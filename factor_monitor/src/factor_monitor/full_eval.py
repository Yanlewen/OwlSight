"""Reusable complete-source evaluation orchestration.

The regular evaluator scores one materialized view at a time.  This module
keeps the suite-level invariants in one place: a benchmark is partitioned by
query into a direct (<= window) view and a lossless windowed (> window) view,
then exactly one score is emitted for every source query.  GPU scoring remains
delegated to :mod:`factor_monitor.training.backend`; this module only handles
configuration, coverage auditing, and deterministic CPU merging.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import ConfigurationError, file_sha256, load_config, stable_hash
from .io import read_jsonl, write_json
from .metrics import score_summary
from .training.encoding import (
    DEFAULT_FIELD_BOUNDARY_POLICY,
    FIELD_BOUNDARY_POLICIES,
)


PARTITIONED_MODE = "partitioned"
FULL_CONTEXT_MODE = "full_context"
ALLOWED_MODES = {PARTITIONED_MODE, FULL_CONTEXT_MODE}
DEFAULT_BENCHMARKS = (
    "aime", "gpqa", "mask_qwen",
    "sycobench_judge", "skilleval", "reward_hacking", "sabotage",
    "sandbagging", "ifbench", "hallulens", "confaide",
    "jailbreakbench", "openai_sycophancy", "openai_wmdp",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    mode: str
    source_directory: Path | None
    source_audit: Path | None
    accepted_directory: Path | None
    overlength_directory: Path | None
    direct_scores: Path
    window_scores: Path | None
    output_directory: Path


def _path(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _expand_tokens(value: Any, *, window_tokens: int, benchmark_id: str) -> Any:
    """Expand the portable path tokens used by a suite config."""
    if not isinstance(value, str):
        return value
    return value.replace("{window}", str(window_tokens)).replace("{benchmark}", benchmark_id)


def _policy_audit_path(path: Path, policy: str) -> Path:
    """Give tokenizer-policy-specific source audits isolated paths."""

    if policy == DEFAULT_FIELD_BOUNDARY_POLICY:
        return path
    suffix = f"_{policy}"
    if path.stem.endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _ids(rows: Iterable[Mapping[str, Any]], *, name: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ConfigurationError(f"{name}[{index}] has no non-empty query_id")
        if query_id in seen:
            raise ConfigurationError(f"{name} contains duplicate query_id={query_id}")
        seen.add(query_id)
        result.append(query_id)
    return result


def _factor_sensitivity_policy(suite: Mapping[str, Any]) -> dict[str, Any] | None:
    value = suite.get("factor_sensitivity")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigurationError("factor_sensitivity must be a mapping")
    variants = value.get("variants")
    if variants != ["matched", "no_factor", "shuffled"]:
        raise ConfigurationError(
            "factor_sensitivity.variants must be exactly "
            "[matched, no_factor, shuffled]"
        )
    if value.get("mixed_labels_only") is not False:
        raise ConfigurationError(
            "formal Factor permutation requires mixed_labels_only: false"
        )
    if value.get("shuffle_stratum") != [
        "family",
        "factor_level",
        "relation",
        "factor_token_length",
    ]:
        raise ConfigurationError(
            "formal Factor permutation must shuffle within "
            "[family, factor_level, relation, factor_token_length]"
        )
    return {
        "variants": list(variants),
        "mixed_labels_only": False,
        "shuffle_stratum": list(value["shuffle_stratum"]),
        "primary_metric": "matched_auroc_minus_shuffled_auroc",
        "bootstrap": int(value.get("bootstrap", 1000)),
        "seed": int(value.get("seed", 1729)),
    }


def _variant_key(row: Mapping[str, Any]) -> tuple[str, str]:
    source = row.get("source_query_id")
    query_id = row.get("query_id")
    if not isinstance(source, str) or not source:
        if not isinstance(query_id, str) or not query_id:
            raise ConfigurationError("Factor sensitivity score row lacks query_id")
        source, separator, suffix = query_id.partition("::factor=")
        if not separator or not source or not suffix:
            raise ConfigurationError(
                "Factor sensitivity score row lacks source_query_id/factor_intervention"
            )
    variant = row.get("factor_intervention")
    if not isinstance(variant, str) or not variant:
        if isinstance(query_id, str) and "::factor=" in query_id:
            variant = query_id.rsplit("::factor=", 1)[1].split("::", 1)[0]
        else:
            raise ConfigurationError(
                "Factor sensitivity score row lacks factor_intervention"
            )
    return source, variant


def _valid_permutation_row(row: Mapping[str, Any]) -> bool:
    """Whether a shuffled row actually changed its Factor donor.

    Singleton strata are retained in the complete variant grid for coverage,
    but cannot provide a valid cross-trajectory permutation.  They are marked
    by an identity donor and excluded from the reviewer-facing Δ_perm.
    """

    if _variant_key(row)[1] != "shuffled":
        return True
    source = row.get("source_query_id")
    donor = row.get("donor_query_id")
    return not isinstance(source, str) or not isinstance(donor, str) or source != donor


def _bootstrap_delta(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    seed: int,
    replicates: int,
) -> list[float] | None:
    trajectories = sorted(
        {
            (str(row.get("benchmark", "")), str(row.get("trajectory_id", "")))
            for row in rows
        }
    )
    if not trajectories:
        return None
    randomizer = __import__("random").Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = [randomizer.choice(trajectories) for _ in trajectories]
        selected = [
            row
            for trajectory in sampled
            for row in rows
            if (str(row.get("benchmark", "")), str(row.get("trajectory_id", "")))
            == trajectory
        ]
        selected = [row for row in selected if _valid_permutation_row(row)]
        by_variant = {
            variant: [row for row in selected if _variant_key(row)[1] == variant]
            for variant in ("matched", "shuffled")
        }
        if not by_variant["matched"] or not by_variant["shuffled"]:
            continue
        matched = score_summary(by_variant["matched"], threshold)["auroc"]
        shuffled = score_summary(by_variant["shuffled"], threshold)["auroc"]
        if matched is not None and shuffled is not None:
            values.append(float(matched) - float(shuffled))
    return values or None


def _factor_sensitivity_summary(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one complete-source run with one reviewer-facing metric."""

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        invalid_shuffled = sum(
            1
            for row in selected
            if _variant_key(row)[1] == "shuffled" and not _valid_permutation_row(row)
        )
        by_variant = {
            variant: [
                row
                for row in selected
                if _variant_key(row)[1] == variant
                and (variant != "shuffled" or _valid_permutation_row(row))
            ]
            for variant in policy["variants"]
        }
        metrics = {
            variant: score_summary(by_variant[variant], threshold)
            for variant in policy["variants"]
        }
        matched = metrics["matched"]["auroc"]
        shuffled = metrics["shuffled"]["auroc"]
        no_factor = metrics["no_factor"]["auroc"]
        delta = None if matched is None or shuffled is None else float(matched) - float(shuffled)
        no_factor_delta = None if matched is None or no_factor is None else float(matched) - float(no_factor)
        return {
            "rows_by_variant": {variant: len(by_variant[variant]) for variant in policy["variants"]},
            "invalid_shuffled_rows_excluded": invalid_shuffled,
            "matched_auroc": matched,
            "shuffled_auroc": shuffled,
            "no_factor_auroc": no_factor,
            "delta_perm": delta,
            "delta_no_factor": no_factor_delta,
            "by_variant": metrics,
        }

    families = sorted({str(row.get("family")) for row in rows})
    by_family = {
        family: summarize([row for row in rows if str(row.get("family")) == family])
        for family in families
    }
    overall = summarize(rows)
    family_deltas = [
        item["delta_perm"] for item in by_family.values() if item["delta_perm"] is not None
    ]
    macro_delta = sum(family_deltas) / len(family_deltas) if family_deltas else None
    bootstrap = _bootstrap_delta(
        rows,
        threshold=threshold,
        seed=int(policy["seed"]),
        replicates=int(policy["bootstrap"]),
    )
    ci = None
    if bootstrap:
        ordered = sorted(bootstrap)
        ci = [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]
    return {
        "schema_version": 1,
        "primary_metric": "delta_perm",
        "primary_definition": "AUROC(matched) - AUROC(shuffled)",
        "shuffle_stratum": policy["shuffle_stratum"],
        "threshold": threshold,
        "overall": overall,
        "family_macro": {
            "matched_auroc": (
                sum(item["matched_auroc"] for item in by_family.values() if item["matched_auroc"] is not None)
                / len([item for item in by_family.values() if item["matched_auroc"] is not None])
                if any(item["matched_auroc"] is not None for item in by_family.values()) else None
            ),
            "shuffled_auroc": (
                sum(item["shuffled_auroc"] for item in by_family.values() if item["shuffled_auroc"] is not None)
                / len([item for item in by_family.values() if item["shuffled_auroc"] is not None])
                if any(item["shuffled_auroc"] is not None for item in by_family.values()) else None
            ),
            "delta_perm": macro_delta,
            "bootstrap_95_ci": ci,
            "bootstrap_replicates": len(bootstrap or []),
        },
        "by_family": by_family,
    }


def _check_label_consistency(
    rows: Iterable[Mapping[str, Any]],
    source_by_id: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> None:
    """Reject a prediction file that silently changes frozen labels."""

    for index, row in enumerate(rows):
        query_id = row["query_id"]
        source = source_by_id.get(query_id)
        if source is None:
            continue
        if "label_id" in source and "label_id" in row and row["label_id"] != source["label_id"]:
            raise ConfigurationError(
                f"{name}[{index}] label_id disagrees with frozen source for query_id={query_id}"
            )


def _score_rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigurationError(f"{name} does not exist: {path}")
    rows = read_jsonl(path)
    _ids(rows, name=name)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        # Window workers persist the complete aggregation vector and may omit
        # the top-level score.  The suite's primary pooling policy is
        # ``top2_mean``; materialize it before coverage auditing so audit and
        # merge validate the same representation.
        if "score" not in row:
            aggregate_scores = row.get("aggregation_scores")
            if isinstance(aggregate_scores, dict) and "top2_mean" in aggregate_scores:
                row = {
                    **row,
                    "score": aggregate_scores["top2_mean"],
                    "aggregation": "top2_mean",
                }
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ConfigurationError(f"{name} contains a non-numeric score")
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ConfigurationError(f"{name} contains score outside finite [0,1]")
        normalized.append(row)
    return normalized


def _resolve_model_binding(
    model_output: Path,
    *,
    checkpoint: str | None = None,
    threshold_source: Path | None = None,
) -> dict[str, Path]:
    """Resolve one immutable training output into eval model inputs.

    A training run normally writes ``model_bundle`` and
    ``validation_threshold.json`` at its root.  If only checkpoint directories
    exist, an explicit checkpoint is required when more than one is present;
    this prevents silently evaluating the wrong checkpoint.
    """

    root = model_output.expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"model output directory does not exist: {root}")
    selected = root
    if checkpoint:
        requested = Path(checkpoint).expanduser()
        selected = (requested if requested.is_absolute() else root / requested).resolve()
        if not selected.is_dir():
            raise ConfigurationError(f"requested checkpoint does not exist: {selected}")
    bundle = selected / "model_bundle"
    if not bundle.is_dir() and not checkpoint:
        bundle = root / "model_bundle"
    if not bundle.is_dir():
        candidates = sorted(
            (path for path in root.glob("checkpoint-*") if path.is_dir() and (path / "model_bundle").is_dir()),
            key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else -1,
        )
        if len(candidates) != 1:
            raise ConfigurationError(
                "model output has no unambiguous model_bundle; pass --checkpoint "
                f"(found {len(candidates)} checkpoint bundles under {root})"
            )
        selected = candidates[0]
        bundle = selected / "model_bundle"
    if threshold_source is not None:
        threshold = threshold_source.expanduser().resolve()
        if not threshold.is_file():
            raise ConfigurationError(f"threshold source does not exist: {threshold}")
    else:
        threshold_candidates = [root / "validation_threshold.json", selected / "validation_threshold.json"]
        threshold = next((path for path in threshold_candidates if path.is_file()), None)
    if threshold is None:
        raise ConfigurationError(f"validation_threshold.json not found under {root} or {selected}")
    tokenizer_candidates = [bundle / "tokenizer", selected / "tokenizer", root / "tokenizer"]
    tokenizer = next((path for path in tokenizer_candidates if path.is_dir()), None)
    if tokenizer is None:
        raise ConfigurationError(f"tokenizer directory not found beside model bundle: {bundle}")
    return {
        "model_bundle": bundle.resolve(),
        "threshold_source": threshold.resolve(),
        "tokenizer": tokenizer.resolve(),
        "checkpoint": selected.resolve(),
    }


def _load_bundle_effective_config(bundle: Path) -> dict[str, Any] | None:
    """Load the saved run contract when a model bundle has one."""

    path = bundle / "effective_config.json"
    if not path.is_file():
        return None
    value = load_config(path)
    if not isinstance(value, dict):
        raise ConfigurationError(f"model bundle effective config is not a mapping: {path}")
    return value


def _bundle_model_identity(bundle_effective: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the model fields that must match a saved bundle at eval time."""

    if bundle_effective is None:
        return None
    model = bundle_effective.get("model")
    if not isinstance(model, Mapping):
        return None
    required = ("checkpoint", "checkpoint_revision", "tokenizer", "tokenizer_revision")
    if any(not isinstance(model.get(key), str) or not model[key].strip() for key in required):
        return None
    identity = {key: model[key] for key in required}
    load_options = model.get("load_options")
    if isinstance(load_options, Mapping):
        identity["load_options"] = dict(load_options)
    return identity


def _resolve_bundle_architecture(
    bundle_effective: Mapping[str, Any] | None,
    *,
    template_path: Path,
    fallback: Path,
) -> Path:
    """Resolve the architecture named by a bundle to a source-snapshot path."""

    if bundle_effective is None:
        return fallback.resolve()
    options = bundle_effective.get("monitor", {}).get("backend_options", {})
    configured = options.get("architecture_config") if isinstance(options, Mapping) else None
    if not isinstance(configured, str) or not configured.strip():
        return fallback.resolve()
    configured_path = Path(configured).expanduser()
    config_root = template_path.resolve().parent
    candidates = [
        configured_path if configured_path.is_absolute() else config_root / configured_path,
        config_root / configured_path.name,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise ConfigurationError(
        "saved model bundle architecture cannot be resolved in the repository: "
        f"{configured}"
    )


def materialize_bound_suite(
    template_path: Path,
    *,
    model_output: Path,
    checkpoint: str | None = None,
    threshold_source: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a model-bound, immutable suite manifest from a reusable template."""

    template = load_suite(template_path)
    binding = _resolve_model_binding(
        model_output,
        checkpoint=checkpoint,
        threshold_source=threshold_source,
    )
    root = model_output.expanduser().resolve()
    run_name = root.name if checkpoint is None else f"{root.name}_{binding['checkpoint'].name}"
    default_eval_root = Path(template["config_path"]).resolve().parents[1] / "runs" / "eval" / run_name / template["suite_id"]
    eval_root = (output_root or default_eval_root).resolve()
    # Keep the bound manifest under configs so every worker receives the exact
    # same suite file, while predictions and merged results stay under runs/eval.
    manifest_path = Path(template["config_path"]).resolve().parents[1] / "configs" / "bound_eval" / run_name / template["suite_id"] / "suite.yaml"
    if eval_root.exists() and any(eval_root.iterdir()) and not overwrite:
        raise ConfigurationError(f"eval output root is non-empty: {eval_root}; pass --overwrite")
    if manifest_path.exists() and not overwrite:
        raise ConfigurationError(f"bound suite already exists: {manifest_path}; pass --overwrite")
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    bound: dict[str, Any] = {
        key: value
        for key, value in template.items()
        if key not in {"config_path", "suite_hash"}
    }
    bundle_effective = _load_bundle_effective_config(binding["model_bundle"])
    saved_model = _bundle_model_identity(bundle_effective)
    fallback_architecture = Path(str(bound["model"].get("architecture")))
    if not fallback_architecture.is_absolute():
        fallback_architecture = Path(template["config_path"]).resolve().parent / fallback_architecture
    architecture = _resolve_bundle_architecture(
        bundle_effective,
        template_path=Path(template["config_path"]),
        fallback=fallback_architecture,
    )
    bound["model"] = {
        **dict(bound["model"]),
        "model_bundle": str(binding["model_bundle"]),
        "threshold_source": str(binding["threshold_source"]),
        "tokenizer": str(binding["tokenizer"]),
        "checkpoint": str(binding["checkpoint"]),
        "architecture": str(architecture),
    }
    if saved_model is not None:
        # The selected training checkpoint remains in provenance; runtime
        # identity must use the base checkpoint recorded by the saved bundle.
        bound["model"].update(saved_model)
    saved_options = (
        bundle_effective.get("monitor", {}).get("backend_options", {})
        if isinstance(bundle_effective, Mapping)
        else {}
    )
    saved_policy = saved_options.get("field_boundary_policy") if isinstance(saved_options, Mapping) else None
    bound["model"]["field_boundary_policy"] = saved_policy or DEFAULT_FIELD_BOUNDARY_POLICY
    benchmarks: list[dict[str, Any]] = []
    for item in template["benchmarks"]:
        copied = dict(item)
        if copied.get("source_audit"):
            copied["source_audit"] = str(
                _policy_audit_path(
                    Path(str(copied["source_audit"])),
                    str(bound["model"].get("field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY)),
                )
            )
        benchmark_root = eval_root / "benchmarks" / str(item["id"])
        copied["direct_scores"] = str(benchmark_root / "direct" / "scores.jsonl")
        copied["output_directory"] = str(benchmark_root)
        if item["mode"] == PARTITIONED_MODE:
            copied["window_scores"] = str(benchmark_root / "window" / "aggregated_scores.jsonl")
        else:
            copied.pop("window_scores", None)
        benchmarks.append(copied)
    bound["benchmarks"] = benchmarks
    bound["output"] = {"directory": str(eval_root / "suite")}
    bound["provenance"] = {
        "template_config": str(Path(template["config_path"]).resolve()),
        "model_output": str(root),
        "checkpoint": str(binding["checkpoint"]),
        "model_bundle": str(binding["model_bundle"]),
        "threshold_source": str(binding["threshold_source"]),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(bound, handle, sort_keys=False)
    resolved = load_suite(manifest_path)
    write_json(eval_root / "worker_plan.json", plan_workers(resolved))
    return {
        "suite_id": resolved["suite_id"],
        "suite_hash": resolved["suite_hash"],
        "manifest": str(manifest_path),
        "output_root": str(eval_root),
        "model": resolved["model"],
    }


def plan_workers(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic worker matrix for a bound suite."""

    workers: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in suite["benchmarks"]:
        benchmark = str(item["id"])
        if item["mode"] == FULL_CONTEXT_MODE:
            workers.append({"benchmark": benchmark, "stages": ["direct"], "mode": item["mode"]})
            continue
        overlength = Path(str(item["overlength_directory"])) / "eval_queries.jsonl"
        stages = ["direct"]
        if overlength.is_file() and overlength.stat().st_size > 0:
            stages.append("window")
        elif not overlength.is_file():
            # Preparation runs in the cluster worker and may not have created
            # the partition yet.  Keep the window stage planned instead of
            # incorrectly reporting it as empty.
            stages.append("window")
        else:
            skipped.append({"benchmark": benchmark, "stage": "window", "reason": "empty_overlength_partition"})
        workers.append({"benchmark": benchmark, "stages": stages, "mode": item["mode"]})
    return {
        "suite_id": suite["suite_id"],
        "suite_hash": suite["suite_hash"],
        "workers": workers,
        "skipped": skipped,
        "worker_count": len(workers),
    }


def load_suite(path: Path) -> dict[str, Any]:
    """Resolve and validate a complete-evaluation suite config."""

    config_path = path.resolve()
    config = load_config(config_path)
    if config.get("kind") != "factor_monitor_full_eval":
        raise ConfigurationError("full-eval config requires kind=factor_monitor_full_eval")
    required = {"schema_version", "suite_id", "model", "window", "coverage", "benchmarks", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ConfigurationError(f"full-eval config missing required fields: {missing}")
    window = config["window"]
    if not isinstance(window, dict):
        raise ConfigurationError("window must be a mapping")
    max_tokens = window.get("max_endpoint_tokens")
    overlap = window.get("overlap_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigurationError("window.max_endpoint_tokens must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 or overlap >= max_tokens:
        raise ConfigurationError("window.overlap_tokens must satisfy 0 <= overlap < max_endpoint_tokens")
    if window.get("truncation_policy") != "error":
        raise ConfigurationError("complete evaluation requires window.truncation_policy=error")
    coverage = config["coverage"]
    if not isinstance(coverage, dict) or coverage.get("source_unit") != "query":
        raise ConfigurationError("coverage.source_unit must be 'query'")
    benchmarks = config["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ConfigurationError("benchmarks must be a non-empty list")
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(benchmarks):
        if not isinstance(item, dict):
            raise ConfigurationError(f"benchmarks[{index}] must be a mapping")
        benchmark_id = item.get("id")
        mode = item.get("mode", PARTITIONED_MODE)
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise ConfigurationError(f"benchmarks[{index}].id must be non-empty")
        if benchmark_id in seen:
            raise ConfigurationError(f"duplicate benchmark id: {benchmark_id}")
        if mode not in ALLOWED_MODES:
            raise ConfigurationError(f"unsupported mode for {benchmark_id}: {mode}")
        seen.add(benchmark_id)
        item = dict(item)
        item["id"] = benchmark_id
        item["mode"] = mode
        if item.get("parent_config"):
            item["parent_config"] = str(_path(_expand_tokens(item["parent_config"], window_tokens=max_tokens, benchmark_id=benchmark_id), base=config_path.parent, field=f"benchmarks[{index}].parent_config"))
        if mode == PARTITIONED_MODE:
            for field in ("source_directory", "source_audit", "accepted_directory", "overlength_directory"):
                item[field] = str(_path(_expand_tokens(item.get(field), window_tokens=max_tokens, benchmark_id=benchmark_id), base=config_path.parent, field=f"benchmarks[{index}].{field}"))
            item["window_scores"] = str(_path(_expand_tokens(item.get("window_scores"), window_tokens=max_tokens, benchmark_id=benchmark_id), base=config_path.parent, field=f"benchmarks[{index}].window_scores")) if item.get("window_scores") else None
        item["direct_scores"] = str(_path(_expand_tokens(item.get("direct_scores"), window_tokens=max_tokens, benchmark_id=benchmark_id), base=config_path.parent, field=f"benchmarks[{index}].direct_scores"))
        item["output_directory"] = str(_path(_expand_tokens(item.get("output_directory"), window_tokens=max_tokens, benchmark_id=benchmark_id), base=config_path.parent, field=f"benchmarks[{index}].output_directory"))
        resolved.append(item)
    model = config["model"]
    if not isinstance(model, dict):
        raise ConfigurationError("model must be a mapping")
    for field in ("model_bundle", "threshold_source", "tokenizer", "architecture"):
        if field not in model and field in {"tokenizer", "architecture"}:
            raise ConfigurationError(f"model.{field} is required for cluster-side preparation")
        model[field] = str(_path(model.get(field), base=config_path.parent, field=f"model.{field}"))
    field_policy = model.get("field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY)
    if field_policy not in FIELD_BOUNDARY_POLICIES:
        raise ConfigurationError(
            "model.field_boundary_policy must be one of "
            f"{FIELD_BOUNDARY_POLICIES}"
        )
    model["field_boundary_policy"] = field_policy
    pad_token = model.get("pad_token")
    if pad_token is not None and (not isinstance(pad_token, str) or not pad_token):
        raise ConfigurationError("model.pad_token must be non-empty text when provided")
    output = config["output"]
    if not isinstance(output, dict) or not output.get("directory"):
        raise ConfigurationError("output.directory is required")
    config["output"] = {**output, "directory": str(_path(_expand_tokens(output["directory"], window_tokens=max_tokens, benchmark_id="suite"), base=config_path.parent, field="output.directory"))}
    _factor_sensitivity_policy(config)
    config["model"] = model
    config["benchmarks"] = resolved
    config["config_path"] = str(config_path)
    config["suite_hash"] = stable_hash({k: v for k, v in config.items() if k not in {"suite_hash", "config_path"}})
    return config


def _spec(item: Mapping[str, Any]) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=str(item["id"]), mode=str(item["mode"]),
        source_directory=Path(item["source_directory"]) if item.get("source_directory") else None,
        source_audit=Path(item["source_audit"]) if item.get("source_audit") else None,
        accepted_directory=Path(item["accepted_directory"]) if item.get("accepted_directory") else None,
        overlength_directory=Path(item["overlength_directory"]) if item.get("overlength_directory") else None,
        direct_scores=Path(item["direct_scores"]),
        window_scores=Path(item["window_scores"]) if item.get("window_scores") else None,
        output_directory=Path(item["output_directory"]),
    )


def audit_coverage(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Check complete source coverage without loading a tokenizer."""

    reports: dict[str, Any] = {}
    sensitivity = _factor_sensitivity_policy(suite)
    expected_variants = tuple(sensitivity["variants"]) if sensitivity else ()

    def row_keys(rows: list[dict[str, Any]], name: str) -> set[tuple[str, str]]:
        if not sensitivity:
            return {(str(row["query_id"]), "matched") for row in rows}
        return {_variant_key(row) for row in rows}

    for item in suite["benchmarks"]:
        spec = _spec(item)
        direct = _score_rows(spec.direct_scores, name=f"{spec.benchmark_id}.direct_scores")
        if spec.mode == FULL_CONTEXT_MODE:
            if sensitivity:
                reports[spec.benchmark_id] = {
                    "mode": spec.mode,
                    "source_queries": len(direct),
                    "scored_queries": len(direct),
                    "diagnostic_rows": len(direct),
                    "variants": ["matched"],
                    "factor_diagnostic": False,
                    "missing": [],
                    "duplicates": 0,
                    "direct_scores_sha256": file_sha256(spec.direct_scores),
                }
            else:
                reports[spec.benchmark_id] = {"mode": spec.mode, "source_queries": len(direct), "scored_queries": len(direct), "missing": [], "duplicates": 0, "direct_scores_sha256": file_sha256(spec.direct_scores)}
            continue
        assert spec.source_directory and spec.accepted_directory and spec.overlength_directory
        if not spec.source_directory.is_dir():
            raise ConfigurationError(f"{spec.benchmark_id}: source directory does not exist: {spec.source_directory}")
        if not spec.source_audit or not spec.source_audit.is_file():
            raise ConfigurationError(f"{spec.benchmark_id}: source audit does not exist: {spec.source_audit}")
        source_rows = read_jsonl(spec.source_directory / "eval_queries.jsonl")
        accepted_rows = read_jsonl(spec.accepted_directory / "eval_queries.jsonl")
        overlength_rows = read_jsonl(spec.overlength_directory / "eval_queries.jsonl")
        source = set(_ids(source_rows, name=f"{spec.benchmark_id}.source"))
        source_by_id = {row["query_id"]: row for row in source_rows}
        accepted = set(_ids(accepted_rows, name=f"{spec.benchmark_id}.accepted"))
        overlength = set(_ids(overlength_rows, name=f"{spec.benchmark_id}.overlength"))
        if accepted & overlength:
            raise ConfigurationError(f"{spec.benchmark_id}: accepted/overlength query overlap")
        if accepted | overlength != source:
            raise ConfigurationError(f"{spec.benchmark_id}: partition does not exactly cover source")
        # An empty overlength partition is a valid complete-source case.  Do
        # not require a synthetic empty score file; a non-empty partition
        # still fails closed when its window output is missing.
        if overlength:
            if spec.window_scores is None:
                raise ConfigurationError(f"{spec.benchmark_id}: window scores are required for overlength queries")
            window = _score_rows(spec.window_scores, name=f"{spec.benchmark_id}.window_scores")
        elif spec.window_scores and spec.window_scores.is_file():
            window = _score_rows(spec.window_scores, name=f"{spec.benchmark_id}.window_scores")
        else:
            window = []
        direct_ids, window_ids = row_keys(direct, f"{spec.benchmark_id}.direct_scores"), row_keys(window, f"{spec.benchmark_id}.window_scores")
        _check_label_consistency(direct, source_by_id, name=f"{spec.benchmark_id}.direct_scores")
        _check_label_consistency(window, source_by_id, name=f"{spec.benchmark_id}.window_scores")
        expected_direct = (
            {(query_id, variant) for query_id in accepted for variant in expected_variants}
            if sensitivity else {(query_id, "matched") for query_id in accepted}
        )
        expected_window = (
            {(query_id, variant) for query_id in overlength for variant in expected_variants}
            if sensitivity else {(query_id, "matched") for query_id in overlength}
        )
        if direct_ids != expected_direct:
            raise ConfigurationError(f"{spec.benchmark_id}: direct scores do not match accepted partition")
        if window_ids != expected_window:
            raise ConfigurationError(f"{spec.benchmark_id}: window scores do not match overlength partition")
        reports[spec.benchmark_id] = {"mode": spec.mode, "source_queries": len(source), "accepted_queries": len(accepted), "overlength_queries": len(overlength), "scored_queries": len(source), "diagnostic_rows": len(direct) + len(window), "variants": list(expected_variants) if sensitivity else ["matched"], "missing": sorted(source - {query_id for query_id, _ in direct_ids | window_ids}), "duplicates": 0, "source_query_file_sha256": file_sha256(spec.source_directory / "eval_queries.jsonl"), "source_audit_sha256": file_sha256(spec.source_audit), "direct_scores_sha256": file_sha256(spec.direct_scores), "window_scores_sha256": file_sha256(spec.window_scores) if spec.window_scores and spec.window_scores.is_file() else None}
    return reports


def merge_suite(suite: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    reports = audit_coverage(suite)
    merged: list[dict[str, Any]] = []
    for item in suite["benchmarks"]:
        spec = _spec(item)
        direct = _score_rows(spec.direct_scores, name=f"{spec.benchmark_id}.direct_scores")
        rows = list(direct)
        if spec.mode == PARTITIONED_MODE:
            assert spec.window_scores
            overlength = Path(str(item["overlength_directory"])) / "eval_queries.jsonl"
            if overlength.is_file() and overlength.stat().st_size > 0:
                window = _score_rows(spec.window_scores, name=f"{spec.benchmark_id}.window_scores")
            elif spec.window_scores.is_file():
                window = _score_rows(spec.window_scores, name=f"{spec.benchmark_id}.window_scores")
            else:
                window = []
            for row in window:
                aggregate_scores = row.get("aggregation_scores")
                if isinstance(aggregate_scores, dict):
                    if "top2_mean" not in aggregate_scores:
                        raise ConfigurationError(f"{spec.benchmark_id}: window aggregate lacks top2_mean")
                    row = {**row, "score": aggregate_scores["top2_mean"], "aggregation": "top2_mean"}
                rows.append(row)
        for row in rows:
            merged.append({**row, "benchmark": spec.benchmark_id})
        reports[spec.benchmark_id]["merged_queries"] = len(rows)
    output = Path(str(suite["output"]["directory"]))
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ConfigurationError(f"suite output is non-empty: {output}; pass overwrite explicitly")
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "merged_scores.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    sensitivity = _factor_sensitivity_policy(suite)
    sensitivity_summary = None
    if sensitivity:
        threshold_payload = load_config(Path(str(suite["model"]["threshold_source"])))
        threshold_value = threshold_payload.get("threshold")
        if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
            raise ConfigurationError("Factor sensitivity threshold source lacks numeric threshold")
        diagnostic_rows = [
            row
            for row in merged
            if str(row.get("factor_intervention", "")) in set(sensitivity["variants"])
            and isinstance(row.get("label_id"), int)
            and not isinstance(row.get("label_id"), bool)
            and row["label_id"] in {0, 1}
        ]
        sensitivity_summary = _factor_sensitivity_summary(
            diagnostic_rows,
            threshold=float(threshold_value),
            policy=sensitivity,
        )
        write_json(output / "factor_sensitivity_summary.json", sensitivity_summary)
    result = {"schema_version": 1, "suite_id": suite["suite_id"], "suite_hash": suite["suite_hash"], "window": suite["window"], "benchmarks": reports, "rows": len(merged), "complete": True, "source_config": suite["config_path"], "model": suite["model"], "merged_scores": str(merged_path)}
    if sensitivity_summary is not None:
        result["factor_sensitivity"] = sensitivity_summary
    write_json(output / "coverage.json", reports)
    write_json(output / "summary.json", result)
    write_json(output / "suite_manifest.json", dict(suite))
    write_json(
        output / "provenance.json",
        {
            "suite_id": suite["suite_id"],
            "suite_hash": suite["suite_hash"],
            "source_config": suite["config_path"],
            "model": suite["model"],
            "benchmark_count": len(suite["benchmarks"]),
            "coverage_file": str(output / "coverage.json"),
        },
    )
    lines = ["# Complete-source evaluation", "", f"- suite: `{suite['suite_id']}`", f"- window: `{suite['window']['max_endpoint_tokens']}` tokens", f"- rows: `{len(merged)}`", ""]
    if sensitivity_summary is not None:
        lines.extend([
            "## Factor permutation diagnostic",
            "",
            "Primary metric: `Δ_perm = AUROC(matched) − AUROC(shuffled)`; shuffling is within family × factor level × relation.",
            f"- overall Δ_perm: `{sensitivity_summary['overall']['delta_perm']}`",
            f"- family-macro Δ_perm: `{sensitivity_summary['family_macro']['delta_perm']}`",
            f"- family-macro bootstrap 95% CI: `{sensitivity_summary['family_macro']['bootstrap_95_ci']}`",
            "",
        ])
    lines.extend(["| benchmark | source | accepted | overlength | merged |", "|---|---:|---:|---:|---:|"])
    for benchmark_id, report in reports.items():
        lines.append(f"| {benchmark_id} | {report['source_queries']} | {report.get('accepted_queries', report['source_queries'])} | {report.get('overlength_queries', 0)} | {report['merged_queries']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_worker(suite_path: Path, benchmark_id: str, stage: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Run one independent benchmark stage through the existing evaluator."""
    if stage not in {"direct", "window"}:
        raise ConfigurationError("worker stage must be direct or window")
    suite = load_suite(suite_path)
    item = next((value for value in suite["benchmarks"] if value["id"] == benchmark_id), None)
    if item is None:
        raise ConfigurationError(f"unknown benchmark: {benchmark_id}")
    if item["mode"] == FULL_CONTEXT_MODE and stage != "direct":
        raise ConfigurationError("full-context intervention benchmarks have no window stage")
    if item["mode"] == PARTITIONED_MODE and stage == "window":
        overlength_queries = Path(item["overlength_directory"]) / "eval_queries.jsonl"
        if not overlength_queries.is_file() or overlength_queries.stat().st_size == 0:
            return {
                "suite_id": suite["suite_id"],
                "benchmark": benchmark_id,
                "stage": stage,
                "skipped": True,
                "reason": "empty_overlength_partition",
            }
    parent_value = item.get("parent_config")
    if not parent_value:
        raise ConfigurationError(f"{benchmark_id} is missing parent_config")
    parent = _path(parent_value, base=Path(suite["config_path"]).parent, field=f"{benchmark_id}.parent_config")
    output = Path(item["output_directory"]) / stage
    overrides: dict[str, Any] = {
        "monitor": {
            "backend_options": {
                "model_bundle": suite["model"]["model_bundle"],
                "field_boundary_policy": suite["model"].get(
                    "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
                ),
            }
        },
        "evaluation": {"threshold_source": suite["model"]["threshold_source"]},
        "output": {"directory": str(output)},
    }
    bundle_effective = _load_bundle_effective_config(
        Path(str(suite["model"]["model_bundle"]))
    )
    model_override = _bundle_model_identity(bundle_effective)
    if model_override is None:
        # Keep the generated suite usable for legacy/minimal bundles while
        # still making all fields explicit for normal saved bundles.
        model_override = {
            key: suite["model"][key]
            for key in ("checkpoint", "checkpoint_revision", "tokenizer", "tokenizer_revision")
            if key in suite["model"]
        }
    if suite["model"].get("pad_token") is not None:
        load_options = model_override.setdefault("load_options", {})
        if isinstance(load_options, Mapping):
            load_options["pad_token"] = suite["model"]["pad_token"]
    if model_override:
        overrides["model"] = model_override
    architecture_path = Path(str(suite["model"]["architecture"]))
    if architecture_path.is_file():
        from .training.config import load_architecture

        architecture = load_architecture(architecture_path)
        overrides["monitor"]["backend_options"].update(
            {
                "architecture_config": str(architecture_path),
                "evidence_routes": dict(architecture.family_routes),
            }
        )
    if item["mode"] == FULL_CONTEXT_MODE:
        # Parent OpenAI configs predate tokenizer-policy-specific audits.  A
        # non-default Llama boundary policy must bind the sibling audit
        # materialized by prepare_full_eval.py, otherwise config validation
        # rejects the worker before model loading.
        parent_effective = load_config(parent)
        parent_backend = parent_effective.get("monitor", {}).get(
            "backend_options", {}
        )
        parent_audit = parent_backend.get("endpoint_length_audit")
        if isinstance(parent_audit, str) and parent_audit:
            audit_path = _path(
                parent_audit,
                base=parent.parent,
                field=f"{benchmark_id}.parent_endpoint_length_audit",
            )
            policy = str(
                suite["model"].get(
                    "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
                )
            )
            overrides["monitor"]["backend_options"]["endpoint_length_audit"] = str(
                _policy_audit_path(audit_path, policy)
            )
    sensitivity = _factor_sensitivity_policy(suite)
    if sensitivity and item["mode"] == PARTITIONED_MODE:
        threshold_payload = load_config(Path(str(suite["model"]["threshold_source"])))
        native_threshold = threshold_payload.get("threshold")
        if isinstance(native_threshold, bool) or not isinstance(native_threshold, (int, float)):
            raise ConfigurationError("Factor sensitivity threshold source lacks numeric threshold")
        overrides["monitor"]["backend_options"]["evaluation_scope"] = {
            "mode": "diagnostic",
            "factor_interventions": list(sensitivity["variants"]),
            "mixed_labels_only": False,
            "intervention_seed": int(sensitivity["seed"]),
            "native_checkpoint_threshold": float(native_threshold),
        }
    if item["mode"] == PARTITIONED_MODE:
        dataset_dir = item["accepted_directory"] if stage == "direct" else item["overlength_directory"]
        overrides["dataset"] = {"directory": dataset_dir, "trajectory_file": "trajectories.jsonl", "query_file": "eval_queries.jsonl"}
        overrides["runtime"] = {"max_sequence_length": suite["window"]["max_endpoint_tokens"]}
        window_tokens = int(suite["window"]["max_endpoint_tokens"])
        overrides["monitor"]["backend_options"]["sequence"] = {
            "length_buckets": [
                bucket
                for bucket in (8192, 16384, 24576, window_tokens)
                if bucket <= window_tokens
            ]
        }
        # Each stage reads its own materialized trajectories.  The accepted
        # and overlength partitions have separate audits; using the original
        # joint/source audit for the window partition fails binding validation.
        overrides["monitor"]["backend_options"]["endpoint_length_audit"] = str(
            Path(dataset_dir) / "length_audit.json"
        )
        if stage == "window":
            window = suite["window"]
            overrides["monitor"]["backend_options"]["endpoint_max_padded_tokens_per_backbone_forward"] = window["max_endpoint_tokens"]
            # The overlength partition contains queries excluded from the
            # direct (<= window) partition, so it has no matching direct
            # reference rows by construction.  Aggregate this partition
            # independently; the suite merger later combines its query-level
            # rows with the direct partition without requiring cross-partition
            # comparison.
            scope_override = overrides["monitor"]["backend_options"].get(
                "evaluation_scope",
                {"mode": "formal"},
            )
            scope_override["long_cot_windowing"] = {
                    "enabled": True,
                    "trigger_endpoint_tokens": window["max_endpoint_tokens"],
                    "max_endpoint_tokens": window["max_endpoint_tokens"],
                    "cot_overlap_tokens": window["overlap_tokens"],
                    "context_policy": window["context_policy"],
                    "coverage_policy": "complete_no_truncation",
                    "selection_unit": "trajectory_any_query_over_trigger",
                    "aggregations": ["max", "top2_mean", "top2_logmeanexp", "noisy_or"],
                    "comparison_mode": "standalone",
                }
            overrides["monitor"]["backend_options"]["evaluation_scope"] = scope_override
    import yaml

    # Keep the generated child beside its parent config. Runtime path
    # resolution is relative to the child file, and the inherited eval configs
    # intentionally contain paths such as ``../runs`` and a sibling
    # architecture filename. Writing the child under system /tmp silently
    # changes the meaning of all of those paths.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".factor-monitor-worker-",
        suffix=".yaml",
        dir=parent.parent,
        delete=False,
    ) as handle:
        child = Path(handle.name)
        yaml.safe_dump({"extends": str(parent), **overrides}, handle, sort_keys=False)
    try:
        from .runtime import run_from_config
        return run_from_config(child, dry_run=dry_run)
    finally:
        child.unlink(missing_ok=True)
