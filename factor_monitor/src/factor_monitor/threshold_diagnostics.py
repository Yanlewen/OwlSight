"""Diagnostic-only decomposition of frozen-threshold and separability error."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    repository_state,
    require_keys,
    require_mapping,
    resolve_path,
)
from .io import make_tree_group_accessible, read_jsonl, write_csv, write_json, write_jsonl
from .metrics import (
    auroc,
    average_precision,
    best_threshold_metrics,
    grouped_fold_partitions,
    threshold_curve,
    threshold_metrics,
)
from .openai_intervention import flatten_predictions, score_frozen_plan
from .process_eval import aggregate_human_trajectory_rows


OWNED_OUTPUTS = {"effective_config.json", "summary.json", "table.csv", "curves.jsonl"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(values: Iterable[float], confidence_level: float) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "lower": _quantile(finite, tail),
        "median": _quantile(finite, 0.5),
        "upper": _quantile(finite, 1.0 - tail),
        "replicates": len(finite),
    }


def _score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, name in ((0, "negative"), (1, "positive")):
        scores = [float(row["score"]) for row in rows if row["label_id"] == label]
        result[name] = {
            "count": len(scores),
            "min": _quantile(scores, 0.0),
            "p05": _quantile(scores, 0.05),
            "p25": _quantile(scores, 0.25),
            "p50": _quantile(scores, 0.5),
            "p75": _quantile(scores, 0.75),
            "p95": _quantile(scores, 0.95),
            "max": _quantile(scores, 1.0),
        }
    return result


def _decomposition(frozen_gmean2: float, oracle_gmean2: float) -> dict[str, float | None]:
    total = 1.0 - frozen_gmean2
    recoverable = oracle_gmean2 - frozen_gmean2
    residual = 1.0 - oracle_gmean2
    return {
        "total_gap_to_one": total,
        "threshold_recoverable_gap": recoverable,
        "ranking_separability_residual_gap": residual,
        "threshold_share_of_total_gap": recoverable / total if total > 0 else None,
        "ranking_separability_share_of_total_gap": residual / total if total > 0 else None,
    }


def _slice_key(filters: dict[str, Any]) -> str:
    if not filters:
        return "overall"
    return "__".join(f"{key}={filters[key]}" for key in sorted(filters))


def _slices(
    rows: list[dict[str, Any]], dimensions: list[str]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    result: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [({}, rows)]
    for width in range(1, len(dimensions) + 1):
        for selected in itertools.combinations(dimensions, width):
            values = sorted({tuple(row.get(key) for key in selected) for row in rows}, key=str)
            for combination in values:
                filters = dict(zip(selected, combination))
                chosen = [
                    row
                    for row in rows
                    if all(row.get(key) == value for key, value in filters.items())
                ]
                result.append((filters, chosen))
    return result


def _crossfit(
    rows: list[dict[str, Any]], *, group_field: str, folds: int, seed: int
) -> dict[str, Any]:
    oof_predictions: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    try:
        partitions = grouped_fold_partitions(
            rows, group_field=group_field, folds=folds, seed=seed
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    for fold_index, (train, test) in enumerate(partitions):
        selected = best_threshold_metrics(train, metric="gmean2")
        threshold = float(selected["threshold"])
        test_metrics = threshold_metrics(test, threshold)
        train_groups = {str(row[group_field]) for row in train}
        test_groups = {str(row[group_field]) for row in test}
        fold_results.append(
            {
                "fold": fold_index,
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "train_rows": len(train),
                "test_rows": len(test),
                "selected_threshold": threshold,
                "test_metrics": test_metrics,
            }
        )
        oof_predictions.extend(
            {"label_id": int(row["label_id"]), "score": float(row["score"] >= threshold)}
            for row in test
        )
    combined = threshold_metrics(oof_predictions, 0.5)
    combined["threshold"] = None
    return {
        "selection_metric": "gmean2",
        "assignment": "label-presence-stratified trajectory groups",
        "folds": folds,
        "seed": seed,
        "group_field": group_field,
        "out_of_fold_metrics": combined,
        "fold_results": fold_results,
    }


def _bootstrap(
    rows: list[dict[str, Any]],
    *,
    frozen_threshold: float,
    group_field: str,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for _ in range(replicates):
        selected = [rng.choice(group_ids) for _ in group_ids]
        sample = [row for group in selected for row in grouped[group]]
        if len({int(row["label_id"]) for row in sample}) < 2:
            skipped += 1
            continue
        frozen = threshold_metrics(sample, frozen_threshold)
        oracle = best_threshold_metrics(
            sample, metric="gmean2", extra_thresholds=(frozen_threshold,)
        )
        decomposition = _decomposition(float(frozen["gmean2"]), float(oracle["gmean2"]))
        samples["frozen_gmean2"].append(float(frozen["gmean2"]))
        samples["oracle_gmean2"].append(float(oracle["gmean2"]))
        for key, value in decomposition.items():
            if value is not None:
                samples[key].append(float(value))
    return {
        "method": "group bootstrap with replacement",
        "group_field": group_field,
        "requested_replicates": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_single_class_replicates": skipped,
        "seed": seed,
        "confidence_level": confidence_level,
        "intervals": {
            key: _interval(values, confidence_level) for key, values in sorted(samples.items())
        },
    }


def _binary_diagnostic(
    rows: list[dict[str, Any]],
    *,
    frozen_threshold: float,
    crossfit: dict[str, Any],
    bootstrap: dict[str, Any],
    seed_salt: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = {int(row["label_id"]) for row in rows}
    if labels != {0, 1}:
        raise ConfigurationError(f"binary diagnostic slice must contain both classes, found {labels}")
    curve = threshold_curve(rows, extra_thresholds=(frozen_threshold,))
    frozen = threshold_metrics(rows, frozen_threshold)
    oracle = max(
        curve,
        key=lambda row: (
            float(row["gmean2"]),
            -abs(float(row["threshold"]) - 0.5),
            float(row["threshold"]),
        ),
    )
    balanced = max(
        curve,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            -abs(float(row["threshold"]) - 0.5),
            float(row["threshold"]),
        ),
    )
    area = auroc(rows)
    salt = int(hashlib.sha256(seed_salt.encode("utf-8")).hexdigest()[:8], 16)
    result = {
        "rows": len(rows),
        "groups": len({str(row[crossfit["group_field"]]) for row in rows}),
        "class_counts": {
            "not_applied": sum(int(row["label_id"]) == 0 for row in rows),
            "applied": sum(int(row["label_id"]) == 1 for row in rows),
        },
        "frozen": frozen,
        "gmean2_oracle": {**oracle, "diagnostic_only": True},
        "balanced_accuracy_oracle": {**balanced, "diagnostic_only": True},
        "decomposition": _decomposition(float(frozen["gmean2"]), float(oracle["gmean2"])),
        "ranking": {
            "auroc": area,
            "pairwise_inversion_rate": None if area is None else 1.0 - area,
            "auprc": average_precision(rows),
        },
        "score_distribution": _score_distribution(rows),
        "crossfit": _crossfit(
            rows,
            group_field=crossfit["group_field"],
            folds=crossfit["folds"],
            seed=crossfit["seed"],
        ),
        "uncertainty": _bootstrap(
            rows,
            frozen_threshold=frozen_threshold,
            group_field=bootstrap["group_field"],
            replicates=bootstrap["replicates"],
            seed=bootstrap["seed"] + salt,
            confidence_level=bootstrap["confidence_level"],
        ),
    }
    return result, curve


def _resolve_input(entry: dict[str, Any], *, config_directory: Path) -> dict[str, Any]:
    reject_unknown(
        entry,
        {
            "id",
            "adapter",
            "scores",
            "effective_config",
            "slice_dimensions",
            "artifact",
        },
        "threshold diagnostic input",
    )
    require_keys(entry, {"id", "adapter", "scores", "effective_config"}, "threshold diagnostic input")
    identifier = entry["id"]
    adapter = entry["adapter"]
    if not isinstance(identifier, str) or not identifier:
        raise ConfigurationError("threshold diagnostic input.id must be nonempty")
    if adapter not in {"binary", "process_any_path", "openai_intervention"}:
        raise ConfigurationError(f"unsupported threshold diagnostic adapter: {adapter}")
    dimensions = entry.get("slice_dimensions", [])
    if not isinstance(dimensions, list) or any(not isinstance(value, str) for value in dimensions):
        raise ConfigurationError(f"{identifier}.slice_dimensions must be a list of strings")
    resolved = {
        **entry,
        "scores": resolve_path(entry["scores"], relative_to=config_directory),
        "effective_config": resolve_path(entry["effective_config"], relative_to=config_directory),
        "slice_dimensions": dimensions,
    }
    if adapter == "openai_intervention":
        if "artifact" not in entry:
            raise ConfigurationError(f"{identifier} openai_intervention input requires artifact")
        resolved["artifact"] = resolve_path(entry["artifact"], relative_to=config_directory)
    elif "artifact" in entry:
        raise ConfigurationError(f"{identifier}.artifact is only valid for openai_intervention")
    for key in ("scores", "effective_config"):
        if not resolved[key].is_file():
            raise ConfigurationError(f"{identifier}.{key} does not exist: {resolved[key]}")
    if adapter == "openai_intervention" and not resolved["artifact"].is_dir():
        raise ConfigurationError(f"{identifier}.artifact does not exist: {resolved['artifact']}")
    return resolved


def _resolve(config_path: Path) -> dict[str, Any]:
    raw = load_config(config_path)
    reject_unknown(
        raw,
        {
            "schema_version",
            "kind",
            "diagnostic_only",
            "metric",
            "threshold_source",
            "crossfit",
            "bootstrap",
            "inputs",
            "output",
        },
        "threshold diagnostic config",
    )
    require_keys(
        raw,
        {
            "schema_version",
            "kind",
            "diagnostic_only",
            "metric",
            "threshold_source",
            "crossfit",
            "bootstrap",
            "inputs",
            "output",
        },
        "threshold diagnostic config",
    )
    if raw["schema_version"] != 1 or raw["kind"] != "threshold_decomposition":
        raise ConfigurationError("threshold diagnostic requires schema_version=1 and kind=threshold_decomposition")
    if raw["diagnostic_only"] is not True:
        raise ConfigurationError("threshold decomposition must explicitly set diagnostic_only=true")
    metric = require_mapping(raw["metric"], "metric")
    reject_unknown(metric, {"primary", "definition", "frozen_selection_metric"}, "metric")
    require_keys(metric, {"primary", "definition", "frozen_selection_metric"}, "metric")
    if metric["primary"] != "gmean2" or metric["definition"] != "tpr_times_tnr":
        raise ConfigurationError("this diagnostic requires gmean2=tpr_times_tnr")
    crossfit = require_mapping(raw["crossfit"], "crossfit")
    reject_unknown(crossfit, {"folds", "seed", "group_field"}, "crossfit")
    require_keys(crossfit, {"folds", "seed", "group_field"}, "crossfit")
    bootstrap = require_mapping(raw["bootstrap"], "bootstrap")
    reject_unknown(
        bootstrap, {"replicates", "seed", "group_field", "confidence_level"}, "bootstrap"
    )
    require_keys(
        bootstrap, {"replicates", "seed", "group_field", "confidence_level"}, "bootstrap"
    )
    resolved_crossfit = {
        "folds": _integer(crossfit["folds"], "crossfit.folds", minimum=2),
        "seed": _integer(crossfit["seed"], "crossfit.seed", minimum=0),
        "group_field": str(crossfit["group_field"]),
    }
    resolved_bootstrap = {
        "replicates": _integer(bootstrap["replicates"], "bootstrap.replicates", minimum=1),
        "seed": _integer(bootstrap["seed"], "bootstrap.seed", minimum=0),
        "group_field": str(bootstrap["group_field"]),
        "confidence_level": _number(bootstrap["confidence_level"], "bootstrap.confidence_level"),
    }
    if not 0 < resolved_bootstrap["confidence_level"] < 1:
        raise ConfigurationError("bootstrap.confidence_level must be in (0,1)")
    threshold_source = resolve_path(raw["threshold_source"], relative_to=config_path.parent)
    threshold_payload = load_config(threshold_source)
    frozen_threshold = _number(threshold_payload.get("threshold"), "threshold_source.threshold")
    if not 0 <= frozen_threshold <= 1:
        raise ConfigurationError("frozen threshold must be in [0,1]")
    inputs = raw["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ConfigurationError("inputs must be a nonempty list")
    resolved_inputs = [_resolve_input(entry, config_directory=config_path.parent) for entry in inputs]
    identifiers = [entry["id"] for entry in resolved_inputs]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("threshold diagnostic input IDs must be unique")
    output = require_mapping(raw["output"], "output")
    reject_unknown(output, {"directory"}, "output")
    require_keys(output, {"directory"}, "output")
    return {
        **raw,
        "config_path": config_path.resolve(),
        "config_sha256": file_sha256(config_path.resolve()),
        "threshold_source": threshold_source,
        "threshold_source_sha256": file_sha256(threshold_source),
        "frozen_threshold": frozen_threshold,
        "crossfit": resolved_crossfit,
        "bootstrap": resolved_bootstrap,
        "inputs": resolved_inputs,
        "output_directory": resolve_path(output["directory"], relative_to=config_path.parent),
    }


def _validate_source_effective(entry: dict[str, Any], frozen_threshold: float) -> dict[str, Any]:
    effective = load_config(entry["effective_config"])
    if effective.get("kind") != "eval_run":
        raise ConfigurationError(f"{entry['id']} source effective config must have kind=eval_run")
    evaluation = require_mapping(effective.get("evaluation"), f"{entry['id']}.evaluation")
    if evaluation.get("threshold_policy") != "frozen_validation":
        raise ConfigurationError(f"{entry['id']} did not use frozen_validation threshold policy")
    source_threshold = _number(evaluation.get("threshold_value"), f"{entry['id']}.threshold_value")
    if source_threshold != frozen_threshold:
        raise ConfigurationError(
            f"{entry['id']} threshold {source_threshold} differs from frozen {frozen_threshold}"
        )
    return effective


def _openai_metric_at_threshold(
    *,
    queries: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    source_id: str,
    level: str,
    plan: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = [
        {
            "query_id": row["query_id"],
            "prediction_label": "applied" if float(row["score"]) >= threshold else "not_applied",
        }
        for row in scores
    ]
    monitored = [
        row for row in flatten_predictions(queries, predictions) if row["source_id"] == source_id
    ]
    final, per_bootstrap, _, _ = score_frozen_plan(
        monitored, plan, z_col="z_l1" if level == "L1" else "z_l3"
    )
    if len(final) != 1 or final[0].get("gmean2_mean") is None:
        raise ConfigurationError(
            f"OpenAI source {source_id}/{level} must produce exactly one defined metric series"
        )
    return final[0], per_bootstrap


def _openai_diagnostics(
    entry: dict[str, Any], *, frozen_threshold: float, confidence_level: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact = entry["artifact"]
    queries = read_jsonl(artifact / "eval_queries.jsonl")
    scores = read_jsonl(entry["scores"])
    score_by_id = {str(row.get("query_id")): row for row in scores}
    if len(score_by_id) != len(scores) or set(score_by_id) != {str(row["query_id"]) for row in queries}:
        raise ConfigurationError(f"{entry['id']} scores must exactly cover unique artifact query IDs")
    ordered_scores = [score_by_id[str(query["query_id"])] for query in queries]
    manifest = load_config(artifact / "plan_manifest.json")
    plan_by_source = {
        str(source["source_id"]): read_jsonl(artifact / source["copied_plan"]["bootstrap_plan"])
        for source in manifest["sources"]
    }
    results: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for source_id in sorted(plan_by_source):
        for level in ("L1", "L3"):
            series_scores = [
                float(score_by_id[str(query["query_id"])]["score"])
                for query in queries
                if query["evaluation"]["source_id"] == source_id
                and query["factor_level"] == level
            ]
            unique = sorted(set(series_scores))
            candidates = sorted(
                {unique[0] - 1e-12, *unique, unique[-1] + 1e-12, frozen_threshold}
            )
            evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for threshold in candidates:
                final, per_bootstrap = _openai_metric_at_threshold(
                    queries=queries,
                    scores=ordered_scores,
                    source_id=source_id,
                    level=level,
                    plan=plan_by_source[source_id],
                    threshold=threshold,
                )
                point = {
                    "input_id": entry["id"],
                    "slice_id": f"source_id={source_id}__factor_level={level}",
                    "adapter": "openai_intervention",
                    "threshold": threshold,
                    **{key: value for key, value in final.items() if key.endswith("_mean")},
                }
                curves.append(point)
                evaluated.append((point, per_bootstrap))
            frozen_point, frozen_bootstrap = next(
                item for item in evaluated if item[0]["threshold"] == frozen_threshold
            )
            oracle_point, oracle_bootstrap = max(
                evaluated,
                key=lambda item: (
                    float(item[0]["gmean2_mean"]),
                    -abs(float(item[0]["threshold"]) - 0.5),
                    float(item[0]["threshold"]),
                ),
            )
            frozen_gmean2 = float(frozen_point["gmean2_mean"])
            oracle_gmean2 = float(oracle_point["gmean2_mean"])
            results.append(
                {
                    "input_id": entry["id"],
                    "adapter": "openai_intervention",
                    "slice": {"source_id": source_id, "factor_level": level},
                    "slice_id": f"source_id={source_id}__factor_level={level}",
                    "rows": len(series_scores),
                    "frozen": {**frozen_point, "plan_bootstrap_gmean2_ci": _interval(
                        (row["gmean2"] for row in frozen_bootstrap if row.get("gmean2") is not None),
                        confidence_level,
                    )},
                    "gmean2_oracle": {
                        **oracle_point,
                        "diagnostic_only": True,
                        "plan_bootstrap_gmean2_ci": _interval(
                            (row["gmean2"] for row in oracle_bootstrap if row.get("gmean2") is not None),
                            confidence_level,
                        ),
                    },
                    "decomposition": _decomposition(frozen_gmean2, oracle_gmean2),
                    "ranking": {
                        "status": "not_defined",
                        "reason": "behavioral outcome y is not Factor-application ground truth",
                    },
                    "crossfit": {
                        "status": "not_applicable",
                        "reason": "the frozen intervention plan is replayed intact; no threshold-calibration split is defined",
                    },
                }
            )
    return results, curves


def _table_row(result: dict[str, Any]) -> dict[str, Any]:
    frozen = result["frozen"]
    oracle = result["gmean2_oracle"]
    frozen_gmean2 = frozen.get("gmean2", frozen.get("gmean2_mean"))
    oracle_gmean2 = oracle.get("gmean2", oracle.get("gmean2_mean"))
    crossfit = result.get("crossfit", {})
    oof = crossfit.get("out_of_fold_metrics", {})
    ranking = result.get("ranking", {})
    decomposition = result["decomposition"]
    return {
        "input_id": result["input_id"],
        "adapter": result["adapter"],
        "slice_id": result["slice_id"],
        "rows": result["rows"],
        "frozen_threshold": frozen.get("threshold"),
        "frozen_gmean2": frozen_gmean2,
        "oracle_threshold": oracle.get("threshold"),
        "oracle_gmean2": oracle_gmean2,
        "crossfit_gmean2": oof.get("gmean2"),
        "auroc": ranking.get("auroc"),
        "pairwise_inversion_rate": ranking.get("pairwise_inversion_rate"),
        **decomposition,
    }


def run_threshold_decomposition(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Run a CPU-only, diagnostic-only audit over preserved evaluation scores."""

    resolved = _resolve(config_path.resolve())
    output = resolved["output_directory"]
    if output.exists():
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in OWNED_OUTPUTS)
        if unexpected:
            raise ConfigurationError(f"refusing to overwrite unowned diagnostic outputs: {unexpected}")
        if not overwrite and any(output.iterdir()):
            raise FileExistsError(f"threshold diagnostic output is not empty; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    permission_probe = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.permission-probe-", dir=output.parent)
    )
    shutil.rmtree(permission_probe)

    source_effective: dict[str, dict[str, Any]] = {}
    source_provenance: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for entry in resolved["inputs"]:
        effective = _validate_source_effective(entry, resolved["frozen_threshold"])
        source_effective[entry["id"]] = effective
        source_provenance.append(
            {
                "id": entry["id"],
                "adapter": entry["adapter"],
                "scores": str(entry["scores"]),
                "scores_sha256": file_sha256(entry["scores"]),
                "effective_config": str(entry["effective_config"]),
                "effective_config_sha256": file_sha256(entry["effective_config"]),
                "dataset": effective.get("dataset"),
            }
        )
        if entry["adapter"] == "openai_intervention":
            special_results, special_curves = _openai_diagnostics(
                entry,
                frozen_threshold=resolved["frozen_threshold"],
                confidence_level=resolved["bootstrap"]["confidence_level"],
            )
            results.extend(special_results)
            curves.extend(special_curves)
            continue
        rows = read_jsonl(entry["scores"])
        if entry["adapter"] == "process_any_path":
            rows = aggregate_human_trajectory_rows(rows)
        for filters, selected in _slices(rows, entry["slice_dimensions"]):
            slice_id = _slice_key(filters)
            diagnostic, curve = _binary_diagnostic(
                selected,
                frozen_threshold=resolved["frozen_threshold"],
                crossfit=resolved["crossfit"],
                bootstrap=resolved["bootstrap"],
                seed_salt=f"{entry['id']}:{slice_id}",
            )
            result = {
                "input_id": entry["id"],
                "adapter": entry["adapter"],
                "slice": filters,
                "slice_id": slice_id,
                **diagnostic,
            }
            results.append(result)
            curves.extend(
                {
                    "input_id": entry["id"],
                    "slice_id": slice_id,
                    "adapter": entry["adapter"],
                    **point,
                }
                for point in curve
            )

    first = next(iter(source_effective.values()))
    runtime = require_mapping(first.get("runtime"), "first source runtime")
    model = require_mapping(first.get("model"), "first source model")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        effective_config = {
            "schema_version": 1,
            "kind": "threshold_decomposition_effective_config",
            "diagnostic_only": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(resolved["config_path"]),
            "config_sha256": resolved["config_sha256"],
            "model_checkpoint": model.get("checkpoint"),
            "tokenizer": model.get("tokenizer"),
            "datasets": source_provenance,
            "split": "preserved formal evaluation suite",
            "max_sequence_length": runtime.get("max_sequence_length"),
            "truncation_policy": runtime.get("truncation_policy"),
            "padding_policy": runtime.get("padding_policy"),
            "batch_size": "not_applicable_cpu_postprocess",
            "learning_rate": "not_applicable_no_training",
            "epochs_or_steps": "not_applicable_no_training",
            "seed": {
                "crossfit": resolved["crossfit"]["seed"],
                "bootstrap": resolved["bootstrap"]["seed"],
            },
            "evaluation_metric": resolved["metric"],
            "frozen_threshold": resolved["frozen_threshold"],
            "threshold_source": str(resolved["threshold_source"]),
            "threshold_source_sha256": resolved["threshold_source_sha256"],
            "crossfit": resolved["crossfit"],
            "bootstrap": resolved["bootstrap"],
            "output_directory": str(output),
            "repository": repository_state(resolved["config_path"].parents[2]),
        }
        write_json(stage / "effective_config.json", effective_config)
        summary = {
            "schema_version": 1,
            "kind": "threshold_decomposition_summary",
            "diagnostic_only": True,
            "formal_score_policy": "frozen validation threshold remains authoritative",
            "oracle_policy": "evaluation labels are used only to measure recoverable threshold gap",
            "metric_definition": "gmean2 = TPR * TNR",
            "frozen_threshold": resolved["frozen_threshold"],
            "series": results,
            "outputs": {
                "effective_config": "effective_config.json",
                "table": "table.csv",
                "curves": "curves.jsonl",
            },
        }
        write_json(stage / "summary.json", summary)
        write_csv(stage / "table.csv", [_table_row(result) for result in results])
        write_jsonl(stage / "curves.jsonl", curves)
        if output.exists() and overwrite:
            for path in output.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        output.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            shutil.move(str(path), output / path.name)
        make_tree_group_accessible(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_config(output / "summary.json")
