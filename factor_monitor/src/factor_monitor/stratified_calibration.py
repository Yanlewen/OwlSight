"""Fit and audit validation-only hierarchical monitor thresholds."""

from __future__ import annotations

import itertools
import math
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from .io import make_tree_group_accessible, read_jsonl, write_csv, write_json
from .metrics import (
    best_threshold_metrics,
    grouped_fold_partitions,
    score_summary,
    threshold_metrics,
)
from .process_eval import aggregate_human_trajectory_rows


OWNED_OUTPUTS = {"effective_config.json", "summary.json", "thresholds.json", "table.csv"}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _metric_value(summary: Mapping[str, Any], metric: str) -> float:
    value = summary.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"cross-fit {metric} is undefined")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"cross-fit {metric} is not finite")
    return result


def _support(rows: Sequence[dict[str, Any]], group_field: str) -> dict[str, Any]:
    counts = Counter(int(row["label_id"]) for row in rows)
    return {
        "rows": len(rows),
        "groups": len({str(row[group_field]) for row in rows}),
        "negative_rows": counts[0],
        "positive_rows": counts[1],
    }


def _binary_predictions(
    rows: Iterable[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            "label_id": int(row["label_id"]),
            "score": float(float(row["score"]) >= threshold),
        }
        for row in rows
    ]


def _crossfit_local_vs_parent(
    local_rows: list[dict[str, Any]],
    *,
    parent_rows: list[dict[str, Any]] | None,
    group_field: str,
    folds: int,
    seed: int,
    metric: str,
) -> dict[str, Any]:
    try:
        partitions = grouped_fold_partitions(
            local_rows, group_field=group_field, folds=folds, seed=seed
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    local_oof: list[dict[str, Any]] = []
    parent_oof: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for index, (local_train, test) in enumerate(partitions):
        held_out_groups = {str(row[group_field]) for row in test}
        try:
            local_selected = best_threshold_metrics(local_train, metric=metric)
        except ValueError as exc:
            raise ConfigurationError(f"fold {index} local threshold is undefined: {exc}") from exc
        local_threshold = float(local_selected["threshold"])
        local_oof.extend(_binary_predictions(test, local_threshold))

        parent_threshold: float | None = None
        if parent_rows is not None:
            parent_train = [
                row
                for row in parent_rows
                if str(row[group_field]) not in held_out_groups
            ]
            try:
                parent_selected = best_threshold_metrics(parent_train, metric=metric)
            except ValueError as exc:
                raise ConfigurationError(
                    f"fold {index} parent threshold is undefined: {exc}"
                ) from exc
            parent_threshold = float(parent_selected["threshold"])
            parent_oof.extend(_binary_predictions(test, parent_threshold))
        fold_rows.append(
            {
                "fold": index,
                "train_rows": len(local_train),
                "test_rows": len(test),
                "test_groups": len(held_out_groups),
                "local_threshold": local_threshold,
                "parent_threshold": parent_threshold,
            }
        )

    local_summary = threshold_metrics(local_oof, 0.5)
    local_summary["threshold"] = None
    parent_summary = None
    if parent_rows is not None:
        parent_summary = threshold_metrics(parent_oof, 0.5)
        parent_summary["threshold"] = None
    return {
        "folds": folds,
        "seed": seed,
        "group_field": group_field,
        "assignment": "label-presence-stratified groups",
        "local_out_of_fold": local_summary,
        "parent_out_of_fold": parent_summary,
        "fold_results": fold_rows,
    }


def _fit_node(
    *,
    node_id: str,
    filters: dict[str, str],
    rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]] | None,
    parent_node: dict[str, Any] | None,
    group_field: str,
    folds: int,
    seed: int,
    metric: str,
) -> dict[str, Any]:
    try:
        fitted = best_threshold_metrics(rows, metric=metric)
        crossfit = _crossfit_local_vs_parent(
            rows,
            parent_rows=parent_rows,
            group_field=group_field,
            folds=folds,
            seed=seed,
            metric=metric,
        )
    except (ValueError, ConfigurationError) as exc:
        if parent_node is None:
            raise ConfigurationError(f"root calibration is not identifiable: {exc}") from exc
        return {
            "node_id": node_id,
            "filters": filters,
            "support": _support(rows, group_field),
            "fitted_threshold": None,
            "selected_threshold": parent_node["selected_threshold"],
            "selected_source": parent_node["node_id"],
            "selection_reason": f"fallback_unidentifiable:{exc}",
            "crossfit": None,
        }

    selected_threshold = float(fitted["threshold"])
    selected_source = node_id
    reason = "root_threshold"
    if parent_node is not None:
        local_value = _metric_value(crossfit["local_out_of_fold"], metric)
        parent_value = _metric_value(crossfit["parent_out_of_fold"], metric)
        if local_value <= parent_value:
            selected_threshold = float(parent_node["selected_threshold"])
            selected_source = str(parent_node["node_id"])
            reason = "fallback_parent_crossfit_not_improved"
        else:
            reason = "local_crossfit_improved_parent"
    return {
        "node_id": node_id,
        "filters": filters,
        "support": _support(rows, group_field),
        "fitted_threshold": float(fitted["threshold"]),
        "fitted_metrics": fitted,
        "selected_threshold": selected_threshold,
        "selected_source": selected_source,
        "selection_reason": reason,
        "crossfit": crossfit,
    }


def _enrich_scores(
    scores: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    query_by_id: dict[str, dict[str, Any]] = {}
    for index, query in enumerate(queries):
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in query_by_id:
            raise ConfigurationError(f"query row {index} has missing/duplicate query_id")
        query_by_id[query_id] = query
    score_by_id: dict[str, dict[str, Any]] = {}
    for index, score in enumerate(scores):
        query_id = score.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in score_by_id:
            raise ConfigurationError(f"score row {index} has missing/duplicate query_id")
        score_by_id[query_id] = score
    if set(query_by_id) != set(score_by_id):
        raise ConfigurationError(
            "calibration score/query IDs differ "
            f"(missing={len(set(query_by_id)-set(score_by_id))}, "
            f"extra={len(set(score_by_id)-set(query_by_id))})"
        )

    result: list[dict[str, Any]] = []
    for query_id, query in query_by_id.items():
        score = score_by_id[query_id]
        row = {
            "query_id": query_id,
            "trajectory_id": query["trajectory_id"],
            "label_id": query["label_id"],
            "score": score["score"],
            "family": query["family"],
            "factor_level": query["factor_level"],
        }
        scope = require_mapping(query.get("metadata", {}), f"query {query_id}.metadata").get(
            "factor_scope"
        )
        if scope is not None:
            row["factor_scope"] = scope
        for field in ("trajectory_id", "label_id", "family", "factor_level"):
            if field in score and score[field] != row[field]:
                raise ConfigurationError(f"score {query_id} disagrees on {field}")
        # Reuse the binary metric validator for score and label validation.
        score_summary([row], 0.5)
        result.append(row)
    return result


def _load_source(
    *, scores_path: Path, queries_path: Path, effective_path: Path
) -> list[dict[str, Any]]:
    effective = load_config(effective_path)
    dataset = require_mapping(effective.get("dataset"), "calibration effective dataset")
    files = require_mapping(dataset.get("files"), "calibration effective dataset.files")
    query_entry = require_mapping(files.get("query_file"), "calibration effective query_file")
    if query_entry.get("sha256") != file_sha256(queries_path):
        raise ConfigurationError("calibration effective config is not bound to query file")
    return _enrich_scores(read_jsonl(scores_path), read_jsonl(queries_path))


def _lookup_threshold(
    thresholds: Mapping[tuple[str, str, str], float], row: Mapping[str, Any], paradigm: str
) -> float:
    key = (paradigm, str(row.get("family")), str(row.get("factor_level")))
    if key not in thresholds:
        raise ConfigurationError(f"no calibrated threshold for {key}")
    return thresholds[key]


def _slice_rows(
    rows: list[dict[str, Any]], dimensions: Sequence[str]
) -> list[tuple[str, list[dict[str, Any]]]]:
    result = [("overall", rows)]
    for width in range(1, len(dimensions) + 1):
        for selected in itertools.combinations(dimensions, width):
            values = sorted({tuple(row.get(key) for key in selected) for row in rows}, key=str)
            for combination in values:
                filters = dict(zip(selected, combination, strict=True))
                chosen = [
                    row
                    for row in rows
                    if all(row.get(key) == value for key, value in filters.items())
                ]
                slice_id = "__".join(f"{key}={filters[key]}" for key in sorted(filters))
                result.append((slice_id, chosen))
    return result


def _evaluate_binary(
    rows: list[dict[str, Any]],
    *,
    paradigm: str,
    baseline_threshold: float,
    thresholds: Mapping[tuple[str, str, str], float],
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for slice_id, selected in _slice_rows(rows, dimensions):
        calibrated = [
            {
                "label_id": int(row["label_id"]),
                "score": float(
                    float(row["score"])
                    >= _lookup_threshold(thresholds, row, paradigm)
                ),
            }
            for row in selected
        ]
        baseline = score_summary(selected, baseline_threshold)
        result = score_summary(calibrated, 0.5)
        table.append(
            {
                "slice_id": slice_id,
                "rows": len(selected),
                "baseline_gmean2": baseline["gmean2"],
                "calibrated_gmean2": result["gmean2"],
                "delta_gmean2": (
                    None
                    if baseline["gmean2"] is None or result["gmean2"] is None
                    else float(result["gmean2"]) - float(baseline["gmean2"])
                ),
                "baseline": baseline,
                "calibrated": result,
            }
        )
    return table


def _evaluate_process(
    rows: list[dict[str, Any]],
    *,
    paradigm: str,
    baseline_threshold: float,
    thresholds: Mapping[tuple[str, str, str], float],
) -> list[dict[str, Any]]:
    baseline_trajectory = aggregate_human_trajectory_rows(rows)
    thresholded = [
        {
            **row,
            "score": float(
                float(row["score"]) >= _lookup_threshold(thresholds, row, paradigm)
            ),
        }
        for row in rows
    ]
    calibrated_trajectory = aggregate_human_trajectory_rows(thresholded)
    table: list[dict[str, Any]] = []
    for level in ("L1", "L3"):
        baseline = [row for row in baseline_trajectory if row["factor_level"] == level]
        calibrated = [row for row in calibrated_trajectory if row["factor_level"] == level]
        base_summary = score_summary(baseline, baseline_threshold)
        calibrated_summary = score_summary(calibrated, 0.5)
        table.append(
            {
                "slice_id": f"factor_level={level}",
                "rows": len(calibrated),
                "baseline_gmean2": base_summary["gmean2"],
                "calibrated_gmean2": calibrated_summary["gmean2"],
                "delta_gmean2": (
                    None
                    if base_summary["gmean2"] is None or calibrated_summary["gmean2"] is None
                    else float(calibrated_summary["gmean2"]) - float(base_summary["gmean2"])
                ),
                "baseline": base_summary,
                "calibrated": calibrated_summary,
                "paper_aligned_tpr": {
                    "baseline": sum(float(row["score"]) >= baseline_threshold for row in baseline)
                    / len(baseline),
                    "calibrated": sum(float(row["score"]) >= 0.5 for row in calibrated)
                    / len(calibrated),
                    "note": "solver-correct Process convention; GMean2 equals TPR",
                },
            }
        )
    return table


def _resolve_config(path: Path) -> dict[str, Any]:
    raw = load_config(path)
    fields = {
        "schema_version",
        "kind",
        "diagnostic_only",
        "calibration",
        "hierarchy",
        "family_paradigms",
        "unavailable_intervention",
        "evaluations",
        "output",
    }
    require_keys(raw, fields, "stratified calibration config")
    reject_unknown(raw, fields, "stratified calibration config")
    if raw["schema_version"] != 1 or raw["kind"] != "stratified_threshold_calibration":
        raise ConfigurationError("invalid stratified calibration kind/schema")
    if raw["diagnostic_only"] is not True:
        raise ConfigurationError("first calibration pass must set diagnostic_only=true")
    base = path.parent

    calibration = require_mapping(raw["calibration"], "calibration")
    calibration_fields = {
        "scores",
        "effective_config",
        "queries",
        "baseline_threshold_source",
    }
    require_keys(calibration, calibration_fields, "calibration")
    reject_unknown(calibration, calibration_fields, "calibration")
    resolved_calibration = {
        key: resolve_path(value, relative_to=base) for key, value in calibration.items()
    }
    for key, value in resolved_calibration.items():
        if not value.is_file():
            raise ConfigurationError(f"calibration.{key} does not exist: {value}")

    hierarchy = require_mapping(raw["hierarchy"], "hierarchy")
    hierarchy_fields = {
        "dimensions",
        "selection_metric",
        "folds",
        "seed",
        "group_field",
        "child_policy",
    }
    require_keys(hierarchy, hierarchy_fields, "hierarchy")
    reject_unknown(hierarchy, hierarchy_fields, "hierarchy")
    if hierarchy["dimensions"] != ["family", "factor_level"]:
        raise ConfigurationError("first calibration hierarchy must be [family, factor_level]")
    if hierarchy["selection_metric"] != "gmean2":
        raise ConfigurationError("stratified calibration selection_metric must be gmean2")
    if hierarchy["child_policy"] != "crossfit_strictly_improves_parent":
        raise ConfigurationError("unsupported child threshold policy")
    normalized_hierarchy = {
        **hierarchy,
        "folds": _integer(hierarchy["folds"], "hierarchy.folds", minimum=2),
        "seed": _integer(hierarchy["seed"], "hierarchy.seed", minimum=0),
        "group_field": _text(hierarchy["group_field"], "hierarchy.group_field"),
    }

    paradigms = require_mapping(raw["family_paradigms"], "family_paradigms")
    if not paradigms:
        raise ConfigurationError("family_paradigms cannot be empty")
    normalized_paradigms = {
        _text(key, "family_paradigms key"): _text(value, f"family_paradigms.{key}")
        for key, value in paradigms.items()
    }
    if set(normalized_paradigms.values()) - {"process", "outcome"}:
        raise ConfigurationError("binary validation may calibrate only process/outcome strata")

    unavailable = raw["unavailable_intervention"]
    if not isinstance(unavailable, list) or not unavailable:
        raise ConfigurationError("unavailable_intervention must explicitly list blocked targets")
    normalized_unavailable: list[dict[str, Any]] = []
    for index, value in enumerate(unavailable):
        item = require_mapping(value, f"unavailable_intervention[{index}]")
        require_keys(item, {"family", "factor_levels", "reason"}, f"unavailable_intervention[{index}]")
        reject_unknown(item, {"family", "factor_levels", "reason"}, f"unavailable_intervention[{index}]")
        if item["factor_levels"] != ["L1", "L3"]:
            raise ConfigurationError("intervention blocked target levels must be [L1, L3]")
        normalized_unavailable.append(
            {
                "paradigm": "intervention",
                "family": _text(item["family"], "unavailable intervention family"),
                "factor_levels": ["L1", "L3"],
                "reason": _text(item["reason"], "unavailable intervention reason"),
            }
        )

    evaluations = raw["evaluations"]
    if not isinstance(evaluations, list):
        raise ConfigurationError("evaluations must be a list")
    resolved_evaluations: list[dict[str, Any]] = []
    for index, value in enumerate(evaluations):
        item = require_mapping(value, f"evaluations[{index}]")
        fields = {"id", "adapter", "paradigm", "scores", "slice_dimensions"}
        require_keys(item, fields, f"evaluations[{index}]")
        reject_unknown(item, fields, f"evaluations[{index}]")
        adapter = _text(item["adapter"], f"evaluations[{index}].adapter")
        if adapter not in {"binary", "process_any_path"}:
            raise ConfigurationError("calibrated evaluation adapter must be binary or process_any_path")
        dimensions = item["slice_dimensions"]
        if not isinstance(dimensions, list) or any(not isinstance(x, str) for x in dimensions):
            raise ConfigurationError("slice_dimensions must be a list of strings")
        score_path = resolve_path(item["scores"], relative_to=base)
        if not score_path.is_file():
            raise ConfigurationError(f"calibrated evaluation scores do not exist: {score_path}")
        resolved_evaluations.append(
            {
                "id": _text(item["id"], f"evaluations[{index}].id"),
                "adapter": adapter,
                "paradigm": _text(item["paradigm"], f"evaluations[{index}].paradigm"),
                "scores": score_path,
                "slice_dimensions": dimensions,
            }
        )
    ids = [item["id"] for item in resolved_evaluations]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("calibrated evaluation IDs must be unique")

    output = require_mapping(raw["output"], "output")
    require_keys(output, {"directory"}, "output")
    reject_unknown(output, {"directory"}, "output")
    return {
        "calibration": resolved_calibration,
        "hierarchy": normalized_hierarchy,
        "family_paradigms": normalized_paradigms,
        "unavailable_intervention": normalized_unavailable,
        "evaluations": resolved_evaluations,
        "output": resolve_path(output["directory"], relative_to=base),
    }


def run_stratified_calibration(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    config_path = config_path.resolve()
    resolved = _resolve_config(config_path)
    output = resolved["output"]
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"calibration output is not empty; pass --overwrite: {output}")

    calibration = resolved["calibration"]
    rows = _load_source(
        scores_path=calibration["scores"],
        queries_path=calibration["queries"],
        effective_path=calibration["effective_config"],
    )
    observed_families = {str(row["family"]) for row in rows}
    if observed_families != set(resolved["family_paradigms"]):
        raise ConfigurationError(
            "calibration families differ from family_paradigms "
            f"(observed={sorted(observed_families)}, configured={sorted(resolved['family_paradigms'])})"
        )
    baseline_payload = load_config(calibration["baseline_threshold_source"])
    baseline_threshold = baseline_payload.get("threshold")
    if isinstance(baseline_threshold, bool) or not isinstance(baseline_threshold, (int, float)):
        raise ConfigurationError("baseline threshold source lacks a numeric threshold")
    baseline_threshold = float(baseline_threshold)

    hierarchy = resolved["hierarchy"]
    common = {
        "group_field": hierarchy["group_field"],
        "folds": hierarchy["folds"],
        "seed": hierarchy["seed"],
        "metric": hierarchy["selection_metric"],
    }
    root = _fit_node(
        node_id="global",
        filters={},
        rows=rows,
        parent_rows=None,
        parent_node=None,
        **common,
    )
    nodes = [root]
    family_nodes: dict[str, dict[str, Any]] = {}
    for family in sorted(observed_families):
        family_rows = [row for row in rows if row["family"] == family]
        node = _fit_node(
            node_id=f"family={family}",
            filters={"family": family},
            rows=family_rows,
            parent_rows=rows,
            parent_node=root,
            **common,
        )
        nodes.append(node)
        family_nodes[family] = node

    leaves: list[dict[str, Any]] = []
    threshold_lookup: dict[tuple[str, str, str], float] = {}
    for family in sorted(observed_families):
        family_rows = [row for row in rows if row["family"] == family]
        for level in ("L1", "L3"):
            level_rows = [row for row in family_rows if row["factor_level"] == level]
            node = _fit_node(
                node_id=f"family={family}__factor_level={level}",
                filters={"family": family, "factor_level": level},
                rows=level_rows,
                parent_rows=family_rows,
                parent_node=family_nodes[family],
                **common,
            )
            paradigm = resolved["family_paradigms"][family]
            node["paradigm"] = paradigm
            nodes.append(node)
            leaves.append(node)
            threshold_lookup[(paradigm, family, level)] = float(node["selected_threshold"])

    evaluation_summaries: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for entry in resolved["evaluations"]:
        eval_rows = read_jsonl(entry["scores"])
        if entry["adapter"] == "process_any_path":
            results = _evaluate_process(
                eval_rows,
                paradigm=entry["paradigm"],
                baseline_threshold=baseline_threshold,
                thresholds=threshold_lookup,
            )
        else:
            results = _evaluate_binary(
                eval_rows,
                paradigm=entry["paradigm"],
                baseline_threshold=baseline_threshold,
                thresholds=threshold_lookup,
                dimensions=entry["slice_dimensions"],
            )
        evaluation_summaries.append({**entry, "scores": str(entry["scores"]), "results": results})
        table.extend(
            {
                "evaluation_id": entry["id"],
                "adapter": entry["adapter"],
                "paradigm": entry["paradigm"],
                **{key: value for key, value in result.items() if key not in {"baseline", "calibrated", "paper_aligned_tpr"}},
                "baseline_process_tpr": (result.get("paper_aligned_tpr") or {}).get("baseline"),
                "calibrated_process_tpr": (result.get("paper_aligned_tpr") or {}).get("calibrated"),
            }
            for result in results
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        thresholds_payload = {
            "schema_version": 1,
            "kind": "stratified_threshold_bundle",
            "diagnostic_only": True,
            "baseline_threshold": baseline_threshold,
            "selection_metric": hierarchy["selection_metric"],
            "hierarchy": hierarchy["dimensions"],
            "nodes": nodes,
            "leaf_thresholds": [
                {
                    "paradigm": node["paradigm"],
                    **node["filters"],
                    "threshold": node["selected_threshold"],
                    "selected_source": node["selected_source"],
                }
                for node in leaves
            ],
            "unavailable_intervention": resolved["unavailable_intervention"],
        }
        write_json(stage / "thresholds.json", thresholds_payload)
        effective = {
            "schema_version": 1,
            "kind": "stratified_threshold_calibration_effective_config",
            "diagnostic_only": True,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "calibration": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in calibration.items()
            },
            "source_monitor_configuration": load_config(calibration["effective_config"]),
            "hierarchy": hierarchy,
            "family_paradigms": resolved["family_paradigms"],
            "baseline_threshold": baseline_threshold,
            "evaluation_inputs": [
                {
                    "id": entry["id"],
                    "adapter": entry["adapter"],
                    "paradigm": entry["paradigm"],
                    "scores": {
                        "path": str(entry["scores"]),
                        "sha256": file_sha256(entry["scores"]),
                    },
                    "slice_dimensions": entry["slice_dimensions"],
                }
                for entry in resolved["evaluations"]
            ],
            "output_directory": str(output),
            "repository": repository_state(Path(__file__).resolve().parents[3]),
        }
        write_json(stage / "effective_config.json", effective)
        summary = {
            "schema_version": 1,
            "kind": "stratified_threshold_calibration_summary",
            "diagnostic_only": True,
            "calibration_rows": len(rows),
            "baseline_threshold": baseline_threshold,
            "leaf_thresholds": thresholds_payload["leaf_thresholds"],
            "unavailable_intervention": resolved["unavailable_intervention"],
            "evaluations": evaluation_summaries,
            "formal_policy": "formal eval labels were read only for post-fit reporting; calibration used validation scores only",
            "outputs": {
                "thresholds": "thresholds.json",
                "effective_config": "effective_config.json",
                "table": "table.csv",
            },
        }
        write_json(stage / "summary.json", summary)
        write_csv(stage / "table.csv", table)
        if output.exists():
            unexpected = sorted(path.name for path in output.iterdir() if path.name not in OWNED_OUTPUTS)
            if unexpected:
                raise ConfigurationError(f"refusing to overwrite unowned calibration outputs: {unexpected}")
            if not overwrite and any(output.iterdir()):
                raise FileExistsError(f"calibration output is not empty: {output}")
            shutil.rmtree(output)
        stage.replace(output)
        make_tree_group_accessible(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary
