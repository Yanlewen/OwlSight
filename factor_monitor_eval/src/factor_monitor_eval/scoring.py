"""Standalone binary artifact scorer."""

from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .adapters import normalize_prediction
from .io import read_json, read_jsonl, sha256, write_json, write_jsonl
from .metrics import score_decisions, score_rows


def _slices(rows: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    def run(items: list[dict[str, Any]]) -> dict[str, Any]:
        return score_rows(items, threshold) if threshold is not None else score_decisions(items)

    result: dict[str, Any] = {"overall": run(rows)}
    for field in ("factor_level", "source", "family"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(field) is not None:
                groups[str(row[field])].append(row)
        result[f"by_{field}"] = {key: run(groups[key]) for key in sorted(groups)}
    return result


def _has_score_field(rows: list[dict[str, Any]]) -> bool:
    return any("score" in row for row in rows)


def score_binary_artifact(
    *, artifact: Path, predictions: Path, output_directory: Path,
    threshold: float | None = None, label_mapping: dict[str, int] | None = None,
    threshold_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    queries_path = artifact / "eval_queries.jsonl"
    if not queries_path.is_file():
        raise ValueError(f"missing eval_queries.jsonl: {artifact}")
    queries = read_jsonl(queries_path)
    query_by_id = {row.get("query_id"): row for row in queries}
    if len(query_by_id) != len(queries) or None in query_by_id:
        raise ValueError("artifact has missing or duplicate query_id")
    raw_predictions = [normalize_prediction(row, label_mapping) for row in read_jsonl(predictions.resolve())]
    pred_by_id = {row["query_id"]: row for row in raw_predictions}
    if len(pred_by_id) != len(raw_predictions):
        raise ValueError("predictions have duplicate query_id")
    missing, extra = set(query_by_id) - set(pred_by_id), set(pred_by_id) - set(query_by_id)
    if missing or extra:
        raise ValueError(f"prediction IDs mismatch (missing={len(missing)}, extra={len(extra)})")
    joined = []
    use_score = threshold is not None
    for query_id in sorted(query_by_id):
        query, prediction = query_by_id[query_id], pred_by_id[query_id]
        if "label_id" in prediction and prediction["label_id"] != query.get("label_id"):
            raise ValueError(f"prediction {query_id} label_id disagrees with artifact")
        row = {
            "query_id": query_id,
            "sample_id": query.get("sample_id"),
            "trajectory_id": query.get("trajectory_id"),
            "factor_instance_id": query.get("factor_instance_id"),
            "factor_level": query.get("factor_level"),
            "family": query.get("family"),
            "source": query.get("source"),
            "label_id": query.get("label_id"),
            "decision": prediction.get("decision"),
            "parse_status": prediction.get("parse_status"),
        }
        if use_score:
            row["score"] = prediction.get("score")
        joined.append(row)
    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.stage-", dir=output_directory.parent))
    try:
        write_jsonl(stage / "joined_predictions.jsonl", joined)
        summary = {
            "schema_version": 1,
            "kind": "factor_monitor_eval_score_summary",
            "metric_adapter": "binary_factor_application",
            "artifact": str(artifact),
            "artifact_query_sha256": sha256(queries_path),
            "predictions": str(predictions.resolve()),
            "predictions_sha256": sha256(predictions.resolve()),
            "formal_metric": "gmean2",
            "threshold": threshold,
            "threshold_source": threshold_source,
            "threshold_rule": "score >= threshold" if threshold is not None else "decision label",
            "ranking_metrics": "available for score predictions only",
            "slices": _slices(joined, threshold),
            "outputs": {"joined_predictions": "joined_predictions.jsonl", "effective_config": "effective_config.json"},
        }
        write_json(stage / "summary.json", summary)
        write_json(stage / "effective_config.json", {
            "schema_version": 1,
            "kind": "factor_monitor_eval_effective_config",
            "scorer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "artifact": str(artifact), "predictions": str(predictions.resolve()),
            "threshold": threshold, "threshold_source": threshold_source,
            "label_mapping": label_mapping or {"applied": 1, "not_applied": 0},
            "output_directory": str(output_directory),
        })
        write_json(stage / "provenance.json", {
            "schema_version": 1,
            "artifact": str(artifact),
            "artifact_query_sha256": sha256(queries_path),
            "predictions": str(predictions.resolve()),
            "predictions_sha256": sha256(predictions.resolve()),
            "scorer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "formal_metric": "gmean2",
            "threshold": threshold,
            "threshold_source": threshold_source,
            "label_mapping": label_mapping or {"applied": 1, "not_applied": 0},
        })
        if output_directory.exists() and any(output_directory.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output_directory}")
            for path in output_directory.iterdir():
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
        output_directory.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir(): shutil.move(str(path), output_directory / path.name)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return read_json(output_directory / "summary.json")
