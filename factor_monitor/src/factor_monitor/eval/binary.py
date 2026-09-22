"""Manifest-validated scorer for ordinary binary Factor evaluation."""

from __future__ import annotations

import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..config import ConfigurationError, file_sha256, load_config, require_mapping
from ..io import read_jsonl, write_json, write_jsonl
from ..metrics import score_summary


OWNED_OUTPUTS = {
    "joined_predictions.jsonl",
    "summary.json",
    "effective_config.json",
}


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{name} must be finite and in [0,1]")
    return result


def _threshold_contract(config: dict[str, Any], query_sha256: str) -> float:
    if config.get("kind") != "eval_run":
        raise ConfigurationError("monitor effective config must have kind=eval_run")
    dataset = require_mapping(config.get("dataset"), "monitor effective dataset")
    files = require_mapping(dataset.get("files"), "monitor effective dataset.files")
    query_file = require_mapping(files.get("query_file"), "monitor effective query_file")
    if query_file.get("sha256") != query_sha256:
        raise ConfigurationError("monitor effective config query hash mismatch")
    evaluation = require_mapping(config.get("evaluation"), "monitor effective evaluation")
    policy = evaluation.get("threshold_policy")
    if policy not in {"frozen_validation", "fixed_choice_0.5", "zero_shot_forced_choice"}:
        raise ConfigurationError(
            "binary formal evaluation requires a frozen validation or fixed-choice threshold"
        )
    threshold = _finite_unit(evaluation.get("threshold_value"), "threshold_value")
    if policy in {"fixed_choice_0.5", "zero_shot_forced_choice"} and threshold != 0.5:
        raise ConfigurationError("fixed-choice threshold must be 0.5")
    return threshold


def _join(
    queries: list[dict[str, Any]], predictions: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    query_by_id: dict[str, dict[str, Any]] = {}
    for index, query in enumerate(queries):
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in query_by_id:
            raise ConfigurationError(f"query row {index} has missing/duplicate query_id")
        label = query.get("label_id")
        if label not in {0, 1} or isinstance(label, bool):
            raise ConfigurationError(f"query {query_id} label_id must be integer 0 or 1")
        query_by_id[query_id] = query

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        query_id = prediction.get("query_id")
        if (
            not isinstance(query_id, str)
            or not query_id
            or query_id in prediction_by_id
        ):
            raise ConfigurationError(f"prediction row {index} has missing/duplicate query_id")
        score = _finite_unit(prediction.get("score"), f"prediction {query_id}.score")
        if "label_id" in prediction and prediction["label_id"] != query_by_id.get(query_id, {}).get("label_id"):
            raise ConfigurationError(f"prediction {query_id} label_id disagrees with frozen query")
        prediction_by_id[query_id] = {"query_id": query_id, "score": score}

    missing = set(query_by_id) - set(prediction_by_id)
    extra = set(prediction_by_id) - set(query_by_id)
    if missing or extra:
        raise ConfigurationError(
            "prediction IDs must exactly match queries "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    joined: list[dict[str, Any]] = []
    for query_id in sorted(query_by_id):
        query = query_by_id[query_id]
        prediction = prediction_by_id[query_id]
        joined.append(
            {
                "query_id": query_id,
                "sample_id": query.get("sample_id"),
                "trajectory_id": query.get("trajectory_id"),
                "factor_instance_id": query.get("factor_instance_id"),
                "factor_level": query.get("factor_level"),
                "family": query.get("family"),
                "source": query.get("source"),
                "label_id": int(query["label_id"]),
                "score": prediction["score"],
            }
        )
    return joined


def _score_slices(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    def summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        result = score_summary(
            [{"label_id": row["label_id"], "score": row["score"]} for row in selected],
            threshold,
        )
        result["statistics_unit"] = "query"
        result["class_counts"] = {
            "not_applied": sum(row["label_id"] == 0 for row in selected),
            "applied": sum(row["label_id"] == 1 for row in selected),
        }
        return result

    result: dict[str, Any] = {"overall": summary(rows)}
    for field in ("factor_level", "source", "family"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if value is not None:
                groups[str(value)].append(row)
        result[f"by_{field}"] = {
            key: summary(groups[key]) for key in sorted(groups)
        }
    return result


def score_binary_artifact(
    *,
    artifact: Path,
    predictions: Path,
    monitor_effective_config: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score a frozen binary artifact using persisted predictions only."""

    artifact = artifact.resolve()
    queries_path = artifact / "eval_queries.jsonl"
    artifact_config_path = artifact / "effective_config.json"
    if not queries_path.is_file() or not artifact_config_path.is_file():
        raise ConfigurationError(
            "binary artifact must contain eval_queries.jsonl and effective_config.json"
        )
    artifact_config = load_config(artifact_config_path)
    evaluation = require_mapping(artifact_config.get("evaluation"), "artifact evaluation")
    if evaluation.get("metric_adapter") != "binary_factor_application":
        raise ConfigurationError("artifact does not use the binary_factor_application adapter")

    queries = read_jsonl(queries_path)
    query_sha256 = file_sha256(queries_path)
    monitor_path = monitor_effective_config.resolve()
    monitor_config = load_config(monitor_path)
    threshold = _threshold_contract(monitor_config, query_sha256)
    predictions = predictions.resolve()
    if not predictions.is_file():
        raise ConfigurationError(f"predictions file does not exist: {predictions}")
    joined = _join(queries, read_jsonl(predictions))
    slices = _score_slices(joined, threshold)

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.stage-", dir=output_directory.parent)
    )
    try:
        write_jsonl(stage / "joined_predictions.jsonl", joined)
        effective = {
            "schema_version": 1,
            "kind": "factor_monitor_binary_score",
            "metric_adapter": "binary_factor_application",
            "scorer": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "artifact": str(artifact),
            "artifact_effective_config_sha256": file_sha256(artifact_config_path),
            "query_sha256": query_sha256,
            "predictions": str(predictions),
            "predictions_sha256": file_sha256(predictions),
            "monitor_effective_config": str(monitor_path),
            "monitor_effective_config_sha256": file_sha256(monitor_path),
            "threshold": threshold,
            "threshold_policy": monitor_config["evaluation"]["threshold_policy"],
            "threshold_source": monitor_config["evaluation"].get("threshold_source"),
            "threshold_source_sha256": monitor_config["evaluation"].get(
                "threshold_source_sha256"
            ),
            "output_directory": str(output_directory),
        }
        write_json(stage / "effective_config.json", effective)
        summary = {
            "schema_version": 1,
            "kind": "factor_monitor_binary_score_summary",
            "metric_adapter": "binary_factor_application",
            "rows": len(joined),
            "threshold": threshold,
            "formal_metric": "gmean2",
            "threshold_rule": "score >= threshold",
            "slices": slices,
            "outputs": {
                "joined_predictions": "joined_predictions.jsonl",
                "effective_config": "effective_config.json",
            },
        }
        write_json(stage / "summary.json", summary)

        if output_directory.exists():
            unexpected = sorted(
                path.name for path in output_directory.iterdir() if path.name not in OWNED_OUTPUTS
            )
            if unexpected:
                raise ConfigurationError(
                    f"refusing to overwrite unowned score outputs: {unexpected}"
                )
            if not overwrite and any(output_directory.iterdir()):
                raise FileExistsError(
                    f"score output is not empty; pass --overwrite: {output_directory}"
                )
            if overwrite:
                for path in output_directory.iterdir():
                    path.unlink()
        output_directory.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            shutil.move(str(path), output_directory / path.name)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_config(output_directory / "summary.json")
