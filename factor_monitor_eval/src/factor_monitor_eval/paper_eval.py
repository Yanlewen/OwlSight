"""One-command rescoring for the frozen paper models and mixed metric adapters."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

from .io import read_json, read_jsonl, sha256, write_json, write_jsonl
from .manifest import load_manifest, resolve_eval_path
from .runner import run_suite


BOUND_TO_EVAL_ID = {
    "aime": "process_aime",
    "gpqa": "process_gpqa",
    "sabotage": "shadearena",
}


def _resolve_path(value: str, *, config_path: Path) -> Path:
    raw = Path(value).expanduser()
    candidates = (
        raw,
        config_path.parent / raw,
        Path(__file__).resolve().parents[3] / raw,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[-1].resolve()


def _load_models(path: Path) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "factor_monitor_paper_models":
        raise ValueError("models config must have kind=factor_monitor_paper_models")
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models config must contain a non-empty models list")
    ids = [item.get("id") for item in models if isinstance(item, dict)]
    if len(ids) != len(models) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every model requires a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError("model ids must be unique")
    return models


def _load_bound_suite(path: Path) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[3] / "factor_monitor" / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from factor_monitor.full_eval import load_suite

    return load_suite(path)


def _prediction_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    direct = Path(str(entry["direct_scores"]))
    if not direct.is_file():
        raise FileNotFoundError(f"missing direct scores: {direct}")
    rows = read_jsonl(direct)
    window_value = entry.get("window_scores")
    if window_value:
        window = Path(str(window_value))
        if window.is_file():
            for row in read_jsonl(window):
                aggregations = row.get("aggregation_scores")
                if isinstance(aggregations, dict):
                    if "top2_mean" not in aggregations:
                        raise ValueError(f"{window} row lacks aggregation_scores.top2_mean")
                    row = {**row, "score": aggregations["top2_mean"], "aggregation": "top2_mean"}
                rows.append(row)
    query_ids = [row.get("query_id") for row in rows]
    if any(not isinstance(item, str) or not item for item in query_ids):
        raise ValueError(f"{entry['id']} predictions contain an invalid query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{entry['id']} predictions contain duplicate query_id values")
    return rows


def _scoring_contract(
    *,
    source: Path,
    artifact: Path,
    output: Path,
) -> dict[str, Any]:
    value = copy.deepcopy(read_json(source))
    query_file = artifact / "eval_queries.jsonl"
    if not query_file.is_file():
        raise FileNotFoundError(query_file)
    dataset = value.setdefault("dataset", {})
    dataset["directory"] = str(artifact)
    files = dataset.setdefault("files", {})
    files["query_file"] = {"path": str(query_file), "sha256": sha256(query_file)}
    trajectory_file = artifact / "trajectories.jsonl"
    if trajectory_file.is_file():
        files["trajectory_file"] = {
            "path": str(trajectory_file),
            "sha256": sha256(trajectory_file),
        }
    value["output_directory"] = str(output)
    value["derived_for_paper_rescoring"] = {
        "source_effective_config": str(source),
        "source_effective_config_sha256": sha256(source),
        "change": "query artifact identity rebound after lossless direct/window merge",
    }
    return value


def _prepare_model_inputs(
    *,
    manifest_path: Path,
    bound_suite_path: Path,
    model_root: Path,
) -> tuple[Path, Path]:
    manifest = load_manifest(manifest_path)
    active = {str(item["id"]): item for item in manifest["benchmarks"]}
    suite = _load_bound_suite(bound_suite_path)
    predictions_root = model_root / "predictions"
    configs_root = model_root / "monitor_configs"
    prepared: set[str] = set()
    for bound_entry in suite["benchmarks"]:
        bound_id = str(bound_entry["id"])
        eval_id = BOUND_TO_EVAL_ID.get(bound_id, bound_id)
        if eval_id not in active:
            continue
        rows = _prediction_rows(bound_entry)
        effective = Path(str(bound_entry["output_directory"])) / "direct" / "effective_config.json"
        if not effective.is_file():
            raise FileNotFoundError(f"missing monitor effective config: {effective}")
        effective_value = read_json(effective)
        if active[eval_id]["track"] == "intervention":
            threshold = effective_value.get("evaluation", {}).get("threshold_value")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError(f"{eval_id} monitor config lacks a numeric threshold_value")
            rows = [
                {
                    **row,
                    "prediction_label": (
                        "applied" if float(row["score"]) >= float(threshold) else "not_applied"
                    ),
                }
                for row in rows
            ]
        artifact = resolve_eval_path(active[eval_id]["artifact"], manifest_path=manifest_path)
        expected_ids = {
            str(row["query_id"])
            for row in read_jsonl(artifact / "eval_queries.jsonl")
        }
        actual_ids = {str(row["query_id"]) for row in rows}
        if actual_ids != expected_ids:
            raise ValueError(
                f"{eval_id} score coverage mismatch "
                f"(missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)})"
            )
        write_jsonl(predictions_root / f"{eval_id}.jsonl", rows)
        write_json(
            configs_root / f"{eval_id}.json",
            _scoring_contract(
                source=effective,
                artifact=artifact,
                output=model_root / "scored" / eval_id,
            ),
        )
        prepared.add(eval_id)
    missing = sorted(set(active) - prepared)
    if missing:
        raise ValueError(f"bound suite does not cover active evaluation ids: {missing}")
    return predictions_root, configs_root


def run_paper_eval(
    *,
    manifest_path: Path,
    models_config: Path,
    output_root: Path,
    model_ids: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize preserved monitor scores and apply all declared metric adapters."""

    manifest_path = manifest_path.resolve()
    models_config = models_config.resolve()
    selected = set(model_ids or [])
    models = _load_models(models_config)
    if selected:
        known = {str(item["id"]) for item in models}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"unknown model ids: {unknown}")
        models = [item for item in models if item["id"] in selected]
    results: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        model_root = output_root.resolve() / model_id
        try:
            bound_suite_value = model.get("bound_suite")
            if bound_suite_value:
                bound_suite = _resolve_path(str(bound_suite_value), config_path=models_config)
                predictions_root, configs_root = _prepare_model_inputs(
                    manifest_path=manifest_path,
                    bound_suite_path=bound_suite,
                    model_root=model_root,
                )
                source = {"bound_suite": str(bound_suite)}
            else:
                predictions_value = model.get("predictions_root")
                configs_value = model.get("monitor_config_root")
                if not isinstance(predictions_value, str) or not predictions_value:
                    raise ValueError(f"{model_id} requires predictions_root")
                if not isinstance(configs_value, str) or not configs_value:
                    raise ValueError(f"{model_id} requires monitor_config_root")
                predictions_root = _resolve_path(
                    predictions_value,
                    config_path=models_config,
                )
                configs_root = _resolve_path(
                    configs_value,
                    config_path=models_config,
                )
                if not predictions_root.is_dir() or not configs_root.is_dir():
                    raise FileNotFoundError(
                        f"{model_id} requires existing predictions_root and monitor_config_root"
                    )
                source = {
                    "predictions_root": str(predictions_root),
                    "monitor_config_root": str(configs_root),
                }
            summary = run_suite(
                manifest_path=manifest_path,
                predictions_root=predictions_root,
                output_root=model_root / "scored",
                monitor_config_root=configs_root,
                overwrite=overwrite,
            )
            results.append(
                {
                    "id": model_id,
                    "status": "ok" if summary["complete"] else "failed",
                    **source,
                    "summary": summary,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": model_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    combined = {
        "schema_version": 1,
        "kind": "factor_monitor_paper_eval_summary",
        "manifest": str(manifest_path),
        "models_config": str(models_config),
        "complete": all(item["status"] == "ok" for item in results),
        "models": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "summary.json", combined)
    return combined
