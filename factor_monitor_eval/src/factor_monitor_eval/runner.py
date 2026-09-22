"""Manifest-driven scoring runner.

Inference is intentionally out-of-process: collaborators can generate one
prediction file per benchmark with any provider, then run this CPU-only runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import sha256, write_json
from .manifest import load_manifest, resolve_eval_path
from .report import write_report, write_suite_report
from .scoring import score_binary_artifact
from .io import read_jsonl
from .thresholds import resolve_threshold


def _level_metrics(summary: dict[str, Any]) -> dict[str, list[float]]:
    """Collect declared L1/L3 G-Mean² values without inventing an aggregate."""
    result: dict[str, list[float]] = {"L1": [], "L3": []}
    slices = summary.get("slices", {})
    for level, value in slices.get("by_factor_level", {}).items():
        if level in result and value.get("gmean2") is not None:
            result[level].append(float(value["gmean2"]))
    # Process exposes two semantic views; the factor-level human-semantic view
    # is the declared comparison view, while trajectory-any-path is retained
    # in the per-unit report and is not double-counted in the suite macro.
    for level, value in summary.get("factor_level_human_semantic", {}).items():
        if level in result and value.get("gmean2") is not None:
            result[level].append(float(value["gmean2"]))
    for item in summary.get("by_source_and_factor_level", []):
        level = item.get("factor_level")
        final = item.get("final_summary", {})
        if level in result and isinstance(final, list):
            for row in final:
                if isinstance(row, dict):
                    value = row.get("gmean2", row.get("gmean2_mean"))
                    if value is not None:
                        result[level].append(float(value))
    return result


def _prediction_path(root: Path, benchmark_id: str) -> Path:
    direct = root / f"{benchmark_id}.jsonl"
    nested = root / benchmark_id / "predictions.jsonl"
    if direct.is_file(): return direct
    if nested.is_file(): return nested
    raise FileNotFoundError(f"missing predictions for {benchmark_id}: {direct} or {nested}")


def _legacy_score(entry: dict[str, Any], predictions: Path, output: Path, monitor_config: Path | None, *, manifest_path: Path, overwrite: bool) -> dict[str, Any]:
    if monitor_config is None or not monitor_config.is_file():
        raise ValueError(f"{entry['id']} requires --monitor-config-root/<id>.json")
    import sys
    source_root = Path(__file__).resolve().parents[3] / "factor_monitor" / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    artifact = resolve_eval_path(entry["artifact"], manifest_path=manifest_path)
    if entry["track"] == "intervention":
        from factor_monitor.openai_intervention import score_artifact
        return score_artifact(artifact=artifact, predictions=predictions, monitor_effective_config=monitor_config, output_directory=output, overwrite=overwrite)
    if entry["track"] == "process":
        from factor_monitor.process_eval import score_process_artifact
        return score_process_artifact(artifact=artifact, predictions=predictions, monitor_effective_config=monitor_config, output_directory=output, overwrite=overwrite)
    raise ValueError(f"unsupported legacy track: {entry['track']}")


def run_suite(*, manifest_path: Path, predictions_root: Path, output_root: Path,
              monitor_config_root: Path | None = None, threshold_root: Path | None = None,
              thresholds_file: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["benchmarks"]:
        benchmark_id = str(entry["id"])
        output = output_root / benchmark_id
        try:
            if overwrite and (output / "report.md").is_file():
                (output / "report.md").unlink()
            predictions = _prediction_path(predictions_root, benchmark_id)
            if entry["track"] == "binary":
                raw = read_jsonl(predictions)
                has_score = any("score" in row for row in raw)
                threshold_info = resolve_threshold(
                    entry, manifest_path=manifest_path.resolve(),
                    thresholds_file=thresholds_file, threshold_root=threshold_root,
                    monitor_config_root=monitor_config_root,
                ) if has_score else None
                if has_score and threshold_info is None:
                    raise ValueError(
                        f"{benchmark_id} contains score predictions but no frozen threshold; "
                        "provide manifest threshold, --thresholds-file, --threshold-root, or --monitor-config-root"
                    )
                result = score_binary_artifact(
                    artifact=resolve_eval_path(entry["artifact"], manifest_path=manifest_path), predictions=predictions,
                    output_directory=output,
                    threshold=threshold_info["value"] if threshold_info else None,
                    threshold_source=threshold_info["source"] if threshold_info else None,
                    overwrite=overwrite,
                )
            else:
                config = monitor_config_root / f"{benchmark_id}.json" if monitor_config_root else None
                result = _legacy_score(entry, predictions, output, config, manifest_path=manifest_path, overwrite=overwrite)
            write_report(result, output / "report.md")
            results.append({"id": benchmark_id, "track": entry["track"], "status": "ok", "summary": result})
        except Exception as exc:
            results.append({"id": benchmark_id, "track": entry["track"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    successful = [item for item in results if item["status"] == "ok"]
    gmeans = [item["summary"].get("slices", {}).get("overall", {}).get("gmean2") for item in successful]
    gmeans = [float(value) for value in gmeans if value is not None]
    by_track: dict[str, list[float]] = {}
    by_level: dict[str, list[float]] = {"L1": [], "L3": []}
    for item in successful:
        value = item["summary"].get("slices", {}).get("overall", {}).get("gmean2")
        if value is not None:
            by_track.setdefault(item["track"], []).append(float(value))
        for level, values in _level_metrics(item["summary"]).items():
            by_level[level].extend(values)
    suite = {
        "schema_version": 1,
        "kind": "factor_monitor_eval_suite_summary",
        "manifest": str(manifest_path.resolve()),
        "formal_metric": "gmean2",
        "macro_gmean2": sum(gmeans) / len(gmeans) if gmeans else None,
        "macro_gmean2_by_track": {
            track: sum(values) / len(values) for track, values in sorted(by_track.items()) if values
        },
        "macro_gmean2_by_factor_level": {
            level: sum(values) / len(values) for level, values in by_level.items() if values
        },
        "complete": len(successful) == len(results),
        "failed_units": [item["id"] for item in results if item["status"] != "ok"],
        "results": results,
    }
    write_json(output_root / "suite_summary.json", suite)
    write_json(output_root / "provenance.json", {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path.resolve()),
        "predictions_root": str(predictions_root.resolve()),
        "output_root": str(output_root.resolve()),
        "benchmark_count": len(manifest["benchmarks"]),
    })
    write_suite_report(suite, output_root / "report.md")
    return suite
