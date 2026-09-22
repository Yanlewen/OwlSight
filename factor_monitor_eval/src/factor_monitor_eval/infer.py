"""Full-suite chat monitor inference with resume and trajectory indexing."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .io import append_jsonl, iter_jsonl, write_json, write_jsonl
from .manifest import load_manifest, resolve_eval_path, select_benchmarks
from .providers import (
    MONITOR_DECISION_PROMPT_ID,
    MONITOR_DECISION_SYSTEM_PROMPT,
    OpenAICompatibleProvider,
    Request,
    build_monitor_decision_prompt,
)
from .smoke import _context, _smoke_decision


DONE_STATUSES = {"valid", "invalid"}


def _index_trajectories(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        trajectory_id = row.get("trajectory_id")
        if isinstance(trajectory_id, str) and trajectory_id:
            index[trajectory_id] = row
    return index


def _load_kept_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    kept = []
    for row in iter_jsonl(path):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            continue
        if row.get("parse_status") in DONE_STATUSES:
            kept.append(row)
    return kept


def _prediction_row(query_id: str, response: dict[str, Any]) -> dict[str, Any]:
    if response.get("error"):
        return {
            "query_id": query_id,
            "decision": None,
            "parse_status": "request_error",
            "raw_output": response.get("raw_output"),
            "error": response.get("error"),
        }
    decision, status = _smoke_decision(response.get("raw_output", response.get("decision")))
    row: dict[str, Any] = {
        "query_id": query_id,
        "decision": decision,
        "parse_status": status,
        "raw_output": response.get("raw_output"),
        "error": None,
    }
    if status == "valid" and decision == "applied":
        row["score"] = 1.0
    elif status == "valid" and decision == "not_applied":
        row["score"] = 0.0
    return row


def _build_request(query: dict[str, Any], trajectory: dict[str, Any], *, max_prompt_chars: int) -> Request:
    return Request(
        query_id=str(query["query_id"]),
        prompt=build_monitor_decision_prompt(
            factor_spec=query.get("factor_spec", {}),
            task_context=_context(trajectory.get("task_context", query.get("task_context"))),
            model_cot=str(trajectory.get("model_cot") or ""),
            max_prompt_chars=max_prompt_chars,
        ),
        metadata={},
    )


def _infer_benchmark(
    *,
    artifact: Path,
    output_path: Path,
    provider: Any,
    workers: int,
    max_prompt_chars: int,
    limit: int | None,
    lock: threading.Lock,
) -> dict[str, Any]:
    queries = list(iter_jsonl(artifact / "eval_queries.jsonl"))
    trajectories = _index_trajectories(artifact / "trajectories.jsonl")
    kept = _load_kept_rows(output_path)
    done = {str(row["query_id"]) for row in kept}
    write_jsonl(output_path, kept)
    pending_queries = [query for query in queries if str(query.get("query_id")) not in done]
    if limit is not None:
        pending_queries = pending_queries[: max(0, int(limit))]

    def call(query: dict[str, Any]) -> dict[str, Any]:
        query_id = str(query["query_id"])
        trajectory_id = query.get("trajectory_id")
        trajectory = trajectories.get(str(trajectory_id)) if trajectory_id is not None else None
        if trajectory is None:
            return _prediction_row(query_id, {"error": f"missing trajectory_id={trajectory_id}"})
        request = _build_request(query, trajectory, max_prompt_chars=max_prompt_chars)
        try:
            return _prediction_row(query_id, provider.predict_batch([request])[0])
        except Exception as exc:
            return _prediction_row(query_id, {"error": f"{type(exc).__name__}: {exc}"})

    completed = 0
    valid = sum(1 for row in kept if row.get("parse_status") == "valid")
    invalid = sum(1 for row in kept if row.get("parse_status") == "invalid")
    errors = 0
    if pending_queries:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(call, query) for query in pending_queries]
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    append_jsonl(output_path, row)
                completed += 1
                if row.get("parse_status") == "valid":
                    valid += 1
                elif row.get("parse_status") == "invalid":
                    invalid += 1
                else:
                    errors += 1
                if completed % 50 == 0 or completed == len(pending_queries):
                    print(
                        f"{artifact.name} {completed}/{len(pending_queries)} "
                        f"valid={valid} invalid={invalid} errors={errors}",
                        file=sys.stderr,
                        flush=True,
                    )
    covered = valid + invalid
    return {
        "id": artifact.name,
        "queries": len(queries),
        "kept": len(kept),
        "attempted": len(pending_queries),
        "valid": valid,
        "invalid": invalid,
        "request_errors": errors,
        "complete": limit is None and covered == len(queries),
        "predictions": str(output_path),
    }


def run_infer(
    *,
    manifest_path: Path,
    dataset_root: Path,
    base_url: str,
    model: str,
    output_root: Path,
    api_key_env: str = "OPENAI_API_KEY",
    timeout: float = 60,
    max_tokens: int = 16,
    max_prompt_chars: int = 16000,
    workers: int = 8,
    ids: list[str] | None = None,
    thinking_off_mode: str | None = None,
    limit: int | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = resolve_eval_path(dataset_root, manifest_path=manifest_path)
    entries = select_benchmarks(manifest, ids)
    output_root.mkdir(parents=True, exist_ok=True)
    client = provider or OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout=timeout,
        max_tokens=max_tokens,
        thinking_off_mode=thinking_off_mode,
    )
    config = {
        "schema_version": 1,
        "kind": "factor_monitor_eval_api_infer",
        "model": model,
        "base_url": base_url,
        "thinking_off_mode": getattr(client, "thinking_off_mode", thinking_off_mode),
        "prompt_id": MONITOR_DECISION_PROMPT_ID,
        "system_prompt": MONITOR_DECISION_SYSTEM_PROMPT,
        "max_tokens": max_tokens,
        "max_prompt_chars": max_prompt_chars,
        "timeout": timeout,
        "workers": workers,
        "benchmark_ids": [str(entry["id"]) for entry in entries],
    }
    write_json(output_root / "run_config.json", config)
    lock = threading.Lock()
    benchmarks = []
    for entry in entries:
        benchmark_id = str(entry["id"])
        artifact = resolve_eval_path(entry["artifact"], manifest_path=manifest_path)
        if not artifact.is_dir():
            artifact = dataset / benchmark_id
        benchmarks.append(
            _infer_benchmark(
                artifact=artifact,
                output_path=output_root / f"{benchmark_id}.jsonl",
                provider=client,
                workers=workers,
                max_prompt_chars=max_prompt_chars,
                limit=limit,
                lock=lock,
            )
        )
    summary = {
        **config,
        "output_root": str(output_root.resolve()),
        "benchmarks": benchmarks,
        "all_complete": all(item["complete"] for item in benchmarks),
        "all_decisions_parse": all(item["invalid"] == 0 and item["request_errors"] == 0 for item in benchmarks),
    }
    write_json(output_root / "infer_summary.json", summary)
    return summary
