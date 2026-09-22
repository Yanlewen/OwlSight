"""Small API smoke test: one first query from every active benchmark."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .adapters import normalize_decision
from .io import read_jsonl, write_json
from .manifest import load_manifest, resolve_eval_path, select_benchmarks
from .providers import (
    MONITOR_DECISION_PROMPT_ID,
    MONITOR_DECISION_SYSTEM_PROMPT,
    OpenAICompatibleProvider,
    Request,
    build_monitor_decision_prompt,
)


def _smoke_decision(value: Any) -> tuple[str | None, str]:
    decision, status = normalize_decision(value)
    if status == "valid":
        return decision, status
    text = str(value or "").strip().lower()
    tokens = re.findall(r"\b(?:not_applied|applied)\b", text)
    if tokens:
        token = tokens[-1]
        return token, "valid"
    return None, "invalid"


def _context(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(f"{item.get('role', 'user')}: {item.get('content', '')}")
        if parts:
            return "\n".join(parts)
    return str(value or "")


def _request(dataset: Path, entry: dict[str, Any], *, max_prompt_chars: int = 16000) -> Request:
    artifact = dataset / str(entry["id"])
    query = read_jsonl(artifact / "eval_queries.jsonl")[0]
    trajectory_id = query.get("trajectory_id")
    trajectory = next(
        row for row in read_jsonl(artifact / "trajectories.jsonl")
        if row.get("trajectory_id") == trajectory_id
    )
    prompt = build_monitor_decision_prompt(
        factor_spec=query.get("factor_spec", {}),
        task_context=_context(trajectory.get("task_context", query.get("task_context"))),
        model_cot=str(trajectory.get("model_cot") or ""),
        max_prompt_chars=max_prompt_chars,
    )
    return Request(
        query_id=str(query["query_id"]),
        prompt=prompt,
        metadata={"benchmark": entry["id"], "family": entry.get("family"), "factor_level": query.get("factor_level")},
    )


def run_smoke(*, manifest_path: Path, dataset_root: Path, base_url: str, model: str,
              output: Path | None = None, api_key_env: str = "OPENAI_API_KEY",
              timeout: float = 60, max_tokens: int = 16,
              max_prompt_chars: int = 16000, ids: list[str] | None = None,
              thinking_off_mode: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = resolve_eval_path(dataset_root, manifest_path=manifest_path)
    entries = select_benchmarks(manifest, ids)
    requests = [_request(dataset, entry, max_prompt_chars=max_prompt_chars) for entry in entries]
    provider = OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout=timeout,
        max_tokens=max_tokens,
        thinking_off_mode=thinking_off_mode,
    )
    def call(request: Request) -> dict[str, Any]:
        try:
            return provider.predict_batch([request])[0]
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=min(8, len(requests))) as pool:
        responses = list(pool.map(call, requests))
    rows = []
    for request, response in zip(requests, responses):
        if response.get("error"):
            decision, status = None, "request_error"
        else:
            decision, status = _smoke_decision(response.get("raw_output", response.get("decision")))
        rows.append({
            "benchmark": request.metadata["benchmark"],
            "family": request.metadata["family"],
            "factor_level": request.metadata["factor_level"],
            "query_id": request.query_id,
            "decision": decision,
            "parse_status": status,
            "raw_output": response.get("raw_output"),
            "error": response.get("error"),
        })
    result = {
        "schema_version": 1,
        "kind": "factor_monitor_eval_api_smoke",
        "model": model,
        "base_url": base_url,
        "thinking_off_mode": provider.thinking_off_mode,
        "prompt_id": MONITOR_DECISION_PROMPT_ID,
        "system_prompt": MONITOR_DECISION_SYSTEM_PROMPT,
        "benchmark_ids": [str(entry["id"]) for entry in entries],
        "benchmarks": rows,
        "all_requests_returned": all(row["parse_status"] != "request_error" for row in rows),
        "all_decisions_parse": all(row["parse_status"] == "valid" for row in rows),
    }
    if output is not None:
        write_json(output, result)
    return result
