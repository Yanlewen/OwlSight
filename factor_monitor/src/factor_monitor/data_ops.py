"""Reusable data acquisition and OpenAI-compatible batch execution.

Dataset-specific adapters construct requests and interpret outputs.  This module
owns transport, resumption, progress, error queues, and source snapshots only.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    require_keys,
    require_mapping,
    stable_hash,
)
from .io import read_jsonl, write_json, write_jsonl


ClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class JudgeProfile:
    """Shared, immutable request and transport contract for one Judge model."""

    profile_id: str
    model: str
    request_options: dict[str, Any]
    base_url_env: str
    api_key_env: str
    workers: int
    retries: int
    timeout_seconds: float
    checkpoint_every: int
    response_contract: dict[str, bool]
    config_path: Path
    config_sha256: str

    @property
    def request_contract(self) -> dict[str, Any]:
        return {"model": self.model, **self.request_options}

    def effective_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "factor_judge_profile",
            "profile_id": self.profile_id,
            "model": self.model,
            "request": self.request_options,
            "transport": {
                "base_url_env": self.base_url_env,
                "api_key_env": self.api_key_env,
                "workers": self.workers,
                "retries": self.retries,
                "timeout_seconds": self.timeout_seconds,
                "checkpoint_every": self.checkpoint_every,
            },
            "response": self.response_contract,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be boolean")
    return value


def load_judge_profile(path: Path) -> JudgeProfile:
    """Load one shared Judge profile with no dataset-specific overrides."""

    path = path.resolve()
    value = load_config(path)
    fields = {
        "schema_version",
        "kind",
        "profile_id",
        "model",
        "request",
        "transport",
        "response",
    }
    require_keys(value, fields, "Judge profile")
    reject_unknown(value, fields, "Judge profile")
    if value["schema_version"] != 1 or value["kind"] != "factor_judge_profile":
        raise ConfigurationError(
            "Judge profile requires schema_version=1 and kind=factor_judge_profile"
        )

    request = require_mapping(value["request"], "Judge profile.request")
    require_keys(
        request,
        {"temperature", "top_p", "max_tokens", "extra_body"},
        "Judge profile.request",
    )
    if "model" in request or "messages" in request:
        raise ConfigurationError(
            "Judge profile.request must not contain model or messages"
        )
    for name in ("temperature", "top_p"):
        number = request[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ConfigurationError(f"Judge profile.request.{name} must be numeric")
    if float(request["temperature"]) < 0:
        raise ConfigurationError("Judge profile.request.temperature must be non-negative")
    if not 0 < float(request["top_p"]) <= 1:
        raise ConfigurationError("Judge profile.request.top_p must be in (0, 1]")
    _positive_integer(request["max_tokens"], "Judge profile.request.max_tokens")
    require_mapping(request["extra_body"], "Judge profile.request.extra_body")

    transport = require_mapping(value["transport"], "Judge profile.transport")
    transport_fields = {
        "base_url_env",
        "api_key_env",
        "workers",
        "retries",
        "timeout_seconds",
        "checkpoint_every",
    }
    require_keys(transport, transport_fields, "Judge profile.transport")
    reject_unknown(transport, transport_fields, "Judge profile.transport")
    retries = transport["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ConfigurationError("Judge profile.transport.retries must be non-negative")
    timeout = transport["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigurationError(
            "Judge profile.transport.timeout_seconds must be positive"
        )

    response = require_mapping(value["response"], "Judge profile.response")
    response_fields = {"require_content", "forbid_reasoning"}
    require_keys(response, response_fields, "Judge profile.response")
    reject_unknown(response, response_fields, "Judge profile.response")
    response_contract = {
        name: _boolean(response[name], f"Judge profile.response.{name}")
        for name in sorted(response_fields)
    }
    return JudgeProfile(
        profile_id=_nonempty_text(value["profile_id"], "Judge profile.profile_id"),
        model=_nonempty_text(value["model"], "Judge profile.model"),
        request_options=dict(request),
        base_url_env=_nonempty_text(
            transport["base_url_env"], "Judge profile.transport.base_url_env"
        ),
        api_key_env=_nonempty_text(
            transport["api_key_env"], "Judge profile.transport.api_key_env"
        ),
        workers=_positive_integer(
            transport["workers"], "Judge profile.transport.workers"
        ),
        retries=retries,
        timeout_seconds=float(timeout),
        checkpoint_every=_positive_integer(
            transport["checkpoint_every"],
            "Judge profile.transport.checkpoint_every",
        ),
        response_contract=response_contract,
        config_path=path,
        config_sha256=file_sha256(path),
    )


def _validate_messages(messages: Any, name: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ConfigurationError(f"{name} must be a non-empty list")
    allowed_roles = {"system", "user", "assistant", "tool"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ConfigurationError(
                f"{name}[{index}] must contain exactly role and content"
            )
        if message["role"] not in allowed_roles:
            raise ConfigurationError(f"{name}[{index}].role is unsupported")
        _nonempty_text(message["content"], f"{name}[{index}].content")


def build_judge_request(
    profile: JudgeProfile,
    *,
    request_id: str,
    purpose: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one request while preventing dataset-specific parameter drift."""

    request_id = _nonempty_text(request_id, "request_id")
    purpose = _nonempty_text(purpose, "purpose")
    _validate_messages(messages, f"Judge request {request_id}.messages")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ConfigurationError("Judge request metadata must be a mapping")
    return {
        "request_id": request_id,
        "profile_id": profile.profile_id,
        "purpose": purpose,
        "metadata": dict(metadata),
        "request": {
            "model": profile.model,
            "messages": [dict(message) for message in messages],
            **profile.request_options,
        },
    }


def export_huggingface_split(
    config: dict[str, Any],
    *,
    output_path: Path,
    summary_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Snapshot one explicit Hugging Face dataset split as JSONL."""

    required = {"repo_id", "config_name", "split", "revision", "streaming"}
    missing = sorted(required - set(config))
    unknown = sorted(set(config) - (required | {"cache_dir", "trust_remote_code"}))
    if missing or unknown:
        raise ConfigurationError(
            f"Hugging Face config missing={missing} unknown={unknown}"
        )
    for field in ("repo_id", "split", "revision"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ConfigurationError(f"Hugging Face {field} must be non-empty")
    if config["config_name"] is not None and (
        not isinstance(config["config_name"], str) or not config["config_name"].strip()
    ):
        raise ConfigurationError("Hugging Face config_name must be null or non-empty")
    if not isinstance(config["streaming"], bool):
        raise ConfigurationError("Hugging Face streaming must be boolean")
    effective = {
        "schema_version": 1,
        "operation": "huggingface_split_snapshot",
        "config": dict(config),
        "output": str(output_path.resolve()),
        "summary_output": str(summary_path.resolve()),
        "dry_run": dry_run,
    }
    if dry_run:
        return effective
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional 'datasets' package first") from exc
    kwargs: dict[str, Any] = {
        "path": config["repo_id"],
        "name": config["config_name"],
        "split": config["split"],
        "revision": config["revision"],
        "streaming": config["streaming"],
    }
    for field in ("cache_dir", "trust_remote_code"):
        if field in config:
            kwargs[field] = config[field]
    dataset = load_dataset(**kwargs)
    row_count = 0

    def rows():
        nonlocal row_count
        for value in dataset:
            row_count += 1
            yield dict(value)

    write_jsonl(output_path, rows())
    summary = {
        **effective,
        "dry_run": False,
        "rows": row_count,
        "sha256": file_sha256(output_path),
    }
    write_json(summary_path, summary)
    return summary


def _validate_chat_requests(
    rows: list[dict[str, Any]],
    *,
    request_contract: dict[str, Any] | None = None,
    profile_id: str | None = None,
) -> None:
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        request_id = row.get("request_id")
        request = row.get("request")
        if not isinstance(request_id, str) or not request_id:
            raise ConfigurationError(f"chat request row {index} has invalid request_id")
        if not isinstance(request, dict):
            raise ConfigurationError(f"chat request {request_id} requires request mapping")
        for field in ("model", "messages", "temperature", "max_tokens"):
            if field not in request:
                raise ConfigurationError(
                    f"chat request {request_id} missing explicit field {field}"
                )
        if not isinstance(request["model"], str) or not request["model"]:
            raise ConfigurationError(f"chat request {request_id} has invalid model")
        _validate_messages(request["messages"], f"chat request {request_id}.messages")
        if profile_id is not None and row.get("profile_id") != profile_id:
            raise ConfigurationError(
                f"chat request {request_id} profile_id does not match {profile_id!r}"
            )
        if request_contract is not None:
            static_request = {
                key: value for key, value in request.items() if key != "messages"
            }
            if static_request != request_contract:
                raise ConfigurationError(
                    f"chat request {request_id} settings drifted from the shared "
                    "request contract"
                )
        ids.append(request_id)
    if len(ids) != len(set(ids)):
        raise ConfigurationError("chat request file contains duplicate request_id values")


def _message_field(message: Any, name: str) -> Any:
    value = getattr(message, name, None)
    if value is not None:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _response_row(row: dict[str, Any], response: Any, attempts: int) -> dict[str, Any]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("provider response has no choices")
    choice = choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)
    if usage is None:
        usage_value = None
    elif hasattr(usage, "model_dump"):
        usage_value = usage.model_dump()
    else:
        usage_value = dict(usage)
    reasoning_content = _message_field(message, "reasoning_content")
    reasoning = _message_field(message, "reasoning")
    canonical_reasoning = (
        reasoning_content if _has_content(reasoning_content) else reasoning
    )
    return {
        "request_id": row["request_id"],
        "profile_id": row.get("profile_id"),
        "request_hash": stable_hash(row["request"]),
        "purpose": row.get("purpose"),
        "metadata": row.get("metadata", {}),
        "model": row["request"]["model"],
        "response_model": getattr(response, "model", None),
        "content": getattr(message, "content", None),
        # vLLM reasoning parsers do not all expose the same OpenAI extension
        # field.  Keep one stable transport contract for dataset adapters while
        # retaining the provider-native field for provenance/debugging.
        "reasoning_content": canonical_reasoning,
        "reasoning": reasoning,
        "finish_reason": getattr(choice, "finish_reason", None),
        "provider_response_id": getattr(response, "id", None),
        "usage": usage_value,
        "attempts": attempts,
    }


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != []


def unwrap_single_json_fence(content: str) -> str:
    """Remove one complete JSON Markdown fence while leaving other text untouched."""

    stripped = content.strip()
    lines = stripped.splitlines()
    if not lines or lines[0].strip().lower() not in {"```", "```json"}:
        return stripped
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    body = "\n".join(lines[1:-1]).strip()
    return stripped if "```" in body else body


def replace_judge_request_pool(
    requests_path: Path,
    requests: list[dict[str, Any]],
    *,
    reset_paths: list[Path],
    overwrite: bool,
) -> bool:
    """Replace a changed Judge contract and only its explicitly owned caches."""

    existing = read_jsonl(requests_path) if requests_path.is_file() else None
    if existing == requests:
        return False
    stale_outputs = [path for path in reset_paths if path.is_file()]
    if (existing is not None or stale_outputs) and not overwrite:
        raise ConfigurationError(
            f"changed Judge contract requires --overwrite: {requests_path}"
        )
    for path in stale_outputs:
        path.unlink()
    write_jsonl(requests_path, requests)
    return True


def prune_judge_output_cache(
    output_path: Path,
    *,
    request_ids: set[str],
) -> int:
    """Remove explicitly identified invalid responses from a resumable cache."""

    if not request_ids or not output_path.is_file():
        return 0
    existing = read_jsonl(output_path)
    observed_ids = [str(row.get("request_id")) for row in existing]
    if len(observed_ids) != len(set(observed_ids)):
        raise ConfigurationError(
            "cannot prune Judge output cache containing duplicate request IDs"
        )
    retained = [
        row for row in existing if str(row.get("request_id")) not in request_ids
    ]
    removed = len(existing) - len(retained)
    if removed:
        write_jsonl(output_path, retained)
    return removed


def _validate_response_contract(
    row: dict[str, Any], response_contract: dict[str, bool] | None
) -> None:
    if response_contract is None:
        return
    if response_contract.get("require_content") and not _has_content(row.get("content")):
        raise ValueError("provider response violates Judge contract: empty content")
    if response_contract.get("forbid_reasoning") and any(
        _has_content(row.get(name)) for name in ("reasoning_content", "reasoning")
    ):
        raise ValueError(
            "provider response violates Judge contract: thinking was not disabled"
        )


def run_openai_chat_batch(
    *,
    requests_path: Path,
    output_path: Path,
    errors_path: Path,
    summary_path: Path,
    base_url: str | None,
    api_key: str | None,
    workers: int,
    retries: int,
    timeout_seconds: float,
    checkpoint_every: int,
    dry_run: bool,
    client_factory: ClientFactory | None = None,
    request_contract: dict[str, Any] | None = None,
    response_contract: dict[str, bool] | None = None,
    profile_id: str | None = None,
    prune_unexpected_outputs: bool = False,
) -> dict[str, Any]:
    """Run explicit chat requests with resumable successes and a retry queue."""

    if workers <= 0 or retries < 0 or timeout_seconds <= 0 or checkpoint_every <= 0:
        raise ConfigurationError("invalid workers/retries/timeout/checkpoint configuration")
    requests = read_jsonl(requests_path)
    _validate_chat_requests(
        requests, request_contract=request_contract, profile_id=profile_id
    )
    existing = read_jsonl(output_path) if output_path.is_file() else []
    existing_ids = [str(row.get("request_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ConfigurationError("existing chat outputs contain duplicate request_id values")
    request_ids = {row["request_id"] for row in requests}
    unexpected = sorted(set(existing_ids) - request_ids)
    if unexpected:
        if not prune_unexpected_outputs:
            raise ConfigurationError(f"existing outputs contain unexpected IDs: {unexpected[:10]}")
        existing = [row for row in existing if str(row.get("request_id")) in request_ids]
        existing_ids = [str(row["request_id"]) for row in existing]
        if not dry_run:
            write_jsonl(output_path, existing)
    requests_by_id = {row["request_id"]: row for row in requests}
    mismatched = []
    for row in existing:
        request_id = str(row.get("request_id"))
        expected_hash = stable_hash(requests_by_id[request_id]["request"])
        if row.get("request_hash") != expected_hash:
            mismatched.append(request_id)
    if mismatched:
        if not prune_unexpected_outputs:
            raise ConfigurationError(
                f"existing output {mismatched[0]} does not match the current request"
            )
        mismatched_set = set(mismatched)
        existing = [
            row
            for row in existing
            if str(row.get("request_id")) not in mismatched_set
        ]
        existing_ids = [str(row["request_id"]) for row in existing]
        if not dry_run:
            write_jsonl(output_path, existing)
    for row in existing:
        _validate_response_contract(row, response_contract)
    completed = set(existing_ids)
    pending = [row for row in requests if row["request_id"] not in completed]
    effective = {
        "schema_version": 1,
        "operation": "openai_compatible_chat_batch",
        "requests": str(requests_path.resolve()),
        "requests_sha256": file_sha256(requests_path),
        "output": str(output_path.resolve()),
        "errors_output": str(errors_path.resolve()),
        "summary_output": str(summary_path.resolve()),
        "requests_total": len(requests),
        "already_complete": len(completed),
        "calls_planned": len(pending),
        "workers": workers,
        "retries": retries,
        "timeout_seconds": timeout_seconds,
        "checkpoint_every": checkpoint_every,
        "prune_unexpected_outputs": prune_unexpected_outputs,
        "unexpected_outputs_pruned": len(unexpected),
        "request_hash_mismatched_outputs_pruned": len(mismatched),
        "base_url_configured": bool(base_url),
        "api_key_configured": bool(api_key),
        "profile_id": profile_id,
        "request_contract": request_contract,
        "request_contract_hash": (
            stable_hash(request_contract) if request_contract is not None else None
        ),
        "response_contract": response_contract,
        "dry_run": dry_run,
    }
    if dry_run:
        return {**effective, "remaining": len(pending)}
    if not pending:
        existing_errors = read_jsonl(errors_path) if errors_path.is_file() else []
        retained_errors = [
            row
            for row in existing_errors
            if str(row.get("request_id")) not in completed
        ]
        resolved_stale_errors = len(existing_errors) - len(retained_errors)
        if resolved_stale_errors:
            write_jsonl(errors_path, retained_errors)
        existing_errors = retained_errors
        return {
            **effective,
            "outputs_total": len(existing),
            "remaining": 0,
            "provider_attempts_total": sum(
                int(row.get("attempts", 1)) for row in existing
            ),
            "outputs_requiring_retry": sum(
                int(row.get("attempts", 1)) > 1 for row in existing
            ),
            "stale_errors_resolved": resolved_stale_errors,
            "errors_total": len(existing_errors),
            "output_sha256": file_sha256(output_path),
            "errors_sha256": (
                file_sha256(errors_path) if errors_path.is_file() else None
            ),
        }
    if not base_url or not api_key:
        raise ConfigurationError("base_url and api_key must be supplied at runtime")
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the optional 'openai' package first") from exc
        client_factory = OpenAI
    client = client_factory(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    def call(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                response = client.chat.completions.create(**row["request"])
                result = _response_row(row, response, attempt)
                _validate_response_contract(result, response_contract)
                return result, None
            except Exception as exc:  # provider exceptions vary by SDK/backend
                last_error = exc
                if attempt <= retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        return None, {
            "request_id": row["request_id"],
            "profile_id": row.get("profile_id"),
            "request_hash": stable_hash(row["request"]),
            "purpose": row.get("purpose"),
            "metadata": row.get("metadata", {}),
            "error_type": type(last_error).__name__,
            "error": str(last_error),
            "attempts": retries + 1,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    succeeded = 0
    handle = output_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(call, row): row["request_id"] for row in pending}
            iterator: Any = as_completed(futures)
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc="data chat requests")
            except ImportError:
                pass
            for observed, future in enumerate(iterator, start=1):
                result, error = future.result()
                if result is not None:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    succeeded += 1
                else:
                    assert error is not None
                    errors.append(error)
                if observed % checkpoint_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
    finally:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    write_jsonl(errors_path, errors)
    observed_outputs = read_jsonl(output_path)
    final_ids = {str(row["request_id"]) for row in observed_outputs}
    summary = {
        **effective,
        "dry_run": False,
        "succeeded_this_run": succeeded,
        "errors_this_run": len(errors),
        "outputs_total": len(final_ids),
        "remaining": len(request_ids - final_ids),
        "provider_attempts_total": sum(
            int(row.get("attempts", 1)) for row in observed_outputs
        ),
        "outputs_requiring_retry": sum(
            int(row.get("attempts", 1)) > 1 for row in observed_outputs
        ),
        "errors_total": len(errors),
        "output_sha256": file_sha256(output_path),
        "errors_sha256": file_sha256(errors_path),
    }
    write_json(summary_path, summary)
    return summary


def run_judge_batch(
    *,
    profile: JudgeProfile | Path,
    requests_path: Path,
    output_path: Path,
    errors_path: Path,
    summary_path: Path,
    base_url: str | None = None,
    api_key: str | None = None,
    dry_run: bool = False,
    client_factory: ClientFactory | None = None,
    prune_unexpected_outputs: bool = False,
) -> dict[str, Any]:
    """Run a Judge request file using one shared, fail-closed model profile."""

    selected = load_judge_profile(profile) if isinstance(profile, Path) else profile
    resolved_url = base_url or os.environ.get(selected.base_url_env)
    resolved_key = api_key or os.environ.get(selected.api_key_env)
    runtime_credentials = {
        "base_url_configured": bool(resolved_url),
        "api_key_configured": bool(resolved_key),
    }
    if dry_run:
        resolved_url = resolved_url or "http://dry-run.invalid/v1"
        resolved_key = resolved_key or "dry-run-placeholder"
    result = run_openai_chat_batch(
        requests_path=requests_path,
        output_path=output_path,
        errors_path=errors_path,
        summary_path=summary_path,
        base_url=resolved_url,
        api_key=resolved_key,
        workers=selected.workers,
        retries=selected.retries,
        timeout_seconds=selected.timeout_seconds,
        checkpoint_every=selected.checkpoint_every,
        dry_run=dry_run,
        client_factory=client_factory,
        request_contract=selected.request_contract,
        response_contract=selected.response_contract,
        profile_id=selected.profile_id,
        prune_unexpected_outputs=prune_unexpected_outputs,
    )
    result = {
        **result,
        **runtime_credentials,
        "judge_profile": selected.effective_config(),
    }
    if not dry_run and summary_path.is_file():
        write_json(summary_path, result)
    return result
