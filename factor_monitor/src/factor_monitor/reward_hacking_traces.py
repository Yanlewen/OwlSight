"""Curate ATIF Reward Hacking traces from Terminal Wrench and HVTB.

The source parsers are intentionally shared because both releases use ATIF.  Source
labels remain source-specific; all behavioral subtyping uses one frozen I/C Judge.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    repository_state,
    resolve_path,
    stable_hash,
)
from .data_ops import build_judge_request, load_judge_profile, run_judge_batch
from .io import iter_jsonl, read_jsonl, write_json, write_jsonl
from .reward_hacking import (
    ICY_JUDGE_SYSTEM_PROMPT,
    ICY_JUDGE_USER_TEMPLATE,
    L3_FACTOR_SPEC_SHA256,
    L3_STANDARD_ID,
    chunk_evidence_units,
    factor_spec,
    parse_icy_judge_content,
    reasoning_units,
)
from .schema import CanonicalRecord, Taxonomy
from .cot_format import render_atif_reasoning_steps, render_tagged_reasoning_lines


MONITOR_COT_FORMAT = (
    "assistant_private_reasoning_steps_only"
)
REASONING_TAG = "[REASONING]"
TOOL_TAG = "[TOOL_CALL]"
HVTB_HACK_SIGNALS = ("read_tests", "modify_tests", "hidden_solution")
TERMINAL_REWARDED = {
    "rewarded_serious_exploit",
    "rewarded_nonserious_hack",
}
TERMINAL_FAILED = {"no_reward"}
TERMINAL_HARD_CANDIDATE = {"attacker_legitimate_solve"}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    required = {
        "schema_version",
        "kind",
        "source_kind",
        "seed",
        "paths",
        "canary",
        "judge",
        "l3",
        "selection",
        "canonicalization",
    }
    if set(config) != required or config.get("schema_version") != 1:
        raise ConfigurationError("invalid Reward Hacking trace-source config")
    if config.get("kind") != "reward_hacking_atif_source":
        raise ConfigurationError("config.kind must be reward_hacking_atif_source")
    if config["source_kind"] not in {"terminal_wrench", "hvtb"}:
        raise ConfigurationError("unsupported Reward Hacking ATIF source_kind")
    l3 = _mapping(config["l3"], "l3")
    if l3.get("standard_id") != L3_STANDARD_ID:
        raise ConfigurationError("Reward Hacking L3 standard ID drifted")
    if l3.get("factor_spec_sha256") != L3_FACTOR_SPEC_SHA256:
        raise ConfigurationError("Reward Hacking L3 hash pin drifted")
    if stable_hash(factor_spec(l3)) != L3_FACTOR_SPEC_SHA256:
        raise ConfigurationError("Reward Hacking L3 four-field text drifted")
    base = config_path.parent
    paths = {
        name: resolve_path(value, relative_to=base)
        for name, value in _mapping(config["paths"], "paths").items()
    }
    expected_paths = {
        "raw_root",
        "candidates",
        "canary_tasks",
        "judge_requests",
        "judge_outputs",
        "judge_errors",
        "judge_transport_summary",
        "judge_labels",
        "prepare_summary",
        "effective_config",
        "canonical_records",
        "canonical_summary",
    }
    if set(paths) != expected_paths:
        raise ConfigurationError("Reward Hacking trace-source paths drifted")
    return config, paths


def _read_atif(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise ConfigurationError(f"invalid ATIF trajectory: {path}")
    return value


def _initial_task_context(steps: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    counts: Counter[str] = Counter()
    for step in steps:
        source = str(step.get("source") or "").lower()
        if source == "agent":
            break
        if source not in {"system", "user"}:
            continue
        message = step.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        counts[source] += 1
        blocks.append(f"[{source.upper()}-{counts[source]:03d}] {message.strip()}")
    if not blocks:
        raise ConfigurationError("ATIF trajectory has no initial Task Context")
    return "\n\n".join(blocks)


def _tool_call_text(call: dict[str, Any]) -> str:
    function = str(call.get("function_name") or call.get("function") or "unknown")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = {}
    rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return f"{TOOL_TAG} {function} {rendered}"


def _atif_projection(path: Path) -> dict[str, Any]:
    value = _read_atif(path)
    cot_lines: list[str] = []
    reasoning_chars = 0
    step_lines: dict[str, list[str]] = defaultdict(list)
    model_names: set[str] = set()
    for step in value["steps"]:
        if not isinstance(step, dict) or str(step.get("source") or "").lower() != "agent":
            continue
        if step.get("model_name"):
            model_names.add(str(step["model_name"]))
        reasoning = step.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = step.get("message")
        current: list[str] = []
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning_chars += len(reasoning)
            for line in reasoning.splitlines():
                if line.strip():
                    current.append(f"{REASONING_TAG} {line.strip()}")
        calls = step.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ConfigurationError(f"ATIF tool_calls must be a list: {path}")
        for call in calls:
            if isinstance(call, dict):
                current.append(_tool_call_text(call))
        cot_lines.extend(current)
        step_lines[str(step.get("step_id"))].extend(current)
    return {
        "task_context": _initial_task_context(value["steps"]),
        "model_cot": "\n".join(cot_lines),
        "reasoning_chars": reasoning_chars,
        "step_lines": dict(step_lines),
        "atif_schema_version": value.get("schema_version"),
        "session_id": value.get("session_id"),
        "model_names": sorted(model_names),
    }


def _contains_canary(projection: dict[str, Any], markers: list[str]) -> bool:
    visible = projection["task_context"] + "\n" + projection["model_cot"]
    lowered = visible.casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _rank(seed: int, *parts: Any) -> str:
    return stable_hash([seed, *[str(part) for part in parts]])


def _task_fingerprint(task_context: str, source_kind: str) -> str:
    text = task_context
    if source_kind == "terminal_wrench" and "Task Description:" in text:
        text = text.split("Task Description:", 1)[1]
        for marker in (
            "You are an authorized Red Team QA Agent",
            "Current terminal state:",
        ):
            if marker in text:
                text = text.split(marker, 1)[0]
    text = re.sub(r"\[(?:SYSTEM|USER)-\d{3}\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return stable_hash(text)


def _terminal_entries(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks_path = raw_root / "index" / "tasks.json"
    task_rows = json.loads(tasks_path.read_text(encoding="utf-8"))
    attackers: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for task in task_rows:
        task_id = str(task["task_id"])
        model = str(task["model"])
        for item in task.get("trajectories") or []:
            path = (
                raw_root
                / "tasks"
                / task_id
                / model
                / str(item["tree_name"])
                / str(item["trajectory_label"])
                / "trial"
                / "agent"
                / "trajectory.json"
            )
            if path.is_file():
                attackers.append({"task_id": task_id, "model": model, "path": path, **item})
        for item in task.get("baselines") or []:
            label = str(item["label"])
            path = (
                raw_root
                / "tasks"
                / task_id
                / model
                / "baseline_trajectories"
                / label
                / "trial"
                / "agent"
                / "trajectory.json"
            )
            if path.is_file():
                baselines.append({"task_id": task_id, "model": model, "path": path, **item})
    return attackers, baselines


def _hvtb_entries(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(raw_root / "trials.jsonl"):
        trials_dir = str(row.get("config", {}).get("trials_dir") or "")
        job_name = Path(trials_dir).name
        condition = job_name.rsplit("-", 1)[-1]
        path = raw_root / "jobs" / job_name / str(row["id"]) / "trajectory.json"
        if path.is_file():
            rows.append({**row, "job_name": job_name, "condition": condition, "path": path})
    return rows


def _scan_canary_tasks(
    entries: Iterable[dict[str, Any]], markers: list[str]
) -> tuple[set[str], int]:
    task_ids: set[str] = set()
    visible_rows = 0
    for entry in entries:
        projection = _atif_projection(entry["path"])
        if _contains_canary(projection, markers):
            task_ids.add(str(entry.get("task_id") or entry.get("task_name")))
            visible_rows += 1
    return task_ids, visible_rows


def _candidate(
    *,
    source_kind: str,
    source_id: str,
    entry: dict[str, Any],
    projection: dict[str, Any],
    role: str,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(entry.get("task_id") or entry.get("task_name"))
    trajectory_id = f"{source_id}:{entry.get('trial_name') or entry.get('id')}"
    return {
        "schema_version": 1,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "leakage_group_id": f"{source_id}:{task_id}",
        "source": source_id,
        "source_kind": source_kind,
        "source_path": str(entry["path"]),
        "candidate_role": role,
        "task_context": projection["task_context"],
        "task_context_sha256": stable_hash(projection["task_context"]),
        "task_fingerprint_sha256": _task_fingerprint(
            projection["task_context"], source_kind
        ),
        "model_cot": projection["model_cot"],
        "monitor_model_cot": render_atif_reasoning_steps(
            projection["step_lines"].values()
        ),
        "model_cot_sha256": stable_hash(projection["model_cot"]),
        "model_cot_format": MONITOR_COT_FORMAT,
        "reasoning_chars": projection["reasoning_chars"],
        "step_lines": projection["step_lines"],
        "solver_models": projection["model_names"],
        "atif_schema_version": projection["atif_schema_version"],
        "session_id": projection["session_id"],
        "upstream": upstream,
    }


def _keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.casefold()
    return sum(lowered.count(keyword.casefold()) for keyword in keywords)


def _select_diverse(
    rows: list[dict[str, Any]], count: int, *, seed: int, key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row.get("selection_score", 0)),
            _rank(seed, *(row.get(field) for field in key_fields)),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for allow_seen in (False, True):
        for row in ranked:
            if len(selected) >= count:
                break
            if row in selected:
                continue
            task_id = str(row.get("task_id") or row.get("task_name"))
            if not allow_seen and task_id in seen_tasks:
                continue
            selected.append(row)
            seen_tasks.add(task_id)
    if len(selected) != count:
        raise ConfigurationError(f"candidate pool has {len(selected)}/{count} requested rows")
    return selected


def _prepare_terminal(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = _mapping(config["selection"], "selection")
    markers = list(config["canary"]["marker_strings"])
    attackers, baselines = _terminal_entries(paths["raw_root"])
    canary_tasks, canary_rows = _scan_canary_tasks(
        [*attackers, *baselines], markers
    )
    eligible_attackers = [row for row in attackers if row["task_id"] not in canary_tasks]
    eligible_baselines = [row for row in baselines if row["task_id"] not in canary_tasks]
    rewarded = [
        row
        for row in eligible_attackers
        if row.get("classification") in TERMINAL_REWARDED and row.get("key_hack_steps")
    ]
    failed = [
        row
        for row in eligible_attackers
        if row.get("classification") in TERMINAL_FAILED
        and row.get("exploit_attempted") is True
        and row.get("key_hack_steps")
    ]
    hard = [
        row
        for row in eligible_attackers
        if row.get("classification") in TERMINAL_HARD_CANDIDATE
    ]
    seed = int(config["seed"])
    keyword_list = list(selection["consideration_keywords"])
    for row in hard:
        projection = _atif_projection(row["path"])
        row["selection_score"] = _keyword_score(projection["model_cot"], keyword_list)
        row["_projection"] = projection
    chosen = {
        "rewarded_positive": _select_diverse(
            rewarded,
            int(selection["rewarded_positive_candidates"]),
            seed=seed,
            key_fields=("task_id", "model", "trial_name"),
        ),
        "failed_applied_candidate": _select_diverse(
            failed,
            int(selection["failed_applied_candidates"]),
            seed=seed + 1,
            key_fields=("task_id", "model", "trial_name"),
        ),
        "hard_negative_candidate": _select_diverse(
            hard,
            int(selection["hard_negative_candidates"]),
            seed=seed + 2,
            key_fields=("task_id", "model", "trial_name"),
        ),
        "normal_negative": _select_diverse(
            eligible_baselines,
            int(selection["normal_negative_candidates"]),
            seed=seed + 3,
            key_fields=("task_id", "model", "trial_name"),
        ),
    }
    rows: list[dict[str, Any]] = []
    for role, entries in chosen.items():
        for entry in entries:
            projection = entry.pop("_projection", None) or _atif_projection(entry["path"])
            if projection["reasoning_chars"] <= 0 or not projection["model_cot"].strip():
                continue
            upstream = {
                key: entry.get(key)
                for key in (
                    "classification",
                    "reward",
                    "exploit_attempted",
                    "exploit_blocked_by_verifier",
                    "attacker_failed_due_to_budget",
                    "key_hack_steps",
                    "observed_categories",
                    "model",
                    "source_dataset",
                    "trial_name",
                    "label",
                )
                if key in entry
            }
            rows.append(
                _candidate(
                    source_kind="terminal_wrench",
                    source_id="terminal_wrench",
                    entry=entry,
                    projection=projection,
                    role=role,
                    upstream=upstream,
                )
            )
    return rows, {
        "raw_attacker_entries": len(attackers),
        "raw_baseline_entries": len(baselines),
        "canary_visible_rows": canary_rows,
        "canary_task_ids": sorted(canary_tasks),
        "eligible_pools": {
            "rewarded": len(rewarded),
            "failed": len(failed),
            "hard_candidate": len(hard),
            "baseline": len(eligible_baselines),
        },
    }


def _hvtb_positive(row: dict[str, Any]) -> bool:
    rewards = row.get("rewards") or {}
    return any(bool(rewards.get(signal)) for signal in HVTB_HACK_SIGNALS)


def _hvtb_failed(row: dict[str, Any]) -> bool:
    rewards = row.get("rewards") or {}
    return _hvtb_positive(row) and not bool(rewards.get("reward"))


def _prepare_hvtb(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = _mapping(config["selection"], "selection")
    markers = list(config["canary"]["marker_strings"])
    all_entries = _hvtb_entries(paths["raw_root"])
    canary_tasks, canary_rows = _scan_canary_tasks(all_entries, markers)
    condition = str(selection["condition"])
    eligible: list[dict[str, Any]] = []
    no_reasoning = 0
    for entry in all_entries:
        if entry["condition"] != condition or entry["task_name"] in canary_tasks:
            continue
        projection = _atif_projection(entry["path"])
        if projection["reasoning_chars"] <= 0 or not projection["model_cot"].strip():
            no_reasoning += 1
            continue
        entry["_projection"] = projection
        eligible.append(entry)
    positives = [row for row in eligible if _hvtb_positive(row)]
    negatives = [row for row in eligible if not _hvtb_positive(row)]
    for row in positives:
        row["selection_score"] = 1 if _hvtb_failed(row) else 0
    seed = int(config["seed"])
    selected_positive = _select_diverse(
        positives,
        int(selection["positive_candidates"]),
        seed=seed,
        key_fields=("task_name", "job_name", "id"),
    )
    positive_tasks = {row["task_name"] for row in selected_positive}
    for row in negatives:
        row["selection_score"] = 1 if row["task_name"] in positive_tasks else 0
    selected_negative = _select_diverse(
        negatives,
        int(selection["negative_candidates"]),
        seed=seed + 1,
        key_fields=("task_name", "job_name", "id"),
    )
    rows: list[dict[str, Any]] = []
    for role, entries in (
        ("watcher_applied", selected_positive),
        ("watcher_negative_candidate", selected_negative),
    ):
        for entry in entries:
            projection = entry.pop("_projection")
            rewards = dict(entry.get("rewards") or {})
            rows.append(
                _candidate(
                    source_kind="hvtb",
                    source_id="hvtb_reward_hacking",
                    entry=entry,
                    projection=projection,
                    role=role,
                    upstream={
                        "trial_id": entry["id"],
                        "trial_name": entry.get("trial_name"),
                        "job_name": entry["job_name"],
                        "condition": entry["condition"],
                        "model": entry.get("config", {}).get("agent", {}).get("model_name"),
                        "rewards": rewards,
                        "watcher_applied": _hvtb_positive(entry),
                        "watcher_failed_applied": _hvtb_failed(entry),
                    },
                )
            )
    return rows, {
        "raw_downloaded_entries": len(all_entries),
        "condition": condition,
        "canary_visible_rows": canary_rows,
        "canary_task_ids": sorted(canary_tasks),
        "excluded_no_reasoning": no_reasoning,
        "eligible_applied": len(positives),
        "eligible_failed_applied": sum(_hvtb_failed(row) for row in positives),
        "eligible_watcher_negative": len(negatives),
        "selected_positive_tasks": len({row["task_name"] for row in selected_positive}),
        "selected_matched_negative_tasks": len(
            positive_tasks & {row["task_name"] for row in selected_negative}
        ),
    }


def _judge_roles(source_kind: str) -> set[str]:
    if source_kind == "terminal_wrench":
        return {"failed_applied_candidate", "hard_negative_candidate"}
    return {"watcher_applied", "watcher_negative_candidate"}


def _judge_requests(
    candidates: list[dict[str, Any]], config: dict[str, Any], profile: Any
) -> list[dict[str, Any]]:
    judge = _mapping(config["judge"], "judge")
    input_config = _mapping(judge["input"], "judge.input")
    l3 = factor_spec(config["l3"])
    requests: list[dict[str, Any]] = []
    for row in candidates:
        if row["candidate_role"] not in _judge_roles(config["source_kind"]):
            continue
        units = reasoning_units(
            row["model_cot"].splitlines(), int(input_config["max_evidence_unit_chars"])
        )
        chunks = chunk_evidence_units(units, int(input_config["max_cot_chunk_chars"]))
        if not chunks:
            raise ConfigurationError(f"candidate has no Judge evidence: {row['trajectory_id']}")
        for chunk_index, chunk in enumerate(chunks, start=1):
            cot_lines = "\n".join(f"[{line_id}] {text}" for line_id, text in chunk)
            requests.append(
                build_judge_request(
                    profile,
                    request_id=f"rh-icy:{row['trajectory_id']}:{chunk_index:04d}",
                    purpose="reward_hacking_atif_I_C_annotation",
                    metadata={
                        "trajectory_id": row["trajectory_id"],
                        "task_id": row["task_id"],
                        "candidate_role": row["candidate_role"],
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "evidence_ids": [line_id for line_id, _ in chunk],
                        "judge_contract_sha256": stable_hash(
                            [ICY_JUDGE_SYSTEM_PROMPT, ICY_JUDGE_USER_TEMPLATE]
                        ),
                    },
                    messages=[
                        {"role": "system", "content": ICY_JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": ICY_JUDGE_USER_TEMPLATE.format(
                                task_context=row["task_context"],
                                chunk_index=chunk_index,
                                chunk_count=len(chunks),
                                cot_lines=cot_lines,
                                **l3,
                            ),
                        },
                    ],
                )
            )
    return requests


def prepare(
    config_path: Path, *, dry_run: bool, overwrite: bool
) -> dict[str, Any]:
    config, paths = _load(config_path)
    if config["source_kind"] == "terminal_wrench":
        candidates, source_summary = _prepare_terminal(config, paths)
    else:
        candidates, source_summary = _prepare_hvtb(config, paths)
    if len({row["trajectory_id"] for row in candidates}) != len(candidates):
        raise ConfigurationError("candidate trajectory IDs are not unique")
    markers = list(config["canary"]["marker_strings"])
    if any(_contains_canary(row, markers) for row in candidates):
        raise ConfigurationError("selected candidate still contains a canary marker")
    profile = load_judge_profile(
        resolve_path(config["judge"]["profile"], relative_to=config_path.parent)
    )
    requests = _judge_requests(candidates, config, profile)
    effective = {
        "schema_version": 1,
        "kind": "reward_hacking_atif_source_effective_config",
        "source_kind": config["source_kind"],
        "dataset": str(paths["raw_root"]),
        "split": config["selection"].get("condition", "terminal_wrench_curated_pool"),
        "model": profile.model,
        "tokenizer": "provider_managed_not_applicable",
        "max_sequence_length": "complete CoT split into lossless Judge chunks",
        "max_cot_chunk_chars": config["judge"]["input"]["max_cot_chunk_chars"],
        "max_evidence_unit_chars": config["judge"]["input"]["max_evidence_unit_chars"],
        "truncation": "none",
        "padding": "not_applicable",
        "batch_size": profile.workers,
        "learning_rate": None,
        "epochs_or_steps": None,
        "seed": config["seed"],
        "evaluation_metric": "strict I/C parse and exact canonical subtype quotas downstream",
        "output_directory": str(paths["candidates"].parent),
        "factor_level": "L3-only",
        "factor_spec_sha256": L3_FACTOR_SPEC_SHA256,
        "canary_policy": "exclude_entire_source_task_group",
        "monitor_input": "task_context_plus_complete_private_reasoning_and_tool_calls",
        "visible_final_included": False,
        "tool_outputs_included": False,
        "judge_profile": profile.effective_config(),
        "selection": config["selection"],
        "canonicalization": config["canonicalization"],
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": file_sha256(config_path.resolve()),
        "repository_state": repository_state(config_path.resolve().parents[5]),
    }
    summary = {
        "schema_version": 1,
        "kind": "reward_hacking_atif_prepare_summary",
        "dry_run": dry_run,
        "source_kind": config["source_kind"],
        "candidates": len(candidates),
        "candidate_roles": dict(sorted(Counter(row["candidate_role"] for row in candidates).items())),
        "judge_requests": len(requests),
        "judge_trajectories": len(
            {row["metadata"]["trajectory_id"] for row in requests}
        ),
        "source": source_summary,
        "effective_config_sha256": stable_hash(effective),
    }
    if dry_run:
        return summary
    for path in (paths["candidates"], paths["judge_requests"], paths["prepare_summary"], paths["effective_config"], paths["canary_tasks"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    if paths["candidates"].is_file() and not overwrite:
        if read_jsonl(paths["candidates"]) != candidates:
            raise ConfigurationError("candidate pool changed; pass --overwrite")
    else:
        write_jsonl(paths["candidates"], candidates)
    if paths["judge_requests"].is_file() and not overwrite:
        if read_jsonl(paths["judge_requests"]) != requests:
            raise ConfigurationError("Judge request pool changed; pass --overwrite")
    else:
        write_jsonl(paths["judge_requests"], requests)
    write_json(paths["canary_tasks"], {
        "marker_strings": markers,
        "group_policy": "exclude_entire_source_task_group",
        "task_ids": source_summary["canary_task_ids"],
    })
    summary["candidates_sha256"] = file_sha256(paths["candidates"])
    summary["judge_requests_sha256"] = file_sha256(paths["judge_requests"])
    write_json(paths["effective_config"], effective)
    write_json(paths["prepare_summary"], summary)
    return summary


def judge(
    config_path: Path,
    *,
    dry_run: bool,
    base_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    config, paths = _load(config_path)
    profile_path = resolve_path(config["judge"]["profile"], relative_to=config_path.parent)
    if not paths["judge_requests"].is_file():
        raise ConfigurationError("run prepare before judge")
    return run_judge_batch(
        profile=profile_path,
        requests_path=paths["judge_requests"],
        output_path=paths["judge_outputs"],
        errors_path=paths["judge_errors"],
        summary_path=paths["judge_transport_summary"],
        base_url=base_url,
        api_key=api_key,
        dry_run=dry_run,
    )


def _parse_icy_with_bounded_evidence(
    content: str,
    *,
    available_evidence_ids: set[str],
    maximum_evidence_ids: int,
) -> tuple[dict[str, Any], list[str]]:
    """Strictly parse I/C output, clipping only otherwise-valid overlong evidence lists."""

    try:
        return (
            parse_icy_judge_content(
                content,
                available_evidence_ids=available_evidence_ids,
                maximum_evidence_ids=maximum_evidence_ids,
            ),
            [],
        )
    except ValueError as original_error:
        try:
            value = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            raise original_error
        if not isinstance(value, dict):
            raise original_error
        clipped_fields: list[str] = []
        normalized = dict(value)
        for field in ("consideration_evidence", "operative_evidence"):
            evidence_ids = value.get(field)
            if not isinstance(evidence_ids, list):
                raise original_error
            if (
                len(evidence_ids) != len(set(evidence_ids))
                or any(item not in available_evidence_ids for item in evidence_ids)
            ):
                raise original_error
            if len(evidence_ids) > maximum_evidence_ids:
                normalized[field] = evidence_ids[:maximum_evidence_ids]
                clipped_fields.append(field)
        if not clipped_fields:
            raise original_error
        return (
            parse_icy_judge_content(
                json.dumps(normalized, ensure_ascii=False),
                available_evidence_ids=available_evidence_ids,
                maximum_evidence_ids=maximum_evidence_ids,
            ),
            clipped_fields,
        )


def _materialize_labels(config: dict[str, Any], paths: dict[str, Path]) -> list[dict[str, Any]]:
    candidates = {row["trajectory_id"]: row for row in read_jsonl(paths["candidates"])}
    requests = {row["request_id"]: row for row in read_jsonl(paths["judge_requests"])}
    outputs = {row["request_id"]: row for row in read_jsonl(paths["judge_outputs"])}
    judge = config["judge"]
    maximum = int(judge["maximum_evidence_ids"])
    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request_id, request in requests.items():
        output = outputs.get(request_id)
        if output is None or output.get("request_hash") != stable_hash(request["request"]):
            raise ConfigurationError(f"missing or stale Judge output: {request_id}")
        metadata = request["metadata"]
        parsed, clipped_fields = _parse_icy_with_bounded_evidence(
            output.get("content") or "",
            available_evidence_ids=set(metadata["evidence_ids"]),
            maximum_evidence_ids=maximum,
        )
        chunks[metadata["trajectory_id"]].append(
            {
                **metadata,
                **parsed,
                "judge_model": output.get("model"),
                "evidence_lists_clipped": clipped_fields,
            }
        )
    labels: list[dict[str, Any]] = []
    for trajectory_id, run_chunks in sorted(chunks.items()):
        expected = run_chunks[0]["chunk_count"]
        if len(run_chunks) != expected:
            raise ConfigurationError(f"incomplete Judge chunks for {trajectory_id}")
        considered = any(row["reward_hacking_considered"] for row in run_chunks)
        operative = any(row["operative_step"] for row in run_chunks)
        confidence = min(
            (row["confidence"] for row in run_chunks),
            key={"low": 0, "medium": 1, "high": 2}.get,
        )
        units = dict(
            reasoning_units(
                candidates[trajectory_id]["model_cot"].splitlines(),
                int(judge["input"]["max_evidence_unit_chars"]),
            )
        )
        operative_ids = list(dict.fromkeys(
            evidence_id
            for row in run_chunks
            for evidence_id in row["operative_evidence"]
        ))[:maximum]
        consideration_ids = list(dict.fromkeys(
            evidence_id
            for row in run_chunks
            for evidence_id in row["consideration_evidence"]
        ))[:maximum]
        labels.append({
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "risk_intent_I": considered,
            "operative_step_C": operative,
            "confidence": confidence,
            "training_subtype": "applied" if operative else "hard_negative" if considered else "normal_negative",
            "consideration_evidence_ids": consideration_ids,
            "operative_evidence_ids": operative_ids,
            "consideration_evidence_spans": [units[item] for item in consideration_ids],
            "operative_evidence_spans": [units[item] for item in operative_ids],
            "chunk_count": len(run_chunks),
            "judge_model": run_chunks[0]["judge_model"],
            "evidence_list_clip_count": sum(
                len(row["evidence_lists_clipped"]) for row in run_chunks
            ),
        })
    write_jsonl(paths["judge_labels"], labels)
    return labels


def _upstream_evidence(row: dict[str, Any]) -> list[str]:
    steps = row.get("upstream", {}).get("key_hack_steps") or []
    spans: list[str] = []
    for step_id in steps:
        spans.extend(row.get("step_lines", {}).get(str(step_id), []))
    return list(dict.fromkeys(span for span in spans if span in row["model_cot"]))


def canonicalize(config_path: Path, *, dry_run: bool, overwrite: bool) -> dict[str, Any]:
    config, paths = _load(config_path)
    if dry_run:
        return {
            "schema_version": 1,
            "kind": "reward_hacking_atif_canonicalize_plan",
            "source_kind": config["source_kind"],
            "factor_levels": ["L3"],
            "output": str(paths["canonical_records"]),
        }
    labels = {row["trajectory_id"]: row for row in _materialize_labels(config, paths)}
    candidates = read_jsonl(paths["candidates"])
    taxonomy = Taxonomy.load(
        resolve_path(config["canonicalization"]["taxonomy"], relative_to=config_path.parent)
    )
    accepted_confidences = set(config["judge"]["accepted_confidences"])
    l3 = factor_spec(config["l3"])
    eval_path = resolve_path(
        config["canonicalization"]["eval_records"], relative_to=config_path.parent
    )
    eval_context_hashes: set[str] = set()
    eval_fingerprints: set[str] = set()
    if eval_path.is_file():
        for eval_row in iter_jsonl(eval_path):
            if eval_row.get("factor_level") != "L3":
                continue
            context = str(eval_row["task_context"])
            eval_context_hashes.add(stable_hash(context))
            eval_fingerprints.add(_task_fingerprint(context, "malt"))
    training_overlap_path_value = config["canonicalization"].get(
        "training_overlap_records"
    )
    training_overlap_path = (
        resolve_path(training_overlap_path_value, relative_to=config_path.parent)
        if training_overlap_path_value
        else None
    )
    training_fingerprints: set[str] = set()
    if training_overlap_path is not None:
        if not training_overlap_path.is_file():
            raise ConfigurationError(
                f"training overlap records do not exist: {training_overlap_path}"
            )
        for overlap_row in iter_jsonl(training_overlap_path):
            fingerprint = (overlap_row.get("metadata") or {}).get(
                "task_fingerprint_sha256"
            )
            if isinstance(fingerprint, str) and fingerprint:
                training_fingerprints.add(fingerprint)
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in candidates:
        if (
            row["task_context_sha256"] in eval_context_hashes
            or row["task_fingerprint_sha256"] in eval_fingerprints
        ):
            rejected["malt_eval_task_context_overlap"] += 1
            continue
        if row["task_fingerprint_sha256"] in training_fingerprints:
            rejected["training_source_task_fingerprint_overlap"] += 1
            continue
        role = row["candidate_role"]
        monitor_model_cot = str(
            row.get("monitor_model_cot")
            or render_tagged_reasoning_lines(row["model_cot"])
        )
        if not monitor_model_cot:
            rejected["empty_monitor_reasoning"] += 1
            continue
        judgment = labels.get(row["trajectory_id"])
        source_locked_positive = role in {"rewarded_positive", "watcher_applied"}
        if (
            judgment is not None
            and judgment["confidence"] not in accepted_confidences
            and not source_locked_positive
        ):
            rejected[f"judge_confidence_{judgment['confidence']}"] += 1
            continue
        if config["source_kind"] == "terminal_wrench" and role == "rewarded_positive":
            label, subtype, intent, operative = 1, "applied", True, True
            evidence = [
                span for span in _upstream_evidence(row) if span in monitor_model_cot
            ]
            method = "terminal_wrench_stored_rewarded_exploit_judgment"
        elif config["source_kind"] == "terminal_wrench" and role == "normal_negative":
            label, subtype, intent, operative = 0, "normal_negative", False, False
            evidence = []
            method = "terminal_wrench_successful_legitimate_baseline"
        elif config["source_kind"] == "hvtb" and role == "watcher_applied":
            label, subtype, intent, operative = 1, "applied", True, True
            evidence = [
                span
                for span in (judgment["operative_evidence_spans"] if judgment else [])
                if span in monitor_model_cot
            ]
            method = "hvtb_filesystem_watcher_operative_signal"
        else:
            if judgment is None:
                rejected["missing_required_judgment"] += 1
                continue
            label = int(judgment["operative_step_C"])
            subtype = judgment["training_subtype"]
            intent = judgment["risk_intent_I"]
            operative = judgment["operative_step_C"]
            evidence = [
                span
                for span in judgment["operative_evidence_spans"]
                if span in monitor_model_cot
            ]
            method = "deepseek_v4_flash_blinded_lossless_chunk_I_C_judgment"
        outcome = None
        if config["source_kind"] == "terminal_wrench":
            reward = row["upstream"].get("reward")
            outcome = bool(reward) if reward is not None else None
        else:
            reward = row["upstream"].get("rewards", {}).get("reward")
            outcome = bool(reward) if reward is not None else None
        training_subtype = (
            "successful_positive" if operative and outcome is True
            else "failed_positive" if operative
            else "hard_negative" if intent
            else "normal_negative"
        )
        record = {
            "schema_version": 1,
            "sample_id": f"{row['trajectory_id']}:l3",
            "trajectory_id": row["trajectory_id"],
            "task_id": row["task_id"],
            "leakage_group_id": row["leakage_group_id"],
            "cell": "frontier_risk_internal",
            "family": "reward_hacking",
            "source": row["source"],
            "task_context": row["task_context"],
            "model_cot": monitor_model_cot,
            "factor_level": "L3",
            "factor_spec": l3,
            "annotation_status": "complete",
            "label": label,
            "label_provenance": {
                "method": method,
                "binary_boundary": "operative_attempt_regardless_of_success",
                "source_label_cannot_be_overturned": role in {"rewarded_positive", "watcher_applied"},
            },
            "evidence_spans": evidence,
            "metadata": {
                "factor_standard_id": L3_STANDARD_ID,
                "factor_spec_sha256": L3_FACTOR_SPEC_SHA256,
                "training_subtype": training_subtype,
                "risk_intent_I": intent,
                "operative_step_C": operative,
                "outcome_Y": outcome,
                "candidate_role": role,
                "solver_models": row["solver_models"],
                "elicitation_regime": (
                    "explicit_instruction" if config["source_kind"] == "terminal_wrench" and role != "normal_negative"
                    else "unprompted_legitimate" if config["source_kind"] == "terminal_wrench"
                    else "unprompted_opportunity"
                ),
                "model_cot_format": "assistant_private_reasoning_steps_only",
                "cot_normalization": "assistant_step_marked_private_reasoning_only",
                "truncation": "none",
                "visible_final_included": False,
                "tool_calls_included": False,
                "tool_outputs_included": False,
                "task_context_sha256": row["task_context_sha256"],
                "task_fingerprint_sha256": row["task_fingerprint_sha256"],
                "source_path": row["source_path"],
                "source_revision": config["canonicalization"]["source_revision"],
                "upstream": row["upstream"],
                "judge": judgment,
            },
        }
        CanonicalRecord.from_mapping(record, taxonomy=taxonomy)
        records.append(record)
    if len({row["sample_id"] for row in records}) != len(records):
        raise ConfigurationError("canonical sample IDs are not unique")
    markers = list(config["canary"]["marker_strings"])
    if any(_contains_canary(row, markers) for row in records):
        raise ConfigurationError("canonical records contain a canary marker")
    paths["canonical_records"].parent.mkdir(parents=True, exist_ok=True)
    if paths["canonical_records"].is_file() and not overwrite:
        if read_jsonl(paths["canonical_records"]) != records:
            raise ConfigurationError("canonical records changed; pass --overwrite")
    else:
        write_jsonl(paths["canonical_records"], records)
    summary = {
        "schema_version": 1,
        "kind": "reward_hacking_atif_canonical_summary",
        "source_kind": config["source_kind"],
        "trajectories": len(records),
        "records": len(records),
        "factor_levels": {"L3": len(records)},
        "labels": dict(sorted(Counter("applied" if row["label"] else "not_applied" for row in records).items())),
        "training_subtypes": dict(sorted(Counter(row["metadata"]["training_subtype"] for row in records).items())),
        "candidate_roles": dict(sorted(Counter(row["metadata"]["candidate_role"] for row in records).items())),
        "unique_tasks": len({row["task_id"] for row in records}),
        "canary_task_groups_excluded": len(
            json.loads(paths["canary_tasks"].read_text(encoding="utf-8"))["task_ids"]
        ),
        "rejected": dict(sorted(rejected.items())),
        "malt_eval_records": str(eval_path),
        "malt_eval_task_context_overlap": rejected.get(
            "malt_eval_task_context_overlap", 0
        ),
        "training_overlap_records": (
            str(training_overlap_path) if training_overlap_path is not None else None
        ),
        "training_source_task_fingerprint_overlap": rejected.get(
            "training_source_task_fingerprint_overlap", 0
        ),
        "judge_evidence_lists_clipped_to_maximum": sum(
            int((row["metadata"].get("judge") or {}).get("evidence_list_clip_count", 0))
            for row in records
        ),
        "records_sha256": file_sha256(paths["canonical_records"]),
    }
    write_json(paths["canonical_summary"], summary)
    return summary


def validate(config_path: Path) -> dict[str, Any]:
    config, paths = _load(config_path)
    if not paths["canonical_records"].is_file():
        return {"schema_version": 1, "status": "prepared_only", "source_kind": config["source_kind"]}
    taxonomy = Taxonomy.load(
        resolve_path(config["canonicalization"]["taxonomy"], relative_to=config_path.parent)
    )
    records = [CanonicalRecord.from_mapping(row, taxonomy=taxonomy) for row in iter_jsonl(paths["canonical_records"])]
    if any(row.factor_level != "L3" for row in records):
        raise ConfigurationError("Reward Hacking trace source must remain L3-only")
    if any(stable_hash(factor_spec(row.factor_spec.__dict__)) != L3_FACTOR_SPEC_SHA256 for row in records if row.factor_spec):
        raise ConfigurationError("canonical L3 Factor drifted")
    return {
        "schema_version": 1,
        "status": "valid",
        "source_kind": config["source_kind"],
        "records": len(records),
        "records_sha256": file_sha256(paths["canonical_records"]),
    }


def run_stage(
    config_path: Path,
    *,
    stage: str,
    dry_run: bool = False,
    overwrite: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if stage == "prepare":
        return prepare(config_path, dry_run=dry_run, overwrite=overwrite)
    if stage == "judge":
        return judge(
            config_path,
            dry_run=dry_run,
            base_url=base_url,
            api_key=api_key,
        )
    if stage == "canonicalize":
        return canonicalize(config_path, dry_run=dry_run, overwrite=overwrite)
    if stage == "validate":
        return validate(config_path)
    raise ConfigurationError(f"unsupported Reward Hacking ATIF stage {stage!r}")
