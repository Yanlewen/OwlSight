"""Zero-shot Factor monitoring with an offline vLLM forced-choice backend."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    repository_state,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .io import read_jsonl, write_json, write_jsonl
from .metrics import best_threshold_metrics
from .openai_intervention import score_artifact
from .prompt_contract import (
    FACTOR_SUFFIX_CHOICE_CONTRACT_ID,
    factor_suffix_choice_contract,
    factor_suffix_choice_token_ids,
    render_factor_suffix_choice_text,
    tokenize_factor_suffix_choice_prompt,
)
from .process_eval import score_process_artifact
from .schema import Taxonomy
from .training.backend import _evaluation_summary, _prediction_only_summary
from .training.data import FactorQueryRecord, TrajectoryGroup, load_trajectory_groups
from .training.encoding import FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID, contract_spec


BACKEND_NAME = "vllm_forced_choice"
SUITE_KIND = "vllm_forced_choice_eval_suite"
EFFECTIVE_KIND = "vllm_forced_choice_eval_suite_effective_config"
CHOICE_CONTRACT_ID = FACTOR_SUFFIX_CHOICE_CONTRACT_ID
_BENCHMARK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SCORERS = {"binary", "openai_process", "openai_intervention"}
_PARTIAL_FILE = "scores.partial.jsonl"
_FINAL_FILE = "scores.jsonl"
_PROGRESS_FILE = "progress.json"


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not lower <= result <= upper:
        raise ConfigurationError(f"{name} must be in [{lower}, {upper}]")
    return result


choice_contract = factor_suffix_choice_contract


def _source_eval_dataset(config_path: Path) -> dict[str, Any]:
    source = load_config(config_path)
    require_keys(source, {"dataset", "runtime", "evaluation"}, str(config_path))
    dataset = require_mapping(source["dataset"], f"{config_path}.dataset")
    require_keys(dataset, {"directory", "trajectory_file", "query_file"}, "dataset")
    directory = resolve_path(dataset["directory"], relative_to=config_path.parent)
    trajectory = resolve_path(dataset["trajectory_file"], relative_to=directory)
    query = resolve_path(dataset["query_file"], relative_to=directory)
    for name, path in (("trajectory_file", trajectory), ("query_file", query)):
        if not path.is_file():
            raise ConfigurationError(f"{config_path} {name} does not exist: {path}")
    evaluation = require_mapping(source["evaluation"], f"{config_path}.evaluation")
    return {
        "source_eval_config": str(config_path),
        "source_eval_config_sha256": file_sha256(config_path),
        "source_eval_config_resolved_hash": stable_hash(source),
        "directory": str(directory),
        "files": {
            "trajectory_file": {"path": str(trajectory), "sha256": file_sha256(trajectory)},
            "query_file": {"path": str(query), "sha256": file_sha256(query)},
        },
        "source_metric": evaluation.get("metric"),
        "reporting_dimensions": evaluation.get("reporting_dimensions"),
    }


def resolve_suite_config(config_path: Path) -> dict[str, Any]:
    """Resolve one strict suite config without importing or initializing vLLM."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    fields = {
        "schema_version",
        "kind",
        "model",
        "taxonomy",
        "monitor",
        "runtime",
        "evaluation",
        "cluster",
        "benchmarks",
        "output",
    }
    require_keys(config, fields, "vLLM baseline suite")
    reject_unknown(config, fields, "vLLM baseline suite")
    if config["schema_version"] != 1 or config["kind"] != SUITE_KIND:
        raise ConfigurationError(
            f"suite requires schema_version=1 and kind={SUITE_KIND!r}"
        )
    base = config_path.parent

    model = require_mapping(config["model"], "model")
    required_model_fields = {
        "checkpoint",
        "tokenizer",
        "checkpoint_revision",
        "tokenizer_revision",
        "dtype",
        "trust_remote_code",
    }
    model_fields = required_model_fields | {"adapter"}
    require_keys(model, required_model_fields, "model")
    reject_unknown(model, model_fields, "model")
    checkpoint = resolve_path(model["checkpoint"], relative_to=base)
    tokenizer = resolve_path(model["tokenizer"], relative_to=base)
    if not isinstance(model["trust_remote_code"], bool):
        raise ConfigurationError("model.trust_remote_code must be boolean")
    if model["dtype"] not in {"bfloat16", "float16", "float32", "auto"}:
        raise ConfigurationError("model.dtype must be bfloat16/float16/float32/auto")
    adapter_value = model.get("adapter")
    model_adapter = None
    if adapter_value is not None:
        adapter_config = require_mapping(adapter_value, "model.adapter")
        adapter_fields = {"path", "name", "id", "rank"}
        require_keys(adapter_config, adapter_fields, "model.adapter")
        reject_unknown(adapter_config, adapter_fields, "model.adapter")
        model_adapter = {
            "path": str(resolve_path(adapter_config["path"], relative_to=base)),
            "name": _nonempty(adapter_config["name"], "model.adapter.name"),
            "id": _integer(adapter_config["id"], "model.adapter.id"),
            "rank": _integer(adapter_config["rank"], "model.adapter.rank"),
            "access_validation": "worker_preflight",
        }

    taxonomy_path = resolve_path(config["taxonomy"], relative_to=base)
    if not taxonomy_path.is_file():
        raise ConfigurationError(f"taxonomy does not exist: {taxonomy_path}")

    monitor = require_mapping(config["monitor"], "monitor")
    monitor_fields = {
        "backend",
        "endpoint_encoding_contract_id",
        "choice_contract_id",
        "chat_template",
        "chat_template_kwargs",
    }
    require_keys(monitor, monitor_fields, "monitor")
    reject_unknown(monitor, monitor_fields, "monitor")
    if monitor["backend"] != BACKEND_NAME:
        raise ConfigurationError(f"monitor.backend must be {BACKEND_NAME!r}")
    if monitor["endpoint_encoding_contract_id"] != FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID:
        raise ConfigurationError(
            f"only {FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID!r} is supported"
        )
    if monitor["choice_contract_id"] != CHOICE_CONTRACT_ID:
        raise ConfigurationError(f"monitor.choice_contract_id must be {CHOICE_CONTRACT_ID!r}")
    if monitor["chat_template"] != "tokenizer_native":
        raise ConfigurationError("monitor.chat_template must be tokenizer_native")
    chat_kwargs = require_mapping(monitor["chat_template_kwargs"], "chat_template_kwargs")

    runtime = require_mapping(config["runtime"], "runtime")
    runtime_fields = {
        "max_sequence_length",
        "truncation_policy",
        "padding_policy",
        "seed",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "max_num_seqs",
        "request_chunk_size",
        "progress_interval_queries",
    }
    optional_runtime_fields = {"enforce_eager", "natural_label_audit_queries"}
    require_keys(runtime, runtime_fields, "runtime")
    reject_unknown(runtime, runtime_fields | optional_runtime_fields, "runtime")
    if runtime["truncation_policy"] != "error":
        raise ConfigurationError("runtime.truncation_policy must be error")
    if runtime["padding_policy"] != "vllm_continuous_batching":
        raise ConfigurationError(
            "runtime.padding_policy must be vllm_continuous_batching"
        )
    for key in ("enable_prefix_caching",):
        if not isinstance(runtime[key], bool):
            raise ConfigurationError(f"runtime.{key} must be boolean")
    if "enforce_eager" in runtime and not isinstance(runtime["enforce_eager"], bool):
        raise ConfigurationError("runtime.enforce_eager must be boolean")
    resolved_runtime = {
        **runtime,
        "enforce_eager": bool(runtime.get("enforce_eager", False)),
        "max_sequence_length": _integer(
            runtime["max_sequence_length"], "runtime.max_sequence_length"
        ),
        "seed": _integer(runtime["seed"], "runtime.seed", minimum=0),
        "tensor_parallel_size": _integer(
            runtime["tensor_parallel_size"], "runtime.tensor_parallel_size"
        ),
        "gpu_memory_utilization": _number(
            runtime["gpu_memory_utilization"],
            "runtime.gpu_memory_utilization",
            lower=0.1,
            upper=1.0,
        ),
        "max_num_seqs": _integer(runtime["max_num_seqs"], "runtime.max_num_seqs"),
        "request_chunk_size": _integer(
            runtime["request_chunk_size"], "runtime.request_chunk_size"
        ),
        "progress_interval_queries": _integer(
            runtime["progress_interval_queries"], "runtime.progress_interval_queries"
        ),
    }
    if "natural_label_audit_queries" in runtime:
        resolved_runtime["natural_label_audit_queries"] = _integer(
            runtime["natural_label_audit_queries"],
            "runtime.natural_label_audit_queries",
            minimum=0,
        )
    if (
        resolved_runtime["progress_interval_queries"]
        != resolved_runtime["request_chunk_size"]
    ):
        raise ConfigurationError(
            "runtime.progress_interval_queries must equal request_chunk_size so every "
            "completed vLLM request chunk is durably checkpointed"
        )

    evaluation = require_mapping(config["evaluation"], "evaluation")
    evaluation_fields = {"metric", "threshold_policy", "threshold_value", "oracle_metric"}
    require_keys(evaluation, evaluation_fields, "evaluation")
    reject_unknown(evaluation, evaluation_fields, "evaluation")
    if evaluation["metric"] != "gmean2" or evaluation["oracle_metric"] != "gmean2":
        raise ConfigurationError("evaluation metric and oracle_metric must both be gmean2")
    expected_threshold_policy = (
        "fixed_choice_0.5"
        if model_adapter is not None
        else "zero_shot_forced_choice"
    )
    if evaluation["threshold_policy"] != expected_threshold_policy:
        raise ConfigurationError(
            f"evaluation.threshold_policy must be {expected_threshold_policy}"
        )
    if float(evaluation["threshold_value"]) != 0.5:
        raise ConfigurationError("zero-shot forced-choice threshold must be 0.5")

    output = require_mapping(config["output"], "output")
    require_keys(output, {"directory"}, "output")
    reject_unknown(output, {"directory"}, "output")
    output_root = resolve_path(output["directory"], relative_to=base)

    benchmarks = config["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ConfigurationError("benchmarks must be a non-empty list")
    resolved_benchmarks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    observed_families: set[str] = set()
    for index, value in enumerate(benchmarks):
        item = require_mapping(value, f"benchmarks[{index}]")
        item_fields = {"id", "family", "source_eval_config", "metric_adapter", "output_subdir"}
        require_keys(item, item_fields, f"benchmarks[{index}]")
        reject_unknown(item, item_fields, f"benchmarks[{index}]")
        benchmark_id = _nonempty(item["id"], f"benchmarks[{index}].id")
        if not _BENCHMARK_ID.fullmatch(benchmark_id) or benchmark_id in seen_ids:
            raise ConfigurationError(f"invalid or duplicate benchmark id: {benchmark_id!r}")
        seen_ids.add(benchmark_id)
        family = _nonempty(item["family"], f"benchmarks[{index}].family")
        observed_families.add(family)
        metric_adapter = _nonempty(
            item["metric_adapter"], f"benchmarks[{index}].metric_adapter"
        )
        if metric_adapter not in _SCORERS:
            raise ConfigurationError(
                f"unsupported metric_adapter: {metric_adapter!r}"
            )
        source_config = resolve_path(item["source_eval_config"], relative_to=base)
        dataset = _source_eval_dataset(source_config)
        output_directory = resolve_path(item["output_subdir"], relative_to=output_root)
        if output_directory in seen_outputs:
            raise ConfigurationError(f"duplicate benchmark output: {output_directory}")
        seen_outputs.add(output_directory)
        resolved_benchmarks.append(
            {
                "id": benchmark_id,
                "family": family,
                "metric_adapter": metric_adapter,
                "dataset": dataset,
                "output_directory": str(output_directory),
            }
        )
    if len(observed_families) != 9:
        raise ConfigurationError(
            f"nine-family suite must contain exactly 9 families, found {sorted(observed_families)}"
        )

    cluster = require_mapping(config["cluster"], "cluster")
    cluster_fields = {
        "gpu",
        "cpu",
        "memory_mb",
        "namespace",
        "charged_group",
        "private_machine",
        "image",
        "host_network",
        "mounts",
        "job_name",
    }
    require_keys(cluster, cluster_fields, "cluster")
    reject_unknown(cluster, cluster_fields, "cluster")
    mounts = cluster["mounts"]
    if not isinstance(mounts, list) or not mounts or not all(
        isinstance(item, str) and item for item in mounts
    ):
        raise ConfigurationError("cluster.mounts must be a non-empty string list")
    resolved_cluster = {
        **cluster,
        "gpu": _integer(cluster["gpu"], "cluster.gpu"),
        "cpu": _integer(cluster["cpu"], "cluster.cpu"),
        "memory_mb": _integer(cluster["memory_mb"], "cluster.memory_mb"),
        "namespace": _nonempty(cluster["namespace"], "cluster.namespace"),
        "charged_group": _nonempty(cluster["charged_group"], "cluster.charged_group"),
        "private_machine": _nonempty(cluster["private_machine"], "cluster.private_machine"),
        "image": _nonempty(cluster["image"], "cluster.image"),
        "job_name": _nonempty(cluster["job_name"], "cluster.job_name"),
    }
    if resolved_cluster["gpu"] != resolved_runtime["tensor_parallel_size"]:
        raise ConfigurationError("cluster.gpu must equal runtime.tensor_parallel_size")
    if not isinstance(cluster["host_network"], bool):
        raise ConfigurationError("cluster.host_network must be boolean")

    choice = choice_contract()
    effective = {
        "schema_version": 1,
        "kind": EFFECTIVE_KIND,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_hash": stable_hash(config),
        "model": {
            **model,
            "checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer),
            "access_validation": "worker_preflight",
            **(
                {"adapter": model_adapter}
                if model_adapter is not None
                else {}
            ),
        },
        "taxonomy": {"path": str(taxonomy_path), "sha256": file_sha256(taxonomy_path)},
        "monitor": {
            **monitor,
            "backend": BACKEND_NAME,
            "visibility_scope": "cot_only",
            "endpoint_encoding_contract": {
                "id": FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
                "spec": contract_spec(FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID),
                "hash": stable_hash(contract_spec(FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID)),
            },
            "choice_contract": {**choice, "hash": stable_hash(choice)},
            "chat_template_kwargs": chat_kwargs,
        },
        "runtime": resolved_runtime,
        "optimization": {
            "strategy": "not_applicable",
            "objective": (
                "causal_lm_choice_adapter_inference"
                if model_adapter is not None
                else "zero_shot_forced_choice_inference"
            ),
            "batch_size": resolved_runtime["request_chunk_size"],
            "learning_rate": "not_applicable",
            "epochs_or_steps": "not_applicable",
            "optimizer": "not_applicable",
            "scheduler": "not_applicable",
            "checkpoint_policy": "not_applicable",
        },
        "evaluation": {**evaluation, "threshold_value": 0.5},
        "benchmarks": resolved_benchmarks,
        "output_directory": str(output_root),
        "cluster": resolved_cluster,
        "repository": repository_state(config_path.parent),
    }
    effective["suite_identity"] = stable_hash(
        {key: value for key, value in effective.items() if key not in {"repository", "suite_identity"}}
    )
    return effective


def _prompt_text(group: TrajectoryGroup, query: FactorQueryRecord, instruction: str) -> str:
    return render_factor_suffix_choice_text(group, query, instruction)


def _prompt_token_ids(tokenizer: Any, text: str, chat_kwargs: Mapping[str, Any]) -> list[int]:
    return tokenize_factor_suffix_choice_prompt(tokenizer, text, chat_kwargs)


def _choice_token_ids(tokenizer: Any) -> dict[str, int]:
    return factor_suffix_choice_token_ids(tokenizer)


def normalized_applied_score(logp_a: float, logp_b: float) -> float:
    """Return the stable two-choice probability assigned to applied (choice B)."""

    if not math.isfinite(logp_a) or not math.isfinite(logp_b):
        raise RuntimeError("choice log probabilities must be finite")
    difference = logp_a - logp_b
    if difference >= 0:
        return math.exp(-difference) / (1.0 + math.exp(-difference))
    return 1.0 / (1.0 + math.exp(difference))


def _available_choice_logprobs(
    output: Any, choice_ids: Mapping[str, int]
) -> dict[str, float]:
    candidates = getattr(output, "outputs", None)
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("vLLM forced-choice request must return exactly one sequence")
    logprobs = getattr(candidates[0], "logprobs", None)
    if not isinstance(logprobs, list) or len(logprobs) != 1:
        raise RuntimeError("vLLM forced-choice request must return one token logprob map")
    values = logprobs[0]
    result: dict[str, float] = {}
    for choice, token_id in choice_ids.items():
        item = values.get(token_id) if hasattr(values, "get") else None
        value = getattr(item, "logprob", item)
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            result[choice] = float(value)
    return result


def _extract_choice_logprobs(output: Any, choice_ids: Mapping[str, int]) -> tuple[float, float]:
    result = _available_choice_logprobs(output, choice_ids)
    for choice, token_id in choice_ids.items():
        if choice not in result:
            raise RuntimeError(f"vLLM output omitted forced choice {choice} token {token_id}")
    return result["A"], result["B"]


def _query_examples(groups: Sequence[TrajectoryGroup]) -> list[tuple[TrajectoryGroup, FactorQueryRecord]]:
    return [(group, query) for group in groups for query in group.queries]


def _benchmark_effective(
    suite: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = benchmark["dataset"]
    output = benchmark["output_directory"]
    return {
        "schema_version": 1,
        "kind": "eval_run",
        "suite_identity": suite["suite_identity"],
        "benchmark_id": benchmark["id"],
        "model": dict(suite["model"]),
        "monitor": dict(suite["monitor"]),
        "dataset": {
            "directory": dataset["directory"],
            "split": "benchmark",
            "files": dataset["files"],
            "source_eval_config": dataset["source_eval_config"],
            "source_eval_config_sha256": dataset["source_eval_config_sha256"],
        },
        "runtime": {
            "max_sequence_length": suite["runtime"]["max_sequence_length"],
            "truncation_policy": "error",
            "padding_policy": suite["runtime"]["padding_policy"],
            "batch_size": suite["runtime"]["request_chunk_size"],
            "seed": suite["runtime"]["seed"],
        },
        "optimization": suite["optimization"],
        "evaluation": {
            **suite["evaluation"],
            **(
                {"reporting_dimensions": dataset["reporting_dimensions"]}
                if dataset.get("reporting_dimensions") is not None
                else {}
            ),
        },
        "output_directory": output,
        "repository": suite["repository"],
    }


def _resume_identity(effective: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in effective.items() if key != "repository"})


def _load_resume(
    output: Path,
    effective: Mapping[str, Any],
    examples: Sequence[tuple[TrajectoryGroup, FactorQueryRecord]],
) -> tuple[list[dict[str, Any]], int, bool]:
    expected_identity = _resume_identity(effective)
    progress_path = output / _PROGRESS_FILE
    partial_path = output / _PARTIAL_FILE
    final_path = output / _FINAL_FILE
    if not progress_path.exists() and not partial_path.exists() and not final_path.exists():
        return [], 0, False
    if not progress_path.is_file():
        raise RuntimeError(f"evaluation resume lacks {_PROGRESS_FILE}: {output}")
    progress = load_config(progress_path)
    if progress.get("resume_identity") != expected_identity:
        raise RuntimeError(f"evaluation resume identity mismatch: {output}")
    completed = progress.get("completed_queries")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise RuntimeError("completed_queries must be an integer")
    source = final_path if progress.get("status") == "completed" else partial_path
    if not source.is_file():
        raise RuntimeError(f"evaluation resume score file is missing: {source}")
    rows = read_jsonl(source)
    expected_ids = [query.query_id for _, query in examples[: len(rows)]]
    if [row.get("query_id") for row in rows] != expected_ids or completed != len(rows):
        raise RuntimeError(f"durable score prefix disagrees with progress: {output}")
    if len(rows) > len(examples):
        raise RuntimeError("evaluation resume contains more scores than frozen queries")
    is_complete = progress.get("status") == "completed"
    if is_complete and len(rows) != len(examples):
        raise RuntimeError("completed evaluation does not cover all frozen queries")
    return rows, len(rows), is_complete


def _write_progress(
    output: Path,
    effective: Mapping[str, Any],
    *,
    completed_queries: int,
    total_queries: int,
    status: str,
) -> None:
    write_json(
        output / _PROGRESS_FILE,
        {
            "schema_version": 1,
            "status": status,
            "completed_queries": completed_queries,
            "total_queries": total_queries,
            "resume_identity": _resume_identity(effective),
            "partial_scores_file": _PARTIAL_FILE,
            "resume_policy": "same_effective_config_validated_query_prefix",
        },
    )


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o664)


def _update_length_audit(
    output: Path,
    *,
    completed_queries: int,
    total_queries: int,
    batch_max_prompt_tokens: int,
    max_sequence_length: int,
) -> None:
    path = output / "length_audit.json"
    previous_max = 0
    if path.is_file():
        previous = load_config(path)
        previous_max = int(previous.get("observed_max_prompt_tokens", 0))
    observed_max = max(previous_max, batch_max_prompt_tokens)
    write_json(
        path,
        {
            "schema_version": 1,
            "completed_queries": completed_queries,
            "total_queries": total_queries,
            "observed_max_prompt_tokens": observed_max,
            "generation_tokens": 1,
            "max_sequence_length": max_sequence_length,
            "truncation_policy": "error",
            "within_limit": observed_max + 1 <= max_sequence_length,
        },
    )


def _score_rows(
    llm: Any,
    sampling_params: Any,
    fallback_sampling_params: Mapping[str, Any],
    tokenizer: Any,
    selected: Sequence[tuple[TrajectoryGroup, FactorQueryRecord]],
    *,
    choice_ids: Mapping[str, int],
    instruction: str,
    chat_kwargs: Mapping[str, Any],
    max_sequence_length: int,
    lora_request: Any | None = None,
) -> tuple[list[dict[str, Any]], int]:
    prompts: list[dict[str, list[int]]] = []
    maximum = 0
    for group, query in selected:
        ids = _prompt_token_ids(
            tokenizer,
            _prompt_text(group, query, instruction),
            chat_kwargs,
        )
        maximum = max(maximum, len(ids))
        if len(ids) + 1 > max_sequence_length:
            raise RuntimeError(
                f"query {query.query_id} requires {len(ids) + 1} tokens, above "
                f"max_sequence_length={max_sequence_length}; truncation is forbidden"
            )
        prompts.append({"prompt_token_ids": ids})
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    if len(outputs) != len(selected):
        raise RuntimeError("vLLM output count differs from request count")
    choice_logprobs = [
        _available_choice_logprobs(result, choice_ids) for result in outputs
    ]
    fallback_counts: dict[str, int] = {}
    for choice, token_id in choice_ids.items():
        missing = [
            index for index, values in enumerate(choice_logprobs) if choice not in values
        ]
        if not missing:
            continue
        fallback_counts[choice] = len(missing)
        fallback_outputs = llm.generate(
            [prompts[index] for index in missing],
            fallback_sampling_params[choice],
            use_tqdm=False,
            lora_request=lora_request,
        )
        if len(fallback_outputs) != len(missing):
            raise RuntimeError("vLLM fallback output count differs from request count")
        for index, result in zip(missing, fallback_outputs, strict=True):
            values = _available_choice_logprobs(result, {choice: token_id})
            if choice not in values:
                raise RuntimeError(
                    f"vLLM single-choice fallback omitted {choice} token {token_id}"
                )
            choice_logprobs[index][choice] = values[choice]
    if fallback_counts:
        print(
            json.dumps(
                {
                    "event": "vllm_choice_logprob_fallback",
                    "missing_choice_counts": fallback_counts,
                    "method": "single_choice_forcing",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows: list[dict[str, Any]] = []
    for (group, query), values in zip(selected, choice_logprobs, strict=True):
        missing = sorted(set(choice_ids) - set(values))
        if missing:
            raise RuntimeError(f"vLLM choice logprobs remain incomplete: {missing}")
        logp_a, logp_b = values["A"], values["B"]
        score = normalized_applied_score(logp_a, logp_b)
        row = {
            "query_id": query.query_id,
            "trajectory_id": query.trajectory_id,
            "score": score,
            "logit": logp_b - logp_a,
            "choice_logprobs": {"A_not_applied": logp_a, "B_applied": logp_b},
            "prediction_label": "applied" if score >= 0.5 else "not_applied",
            "family": query.family,
            "factor_level": query.factor_level,
            "evidence_route": (
                "lora_choice_sft_factor_suffix_prompt"
                if lora_request is not None
                else "zero_shot_factor_suffix_prompt"
            ),
            "negative_type": query.negative_type,
            "factor_intervention": query.intervention,
            "source_query_id": query.source_query_id or query.query_id,
            "donor_query_id": query.donor_query_id or query.query_id,
        }
        if query.factor_scope is not None:
            row["factor_scope"] = query.factor_scope
        if query.label is not None:
            row["label_id"] = query.label
        rows.append(row)
    return rows, maximum


def _postprocess(
    benchmark: Mapping[str, Any],
    effective: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output = Path(benchmark["output_directory"])
    threshold = float(effective["evaluation"]["threshold_value"])
    prediction_rows = [
        {"query_id": row["query_id"], "prediction_label": row["prediction_label"]}
        for row in rows
    ]
    write_jsonl(output / "predictions.jsonl", prediction_rows)
    labelled = all("label_id" in row for row in rows)
    if labelled:
        summary = _evaluation_summary(
            list(rows),
            threshold,
            effective["evaluation"].get("reporting_dimensions"),
        )
        summary["oracle_gmean2"] = best_threshold_metrics(rows, metric="gmean2")
    else:
        summary = _prediction_only_summary(list(rows), threshold)
    summary.update(
        {
            "backend": BACKEND_NAME,
            "choice_contract_id": CHOICE_CONTRACT_ID,
            "metric_adapter": benchmark["metric_adapter"],
        }
    )
    write_json(output / "metrics.json", summary)

    adapter = benchmark["metric_adapter"]
    if adapter == "openai_process":
        scored = score_process_artifact(
            artifact=Path(benchmark["dataset"]["directory"]),
            predictions=output / _FINAL_FILE,
            monitor_effective_config=output / "effective_config.json",
            output_directory=output.parent / "scored",
            overwrite=False,
        )
        return {"inference": summary, "formal_scored": scored}
    if adapter == "openai_intervention":
        scored = score_artifact(
            artifact=Path(benchmark["dataset"]["directory"]),
            predictions=output / "predictions.jsonl",
            monitor_effective_config=output / "effective_config.json",
            output_directory=output.parent / "scored",
            overwrite=False,
        )
        return {"inference": summary, "formal_scored": scored}
    return {"inference": summary}


def _load_vllm(
    effective: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Any, dict[str, int], Any | None]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ModuleNotFoundError as exc:
        raise RuntimeError("vLLM baseline evaluation requires transformers and vllm") from exc

    model = effective["model"]
    for field in ("checkpoint", "tokenizer"):
        path = Path(model[field])
        if not path.is_dir():
            raise RuntimeError(f"worker cannot access model {field}: {path}")
    adapter = model.get("adapter")
    lora_request = None
    if adapter is not None:
        adapter_path = Path(adapter["path"])
        required_adapter_files = (
            adapter_path / "adapter_config.json",
            adapter_path / "adapter_model.safetensors",
        )
        if not all(path.is_file() for path in required_adapter_files):
            raise RuntimeError(f"worker cannot access complete LoRA adapter: {adapter_path}")
        lora_request = LoRARequest(
            adapter["name"], int(adapter["id"]), str(adapter_path)
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model["tokenizer"],
        local_files_only=True,
        trust_remote_code=bool(model["trust_remote_code"]),
        use_fast=True,
    )
    choice_ids = _choice_token_ids(tokenizer)
    runtime = effective["runtime"]
    llm = LLM(
        model=model["checkpoint"],
        tokenizer=model["tokenizer"],
        dtype=model["dtype"],
        trust_remote_code=bool(model["trust_remote_code"]),
        tensor_parallel_size=int(runtime["tensor_parallel_size"]),
        max_model_len=int(runtime["max_sequence_length"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        enable_prefix_caching=bool(runtime["enable_prefix_caching"]),
        enforce_eager=bool(runtime.get("enforce_eager", False)),
        max_num_seqs=int(runtime["max_num_seqs"]),
        seed=int(runtime["seed"]),
        enable_lora=adapter is not None,
        **(
            {"max_lora_rank": int(adapter["rank"])}
            if adapter is not None
            else {}
        ),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2,
        allowed_token_ids=[choice_ids["A"], choice_ids["B"]],
        detokenize=False,
    )
    fallback_sampling = {
        choice: SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=0,
            allowed_token_ids=[token_id],
            detokenize=False,
        )
        for choice, token_id in choice_ids.items()
    }
    natural_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=0,
        detokenize=True,
    )
    return (
        llm,
        sampling,
        fallback_sampling,
        natural_sampling,
        tokenizer,
        choice_ids,
        lora_request,
    )


def _natural_label_audit(
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    prepared: Sequence[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Sequence[tuple[TrajectoryGroup, FactorQueryRecord]],
        ]
    ],
    *,
    choice_ids: Mapping[str, int],
    instruction: str,
    chat_kwargs: Mapping[str, Any],
    max_sequence_length: int,
    requested_queries: int,
    lora_request: Any | None,
    output_root: Path,
) -> dict[str, Any] | None:
    """Audit natural one-token A/B compliance without constraining generation."""

    if requested_queries <= 0:
        return None
    selected: list[tuple[str, TrajectoryGroup, FactorQueryRecord]] = []
    cursors = [0] * len(prepared)
    while len(selected) < requested_queries:
        advanced = False
        for index, (benchmark, _, examples) in enumerate(prepared):
            cursor = cursors[index]
            if cursor >= len(examples):
                continue
            group, query = examples[cursor]
            cursors[index] += 1
            selected.append((str(benchmark["id"]), group, query))
            advanced = True
            if len(selected) == requested_queries:
                break
        if not advanced:
            break
    prompts: list[dict[str, list[int]]] = []
    for _, group, query in selected:
        ids = _prompt_token_ids(
            tokenizer,
            _prompt_text(group, query, instruction),
            chat_kwargs,
        )
        if len(ids) + 1 > max_sequence_length:
            raise RuntimeError(
                f"natural audit query {query.query_id} exceeds max_sequence_length"
            )
        prompts.append({"prompt_token_ids": ids})
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    if len(outputs) != len(selected):
        raise RuntimeError("natural label audit output count differs from request count")
    choice_by_id = {token_id: choice for choice, token_id in choice_ids.items()}
    rows: list[dict[str, Any]] = []
    for (benchmark_id, _, query), output in zip(selected, outputs, strict=True):
        candidates = getattr(output, "outputs", None)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise RuntimeError("natural label audit requires exactly one completion")
        candidate = candidates[0]
        token_ids = getattr(candidate, "token_ids", None)
        token_id = (
            int(token_ids[0])
            if isinstance(token_ids, (list, tuple)) and len(token_ids) == 1
            else None
        )
        choice = choice_by_id.get(token_id)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "query_id": query.query_id,
                "token_id": token_id,
                "text": str(getattr(candidate, "text", "")),
                "choice": choice,
                "compliant": choice is not None,
            }
        )
    write_jsonl(output_root / "natural_label_audit.jsonl", rows)
    summary = {
        "schema_version": 1,
        "requested_queries": requested_queries,
        "audited_queries": len(rows),
        "compliant_queries": sum(bool(row["compliant"]) for row in rows),
        "compliance_rate": (
            sum(bool(row["compliant"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "generation": "one_token_unconstrained_greedy",
    }
    write_json(output_root / "natural_label_audit_summary.json", summary)
    return summary


def run_suite(config_path: Path, *, dry_run: bool) -> dict[str, Any]:
    effective = resolve_suite_config(config_path)
    if dry_run:
        return {
            "status": "dry_run",
            "effective_config": effective,
            "model_access_note": "checkpoint/tokenizer existence is checked inside the rjob",
        }

    output_root = Path(effective["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    suite_effective_path = output_root / "suite_effective_config.json"
    if suite_effective_path.is_file():
        existing = load_config(suite_effective_path)
        if existing.get("suite_identity") != effective["suite_identity"]:
            raise RuntimeError(f"suite output belongs to a different effective config: {output_root}")
    write_json(suite_effective_path, effective)

    taxonomy = Taxonomy.load(Path(effective["taxonomy"]["path"]))
    prepared: list[tuple[dict[str, Any], dict[str, Any], list[tuple[TrajectoryGroup, FactorQueryRecord]]]] = []
    for benchmark in effective["benchmarks"]:
        dataset = benchmark["dataset"]
        allow_unlabelled = dataset["source_metric"] == "prediction_only"
        groups = load_trajectory_groups(
            Path(dataset["files"]["trajectory_file"]["path"]),
            Path(dataset["files"]["query_file"]["path"]),
            taxonomy=taxonomy,
            allow_unlabelled=allow_unlabelled,
        )
        families = {group.family for group in groups}
        if families != {benchmark["family"]}:
            raise RuntimeError(
                f"benchmark {benchmark['id']} expected family {benchmark['family']!r}, "
                f"found {sorted(families)}"
            )
        benchmark_effective = _benchmark_effective(effective, benchmark)
        examples = _query_examples(groups)
        prepared.append((benchmark, benchmark_effective, examples))

    (
        llm,
        sampling,
        fallback_sampling,
        natural_sampling,
        tokenizer,
        choice_ids,
        lora_request,
    ) = _load_vllm(effective)
    instruction = effective["monitor"]["choice_contract"]["instruction"]
    chat_kwargs = effective["monitor"]["chat_template_kwargs"]
    suite_results: dict[str, Any] = {}
    for benchmark, benchmark_effective, examples in prepared:
        output = Path(benchmark["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        effective_path = output / "effective_config.json"
        if effective_path.is_file():
            existing = load_config(effective_path)
            if _resume_identity(existing) != _resume_identity(benchmark_effective):
                raise RuntimeError(f"benchmark output effective config mismatch: {output}")
        write_json(effective_path, benchmark_effective)
        rows, completed, complete = _load_resume(
            output, benchmark_effective, examples
        )
        if not complete:
            partial = output / _PARTIAL_FILE
            if completed == 0:
                partial.write_text("", encoding="utf-8")
                os.chmod(partial, 0o664)
                _write_progress(
                    output,
                    benchmark_effective,
                    completed_queries=0,
                    total_queries=len(examples),
                    status="running",
                )
            chunk_size = int(effective["runtime"]["request_chunk_size"])
            for start in range(completed, len(examples), chunk_size):
                selected = examples[start : start + chunk_size]
                batch_rows, batch_max_prompt_tokens = _score_rows(
                    llm,
                    sampling,
                    fallback_sampling,
                    tokenizer,
                    selected,
                    choice_ids=choice_ids,
                    instruction=instruction,
                    chat_kwargs=chat_kwargs,
                    max_sequence_length=int(effective["runtime"]["max_sequence_length"]),
                    lora_request=lora_request,
                )
                _append_rows(partial, batch_rows)
                rows.extend(batch_rows)
                completed = len(rows)
                _update_length_audit(
                    output,
                    completed_queries=completed,
                    total_queries=len(examples),
                    batch_max_prompt_tokens=batch_max_prompt_tokens,
                    max_sequence_length=int(
                        effective["runtime"]["max_sequence_length"]
                    ),
                )
                _write_progress(
                    output,
                    benchmark_effective,
                    completed_queries=completed,
                    total_queries=len(examples),
                    status="running",
                )
                print(
                    json.dumps(
                        {
                            "event": "vllm_baseline_progress",
                            "benchmark": benchmark["id"],
                            "completed_queries": completed,
                            "total_queries": len(examples),
                            "percent": round(100.0 * completed / len(examples), 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            os.replace(partial, output / _FINAL_FILE)
            _write_progress(
                output,
                benchmark_effective,
                completed_queries=len(rows),
                total_queries=len(examples),
                status="completed",
            )
        metrics_path = output / "metrics.json"
        scored_path = output.parent / "scored" / "summary.json"
        needs_formal = benchmark["metric_adapter"] != "binary"
        if not metrics_path.is_file() or (needs_formal and not scored_path.is_file()):
            suite_results[benchmark["id"]] = _postprocess(
                benchmark, benchmark_effective, rows
            )
        else:
            suite_results[benchmark["id"]] = {
                "inference": load_config(metrics_path),
                **(
                    {"formal_scored": load_config(scored_path)}
                    if needs_formal
                    else {}
                ),
            }
        write_json(
            output_root / "suite_progress.json",
            {
                "schema_version": 1,
                "status": "running",
                "completed_benchmarks": list(suite_results),
                "total_benchmarks": len(effective["benchmarks"]),
            },
        )
    natural_label_audit = _natural_label_audit(
        llm,
        natural_sampling,
        tokenizer,
        prepared,
        choice_ids=choice_ids,
        instruction=instruction,
        chat_kwargs=chat_kwargs,
        max_sequence_length=int(effective["runtime"]["max_sequence_length"]),
        requested_queries=int(
            effective["runtime"].get("natural_label_audit_queries", 0)
        ),
        lora_request=lora_request,
        output_root=output_root,
    )
    summary = {
        "schema_version": 1,
        "kind": "vllm_forced_choice_eval_suite_summary",
        "status": "completed",
        "suite_identity": effective["suite_identity"],
        "model": effective["model"],
        "monitor": effective["monitor"],
        "evaluation": effective["evaluation"],
        "benchmarks": suite_results,
        "natural_label_audit": natural_label_audit,
    }
    write_json(output_root / "summary.json", summary)
    write_json(
        output_root / "suite_progress.json",
        {
            "schema_version": 1,
            "status": "completed",
            "completed_benchmarks": list(suite_results),
            "total_benchmarks": len(effective["benchmarks"]),
        },
    )
    return summary
