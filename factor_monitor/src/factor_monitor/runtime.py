"""Backend-neutral training/evaluation run contracts and effective configs."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    repository_state,
    reject_unknown,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .io import write_json
from .prompt_contract import CONTRACT_ID, contract_hash
from .training.config import BACKEND_NAME, validate_factor_monitor_effective_config


RunBackend = Callable[[dict[str, Any]], dict[str, Any]]
_TRAIN_BACKENDS: dict[str, RunBackend] = {}
_EVAL_BACKENDS: dict[str, RunBackend] = {}


def is_world_process_zero() -> bool:
    """Return whether this process owns shared run-directory writes."""
    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ConfigurationError("RANK and WORLD_SIZE must be integers") from exc
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ConfigurationError(
            f"invalid distributed process coordinates: RANK={rank}, WORLD_SIZE={world_size}"
        )
    return rank == 0


def register_training_backend(name: str, backend: RunBackend) -> None:
    if not name or name in _TRAIN_BACKENDS:
        raise ValueError(f"invalid or duplicate training backend: {name!r}")
    _TRAIN_BACKENDS[name] = backend


def register_evaluation_backend(name: str, backend: RunBackend) -> None:
    if not name or name in _EVAL_BACKENDS:
        raise ValueError(f"invalid or duplicate evaluation backend: {name!r}")
    _EVAL_BACKENDS[name] = backend


def _reusable_train_output_shell(
    output_dir: Path, effective: dict[str, Any]
) -> bool:
    """Return whether an existing train output contains provably no run state."""

    if output_dir.is_symlink() or not output_dir.is_dir():
        return False
    entries = tuple(output_dir.iterdir())
    if not entries:
        return True
    if len(entries) != 1 or entries[0].name != "effective_config.json":
        return False
    try:
        existing = load_config(entries[0])
    except Exception:
        return False
    expected_files = effective.get("dataset", {}).get("files")
    existing_files = existing.get("dataset", {}).get("files")
    return (
        existing.get("kind") == "train_run"
        and isinstance(effective.get("config_hash"), str)
        and existing.get("config_hash") == effective["config_hash"]
        and isinstance(expected_files, dict)
        and existing_files == expected_files
        and existing.get("output_directory") == effective.get("output_directory")
    )


def _validate_train_output_preflight(effective: dict[str, Any]) -> None:
    """Fail before GPU submission unless a train output is new, resumable, or empty."""

    if effective["kind"] != "train_run":
        return
    output_dir = Path(effective["output_directory"])
    resume = effective["optimization"].get("resume_from_checkpoint")
    if resume is not None:
        if Path(resume).parent.resolve() != output_dir.resolve():
            raise ConfigurationError(
                "resume checkpoint must be an immediate checkpoint directory inside output_directory"
            )
        return
    if not output_dir.exists() or _reusable_train_output_shell(output_dir, effective):
        return
    entries = sorted(path.name for path in output_dir.iterdir()) if output_dir.is_dir() else []
    raise ConfigurationError(
        "training output preflight refused existing preserved or mismatched state: "
        f"{output_dir}; entries={entries}. Move/delete it or configure an explicit "
        "checkpoint resume before requesting GPUs. Automatic reuse is limited to an "
        "empty directory or a matching effective_config.json-only shell."
    )


def _validate_eval_output_preflight(effective: dict[str, Any]) -> None:
    """Reject completed or structurally unsafe eval outputs before requesting GPUs."""

    if effective["kind"] != "eval_run":
        return
    output_dir = Path(effective["output_directory"])
    if not output_dir.exists():
        return
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ConfigurationError(f"evaluation output is not a safe directory: {output_dir}")
    entries = {path.name for path in output_dir.iterdir()}
    if not entries:
        return
    progress_path = output_dir / "progress.json"
    effective_path = output_dir / "effective_config.json"
    if effective_path.is_file():
        try:
            existing = load_config(effective_path)
        except Exception as exc:
            raise ConfigurationError(
                f"cannot validate existing evaluation effective config: {effective_path}"
            ) from exc
        if (
            existing.get("config_hash") != effective.get("config_hash")
            or existing.get("dataset", {}).get("files")
            != effective.get("dataset", {}).get("files")
            or existing.get("output_directory") != effective.get("output_directory")
        ):
            raise ConfigurationError(
                "evaluation output belongs to a different config or frozen dataset: "
                f"{output_dir}"
            )
    if progress_path.is_file():
        try:
            progress = load_config(progress_path)
        except Exception as exc:
            raise ConfigurationError(
                f"cannot validate evaluation progress: {progress_path}"
            ) from exc
        if progress.get("status") == "completed":
            raise ConfigurationError(
                f"evaluation output is already completed and will not be overwritten: {output_dir}"
            )
        if progress.get("status") == "running" and (
            output_dir / "scores.partial.jsonl"
        ).is_file() and effective_path.is_file():
            return
    if entries == {"effective_config.json"} and effective_path.is_file():
        return
    raise ConfigurationError(
        "evaluation output preflight found incomplete or unowned state: "
        f"{output_dir}; entries={sorted(entries)}. Preserve/inspect it before resubmission."
    )


def _validate_output_preflight(effective: dict[str, Any]) -> None:
    _validate_train_output_preflight(effective)
    _validate_eval_output_preflight(effective)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _existing_file(path_value: Any, *, base: Path, name: str) -> Path:
    path = resolve_path(path_value, relative_to=base)
    if not path.is_file():
        raise ConfigurationError(f"{name} does not exist: {path}")
    return path


def _common(config: dict[str, Any], *, config_path: Path, kind: str) -> dict[str, Any]:
    require_keys(config, {"dataset", "model", "monitor", "runtime", "evaluation", "output"}, "run config")
    base = config_path.parent
    dataset = require_mapping(config["dataset"], "dataset")
    model = require_mapping(config["model"], "model")
    monitor = require_mapping(config["monitor"], "monitor")
    runtime = require_mapping(config["runtime"], "runtime")
    evaluation = require_mapping(config["evaluation"], "evaluation")
    output = require_mapping(config["output"], "output")
    require_keys(model, {"checkpoint", "tokenizer"}, "model")
    require_keys(monitor, {"backend"}, "monitor")
    require_keys(
        runtime,
        {"max_sequence_length", "truncation_policy", "padding_policy", "seed"},
        "runtime",
    )
    require_keys(evaluation, {"metric"}, "evaluation")
    require_keys(output, {"directory"}, "output")
    model_fields = {"checkpoint", "tokenizer", "load_options"}
    reject_unknown(monitor, {"backend", "prompt_contract_id", "backend_options"}, "monitor")
    runtime_fields = {"max_sequence_length", "truncation_policy", "padding_policy", "seed"}
    if kind == "train_run":
        runtime_fields.add("expected_world_size")
    if kind == "eval_run":
        runtime_fields.add("batch_size")
    reject_unknown(runtime, runtime_fields, "runtime")
    reject_unknown(output, {"directory"}, "output")
    backend = _nonempty(monitor["backend"], "monitor.backend")
    if backend == BACKEND_NAME:
        require_keys(monitor, {"backend_options"}, "monitor")
        require_keys(model, {"checkpoint_revision", "tokenizer_revision"}, "model")
        model_fields.update({"checkpoint_revision", "tokenizer_revision"})
        if "prompt_contract_id" in monitor:
            raise ConfigurationError(
                "FactorMonitor dual raw encoding must not set legacy monitor.prompt_contract_id"
            )
    else:
        require_keys(monitor, {"prompt_contract_id"}, "monitor")
        if monitor["prompt_contract_id"] != CONTRACT_ID:
            raise ConfigurationError(
                f"monitor.prompt_contract_id must be {CONTRACT_ID!r}, got "
                f"{monitor['prompt_contract_id']!r}"
            )
    reject_unknown(model, model_fields, "model")
    if runtime["truncation_policy"] != "error":
        raise ConfigurationError("runtime.truncation_policy must be error for complete-CoT runs")
    return {
        "base": base,
        "dataset": dataset,
        "model": {
            **model,
            "checkpoint": _nonempty(model["checkpoint"], "model.checkpoint"),
            "tokenizer": _nonempty(model["tokenizer"], "model.tokenizer"),
        },
        "monitor": {
            **monitor,
            "backend": backend,
        },
        "runtime": {
            **runtime,
            "max_sequence_length": _integer(
                runtime["max_sequence_length"], "runtime.max_sequence_length", positive=True
            ),
            "padding_policy": _nonempty(runtime["padding_policy"], "runtime.padding_policy"),
            "seed": _integer(runtime["seed"], "runtime.seed"),
            **(
                {
                    "expected_world_size": _integer(
                        runtime["expected_world_size"],
                        "runtime.expected_world_size",
                        positive=True,
                    )
                }
                if "expected_world_size" in runtime
                else {}
            ),
        },
        "evaluation": {**evaluation, "metric": _nonempty(evaluation["metric"], "evaluation.metric")},
        "output_directory": resolve_path(output["directory"], relative_to=base),
    }


def _apply_validation_only_override(
    effective: dict[str, Any],
    *,
    validation_checkpoint: Path | None,
    output_directory: Path | None,
    mil_short_validation_auroc_floor: float | None = None,
) -> dict[str, Any]:
    """Bind one immutable training checkpoint to an independent validation run."""

    if (validation_checkpoint is None) != (output_directory is None):
        raise ConfigurationError(
            "--validation-checkpoint and --output-directory must be provided together"
        )
    if mil_short_validation_auroc_floor is not None and validation_checkpoint is None:
        raise ConfigurationError(
            "--mil-short-validation-auroc-floor requires --validation-checkpoint"
        )
    if validation_checkpoint is None:
        return effective
    if effective["kind"] != "train_run" or effective["monitor"]["backend"] != BACKEND_NAME:
        raise ConfigurationError(
            "validation-only checkpoint overrides require a FactorMonitor train config"
        )
    if effective["optimization"].get("resume_from_checkpoint") is not None:
        raise ConfigurationError(
            "validation-only checkpoint override cannot be combined with training resume"
        )
    checkpoint = validation_checkpoint.resolve()
    state_path = checkpoint / "trainer_state.json"
    bundle = checkpoint / "model_bundle"
    bundle_effective_path = bundle / "effective_config.json"
    factor_weights = bundle / "factor_monitor.safetensors"
    adapter_weights = bundle / "backbone" / "adapter_model.safetensors"
    required = (state_path, bundle_effective_path, factor_weights, adapter_weights)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ConfigurationError(
            f"validation checkpoint is incomplete; missing={missing}"
        )
    state = load_config(state_path)
    global_step = state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ConfigurationError(
            "validation checkpoint Trainer state lacks a positive global_step"
        )
    if checkpoint.name != f"checkpoint-{global_step}":
        raise ConfigurationError(
            "validation checkpoint directory name does not match Trainer global_step"
        )
    source_config_hash = effective["config_hash"]
    bundle_effective = load_config(bundle_effective_path)
    if bundle_effective.get("config_hash") != source_config_hash:
        raise ConfigurationError(
            "validation checkpoint was not produced by the selected training config"
        )

    if mil_short_validation_auroc_floor is not None:
        try:
            floor = float(mil_short_validation_auroc_floor)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "--mil-short-validation-auroc-floor must be a finite number in (0,1]"
            ) from exc
        if not 0.0 < floor <= 1.0:
            raise ConfigurationError(
                "--mil-short-validation-auroc-floor must be a finite number in (0,1]"
            )
        backend_options = effective["monitor"].get("backend_options")
        if not isinstance(backend_options, dict) or not isinstance(
            backend_options.get("long_cot_mil"), dict
        ):
            raise ConfigurationError(
                "--mil-short-validation-auroc-floor requires monitor.backend_options.long_cot_mil"
            )
        backend_options["long_cot_mil"] = {
            **backend_options["long_cot_mil"],
            "short_validation_auroc_floor": floor,
        }

    output = output_directory.resolve()
    if output == checkpoint.parent or output == checkpoint or checkpoint in output.parents:
        raise ConfigurationError(
            "validation-only output must be independent of the source training checkpoint"
        )
    validation_only = {
        "checkpoint": str(checkpoint),
        "global_step": global_step,
        "source_config_hash": source_config_hash,
        "trainer_state_sha256": file_sha256(state_path),
        "factor_monitor_sha256": file_sha256(factor_weights),
        "adapter_model_sha256": file_sha256(adapter_weights),
        "source_policy": "immutable",
        "training_steps": 0,
    }
    effective["runtime"]["expected_world_size"] = 1
    effective["evaluation"]["during_training"] = False
    effective["evaluation"]["validation_only"] = validation_only
    effective["output_directory"] = str(output)
    effective["config_hash"] = stable_hash(
        {
            "source_config_hash": source_config_hash,
            "validation_only": validation_only,
            "output_directory": str(output),
        }
    )
    return effective


def _apply_evaluation_override(
    effective: dict[str, Any],
    *,
    model_bundle: Path | None,
    threshold_source: Path | None,
    output_directory: Path | None,
    architecture: Path | None,
    native_checkpoint_threshold: float | None,
    long_cot_aggregation: str | None,
    direct_reference_scores: Path | None,
) -> dict[str, Any]:
    """Bind one saved model and its frozen threshold to an existing eval recipe."""

    core = (model_bundle, threshold_source, output_directory)
    if any(value is not None for value in core) and not all(
        value is not None for value in core
    ):
        raise ConfigurationError(
            "--model-bundle, --threshold-source, and --output-directory must be "
            "provided together for evaluation overrides"
        )
    window = (long_cot_aggregation, direct_reference_scores)
    if any(value is not None for value in window) and not all(
        value is not None for value in window
    ):
        raise ConfigurationError(
            "--long-cot-aggregation and --direct-reference-scores must be "
            "provided together"
        )
    if model_bundle is None:
        if any(value is not None for value in window):
            raise ConfigurationError(
                "long-CoT evaluation overrides require the model/threshold/output group"
            )
        return effective
    if effective["kind"] != "eval_run" or effective["monitor"]["backend"] != BACKEND_NAME:
        raise ConfigurationError(
            "evaluation overrides require a FactorMonitor eval config"
        )

    bundle = model_bundle.resolve()
    factor_weights = bundle / "factor_monitor.safetensors"
    adapter_weights = bundle / "backbone" / "adapter_model.safetensors"
    bundle_effective = bundle / "effective_config.json"
    required = (factor_weights, adapter_weights, bundle_effective)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ConfigurationError(f"evaluation model bundle is incomplete; missing={missing}")

    threshold_path = threshold_source.resolve()
    if not threshold_path.is_file():
        raise ConfigurationError(
            f"evaluation threshold source does not exist: {threshold_path}"
        )
    threshold_payload = load_config(threshold_path)
    checkpoint_reference = threshold_payload.get("best_model_checkpoint")
    if not isinstance(checkpoint_reference, str) or not checkpoint_reference.strip():
        raise ConfigurationError(
            "evaluation threshold source lacks best_model_checkpoint"
        )
    checkpoint = Path(checkpoint_reference)
    if not checkpoint.is_absolute():
        checkpoint = threshold_path.parent / checkpoint
    checkpoint = checkpoint.resolve()
    checkpoint_bundle = checkpoint / "model_bundle"
    checkpoint_factor = checkpoint_bundle / "factor_monitor.safetensors"
    checkpoint_adapter = checkpoint_bundle / "backbone" / "adapter_model.safetensors"
    if not checkpoint_factor.is_file() or not checkpoint_adapter.is_file():
        raise ConfigurationError(
            "evaluation threshold checkpoint lacks a complete model bundle: "
            f"{checkpoint}"
        )
    factor_sha = file_sha256(factor_weights)
    adapter_sha = file_sha256(adapter_weights)
    if factor_sha != file_sha256(checkpoint_factor) or adapter_sha != file_sha256(
        checkpoint_adapter
    ):
        raise ConfigurationError(
            "evaluation model bundle weights do not match the frozen-threshold checkpoint"
        )

    options = effective["monitor"]["backend_options"]
    windowing = options.get("evaluation_scope", {}).get("long_cot_windowing")
    if long_cot_aggregation is not None:
        if not isinstance(windowing, dict):
            raise ConfigurationError(
                "--long-cot-aggregation requires a long-CoT windowing eval config"
            )
        expected_pooling = {
            "top2_mean": "top2_probability_mean",
            "top2_logmeanexp": "top2_logmeanexp",
        }.get(long_cot_aggregation)
        if expected_pooling is None:
            raise ConfigurationError(
                "--long-cot-aggregation must be top2_mean or top2_logmeanexp"
            )
        if threshold_payload.get("pooling") != expected_pooling:
            raise ConfigurationError(
                "long-CoT aggregation does not match the frozen threshold pooling"
            )
        reference = direct_reference_scores.resolve()
        if not reference.is_file():
            raise ConfigurationError(
                f"direct reference scores do not exist: {reference}"
            )
        reference_effective_path = reference.parent / "effective_config.json"
        if not reference_effective_path.is_file():
            raise ConfigurationError(
                "direct reference scores lack their effective config: "
                f"{reference_effective_path}"
            )
        reference_effective = load_config(reference_effective_path)
        reference_binding = reference_effective.get("evaluation", {}).get(
            "override_binding", {}
        )
        if (
            reference_binding.get("factor_monitor_sha256") != factor_sha
            or reference_binding.get("adapter_model_sha256") != adapter_sha
        ):
            raise ConfigurationError(
                "direct reference scores were not produced by the selected model bundle"
            )
        windowing["aggregations"] = [long_cot_aggregation]
        windowing["direct_reference_scores"] = str(reference)
    elif windowing is not None and windowing.get("comparison_mode") != "standalone":
        raise ConfigurationError(
            "overridden long-CoT evals must explicitly provide their aggregation and "
            "matching direct-reference scores"
        )

    source_config_hash = effective["config_hash"]
    output = output_directory.resolve()
    if architecture is not None:
        from .training.config import load_architecture

        architecture_path = architecture.resolve()
        if not architecture_path.is_file():
            raise ConfigurationError(
                f"evaluation architecture override does not exist: {architecture_path}"
            )
        parsed_architecture = load_architecture(architecture_path)
        effective["monitor"]["backend_options"]["architecture_config"] = str(
            architecture_path
        )
        effective["monitor"]["backend_options"]["evidence_routes"] = dict(
            parsed_architecture.family_routes
        )
    effective["monitor"]["backend_options"]["model_bundle"] = str(bundle)
    effective["evaluation"]["threshold_source"] = str(threshold_path)
    effective["evaluation"]["threshold_source_sha256"] = file_sha256(threshold_path)
    if native_checkpoint_threshold is not None:
        if not 0.0 <= float(native_checkpoint_threshold) <= 1.0:
            raise ConfigurationError(
                "--native-checkpoint-threshold must be in [0,1]"
            )
        scope = effective["monitor"]["backend_options"].get("evaluation_scope")
        if not isinstance(scope, dict) or scope.get("mode") != "diagnostic":
            raise ConfigurationError(
                "--native-checkpoint-threshold requires diagnostic evaluation"
            )
        scope["native_checkpoint_threshold"] = float(native_checkpoint_threshold)
    effective["evaluation"]["override_binding"] = {
        "checkpoint": str(checkpoint),
        "factor_monitor_sha256": factor_sha,
        "adapter_model_sha256": adapter_sha,
        "threshold_source_sha256": file_sha256(threshold_path),
        **(
            {"architecture": str(architecture.resolve())}
            if architecture is not None
            else {}
        ),
            **(
                {"native_checkpoint_threshold": float(native_checkpoint_threshold)}
                if native_checkpoint_threshold is not None
                else {}
            ),
        **(
            {"long_cot_aggregation": long_cot_aggregation}
            if long_cot_aggregation is not None
            else {}
        ),
    }
    effective["output_directory"] = str(output)
    effective["config_hash"] = stable_hash(
        {
            "source_config_hash": source_config_hash,
            "model_bundle": str(bundle),
            "threshold_source": str(threshold_path),
            "output_directory": str(output),
            "architecture": (
                str(architecture.resolve()) if architecture is not None else None
            ),
            "native_checkpoint_threshold": (
                float(native_checkpoint_threshold)
                if native_checkpoint_threshold is not None
                else None
            ),
            "long_cot_aggregation": long_cot_aggregation,
            "direct_reference_scores": (
                str(direct_reference_scores.resolve())
                if direct_reference_scores is not None
                else None
            ),
        }
    )
    return effective


def resolve_run_config(
    config_path: Path,
    *,
    validation_checkpoint: Path | None = None,
    output_directory: Path | None = None,
    model_bundle: Path | None = None,
    threshold_source: Path | None = None,
    architecture: Path | None = None,
    long_cot_aggregation: str | None = None,
    direct_reference_scores: Path | None = None,
    native_checkpoint_threshold: float | None = None,
    mil_short_validation_auroc_floor: float | None = None,
) -> dict[str, Any]:
    evaluation_override = {
        "model_bundle": model_bundle,
        "threshold_source": threshold_source,
        "architecture": architecture,
        "native_checkpoint_threshold": native_checkpoint_threshold,
        "long_cot_aggregation": long_cot_aggregation,
        "direct_reference_scores": direct_reference_scores,
    }
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("schema_version") != 1:
        raise ConfigurationError("run config schema_version must be 1")
    kind = config.get("kind")
    if kind not in {"train_run", "eval_run"}:
        raise ConfigurationError("run config kind must be train_run or eval_run")
    allowed_top = {
        "schema_version",
        "kind",
        "dataset",
        "model",
        "monitor",
        "runtime",
        "evaluation",
        "output",
    }
    if kind == "train_run":
        allowed_top.update({"optimization", "initialization"})
    reject_unknown(config, allowed_top, "run config")
    common = _common(config, config_path=config_path, kind=kind)
    dataset = common["dataset"]
    base = common["base"]
    require_keys(dataset, {"directory", "trajectory_file"}, "dataset")
    dataset_dir = resolve_path(dataset["directory"], relative_to=base)
    trajectory_file = _existing_file(
        dataset["trajectory_file"], base=dataset_dir, name="dataset.trajectory_file"
    )
    dataset_files: dict[str, Path] = {"trajectory_file": trajectory_file}

    optimization: dict[str, Any]
    initialization: dict[str, Any] | None = None
    if kind == "train_run":
        require_keys(dataset, {"train_file", "validation_file"}, "dataset")
        factor_backend = common["monitor"]["backend"] == BACKEND_NAME
        if factor_backend:
            require_keys(dataset, {"manifest", "holdout_file"}, "dataset")
        reject_unknown(
            dataset,
            {
                "directory",
                "trajectory_file",
                "train_file",
                "validation_file",
                *(("manifest", "holdout_file") if factor_backend else ()),
            },
            "dataset",
        )
        dataset_files.update(
            {
                "train_file": _existing_file(dataset["train_file"], base=dataset_dir, name="dataset.train_file"),
                "validation_file": _existing_file(
                    dataset["validation_file"], base=dataset_dir, name="dataset.validation_file"
                ),
            }
        )
        if factor_backend:
            dataset_files.update(
                {
                    "manifest": _existing_file(
                        dataset["manifest"], base=dataset_dir, name="dataset.manifest"
                    ),
                    "holdout_file": _existing_file(
                        dataset["holdout_file"], base=dataset_dir, name="dataset.holdout_file"
                    ),
                }
            )
        optimization_value = require_mapping(config.get("optimization"), "optimization")
        initialization_value = config.get("initialization")
        if initialization_value is not None:
            initialization_block = require_mapping(
                initialization_value, "initialization"
            )
            initialization_fields = {"model_bundle", "policy"}
            require_keys(initialization_block, initialization_fields, "initialization")
            reject_unknown(initialization_block, initialization_fields, "initialization")
            if initialization_block["policy"] != "weights_only_fresh_optimizer":
                raise ConfigurationError(
                    "initialization.policy must be weights_only_fresh_optimizer"
                )
            model_bundle = resolve_path(
                initialization_block["model_bundle"], relative_to=base
            )
            required_bundle_files = (
                model_bundle / "factor_monitor.safetensors",
                model_bundle / "effective_config.json",
                model_bundle / "architecture_config.json",
                model_bundle / "backbone",
            )
            if not all(path.exists() for path in required_bundle_files):
                raise ConfigurationError(
                    f"initialization model bundle is incomplete: {model_bundle}"
                )
            initialization = {
                "model_bundle": str(model_bundle),
                "policy": "weights_only_fresh_optimizer",
                "factor_monitor_sha256": file_sha256(
                    model_bundle / "factor_monitor.safetensors"
                ),
                "effective_config_sha256": file_sha256(
                    model_bundle / "effective_config.json"
                ),
            }
        required = {
            "strategy",
            "objective",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "epochs",
            "max_steps",
            "optimizer",
            "scheduler",
            "checkpoint_policy",
        }
        if factor_backend:
            required.update(
                {
                    "weight_decay",
                    "warmup_ratio",
                    "max_grad_norm",
                    "mixed_precision",
                    "logging_steps",
                    "save_steps",
                    "resume_from_checkpoint",
                    "gradient_checkpointing",
                    "gradient_checkpointing_use_reentrant",
                }
            )
        optimization_fields = set(required)
        if factor_backend:
            optimization_fields.add("save_total_limit")
        require_keys(optimization_value, required, "optimization")
        reject_unknown(optimization_value, optimization_fields, "optimization")
        epochs = optimization_value["epochs"]
        max_steps = optimization_value["max_steps"]
        if (epochs is None) == (max_steps is None):
            raise ConfigurationError("exactly one of optimization.epochs or max_steps must be set")
        optimization = {
            **optimization_value,
            "strategy": _nonempty(optimization_value["strategy"], "optimization.strategy"),
            "objective": _nonempty(optimization_value["objective"], "optimization.objective"),
            "batch_size": _integer(optimization_value["batch_size"], "optimization.batch_size", positive=True),
            "eval_batch_size": _integer(
                optimization_value["eval_batch_size"], "optimization.eval_batch_size", positive=True
            ),
            "gradient_accumulation_steps": _integer(
                optimization_value["gradient_accumulation_steps"],
                "optimization.gradient_accumulation_steps",
                positive=True,
            ),
            "learning_rate": _number(
                optimization_value["learning_rate"], "optimization.learning_rate", positive=True
            ),
            "epochs": None if epochs is None else _number(epochs, "optimization.epochs", positive=True),
            "max_steps": None
            if max_steps is None
            else _integer(max_steps, "optimization.max_steps", positive=True),
            "optimizer": _nonempty(optimization_value["optimizer"], "optimization.optimizer"),
            "scheduler": _nonempty(optimization_value["scheduler"], "optimization.scheduler"),
            "checkpoint_policy": _nonempty(
                optimization_value["checkpoint_policy"], "optimization.checkpoint_policy"
            ),
        }
        if factor_backend:
            mil_options_value = common["monitor"].get("backend_options", {}).get(
                "long_cot_mil", {}
            )
            allow_auto_save_steps = (
                optimization_value["save_steps"] == "auto"
                and optimization_value["epochs"] == 3
                and optimization_value["max_steps"] is None
                and isinstance(mil_options_value, Mapping)
                and mil_options_value.get("training_schedule") == "full_epoch_joint"
            )
            save_steps_effective = (
                "auto"
                if allow_auto_save_steps
                else _integer(
                    optimization_value["save_steps"],
                    "optimization.save_steps",
                    positive=True,
                )
            )
            optimization.update(
                {
                    "weight_decay": _number(
                        optimization_value["weight_decay"], "optimization.weight_decay"
                    ),
                    "warmup_ratio": _number(
                        optimization_value["warmup_ratio"], "optimization.warmup_ratio"
                    ),
                    "max_grad_norm": _number(
                        optimization_value["max_grad_norm"], "optimization.max_grad_norm", positive=True
                    ),
                    "mixed_precision": _nonempty(
                        optimization_value["mixed_precision"], "optimization.mixed_precision"
                    ),
                    "logging_steps": _integer(
                        optimization_value["logging_steps"], "optimization.logging_steps", positive=True
                    ),
                    "save_steps": save_steps_effective,
                    "save_total_limit": (
                        None
                        if optimization_value.get("save_total_limit", 2) is None
                        else _integer(
                            optimization_value.get("save_total_limit", 2),
                            "optimization.save_total_limit",
                            positive=True,
                        )
                    ),
                    "resume_from_checkpoint": None,
                    "gradient_checkpointing": optimization_value[
                        "gradient_checkpointing"
                    ],
                    "gradient_checkpointing_use_reentrant": optimization_value[
                        "gradient_checkpointing_use_reentrant"
                    ],
                }
            )
            if not isinstance(optimization["gradient_checkpointing"], bool):
                raise ConfigurationError(
                    "optimization.gradient_checkpointing must be boolean"
                )
            if not isinstance(
                optimization["gradient_checkpointing_use_reentrant"], bool
            ):
                raise ConfigurationError(
                    "optimization.gradient_checkpointing_use_reentrant must be boolean"
                )
            if (
                not optimization["gradient_checkpointing"]
                and optimization["gradient_checkpointing_use_reentrant"]
            ):
                raise ConfigurationError(
                    "gradient_checkpointing_use_reentrant cannot be true when "
                    "gradient checkpointing is disabled"
                )
            resume_value = optimization_value["resume_from_checkpoint"]
            if resume_value is not None:
                resume_path = resolve_path(resume_value, relative_to=base)
                if not (resume_path / "model_bundle").is_dir():
                    raise ConfigurationError(
                        f"resume checkpoint lacks model_bundle: {resume_path}"
                    )
                optimization["resume_from_checkpoint"] = str(resume_path)
            if optimization["weight_decay"] < 0:
                raise ConfigurationError("optimization.weight_decay must be nonnegative")
            if not 0 <= optimization["warmup_ratio"] < 1:
                raise ConfigurationError("optimization.warmup_ratio must be in [0,1)")
            if optimization["mixed_precision"] not in {"bf16", "fp16", "fp32"}:
                raise ConfigurationError("optimization.mixed_precision must be bf16/fp16/fp32")
        evaluation_fields = {"metric", "cadence"}
        if factor_backend:
            evaluation_fields.update(
                {"threshold_policy", "early_stopping_patience", "during_training"}
            )
        require_keys(
            common["evaluation"],
            evaluation_fields - {"metric", "during_training"},
            "evaluation",
        )
        reject_unknown(common["evaluation"], evaluation_fields, "evaluation")
        if factor_backend:
            during_training = common["evaluation"].get("during_training", True)
            if not isinstance(during_training, bool):
                raise ConfigurationError("evaluation.during_training must be boolean")
            common["evaluation"]["during_training"] = during_training
    else:
        require_keys(dataset, {"query_file"}, "dataset")
        reject_unknown(
            dataset, {"directory", "trajectory_file", "query_file"}, "dataset"
        )
        dataset_files["query_file"] = _existing_file(
            dataset["query_file"], base=dataset_dir, name="dataset.query_file"
        )
        require_keys(
            common["runtime"], {"batch_size"}, "runtime"
        )
        common["runtime"]["batch_size"] = _integer(
            common["runtime"]["batch_size"], "runtime.batch_size", positive=True
        )
        require_keys(
            common["evaluation"], {"threshold_policy", "threshold_source"}, "evaluation"
        )
        reject_unknown(
            common["evaluation"],
            {"metric", "threshold_policy", "threshold_source", "reporting_dimensions"},
            "evaluation",
        )
        if common["monitor"]["backend"] == BACKEND_NAME:
            threshold_source = _existing_file(
                common["evaluation"]["threshold_source"],
                base=base,
                name="evaluation.threshold_source",
            )
            common["evaluation"]["threshold_source"] = str(threshold_source)
            common["evaluation"]["threshold_source_sha256"] = file_sha256(
                threshold_source
            )
        optimization = {
            "strategy": "not_applicable",
            "objective": "inference_only",
            "batch_size": common["runtime"]["batch_size"],
            "eval_batch_size": common["runtime"]["batch_size"],
            "gradient_accumulation_steps": "not_applicable",
            "learning_rate": "not_applicable",
            "epochs": "not_applicable",
            "max_steps": "not_applicable",
            "optimizer": "not_applicable",
            "scheduler": "not_applicable",
            "checkpoint_policy": "not_applicable",
        }

    effective_monitor = {
        **common["monitor"],
        "visibility_scope": "cot_only",
    }
    if common["monitor"]["backend"] != BACKEND_NAME:
        effective_monitor["prompt_contract_hash"] = contract_hash()
    effective = {
        "schema_version": 1,
        "kind": kind,
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "model": common["model"],
        "monitor": effective_monitor,
        "dataset": {
            "directory": str(dataset_dir),
            "split": "train+validation" if kind == "train_run" else "benchmark",
            "files": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in dataset_files.items()
            },
        },
        "runtime": common["runtime"],
        "optimization": optimization,
        "evaluation": common["evaluation"],
        "output_directory": str(common["output_directory"]),
        "repository": repository_state(Path(__file__).resolve().parent),
        **({"initialization": initialization} if initialization is not None else {}),
    }
    if kind == "train_run":
        if any(
            value is not None
            for value in (
                evaluation_override["model_bundle"],
                evaluation_override["threshold_source"],
                evaluation_override["architecture"],
                evaluation_override["native_checkpoint_threshold"],
                evaluation_override["long_cot_aggregation"],
                evaluation_override["direct_reference_scores"],
            )
        ):
            raise ConfigurationError("evaluation overrides cannot be used with train configs")
        return _apply_validation_only_override(
            effective,
            validation_checkpoint=validation_checkpoint,
            output_directory=output_directory,
            mil_short_validation_auroc_floor=mil_short_validation_auroc_floor,
        )
    if mil_short_validation_auroc_floor is not None:
        raise ConfigurationError(
            "--mil-short-validation-auroc-floor cannot be used with eval configs"
        )
    if validation_checkpoint is not None:
        raise ConfigurationError(
            "--validation-checkpoint cannot be used with eval configs"
        )
    return _apply_evaluation_override(
        effective,
        model_bundle=evaluation_override["model_bundle"],
        threshold_source=evaluation_override["threshold_source"],
        output_directory=output_directory,
        architecture=evaluation_override["architecture"],
        native_checkpoint_threshold=evaluation_override["native_checkpoint_threshold"],
        long_cot_aggregation=evaluation_override["long_cot_aggregation"],
        direct_reference_scores=evaluation_override["direct_reference_scores"],
    )


def run_from_config(
    config_path: Path,
    *,
    dry_run: bool,
    validation_checkpoint: Path | None = None,
    output_directory: Path | None = None,
    model_bundle: Path | None = None,
    threshold_source: Path | None = None,
    architecture: Path | None = None,
    long_cot_aggregation: str | None = None,
    direct_reference_scores: Path | None = None,
    native_checkpoint_threshold: float | None = None,
    mil_short_validation_auroc_floor: float | None = None,
) -> dict[str, Any]:
    effective = resolve_run_config(
        config_path,
        validation_checkpoint=validation_checkpoint,
        output_directory=output_directory,
        model_bundle=model_bundle,
        threshold_source=threshold_source,
        architecture=architecture,
        long_cot_aggregation=long_cot_aggregation,
        direct_reference_scores=direct_reference_scores,
        native_checkpoint_threshold=native_checkpoint_threshold,
        mil_short_validation_auroc_floor=mil_short_validation_auroc_floor,
    )
    backend_name = str(effective["monitor"]["backend"])
    if backend_name == BACKEND_NAME:
        effective = validate_factor_monitor_effective_config(
            effective, require_repository_provenance=not dry_run
        )
        backend_module = importlib.import_module("factor_monitor.training.backend")
        if dry_run and effective["kind"] == "eval_run":
            backend_module.validate_evaluation_bundle_identity(
                effective, require_encoding_contracts=False
            )
    if dry_run:
        _validate_output_preflight(effective)
        return effective
    registry = _TRAIN_BACKENDS if effective["kind"] == "train_run" else _EVAL_BACKENDS
    if backend_name not in registry:
        raise ConfigurationError(
            f"backend {backend_name!r} is not registered; available={sorted(registry)}. "
            "The run was not started."
        )
    if is_world_process_zero():
        output_dir = Path(effective["output_directory"])
        _validate_output_preflight(effective)
        output_dir.mkdir(parents=True, exist_ok=True)
        if effective["kind"] == "train_run":
            write_json(output_dir / "effective_config.json", effective)
    return registry[backend_name](effective)
