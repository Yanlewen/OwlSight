"""Transformers/PEFT backend for the config-driven FactorMonitor framework."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import file_sha256, load_config, stable_hash
from ..io import read_jsonl, write_json, write_jsonl
from ..metrics import score_summary, select_balanced_threshold
from ..prompt_contract import (
    factor_suffix_choice_contract,
    factor_suffix_choice_token_ids,
)
from ..runtime import (
    is_world_process_zero,
    register_evaluation_backend,
    register_training_backend,
)
from .checkpoint import load_model_bundle, save_model_bundle
from .config import (
    BACKEND_NAME,
    FACTOR_PATH_INTERVENTIONS,
    PREDICTION_ONLY_METRIC,
    architecture_from_dict,
)
from .data import (
    EndpointTrajectoryCollator,
    FactorQueryRecord,
    FactorInstanceContrastCollator,
    FactorInstanceContrastDataset,
    FactorInstanceContrastLengthBucketBatchSampler,
    FamilyUniformLengthBucketBatchSampler,
    LongCotMilCollator,
    LongCotMilDataset,
    LongCotMilFullEpochBatchSampler,
    LongCotMilMiningState,
    LongCotMilTrainingUnit,
    LongCotMilTwoStreamBatchSampler,
    SynchronizedLengthBucketBatchSampler,
    TrajectoryCollator,
    TrajectoryGroup,
    TrajectoryGroupDataset,
    causal_choice_token_lengths,
    evidence_token_lengths,
    _evidence_window_parts,
    endpoint_token_lengths,
    factor_token_lengths,
    factor_intervention_groups,
    build_factor_instance_contrast_units,
    build_long_cot_mil_units,
    load_trajectory_groups,
)
from .encoding import (
    DEFAULT_FIELD_BOUNDARY_POLICY,
    ENDPOINT_CONTRACT_ID,
    EVIDENCE_CONTRACT_ID,
    FACTOR_CONTRACT_ID,
    FieldBoundaryError,
    SINGLE_ENDPOINT_DECISION_MODE,
    contract_hash,
    contract_spec,
    endpoint_decision_mode,
    field_boundary_policy,
    render_endpoint_inputs,
    set_field_boundary_policy,
    tokenize_rendered,
)
from .model import (
    CausalLMChoiceMonitor,
    EndpointBinaryMonitor,
    FactorMonitor,
    assert_causal_choice_trainable_parameters,
    assert_endpoint_monitor_trainable_parameters,
    assert_factor_monitor_trainable_parameters,
)
from .objective import (
    factor_instance_uniform_binary_loss,
    factor_instance_uniform_query_loss,
    hierarchical_binary_loss,
    hierarchical_query_loss,
    mil_factor_instance_uniform_binary_loss,
)


EVALUATION_PROGRESS_SCHEMA_VERSION = 1
EVALUATION_PROGRESS_INTERVAL_TRAJECTORIES = 25
DIAGNOSTIC_PROGRESS_INTERVAL_TRAJECTORIES = 5
EVALUATION_PARTIAL_SCORES_FILE = "scores.partial.jsonl"


def _epoch0_sampler_audit(sampler: Any) -> dict[str, Any]:
    """Return the sampler audit without assuming a wrapped sampler layout."""

    base = getattr(sampler, "base", None)
    if base is not None and hasattr(base, "audit"):
        return base.audit()
    if isinstance(sampler, LongCotMilFullEpochBatchSampler):
        return sampler.audit(epoch=0)
    return {
        "sampler_type": type(sampler).__name__,
        "audit_available": False,
    }


EVALUATION_PROGRESS_FILE = "progress.json"
LONG_COT_WINDOW_MANIFEST_FILE = "long_cot_window_manifest.json"
LONG_COT_AGGREGATED_SCORES_FILE = "aggregated_scores.jsonl"


def _model_variant(effective: Mapping[str, Any]) -> str:
    monitor = effective.get("monitor", {})
    options = monitor.get("backend_options", {}) if isinstance(monitor, Mapping) else {}
    return str(options.get("model_variant", "factor_late_fusion"))


def _dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FactorMonitor backend requires: pip install -e 'factor_monitor[training]'"
        ) from exc
    return locals()


def _torch_dtype(torch_module: Any, name: str) -> Any:
    values = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return values[name]


def tokenizer_identity(tokenizer: Any, *, source: str, revision: str) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    return {
        "source": source,
        "revision": revision,
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", source),
        "vocabulary_hash": stable_hash(vocabulary),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "field_boundary_policy": field_boundary_policy(tokenizer),
    }


def _load_tokenizer(effective: Mapping[str, Any], dependencies: Mapping[str, Any]) -> Any:
    model = effective["model"]
    options = model["load_options"]
    monitor_options = effective["monitor"]["backend_options"]
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        model["tokenizer"],
        revision=model["tokenizer_revision"],
        trust_remote_code=options["trust_remote_code"],
        local_files_only=options["local_files_only"],
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("frozen encoding contracts require a fast tokenizer with offsets")
    configured_pad_token = options.get("pad_token")
    if configured_pad_token is not None:
        vocabulary = tokenizer.get_vocab()
        if configured_pad_token not in vocabulary:
            raise RuntimeError(
                "model.load_options.pad_token must already exist in the tokenizer vocabulary: "
                f"{configured_pad_token!r}"
            )
        tokenizer.pad_token = configured_pad_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer must define pad_token_id; refusing to select one implicitly")
    if effective["runtime"]["padding_policy"] != "dynamic_right":
        raise RuntimeError("FactorMonitor backend currently requires runtime.padding_policy=dynamic_right")
    tokenizer.padding_side = "right"
    set_field_boundary_policy(
        tokenizer,
        monitor_options.get("field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY),
    )
    return tokenizer


def _load_base_model(effective: Mapping[str, Any], dependencies: Mapping[str, Any]) -> Any:
    model = effective["model"]
    options = model["load_options"]
    kwargs = {
        "revision": model["checkpoint_revision"],
        "trust_remote_code": options["trust_remote_code"],
        "local_files_only": options["local_files_only"],
        "attn_implementation": options["attn_implementation"],
        "torch_dtype": _torch_dtype(dependencies["torch"], options["torch_dtype"]),
    }
    return dependencies["AutoModelForCausalLM"].from_pretrained(
        model["checkpoint"], **kwargs
    )


def build_training_model(
    effective: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    architecture: Any,
    tokenizer: Any,
) -> tuple[Any, int]:
    """Construct the configured FactorMonitor model and return trainable count.

    The training backend and bounded memory benchmark share this constructor so
    memory measurements cannot silently drift to a different model variant or
    LoRA configuration.
    """

    base = _load_base_model(effective, dependencies)
    if effective["optimization"]["gradient_checkpointing"]:
        if not hasattr(base, "gradient_checkpointing_enable"):
            raise RuntimeError("base model does not implement gradient checkpointing")
        if hasattr(base, "config"):
            base.config.use_cache = False
    lora = effective["monitor"]["backend_options"]["lora"]
    initialization = effective.get("initialization")
    if initialization is not None:
        initialization_bundle = Path(initialization["model_bundle"])
        _verify_bundle_identity(initialization_bundle, effective)
        backbone = dependencies["PeftModel"].from_pretrained(
            base,
            initialization_bundle / "backbone",
            is_trainable=True,
        )
    else:
        peft_config = dependencies["LoraConfig"](
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        )
        backbone = dependencies["get_peft_model"](base, peft_config)
    variant = _model_variant(effective)
    if variant == "causal_lm_choice":
        model = CausalLMChoiceMonitor(
            backbone,
            architecture,
            choice_token_ids=factor_suffix_choice_token_ids(tokenizer),
            max_padded_tokens_per_backbone_forward=effective["monitor"][
                "backend_options"
            ]["endpoint_max_padded_tokens_per_backbone_forward"],
        )
        trainable = assert_causal_choice_trainable_parameters(model)
    elif variant == "endpoint_early_fusion":
        role_embedding = effective["monitor"]["backend_options"][
            "endpoint_role_embedding"
        ]
        model = EndpointBinaryMonitor(
            backbone,
            architecture,
            max_padded_tokens_per_backbone_forward=effective["monitor"][
                "backend_options"
            ]["endpoint_max_padded_tokens_per_backbone_forward"],
            endpoint_role_embedding_fields=(
                tuple(role_embedding["field_order"])
                if role_embedding["enabled"]
                else None
            ),
            endpoint_decision_mode=effective["monitor"]["backend_options"][
                "endpoint_decision_mode"
            ],
            factor_guided_readout_config=effective["monitor"]["backend_options"].get(
                "endpoint_factor_guided_readout"
            ),
            signed_criterion_panel_config=effective["monitor"]["backend_options"].get(
                "endpoint_signed_criterion_panel"
            ),
            layer_mixed_config=effective["monitor"]["backend_options"].get(
                "endpoint_layer_mixed"
            ),
        )
        trainable = assert_endpoint_monitor_trainable_parameters(model)
    else:
        joint_encoding = variant == "factor_joint_fusion"
        model = FactorMonitor(
            backbone,
            architecture,
            joint_encoding=joint_encoding,
            max_padded_tokens_per_backbone_forward=(
                effective["monitor"]["backend_options"][
                    "joint_max_padded_tokens_per_backbone_forward"
                ]
                if joint_encoding
                else None
            ),
        )
        trainable = assert_factor_monitor_trainable_parameters(model)
    if initialization is not None:
        initialization_bundle = Path(initialization["model_bundle"])
        model = load_model_bundle(
            backbone,
            initialization_bundle,
            endpoint_max_padded_tokens_per_backbone_forward=(
                effective["monitor"]["backend_options"].get(
                    "endpoint_max_padded_tokens_per_backbone_forward"
                )
                if variant == "endpoint_early_fusion"
                else None
            ),
            joint_max_padded_tokens_per_backbone_forward=(
                effective["monitor"]["backend_options"].get(
                    "joint_max_padded_tokens_per_backbone_forward"
                )
                if variant == "factor_joint_fusion"
                else None
            ),
        )
        if model.architecture.source_hash != architecture.source_hash:
            raise RuntimeError("initialization architecture differs from the current run")
        if isinstance(model, CausalLMChoiceMonitor):
            trainable = assert_causal_choice_trainable_parameters(model)
        elif isinstance(model, EndpointBinaryMonitor):
            trainable = assert_endpoint_monitor_trainable_parameters(model)
        else:
            trainable = assert_factor_monitor_trainable_parameters(model)
    return model, trainable


def _contract_enriched_config(
    effective: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    enriched = dict(effective)
    monitor = dict(enriched["monitor"])
    identity = tokenizer_identity(
        tokenizer,
        source=str(enriched["model"]["tokenizer"]),
        revision=str(enriched["model"]["tokenizer_revision"]),
    )
    contracts = {
        "factor": {
            "id": FACTOR_CONTRACT_ID,
            "hash": contract_hash(FACTOR_CONTRACT_ID, identity),
            "spec": contract_spec(FACTOR_CONTRACT_ID),
        },
        "evidence": {
            "id": EVIDENCE_CONTRACT_ID,
            "hash": contract_hash(EVIDENCE_CONTRACT_ID, identity),
            "spec": contract_spec(EVIDENCE_CONTRACT_ID),
        },
        "tokenizer_identity": identity,
    }
    variant = _model_variant(enriched)
    if variant in {"endpoint_early_fusion", "factor_joint_fusion", "causal_lm_choice"}:
        option_prefix = (
            "endpoint"
            if variant in {"endpoint_early_fusion", "causal_lm_choice"}
            else "joint"
        )
        serialization_contract = enriched["monitor"]["backend_options"][
            f"{option_prefix}_encoding_contract_id"
        ]
        contracts[
            "endpoint"
            if variant in {"endpoint_early_fusion", "causal_lm_choice"}
            else "joint"
        ] = {
            "id": serialization_contract,
            "hash": contract_hash(serialization_contract, identity),
            "spec": contract_spec(serialization_contract),
        }
    if variant == "causal_lm_choice":
        choice = factor_suffix_choice_contract()
        choice_ids = factor_suffix_choice_token_ids(tokenizer)
        contracts["choice"] = {
            "id": choice["contract_id"],
            "hash": stable_hash({"contract": choice, "token_ids": choice_ids}),
            "spec": choice,
            "token_ids": choice_ids,
            "chat_template": "tokenizer_native",
            "chat_template_kwargs": {"enable_thinking": False},
        }
    monitor["encoding_contracts"] = contracts
    enriched["monitor"] = monitor
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise RuntimeError("WORLD_SIZE must be an integer") from exc
    if world_size <= 0:
        raise RuntimeError("WORLD_SIZE must be positive")
    optimization = enriched.get("optimization")
    if enriched.get("kind") == "train_run" and isinstance(optimization, Mapping):
        expected_world_size = enriched.get("runtime", {}).get("expected_world_size")
        if expected_world_size is not None and world_size != int(expected_world_size):
            raise RuntimeError(
                "training world size differs from runtime.expected_world_size: "
                f"expected={expected_world_size}, actual={world_size}"
            )
        per_device_batch = int(optimization["batch_size"])
        accumulation = int(optimization["gradient_accumulation_steps"])
        enriched["distributed"] = {
            "world_size": world_size,
            "per_device_trajectory_batch_size": per_device_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_global_trajectory_batch_size": (
                world_size * per_device_batch * accumulation
            ),
        }
    return enriched


def _verify_bundle_identity(
    bundle: Path,
    effective: Mapping[str, Any],
    *,
    verify_encoding_contracts: bool = True,
) -> None:
    path = bundle / "effective_config.json"
    if not path.is_file():
        raise RuntimeError(f"model bundle lacks effective_config.json: {bundle}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in ("checkpoint", "checkpoint_revision", "tokenizer", "tokenizer_revision"):
        if saved.get("model", {}).get(key) != effective.get("model", {}).get(key):
            raise RuntimeError(f"evaluation {key} differs from the saved model bundle")
    saved_monitor = saved.get("monitor", {})
    current_monitor = effective.get("monitor", {})
    saved_variant = saved_monitor.get("backend_options", {}).get(
        "model_variant", "factor_late_fusion"
    )
    current_variant = current_monitor.get("backend_options", {}).get(
        "model_variant", "factor_late_fusion"
    )
    if saved_variant != current_variant:
        raise RuntimeError("evaluation model_variant differs from the saved model bundle")
    saved_boundary_policy = saved_monitor.get("backend_options", {}).get(
        "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
    )
    current_boundary_policy = current_monitor.get("backend_options", {}).get(
        "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
    )
    if saved_boundary_policy != current_boundary_policy:
        raise RuntimeError(
            "evaluation field boundary policy differs from the saved model bundle"
        )

    def guided_readout_identity(
        monitor: Mapping[str, Any], *, owner: str
    ) -> dict[str, Any]:
        options = monitor.get("backend_options", {})
        if not isinstance(options, Mapping):
            raise RuntimeError(f"{owner} model bundle backend_options are invalid")
        value = options.get("endpoint_factor_guided_readout")
        if value is None:
            return {"enabled": False}
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{owner} guided readout identity is invalid")
        if value.get("enabled") is False:
            return {"enabled": False}
        return dict(value)

    if guided_readout_identity(
        saved_monitor, owner="saved"
    ) != guided_readout_identity(current_monitor, owner="evaluation"):
        raise RuntimeError(
            "evaluation Factor-guided readout differs from the saved model bundle"
        )

    def optional_endpoint_block_identity(
        monitor: Mapping[str, Any], *, owner: str, key: str
    ) -> dict[str, Any]:
        options = monitor.get("backend_options", {})
        if not isinstance(options, Mapping):
            raise RuntimeError(f"{owner} model bundle backend_options are invalid")
        value = options.get(key)
        if value is None:
            return {"enabled": False}
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{owner} {key} identity is invalid")
        if value.get("enabled") is False:
            return {"enabled": False}
        return dict(value)

    for block_key, label in (
        ("endpoint_signed_criterion_panel", "Signed Criterion Panel"),
        ("endpoint_layer_mixed", "Layer-Mixed Endpoint"),
    ):
        if optional_endpoint_block_identity(
            saved_monitor, owner="saved", key=block_key
        ) != optional_endpoint_block_identity(
            current_monitor, owner="evaluation", key=block_key
        ):
            raise RuntimeError(
                f"evaluation {label} differs from the saved model bundle"
            )

    def role_identity(monitor: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
        options = monitor.get("backend_options", {})
        if not isinstance(options, Mapping):
            raise RuntimeError(f"{owner} model bundle backend_options are invalid")
        value = options.get("endpoint_role_embedding")
        if value is None:
            return {"enabled": False}
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{owner} endpoint role embedding identity is invalid")
        if value.get("enabled") is False:
            return {"enabled": False}
        return dict(value)

    saved_role = role_identity(saved_monitor, owner="saved")
    current_role = role_identity(current_monitor, owner="evaluation")
    if saved_role != current_role:
        raise RuntimeError(
            "evaluation endpoint role embedding differs from the saved model bundle"
        )
    def decision_identity(monitor: Mapping[str, Any], *, owner: str) -> str:
        options = monitor.get("backend_options", {})
        if not isinstance(options, Mapping):
            raise RuntimeError(f"{owner} model bundle backend_options are invalid")
        value = options.get("endpoint_decision_mode")
        if value is not None:
            return str(value)
        contract_id = options.get("endpoint_encoding_contract_id")
        if contract_id is None:
            contract_id = (
                monitor.get("encoding_contracts", {}).get("endpoint", {}).get("id")
            )
        if contract_id is None:
            return SINGLE_ENDPOINT_DECISION_MODE
        try:
            return endpoint_decision_mode(str(contract_id))
        except ValueError as exc:
            raise RuntimeError(
                f"{owner} endpoint decision identity cannot be derived"
            ) from exc

    if current_variant == "endpoint_early_fusion" and decision_identity(
        saved_monitor, owner="saved"
    ) != decision_identity(current_monitor, owner="evaluation"):
        raise RuntimeError(
            "evaluation endpoint decision mode differs from the saved model bundle"
        )
    for key in ("architecture_hash", "route_mapping_hash"):
        if saved_monitor.get(key) != current_monitor.get(key):
            raise RuntimeError(f"evaluation {key} differs from the saved model bundle")
    if not verify_encoding_contracts:
        return

    saved_contracts = saved_monitor.get("encoding_contracts", {})
    current_contracts = current_monitor.get("encoding_contracts", {})

    def encoding_hash_matches(stream: str) -> bool:
        saved_encoding = saved_contracts.get(stream, {})
        current_encoding = current_contracts.get(stream, {})
        saved_hash = saved_encoding.get("hash")
        current_hash = current_encoding.get("hash")
        if saved_hash and saved_hash == current_hash:
            return True

        # Bundles created before field_boundary_policy became part of the
        # tokenizer identity have the same strict renderer/tokenizer behavior,
        # but their persisted contract hash omits that metadata field. Accept
        # only this exact legacy shape; all other tokenizer or spec drift must
        # remain a hard failure.
        saved_identity = saved_contracts.get("tokenizer_identity")
        current_identity = current_contracts.get("tokenizer_identity")
        if not isinstance(saved_identity, Mapping) or not isinstance(
            current_identity, Mapping
        ):
            return False
        if "field_boundary_policy" in saved_identity:
            return False
        if current_identity.get("field_boundary_policy") != DEFAULT_FIELD_BOUNDARY_POLICY:
            return False
        legacy_identity = dict(current_identity)
        legacy_identity.pop("field_boundary_policy", None)
        if dict(saved_identity) != legacy_identity:
            return False
        if saved_encoding.get("id") != current_encoding.get("id"):
            return False
        if saved_encoding.get("spec") != current_encoding.get("spec"):
            return False
        contract_id = current_encoding.get("id")
        if not isinstance(contract_id, str) or not contract_id:
            return False
        return saved_hash == contract_hash(contract_id, legacy_identity)

    for stream in ("factor", "evidence"):
        if not encoding_hash_matches(stream):
            raise RuntimeError(
                f"evaluation {stream} encoding contract differs from the saved model bundle"
            )
    if current_variant in {"endpoint_early_fusion", "factor_joint_fusion"}:
        contract_key = "endpoint" if current_variant == "endpoint_early_fusion" else "joint"
        if not encoding_hash_matches(contract_key):
            raise RuntimeError(
                "evaluation joint encoding contract differs from the saved model bundle"
            )


def validate_evaluation_bundle_identity(
    effective: Mapping[str, Any], *, require_encoding_contracts: bool = True
) -> None:
    """Fail before model loading when an eval config drifts from its saved bundle."""

    bundle = Path(effective["monitor"]["backend_options"]["model_bundle"])
    _verify_bundle_identity(
        bundle,
        effective,
        verify_encoding_contracts=require_encoding_contracts,
    )


def _dataset_paths(effective: Mapping[str, Any]) -> dict[str, Path]:
    return {
        key: Path(value["path"]).resolve()
        for key, value in effective["dataset"]["files"].items()
    }


def _query_split_path(paths: Mapping[str, Path], split: str) -> Path:
    key = "query_file" if split == "query" else f"{split}_file"
    try:
        return paths[key]
    except KeyError as error:
        raise KeyError(
            f"dataset files do not define the {split!r} query split key {key!r}"
        ) from error


def _load_groups(
    effective: Mapping[str, Any], *, split: str, allow_unlabelled: bool = False
) -> list[TrajectoryGroup]:
    paths = _dataset_paths(effective)
    return load_trajectory_groups(
        paths["trajectory_file"],
        _query_split_path(paths, split),
        architecture=architecture_from_dict(effective["monitor"]["architecture"]),
        allow_unlabelled=allow_unlabelled,
    )


def _collator(
    effective: Mapping[str, Any],
    tokenizer: Any,
    *,
    pad_queries_to: int | None = None,
) -> TrajectoryCollator | EndpointTrajectoryCollator:
    options = effective["monitor"]["backend_options"]
    if _model_variant(effective) in {
        "endpoint_early_fusion",
        "factor_joint_fusion",
        "causal_lm_choice",
    }:
        option_prefix = (
            "endpoint"
            if _model_variant(effective) in {
                "endpoint_early_fusion",
                "causal_lm_choice",
            }
            else "joint"
        )
        return EndpointTrajectoryCollator(
            tokenizer,
            architecture_from_dict(effective["monitor"]["architecture"]),
            max_sequence_length=int(effective["runtime"]["max_sequence_length"]),
            max_queries_per_trajectory=options["batching"]["max_queries_per_trajectory"],
            padding_side="right",
            pad_queries_to=pad_queries_to,
            output_mode=(
                "endpoint"
                if _model_variant(effective) in {
                    "endpoint_early_fusion",
                    "causal_lm_choice",
                }
                else "factor_joint"
            ),
            endpoint_contract_id=options[f"{option_prefix}_encoding_contract_id"],
            include_endpoint_span_masks=bool(
                _model_variant(effective) == "endpoint_early_fusion"
                and (
                    options.get("endpoint_factor_guided_readout", {}).get("enabled")
                    or options.get("endpoint_signed_criterion_panel", {}).get("enabled")
                )
            ),
            choice_contract_id=(
                options["choice_contract_id"]
                if _model_variant(effective) == "causal_lm_choice"
                else None
            ),
            chat_template_kwargs=(
                options["chat_template_kwargs"]
                if _model_variant(effective) == "causal_lm_choice"
                else None
            ),
        )
    return TrajectoryCollator(
        tokenizer,
        architecture_from_dict(effective["monitor"]["architecture"]),
        max_evidence_tokens=int(effective["runtime"]["max_sequence_length"]),
        max_factor_tokens=int(options["sequence"]["max_factor_tokens"]),
        max_queries_per_trajectory=options["batching"]["max_queries_per_trajectory"],
        padding_side="right",
        pad_queries_to=pad_queries_to,
    )


def _validate_token_limits(
    effective: Mapping[str, Any],
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
) -> dict[str, list[int]]:
    maximum_evidence = int(effective["runtime"]["max_sequence_length"])
    maximum_factor = int(
        effective["monitor"]["backend_options"]["sequence"]["max_factor_tokens"]
    )
    lengths_by_split: dict[str, list[int]] = {}
    for split, groups in groups_by_split.items():
        if _model_variant(effective) in {
            "endpoint_early_fusion",
            "factor_joint_fusion",
            "causal_lm_choice",
        }:
            option_prefix = (
                "endpoint"
                if _model_variant(effective) in {
                    "endpoint_early_fusion",
                    "causal_lm_choice",
                }
                else "joint"
            )
            query_lengths = (
                causal_choice_token_lengths(
                    tokenizer,
                    groups,
                    chat_template_kwargs=effective["monitor"]["backend_options"][
                        "chat_template_kwargs"
                    ],
                )
                if _model_variant(effective) == "causal_lm_choice"
                else endpoint_token_lengths(
                    tokenizer,
                    groups,
                    contract_id=effective["monitor"]["backend_options"][
                        f"{option_prefix}_encoding_contract_id"
                    ],
                )
            )
            overlength = [
                query_id for query_id, length in query_lengths if length > maximum_evidence
            ]
            if overlength:
                raise RuntimeError(
                    f"{split} artifact contains {len(overlength)} joint inputs above "
                    f"max_sequence_length: {overlength[:3]}"
                )
            if _model_variant(effective) == "factor_joint_fusion":
                factor_overlength = [
                    query_id
                    for query_id, length in factor_token_lengths(tokenizer, groups)
                    if length > maximum_factor
                ]
                if factor_overlength:
                    raise RuntimeError(
                        f"{split} artifact contains {len(factor_overlength)} Factors "
                        f"above max_factor_tokens: {factor_overlength[:3]}"
                    )
            by_query = dict(query_lengths)
            lengths_by_split[split] = [
                max(by_query[query.query_id] for query in group.queries)
                for group in groups
            ]
            continue
        lengths = evidence_token_lengths(tokenizer, groups)
        lengths_by_split[split] = lengths
        overlength = [
            group.trajectory_id
            for group, length in zip(groups, lengths, strict=True)
            if length > maximum_evidence
        ]
        if overlength:
            raise RuntimeError(
                f"{split} artifact contains {len(overlength)} trajectories above "
                f"max_sequence_length: {overlength[:3]}"
            )
        factor_overlength = [
            query_id
            for query_id, length in factor_token_lengths(tokenizer, groups)
            if length > maximum_factor
        ]
        if factor_overlength:
            raise RuntimeError(
                f"{split} artifact contains {len(factor_overlength)} Factors above "
                f"max_factor_tokens: {factor_overlength[:3]}"
            )
    return lengths_by_split


def _select_evaluation_groups(
    effective: Mapping[str, Any],
    tokenizer: Any,
    groups: Sequence[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Apply an explicitly configured engineering-only evaluation selection."""

    scope = effective["monitor"]["backend_options"]["evaluation_scope"]
    selection = scope.get("trajectory_selection")
    maximum = scope.get("max_trajectories")
    if selection is None and maximum is None:
        return list(groups)
    if scope["mode"] != "smoke_reload":
        raise RuntimeError("formal evaluation cannot select a trajectory subset")
    if selection != "longest_evidence" or maximum != 1:
        raise RuntimeError(
            "the supported evaluation gate is longest_evidence with max_trajectories=1"
        )
    lengths = evidence_token_lengths(tokenizer, groups)
    selected_index = max(range(len(groups)), key=lambda index: lengths[index])
    selected = groups[selected_index]
    print(
        json.dumps(
            {
                "event": "evaluation_scope_selection",
                "mode": scope["mode"],
                "trajectory_selection": selection,
                "trajectory_id": selected.trajectory_id,
                "evidence_tokens": lengths[selected_index],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return [selected]


def _field_token_boundaries(tokenizer: Any, text: str, *, field: str) -> list[int]:
    """Map one raw evidence field's token boundaries to lossless character slices."""

    if field not in {"context", "cot"}:
        raise ValueError(f"unsupported endpoint evidence field: {field}")
    if not text:
        return [0]

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise RuntimeError(f"long-CoT windowing requires {field} tokenizer IDs and offsets")
    if not input_ids or len(input_ids) != len(offsets):
        raise RuntimeError(f"long-CoT windowing received invalid raw {field} token offsets")
    parsed: list[tuple[int, int]] = []
    for value in offsets:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], int)
            or not isinstance(value[1], int)
            or value[0] < 0
            or value[1] < value[0]
            or value[1] > len(text)
        ):
            raise RuntimeError(f"long-CoT windowing received malformed {field} token offsets")
        parsed.append((value[0], value[1]))
    boundaries = [0, *(start for start, _ in parsed[1:]), len(text)]
    if boundaries != sorted(boundaries):
        raise RuntimeError(f"long-CoT {field} token offsets are not monotonic")
    return boundaries


def _window_query(
    query: FactorQueryRecord,
    *,
    trajectory_id: str,
    index: int,
    field: str,
) -> FactorQueryRecord:
    source_query_id = query.source_query_id or query.query_id
    query_prefix = query.query_id if query.intervention != "matched" else source_query_id
    marker = f"{field}-window"
    return replace(
        query,
        query_id=f"{query_prefix}::{marker}={index:04d}",
        trajectory_id=trajectory_id,
        factor_instance_id=f"{query.factor_instance_id}::{marker}={index:04d}",
        source_query_id=source_query_id,
        donor_query_id=query.donor_query_id or query.query_id,
    )


def _long_cot_window_groups(
    tokenizer: Any,
    groups: Sequence[TrajectoryGroup],
    *,
    contract_id: str,
    trigger_endpoint_tokens: int,
    max_endpoint_tokens: int,
    cot_overlap_tokens: int,
    context_policy: str = "repeat_full",
    overlap_fallback_policy: str = "error",
) -> tuple[list[TrajectoryGroup], list[TrajectoryGroup], dict[str, Any]]:
    """Select long trajectories and losslessly window the binding evidence field."""

    if context_policy not in {
        "repeat_full",
        "repeat_full_when_feasible_else_window_context",
    }:
        raise ValueError(f"unsupported long-CoT context policy: {context_policy}")
    if overlap_fallback_policy not in {"error", "half_field_capacity_when_needed"}:
        raise ValueError(
            f"unsupported long-CoT overlap fallback: {overlap_fallback_policy}"
        )

    full_lengths = dict(
        endpoint_token_lengths(tokenizer, groups, contract_id=contract_id)
    )
    selected = [
        group
        for group in groups
        if any(full_lengths[query.query_id] > trigger_endpoint_tokens for query in group.queries)
    ]
    if not selected:
        raise RuntimeError("long_cot_windowing selected no trajectories above its trigger")
    directly_over = sum(
        full_lengths[query.query_id] > trigger_endpoint_tokens
        for group in selected
        for query in group.queries
    )
    window_groups: list[TrajectoryGroup] = []
    manifest_rows: list[dict[str, Any]] = []
    for group in selected:
        cot_boundaries = _field_token_boundaries(
            tokenizer, group.model_cot, field="cot"
        )
        context_boundaries = _field_token_boundaries(
            tokenizer, group.task_context, field="context"
        )
        cot_tokens = len(cot_boundaries) - 1
        context_tokens = len(context_boundaries) - 1
        cot_base_maximum = max(
            length
            for _, length in endpoint_token_lengths(
                tokenizer, [replace(group, model_cot="")], contract_id=contract_id
            )
        )
        cot_capacity = max_endpoint_tokens - cot_base_maximum
        window_field = "cot"
        base_maximum = cot_base_maximum
        capacity = cot_capacity
        if (
            context_policy == "repeat_full_when_feasible_else_window_context"
            and cot_capacity <= 0
        ):
            window_field = "context"
            base_maximum = max(
                length
                for _, length in endpoint_token_lengths(
                    tokenizer,
                    [replace(group, task_context="")],
                    contract_id=contract_id,
                )
            )
            capacity = max_endpoint_tokens - base_maximum
        if capacity <= 0:
            raise RuntimeError(
                f"trajectory {group.trajectory_id} cannot fit Factor Suffix plus the "
                f"complete non-windowed evidence field while windowing {window_field}"
            )
        effective_overlap = cot_overlap_tokens
        overlap_fallback_applied = False
        if capacity <= cot_overlap_tokens:
            if overlap_fallback_policy == "error":
                raise RuntimeError(
                    f"trajectory {group.trajectory_id} leaves only {capacity} "
                    f"{window_field} tokens, not enough for the configured "
                    f"{cot_overlap_tokens}-token overlap"
                )
            effective_overlap = capacity // 2
            overlap_fallback_applied = True
        boundaries = cot_boundaries if window_field == "cot" else context_boundaries
        field_tokens = len(boundaries) - 1
        if field_tokens <= 0:
            raise RuntimeError(
                f"trajectory {group.trajectory_id} selected an empty {window_field} field"
            )
        start = 0
        windows: list[dict[str, Any]] = []
        while start < field_tokens:
            nominal_start = start
            previous_start = windows[-1][f"{window_field}_token_start"] if windows else -1
            previous_end = windows[-1][f"{window_field}_token_end"] if windows else -1

            def materialize(end: int, index: int) -> tuple[TrajectoryGroup, dict[str, int]]:
                source_text = (
                    group.model_cot if window_field == "cot" else group.task_context
                )
                chunk = source_text[boundaries[start] : boundaries[end]]
                marker = f"{window_field}-window"
                trajectory_id = f"{group.trajectory_id}::{marker}={index:04d}"
                window_group = TrajectoryGroup(
                    trajectory_id=trajectory_id,
                    task_context=(chunk if window_field == "context" else group.task_context),
                    model_cot=(chunk if window_field == "cot" else group.model_cot),
                    family=group.family,
                    leakage_group_id=group.leakage_group_id,
                    queries=tuple(
                        _window_query(
                            query,
                            trajectory_id=trajectory_id,
                            index=index,
                            field=window_field,
                        )
                        for query in group.queries
                    ),
                )
                lengths = {
                    query.query_id: max(
                        len(tokenize_rendered(tokenizer, rendered).input_ids)
                        for rendered in render_endpoint_inputs(
                            query.factor_spec,
                            task_context=window_group.task_context,
                            model_cot=window_group.model_cot,
                            contract_id=contract_id,
                            omit_factor=query.intervention == "no_factor",
                        )
                    )
                    for query in window_group.queries
                }
                return window_group, lengths

            index = len(windows)
            # Raw field-token boundaries are not sufficient by themselves: once a
            # chunk is embedded in the endpoint contract, the tokenizer may merge
            # either edge with the adjacent structural newline. First move the end
            # backward; if the fixed start itself is unsafe, move it backward too.
            # Moving a later start backward only increases overlap. Requiring the
            # new end to pass the previous end preserves forward progress.
            minimum_start = 0 if not windows else previous_start + 1
            found = False
            while start >= minimum_start and not found:
                candidate_end = min(field_tokens, start + capacity)
                minimum_end = max(start + 1, previous_end + 1)
                while candidate_end >= minimum_end:
                    try:
                        window_group, window_lengths = materialize(candidate_end, index)
                    except FieldBoundaryError as exc:
                        if exc.field != window_field:
                            raise
                        if exc.side in {"start", "both"}:
                            break
                        candidate_end -= 1
                        continue
                    if (
                        window_lengths
                        and max(window_lengths.values()) <= max_endpoint_tokens
                    ):
                        found = True
                        break
                    candidate_end -= 1
                if not found:
                    start -= 1
            if not found:
                start = nominal_start
                candidate_end = min(field_tokens, start + capacity)
                window_lengths = {}
                window_group = None
            if not found or window_group is None:
                raise RuntimeError(
                    f"trajectory {group.trajectory_id} has no non-empty {window_field} window "
                    "that fits while preserving embedded endpoint field boundaries"
                )
            window_groups.append(window_group)
            cot_start = start if window_field == "cot" else 0
            cot_end = candidate_end if window_field == "cot" else cot_tokens
            context_start = start if window_field == "context" else 0
            context_end = candidate_end if window_field == "context" else context_tokens
            windows.append(
                {
                    "window_index": index,
                    "window_field": window_field,
                    "context_token_start": context_start,
                    "context_token_end": context_end,
                    "context_char_start": context_boundaries[context_start],
                    "context_char_end": context_boundaries[context_end],
                    "cot_token_start": cot_start,
                    "cot_token_end": cot_end,
                    "cot_char_start": cot_boundaries[cot_start],
                    "cot_char_end": cot_boundaries[cot_end],
                    "endpoint_max_tokens": max(window_lengths.values()),
                    "endpoint_tokens_by_query": {
                        query.source_query_id or query.query_id: window_lengths[query.query_id]
                        for query in window_group.queries
                    },
                }
            )
            if candidate_end == field_tokens:
                break
            next_start = candidate_end - effective_overlap
            if next_start <= start:
                raise RuntimeError(
                    f"long-CoT {window_field} windowing failed to make forward progress"
                )
            start = next_start
        start_key = f"{window_field}_token_start"
        end_key = f"{window_field}_token_end"
        if windows[0][start_key] != 0 or windows[-1][end_key] != field_tokens:
            raise RuntimeError(
                f"long-CoT windows do not cover both {window_field} endpoints"
            )
        for left, right in zip(windows, windows[1:]):
            if right[start_key] > left[end_key]:
                raise RuntimeError(
                    f"long-CoT {window_field} windows contain an uncovered token gap"
                )
        manifest_rows.append(
            {
                "source_trajectory_id": group.trajectory_id,
                "family": group.family,
                "source_queries": len(group.queries),
                "directly_over_trigger_queries": sum(
                    full_lengths[query.query_id] > trigger_endpoint_tokens
                    for query in group.queries
                ),
                "full_endpoint_tokens_by_query": {
                    query.query_id: full_lengths[query.query_id]
                    for query in group.queries
                },
                "window_field": window_field,
                "context_tokens": context_tokens,
                "cot_tokens": cot_tokens,
                "derived_field_capacity": capacity,
                "derived_cot_capacity": cot_capacity,
                "target_overlap_tokens": cot_overlap_tokens,
                "effective_overlap_tokens": effective_overlap,
                "overlap_fallback_applied": overlap_fallback_applied,
                "windows": windows,
            }
        )
    selected_queries = sum(len(group.queries) for group in selected)
    manifest = {
        "schema_version": 1,
        "policy": {
            "trigger_endpoint_tokens": trigger_endpoint_tokens,
            "max_endpoint_tokens": max_endpoint_tokens,
            "cot_overlap_tokens": cot_overlap_tokens,
            "context_policy": context_policy,
            "overlap_fallback_policy": overlap_fallback_policy,
            "coverage_policy": "complete_no_truncation",
            "selection_unit": "trajectory_any_query_over_trigger",
        },
        "summary": {
            "source_trajectories": len(groups),
            "source_queries": sum(len(group.queries) for group in groups),
            "selected_trajectories": len(selected),
            "selected_queries": selected_queries,
            "directly_over_trigger_queries": directly_over,
            "window_trajectories": len(window_groups),
            "window_queries": sum(len(group.queries) for group in window_groups),
            "windowed_trajectories_by_field": dict(
                sorted(Counter(row["window_field"] for row in manifest_rows).items())
            ),
            "overlap_fallback_trajectories": sum(
                bool(row["overlap_fallback_applied"]) for row in manifest_rows
            ),
            "maximum_source_endpoint_tokens": max(full_lengths.values()),
            "maximum_window_endpoint_tokens": max(
                window["endpoint_max_tokens"]
                for row in manifest_rows
                for window in row["windows"]
            ),
        },
        "trajectories": manifest_rows,
    }
    return selected, window_groups, manifest


def _apply_shared_window_boundaries(
    tokenizer: Any,
    variant_groups: Sequence[TrajectoryGroup],
    matched_source_groups: Sequence[TrajectoryGroup],
    matched_window_groups: Sequence[TrajectoryGroup],
    *,
    variant: str,
    contract_id: str,
    max_endpoint_tokens: int,
) -> list[TrajectoryGroup]:
    """Reuse matched window text/boundaries for one intervention variant.

    Window boundaries are a property of the evidence text, not of the Factor
    perturbation.  This helper therefore never recomputes a split for an
    ablated/permuted Factor.  It only swaps the query metadata into the already
    materialized matched windows and fails closed if the swapped endpoint no
    longer fits the frozen token budget.
    """

    by_source_trajectory = {
        group.trajectory_id: group for group in variant_groups
    }
    expected_sources = {group.trajectory_id for group in matched_source_groups}
    if set(by_source_trajectory) != expected_sources:
        raise RuntimeError(
            f"{variant} intervention changed the selected long-trajectory set"
        )
    output: list[TrajectoryGroup] = []
    for template in matched_window_groups:
        parts = _evidence_window_parts(template.trajectory_id)
        if parts is None:
            raise RuntimeError("matched window template lacks a field-aware ID")
        source_trajectory, marker, index = parts
        field = marker.removesuffix("-window")
        source_group = by_source_trajectory[source_trajectory]
        query_by_source = {
            query.source_query_id or query.query_id: query
            for query in source_group.queries
        }
        window_queries = []
        for matched_query in template.queries:
            source_query_id = matched_query.source_query_id or matched_query.query_id
            query = query_by_source.get(source_query_id)
            if query is None:
                raise RuntimeError(
                    f"{variant} intervention lacks source query {source_query_id}"
                )
            variant_trajectory_id = template.trajectory_id
            if variant != "matched":
                variant_trajectory_id = (
                    f"{source_trajectory}::factor={variant}::{marker}={index:04d}"
                )
            window_queries.append(
                _window_query(
                    query,
                    trajectory_id=variant_trajectory_id,
                    index=index,
                    field=field,
                )
            )
        variant_window = TrajectoryGroup(
            trajectory_id=(
                template.trajectory_id
                if variant == "matched"
                else f"{source_trajectory}::factor={variant}::{marker}={index:04d}"
            ),
            task_context=template.task_context,
            model_cot=template.model_cot,
            family=template.family,
            leakage_group_id=template.leakage_group_id,
            queries=tuple(window_queries),
        )
        lengths = [
            max(
                len(tokenize_rendered(tokenizer, rendered).input_ids)
                for rendered in render_endpoint_inputs(
                    query.factor_spec,
                    task_context=variant_window.task_context,
                    model_cot=variant_window.model_cot,
                    contract_id=contract_id,
                    omit_factor=query.intervention == "no_factor",
                )
            )
            for query in variant_window.queries
        ]
        if lengths and max(lengths) > max_endpoint_tokens:
            raise RuntimeError(
                f"{variant} intervention does not fit the matched window boundary "
                f"for {source_trajectory} window {index}: max={max(lengths)} > "
                f"budget={max_endpoint_tokens}"
            )
        output.append(variant_window)
    return output


def _validate_frozen_long_cot_manifest(
    generated: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless generated windows preserve every frozen manifest field."""

    path = Path(str(binding["path"]))
    expected_sha = str(binding["sha256"])
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError("frozen long-CoT manifest changed after config resolution")
    frozen = load_config(path)

    def require_frozen_subset(
        expected: Any, actual: Any, *, location: str
    ) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                raise RuntimeError(f"frozen long-CoT manifest type drift at {location}")
            missing = sorted(set(expected) - set(actual))
            if missing:
                raise RuntimeError(
                    f"generated long-CoT manifest lacks frozen fields at {location}: "
                    f"{missing}"
                )
            for key, value in expected.items():
                require_frozen_subset(
                    value, actual[key], location=f"{location}.{key}"
                )
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise RuntimeError(
                    f"frozen long-CoT manifest list drift at {location}"
                )
            for index, value in enumerate(expected):
                require_frozen_subset(
                    value, actual[index], location=f"{location}[{index}]"
                )
            return
        if expected != actual:
            raise RuntimeError(f"frozen long-CoT manifest value drift at {location}")

    require_frozen_subset(frozen, generated, location="manifest")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "validation": "all_frozen_fields_match_generated_manifest",
    }


def _load_teacher_distillation_logits(
    effective: Mapping[str, Any],
    teacher: Mapping[str, Any],
    generated_manifest: Mapping[str, Any],
    window_groups: Sequence[TrajectoryGroup],
) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    """Validate and load one immutable all-train-window Teacher score file."""

    frozen_audit = _validate_frozen_long_cot_manifest(
        generated_manifest, teacher["frozen_window_manifest"]
    )

    def revalidate_file(binding: Mapping[str, Any], label: str) -> Path:
        path = Path(str(binding["path"]))
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise RuntimeError(f"Teacher {label} changed after config resolution")
        return path

    score_path = revalidate_file(teacher["score_file"], "score file")
    teacher_effective_path = revalidate_file(
        teacher["score_effective_config"], "effective config"
    )
    teacher_effective = load_config(teacher_effective_path)
    if teacher_effective.get("kind") != "eval_run":
        raise RuntimeError("Teacher scores must come from a formal eval_run")
    teacher_files = teacher_effective.get("dataset", {}).get("files", {})
    current_files = effective.get("dataset", {}).get("files", {})
    for teacher_key, current_key in (
        ("query_file", "train_file"),
        ("trajectory_file", "trajectory_file"),
    ):
        if (
            teacher_files.get(teacher_key, {}).get("sha256")
            != current_files.get(current_key, {}).get("sha256")
        ):
            raise RuntimeError(
                f"Teacher {teacher_key} is not bound to the current training split"
            )
    teacher_options = teacher_effective.get("monitor", {}).get("backend_options", {})
    configured_bundle = teacher["teacher_model_bundle"]
    if Path(str(teacher_options.get("model_bundle", ""))).resolve() != Path(
        str(configured_bundle["path"])
    ).resolve():
        raise RuntimeError("Teacher score model bundle differs from training config")
    model_bundle = Path(str(configured_bundle["path"]))
    factor_monitor_path = model_bundle / "factor_monitor.safetensors"
    adapter_path = model_bundle / "backbone" / "adapter_model.safetensors"
    if (
        file_sha256(factor_monitor_path) != configured_bundle["factor_monitor_sha256"]
        or file_sha256(adapter_path) != configured_bundle["adapter_model_sha256"]
    ):
        raise RuntimeError("Teacher model bundle changed after config resolution")
    teacher_bundle_effective = load_config(model_bundle / "effective_config.json")
    expected_maximum = {"t32": 33024, "t65": 66048}[teacher["teacher_role"]]
    if teacher_bundle_effective.get("runtime", {}).get("max_sequence_length") != expected_maximum:
        raise RuntimeError("Teacher role disagrees with its source training context length")
    teacher_windowing = teacher_options.get("evaluation_scope", {}).get(
        "long_cot_windowing", {}
    )
    if (
        teacher_windowing.get("comparison_mode") != "teacher_window_audit"
        or teacher_windowing.get("frozen_window_manifest", {}).get("sha256")
        != teacher["frozen_window_manifest"]["sha256"]
    ):
        raise RuntimeError("Teacher scoring run used a different frozen window manifest")

    summary = generated_manifest.get("summary", {})
    expected_summary = {
        "selected_trajectories": 286,
        "selected_queries": 572,
        "window_trajectories": 805,
        "window_queries": 1610,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"Teacher train manifest {key} drifted: expected {expected}, "
                f"found {summary.get(key)!r}"
            )

    rows = read_jsonl(score_path)
    expected_rows: list[dict[str, Any]] = []
    for group in window_groups:
        window_index = LongCotMilMiningState._window_index(group)
        for query in group.queries:
            if not isinstance(query.source_query_id, str):
                raise RuntimeError("Teacher window query lacks a source query ID")
            expected_rows.append(
                {
                    "query_id": query.query_id,
                    "source_query_id": query.source_query_id,
                    "trajectory_id": group.trajectory_id,
                    "window_index": window_index,
                    "label_id": query.label,
                    "family": query.family,
                    "factor_level": query.factor_level,
                }
            )
    if len(rows) != len(expected_rows):
        raise RuntimeError(
            f"Teacher score coverage differs: expected {len(expected_rows)}, "
            f"found {len(rows)}"
        )
    logits: dict[tuple[str, int], float] = {}
    positive_sources: set[str] = set()
    positive_window_rows = 0
    for index, (row, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        for field in (
            "query_id",
            "source_query_id",
            "trajectory_id",
            "label_id",
            "family",
            "factor_level",
        ):
            if row.get(field) != expected[field]:
                raise RuntimeError(
                    f"Teacher score row {index} changed {field}: "
                    f"expected {expected[field]!r}, found {row.get(field)!r}"
                )
        logit = row.get("logit")
        if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(logit):
            raise RuntimeError(f"Teacher score row {index} lacks a finite raw logit")
        key = (str(expected["source_query_id"]), int(expected["window_index"]))
        if key in logits:
            raise RuntimeError(f"duplicate Teacher source/window key: {key}")
        logits[key] = float(logit)
        if expected["label_id"] == 1:
            positive_sources.add(str(expected["source_query_id"]))
            positive_window_rows += 1
    if len(positive_sources) != 140 or positive_window_rows != 334:
        raise RuntimeError(
            "Teacher applied-long coverage drifted from 140 queries / 334 query-windows"
        )
    progress_path = score_path.parent / EVALUATION_PROGRESS_FILE
    if not progress_path.is_file():
        raise RuntimeError("Teacher score output lacks completed progress metadata")
    progress = load_config(progress_path)
    if progress.get("status") != "completed" or progress.get("score_rows") != len(rows):
        raise RuntimeError("Teacher scoring progress is incomplete or inconsistent")
    return logits, {
        "teacher_role": teacher["teacher_role"],
        "score_file": dict(teacher["score_file"]),
        "score_effective_config": dict(teacher["score_effective_config"]),
        "teacher_model_bundle": dict(configured_bundle),
        "frozen_window_manifest": frozen_audit,
        "window_score_rows": len(rows),
        "source_queries": len({key[0] for key in logits}),
        "applied_source_queries": len(positive_sources),
        "applied_query_windows": positive_window_rows,
        "validation": "exact_order_metadata_logits_and_completed_progress",
    }


def _audit_long_cot_evidence_spans(
    query_file: Path,
    source_groups: Sequence[TrajectoryGroup],
    window_manifest: Mapping[str, Any],
    *,
    maximum_failure_ratio: float,
) -> dict[str, Any]:
    """Audit, but never train from, source-owned positive evidence strings."""

    raw = {str(row["query_id"]): row for row in read_jsonl(query_file)}
    groups = {group.trajectory_id: group for group in source_groups}
    manifest_rows = {
        str(row["source_trajectory_id"]): row
        for row in window_manifest.get("trajectories", [])
    }
    rows: list[dict[str, Any]] = []
    for group in source_groups:
        if group.trajectory_id not in manifest_rows:
            raise RuntimeError("evidence audit lacks a long-CoT window manifest row")
        windows = manifest_rows[group.trajectory_id]["windows"]
        for query in group.queries:
            source = raw.get(query.query_id)
            if not isinstance(source, Mapping):
                raise RuntimeError(f"evidence audit lacks source query {query.query_id}")
            if query.label != 1:
                continue
            spans = source.get("evidence_spans", [])
            if not isinstance(spans, list) or not all(isinstance(span, str) for span in spans):
                raise RuntimeError("evidence_spans must be a string list")
            nonempty = [span for span in spans if span]
            occurrences: list[tuple[int, int]] = []
            missing: list[str] = []
            for span in nonempty:
                starts: list[int] = []
                offset = 0
                while True:
                    found = group.model_cot.find(span, offset)
                    if found < 0:
                        break
                    starts.append(found)
                    occurrences.append((found, found + len(span)))
                    offset = found + 1
                if not starts:
                    missing.append(span)
            contained = [
                (start, end)
                for start, end in occurrences
                if any(
                    int(window["cot_char_start"]) <= start
                    and end <= int(window["cot_char_end"])
                    for window in windows
                )
            ]
            if not nonempty:
                classification = "unannotated_manual_review"
            elif missing:
                classification = "span_not_found_manual_review"
            elif contained:
                classification = "single_window_witness"
            else:
                classification = "window_boundary_split_manual_review"
            rows.append(
                {
                    "query_id": query.query_id,
                    "trajectory_id": group.trajectory_id,
                    "family": query.family,
                    "factor_level": query.factor_level,
                    "span_count": len(nonempty),
                    "missing_span_count": len(missing),
                    "contained_occurrences": len(contained),
                    "classification": classification,
                }
            )
    if not rows:
        raise RuntimeError("long-CoT evidence audit found no positive queries")
    annotated = [row for row in rows if row["span_count"] > 0]
    failures = sum(
        row["classification"] != "single_window_witness" for row in annotated
    )
    failure_ratio = failures / len(annotated) if annotated else 1.0
    manual_reviews = sum(
        row["classification"] != "single_window_witness" for row in rows
    )
    report = {
        "schema_version": 1,
        "role": "diagnostic_only_never_used_for_window_selection_or_loss",
        "positive_queries": len(rows),
        "annotated_positive_queries": len(annotated),
        "single_window_witness_queries": sum(
            row["classification"] == "single_window_witness" for row in rows
        ),
        "manual_review_queries": manual_reviews,
        "failure_ratio": failure_ratio,
        "maximum_failure_ratio": maximum_failure_ratio,
        "status": "passed" if failure_ratio <= maximum_failure_ratio else "failed",
        "classifications": dict(
            sorted(Counter(row["classification"] for row in rows).items())
        ),
        "rows": rows,
    }
    if failure_ratio > maximum_failure_ratio:
        raise RuntimeError(
            "long-CoT MIL evidence audit exceeded its preregistered failure ratio: "
            f"{failure_ratio:.6f} > {maximum_failure_ratio:.6f}"
        )
    return report


def _metric_rows(predictions: Any, labels: Any) -> list[dict[str, Any]]:
    import numpy as np

    if not isinstance(labels, (tuple, list)) or len(labels) != 6:
        raise RuntimeError("Trainer did not preserve the six FactorMonitor label tensors")
    label_ids, query_mask, factor_instance, factor_level, family_id, evidence_route = labels
    valid = (
        (np.asarray(query_mask) == 1)
        & (np.asarray(factor_instance) >= 0)
        & (np.asarray(factor_level) >= 0)
        & (np.asarray(family_id) >= 0)
        & (np.asarray(evidence_route) >= 0)
    )
    scores = 1.0 / (1.0 + np.exp(-np.asarray(predictions, dtype=np.float64)))
    rows: list[dict[str, Any]] = []
    for index in zip(*np.nonzero(valid), strict=True):
        rows.append(
            {
                "label_id": int(label_ids[index]),
                "score": float(scores[index]),
                "factor_level_id": int(factor_level[index]),
                "family_id": int(family_id[index]),
                "route_id": int(evidence_route[index]),
            }
        )
    return rows


def _metrics(eval_prediction: Any) -> dict[str, float]:
    rows = _metric_rows(eval_prediction.predictions, eval_prediction.label_ids)
    threshold = select_balanced_threshold(rows)
    summary = score_summary(rows, threshold)
    result: dict[str, float] = {"threshold": threshold}
    for key in ("auroc", "auprc", "balanced_accuracy", "gmean", "gmean2", "f1"):
        value = summary[key]
        if value is not None:
            result[key] = float(value)
    return result


def _causal_choice_metrics(eval_prediction: Any) -> dict[str, float]:
    """Score the natural A/B decision boundary without threshold fitting."""

    rows = _metric_rows(eval_prediction.predictions, eval_prediction.label_ids)
    summary = score_summary(rows, 0.5)
    result: dict[str, float] = {"threshold": 0.5}
    for key in ("auroc", "auprc", "balanced_accuracy", "gmean", "gmean2", "f1"):
        value = summary[key]
        if value is not None:
            result[key] = float(value)
    return result


def _top2_mil_probability(logits: Any, *, pooling: str) -> float:
    """Pool one bag of raw logits with the exact MIL validation semantics."""

    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("MIL pooling requires a non-empty finite one-dimensional bag")
    k = min(2, int(values.size))
    top = np.sort(values)[-k:]
    if pooling == "top2_logmeanexp":
        maximum = float(np.max(top))
        pooled_logit = maximum + math.log(float(np.exp(top - maximum).sum()) / k)
        return 1.0 / (
            1.0 + math.exp(-max(-60.0, min(60.0, pooled_logit)))
        )
    if pooling == "top2_probability_mean":
        return float(
            (1.0 / (1.0 + np.exp(-np.clip(top, -60.0, 60.0)))).mean()
        )
    raise ValueError(f"unsupported MIL pooling: {pooling}")


def _mil_metrics(
    eval_prediction: Any,
    *,
    pooling: str,
    short_validation_auroc_floor: float,
) -> dict[str, float]:
    """Aggregate every validation window back to one original Factor query."""

    import numpy as np

    labels = eval_prediction.label_ids
    if not isinstance(labels, (tuple, list)) or len(labels) != 7:
        raise RuntimeError("MIL validation did not preserve its seven label tensors")
    (
        label_ids,
        query_mask,
        factor_level,
        family_id,
        evidence_route,
        source_query_index,
        is_long,
    ) = (np.asarray(value) for value in labels)
    predictions = np.asarray(eval_prediction.predictions, dtype=np.float64)
    valid = (
        (query_mask == 1)
        & (factor_level >= 0)
        & (family_id >= 0)
        & (evidence_route >= 0)
        & (source_query_index >= 0)
        & (is_long >= 0)
    )
    grouped: dict[int, list[tuple[float, int, int, int, int, int]]] = {}
    for index in zip(*np.nonzero(valid), strict=True):
        source = int(source_query_index[index])
        grouped.setdefault(source, []).append(
            (
                float(predictions[index]),
                int(label_ids[index]),
                int(factor_level[index]),
                int(family_id[index]),
                int(evidence_route[index]),
                int(is_long[index]),
            )
        )
    rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        values = grouped[source]
        metadata = {value[1:] for value in values}
        if len(metadata) != 1:
            raise RuntimeError("one MIL validation bag changed source metadata")
        logits = np.asarray([value[0] for value in values], dtype=np.float64)
        try:
            score = _top2_mil_probability(logits, pooling=pooling)
        except ValueError as exc:
            raise RuntimeError(f"unsupported MIL validation pooling: {pooling}") from exc
        label, level, family, route, long_flag = next(iter(metadata))
        rows.append(
            {
                "label_id": label,
                "score": score,
                "factor_level_id": level,
                "family_id": family,
                "route_id": route,
                "is_long": long_flag,
            }
        )
    result: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, selected in (
        ("short", [row for row in rows if row["is_long"] == 0]),
        ("long", [row for row in rows if row["is_long"] == 1]),
    ):
        if not selected:
            raise RuntimeError(f"MIL validation contains no {name} source queries")
        threshold = select_balanced_threshold(selected)
        summary = score_summary(selected, threshold)
        summaries[name] = summary
        result[f"{name}_threshold"] = float(threshold)
        for metric in ("auroc", "auprc", "balanced_accuracy", "gmean", "gmean2", "f1"):
            value = summary.get(metric)
            if value is not None:
                result[f"{name}_{metric}"] = float(value)
    short_auroc = result.get("short_auroc")
    long_auroc = result.get("long_auroc")
    if short_auroc is None or long_auroc is None:
        raise RuntimeError("MIL validation requires defined short and long AUROC")
    result["selection_score"] = (
        long_auroc
        if short_auroc >= short_validation_auroc_floor
        else long_auroc - 10.0
    )
    # The canonical direct threshold remains the short-context threshold.
    result["threshold"] = result["short_threshold"]
    result["auroc"] = short_auroc
    return result


def _trainer_class(dependencies: Mapping[str, Any]) -> type[Any]:
    torch = dependencies["torch"]
    Trainer = dependencies["Trainer"]
    DataLoader = dependencies["DataLoader"]

    class FactorMonitorTrainer(Trainer):
        label_names = [
            "labels",
            "query_mask",
            "factor_instance_index",
            "factor_level",
            "family_id",
            "evidence_route",
        ]

        def __init__(
            self,
            *args: Any,
            factor_batch_sampler: Any,
            factor_eval_batch_sampler: Any | None = None,
            eval_data_collator: Any,
            bundle_context: Mapping[str, Any],
            mil_mining_state: LongCotMilMiningState | None = None,
            mil_refresh_groups: Sequence[TrajectoryGroup] = (),
            mil_refresh_collator: Any | None = None,
            **kwargs: Any,
        ) -> None:
            self.factor_batch_sampler = factor_batch_sampler
            self.factor_eval_batch_sampler = factor_eval_batch_sampler
            self.eval_data_collator = eval_data_collator
            self.bundle_context = dict(bundle_context)
            self.mil_mining_state = mil_mining_state
            self.mil_refresh_groups = tuple(mil_refresh_groups)
            self.mil_refresh_collator = mil_refresh_collator
            self.teacher_metric_accumulator = {
                "microbatches": 0,
                "teacher_kl_loss": 0.0,
                "weighted_teacher_kl_loss": 0.0,
                "eligible_queries": 0,
                "eligible_units": 0,
            }
            self.memory_telemetry_enabled = (
                os.environ.get("FACTOR_MONITOR_MEMORY_TELEMETRY", "0") == "1"
            )
            self.memory_telemetry_rank = int(os.environ.get("LOCAL_RANK", "0"))
            super().__init__(*args, **kwargs)
            effective_config = self.bundle_context.get("effective_config", {})
            monitor_config = effective_config.get("monitor", {})
            mil = monitor_config.get("backend_options", {}).get("long_cot_mil")
            self.label_names = (
                [
                    "labels",
                    "query_mask",
                    "factor_level",
                    "family_id",
                    "evidence_route",
                    "mil_source_query_index",
                    "mil_is_long",
                ]
                if mil is not None
                else list(type(self).label_names)
            )

        def _memory_input_summary(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
            summary: dict[str, Any] = {}
            for key in (
                "endpoint_input_ids",
                "joint_input_ids",
                "evidence_input_ids",
                "factor_input_ids",
            ):
                value = inputs.get(key)
                if value is not None and hasattr(value, "shape"):
                    summary[key] = {
                        "shape": [int(item) for item in value.shape],
                        "numel": int(value.numel()),
                    }
            for key in ("endpoint_attention_mask", "joint_attention_mask", "evidence_attention_mask"):
                value = inputs.get(key)
                if value is not None and hasattr(value, "sum"):
                    summary[f"{key}_attended_tokens"] = int(value.sum().item())
            for key in ("mil_window_index", "mil_bag_index", "mil_is_long"):
                value = inputs.get(key)
                if value is not None and hasattr(value, "numel"):
                    summary[f"{key}_valid_values"] = int((value >= 0).sum().item())
            return summary

        def _write_memory_record(self, record: Mapping[str, Any]) -> None:
            if not self.memory_telemetry_enabled:
                return
            path = Path(self.args.output_dir) / (
                f"memory_telemetry_rank{self.memory_telemetry_rank}.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")

        def training_step(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            num_items_in_batch: Any | None = None,
        ) -> Any:
            enabled = self.memory_telemetry_enabled and torch.cuda.is_available()
            started = time.perf_counter()
            if enabled:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated_before = int(torch.cuda.memory_allocated())
                reserved_before = int(torch.cuda.memory_reserved())
            else:
                allocated_before = reserved_before = None
            error: str | None = None
            try:
                return super().training_step(
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except RuntimeError as exc:
                error = type(exc).__name__ + ": " + str(exc)[:500]
                raise
            finally:
                if enabled:
                    try:
                        torch.cuda.synchronize()
                        record = {
                            "schema_version": 1,
                            "record_type": "training_step_memory",
                            "rank": self.memory_telemetry_rank,
                            "global_step": int(self.state.global_step),
                            "elapsed_seconds": time.perf_counter() - started,
                            "allocated_before": allocated_before,
                            "reserved_before": reserved_before,
                            "peak_allocated": int(torch.cuda.max_memory_allocated()),
                            "peak_reserved": int(torch.cuda.max_memory_reserved()),
                            "error": error,
                            "inputs": self._memory_input_summary(inputs),
                        }
                        self._write_memory_record(record)
                    except Exception:
                        # Telemetry must never change training semantics or hide
                        # the original OOM/forward exception.
                        pass

        def _persist_mil_cache(self, checkpoint: Path | None = None) -> None:
            if self.mil_mining_state is None or not self.is_world_process_zero():
                return
            snapshot = self.mil_mining_state.snapshot()
            audit = self.mil_mining_state.audit()
            write_json(Path(self.args.output_dir) / "mil_mining_cache.json", snapshot)
            write_json(Path(self.args.output_dir) / "mil_mining_cache_audit.json", audit)
            if checkpoint is not None:
                if not checkpoint.is_dir():
                    raise RuntimeError(f"MIL checkpoint directory is missing: {checkpoint}")
                write_json(checkpoint / "mil_mining_cache.json", snapshot)
                write_json(checkpoint / "mil_mining_cache_audit.json", audit)

        def _refresh_mil_cache_if_due(self) -> bool:
            if self.mil_mining_state is None or not self.mil_refresh_groups:
                return False
            mil = self.bundle_context["effective_config"]["monitor"][
                "backend_options"
            ]["long_cot_mil"]
            if mil["replay_only"] or self.state.global_step >= self.state.max_steps:
                return False
            cadence = mil["optimizer_steps_per_epoch"]
            if self.state.global_step <= 0 or self.state.global_step % cadence:
                return False
            model = self.accelerator.unwrap_model(self.model)
            rows = _evaluate_batches(
                model,
                self.mil_refresh_groups,
                self.mil_refresh_collator,
                batch_size=1,
                device=self.args.device,
            )
            self.mil_mining_state.update_rows(rows, full_refresh=True)
            model.train()
            return True

        def get_train_dataloader(self) -> Any:
            if self.train_dataset is None:
                raise ValueError("FactorMonitor training requires train_dataset")
            parameters: dict[str, Any] = {
                "batch_sampler": self.factor_batch_sampler,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }
            if self.args.dataloader_num_workers > 0:
                parameters["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader = DataLoader(self.train_dataset, **parameters)
            if getattr(self.factor_batch_sampler, "batch_size", None) is None:
                # Contrast units have variable dataset-item counts but the
                # sampler already emits an equal number of synchronized rank
                # batches with exactly two trajectory slots each.
                previous_even_batches = self.accelerator.even_batches
                self.accelerator.even_batches = False
                try:
                    return self.accelerator.prepare(dataloader)
                finally:
                    self.accelerator.even_batches = previous_even_batches
            return self.accelerator.prepare(dataloader)

        def get_eval_dataloader(self, eval_dataset: Any | None = None) -> Any:
            # Trainer exposes one collator for both splits.  Training keeps
            # trajectory-local dynamic query padding, while distributed
            # validation must expose one fixed record dimension on every rank
            # so Accelerate's recursive pad/gather sequence is identical.
            selected_dataset = (
                eval_dataset
                if eval_dataset is not None
                else getattr(self, "eval_dataset", None)
            )
            if (
                self.factor_eval_batch_sampler is not None
                and selected_dataset is self.eval_dataset
            ):
                parameters: dict[str, Any] = {
                    "batch_sampler": self.factor_eval_batch_sampler,
                    "collate_fn": self.eval_data_collator,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "persistent_workers": self.args.dataloader_persistent_workers,
                }
                if self.args.dataloader_num_workers > 0:
                    parameters["prefetch_factor"] = self.args.dataloader_prefetch_factor
                return self.accelerator.prepare(DataLoader(selected_dataset, **parameters))
            train_collator = self.data_collator
            self.data_collator = self.eval_data_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = train_collator

        def _maybe_log_save_evaluate(
            self,
            tr_loss: Any,
            grad_norm: Any,
            model: Any,
            trial: Any,
            epoch: float,
            ignore_keys_for_eval: list[str] | None,
            start_time: float,
            learning_rate: float | None = None,
        ) -> None:
            should_log = bool(self.control.should_log)
            if not (self.control.should_save and self.control.should_evaluate):
                should_save = bool(self.control.should_save)
                result = super()._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    model,
                    trial,
                    epoch,
                    ignore_keys_for_eval,
                    start_time,
                    learning_rate,
                )
                refreshed = self._refresh_mil_cache_if_due()
                if should_save:
                    checkpoint = (
                        Path(self._get_output_dir(trial))
                        / f"checkpoint-{self.state.global_step}"
                    )
                    self._persist_mil_cache(checkpoint)
                elif refreshed:
                    self._persist_mil_cache()
                if should_log:
                    self._flush_teacher_metrics()
                return result

            # Transformers evaluates before saving when both events share a
            # cadence.  Persist the resumable checkpoint first, then evaluate.
            # If evaluation fails, the pre-eval Trainer state deliberately
            # remains usable; after success, refresh only Trainer state with
            # the newly selected best metric/checkpoint.
            self.control.should_evaluate = False
            super()._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate,
            )
            self.control.should_evaluate = True
            self.control.should_save = False
            super()._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate,
            )
            checkpoint = (
                Path(self._get_output_dir(trial))
                / f"checkpoint-{self.state.global_step}"
            )
            if self.state.best_global_step == self.state.global_step:
                self.state.best_model_checkpoint = str(checkpoint)
            if self.args.should_save:
                state_path = checkpoint / "trainer_state.json"
                if not state_path.is_file():
                    raise RuntimeError(
                        f"pre-evaluation checkpoint lacks Trainer state: {state_path}"
                    )
                self.state.save_to_json(str(state_path))
            self._refresh_mil_cache_if_due()
            self._persist_mil_cache(checkpoint)
            self._flush_teacher_metrics()

        def _flush_teacher_metrics(self) -> None:
            accumulator = self.teacher_metric_accumulator
            microbatches = int(accumulator["microbatches"])
            if microbatches <= 0:
                return
            self.log(
                {
                    "teacher_kl_loss": accumulator["teacher_kl_loss"] / microbatches,
                    "weighted_teacher_kl_loss": (
                        accumulator["weighted_teacher_kl_loss"] / microbatches
                    ),
                    "teacher_eligible_queries": accumulator["eligible_queries"],
                    "teacher_eligible_units": accumulator["eligible_units"],
                    "teacher_temperature": 2.0,
                    "teacher_loss_weight": 0.25,
                }
            )
            accumulator.update(
                {
                    "microbatches": 0,
                    "teacher_kl_loss": 0.0,
                    "weighted_teacher_kl_loss": 0.0,
                    "eligible_queries": 0,
                    "eligible_units": 0,
                }
            )

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            batch = dict(inputs)
            for key in ("trajectory_id", "leakage_group_id", "query_id"):
                batch.pop(key, None)
            batch.pop("training_unit_id", None)
            source_query_ids = batch.pop("source_query_id", None)
            labels = batch.pop("labels")
            record_loss_weight = batch.pop("loss_weight")
            instance = batch.pop("factor_instance_index")
            level = batch.pop("factor_level")
            family = batch.pop("family_id")
            training_unit = batch.pop("training_unit_index", None)
            metric_weighting = self.bundle_context["effective_config"]["monitor"][
                "backend_options"
            ].get("metric_driven_weighting")
            if metric_weighting is not None:
                architecture = self.bundle_context["effective_config"]["monitor"][
                    "architecture"
                ]
                family_names = architecture["phase1_families"]
                level_names = architecture["factor_levels"]
                level_weights = metric_weighting["resolved"]["level_loss_weights"]
                family_weights = metric_weighting["resolved"][
                    "family_sampling_weights"
                ]
                mil_weighting = (
                    self.bundle_context["effective_config"]["monitor"][
                        "backend_options"
                    ].get("metric_mil_loss_weighting", "none")
                    if training_unit is not None
                    and self.bundle_context["effective_config"]["monitor"][
                        "backend_options"
                    ].get("batching", {}).get("training_unit")
                    == "long_cot_mil"
                    else None
                )
                for family_name, weights in level_weights.items():
                    family_index = family_names.index(family_name)
                    for level_name, weight in weights.items():
                        level_index = level_names.index(level_name)
                        selector = (family == family_index) & (level == level_index)
                        multiplier = float(weight)
                        if mil_weighting == "factor_instance":
                            multiplier *= float(family_weights[family_name])
                        record_loss_weight = torch.where(
                            selector,
                            record_loss_weight * multiplier,
                            record_loss_weight,
                        )
            training_unit_weight = batch.pop("training_unit_weight", None)
            mil_bag = batch.pop("mil_bag_index", None)
            mil_instance = batch.pop("mil_factor_instance_index", None)
            mil_source_index = batch.pop("mil_source_query_index", None)
            mil_window_index = batch.pop("mil_window_index", None)
            mil_is_long = batch.pop("mil_is_long", None)
            mil_teacher_logit = batch.pop("mil_teacher_logit", None)
            label_weighting = self.bundle_context["effective_config"]["monitor"].get(
                "backend_options", {}
            ).get("label_weighting", "none")
            loss_weight = (
                record_loss_weight
                if label_weighting == "record_loss_weight"
                else None
            )
            model_variant = _model_variant(self.bundle_context["effective_config"])
            if model_variant in {"endpoint_early_fusion", "causal_lm_choice"}:
                # Preserve the canonical route as evaluation metadata so the
                # shared metric contract remains six tensors, but never expose
                # it to the M0 endpoint model as an input feature.
                batch.pop("evidence_route")
            if model_variant == "causal_lm_choice":
                logits, per_query_loss = model(**batch, choice_labels=labels)
            else:
                logits = model(**batch)
                per_query_loss = None
            mil = self.bundle_context["effective_config"]["monitor"].get(
                "backend_options", {}
            ).get("long_cot_mil")
            if model_variant == "causal_lm_choice":
                if mil is not None:
                    raise RuntimeError(
                        "causal choice SFT does not support long-CoT MIL batches"
                    )
                if training_unit is None:
                    values = hierarchical_query_loss(
                        per_query_loss,
                        batch["query_mask"],
                        instance,
                        level,
                        family,
                        loss_weight,
                    )
                else:
                    configured_loss = self.bundle_context["effective_config"][
                        "monitor"
                    ]["backend_options"]["loss_aggregation"]
                    if configured_loss != "factor_instance_uniform":
                        raise RuntimeError(
                            "causal choice SFT requires factor_instance_uniform loss"
                        )
                    values = factor_instance_uniform_query_loss(
                        per_query_loss,
                        batch["query_mask"],
                        instance,
                        level,
                        family,
                        training_unit,
                        loss_weight,
                    )
            elif mil is not None:
                if any(
                    value is None
                    for value in (
                        training_unit,
                        mil_bag,
                        mil_instance,
                        mil_source_index,
                        mil_window_index,
                        mil_is_long,
                        source_query_ids,
                    )
                ):
                    raise RuntimeError("MIL batch lacks source hierarchy metadata")
                teacher = mil.get("teacher_distillation")
                if teacher is not None and model.training and mil_teacher_logit is None:
                    raise RuntimeError("Teacher-distilled MIL training lacks Teacher logits")
                if teacher is None and mil_teacher_logit is not None:
                    raise RuntimeError("non-distilled MIL batch unexpectedly carries Teacher logits")
                teacher_training = teacher is not None and model.training
                values = mil_factor_instance_uniform_binary_loss(
                    logits,
                    labels,
                    batch["query_mask"],
                    mil_bag,
                    mil_instance,
                    level,
                    family,
                    training_unit,
                    pooling=mil["pooling"],
                    loss_weight=loss_weight,
                    training_unit_weight=training_unit_weight,
                    teacher_logits=(mil_teacher_logit if teacher_training else None),
                    mil_is_long=(mil_is_long if teacher_training else None),
                    teacher_temperature=(
                        teacher["temperature"] if teacher_training else None
                    ),
                    teacher_loss_weight=(
                        teacher["loss_weight"] if teacher_training else None
                    ),
                )
                if teacher_training:
                    accumulator = self.teacher_metric_accumulator
                    accumulator["microbatches"] += 1
                    accumulator["teacher_kl_loss"] += float(
                        values["teacher_kl_loss"].item()
                    )
                    accumulator["weighted_teacher_kl_loss"] += float(
                        values["weighted_teacher_kl_loss"].item()
                    )
                    accumulator["eligible_queries"] += int(
                        values["teacher_eligible_query_count"]
                    )
                    accumulator["eligible_units"] += int(
                        values["teacher_eligible_unit_count"]
                    )
                if self.mil_mining_state is not None:
                    active_query_mask = batch["query_mask"]
                    if training_unit_weight is not None:
                        active_query_mask = active_query_mask & (
                            training_unit_weight.to(
                                training_unit.device
                            )[training_unit.clamp_min(0)]
                            > 0
                        )
                    self.mil_mining_state.update_batch(
                        source_query_ids,
                        mil_window_index,
                        logits,
                        active_query_mask,
                    )
            elif training_unit is None:
                values = hierarchical_binary_loss(
                    logits,
                    labels,
                    batch["query_mask"],
                    instance,
                    level,
                    family,
                    loss_weight,
                )
            else:
                configured_loss = self.bundle_context["effective_config"]["monitor"][
                    "backend_options"
                ]["loss_aggregation"]
                if configured_loss != "factor_instance_uniform":
                    raise RuntimeError(
                        "contrast-unit batch requires factor_instance_uniform loss"
                    )
                values = factor_instance_uniform_binary_loss(
                    logits,
                    labels,
                    batch["query_mask"],
                    instance,
                    level,
                    family,
                    training_unit,
                    loss_weight,
                )
            outputs = {"logits": logits, "loss": values["loss"]}
            return (values["loss"], outputs) if return_outputs else values["loss"]

        def create_optimizer(self) -> Any:
            if self.optimizer is not None:
                return self.optimizer
            lora_parameters = []
            custom_parameters = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if "lora_" in name:
                    lora_parameters.append(parameter)
                elif name.startswith(self.model.CUSTOM_PREFIXES):
                    custom_parameters.append(parameter)
                else:
                    raise RuntimeError(f"unexpected trainable parameter in optimizer: {name}")
            causal_choice = (
                _model_variant(self.bundle_context["effective_config"])
                == "causal_lm_choice"
            )
            if not lora_parameters:
                raise RuntimeError("optimizer requires a non-empty LoRA parameter group")
            if causal_choice and custom_parameters:
                raise RuntimeError(
                    "causal choice optimizer must not contain custom trainable parameters"
                )
            if not causal_choice and not custom_parameters:
                raise RuntimeError(
                    "FactorMonitor optimizer requires custom trainable parameters"
                )
            optimization = self.bundle_context["effective_config"]["optimization"]
            options = self.bundle_context["effective_config"]["monitor"]["backend_options"]
            if optimization["optimizer"] != "adamw_torch":
                raise RuntimeError("backend currently implements only optimizer=adamw_torch")
            parameter_groups = [
                {"params": lora_parameters, "lr": optimization["learning_rate"]}
            ]
            if custom_parameters:
                parameter_groups.append(
                    {
                        "params": custom_parameters,
                        "lr": options["new_module_learning_rate"],
                    }
                )
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=optimization["weight_decay"],
            )
            return self.optimizer

        def _load_from_checkpoint(
            self, resume_from_checkpoint: str, model: Any | None = None
        ) -> None:
            target_model = self.model if model is None else model
            bundle = Path(resume_from_checkpoint) / "model_bundle"
            restored = load_model_bundle(target_model.backbone, bundle)
            if (
                restored.architecture.source_hash
                != target_model.architecture.source_hash
            ):
                raise RuntimeError("resume architecture differs from the current run")
            target_model.load_custom_state_dict(restored.custom_state_dict())
            adapter_state = dependencies["load_peft_weights"](
                str(bundle / "backbone"), device="cpu"
            )
            result = dependencies["set_peft_model_state_dict"](
                target_model.backbone, adapter_state, adapter_name="default"
            )
            unexpected = getattr(result, "unexpected_keys", ())
            if unexpected:
                raise RuntimeError(
                    f"unexpected LoRA keys while resuming: {sorted(unexpected)}"
                )

        def _load_best_model(self) -> None:
            checkpoint = self.state.best_model_checkpoint
            if not checkpoint:
                raise RuntimeError("Trainer selected no best FactorMonitor checkpoint")
            self._load_from_checkpoint(checkpoint)

        def _save(self, output_dir: str | None = None, state_dict: Any | None = None) -> None:
            del state_dict
            target = Path(output_dir or self.args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            bundle = target / "model_bundle"
            if bundle.exists():
                raise FileExistsError(f"refusing to overwrite model bundle: {bundle}")
            save_model_bundle(
                self.model,
                bundle,
                effective_config=self.bundle_context["effective_config"],
                dataset_manifest=self.bundle_context["dataset_manifest"],
                metrics={
                    "global_step": self.state.global_step,
                    "latest_log": self.state.log_history[-1]
                    if self.state.log_history
                    else None,
                },
                tokenizer=self.processing_class,
                save_backbone=True,
            )
            torch.save(self.args, target / "training_args.bin")

    return FactorMonitorTrainer


def _training_arguments(effective: Mapping[str, Any], dependencies: Mapping[str, Any]) -> Any:
    optimization = effective["optimization"]
    evaluation = effective["evaluation"]
    precision = optimization["mixed_precision"]
    evaluate_during_training = evaluation.get("during_training", True)
    kwargs = {
        "output_dir": effective["output_directory"],
        "per_device_train_batch_size": optimization["batch_size"],
        "per_device_eval_batch_size": optimization["eval_batch_size"],
        "gradient_accumulation_steps": optimization["gradient_accumulation_steps"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
        "warmup_ratio": optimization["warmup_ratio"],
        "max_grad_norm": optimization["max_grad_norm"],
        "num_train_epochs": optimization["epochs"] or 1.0,
        "max_steps": optimization["max_steps"] if optimization["max_steps"] is not None else -1,
        "lr_scheduler_type": optimization["scheduler"],
        "logging_steps": optimization["logging_steps"],
        "eval_strategy": "steps" if evaluate_during_training else "no",
        "save_strategy": "steps",
        "save_steps": optimization["save_steps"],
        "save_total_limit": optimization.get("save_total_limit", 2),
        "load_best_model_at_end": evaluate_during_training,
        "seed": effective["runtime"]["seed"],
        "data_seed": effective["runtime"]["seed"],
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "remove_unused_columns": False,
        "dataloader_drop_last": False,
        "report_to": [],
        "ddp_find_unused_parameters": (
            effective["monitor"]["backend_options"]["batching"]["distributed_unused_route_policy"]
            == "find_unused_parameters_true"
        ),
        "gradient_checkpointing": optimization["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {
            "use_reentrant": optimization["gradient_checkpointing_use_reentrant"]
        },
    }
    if evaluate_during_training:
        kwargs.update(
            {
                "eval_steps": evaluation["cadence"],
                "metric_for_best_model": evaluation["metric"],
                "greater_is_better": evaluation["metric"] != "loss",
            }
        )
    patience = evaluation.get("early_stopping_patience")
    if patience is not None:
        raise RuntimeError(
            "early_stopping_patience is recorded but callback wiring is not yet enabled; set null"
        )
    return dependencies["TrainingArguments"](**kwargs)


def _pending_validation_recovery_checkpoint(
    checkpoint_value: str | None,
    *,
    max_steps: int | None,
    metric: str,
) -> Path | None:
    """Identify a pre-evaluation checkpoint at the configured final step."""

    if checkpoint_value is None or max_steps is None:
        return None
    checkpoint = Path(checkpoint_value)
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"resume checkpoint lacks Trainer state: {state_path}")
    state = load_config(state_path)
    global_step = state.get("global_step")
    if global_step != max_steps:
        return None
    metric_key = metric if metric.startswith("eval_") else f"eval_{metric}"
    completed_rows = [
        row
        for row in state.get("log_history", [])
        if isinstance(row, Mapping)
        and row.get("step") == global_step
        and metric_key in row
    ]
    if completed_rows:
        raise RuntimeError(
            "final resume checkpoint already contains validation metrics; "
            "refusing to execute an extra training step"
        )
    return checkpoint


def _run_pending_validation_only(
    trainer: Any,
    checkpoint: Path,
    *,
    immutable_checkpoint: bool = False,
    metric: str | None = None,
) -> dict[str, Any]:
    """Restore final-step weights and Trainer state, then run only validation."""

    print(
        json.dumps(
            {
                "event": "pending_validation_recovery",
                "checkpoint": str(checkpoint),
                "action": "validation_only_no_training",
                "source_policy": "immutable" if immutable_checkpoint else "in_place_recovery",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    trainer._load_from_checkpoint(str(checkpoint))
    trainer.state = type(trainer.state).load_from_json(
        str(checkpoint / "trainer_state.json")
    )
    trainer.state.is_local_process_zero = trainer.is_local_process_zero()
    trainer.state.is_world_process_zero = trainer.is_world_process_zero()
    metrics = dict(trainer.evaluate())
    recovery_mode = (
        "explicit_validation_only"
        if immutable_checkpoint
        else "pending_validation_only"
    )
    result_metrics = {"recovery_mode": recovery_mode, **metrics}
    metrics_output = getattr(trainer.args, "output_dir", None)
    if trainer.args.should_save and metrics_output is not None:
        write_json(Path(metrics_output) / "train_metrics.json", result_metrics)
    if immutable_checkpoint:
        if metric is None:
            raise RuntimeError("explicit validation-only recovery requires a metric")
        metric_key = metric if metric.startswith("eval_") else f"eval_{metric}"
        metric_value = metrics.get(metric_key)
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
        ):
            raise RuntimeError(
                f"explicit validation did not produce a finite {metric_key}"
            )
        trainer.state.best_metric = float(metric_value)
        trainer.state.best_global_step = trainer.state.global_step
    elif not trainer._determine_best_metric(metrics, None):
        raise RuntimeError("pending validation did not select the final checkpoint")
    trainer.state.best_model_checkpoint = str(checkpoint)
    if trainer.args.should_save and not immutable_checkpoint:
        trainer.state.save_to_json(str(checkpoint / "trainer_state.json"))
    if not immutable_checkpoint:
        trainer.save_model()
    trainer.save_state()
    return result_metrics


def run_training(effective: dict[str, Any]) -> dict[str, Any]:
    dependencies = _dependencies()
    dependencies["set_seed"](effective["runtime"]["seed"])
    tokenizer = _load_tokenizer(effective, dependencies)
    effective = _contract_enriched_config(effective, tokenizer)
    validation_only = effective["evaluation"].get("validation_only")
    if validation_only is not None:
        recovery_checkpoint = Path(validation_only["checkpoint"])
    else:
        recovery_checkpoint = _pending_validation_recovery_checkpoint(
            effective["optimization"]["resume_from_checkpoint"],
            max_steps=effective["optimization"]["max_steps"],
            metric=effective["evaluation"]["metric"],
        )
    output = Path(effective["output_directory"])
    if is_world_process_zero():
        write_json(output / "effective_config.json", effective)
    architecture = architecture_from_dict(effective["monitor"]["architecture"])
    train_groups = _load_groups(effective, split="train")
    validation_groups = _load_groups(effective, split="validation")
    scope = effective["monitor"]["backend_options"]["training_scope"]
    expected_families = set(scope["families"])
    for split, groups in (("training", train_groups), ("validation", validation_groups)):
        present_families = {group.family for group in groups}
        if present_families != expected_families:
            raise RuntimeError(
                f"{split} split families differ from training_scope; "
                f"expected={sorted(expected_families)}, found={sorted(present_families)}"
            )
    boundaries = effective["monitor"]["backend_options"]["sequence"]["length_buckets"]
    validation_query_counts = Counter(len(group.queries) for group in validation_groups)
    validation_pad_queries = max(validation_query_counts)
    base_train_collator = _collator(effective, tokenizer)
    base_eval_collator = _collator(
        effective,
        tokenizer,
        pad_queries_to=validation_pad_queries,
    )
    training_unit = effective["monitor"]["backend_options"]["batching"][
        "training_unit"
    ]
    contrast_audit: dict[str, Any] | None = None
    mil_options = effective["monitor"]["backend_options"].get("long_cot_mil")
    mil_mining_state: LongCotMilMiningState | None = None
    mil_refresh_groups: tuple[TrajectoryGroup, ...] = ()
    mil_refresh_collator: Any | None = None
    mil_manifests: dict[str, Any] | None = None
    teacher_distillation_logits: dict[tuple[str, int], float] | None = None
    teacher_distillation_audit: dict[str, Any] | None = None
    compute_metrics = (
        _causal_choice_metrics
        if _model_variant(effective) == "causal_lm_choice"
        else _metrics
    )
    validation_schedule_audit: dict[str, Any] | None = None
    if training_unit == "long_cot_mil":
        if mil_options is None:
            raise RuntimeError("long_cot_mil training unit lacks MIL options")
        contract_id = effective["monitor"]["backend_options"][
            "endpoint_encoding_contract_id"
        ]
        long_train_groups, train_windows, train_window_manifest = _long_cot_window_groups(
            tokenizer,
            train_groups,
            contract_id=contract_id,
            trigger_endpoint_tokens=mil_options["trigger_endpoint_tokens"],
            max_endpoint_tokens=mil_options["max_endpoint_tokens"],
            cot_overlap_tokens=mil_options["cot_overlap_tokens"],
            context_policy=mil_options["context_policy"],
            overlap_fallback_policy=mil_options["overlap_fallback_policy"],
        )
        long_validation_groups, validation_windows, validation_window_manifest = (
            _long_cot_window_groups(
                tokenizer,
                validation_groups,
                contract_id=contract_id,
                trigger_endpoint_tokens=mil_options["trigger_endpoint_tokens"],
                max_endpoint_tokens=mil_options["max_endpoint_tokens"],
                cot_overlap_tokens=mil_options["cot_overlap_tokens"],
                context_policy=mil_options["context_policy"],
                overlap_fallback_policy=mil_options["overlap_fallback_policy"],
            )
        )
        long_train_ids = {group.trajectory_id for group in long_train_groups}
        long_validation_ids = {group.trajectory_id for group in long_validation_groups}
        short_train_groups = [
            group for group in train_groups if group.trajectory_id not in long_train_ids
        ]
        short_validation_groups = [
            group
            for group in validation_groups
            if group.trajectory_id not in long_validation_ids
        ]
        dataset_paths = _dataset_paths(effective)
        train_evidence_audit = _audit_long_cot_evidence_spans(
            _query_split_path(dataset_paths, "train"),
            long_train_groups,
            train_window_manifest,
            maximum_failure_ratio=mil_options["evidence_audit_max_failure_ratio"],
        )
        validation_evidence_audit = _audit_long_cot_evidence_spans(
            _query_split_path(dataset_paths, "validation"),
            long_validation_groups,
            validation_window_manifest,
            maximum_failure_ratio=mil_options["evidence_audit_max_failure_ratio"],
        )
        mil_split_lengths = _validate_token_limits(
            effective,
            tokenizer,
            {
                "train_short": short_train_groups,
                "train_windows": train_windows,
                "validation_short": short_validation_groups,
                "validation_windows": validation_windows,
            },
        )
        # MIL bags are trajectory-local: a complete Factor contrast pair must
        # not be co-scheduled into one bag because build_long_cot_mil_units
        # attaches windows to exactly one source Factor-instance.
        train_long_sources, train_long_audit = build_factor_instance_contrast_units(
            long_train_groups,
            co_schedule_complete_contrasts=False,
        )
        train_short_units, train_short_audit = build_factor_instance_contrast_units(
            short_train_groups
        )
        validation_long_sources, validation_long_audit = (
            build_factor_instance_contrast_units(
                long_validation_groups,
                co_schedule_complete_contrasts=False,
            )
        )
        validation_short_units, validation_short_audit = (
            build_factor_instance_contrast_units(
                short_validation_groups,
                co_schedule_complete_contrasts=False,
            )
        )
        train_long_units = build_long_cot_mil_units(train_long_sources, train_windows)
        validation_long_units = build_long_cot_mil_units(
            validation_long_sources, validation_windows
        )
        if mil_options["training_schedule"] == "full_epoch_joint":
            resolved_long_units = len(train_long_units)
            resolved_short_units = len(train_short_units)
            resolved_steps_per_epoch = (
                resolved_long_units + resolved_short_units + 3
            ) // 4
            configured_counts = {
                "expected_long_units": mil_options["expected_long_units"],
                "expected_short_units": mil_options["expected_short_units"],
                "sampled_long_units_per_exposure": mil_options[
                    "sampled_long_units_per_exposure"
                ],
                "optimizer_steps_per_epoch": mil_options[
                    "optimizer_steps_per_epoch"
                ],
            }
            resolved_counts = {
                "expected_long_units": resolved_long_units,
                "expected_short_units": resolved_short_units,
                "sampled_long_units_per_exposure": resolved_long_units,
                "optimizer_steps_per_epoch": resolved_steps_per_epoch,
            }
            for name, configured in configured_counts.items():
                if configured != "auto" and int(configured) != resolved_counts[name]:
                    raise RuntimeError(
                        f"long_cot_mil.{name} disagrees with materialized data: "
                        f"configured={configured}, resolved={resolved_counts[name]}"
                    )
            mil_options = {**mil_options, **resolved_counts}
            effective["monitor"]["backend_options"]["long_cot_mil"] = mil_options
            if effective["optimization"].get("save_steps") == "auto":
                effective["optimization"]["save_steps"] = resolved_steps_per_epoch
            if effective["evaluation"].get("cadence") == "auto":
                effective["evaluation"]["cadence"] = resolved_steps_per_epoch
            if is_world_process_zero():
                write_json(output / "effective_config.json", effective)
        train_dataset = LongCotMilDataset(train_long_units, train_short_units)
        eval_dataset = LongCotMilDataset(validation_long_units, validation_short_units)
        train_window_lengths = {
            group.trajectory_id: length
            for group, length in zip(
                train_windows,
                mil_split_lengths["train_windows"],
                strict=True,
            )
        }
        train_short_lengths = {
            group.trajectory_id: length
            for group, length in zip(
                short_train_groups,
                mil_split_lengths["train_short"],
                strict=True,
            )
        }
        train_unit_lengths: list[int] = []
        train_unit_costs: list[int] = []
        for unit in train_long_units:
            unit_lengths = [
                train_window_lengths[group.trajectory_id]
                for group in unit.window_groups
            ]
            train_unit_lengths.append(max(unit_lengths))
            train_unit_costs.append(sum(unit_lengths))
        for unit in train_short_units:
            unit_lengths = [
                train_short_lengths[group.trajectory_id]
                for group in unit.groups
            ]
            train_unit_lengths.append(max(unit_lengths))
            train_unit_costs.append(sum(unit_lengths))
        validation_window_lengths = {
            group.trajectory_id: length
            for group, length in zip(
                validation_windows,
                mil_split_lengths["validation_windows"],
                strict=True,
            )
        }
        validation_short_lengths = {
            group.trajectory_id: length
            for group, length in zip(
                short_validation_groups,
                mil_split_lengths["validation_short"],
                strict=True,
            )
        }
        eval_lengths: list[int] = []
        eval_costs: list[int] = []
        for unit in validation_long_units:
            unit_lengths = [
                validation_window_lengths[group.trajectory_id]
                for group in unit.window_groups
            ]
            eval_lengths.append(max(unit_lengths))
            eval_costs.append(sum(unit_lengths))
        for unit in validation_short_units:
            unit_lengths = [
                validation_short_lengths[group.trajectory_id]
                for group in unit.groups
            ]
            eval_lengths.append(max(unit_lengths))
            eval_costs.append(sum(unit_lengths))
        if len(eval_lengths) != len(eval_dataset):
            raise RuntimeError("MIL validation schedule does not cover every unit")
        mil_mining_state = LongCotMilMiningState(
            exact_max_windows=mil_options["exact_max_windows"],
            active_windows=mil_options["active_windows"],
            exploitation_windows=mil_options["exploitation_windows"],
            exploration_windows=mil_options["exploration_windows"],
            initial_selection=(
                "position_stratified_no_model"
                if mil_options["initial_cache"] == "position_stratified_no_model"
                else "cached_rank_then_position"
            ),
        )
        teacher_options = mil_options.get("teacher_distillation")
        if teacher_options is not None:
            teacher_distillation_logits, teacher_distillation_audit = (
                _load_teacher_distillation_logits(
                    effective,
                    teacher_options,
                    train_window_manifest,
                    train_windows,
                )
            )
        source_query_indices = {
            query.query_id: index
            for index, query in enumerate(
                sorted(
                    (
                        query
                        for group in validation_groups
                        for query in group.queries
                    ),
                    key=lambda query: query.query_id,
                )
            )
        }
        train_collator = LongCotMilCollator(
            base_train_collator,
            mining_state=mil_mining_state,
            teacher_logits=teacher_distillation_logits,
        )
        eval_collator = LongCotMilCollator(
            base_eval_collator,
            mining_state=None,
            source_query_indices=source_query_indices,
            full_long_bags=True,
        )
        # Validation-only recovery never iterates the training sampler. Keep
        # its frozen two-rank schedule contract while the eval sampler uses
        # the actual one-rank runtime and therefore performs no collectives.
        training_sampler_replicas = (
            2
            if recovery_checkpoint is not None
            and effective["distributed"]["world_size"] == 1
            else effective["distributed"]["world_size"]
        )
        if mil_options["training_schedule"] == "full_epoch_joint":
            sampler = LongCotMilFullEpochBatchSampler(
                train_dataset,
                train_unit_lengths,
                train_unit_costs,
                bucket_boundaries=boundaries,
                seed=effective["runtime"]["seed"],
                synchronized_replicas=training_sampler_replicas,
                gradient_accumulation_steps=effective["optimization"][
                    "gradient_accumulation_steps"
                ],
                expected_long_units=mil_options["expected_long_units"],
                expected_short_units=mil_options["expected_short_units"],
            )
        else:
            sampler = LongCotMilTwoStreamBatchSampler(
                train_dataset,
                exposures=mil_options["long_exposures"],
                seed=effective["runtime"]["seed"],
                synchronized_replicas=training_sampler_replicas,
                replay_only=mil_options["replay_only"],
                expected_long_units=mil_options["expected_long_units"],
                sampled_long_units_per_exposure=mil_options[
                    "sampled_long_units_per_exposure"
                ],
            )
        eval_sampler = SynchronizedLengthBucketBatchSampler(
            eval_lengths,
            bucket_boundaries=boundaries,
            batch_size=effective["optimization"]["eval_batch_size"],
            synchronized_replicas=effective["distributed"]["world_size"],
            costs=eval_costs,
        )
        validation_schedule_audit = {
            "ordering": "length_bucket_then_adjacent_cost_groups",
            "replica_assignment": "greedy_least_total_cost",
            "unit_length": "max_exact_endpoint_tokens_across_unit_groups",
            "unit_cost": "sum_exact_endpoint_tokens_across_unit_groups",
            "minimum_unit_cost": min(eval_costs),
            "maximum_unit_cost": max(eval_costs),
            "validation_units": len(eval_costs),
        }
        mil_refresh_groups = tuple(train_windows)
        mil_refresh_collator = base_eval_collator
        mil_manifests = {
            "train": train_window_manifest,
            "validation": validation_window_manifest,
            "train_evidence_audit": train_evidence_audit,
            "validation_evidence_audit": validation_evidence_audit,
        }
        contrast_audit = {
            "long_train": train_long_audit,
            "short_replay_train": train_short_audit,
            "long_validation": validation_long_audit,
            "short_validation": validation_short_audit,
        }
        compute_metrics = lambda prediction: _mil_metrics(
            prediction,
            pooling=mil_options["pooling"],
            short_validation_auroc_floor=mil_options[
                "short_validation_auroc_floor"
            ],
        )
    else:
        split_lengths = _validate_token_limits(
            effective,
            tokenizer,
            {"train": train_groups, "validation": validation_groups},
        )
        lengths = split_lengths["train"]
        train_collator = base_train_collator
        eval_collator = base_eval_collator
        eval_dataset = TrajectoryGroupDataset(validation_groups)
    if training_unit == "trajectory":
        metric_weighting = effective["monitor"]["backend_options"].get(
            "metric_driven_weighting"
        )
        family_sampling_weights = (
            metric_weighting["resolved"]["family_sampling_weights"]
            if metric_weighting is not None
            else None
        )
        train_dataset = TrajectoryGroupDataset(train_groups)
        sampler = FamilyUniformLengthBucketBatchSampler(
            train_groups,
            lengths,
            bucket_boundaries=boundaries,
            batch_size=effective["optimization"]["batch_size"],
            num_samples=len(train_groups),
            seed=effective["runtime"]["seed"],
            drop_last=False,
            synchronized_replicas=effective["distributed"]["world_size"],
            family_weights=family_sampling_weights,
        )
    elif training_unit == "factor_instance_contrast":
        metric_weighting = effective["monitor"]["backend_options"].get(
            "metric_driven_weighting"
        )
        family_sampling_weights = (
            metric_weighting["resolved"]["family_sampling_weights"]
            if metric_weighting is not None
            else None
        )
        contrast_units, contrast_audit = build_factor_instance_contrast_units(train_groups)
        train_dataset = FactorInstanceContrastDataset(contrast_units)
        train_collator = FactorInstanceContrastCollator(train_collator)
        sampler = FactorInstanceContrastLengthBucketBatchSampler(
            contrast_units,
            {
                group.trajectory_id: length
                for group, length in zip(train_groups, lengths, strict=True)
            },
            bucket_boundaries=boundaries,
            batch_size=effective["optimization"]["batch_size"],
            num_trajectory_slots=len(train_groups),
            seed=effective["runtime"]["seed"],
            drop_last=False,
            synchronized_replicas=effective["distributed"]["world_size"],
            family_weights=family_sampling_weights,
        )
    elif training_unit != "long_cot_mil":
        raise RuntimeError(f"unsupported training unit: {training_unit}")
    if training_unit != "long_cot_mil":
        eval_sampler = SynchronizedLengthBucketBatchSampler(
            split_lengths["validation"],
            bucket_boundaries=boundaries,
            batch_size=effective["optimization"]["eval_batch_size"],
            synchronized_replicas=effective["distributed"]["world_size"],
        )
    dataset_manifest = load_config(_dataset_paths(effective)["manifest"])
    model, trainable = build_training_model(effective, dependencies, architecture, tokenizer)
    if (
        recovery_checkpoint is None
        and mil_mining_state is not None
        and not mil_options["replay_only"]
    ):
        cache_path = output / "mil_mining_cache.json"
        if effective["optimization"]["resume_from_checkpoint"] is not None:
            resume_cache = (
                Path(effective["optimization"]["resume_from_checkpoint"])
                / "mil_mining_cache.json"
            )
            restore_path = resume_cache if resume_cache.is_file() else cache_path
            if not restore_path.is_file():
                raise RuntimeError("resumed MIL run lacks mil_mining_cache.json")
            mil_mining_state.restore(load_config(restore_path))
        elif mil_options["initial_cache"] == "warm_start_full_no_grad":
            torch = dependencies["torch"]
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(
                "cuda", local_rank
            ) if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)
            cache_rows = _evaluate_batches(
                model,
                mil_refresh_groups,
                mil_refresh_collator,
                batch_size=1,
                device=device,
            )
            mil_mining_state.update_rows(cache_rows, full_refresh=True)
            model.train()
            if is_world_process_zero():
                write_json(cache_path, mil_mining_state.snapshot())
                write_json(
                    output / "mil_mining_cache_audit.json",
                    mil_mining_state.audit(),
                )
        elif mil_options["initial_cache"] == "position_stratified_no_model":
            if is_world_process_zero():
                write_json(cache_path, mil_mining_state.snapshot())
                write_json(
                    output / "mil_mining_cache_audit.json",
                    mil_mining_state.audit(),
                )
        else:
            raise RuntimeError(
                f"unsupported MIL initial cache: {mil_options['initial_cache']!r}"
            )
    if mil_manifests is not None and is_world_process_zero():
        write_json(output / "mil_train_window_manifest.json", mil_manifests["train"])
        write_json(
            output / "mil_validation_window_manifest.json",
            mil_manifests["validation"],
        )
        write_json(
            output / "mil_train_evidence_audit.json",
            mil_manifests["train_evidence_audit"],
        )
        write_json(
            output / "mil_validation_evidence_audit.json",
            mil_manifests["validation_evidence_audit"],
        )
        if teacher_distillation_audit is not None:
            write_json(
                output / "teacher_distillation_audit.json",
                teacher_distillation_audit,
            )
    effective["training"] = {
        "trainable_parameters": trainable,
        "memory_telemetry": {
            "enabled": os.environ.get("FACTOR_MONITOR_MEMORY_TELEMETRY", "0") == "1",
            "source": "torch.cuda.max_memory_allocated_and_reserved",
            "sampling": "per_training_step",
        },
        "validation_query_padding": {
            "policy": "fixed_global_max",
            "pad_to_records": validation_pad_queries,
            "observed_trajectory_counts": {
                str(query_count): count
                for query_count, count in sorted(validation_query_counts.items())
            },
        },
        "shared_cadence_order": "save_then_evaluate",
        "length_bucket_sampler": {
            "training_unit": training_unit,
            "loss_aggregation": effective["monitor"]["backend_options"][
                "loss_aggregation"
            ],
            "synchronized_replicas": sampler.synchronized_replicas,
            "per_replica_trajectory_slots": getattr(
                sampler, "trajectory_slots_per_replica", sampler.batch_size
            ),
            "remainder_policy": (
                "zero_loss_sync_padding"
                if mil_options is not None
                and mil_options["training_schedule"] == "full_epoch_joint"
                else "repeat_within_bucket_to_full_synchronized_step"
            ),
            "validation_exact_coverage": True,
            "validation_padding_policy": "repeat_at_final_global_step_only",
            **(
                {"validation_cost_balancing": validation_schedule_audit}
                if validation_schedule_audit is not None
                else {}
            ),
        },
        **(
            {
                "metric_driven_weighting": {
                    "sampling_unit": "family_priority_factor_instance_anchor",
                    "loss_unit": "factor_instance_query_level",
                    "mil_loss_application": (
                        effective["monitor"]["backend_options"].get(
                            "metric_mil_loss_weighting"
                        )
                        if effective["monitor"]["backend_options"].get(
                            "batching", {}
                        ).get("training_unit")
                        == "long_cot_mil"
                        else "not_applicable"
                    ),
                    "configured": effective["monitor"]["backend_options"][
                        "metric_driven_weighting"
                    ],
                    "epoch0_sampler_audit": _epoch0_sampler_audit(sampler),
                }
            }
            if effective["monitor"]["backend_options"].get(
                "metric_driven_weighting"
            )
            is not None
            else {}
        ),
        **({"contrast_unit_audit": contrast_audit} if contrast_audit is not None else {}),
        **(
            {
                "long_cot_mil": {
                    "train_window_summary": mil_manifests["train"]["summary"],
                    "validation_window_summary": mil_manifests["validation"]["summary"],
                    **(
                        {
                            "optimizer_steps_per_epoch": sampler.optimizer_steps_per_epoch,
                            "epoch_schedules": [
                                sampler.audit(epoch=epoch)
                                for epoch in range(
                                    int(effective["optimization"]["epochs"] or 1)
                                )
                            ],
                        }
                        if mil_options["training_schedule"] == "full_epoch_joint"
                        else {
                            "optimizer_steps_per_exposure": (
                                sampler.optimizer_steps_per_exposure
                            )
                        }
                    ),
                    "mining": mil_mining_state.audit(),
                    "validation_pooling": mil_options["pooling"],
                    "short_validation_auroc_floor": mil_options[
                        "short_validation_auroc_floor"
                    ],
                    **(
                        {"teacher_distillation": teacher_distillation_audit}
                        if teacher_distillation_audit is not None
                        else {}
                    ),
                }
            }
            if mil_manifests is not None
            else {}
        ),
    }
    if is_world_process_zero():
        write_json(output / "effective_config.json", effective)
    trainer_type = _trainer_class(dependencies)
    trainer = trainer_type(
        model=model,
        args=_training_arguments(effective, dependencies),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        eval_data_collator=eval_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        factor_batch_sampler=sampler,
        factor_eval_batch_sampler=eval_sampler,
        bundle_context={
            "effective_config": effective,
            "dataset_manifest": dataset_manifest,
        },
        mil_mining_state=mil_mining_state,
        mil_refresh_groups=mil_refresh_groups,
        mil_refresh_collator=mil_refresh_collator,
    )
    if recovery_checkpoint is not None:
        recovery_mode = (
            "explicit_validation_only"
            if validation_only is not None
            else "pending_validation_only"
        )
        effective["training"]["recovery"] = {
            "mode": recovery_mode,
            "checkpoint": str(recovery_checkpoint),
            "training_steps_executed": 0,
        }
        if is_world_process_zero():
            write_json(output / "effective_config.json", effective)
        metrics = _run_pending_validation_only(
            trainer,
            recovery_checkpoint,
            immutable_checkpoint=validation_only is not None,
            metric=effective["evaluation"]["metric"],
        )
    else:
        result = trainer.train(
            resume_from_checkpoint=effective["optimization"]["resume_from_checkpoint"]
        )
        trainer.save_model(str(output))
        trainer.save_state()
        metrics = dict(result.metrics)
    if is_world_process_zero():
        write_json(output / "train_metrics.json", metrics)
        if recovery_checkpoint is not None or effective["evaluation"].get(
            "during_training", True
        ):
            materialize_validation_threshold(output, effective)
    return {"status": "completed", "output_directory": str(output), "metrics": metrics}


def _evaluate_batches(
    model: Any,
    groups: Sequence[TrajectoryGroup],
    collator: TrajectoryCollator | EndpointTrajectoryCollator,
    *,
    batch_size: int,
    device: Any,
    start_index: int = 0,
    on_batch: Callable[[int, Sequence[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    dependencies = _dependencies()
    torch = dependencies["torch"]
    model.eval()
    rows: list[dict[str, Any]] = []
    for start in range(start_index, len(groups), batch_size):
        selected = groups[start : start + batch_size]
        batch = collator(selected)
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if hasattr(value, "to")
        }
        if isinstance(model, (EndpointBinaryMonitor, CausalLMChoiceMonitor)):
            input_names = [
                "endpoint_input_ids",
                "endpoint_attention_mask",
                "endpoint_role_ids",
                "query_mask",
            ]
            if (
                isinstance(model, EndpointBinaryMonitor)
                and model.factor_guided_readout is not None
            ):
                input_names.extend(
                    ("factor_span_mask", "context_span_mask", "cot_span_mask")
                )
            if (
                isinstance(model, EndpointBinaryMonitor)
                and model.signed_criterion_panel is not None
            ):
                input_names.extend(
                    (
                        "factor_span_mask",
                        "relation_span_mask",
                        "applied_span_mask",
                        "not_applied_span_mask",
                    )
                )
        elif model.joint_encoding:
            input_names = (
                "joint_input_ids",
                "joint_attention_mask",
                "factor_span_mask",
                "applied_span_mask",
                "not_applied_span_mask",
                "context_span_mask",
                "cot_span_mask",
                "query_mask",
                "relation",
                "evidence_route",
            )
        else:
            input_names = (
                "evidence_input_ids",
                "evidence_attention_mask",
                "context_span_mask",
                "cot_span_mask",
                "factor_input_ids",
                "factor_attention_mask",
                "factor_span_mask",
                "applied_span_mask",
                "not_applied_span_mask",
                "query_mask",
                "relation",
                "evidence_route",
            )
        model_inputs = {key: tensor_batch[key] for key in input_names}
        with torch.no_grad():
            logits = model(**model_inputs).detach().float().cpu()
        batch_rows: list[dict[str, Any]] = []
        for batch_index, group in enumerate(selected):
            for record_index, query in enumerate(group.queries):
                row = {
                        "query_id": query.query_id,
                        "trajectory_id": query.trajectory_id,
                        "score": float(torch.sigmoid(logits[batch_index, record_index]).item()),
                        "logit": float(logits[batch_index, record_index].item()),
                        "family": query.family,
                        "factor_level": query.factor_level,
                        "evidence_route": model.architecture.family_routes[query.family],
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
                batch_rows.append(row)
        if on_batch is not None:
            on_batch(start + len(selected), batch_rows)
    return rows


def _factor_path_records(group: TrajectoryGroup) -> list[tuple[Any, Any, int, int]]:
    """Pair the matched/shuffled donor records for each source query in order."""

    order: list[str] = []
    grouped: dict[str, dict[str, tuple[Any, int]]] = {}
    for index, query in enumerate(group.queries):
        source = query.source_query_id or query.query_id
        if source not in grouped:
            grouped[source] = {}
            order.append(source)
        if query.intervention in grouped[source]:
            raise RuntimeError(
                f"duplicate {query.intervention} Factor path record for {source}"
            )
        grouped[source][query.intervention] = (query, index)
    pairs: list[tuple[Any, Any, int, int]] = []
    for source in order:
        variants = grouped[source]
        if set(variants) != {"matched", "shuffled"}:
            raise RuntimeError(
                f"Factor path source {source} requires exactly matched and shuffled records"
            )
        matched, matched_index = variants["matched"]
        shuffled, shuffled_index = variants["shuffled"]
        if matched.label != shuffled.label or matched.factor_level != shuffled.factor_level:
            raise RuntimeError(f"Factor path donor metadata differs for {source}")
        pairs.append((matched, shuffled, matched_index, shuffled_index))
    return pairs


def _factor_path_score_id(source_query_id: str, intervention: str) -> str:
    return f"{source_query_id}::factor_path={intervention}"


def _evaluate_factor_path_batches(
    model: Any,
    groups: Sequence[TrajectoryGroup],
    collator: TrajectoryCollator,
    *,
    interventions: Sequence[str],
    batch_size: int,
    device: Any,
    start_index: int = 0,
    on_batch: Callable[[int, Sequence[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run direct-route Locator/Matcher activation interventions without retraining."""

    if not isinstance(model, FactorMonitor):
        raise RuntimeError("Factor path attribution requires FactorMonitor")
    requested = tuple(interventions)
    if not requested or requested[0] != "matched":
        raise RuntimeError("Factor path interventions must begin with matched")
    if len(requested) != len(set(requested)) or set(requested) - set(
        FACTOR_PATH_INTERVENTIONS
    ):
        raise RuntimeError("Factor path interventions contain unsupported values")
    dependencies = _dependencies()
    torch = dependencies["torch"]
    model.eval()
    rows: list[dict[str, Any]] = []
    shuffled_paths = {
        "full_shuffled",
        "locator_only_shuffled",
        "matcher_only_shuffled",
    }
    input_names = (
        "evidence_input_ids",
        "evidence_attention_mask",
        "context_span_mask",
        "cot_span_mask",
        "factor_input_ids",
        "factor_attention_mask",
        "factor_span_mask",
        "applied_span_mask",
        "not_applied_span_mask",
        "query_mask",
        "relation",
        "evidence_route",
    )
    for start in range(start_index, len(groups), batch_size):
        selected = groups[start : start + batch_size]
        source_records = [_factor_path_records(group) for group in selected]
        maximum = max(len(records) for records in source_records)
        matched_indices = torch.zeros(
            (len(selected), maximum), dtype=torch.long, device=device
        )
        shuffled_indices = torch.zeros_like(matched_indices)
        source_query_mask = torch.zeros(
            (len(selected), maximum), dtype=torch.bool, device=device
        )
        for batch_index, records in enumerate(source_records):
            for source_index, (_, _, matched_index, shuffled_index) in enumerate(records):
                matched_indices[batch_index, source_index] = matched_index
                shuffled_indices[batch_index, source_index] = shuffled_index
                source_query_mask[batch_index, source_index] = True

        batch = collator(selected)
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if hasattr(value, "to")
        }
        model_inputs = {key: tensor_batch[key] for key in input_names}
        with torch.no_grad():
            logits_by_path = {
                name: values.detach().float().cpu()
                for name, values in model.direct_factor_path_logits(
                    **model_inputs,
                    matched_factor_indices=matched_indices,
                    shuffled_factor_indices=shuffled_indices,
                    source_query_mask=source_query_mask,
                    interventions=requested,
                ).items()
            }
        batch_rows: list[dict[str, Any]] = []
        for batch_index, records in enumerate(source_records):
            for source_index, (matched, shuffled, _, _) in enumerate(records):
                source_query_id = matched.source_query_id or matched.query_id
                for intervention in requested:
                    donor = shuffled if intervention in shuffled_paths else matched
                    logit = logits_by_path[intervention][batch_index, source_index]
                    row = {
                        "query_id": _factor_path_score_id(
                            source_query_id, intervention
                        ),
                        "trajectory_id": matched.trajectory_id,
                        "score": float(torch.sigmoid(logit).item()),
                        "logit": float(logit.item()),
                        "family": matched.family,
                        "factor_level": matched.factor_level,
                        "evidence_route": model.architecture.family_routes[
                            matched.family
                        ],
                        "negative_type": matched.negative_type,
                        "factor_intervention": donor.intervention,
                        "factor_path_intervention": intervention,
                        "source_query_id": source_query_id,
                        "donor_query_id": donor.donor_query_id or donor.query_id,
                    }
                    if matched.factor_scope is not None:
                        row["factor_scope"] = matched.factor_scope
                    if matched.label is not None:
                        row["label_id"] = matched.label
                    rows.append(row)
                    batch_rows.append(row)
        if on_batch is not None:
            on_batch(start + len(selected), batch_rows)
    return rows


def _evaluation_progress_config(effective: Mapping[str, Any]) -> dict[str, Any]:
    identity_source = dict(effective)
    identity_source.pop("evaluation_progress", None)
    scope = (
        effective.get("monitor", {})
        .get("backend_options", {})
        .get("evaluation_scope", {})
    )
    interval = (
        DIAGNOSTIC_PROGRESS_INTERVAL_TRAJECTORIES
        if scope.get("mode") == "diagnostic"
        else EVALUATION_PROGRESS_INTERVAL_TRAJECTORIES
    )
    return {
        "schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
        "interval_trajectories": interval,
        "partial_scores_file": EVALUATION_PARTIAL_SCORES_FILE,
        "progress_file": EVALUATION_PROGRESS_FILE,
        "resume_policy": "same_effective_config_validated_query_prefix",
        "resume_identity": stable_hash(identity_source),
    }


def _expected_group_score_ids(
    group: TrajectoryGroup, effective: Mapping[str, Any]
) -> list[str]:
    scope = (
        effective.get("monitor", {})
        .get("backend_options", {})
        .get("evaluation_scope", {})
    )
    path_interventions = scope.get("factor_path_interventions")
    if not path_interventions:
        return [query.query_id for query in group.queries]
    return [
        _factor_path_score_id(matched.source_query_id or matched.query_id, intervention)
        for matched, _, _, _ in _factor_path_records(group)
        for intervention in path_interventions
    ]


def _load_evaluation_progress(
    output: Path,
    effective: Mapping[str, Any],
    groups: Sequence[TrajectoryGroup],
) -> tuple[list[dict[str, Any]], int]:
    partial = output / EVALUATION_PARTIAL_SCORES_FILE
    progress_path = output / EVALUATION_PROGRESS_FILE
    if not partial.exists() and not progress_path.exists():
        return [], 0
    if not progress_path.is_file():
        raise RuntimeError(
            "evaluation resume requires both scores.partial.jsonl and progress.json"
        )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_progress = _evaluation_progress_config(effective)
    if progress.get("schema_version") != EVALUATION_PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("evaluation progress schema version mismatch")
    if progress.get("resume_identity") != expected_progress["resume_identity"]:
        raise RuntimeError("evaluation partial scores belong to a different effective config")
    recorded_completed = progress.get("completed_trajectories")
    if not isinstance(recorded_completed, int) or isinstance(recorded_completed, bool):
        raise RuntimeError("evaluation completed_trajectories must be an integer")
    if recorded_completed < 0 or recorded_completed > len(groups):
        raise RuntimeError("evaluation completed_trajectories is out of range")
    score_source = partial
    if not partial.is_file():
        final_scores = output / "scores.jsonl"
        if progress.get("status") != "completed" or not final_scores.is_file():
            raise RuntimeError(
                "evaluation resume requires partial scores unless final scores are complete"
            )
        score_source = final_scores
    rows = read_jsonl(score_source)
    observed_query_ids = [row.get("query_id") for row in rows]
    expected_query_ids: list[str] = []
    inferred_completed = 0 if not observed_query_ids else None
    if observed_query_ids:
        for completed, group in enumerate(groups, start=1):
            expected_query_ids.extend(_expected_group_score_ids(group, effective))
            if len(expected_query_ids) == len(observed_query_ids):
                inferred_completed = completed
                break
            if len(expected_query_ids) > len(observed_query_ids):
                break
    if (
        inferred_completed is None
        or observed_query_ids != expected_query_ids
    ):
        raise RuntimeError("evaluation partial scores do not match the frozen query prefix")
    if recorded_completed > inferred_completed:
        raise RuntimeError("evaluation progress is ahead of durable partial scores")
    return rows, inferred_completed


def _write_evaluation_progress(
    output: Path,
    effective: Mapping[str, Any],
    *,
    completed_trajectories: int,
    total_trajectories: int,
    score_rows: int,
    status: str,
) -> None:
    write_json(
        output / EVALUATION_PROGRESS_FILE,
        {
            **_evaluation_progress_config(effective),
            "status": status,
            "completed_trajectories": completed_trajectories,
            "total_trajectories": total_trajectories,
            "score_rows": score_rows,
        },
    )


def _evaluation_checkpoint_due(
    completed_trajectories: int,
    last_checkpointed_trajectories: int,
    total_trajectories: int,
    interval_trajectories: int = EVALUATION_PROGRESS_INTERVAL_TRAJECTORIES,
) -> bool:
    return (
        completed_trajectories == total_trajectories
        or completed_trajectories - last_checkpointed_trajectories
        >= interval_trajectories
    )


def _evaluation_summary(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    reporting_dimensions: Mapping[str, Any] | None = None,
    *,
    require_all_dimension_values: bool = True,
) -> dict[str, Any]:
    levels = sorted({str(row["factor_level"]) for row in rows})
    summary = {
        **score_summary(rows, threshold),
        "by_factor_level": {
            level: score_summary(
                [row for row in rows if row["factor_level"] == level], threshold
            )
            for level in levels
        },
    }
    for dimension, specification in (reporting_dimensions or {}).items():
        required = bool(specification["required"])
        expected_values = tuple(specification["values"])
        missing = [row.get("query_id") for row in rows if not row.get(dimension)]
        if required and missing:
            raise RuntimeError(
                f"evaluation reporting dimension {dimension} is missing from "
                f"{len(missing)} score rows: {missing[:3]}"
            )
        observed_values = {
            str(row[dimension]) for row in rows if row.get(dimension) is not None
        }
        unexpected = observed_values - set(expected_values)
        if unexpected:
            raise RuntimeError(
                f"evaluation reporting dimension {dimension} has unexpected values: "
                f"{sorted(unexpected)}"
            )
        if (
            required
            and require_all_dimension_values
            and observed_values != set(expected_values)
        ):
            raise RuntimeError(
                f"evaluation reporting dimension {dimension} expected values "
                f"{list(expected_values)}, observed {sorted(observed_values)}"
            )
        grouped: dict[str, Any] = {}
        for value in expected_values:
            selected = [row for row in rows if row.get(dimension) == value]
            if not selected:
                continue
            grouped[value] = {
                "overall": score_summary(selected, threshold),
                "by_factor_level": {
                    level: score_summary(
                        [row for row in selected if row["factor_level"] == level],
                        threshold,
                    )
                    for level in levels
                    if any(row["factor_level"] == level for row in selected)
                },
            }
        summary[f"by_{dimension}"] = grouped
    return summary


def _prediction_only_summary(
    rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    levels = sorted({str(row["factor_level"]) for row in rows})

    def counts(selected: Sequence[dict[str, Any]]) -> dict[str, int]:
        return {
            label: sum(
                ("applied" if float(row["score"]) >= threshold else "not_applied")
                == label
                for row in selected
            )
            for label in ("not_applied", "applied")
        }

    return {
        "rows": len(rows),
        "threshold": float(threshold),
        "prediction_counts": counts(rows),
        "by_factor_level": {
            level: {
                "rows": len(selected := [
                    row for row in rows if row["factor_level"] == level
                ]),
                "prediction_counts": counts(selected),
            }
            for level in levels
        },
        "note": (
            "No Factor labels are present; behavioral metrics require the "
            "benchmark-specific scorer."
        ),
    }


def _aggregate_long_cot_window_rows(
    window_rows: Sequence[dict[str, Any]],
    source_groups: Sequence[TrajectoryGroup],
    aggregations: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Aggregate window probabilities back to exactly one row per source query."""

    source_queries: dict[tuple[str, str], tuple[TrajectoryGroup, FactorQueryRecord]] = {}
    for group in source_groups:
        for query in group.queries:
            source_id = query.source_query_id or query.query_id
            key = (source_id, query.intervention)
            if key in source_queries:
                raise RuntimeError("long-CoT source query IDs are not unique")
            source_queries[key] = (group, query)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in window_rows:
        source_query_id = row.get("source_query_id")
        intervention = str(row.get("factor_intervention", "matched"))
        key = (source_query_id, intervention)
        if not isinstance(source_query_id, str) or key not in source_queries:
            raise RuntimeError("window score lacks a valid source_query_id")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(source_queries):
        missing = sorted(set(source_queries) - set(grouped))
        unexpected = sorted(set(grouped) - set(source_queries))
        raise RuntimeError(
            "window scores do not exactly cover source queries; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    by_aggregation: dict[str, list[dict[str, Any]]] = {
        name: [] for name in aggregations
    }
    output_rows: list[dict[str, Any]] = []
    for source_key in sorted(source_queries):
        group, query = source_queries[source_key]
        selected = grouped[source_key]
        scores = [float(row["score"]) for row in selected]
        if any(score < 0.0 or score > 1.0 or not math.isfinite(score) for score in scores):
            raise RuntimeError("window score is not a finite probability")
        ordered = sorted(scores, reverse=True)
        values: dict[str, float] = {}
        for aggregation in aggregations:
            if aggregation == "max":
                values[aggregation] = ordered[0]
            elif aggregation == "top2_mean":
                values[aggregation] = sum(ordered[:2]) / min(2, len(ordered))
            elif aggregation == "top2_logmeanexp":
                logits = [row.get("logit") for row in selected]
                if any(
                    isinstance(logit, bool)
                    or not isinstance(logit, (int, float))
                    or not math.isfinite(float(logit))
                    for logit in logits
                ):
                    raise RuntimeError(
                        "top2_logmeanexp requires one finite raw logit per window"
                    )
                values[aggregation] = _top2_mil_probability(
                    logits, pooling="top2_logmeanexp"
                )
            elif aggregation == "noisy_or":
                values[aggregation] = 1.0 - math.prod(
                    1.0 - score for score in scores
                )
            else:
                raise RuntimeError(
                    f"unsupported long-CoT aggregation: {aggregation!r}"
                )
        base = {
            "query_id": (
                source_key[0]
                if source_key[1] == "matched"
                else f"{source_key[0]}::factor={source_key[1]}"
            ),
            "source_query_id": source_key[0],
            "trajectory_id": group.trajectory_id,
            "family": query.family,
            "factor_level": query.factor_level,
            "evidence_route": selected[0]["evidence_route"],
            "negative_type": query.negative_type,
            "window_count": len(selected),
            "factor_intervention": source_key[1],
            "donor_query_id": query.donor_query_id or query.query_id,
        }
        if query.factor_scope is not None:
            base["factor_scope"] = query.factor_scope
        if query.label is not None:
            base["label_id"] = query.label
        output_rows.append({**base, "aggregation_scores": values})
        for name, score in values.items():
            by_aggregation[name].append({**base, "aggregation": name, "score": score})
    return by_aggregation, output_rows


def _long_cot_reference_summary(
    reference: Mapping[str, Any],
    aggregated: Mapping[str, Sequence[dict[str, Any]]],
    *,
    threshold: float,
    prediction_only: bool,
    reporting_dimensions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare window aggregations to the preserved direct full-context scores."""

    reference_path = Path(str(reference["path"]))
    if file_sha256(reference_path) != reference["sha256"]:
        raise RuntimeError("long-CoT direct reference scores changed after config resolution")
    reference_rows = read_jsonl(reference_path)
    reference_by_query: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id in reference_by_query:
            raise RuntimeError("direct reference scores contain invalid query IDs")
        reference_by_query[query_id] = row
    first = next(iter(aggregated.values()))
    first_by_query = {str(row["query_id"]): row for row in first}
    if len(first_by_query) != len(first):
        raise RuntimeError("long-CoT aggregate rows duplicate source query IDs")
    source_ids = set(first_by_query)
    matched_ids = sorted(source_ids & set(reference_by_query))
    if not matched_ids:
        raise RuntimeError("long-CoT direct reference has no matching source queries")
    direct_rows = [reference_by_query[query_id] for query_id in matched_ids]
    for query_id, direct in zip(matched_ids, direct_rows, strict=True):
        aggregate_row = first_by_query[query_id]
        for field in ("family", "factor_level"):
            if direct.get(field) != aggregate_row.get(field):
                raise RuntimeError(
                    f"direct reference {field} differs for query {query_id}"
                )
        if not prediction_only and direct.get("label_id") != aggregate_row.get("label_id"):
            raise RuntimeError(f"direct reference label differs for query {query_id}")
    summarize = (
        (lambda rows: _prediction_only_summary(rows, threshold))
        if prediction_only
        else (
            lambda rows: _evaluation_summary(
                rows,
                threshold,
                reporting_dimensions,
                require_all_dimension_values=False,
            )
        )
    )
    aggregation_summaries: dict[str, Any] = {}
    for name, rows in aggregated.items():
        by_query = {str(row["query_id"]): row for row in rows}
        matched_rows = [by_query[query_id] for query_id in matched_ids]
        absolute = [
            abs(float(by_query[query_id]["score"]) - float(reference_by_query[query_id]["score"]))
            for query_id in matched_ids
        ]
        signed = [
            float(by_query[query_id]["score"]) - float(reference_by_query[query_id]["score"])
            for query_id in matched_ids
        ]
        aggregation_summaries[name] = {
            "metrics": summarize(matched_rows),
            "mean_absolute_score_delta_from_direct": sum(absolute) / len(absolute),
            "mean_signed_score_delta_from_direct": sum(signed) / len(signed),
        }
    return {
        "path": str(reference["path"]),
        "sha256": str(reference["sha256"]),
        "matched_queries": len(matched_ids),
        "window_only_queries": len(source_ids) - len(matched_ids),
        "direct_metrics": summarize(direct_rows),
        "aggregations_on_matched_queries": aggregation_summaries,
    }


def _paired_intervention_metrics(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    intervention_field: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], float]:
    """Compute matched-relative metrics shared by Factor diagnostic families."""

    import numpy as np

    by_variant = {
        variant: [row for row in rows if row[intervention_field] == variant]
        for variant in dict.fromkeys(str(row[intervention_field]) for row in rows)
    }
    if "matched" not in by_variant:
        raise RuntimeError("Factor diagnostic requires the matched intervention")
    matched = by_variant["matched"]
    matched_by_source = {str(row["source_query_id"]): row for row in matched}
    if len(matched_by_source) != len(matched):
        raise RuntimeError("matched Factor diagnostic rows duplicate source_query_id")

    variant_metrics: dict[str, Any] = {}
    matched_auroc = score_summary(matched, threshold)["auroc"]
    matched_scores = np.asarray([float(row["score"]) for row in matched], dtype=np.float64)
    between_std = float(matched_scores.std())
    for variant, selected in by_variant.items():
        summary = score_summary(selected, threshold)
        paired = []
        if variant != "matched":
            altered_by_source = {
                str(row["source_query_id"]): row for row in selected
            }
            if set(altered_by_source) != set(matched_by_source):
                raise RuntimeError(
                    f"{variant} rows do not cover the matched source-query set"
                )
            paired = [
                (
                    float(matched_by_source[source]["score"]),
                    float(altered_by_source[source]["score"]),
                )
                for source in sorted(matched_by_source)
            ]
        absolute = [abs(left - right) for left, right in paired]
        paired_left = np.asarray([value[0] for value in paired], dtype=np.float64)
        paired_right = np.asarray([value[1] for value in paired], dtype=np.float64)
        pearson = None
        if (
            len(paired) >= 2
            and float(paired_left.std()) > 0.0
            and float(paired_right.std()) > 0.0
        ):
            pearson = float(np.corrcoef(paired_left, paired_right)[0, 1])
        auroc = summary["auroc"]
        variant_metrics[variant] = {
            "binary_metrics": summary,
            "auroc_delta_from_matched": (
                None
                if auroc is None or matched_auroc is None
                else float(auroc) - float(matched_auroc)
            ),
            "mean_absolute_score_delta": (
                None if not absolute else float(np.mean(absolute))
            ),
            "median_absolute_score_delta": (
                None if not absolute else float(np.median(absolute))
            ),
            "normalized_mean_absolute_score_delta": (
                None
                if not absolute or between_std == 0.0
                else float(np.mean(absolute) / between_std)
            ),
            "pearson_with_matched": pearson,
        }
    return by_variant, variant_metrics, between_std


def _pairwise_ranking_summary(
    selected: Sequence[dict[str, Any]], *, seed: int, allow_empty: bool = False
) -> dict[str, Any] | None:
    """Bootstrap within-trajectory applied-vs-not-applied ranking concordance."""

    import numpy as np

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(
            (str(row["trajectory_id"]), str(row["factor_level"])), []
        ).append(row)
    margins_by_trajectory: dict[str, list[float]] = {}
    margins: list[float] = []
    for (trajectory_id, _), group_rows in grouped.items():
        positives = [float(row["score"]) for row in group_rows if row["label_id"] == 1]
        negatives = [float(row["score"]) for row in group_rows if row["label_id"] == 0]
        for positive in positives:
            for negative in negatives:
                margin = positive - negative
                margins.append(margin)
                margins_by_trajectory.setdefault(trajectory_id, []).append(margin)
    if not margins:
        if allow_empty:
            return None
        raise RuntimeError("mixed-label Factor diagnostic produced no pairwise margins")

    def concordance(values: Sequence[float]) -> float:
        return float(
            np.mean([1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values])
        )

    trajectories = sorted(margins_by_trajectory)
    trajectory_margins = {
        trajectory_id: float(np.mean(values))
        for trajectory_id, values in margins_by_trajectory.items()
    }
    trajectory_concordance = {
        trajectory_id: concordance(values)
        for trajectory_id, values in margins_by_trajectory.items()
    }
    randomizer = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(1000):
        sampled = randomizer.choice(trajectories, size=len(trajectories), replace=True)
        bootstrapped.append(float(np.mean([
            trajectory_concordance[str(trajectory_id)]
            for trajectory_id in sampled
        ])))
    return {
        "pairs": len(margins),
        "trajectories": len(trajectories),
        "aggregation": "pairwise_within_level_then_trajectory_mean",
        "mean_margin": float(np.mean(list(trajectory_margins.values()))),
        "median_margin": float(np.median(list(trajectory_margins.values()))),
        "concordance": float(np.mean(list(trajectory_concordance.values()))),
        "concordance_bootstrap_95_ci": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "bootstrap_replicates": 1000,
    }


def _factor_sensitivity_summary(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    native_checkpoint_threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Summarize the preregistered practical Factor-invariance diagnostics."""

    by_variant, variant_metrics, between_std = _paired_intervention_metrics(
        rows,
        threshold,
        intervention_field="factor_intervention",
    )
    matched = by_variant["matched"]

    pairwise_overall = _pairwise_ranking_summary(
        matched, seed=seed, allow_empty=True
    )
    pairwise_by_level = {
        level: _pairwise_ranking_summary(
            [row for row in matched if row["factor_level"] == level],
            seed=seed,
            allow_empty=True,
        )
        for level in sorted({str(row["factor_level"]) for row in matched})
    }
    altered = [
        variant_metrics[name]
        for name in ("shuffled", "no_factor", "constant")
        if name in variant_metrics
    ]
    score_intervention_invariance = bool(altered) and all(
            item["auroc_delta_from_matched"] is not None
            and abs(item["auroc_delta_from_matched"]) < 0.02
            and item["normalized_mean_absolute_score_delta"] is not None
            and item["normalized_mean_absolute_score_delta"] < 0.1
            for item in altered
        )
    practical_invariance = None
    if pairwise_overall is not None:
        ci_low, ci_high = pairwise_overall["concordance_bootstrap_95_ci"]
        practical_invariance = (
            ci_low <= 0.5 <= ci_high and score_intervention_invariance
        )
    return {
        "schema_version": 1,
        "primary_metric": "delta_perm",
        "primary_definition": "AUROC(matched) - AUROC(shuffled)",
        "delta_perm": (
            None
            if "shuffled" not in variant_metrics
            or variant_metrics["shuffled"]["auroc_delta_from_matched"] is None
            else -float(variant_metrics["shuffled"]["auroc_delta_from_matched"])
        ),
        "delta_no_factor": (
            None
            if "no_factor" not in variant_metrics
            or variant_metrics["no_factor"]["auroc_delta_from_matched"] is None
            else -float(variant_metrics["no_factor"]["auroc_delta_from_matched"])
        ),
        "preregistered_thresholds": {
            "absolute_auroc_change_lt": 0.02,
            "normalized_mean_absolute_score_delta_lt": 0.1,
            "pairwise_concordance_ci_must_include": 0.5,
        },
        "practical_factor_invariance": practical_invariance,
        "score_intervention_invariance": score_intervention_invariance,
        "pairwise_unavailable_reason": (
            "no within-trajectory applied/not-applied pairs"
            if pairwise_overall is None
            else None
        ),
        "matched_between_trajectory_score_std": between_std,
        "thresholds": {
            "shared_checkpoint_comparison": threshold,
            "native_checkpoint": native_checkpoint_threshold,
        },
        "native_checkpoint_threshold_metrics": {
            variant: score_summary(selected, native_checkpoint_threshold)
            for variant, selected in by_variant.items()
        },
        "pairwise_overall": pairwise_overall,
        "pairwise_by_factor_level": pairwise_by_level,
        "by_intervention": variant_metrics,
    }


def _factor_path_attribution_summary(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    native_checkpoint_threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Summarize inference-only Locator/raw-Factor matcher path interventions."""

    by_variant, variant_metrics, between_std = _paired_intervention_metrics(
        rows,
        threshold,
        intervention_field="factor_path_intervention",
    )
    levels = sorted({str(row["factor_level"]) for row in rows})
    return {
        "schema_version": 1,
        "diagnostic": "direct_factor_path_attribution",
        "interpretation_warning": (
            "zero interventions create out-of-distribution activations and are "
            "causal path probes, not model-performance estimates"
        ),
        "matched_between_trajectory_score_std": between_std,
        "thresholds": {
            "shared_checkpoint_comparison": threshold,
            "native_checkpoint": native_checkpoint_threshold,
        },
        "native_checkpoint_threshold_metrics": {
            variant: score_summary(selected, native_checkpoint_threshold)
            for variant, selected in by_variant.items()
        },
        "by_path_intervention": variant_metrics,
        "pairwise_by_path_intervention": {
            variant: {
                "overall": _pairwise_ranking_summary(selected, seed=seed),
                "by_factor_level": {
                    level: _pairwise_ranking_summary(
                        [row for row in selected if row["factor_level"] == level],
                        seed=seed,
                    )
                    for level in levels
                },
            }
            for variant, selected in by_variant.items()
        },
    }


def materialize_validation_threshold(
    output: Path, effective: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist the threshold selected at the unique best validation step."""

    output = output.resolve()
    state_path = output / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"training output lacks trainer_state.json: {output}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Trainer state JSON: {state_path}") from exc
    best_step = state.get("best_global_step")
    if isinstance(best_step, bool) or not isinstance(best_step, int) or best_step <= 0:
        raise RuntimeError("Trainer state lacks a positive best_global_step")
    checkpoint_value = state.get("best_model_checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise RuntimeError("Trainer state lacks best_model_checkpoint")
    checkpoint = Path(checkpoint_value).resolve()
    validation_only = effective.get("evaluation", {}).get("validation_only")
    if validation_only is None:
        if checkpoint.parent != output or checkpoint.name != f"checkpoint-{best_step}":
            raise RuntimeError(
                "best_model_checkpoint must be the matching immediate checkpoint inside output"
            )
        checkpoint_reference = checkpoint.name
    else:
        expected_checkpoint = Path(str(validation_only["checkpoint"])).resolve()
        if (
            checkpoint != expected_checkpoint
            or checkpoint.name != f"checkpoint-{best_step}"
            or validation_only.get("global_step") != best_step
        ):
            raise RuntimeError(
                "validation-only Trainer state does not match its immutable source checkpoint"
            )
        checkpoint_reference = str(checkpoint)
    if not checkpoint.is_dir():
        raise RuntimeError(f"best model checkpoint does not exist: {checkpoint}")
    candidates = [
        row
        for row in state.get("log_history", [])
        if isinstance(row, Mapping)
        and row.get("step") == best_step
        and "eval_threshold" in row
    ]
    # A validation-only recovery can append the same evaluation row twice:
    # once restored from the immutable checkpoint and once emitted by the
    # recovery evaluate() call.  Runtime fields are expected to differ, but
    # the selected threshold/metrics must remain identical.  Collapse only
    # such exact selection duplicates; any substantive disagreement remains
    # an error.
    volatile_keys = {
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "eval_model_preparation_time",
    }
    unique_candidates: list[Mapping[str, Any]] = []
    for candidate in candidates:
        signature = {
            key: value
            for key, value in candidate.items()
            if key not in volatile_keys
        }
        if not any(
            {
                key: value
                for key, value in existing.items()
                if key not in volatile_keys
            }
            == signature
            for existing in unique_candidates
        ):
            unique_candidates.append(candidate)
    if len(unique_candidates) != 1:
        raise RuntimeError(
            f"expected exactly one threshold-bearing validation row at step {best_step}, "
            f"found {len(unique_candidates)}"
        )
    row = unique_candidates[0]
    threshold = row["eval_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= float(threshold) <= 1
    ):
        raise RuntimeError("best validation threshold must be numeric in [0,1]")
    metric = effective.get("evaluation", {}).get("metric")
    if not isinstance(metric, str) or not metric:
        raise RuntimeError("effective config lacks the selected validation metric")
    metric_value = row.get(f"eval_{metric}")
    best_metric = state.get("best_metric")
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (int, float))
        or isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isclose(float(metric_value), float(best_metric), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(
            f"best validation {metric} does not match Trainer state best_metric"
        )
    metric_names = {
        "loss",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "gmean",
        "gmean2",
        "f1",
        "selection_score",
        *{
            f"{prefix}_{name}"
            for prefix in ("short", "long")
            for name in (
                "threshold",
                "auroc",
                "auprc",
                "balanced_accuracy",
                "gmean",
                "gmean2",
                "f1",
            )
        },
    }
    validation_metrics = {
        name: float(row[f"eval_{name}"])
        for name in sorted(metric_names)
        if isinstance(row.get(f"eval_{name}"), (int, float))
        and not isinstance(row.get(f"eval_{name}"), bool)
    }
    payload = {
        "global_step": best_step,
        "metric": metric,
        "metric_value": float(metric_value),
        "best_model_checkpoint": checkpoint_reference,
        "source": state_path.name,
        "source_sha256": file_sha256(state_path),
        "threshold": float(threshold),
        "validation_metrics": validation_metrics,
    }
    write_json(output / "validation_threshold.json", payload)
    mil = effective.get("monitor", {}).get("backend_options", {}).get("long_cot_mil")
    if mil is not None:
        short_auroc = row.get("eval_short_auroc")
        long_threshold = row.get("eval_long_threshold")
        if (
            isinstance(short_auroc, bool)
            or not isinstance(short_auroc, (int, float))
            or float(short_auroc) < float(mil["short_validation_auroc_floor"])
        ):
            raise RuntimeError(
                "selected MIL checkpoint violates the preregistered short-validation guard"
            )
        if (
            isinstance(long_threshold, bool)
            or not isinstance(long_threshold, (int, float))
            or not 0 <= float(long_threshold) <= 1
        ):
            raise RuntimeError("selected MIL checkpoint lacks a valid long-bag threshold")
        write_json(
            output / "mil_validation_threshold.json",
            {
                **payload,
                "threshold": float(long_threshold),
                "threshold_role": "full_window_bag",
                "pooling": mil["pooling"],
            },
        )
    return payload


def run_evaluation(effective: dict[str, Any]) -> dict[str, Any]:
    dependencies = _dependencies()
    tokenizer = _load_tokenizer(effective, dependencies)
    effective = _contract_enriched_config(effective, tokenizer)
    bundle = Path(effective["monitor"]["backend_options"]["model_bundle"])
    validate_evaluation_bundle_identity(effective)
    prediction_only = effective["evaluation"]["metric"] == PREDICTION_ONLY_METRIC
    groups = _load_groups(
        effective, split="query", allow_unlabelled=prediction_only
    )
    scope = effective["monitor"]["backend_options"]["evaluation_scope"]
    windowing = scope.get("long_cot_windowing")
    path_interventions = tuple(scope.get("factor_path_interventions", ()))
    intervention_manifest = None
    long_cot_source_groups: list[TrajectoryGroup] | None = None
    long_cot_manifest: dict[str, Any] | None = None
    frozen_long_cot_manifest: dict[str, Any] | None = None
    if windowing is None:
        groups = _select_evaluation_groups(effective, tokenizer, groups)
    if scope["mode"] == "diagnostic":
        if prediction_only:
            raise RuntimeError("Factor sensitivity diagnostics require binary labels")
        source_groups = _select_evaluation_groups(effective, tokenizer, groups)
        transformed_groups, intervention_manifest = factor_intervention_groups(
            source_groups,
            tokenizer,
            variants=scope["factor_interventions"],
            seed=scope["intervention_seed"],
            mixed_labels_only=scope["mixed_labels_only"],
        )
        if windowing is None:
            groups = transformed_groups
        else:
            # Build the split once from the matched source and reuse its exact
            # boundaries for all Factor variants.
            matched_source = [
                replace(
                    group,
                    queries=tuple(
                        query
                        for query in group.queries
                        if query.intervention == "matched"
                    ),
                )
                for group in transformed_groups
            ]
            matched_source, matched_windows, long_cot_manifest = (
                _long_cot_window_groups(
                    tokenizer,
                    matched_source,
                    contract_id=effective["monitor"]["backend_options"][
                        "endpoint_encoding_contract_id"
                    ],
                    trigger_endpoint_tokens=int(windowing["trigger_endpoint_tokens"]),
                    max_endpoint_tokens=int(windowing["max_endpoint_tokens"]),
                    cot_overlap_tokens=int(windowing["cot_overlap_tokens"]),
                    context_policy=str(windowing.get("context_policy", "repeat_full")),
                    overlap_fallback_policy=str(
                        windowing.get("overlap_fallback_policy", "error")
                    ),
                )
            )
            selected_source_ids = {group.trajectory_id for group in matched_source}
            variant_source_groups: list[TrajectoryGroup] = []
            variant_window_groups: list[TrajectoryGroup] = []
            for variant in scope["factor_interventions"]:
                variant_groups = [
                    replace(
                        group,
                        queries=tuple(
                            query
                            for query in group.queries
                            if query.intervention == variant
                        ),
                    )
                    for group in transformed_groups
                    if group.trajectory_id in selected_source_ids
                ]
                variant_source_groups.extend(variant_groups)
                variant_window_groups.extend(
                    _apply_shared_window_boundaries(
                        tokenizer,
                        variant_groups,
                        matched_source,
                        matched_windows,
                        variant=variant,
                        contract_id=effective["monitor"]["backend_options"][
                            "endpoint_encoding_contract_id"
                        ],
                        max_endpoint_tokens=int(windowing["max_endpoint_tokens"]),
                    )
                )
            long_cot_source_groups = variant_source_groups
            groups = variant_window_groups
        if path_interventions:
            if windowing is not None:
                raise RuntimeError("Factor path attribution cannot use long-CoT windows")
            if _model_variant(effective) != "factor_late_fusion":
                raise RuntimeError("Factor path attribution requires factor_late_fusion")
            families = {group.family for group in groups}
            non_direct = {
                family
                for family in families
                if effective["monitor"]["architecture"]["family_routes"][family]
                != "cot_direct"
            }
            if non_direct:
                raise RuntimeError(
                    "Factor path attribution currently supports cot_direct only; "
                    f"found {sorted(non_direct)}"
                )
            intervention_manifest = {
                **intervention_manifest,
                "diagnostic": "direct_factor_path_attribution",
                "factor_path_interventions": list(path_interventions),
                "score_rows": (
                    int(intervention_manifest["source_queries"])
                    * len(path_interventions)
                ),
            }
    elif windowing is not None:
        long_cot_source_groups, groups, long_cot_manifest = _long_cot_window_groups(
            tokenizer,
            groups,
            contract_id=effective["monitor"]["backend_options"][
                "endpoint_encoding_contract_id"
            ],
            trigger_endpoint_tokens=int(windowing["trigger_endpoint_tokens"]),
            max_endpoint_tokens=int(windowing["max_endpoint_tokens"]),
            cot_overlap_tokens=int(windowing["cot_overlap_tokens"]),
            context_policy=str(windowing.get("context_policy", "repeat_full")),
            overlap_fallback_policy=str(
                windowing.get("overlap_fallback_policy", "error")
            ),
        )
    if windowing is not None and long_cot_manifest is not None:
        if windowing.get("frozen_window_manifest") is not None:
            frozen_long_cot_manifest = _validate_frozen_long_cot_manifest(
                long_cot_manifest, windowing["frozen_window_manifest"]
            )
        effective["long_cot_windowing_runtime"] = long_cot_manifest["summary"]
        if frozen_long_cot_manifest is not None:
            effective["long_cot_windowing_runtime"]["frozen_manifest"] = (
                frozen_long_cot_manifest
            )
    _validate_token_limits(effective, tokenizer, {"query": groups})
    output = Path(effective["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    if long_cot_manifest is not None:
        write_json(output / LONG_COT_WINDOW_MANIFEST_FILE, long_cot_manifest)
    if intervention_manifest is not None:
        write_json(output / "factor_intervention_manifest.json", intervention_manifest)
    effective["evaluation_progress"] = _evaluation_progress_config(effective)
    resumed_rows, completed_trajectories = _load_evaluation_progress(
        output, effective, groups
    )
    write_json(output / "effective_config.json", effective)
    if completed_trajectories:
        print(
            json.dumps(
                {
                    "event": "evaluation_resume",
                    "completed_trajectories": completed_trajectories,
                    "total_trajectories": len(groups),
                    "score_rows": len(resumed_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        partial_path = output / EVALUATION_PARTIAL_SCORES_FILE
        partial_path.write_text("", encoding="utf-8")
        os.chmod(partial_path, 0o664)
        _write_evaluation_progress(
            output,
            effective,
            completed_trajectories=0,
            total_trajectories=len(groups),
            score_rows=0,
            status="running",
        )
    partial_path = output / EVALUATION_PARTIAL_SCORES_FILE
    new_rows: list[dict[str, Any]] = []
    if completed_trajectories < len(groups):
        base = _load_base_model(effective, dependencies)
        adapter = bundle / "backbone"
        if not adapter.is_dir():
            raise RuntimeError(f"model bundle lacks LoRA adapter directory: {adapter}")
        backbone = dependencies["PeftModel"].from_pretrained(
            base, adapter, is_trainable=False
        )
        model = load_model_bundle(
            backbone,
            bundle,
            endpoint_max_padded_tokens_per_backbone_forward=(
                effective["monitor"]["backend_options"].get(
                    "endpoint_max_padded_tokens_per_backbone_forward"
                )
                if _model_variant(effective) == "endpoint_early_fusion"
                else None
            ),
            joint_max_padded_tokens_per_backbone_forward=(
                effective["monitor"]["backend_options"].get(
                    "joint_max_padded_tokens_per_backbone_forward"
                )
                if _model_variant(effective) == "factor_joint_fusion"
                else None
            ),
        )
        device = dependencies["torch"].device(
            "cuda" if dependencies["torch"].cuda.is_available() else "cpu"
        )
        model.to(device)
        pending_rows: list[dict[str, Any]] = []
        written_new_rows = 0
        last_checkpointed = completed_trajectories

        def checkpoint(completed: int, batch_rows: Sequence[dict[str, Any]]) -> None:
            nonlocal last_checkpointed, written_new_rows
            pending_rows.extend(batch_rows)
            should_checkpoint = _evaluation_checkpoint_due(
                completed,
                last_checkpointed,
                len(groups),
                int(effective["evaluation_progress"]["interval_trajectories"]),
            )
            if not should_checkpoint:
                return
            with partial_path.open("a", encoding="utf-8", newline="") as handle:
                for row in pending_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            total_rows = len(resumed_rows) + written_new_rows + len(pending_rows)
            written_new_rows += len(pending_rows)
            last_checkpointed = completed
            pending_rows.clear()
            _write_evaluation_progress(
                output,
                effective,
                completed_trajectories=completed,
                total_trajectories=len(groups),
                score_rows=total_rows,
                status="running",
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "completed_trajectories": completed,
                        "total_trajectories": len(groups),
                        "score_rows": total_rows,
                        "percent": round(100.0 * completed / len(groups), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        collator = _collator(effective, tokenizer)
        if path_interventions:
            if not isinstance(collator, TrajectoryCollator):
                raise RuntimeError("Factor path attribution requires TrajectoryCollator")
            new_rows = _evaluate_factor_path_batches(
                model,
                groups,
                collator,
                interventions=path_interventions,
                batch_size=effective["runtime"]["batch_size"],
                device=device,
                start_index=completed_trajectories,
                on_batch=checkpoint,
            )
        else:
            new_rows = _evaluate_batches(
                model,
                groups,
                collator,
                batch_size=effective["runtime"]["batch_size"],
                device=device,
                start_index=completed_trajectories,
                on_batch=checkpoint,
            )
    rows = resumed_rows + new_rows
    policy = effective["evaluation"]["threshold_policy"]
    if policy != "frozen_validation":
        raise RuntimeError("formal evaluation requires threshold_policy=frozen_validation")
    threshold_value = effective["evaluation"]["threshold_value"]
    if windowing is not None:
        if long_cot_source_groups is None or long_cot_manifest is None:
            raise RuntimeError("long-CoT runtime state is incomplete")
        aggregated, aggregate_output_rows = _aggregate_long_cot_window_rows(
            rows,
            long_cot_source_groups,
            windowing["aggregations"],
        )
        write_jsonl(output / LONG_COT_AGGREGATED_SCORES_FILE, aggregate_output_rows)
        if prediction_only:
            aggregation_metrics = {
                name: _prediction_only_summary(selected, float(threshold_value))
                for name, selected in aggregated.items()
            }
            prediction_rows = [
                {
                    "query_id": row["query_id"],
                    "prediction_labels": {
                        name: (
                            "applied"
                            if float(score) >= float(threshold_value)
                            else "not_applied"
                        )
                        for name, score in row["aggregation_scores"].items()
                    },
                }
                for row in aggregate_output_rows
            ]
        else:
            aggregation_metrics = {
                name: _evaluation_summary(
                    selected,
                    float(threshold_value),
                    effective["evaluation"].get("reporting_dimensions"),
                    require_all_dimension_values=False,
                )
                for name, selected in aggregated.items()
            }
            prediction_rows = None
        summary = {
            "mode": (
                "teacher_window_audit"
                if windowing.get("comparison_mode") == "teacher_window_audit"
                else "long_cot_sliding_window"
            ),
            "rows": len(aggregate_output_rows),
            "window_score_rows": len(rows),
            "threshold": float(threshold_value),
            "selection": long_cot_manifest["summary"],
            "window_policy": long_cot_manifest["policy"],
            "aggregations": aggregation_metrics,
            **(
                {"frozen_window_manifest": frozen_long_cot_manifest}
                if frozen_long_cot_manifest is not None
                else {}
            ),
            **(
                {
                    "direct_reference": _long_cot_reference_summary(
                        windowing["direct_reference_scores"],
                        aggregated,
                        threshold=float(threshold_value),
                        prediction_only=prediction_only,
                        reporting_dimensions=effective["evaluation"].get(
                            "reporting_dimensions"
                        ),
                    )
                }
                if windowing.get("direct_reference_scores") is not None
                else {}
            ),
            "threshold_note": (
                "The frozen checkpoint threshold was selected for direct full-context "
                "scores, not recalibrated for window aggregation; AUROC/AUPRC are the "
                "primary sliding-window comparison."
            ),
        }
    elif prediction_only:
        prediction_rows = [
            {
                "query_id": row["query_id"],
                "prediction_label": (
                    "applied" if row["score"] >= float(threshold_value) else "not_applied"
                ),
            }
            for row in rows
        ]
        summary = _prediction_only_summary(rows, float(threshold_value))
    elif path_interventions:
        prediction_rows = None
        summary = _factor_path_attribution_summary(
            rows,
            float(threshold_value),
            native_checkpoint_threshold=float(scope["native_checkpoint_threshold"]),
            seed=int(scope["intervention_seed"]),
        )
    elif scope["mode"] == "diagnostic":
        prediction_rows = None
        summary = _factor_sensitivity_summary(
            rows,
            float(threshold_value),
            native_checkpoint_threshold=float(scope["native_checkpoint_threshold"]),
            seed=int(scope["intervention_seed"]),
        )
    else:
        prediction_rows = None
        summary = _evaluation_summary(
            rows,
            float(threshold_value),
            effective["evaluation"].get("reporting_dimensions"),
        )
    if partial_path.exists():
        os.replace(partial_path, output / "scores.jsonl")
    _write_evaluation_progress(
        output,
        effective,
        completed_trajectories=len(groups),
        total_trajectories=len(groups),
        score_rows=len(rows),
        status="completed",
    )
    if prediction_rows is not None:
        write_jsonl(output / "predictions.jsonl", prediction_rows)
    write_json(output / "metrics.json", summary)
    return {"status": "completed", "output_directory": str(output), "metrics": summary}


register_training_backend(BACKEND_NAME, run_training)
register_evaluation_backend(BACKEND_NAME, run_evaluation)
