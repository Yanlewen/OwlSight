"""Atomic FactorMonitor model-bundle save/load and round-trip checks."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..config import stable_hash
from ..io import make_tree_group_accessible, write_json, write_text
from .config import architecture_from_dict
from .encoding import SINGLE_ENDPOINT_DECISION_MODE
from .model import CausalLMChoiceMonitor, EndpointBinaryMonitor, FactorMonitor


MODEL_WEIGHTS = "factor_monitor.safetensors"


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError("checkpoint support requires safetensors") from exc
    return load_file, save_file


def save_model_bundle(
    model: FactorMonitor | EndpointBinaryMonitor | CausalLMChoiceMonitor,
    output_dir: Path,
    *,
    effective_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tokenizer: Any | None,
    save_backbone: bool,
) -> Path:
    """Write a complete immutable model bundle through an atomic directory rename."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"model bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    _, save_file = _safetensors()
    try:
        architecture = model.architecture.to_dict()
        route_mapping = dict(model.architecture.family_routes)
        write_json(stage / "architecture_config.json", architecture)
        write_json(stage / "effective_config.json", dict(effective_config))
        write_json(stage / "dataset_manifest.json", dict(dataset_manifest))
        write_json(stage / "metrics.json", dict(metrics))
        write_json(stage / "route_mapping.json", route_mapping)
        write_text(stage / "route_mapping.sha256", stable_hash(route_mapping) + "\n")
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in model.custom_state_dict().items()
        }
        save_file(state, str(stage / MODEL_WEIGHTS))
        if save_backbone:
            save = getattr(model.backbone, "save_pretrained", None)
            if save is None:
                raise TypeError("backbone does not implement save_pretrained")
            save(stage / "backbone", safe_serialization=True)
        if tokenizer is not None:
            save_tokenizer = getattr(tokenizer, "save_pretrained", None)
            if save_tokenizer is None:
                raise TypeError("tokenizer does not implement save_pretrained")
            save_tokenizer(stage / "tokenizer")
        make_tree_group_accessible(stage)
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def load_model_bundle(
    backbone: Any,
    model_dir: Path,
    *,
    endpoint_max_padded_tokens_per_backbone_forward: int | None = None,
    joint_max_padded_tokens_per_backbone_forward: int | None = None,
) -> FactorMonitor | EndpointBinaryMonitor | CausalLMChoiceMonitor:
    """Restore a monitor, optionally overriding a joint-input batching budget.

    The budget controls only how independent query sequences are grouped into
    backbone forwards. It is not a learned model parameter, so evaluation may
    raise it after a dataset-bound length audit proves the required limit.
    """

    import json

    model_dir = model_dir.resolve()
    architecture_path = model_dir / "architecture_config.json"
    weights_path = model_dir / MODEL_WEIGHTS
    route_path = model_dir / "route_mapping.json"
    route_hash_path = model_dir / "route_mapping.sha256"
    if not all(
        path.is_file()
        for path in (architecture_path, weights_path, route_path, route_hash_path)
    ):
        raise FileNotFoundError(f"incomplete FactorMonitor model bundle: {model_dir}")
    architecture = architecture_from_dict(
        json.loads(architecture_path.read_text(encoding="utf-8"))
    )
    route_mapping = json.loads(route_path.read_text(encoding="utf-8"))
    recorded_hash = route_hash_path.read_text(encoding="utf-8").strip()
    if route_mapping != architecture.family_routes:
        raise RuntimeError("model bundle route mapping differs from architecture")
    if stable_hash(route_mapping) != recorded_hash:
        raise RuntimeError("model bundle route mapping hash mismatch")
    effective_path = model_dir / "effective_config.json"
    if not effective_path.is_file():
        raise FileNotFoundError(f"model bundle lacks effective config: {model_dir}")
    saved_effective = json.loads(effective_path.read_text(encoding="utf-8"))
    load_file, _ = _safetensors()
    custom_state = load_file(str(weights_path))
    variant = (
        saved_effective.get("monitor", {})
        .get("backend_options", {})
        .get("model_variant", "factor_late_fusion")
    )
    if variant == "factor_late_fusion":
        if (
            endpoint_max_padded_tokens_per_backbone_forward is not None
            or joint_max_padded_tokens_per_backbone_forward is not None
        ):
            raise ValueError("forward budget override requires a joint-input model")
        model = FactorMonitor(backbone, architecture)
    elif variant == "factor_joint_fusion":
        if endpoint_max_padded_tokens_per_backbone_forward is not None:
            raise ValueError("endpoint forward budget override requires an endpoint model")
        saved_budget = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("joint_max_padded_tokens_per_backbone_forward")
        )
        model = FactorMonitor(
            backbone,
            architecture,
            joint_encoding=True,
            max_padded_tokens_per_backbone_forward=(
                joint_max_padded_tokens_per_backbone_forward
                if joint_max_padded_tokens_per_backbone_forward is not None
                else saved_budget
            ),
        )
    elif variant == "endpoint_early_fusion":
        if joint_max_padded_tokens_per_backbone_forward is not None:
            raise ValueError("joint FactorMonitor budget override requires that model")
        saved_budget = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_max_padded_tokens_per_backbone_forward")
        )
        role_embedding = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_role_embedding", {"enabled": False})
        )
        endpoint_decision_mode = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_decision_mode", SINGLE_ENDPOINT_DECISION_MODE)
        )
        factor_guided_readout = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_factor_guided_readout")
        )
        signed_criterion_panel = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_signed_criterion_panel")
        )
        layer_mixed = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_layer_mixed")
        )
        model = EndpointBinaryMonitor(
            backbone,
            architecture,
            max_padded_tokens_per_backbone_forward=(
                endpoint_max_padded_tokens_per_backbone_forward
                if endpoint_max_padded_tokens_per_backbone_forward is not None
                else saved_budget
            ),
            endpoint_role_embedding_fields=(
                tuple(role_embedding["field_order"])
                if role_embedding.get("enabled")
                else None
            ),
            endpoint_decision_mode=endpoint_decision_mode,
            factor_guided_readout_config=factor_guided_readout,
            signed_criterion_panel_config=signed_criterion_panel,
            layer_mixed_config=layer_mixed,
        )
    elif variant == "causal_lm_choice":
        if joint_max_padded_tokens_per_backbone_forward is not None:
            raise ValueError("joint FactorMonitor budget override requires that model")
        token_ids = custom_state.get("choice_token_ids")
        if token_ids is None or tuple(token_ids.shape) != (2,):
            raise RuntimeError("causal choice bundle lacks two choice token IDs")
        saved_budget = (
            saved_effective.get("monitor", {})
            .get("backend_options", {})
            .get("endpoint_max_padded_tokens_per_backbone_forward")
        )
        model = CausalLMChoiceMonitor(
            backbone,
            architecture,
            choice_token_ids={
                "A": int(token_ids[0].item()),
                "B": int(token_ids[1].item()),
            },
            max_padded_tokens_per_backbone_forward=(
                endpoint_max_padded_tokens_per_backbone_forward
                if endpoint_max_padded_tokens_per_backbone_forward is not None
                else saved_budget
            ),
        )
    else:
        raise RuntimeError(f"unknown saved model variant: {variant!r}")
    model.load_custom_state_dict(custom_state)
    return model


def assert_round_trip_logits(
    before: Any,
    after: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("round-trip checks require torch") from exc
    maximum = float((before.detach().float() - after.detach().float()).abs().max().item())
    if not torch.allclose(before.detach().float(), after.detach().float(), atol=atol, rtol=rtol):
        raise RuntimeError(f"save/reload logits differ: max_abs_difference={maximum}")
    return {"max_abs_difference": maximum, "atol": float(atol), "rtol": float(rtol)}
