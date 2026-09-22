"""Strict architecture and run-config validation for the Phase-1 monitor."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .encoding import (
    CRITERION_DIFFERENCE_DECISION_MODE,
    CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID,
    DEFAULT_FIELD_BOUNDARY_POLICY,
    ENDPOINT_ROLE_FIELDS,
    FIELD_BOUNDARY_POLICIES,
    FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_CHOICE_CONTRACT_ID,
    FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
    SINGLE_ENDPOINT_DECISION_MODE,
    endpoint_decision_mode,
)


BACKEND_NAME = "factor_monitor_binary"
ARCHITECTURE_KIND = "factor_monitor_architecture"
FACTOR_CONTRACT_ID = "factor_encoding_contract_v1"
EVIDENCE_CONTRACT_ID = "evidence_encoding_contract_v1"
ENDPOINT_CONTRACT_ID = "endpoint_early_fusion_contract_v1"
ENDPOINT_CONTRACT_IDS = (
    ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID,
    CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID,
)
ROUTES = ("cot_direct", "context_grounded")
FACTOR_ENCODER_MODES = ("learned_queries", "field_mean_pool")
DIRECT_MATCHER_MODES = ("concat_binary", "criterion_margin")
DIRECT_EVIDENCE_MODES = ("locator_state", "cross_attention_readout")
RELATIONS = ("Intrinsic", "Extrinsic")
LEVELS = ("L1", "L3")
PHASE1_FAMILIES = (
    "reasoning_method",
    "deception",
    "memory_use",
    "skill_use",
    "sycophancy",
)
NINE_FAMILY_TRAINING_FAMILIES = PHASE1_FAMILIES + (
    "reward_hacking",
    "sabotage",
    "sandbagging",
    "instruction_following",
)
TRUSTWORTHINESS13_TRAINING_FAMILIES = NINE_FAMILY_TRAINING_FAMILIES + (
    "hallucination",
    "bias",
    "privacy",
    "jailbreak",
)
SUPPORTED_TRAINING_FAMILY_SETS = (
    PHASE1_FAMILIES,
    NINE_FAMILY_TRAINING_FAMILIES,
    TRUSTWORTHINESS13_TRAINING_FAMILIES,
)
PHASE1_FAMILY_ROUTES = {
    "reasoning_method": "cot_direct",
    "deception": "cot_direct",
    "memory_use": "context_grounded",
    "skill_use": "context_grounded",
    "sycophancy": "context_grounded",
    "reward_hacking": "cot_direct",
    "sabotage": "cot_direct",
    "sandbagging": "context_grounded",
    "instruction_following": "context_grounded",
    "hallucination": "context_grounded",
    "bias": "context_grounded",
    "privacy": "context_grounded",
    "jailbreak": "context_grounded",
}
PHASE1_FAMILY_RELATIONS = {
    "reasoning_method": "Intrinsic",
    "deception": "Intrinsic",
    "memory_use": "Intrinsic",
    "skill_use": "Intrinsic",
    "sycophancy": "Extrinsic",
    "reward_hacking": "Intrinsic",
    "sabotage": "Intrinsic",
    "sandbagging": "Extrinsic",
    "instruction_following": "Extrinsic",
    "hallucination": "Intrinsic",
    "bias": "Intrinsic",
    "privacy": "Intrinsic",
    "jailbreak": "Extrinsic",
}
SUPPORTED_METRICS = {
    "loss",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "gmean",
    "gmean2",
    "f1",
    "selection_score",
}
PREDICTION_ONLY_METRIC = "prediction_only"
FACTOR_PATH_INTERVENTIONS = (
    "matched",
    "full_shuffled",
    "locator_only_shuffled",
    "matcher_only_shuffled",
    "zero_matcher_factor",
    "zero_located_evidence",
)
LONG_COT_WINDOW_AGGREGATIONS = (
    "max",
    "top2_mean",
    "top2_logmeanexp",
    "noisy_or",
)


def _integer(value: Any, name: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if nonnegative and result < 0:
        raise ConfigurationError(f"{name} must be nonnegative")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _bounded_mean_one(values: Mapping[str, float], *, lower: float, upper: float) -> dict[str, float]:
    """Scale positive values so their arithmetic mean is one within bounds."""

    if not values:
        raise ConfigurationError("metric-driven weighting requires at least one value")
    if lower <= 0 or lower > 1 or upper < 1 or lower > upper:
        raise ConfigurationError("metric-driven weighting bounds must contain one")
    keys = tuple(values)
    if any(not math.isfinite(float(values[key])) or float(values[key]) <= 0 for key in keys):
        raise ConfigurationError("metric-driven weighting values must be finite and positive")

    def mean_at(scale: float) -> float:
        return sum(min(upper, max(lower, scale * float(values[key]))) for key in keys) / len(keys)

    low, high = 0.0, max(upper / float(values[key]) for key in keys)
    for _ in range(80):
        middle = (low + high) / 2.0
        if mean_at(middle) < 1.0:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    result = {key: min(upper, max(lower, scale * float(values[key]))) for key in keys}
    correction = 1.0 / (sum(result.values()) / len(result))
    return {key: min(upper, max(lower, value * correction)) for key, value in result.items()}


def _metric_driven_weighting(
    value: Any, *, training_families: Sequence[str]
) -> dict[str, Any]:
    """Validate and resolve eval-derived family/level training priorities."""

    block = require_mapping(value, "monitor.backend_options.metric_driven_weighting")
    required_fields = {"enabled", "score_source", "formula", "metrics"}
    allowed_fields = required_fields | {
        "family_sampling_overrides",
        "level_loss_overrides",
    }
    require_keys(
        block,
        required_fields,
        "monitor.backend_options.metric_driven_weighting",
    )
    reject_unknown(
        block,
        allowed_fields,
        "monitor.backend_options.metric_driven_weighting",
    )
    if not isinstance(block["enabled"], bool) or not block["enabled"]:
        raise ConfigurationError("metric_driven_weighting.enabled must be true")
    score_source = _text(
        block["score_source"],
        "metric_driven_weighting.score_source",
    )
    formula = require_mapping(
        block["formula"], "metric_driven_weighting.formula"
    )
    formula_fields = {
        "auroc_weight",
        "gmean2_weight",
        "min_multiplier",
        "max_multiplier",
        "sensitivity",
    }
    require_keys(formula, formula_fields, "metric_driven_weighting.formula")
    reject_unknown(formula, formula_fields, "metric_driven_weighting.formula")
    auroc_weight = _number(formula["auroc_weight"], "metric_driven_weighting.formula.auroc_weight", nonnegative=True)
    gmean2_weight = _number(formula["gmean2_weight"], "metric_driven_weighting.formula.gmean2_weight", nonnegative=True)
    if auroc_weight + gmean2_weight <= 0:
        raise ConfigurationError("metric-driven AUROC/G-Mean² weights must sum to a positive value")
    lower = _number(formula["min_multiplier"], "metric_driven_weighting.formula.min_multiplier")
    upper = _number(formula["max_multiplier"], "metric_driven_weighting.formula.max_multiplier")
    sensitivity = _number(formula["sensitivity"], "metric_driven_weighting.formula.sensitivity", nonnegative=True)
    if sensitivity <= 0 or lower <= 0 or lower > 1 or upper < 1 or lower > upper:
        raise ConfigurationError("invalid metric-driven weighting bounds or sensitivity")

    raw_metrics = require_mapping(block["metrics"], "metric_driven_weighting.metrics")
    unknown_families = set(raw_metrics) - set(training_families)
    if unknown_families:
        raise ConfigurationError(
            "metric-driven metrics contain families outside training_scope: "
            f"{sorted(unknown_families)}"
        )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    known_deficits: list[float] = []
    for family, levels in raw_metrics.items():
        level_mapping = require_mapping(
            levels, f"metric_driven_weighting.metrics.{family}"
        )
        unknown_levels = set(level_mapping) - set(LEVELS)
        if unknown_levels:
            raise ConfigurationError(
                f"metric-driven metrics for {family} contain unknown levels: "
                f"{sorted(unknown_levels)}"
            )
        metrics[family] = {}
        for level, scores in level_mapping.items():
            score_mapping = require_mapping(
                scores, f"metric_driven_weighting.metrics.{family}.{level}"
            )
            require_keys(
                score_mapping,
                {"auroc", "gmean2"},
                f"metric_driven_weighting.metrics.{family}.{level}",
            )
            reject_unknown(
                score_mapping,
                {"auroc", "gmean2"},
                f"metric_driven_weighting.metrics.{family}.{level}",
            )
            auroc = _number(
                score_mapping["auroc"],
                f"metric_driven_weighting.metrics.{family}.{level}.auroc",
            )
            gmean2 = _number(
                score_mapping["gmean2"],
                f"metric_driven_weighting.metrics.{family}.{level}.gmean2",
            )
            if not 0 <= auroc <= 1 or not 0 <= gmean2 <= 1:
                raise ConfigurationError("metric-driven AUROC and G-Mean² must be in [0,1]")
            metrics[family][level] = {"auroc": auroc, "gmean2": gmean2}
            known_deficits.append(
                (
                    auroc_weight * (1.0 - auroc)
                    + gmean2_weight * (1.0 - gmean2)
                )
                / (auroc_weight + gmean2_weight)
            )
    neutral_deficit = sum(known_deficits) / len(known_deficits) if known_deficits else 0.25
    deficits: dict[str, dict[str, float]] = {}
    for family in training_families:
        deficits[family] = {}
        for level in LEVELS:
            score = metrics.get(family, {}).get(level)
            deficits[family][level] = (
                neutral_deficit
                if score is None
                else (
                    auroc_weight * (1.0 - score["auroc"])
                    + gmean2_weight * (1.0 - score["gmean2"])
                )
                / (auroc_weight + gmean2_weight)
            )
    center = sum(value for levels in deficits.values() for value in levels.values()) / (
        len(training_families) * len(LEVELS)
    )
    raw_level = {
        f"{family}:{level}": math.exp(sensitivity * (deficits[family][level] - center))
        for family in training_families
        for level in LEVELS
    }
    level_detail = _bounded_mean_one(raw_level, lower=lower, upper=upper)
    family_raw = {
        family: sum(level_detail[f"{family}:{level}"] for level in LEVELS) / len(LEVELS)
        for family in training_families
    }
    observed_families = tuple(family for family in training_families if family in metrics)
    missing_families = set(training_families) - set(observed_families)
    if observed_families:
        observed_sampling = _bounded_mean_one(
            {family: family_raw[family] for family in observed_families},
            lower=lower,
            upper=upper,
        )
        # A family without a trustworthy eval score is deliberately neutral,
        # not silently shifted by the normalization of scored families.
        family_sampling = {
            family: (1.0 if family in missing_families else observed_sampling[family])
            for family in training_families
        }
    else:
        family_sampling = {family: 1.0 for family in training_families}
    level_loss: dict[str, dict[str, float]] = {}
    for family in training_families:
        if family in missing_families:
            level_loss[family] = {level: 1.0 for level in LEVELS}
            continue
        family_mean = family_raw[family]
        level_loss[family] = _bounded_mean_one(
            {
                level: level_detail[f"{family}:{level}"] / family_mean
                for level in LEVELS
            },
            lower=lower,
            upper=upper,
        )
    raw_family_overrides = block.get("family_sampling_overrides", {})
    family_override_mapping = require_mapping(
        raw_family_overrides,
        "metric_driven_weighting.family_sampling_overrides",
    )
    unknown_family_overrides = set(family_override_mapping) - set(training_families)
    if unknown_family_overrides:
        raise ConfigurationError(
            "metric-driven family sampling overrides contain families outside "
            f"training_scope: {sorted(unknown_family_overrides)}"
        )
    family_sampling_overrides = {
        family: _number(
            multiplier,
            f"metric_driven_weighting.family_sampling_overrides.{family}",
        )
        for family, multiplier in family_override_mapping.items()
    }
    if any(
        not math.isfinite(multiplier) or multiplier <= 0
        for multiplier in family_sampling_overrides.values()
    ):
        raise ConfigurationError(
            "metric-driven family sampling override values must be finite and positive"
        )
    if family_sampling_overrides:
        adjusted_family_sampling = {
            family: family_sampling[family]
            * family_sampling_overrides.get(family, 1.0)
            for family in training_families
        }
        observed_adjusted = {
            family: adjusted_family_sampling[family]
            for family in observed_families
        }
        normalized_adjusted = _bounded_mean_one(
            observed_adjusted,
            lower=lower,
            upper=upper,
        )
        family_sampling = {
            family: (
                normalized_adjusted[family]
                if family in observed_families
                else 1.0
            )
            for family in training_families
        }

    raw_overrides = block.get("level_loss_overrides", {})
    override_mapping = require_mapping(
        raw_overrides,
        "metric_driven_weighting.level_loss_overrides",
    )
    unknown_override_families = set(override_mapping) - set(training_families)
    if unknown_override_families:
        raise ConfigurationError(
            "metric-driven level loss overrides contain families outside training_scope: "
            f"{sorted(unknown_override_families)}"
        )
    level_loss_overrides: dict[str, dict[str, float]] = {}
    for family, values in override_mapping.items():
        override_levels = require_mapping(
            values,
            f"metric_driven_weighting.level_loss_overrides.{family}",
        )
        require_keys(
            override_levels,
            set(LEVELS),
            f"metric_driven_weighting.level_loss_overrides.{family}",
        )
        reject_unknown(
            override_levels,
            set(LEVELS),
            f"metric_driven_weighting.level_loss_overrides.{family}",
        )
        normalized = {
            level: _number(
                override_levels[level],
                f"metric_driven_weighting.level_loss_overrides.{family}.{level}",
            )
            for level in LEVELS
        }
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in normalized.values()
        ):
            raise ConfigurationError(
                "metric-driven level loss override values must be finite and positive"
            )
        # Explicit overrides are a deliberate experiment knob.  They may be
        # below the automatic metric-weighting floor (for example, exact
        # count balancing can require 0.47), but remain bounded above by the
        # configured upper multiplier and must still be positive.
        if any(weight > upper for weight in normalized.values()):
            raise ConfigurationError(
                "metric-driven level loss override values must not exceed the formula upper bound"
            )
        if not math.isclose(
            sum(normalized.values()) / len(normalized),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ConfigurationError(
                "metric-driven level loss overrides must have arithmetic mean one"
            )
        level_loss_overrides[family] = normalized
        level_loss[family] = normalized
    resolved = {
        "family_sampling_weights": family_sampling,
        "level_loss_weights": level_loss,
        "deficits": deficits,
        "neutral_policy": "missing_family_and_level_use_exact_multiplier_one",
        "family_sampling_overrides": family_sampling_overrides,
        "level_loss_overrides": level_loss_overrides,
    }
    return {
        "enabled": True,
        "score_source": score_source,
        "formula": {
            "auroc_weight": auroc_weight,
            "gmean2_weight": gmean2_weight,
            "min_multiplier": lower,
            "max_multiplier": upper,
            "sensitivity": sensitivity,
        },
        "metrics": metrics,
        "family_sampling_overrides": family_sampling_overrides,
        "level_loss_overrides": level_loss_overrides,
        "resolved": resolved,
    }


@dataclass(frozen=True)
class FactorMonitorArchitecture:
    architecture_id: str
    backbone_family: str
    hidden_size: int
    reader_dim: int
    num_queries: int
    num_layers: int
    num_attention_heads: int
    head_dim: int
    ffn_dim: int
    dropout: float
    context_num_latents: int
    relation_dim: int
    fusion_hidden_dim: int
    direct_hidden_dims: tuple[int, int]
    grounded_hidden_dims: tuple[int, int]
    relations: tuple[str, ...]
    routes: tuple[str, ...]
    factor_levels: tuple[str, ...]
    phase1_families: tuple[str, ...]
    family_routes: dict[str, str]
    factor_encoder_mode: str = "learned_queries"
    direct_matcher_mode: str = "concat_binary"
    direct_matcher_include_raw_factor: bool = True
    cot_query_self_attention: bool = True
    direct_evidence_mode: str = "locator_state"
    source_path: str | None = None
    source_sha256: str | None = None
    source_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def architecture_from_dict(value: Mapping[str, Any]) -> FactorMonitorArchitecture:
    """Restore the normalized architecture saved inside a model bundle."""

    fields = set(FactorMonitorArchitecture.__dataclass_fields__)
    defaults = {
        "factor_encoder_mode": "learned_queries",
        "direct_matcher_mode": "concat_binary",
        "direct_matcher_include_raw_factor": True,
        "cot_query_self_attention": True,
        "direct_evidence_mode": "locator_state",
    }
    missing = fields - set(value)
    unexpected = set(value) - fields
    if unexpected or not missing.issubset(defaults):
        raise ConfigurationError(
            f"normalized architecture fields differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    normalized = dict(value)
    for key, default in defaults.items():
        normalized.setdefault(key, default)
    for key in (
        "direct_hidden_dims",
        "grounded_hidden_dims",
        "relations",
        "routes",
        "factor_levels",
        "phase1_families",
    ):
        raw = normalized[key]
        if not isinstance(raw, (list, tuple)):
            raise ConfigurationError(f"normalized architecture {key} must be a sequence")
        normalized[key] = tuple(raw)
    if normalized["relations"] != RELATIONS or normalized["routes"] != ROUTES:
        raise ConfigurationError("normalized architecture relation/route enum drift")
    if normalized["factor_levels"] != LEVELS:
        raise ConfigurationError("normalized architecture factor-level drift")
    if normalized["factor_encoder_mode"] not in FACTOR_ENCODER_MODES:
        raise ConfigurationError("normalized architecture factor-encoder mode drift")
    if normalized["direct_matcher_mode"] not in DIRECT_MATCHER_MODES:
        raise ConfigurationError("normalized architecture direct-matcher mode drift")
    if not isinstance(normalized["direct_matcher_include_raw_factor"], bool):
        raise ConfigurationError(
            "normalized architecture direct-matcher raw-Factor flag must be boolean"
        )
    if not isinstance(normalized["cot_query_self_attention"], bool):
        raise ConfigurationError(
            "normalized architecture CoT query-self-attention flag must be boolean"
        )
    if normalized["direct_evidence_mode"] not in DIRECT_EVIDENCE_MODES:
        raise ConfigurationError(
            "normalized architecture direct-Evidence mode drift"
        )
    if (
        normalized["direct_matcher_mode"] == "criterion_margin"
        and normalized["direct_matcher_include_raw_factor"]
    ):
        raise ConfigurationError(
            "normalized criterion-margin architecture cannot include raw Factor"
        )
    if (
        normalized["direct_evidence_mode"] != "locator_state"
        and normalized["direct_matcher_mode"] != "criterion_margin"
    ):
        raise ConfigurationError(
            "normalized cross-attention readout requires criterion-margin matching"
        )
    architecture = FactorMonitorArchitecture(**normalized)
    if architecture.reader_dim != architecture.num_attention_heads * architecture.head_dim:
        raise ConfigurationError("normalized architecture attention dimensions are invalid")
    if architecture.num_queries != 3 or architecture.context_num_latents != 1:
        raise ConfigurationError("normalized architecture Phase-1 query/latent counts are invalid")
    if set(architecture.family_routes) != set(architecture.phase1_families):
        raise ConfigurationError("normalized architecture family route coverage is invalid")
    if architecture.phase1_families not in SUPPORTED_TRAINING_FAMILY_SETS:
        raise ConfigurationError("normalized architecture training-family set is unsupported")
    expected_routes = {
        family: PHASE1_FAMILY_ROUTES[family]
        for family in architecture.phase1_families
    }
    if architecture.family_routes != expected_routes:
        raise ConfigurationError("normalized architecture family routes drifted")
    return architecture


def _validate_block(
    value: Any,
    name: str,
    *,
    query_self_attention: bool | None = None,
    allow_mode: bool = False,
) -> dict[str, Any]:
    block = require_mapping(value, name)
    required = {
        "num_queries",
        "num_layers",
        "num_attention_heads",
        "head_dim",
        "ffn_dim",
        "activation",
        "dropout",
    }
    allowed = set(required) | {
        "num_latents",
        "query_self_attention",
        "cross_query_record_self_attention",
    }
    if allow_mode:
        allowed.add("mode")
    require_keys(block, required, name)
    reject_unknown(block, allowed, name)
    if block["activation"] != "gelu":
        raise ConfigurationError(f"{name}.activation must be 'gelu'")
    for key in ("num_queries", "num_layers", "num_attention_heads", "head_dim", "ffn_dim"):
        _integer(block[key], f"{name}.{key}")
    dropout = _number(block["dropout"], f"{name}.dropout", nonnegative=True)
    if dropout >= 1:
        raise ConfigurationError(f"{name}.dropout must be less than one")
    if query_self_attention is not None and block.get("query_self_attention") is not query_self_attention:
        raise ConfigurationError(
            f"{name}.query_self_attention must be {query_self_attention}"
        )
    return block


def load_architecture(path: Path) -> FactorMonitorArchitecture:
    path = path.resolve()
    value = load_config(path)
    required = {
        "schema_version",
        "kind",
        "architecture_id",
        "backbone",
        "encoding",
        "reader",
        "factor_encoder",
        "context_locator",
        "context_latent_aggregator",
        "cot_locator",
        "grounded_query_fusion",
        "relation_embedding",
        "matchers",
        "evidence_routes",
        "factor_levels",
        "phase1_families",
    }
    require_keys(value, required, "architecture")
    reject_unknown(value, required, "architecture")
    if value["schema_version"] != 1 or value["kind"] != ARCHITECTURE_KIND:
        raise ConfigurationError(
            "architecture requires schema_version=1 and kind=factor_monitor_architecture"
        )

    backbone = require_mapping(value["backbone"], "backbone")
    require_keys(backbone, {"family", "hidden_size"}, "backbone")
    reject_unknown(backbone, {"family", "hidden_size"}, "backbone")
    encoding = require_mapping(value["encoding"], "encoding")
    require_keys(encoding, {"factor_contract", "evidence_contract"}, "encoding")
    reject_unknown(encoding, {"factor_contract", "evidence_contract"}, "encoding")
    if encoding != {
        "factor_contract": FACTOR_CONTRACT_ID,
        "evidence_contract": EVIDENCE_CONTRACT_ID,
    }:
        raise ConfigurationError("architecture encoding contracts do not match Phase 1")
    reader = require_mapping(value["reader"], "reader")
    require_keys(reader, {"dim"}, "reader")
    reject_unknown(reader, {"dim"}, "reader")
    factor = _validate_block(
        value["factor_encoder"], "factor_encoder", allow_mode=True
    )
    context = _validate_block(
        value["context_locator"], "context_locator", query_self_attention=False
    )
    cot = _validate_block(value["cot_locator"], "cot_locator")
    cot_query_self_attention = cot.get("query_self_attention")
    if not isinstance(cot_query_self_attention, bool):
        raise ConfigurationError(
            "cot_locator.query_self_attention must be an explicit boolean"
        )
    if cot.get("cross_query_record_self_attention") is not False:
        raise ConfigurationError(
            "cot_locator.cross_query_record_self_attention must be false"
        )

    reader_dim = _integer(reader["dim"], "reader.dim")
    heads = _integer(factor["num_attention_heads"], "factor_encoder.num_attention_heads")
    head_dim = _integer(factor["head_dim"], "factor_encoder.head_dim")
    if reader_dim != heads * head_dim:
        raise ConfigurationError("reader.dim must equal attention heads * head_dim")
    shared = ("num_layers", "num_attention_heads", "head_dim", "ffn_dim", "dropout")
    for key in shared:
        if context[key] != factor[key] or cot[key] != factor[key]:
            raise ConfigurationError(f"Factor/Context/CoT blocks must share {key}")
    if factor["num_queries"] != 3 or context["num_queries"] != 1 or cot["num_queries"] != 3:
        raise ConfigurationError("Phase 1 query counts must be Factor=3, Context=1, CoT=3")
    if context.get("num_latents") != 1 or cot.get("num_latents") != 3:
        raise ConfigurationError("Phase 1 latent counts must be Context=1 and CoT=3")
    factor_encoder_mode = factor.get("mode", "learned_queries")
    if factor_encoder_mode not in FACTOR_ENCODER_MODES:
        raise ConfigurationError(
            f"factor_encoder.mode must be one of {FACTOR_ENCODER_MODES}"
        )

    aggregator = require_mapping(value["context_latent_aggregator"], "context_latent_aggregator")
    require_keys(aggregator, {"mode", "required_num_latents"}, "context_latent_aggregator")
    reject_unknown(aggregator, {"mode", "required_num_latents"}, "context_latent_aggregator")
    if aggregator != {"mode": "index_zero", "required_num_latents": 1}:
        raise ConfigurationError("Phase 1 context aggregation must be index_zero with M=1")

    fusion = require_mapping(value["grounded_query_fusion"], "grounded_query_fusion")
    fusion_fields = {"input_dim", "hidden_dim", "output_dim", "activation", "dropout", "residual"}
    require_keys(fusion, fusion_fields, "grounded_query_fusion")
    reject_unknown(fusion, fusion_fields, "grounded_query_fusion")
    if (
        fusion["input_dim"] != 2 * reader_dim
        or fusion["output_dim"] != reader_dim
        or fusion["activation"] != "gelu"
        or fusion["residual"] != "query"
        or _number(fusion["dropout"], "grounded_query_fusion.dropout", nonnegative=True)
        != float(factor["dropout"])
    ):
        raise ConfigurationError("grounded_query_fusion dimensions/semantics are invalid")

    relation = require_mapping(value["relation_embedding"], "relation_embedding")
    require_keys(relation, {"values", "dim"}, "relation_embedding")
    reject_unknown(relation, {"values", "dim"}, "relation_embedding")
    if tuple(relation["values"]) != RELATIONS:
        raise ConfigurationError(f"relation values must be {RELATIONS}")
    relation_dim = _integer(relation["dim"], "relation_embedding.dim")

    matchers = require_mapping(value["matchers"], "matchers")
    require_keys(matchers, {"direct", "grounded"}, "matchers")
    reject_unknown(matchers, {"direct", "grounded"}, "matchers")
    parsed_matchers: dict[str, tuple[int, tuple[int, int]]] = {}
    direct_matcher_mode = "concat_binary"
    direct_matcher_include_raw_factor = True
    direct_evidence_mode = "locator_state"
    for name in ("direct", "grounded"):
        matcher = require_mapping(matchers[name], f"matchers.{name}")
        required_fields = {
            "input_dim",
            "hidden_dims",
            "output_dim",
            "activation",
            "dropout",
        }
        allowed_fields = set(required_fields)
        if name == "direct":
            allowed_fields.update({"mode", "include_raw_factor", "evidence_input"})
            direct_matcher_mode = matcher.get("mode", "concat_binary")
            if direct_matcher_mode not in DIRECT_MATCHER_MODES:
                raise ConfigurationError(
                    f"matchers.direct.mode must be one of {DIRECT_MATCHER_MODES}"
                )
            direct_matcher_include_raw_factor = matcher.get(
                "include_raw_factor", True
            )
            if not isinstance(direct_matcher_include_raw_factor, bool):
                raise ConfigurationError(
                    "matchers.direct.include_raw_factor must be boolean"
                )
            direct_evidence_mode = matcher.get("evidence_input", "locator_state")
            if direct_evidence_mode not in DIRECT_EVIDENCE_MODES:
                raise ConfigurationError(
                    "matchers.direct.evidence_input must be one of "
                    f"{DIRECT_EVIDENCE_MODES}"
                )
            if direct_matcher_mode == "criterion_margin":
                if direct_matcher_include_raw_factor:
                    raise ConfigurationError(
                        "matchers.direct criterion_margin requires "
                        "include_raw_factor=false"
                    )
                expected = reader_dim + relation_dim
            else:
                if direct_evidence_mode != "locator_state":
                    raise ConfigurationError(
                        "cross-attention readout requires criterion-margin matching"
                    )
                expected = (
                    (6 if direct_matcher_include_raw_factor else 3) * reader_dim
                    + relation_dim
                )
        else:
            expected = 7 * reader_dim + relation_dim
        require_keys(matcher, required_fields, f"matchers.{name}")
        reject_unknown(matcher, allowed_fields, f"matchers.{name}")
        hidden_dims = matcher["hidden_dims"]
        if not isinstance(hidden_dims, list) or len(hidden_dims) != 2:
            raise ConfigurationError(f"matchers.{name}.hidden_dims must have two entries")
        parsed_hidden = tuple(_integer(item, f"matchers.{name}.hidden_dims") for item in hidden_dims)
        if matcher["input_dim"] != expected or matcher["output_dim"] != 1:
            raise ConfigurationError(f"matchers.{name} dimensions are invalid")
        if matcher["activation"] != "gelu":
            raise ConfigurationError(f"matchers.{name}.activation must be gelu")
        matcher_dropout = _number(
            matcher["dropout"], f"matchers.{name}.dropout", nonnegative=True
        )
        if matcher_dropout != float(factor["dropout"]) or matcher_dropout >= 1:
            raise ConfigurationError(
                f"matchers.{name}.dropout must match the shared architecture dropout"
            )
        parsed_matchers[name] = (expected, parsed_hidden)  # type: ignore[assignment]

    routes = require_mapping(value["evidence_routes"], "evidence_routes")
    require_keys(routes, {"values", "family_mapping"}, "evidence_routes")
    reject_unknown(routes, {"values", "family_mapping"}, "evidence_routes")
    if tuple(routes["values"]) != ROUTES:
        raise ConfigurationError(f"route values must be {ROUTES}")
    families_value = value["phase1_families"]
    if not isinstance(families_value, list) or not families_value:
        raise ConfigurationError("phase1_families must be a non-empty list")
    families = tuple(_text(item, "phase1 family") for item in families_value)
    if len(families) != len(set(families)):
        raise ConfigurationError("phase1_families contains duplicates")
    if families not in SUPPORTED_TRAINING_FAMILY_SETS:
        raise ConfigurationError(
            "phase1_families must be one of the frozen training-family sets: "
            f"{SUPPORTED_TRAINING_FAMILY_SETS}"
        )
    family_routes = require_mapping(routes["family_mapping"], "evidence_routes.family_mapping")
    if set(family_routes) != set(families):
        raise ConfigurationError("family route mapping must cover exactly phase1_families")
    if any(route not in ROUTES for route in family_routes.values()):
        raise ConfigurationError("family route mapping contains an unknown route")
    expected_routes = {family: PHASE1_FAMILY_ROUTES[family] for family in families}
    if family_routes != expected_routes:
        raise ConfigurationError("Phase-1 family route mapping differs from the frozen design")
    if tuple(value["factor_levels"]) != LEVELS:
        raise ConfigurationError(f"factor_levels must be {LEVELS}")

    architecture = FactorMonitorArchitecture(
        architecture_id=_text(value["architecture_id"], "architecture_id"),
        backbone_family=_text(backbone["family"], "backbone.family"),
        hidden_size=_integer(backbone["hidden_size"], "backbone.hidden_size"),
        reader_dim=reader_dim,
        num_queries=3,
        num_layers=_integer(factor["num_layers"], "factor_encoder.num_layers"),
        num_attention_heads=heads,
        head_dim=head_dim,
        ffn_dim=_integer(factor["ffn_dim"], "factor_encoder.ffn_dim"),
        dropout=float(factor["dropout"]),
        context_num_latents=1,
        relation_dim=relation_dim,
        fusion_hidden_dim=_integer(fusion["hidden_dim"], "grounded_query_fusion.hidden_dim"),
        direct_hidden_dims=parsed_matchers["direct"][1],
        grounded_hidden_dims=parsed_matchers["grounded"][1],
        relations=RELATIONS,
        routes=ROUTES,
        factor_levels=LEVELS,
        phase1_families=families,
        family_routes={str(key): str(route) for key, route in family_routes.items()},
        factor_encoder_mode=factor_encoder_mode,
        direct_matcher_mode=direct_matcher_mode,
        direct_matcher_include_raw_factor=direct_matcher_include_raw_factor,
        cot_query_self_attention=cot_query_self_attention,
        direct_evidence_mode=direct_evidence_mode,
        source_path=str(path),
        source_sha256=file_sha256(path),
        source_hash=stable_hash(value),
    )
    # The monitor head is frozen across the Phase-1 backbone variants.  The
    # backbone identity/width is the one intentional model-dependent field:
    # these causal-LM families expose different hidden sizes, while all
    # reader/locator/matcher dimensions below must remain unchanged.
    supported_backbones = {
        "Qwen3.5-4B": 2560,
        "Qwen3.5-9B": 4096,
        "Llama-3.2-3B": 3072,
    }
    expected_hidden_size = supported_backbones.get(architecture.backbone_family)
    if expected_hidden_size is None or architecture.hidden_size != expected_hidden_size:
        raise ConfigurationError(
            "unsupported Phase-1 backbone family/hidden size: "
            f"{architecture.backbone_family}/{architecture.hidden_size}; "
            f"expected one of {supported_backbones}"
        )
    frozen_dimensions = {
        "reader_dim": 768,
        "num_layers": 2,
        "num_attention_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "dropout": 0.1,
        "relation_dim": 64,
        "fusion_hidden_dim": 1536,
        "direct_hidden_dims": (1536, 512),
        "grounded_hidden_dims": (1536, 512),
    }
    drift = {
        key: (getattr(architecture, key), expected)
        for key, expected in frozen_dimensions.items()
        if getattr(architecture, key) != expected
    }
    if drift:
        raise ConfigurationError(f"Phase-1 frozen architecture drift: {drift}")
    architecture_variant = (
        architecture.factor_encoder_mode,
        architecture.direct_matcher_mode,
        architecture.direct_matcher_include_raw_factor,
        architecture.cot_query_self_attention,
        architecture.direct_evidence_mode,
    )
    expected_architecture_ids = {
        ("learned_queries", "concat_binary", True, True, "locator_state"): "factor_monitor_phase1",
        ("field_mean_pool", "concat_binary", True, True, "locator_state"): "factor_monitor_phase1_factor_pool",
        (
            "field_mean_pool",
            "concat_binary",
            False,
            True,
            "locator_state",
        ): "factor_monitor_phase1_factor_pool_no_direct_factor",
        (
            "field_mean_pool",
            "criterion_margin",
            False,
            True,
            "locator_state",
        ): "factor_monitor_phase1_factor_pool_criterion_margin",
        (
            "field_mean_pool",
            "criterion_margin",
            False,
            False,
            "locator_state",
        ): "factor_monitor_phase1_factor_pool_criterion_margin_isolated_queries",
        (
            "field_mean_pool",
            "criterion_margin",
            False,
            False,
            "cross_attention_readout",
        ): "factor_monitor_phase1_factor_pool_criterion_margin_isolated_queries_evidence_readout",
    }
    if architecture_variant not in expected_architecture_ids:
        raise ConfigurationError(
            "unsupported Phase-1 Factor encoder/direct-matcher/CoT-query mode combination: "
            f"{architecture_variant}"
        )
    expected_architecture_id = expected_architecture_ids[architecture_variant]
    if architecture.architecture_id != expected_architecture_id:
        raise ConfigurationError(
            "architecture_id does not match the architecture variant: "
            f"expected {expected_architecture_id!r}"
        )
    return architecture


def _required_mapping(parent: dict[str, Any], key: str, fields: set[str]) -> dict[str, Any]:
    value = require_mapping(parent.get(key), f"monitor.backend_options.{key}")
    require_keys(value, fields, f"monitor.backend_options.{key}")
    reject_unknown(value, fields, f"monitor.backend_options.{key}")
    return value


def _endpoint_factor_guided_readout(
    value: Any,
) -> dict[str, Any]:
    """Validate the explicit architecture delta without inventing defaults."""

    block = require_mapping(
        value, "monitor.backend_options.endpoint_factor_guided_readout"
    )
    if "enabled" not in block:
        raise ConfigurationError("endpoint_factor_guided_readout requires enabled")
    if not isinstance(block["enabled"], bool):
        raise ConfigurationError("endpoint_factor_guided_readout.enabled must be boolean")
    if not block["enabled"]:
        reject_unknown(block, {"enabled"}, "endpoint_factor_guided_readout")
        return {"enabled": False}
    fields = {
        "enabled",
        "readout_dim",
        "factor_fields",
        "evidence_fields",
        "attention",
        "interaction_features",
        "interaction_hidden_dim",
        "dropout",
        "output_initialization",
    }
    require_keys(block, fields, "endpoint_factor_guided_readout")
    reject_unknown(block, fields, "endpoint_factor_guided_readout")
    if _integer(block["readout_dim"], "endpoint_factor_guided_readout.readout_dim") != 256:
        raise ConfigurationError("guided readout dimension is frozen at 256")
    if block["factor_fields"] != ["factor"]:
        raise ConfigurationError("guided readout Factor fields must be [factor]")
    if block["evidence_fields"] != ["context", "cot"]:
        raise ConfigurationError("guided readout evidence fields must be [context, cot]")
    if block["attention"] != "scaled_dot_product_softmax":
        raise ConfigurationError("guided readout requires scaled dot-product softmax")
    if block["interaction_features"] != ["hadamard", "absolute_difference"]:
        raise ConfigurationError(
            "guided readout interaction features must be hadamard and absolute_difference"
        )
    if _integer(
        block["interaction_hidden_dim"],
        "endpoint_factor_guided_readout.interaction_hidden_dim",
    ) != 256:
        raise ConfigurationError("guided interaction hidden dimension is frozen at 256")
    if _number(
        block["dropout"], "endpoint_factor_guided_readout.dropout", nonnegative=True
    ) != 0.0:
        raise ConfigurationError("guided readout dropout is frozen at zero")
    if block["output_initialization"] != "zeros":
        raise ConfigurationError("guided readout output must use zero initialization")
    return dict(block)


def _endpoint_signed_criterion_panel(value: Any) -> dict[str, Any]:
    block = require_mapping(
        value, "monitor.backend_options.endpoint_signed_criterion_panel"
    )
    if "enabled" not in block:
        raise ConfigurationError("endpoint_signed_criterion_panel requires enabled")
    if not isinstance(block["enabled"], bool):
        raise ConfigurationError("endpoint_signed_criterion_panel.enabled must be boolean")
    if not block["enabled"]:
        reject_unknown(block, {"enabled"}, "endpoint_signed_criterion_panel")
        return {"enabled": False}
    fields = {
        "enabled",
        "projection_dim",
        "pooling",
        "fields",
        "score",
        "residual_scale_initialization",
        "dropout",
    }
    require_keys(block, fields, "endpoint_signed_criterion_panel")
    reject_unknown(block, fields, "endpoint_signed_criterion_panel")
    if _integer(
        block["projection_dim"], "endpoint_signed_criterion_panel.projection_dim"
    ) != 256:
        raise ConfigurationError("signed criterion panel projection dimension is frozen at 256")
    if block["pooling"] != "masked_mean":
        raise ConfigurationError("signed criterion panel requires masked_mean pooling")
    if block["fields"] != ["factor", "relation", "applied", "not_applied"]:
        raise ConfigurationError("signed criterion panel fields differ from the frozen order")
    if block["score"] != "shared_cosine_difference":
        raise ConfigurationError("signed criterion panel requires shared_cosine_difference")
    if block["residual_scale_initialization"] != "zeros":
        raise ConfigurationError("signed criterion panel residual scale must start at zero")
    if _number(
        block["dropout"], "endpoint_signed_criterion_panel.dropout", nonnegative=True
    ) != 0.0:
        raise ConfigurationError("signed criterion panel dropout is frozen at zero")
    return dict(block)


def _endpoint_layer_mixed(value: Any) -> dict[str, Any]:
    block = require_mapping(value, "monitor.backend_options.endpoint_layer_mixed")
    if "enabled" not in block:
        raise ConfigurationError("endpoint_layer_mixed requires enabled")
    if not isinstance(block["enabled"], bool):
        raise ConfigurationError("endpoint_layer_mixed.enabled must be boolean")
    if not block["enabled"]:
        reject_unknown(block, {"enabled"}, "endpoint_layer_mixed")
        return {"enabled": False}
    fields = {
        "enabled",
        "layers_below_final",
        "combination",
        "normalization",
        "heads",
        "residual_initialization",
    }
    require_keys(block, fields, "endpoint_layer_mixed")
    reject_unknown(block, fields, "endpoint_layer_mixed")
    if block["layers_below_final"] != [4, 8]:
        raise ConfigurationError("layer-mixed endpoint layers are frozen at [4, 8]")
    expected = {
        "combination": "zero_initialized_logit_residual",
        "normalization": "separate_layernorm",
        "heads": "separate_linear",
        "residual_initialization": "zeros",
    }
    for key, wanted in expected.items():
        if block[key] != wanted:
            raise ConfigurationError(f"endpoint_layer_mixed.{key} must be {wanted!r}")
    return dict(block)


def _long_cot_windowing(value: Any, *, config_path: Path) -> dict[str, Any]:
    """Validate the explicit inference-only long-CoT window contract."""

    block = require_mapping(value, "evaluation_scope.long_cot_windowing")
    fields = {
        "enabled",
        "trigger_endpoint_tokens",
        "max_endpoint_tokens",
        "cot_overlap_tokens",
        "context_policy",
        "overlap_fallback_policy",
        "coverage_policy",
        "selection_unit",
        "aggregations",
        "comparison_mode",
        "frozen_window_manifest",
        "direct_reference_scores",
    }
    require_keys(
        block,
        {
            "enabled",
            "trigger_endpoint_tokens",
            "max_endpoint_tokens",
            "cot_overlap_tokens",
            "context_policy",
            "coverage_policy",
            "selection_unit",
            "aggregations",
        },
        "evaluation_scope.long_cot_windowing",
    )
    reject_unknown(block, fields, "evaluation_scope.long_cot_windowing")
    if block["enabled"] is not True:
        raise ConfigurationError(
            "long_cot_windowing must be omitted when disabled; enabled must be true"
        )
    trigger = _integer(
        block["trigger_endpoint_tokens"],
        "evaluation_scope.long_cot_windowing.trigger_endpoint_tokens",
    )
    maximum = _integer(
        block["max_endpoint_tokens"],
        "evaluation_scope.long_cot_windowing.max_endpoint_tokens",
    )
    overlap = _integer(
        block["cot_overlap_tokens"],
        "evaluation_scope.long_cot_windowing.cot_overlap_tokens",
    )
    if trigger != maximum:
        raise ConfigurationError(
            "long_cot_windowing trigger_endpoint_tokens must equal max_endpoint_tokens"
        )
    if overlap >= maximum:
        raise ConfigurationError(
            "long_cot_windowing cot_overlap_tokens must be below max_endpoint_tokens"
        )
    context_policy = block["context_policy"]
    if context_policy not in {
        "repeat_full",
        "repeat_full_when_feasible_else_window_context",
    }:
        raise ConfigurationError(
            "long_cot_windowing context_policy must be repeat_full or "
            "repeat_full_when_feasible_else_window_context"
        )
    overlap_fallback_policy = block.get("overlap_fallback_policy", "error")
    if overlap_fallback_policy not in {
        "error",
        "half_field_capacity_when_needed",
    }:
        raise ConfigurationError(
            "long_cot_windowing overlap_fallback_policy must be error or "
            "half_field_capacity_when_needed"
        )
    if (
        context_policy == "repeat_full"
        and overlap_fallback_policy != "error"
    ):
        raise ConfigurationError(
            "repeat_full long_cot_windowing requires overlap_fallback_policy=error"
        )
    if block["coverage_policy"] != "complete_no_truncation":
        raise ConfigurationError(
            "long_cot_windowing requires coverage_policy=complete_no_truncation"
        )
    if block["selection_unit"] != "trajectory_any_query_over_trigger":
        raise ConfigurationError(
            "long_cot_windowing requires trajectory_any_query_over_trigger selection"
        )
    aggregations = block["aggregations"]
    if (
        not isinstance(aggregations, list)
        or not aggregations
        or any(value not in LONG_COT_WINDOW_AGGREGATIONS for value in aggregations)
        or len(aggregations) != len(set(aggregations))
    ):
        raise ConfigurationError(
            "long_cot_windowing aggregations must be a non-empty unique subset of "
            "[max, top2_mean, top2_logmeanexp, noisy_or]"
        )
    comparison_mode = block.get("comparison_mode", "direct_reference")
    if comparison_mode not in {"direct_reference", "standalone", "teacher_window_audit"}:
        raise ConfigurationError(
            "long_cot_windowing comparison_mode must be direct_reference, "
            "standalone, or teacher_window_audit"
        )
    reference_value = block.get("direct_reference_scores")
    frozen_value = block.get("frozen_window_manifest")
    if comparison_mode == "direct_reference":
        if reference_value is None or frozen_value is not None:
            raise ConfigurationError(
                "direct-reference long_cot_windowing requires direct_reference_scores "
                "and forbids frozen_window_manifest"
            )
    elif comparison_mode == "teacher_window_audit":
        if frozen_value is None or reference_value is not None:
            raise ConfigurationError(
                "teacher_window_audit requires frozen_window_manifest and forbids "
                "direct_reference_scores"
            )
        if aggregations != ["top2_mean", "top2_logmeanexp"]:
            raise ConfigurationError(
                "teacher_window_audit freezes aggregations to "
                "[top2_mean, top2_logmeanexp]"
            )
    else:
        if reference_value is not None or frozen_value is not None:
            raise ConfigurationError(
                "standalone long_cot_windowing forbids direct_reference_scores and "
                "frozen_window_manifest"
            )
    reference = None
    if reference_value is not None:
        reference = resolve_path(
            _text(
                reference_value,
                "evaluation_scope.long_cot_windowing.direct_reference_scores",
            ),
            relative_to=config_path.parent,
        )
        if not reference.is_file():
            raise ConfigurationError(
                f"long_cot_windowing direct reference scores do not exist: {reference}"
            )
    frozen = None
    if frozen_value is not None:
        frozen = resolve_path(
            _text(
                frozen_value,
                "evaluation_scope.long_cot_windowing.frozen_window_manifest",
            ),
            relative_to=config_path.parent,
        )
        if not frozen.is_file():
            raise ConfigurationError(
                f"long_cot_windowing frozen manifest does not exist: {frozen}"
            )
        frozen_payload = load_config(frozen)
        if frozen_payload.get("schema_version") != 1:
            raise ConfigurationError("frozen long-CoT manifest schema_version must be 1")
        frozen_policy = require_mapping(
            frozen_payload.get("policy"), "frozen long-CoT manifest policy"
        )
        expected_policy = {
            "trigger_endpoint_tokens": trigger,
            "max_endpoint_tokens": maximum,
            "cot_overlap_tokens": overlap,
            "context_policy": context_policy,
            "coverage_policy": "complete_no_truncation",
            "selection_unit": "trajectory_any_query_over_trigger",
        }
        for key, expected in expected_policy.items():
            if frozen_policy.get(key) != expected:
                raise ConfigurationError(
                    f"frozen long-CoT manifest policy {key} differs from config"
                )
        frozen_fallback = frozen_policy.get("overlap_fallback_policy", "error")
        if frozen_fallback != overlap_fallback_policy:
            raise ConfigurationError(
                "frozen long-CoT manifest overlap fallback differs from config"
            )
        trajectories = frozen_payload.get("trajectories")
        if not isinstance(trajectories, list) or not trajectories:
            raise ConfigurationError(
                "frozen long-CoT manifest must contain selected trajectories"
            )
    return {
        "enabled": True,
        "trigger_endpoint_tokens": trigger,
        "max_endpoint_tokens": maximum,
        "cot_overlap_tokens": overlap,
        "context_policy": context_policy,
        "overlap_fallback_policy": overlap_fallback_policy,
        "coverage_policy": "complete_no_truncation",
        "selection_unit": "trajectory_any_query_over_trigger",
        "aggregations": list(aggregations),
        "comparison_mode": comparison_mode,
        **(
            {
                "direct_reference_scores": {
                    "path": str(reference),
                    "sha256": file_sha256(reference),
                }
            }
            if reference is not None
            else {}
        ),
        **(
            {
                "frozen_window_manifest": {
                    "path": str(frozen),
                    "sha256": file_sha256(frozen),
                }
            }
            if frozen is not None
            else {}
        ),
    }


def _teacher_distillation(
    value: Any, *, config_path: Path
) -> dict[str, Any]:
    """Bind one immutable window-scoring Teacher to the joint MIL run."""

    block = require_mapping(
        value, "monitor.backend_options.long_cot_mil.teacher_distillation"
    )
    fields = {
        "teacher_role",
        "score_file",
        "score_effective_config",
        "teacher_model_bundle",
        "frozen_window_manifest",
        "objective",
        "temperature",
        "loss_weight",
        "target_scope",
        "window_scope",
        "mining_policy",
    }
    require_keys(block, fields, "long_cot_mil.teacher_distillation")
    reject_unknown(block, fields, "long_cot_mil.teacher_distillation")
    frozen_values = {
        "objective": "applied_long_window_listwise_kl",
        "target_scope": "long_applied_only",
        "window_scope": "student_active_windows",
        "mining_policy": "student_only",
    }
    for key, expected in frozen_values.items():
        if block[key] != expected:
            raise ConfigurationError(
                f"teacher_distillation.{key} must be {expected!r}"
            )
    role = block["teacher_role"]
    if role not in {"t32", "t65"}:
        raise ConfigurationError("teacher_distillation.teacher_role must be t32 or t65")
    temperature = _number(
        block["temperature"], "teacher_distillation.temperature", nonnegative=True
    )
    loss_weight = _number(
        block["loss_weight"], "teacher_distillation.loss_weight", nonnegative=True
    )
    if temperature != 2.0 or loss_weight != 0.25:
        raise ConfigurationError(
            "the preregistered Teacher experiment requires temperature=2.0 and "
            "loss_weight=0.25"
        )

    def bound_file(field: str) -> dict[str, str]:
        path = resolve_path(
            _text(block[field], f"teacher_distillation.{field}"),
            relative_to=config_path.parent,
        )
        if not path.is_file():
            raise ConfigurationError(
                f"teacher_distillation.{field} does not exist: {path}"
            )
        return {"path": str(path), "sha256": file_sha256(path)}

    model_bundle = resolve_path(
        _text(
            block["teacher_model_bundle"],
            "teacher_distillation.teacher_model_bundle",
        ),
        relative_to=config_path.parent,
    )
    factor_monitor_path = model_bundle / "factor_monitor.safetensors"
    adapter_path = model_bundle / "backbone" / "adapter_model.safetensors"
    if not factor_monitor_path.is_file() or not adapter_path.is_file():
        raise ConfigurationError(
            f"teacher_distillation Teacher model bundle is incomplete: {model_bundle}"
        )
    return {
        "teacher_role": role,
        "score_file": bound_file("score_file"),
        "score_effective_config": bound_file("score_effective_config"),
        "teacher_model_bundle": {
            "path": str(model_bundle),
            "factor_monitor_sha256": file_sha256(factor_monitor_path),
            "adapter_model_sha256": file_sha256(adapter_path),
        },
        "frozen_window_manifest": bound_file("frozen_window_manifest"),
        **frozen_values,
        "temperature": temperature,
        "loss_weight": loss_weight,
        "record_weighting": "inherit_main_loss",
        "hierarchy": "factor_instance_then_training_unit",
    }


def _long_cot_mil(value: Any, *, config_path: Path) -> dict[str, Any]:
    """Validate the preregistered continuation or full-epoch MIL contract."""

    block = require_mapping(value, "monitor.backend_options.long_cot_mil")
    fields = {
        "enabled",
        "training_schedule",
        "trigger_endpoint_tokens",
        "max_endpoint_tokens",
        "cot_overlap_tokens",
        "context_policy",
        "overlap_fallback_policy",
        "coverage_policy",
        "selection_unit",
        "exact_max_windows",
        "active_windows",
        "exploitation_windows",
        "exploration_windows",
        "initial_cache",
        "cache_refresh",
        "pooling",
        "long_short_ratio",
        "long_exposures",
        "long_sampling",
        "short_replay_sampling",
        "expected_long_units",
        "expected_short_units",
        "sampled_long_units_per_exposure",
        "optimizer_steps_per_epoch",
        "replay_only",
        "evidence_audit_max_failure_ratio",
        "short_validation_auroc_floor",
        "teacher_distillation",
    }
    optional_fields = {
        "training_schedule",
        "expected_short_units",
        "optimizer_steps_per_epoch",
        "teacher_distillation",
    }
    require_keys(
        block,
        fields - optional_fields,
        "monitor.backend_options.long_cot_mil",
    )
    reject_unknown(block, fields, "monitor.backend_options.long_cot_mil")
    if block["enabled"] is not True:
        raise ConfigurationError("long_cot_mil must be omitted when disabled")
    trigger = _integer(block["trigger_endpoint_tokens"], "long_cot_mil.trigger_endpoint_tokens")
    maximum = _integer(block["max_endpoint_tokens"], "long_cot_mil.max_endpoint_tokens")
    overlap = _integer(block["cot_overlap_tokens"], "long_cot_mil.cot_overlap_tokens")
    if trigger != maximum or (maximum, overlap) not in {
        (33024, 6144),
        (16384, 6144),
    }:
        raise ConfigurationError(
            "long_cot_mil requires trigger=max and the registered window pair "
            "(33024, 6144) or (16384, 6144)"
        )
    schedule = block.get("training_schedule", "two_stream_continuation")
    if schedule not in {"two_stream_continuation", "full_epoch_joint"}:
        raise ConfigurationError(
            f"unsupported long_cot_mil.training_schedule: {schedule!r}"
        )
    common_expected = {
        "context_policy": "repeat_full_when_feasible_else_window_context",
        "overlap_fallback_policy": "half_field_capacity_when_needed",
        "coverage_policy": "complete_no_truncation",
        "selection_unit": "trajectory_any_query_over_trigger",
    }
    for key, wanted in common_expected.items():
        if block[key] != wanted:
            raise ConfigurationError(f"long_cot_mil.{key} must be {wanted!r}")
    schedule_expected = (
        {
            "initial_cache": "warm_start_full_no_grad",
            "cache_refresh": "every_long_exposure",
            "long_short_ratio": "1:1",
            "long_sampling": "shuffled_without_replacement_pad_to_288",
            "short_replay_sampling": "nine_family_uniform",
        }
        if schedule == "two_stream_continuation"
        else {
            "initial_cache": "position_stratified_no_model",
            "cache_refresh": "epoch_boundary",
            "long_short_ratio": "natural",
            "long_sampling": "exhaustive_without_replacement",
            "short_replay_sampling": "exhaustive_without_replacement",
        }
    )
    for key, wanted in schedule_expected.items():
        if block[key] != wanted:
            raise ConfigurationError(f"long_cot_mil.{key} must be {wanted!r}")
    integer_expected = {
        "exact_max_windows": 4,
        "active_windows": 4,
        "exploitation_windows": 2,
        "exploration_windows": 2,
    }
    for key, wanted in integer_expected.items():
        if _integer(block[key], f"long_cot_mil.{key}") != wanted:
            raise ConfigurationError(f"long_cot_mil.{key} must be {wanted}")
    expected_long_value = block["expected_long_units"]
    expected_long_units = (
        "auto"
        if expected_long_value == "auto"
        else _integer(expected_long_value, "long_cot_mil.expected_long_units")
    )
    sampled_long_value = block["sampled_long_units_per_exposure"]
    sampled_long_units = (
        "auto"
        if sampled_long_value == "auto"
        else _integer(
            sampled_long_value,
            "long_cot_mil.sampled_long_units_per_exposure",
        )
    )
    if expected_long_units != "auto" and expected_long_units <= 0:
        raise ConfigurationError("long_cot_mil.expected_long_units must be positive")
    if schedule == "full_epoch_joint":
        if (
            expected_long_units != "auto"
            and sampled_long_units != "auto"
            and sampled_long_units != expected_long_units
        ):
            raise ConfigurationError(
                "full_epoch_joint must sample every long unit once per epoch"
            )
    elif expected_long_units != 286 or sampled_long_units != 288:
        raise ConfigurationError(
            "two_stream_continuation requires 286 long units and 288 sampled units"
        )
    exposures = _integer(block["long_exposures"], "long_cot_mil.long_exposures")
    if exposures not in {1, 3}:
        raise ConfigurationError("long_cot_mil.long_exposures must be one or three")
    expected_short_units = block.get("expected_short_units")
    optimizer_steps_per_epoch = block.get("optimizer_steps_per_epoch")
    if schedule == "full_epoch_joint":
        if expected_short_units is None or optimizer_steps_per_epoch is None:
            raise ConfigurationError(
                "full_epoch_joint requires expected_short_units and "
                "optimizer_steps_per_epoch"
            )
        expected_short_units = (
            "auto"
            if expected_short_units == "auto"
            else _integer(expected_short_units, "long_cot_mil.expected_short_units")
        )
        if expected_short_units != "auto" and expected_short_units <= 0:
            raise ConfigurationError(
                "long_cot_mil.expected_short_units must be positive"
            )
        optimizer_steps_per_epoch = (
            "auto"
            if optimizer_steps_per_epoch == "auto"
            else _integer(
                optimizer_steps_per_epoch,
                "long_cot_mil.optimizer_steps_per_epoch",
            )
        )
        if expected_long_units != "auto" and expected_short_units != "auto":
            expected_steps = (expected_long_units + expected_short_units + 3) // 4
            if optimizer_steps_per_epoch != "auto" and optimizer_steps_per_epoch != expected_steps:
                raise ConfigurationError(
                    "full_epoch_joint optimizer_steps_per_epoch must equal "
                    "ceil((expected_long_units + expected_short_units) / 4); "
                    f"expected {expected_steps}, found {optimizer_steps_per_epoch}"
                )
        if exposures != 3:
            raise ConfigurationError(
                "full_epoch_joint requires three long-unit exposures"
            )
        if block["replay_only"] is not False:
            raise ConfigurationError("full_epoch_joint cannot be replay-only")
    else:
        if expected_short_units is not None or optimizer_steps_per_epoch is not None:
            raise ConfigurationError(
                "continuation MIL must omit full-epoch unit and cadence fields"
            )
        optimizer_steps_per_epoch = 144
    if block["pooling"] not in {"top2_probability_mean", "top2_logmeanexp"}:
        raise ConfigurationError("long_cot_mil.pooling is unsupported")
    if not isinstance(block["replay_only"], bool):
        raise ConfigurationError("long_cot_mil.replay_only must be boolean")
    failure_ratio = _number(
        block["evidence_audit_max_failure_ratio"],
        "long_cot_mil.evidence_audit_max_failure_ratio",
        nonnegative=True,
    )
    if failure_ratio != 0.1:
        raise ConfigurationError("MIL evidence audit failure ratio is frozen at 0.1")
    short_floor = _number(
        block["short_validation_auroc_floor"],
        "long_cot_mil.short_validation_auroc_floor",
        nonnegative=True,
    )
    if not 0 < short_floor <= 1:
        raise ConfigurationError("MIL short validation AUROC floor must be in (0,1]")
    teacher_distillation = None
    if block.get("teacher_distillation") is not None:
        if schedule != "full_epoch_joint" or block["replay_only"] is not False:
            raise ConfigurationError(
                "Teacher distillation requires non-replay full_epoch_joint MIL"
            )
        if block["pooling"] != "top2_probability_mean":
            raise ConfigurationError("Teacher distillation requires PMean MIL pooling")
        teacher_distillation = _teacher_distillation(
            block["teacher_distillation"], config_path=config_path
        )
    return {
        **dict(block),
        "training_schedule": schedule,
        "trigger_endpoint_tokens": trigger,
        "max_endpoint_tokens": maximum,
        "cot_overlap_tokens": overlap,
        "long_exposures": exposures,
        "expected_short_units": expected_short_units,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "evidence_audit_max_failure_ratio": failure_ratio,
        "short_validation_auroc_floor": short_floor,
        **(
            {"teacher_distillation": teacher_distillation}
            if teacher_distillation is not None
            else {}
        ),
    }


def validate_factor_monitor_effective_config(
    effective: dict[str, Any], *, require_repository_provenance: bool = False
) -> dict[str, Any]:
    """Validate and enrich a generic runtime config without selecting defaults."""

    if effective.get("kind") not in {"train_run", "eval_run"}:
        raise ConfigurationError("FactorMonitor backend supports train_run or eval_run")
    monitor = require_mapping(effective.get("monitor"), "monitor")
    if monitor.get("backend") != BACKEND_NAME:
        raise ConfigurationError(f"monitor.backend must be {BACKEND_NAME!r}")
    options = require_mapping(monitor.get("backend_options"), "monitor.backend_options")
    config_path = Path(str(effective["config_path"])).resolve()
    long_cot_windowing = None
    long_cot_mil = None
    if effective["kind"] == "eval_run":
        raw_evaluation_scope = require_mapping(
            options.get("evaluation_scope"), "evaluation_scope"
        )
        if raw_evaluation_scope.get("long_cot_windowing") is not None:
            long_cot_windowing = _long_cot_windowing(
                raw_evaluation_scope["long_cot_windowing"],
                config_path=config_path,
            )
    elif options.get("long_cot_mil") is not None:
        long_cot_mil = _long_cot_mil(
            options["long_cot_mil"], config_path=config_path
        )
    common_fields = {
        "architecture_config",
        "factor_encoding_contract_id",
        "evidence_encoding_contract_id",
        "evidence_routes",
        "sequence",
        "batching",
        "repository_policy",
    }
    train_fields = {
        "lora",
        "new_module_learning_rate",
        "label_weighting",
        "loss_aggregation",
        "training_scope",
    }
    eval_fields = {"model_bundle", "evaluation_scope"}
    required_options = common_fields | (
        train_fields if effective["kind"] == "train_run" else eval_fields
    )
    optional_options = {
        "model_variant",
        "endpoint_encoding_contract_id",
        "endpoint_length_audit",
        "endpoint_max_padded_tokens_per_backbone_forward",
        "endpoint_decision_mode",
        "joint_encoding_contract_id",
        "joint_length_audit",
        "joint_max_padded_tokens_per_backbone_forward",
        "endpoint_role_embedding",
        "endpoint_factor_guided_readout",
        "endpoint_signed_criterion_panel",
        "endpoint_layer_mixed",
        "long_cot_mil",
        "metric_mil_loss_weighting",
        "choice_contract_id",
        "chat_template",
        "chat_template_kwargs",
        "field_boundary_policy",
    }
    if effective["kind"] == "train_run":
        optional_options.add("metric_driven_weighting")
    require_keys(options, required_options, "monitor.backend_options")
    reject_unknown(options, required_options | optional_options, "monitor.backend_options")
    field_policy = options.get(
        "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
    )
    if field_policy not in FIELD_BOUNDARY_POLICIES:
        raise ConfigurationError(
            "monitor.backend_options.field_boundary_policy must be one of "
            f"{FIELD_BOUNDARY_POLICIES}"
        )
    # Make the policy visible in every effective config, including legacy
    # strict runs that omit the optional field in their source YAML.
    options = {**options, "field_boundary_policy": field_policy}
    mil_loss_weighting = options.get("metric_mil_loss_weighting")
    if mil_loss_weighting is not None:
        if mil_loss_weighting not in {"factor_instance"}:
            raise ConfigurationError(
                "monitor.backend_options.metric_mil_loss_weighting must be "
                "'factor_instance' when provided"
            )
        if options.get("metric_driven_weighting") is None:
            raise ConfigurationError(
                "metric_mil_loss_weighting requires metric_driven_weighting"
            )
    model_variant = options.get("model_variant", "factor_late_fusion")
    if model_variant not in {
        "factor_late_fusion",
        "factor_joint_fusion",
        "endpoint_early_fusion",
        "causal_lm_choice",
    }:
        raise ConfigurationError(
            "monitor.backend_options.model_variant must be factor_late_fusion, "
            "factor_joint_fusion, endpoint_early_fusion, or causal_lm_choice"
        )
    joint_serialized = model_variant in {
        "factor_joint_fusion",
        "endpoint_early_fusion",
        "causal_lm_choice",
    }
    option_prefix = (
        "endpoint"
        if model_variant in {"endpoint_early_fusion", "causal_lm_choice"}
        else "joint"
    )
    serialization_contract = options.get(f"{option_prefix}_encoding_contract_id")
    serialization_audit_value = options.get(f"{option_prefix}_length_audit")
    serialization_forward_budget = options.get(
        f"{option_prefix}_max_padded_tokens_per_backbone_forward"
    )
    serialization_audit_path: Path | None = None
    serialization_audited_maximum: int | None = None
    endpoint_decision_mode_value = options.get("endpoint_decision_mode")
    causal_choice = model_variant == "causal_lm_choice"
    if causal_choice:
        if effective["kind"] != "train_run":
            raise ConfigurationError(
                "causal_lm_choice model bundles are evaluated through the vLLM choice suite"
            )
        if options.get("choice_contract_id") != FACTOR_SUFFIX_CHOICE_CONTRACT_ID:
            raise ConfigurationError(
                f"causal_lm_choice requires choice_contract_id="
                f"{FACTOR_SUFFIX_CHOICE_CONTRACT_ID!r}"
            )
        if options.get("chat_template") != "tokenizer_native":
            raise ConfigurationError(
                "causal_lm_choice requires chat_template=tokenizer_native"
            )
        chat_template_kwargs = require_mapping(
            options.get("chat_template_kwargs"), "chat_template_kwargs"
        )
        if dict(chat_template_kwargs) != {"enable_thinking": False}:
            raise ConfigurationError(
                "causal_lm_choice freezes chat_template_kwargs to enable_thinking: false"
            )
    else:
        unexpected_choice_fields = [
            key
            for key in ("choice_contract_id", "chat_template", "chat_template_kwargs")
            if options.get(key) is not None
        ]
        if unexpected_choice_fields:
            raise ConfigurationError(
                f"choice prompt fields require causal_lm_choice: {unexpected_choice_fields}"
            )
        chat_template_kwargs = None
    guided_readout_present = "endpoint_factor_guided_readout" in options
    endpoint_factor_guided_readout = (
        _endpoint_factor_guided_readout(options["endpoint_factor_guided_readout"])
        if guided_readout_present
        else {"enabled": False}
    )
    signed_panel_present = "endpoint_signed_criterion_panel" in options
    endpoint_signed_criterion_panel = (
        _endpoint_signed_criterion_panel(options["endpoint_signed_criterion_panel"])
        if signed_panel_present
        else {"enabled": False}
    )
    layer_mixed_present = "endpoint_layer_mixed" in options
    endpoint_layer_mixed = (
        _endpoint_layer_mixed(options["endpoint_layer_mixed"])
        if layer_mixed_present
        else {"enabled": False}
    )
    role_embedding_value = options.get("endpoint_role_embedding")
    if role_embedding_value is None:
        endpoint_role_embedding = {
            "enabled": False,
            "field_order": list(ENDPOINT_ROLE_FIELDS),
            "initialization": "zeros",
            "application": "field_content_tokens_only",
            "structural_tokens": "zero",
        }
    else:
        role_embedding = require_mapping(
            role_embedding_value, "monitor.backend_options.endpoint_role_embedding"
        )
        role_fields = {
            "enabled",
            "field_order",
            "initialization",
            "application",
            "structural_tokens",
        }
        require_keys(role_embedding, role_fields, "endpoint_role_embedding")
        reject_unknown(role_embedding, role_fields, "endpoint_role_embedding")
        if not isinstance(role_embedding["enabled"], bool):
            raise ConfigurationError("endpoint_role_embedding.enabled must be boolean")
        if role_embedding["field_order"] != list(ENDPOINT_ROLE_FIELDS):
            raise ConfigurationError(
                "endpoint_role_embedding.field_order must be the canonical six-field order"
            )
        if role_embedding["initialization"] != "zeros":
            raise ConfigurationError("endpoint role embeddings must use zero initialization")
        if role_embedding["application"] != "field_content_tokens_only":
            raise ConfigurationError(
                "endpoint role embeddings may apply only to field content tokens"
            )
        if role_embedding["structural_tokens"] != "zero":
            raise ConfigurationError("structural endpoint tokens must receive zero role input")
        endpoint_role_embedding = dict(role_embedding)
    if endpoint_role_embedding["enabled"] and model_variant != "endpoint_early_fusion":
        raise ConfigurationError("endpoint role embeddings require endpoint_early_fusion")
    if (
        endpoint_factor_guided_readout["enabled"]
        and model_variant != "endpoint_early_fusion"
    ):
        raise ConfigurationError("guided readout requires endpoint_early_fusion")
    if endpoint_signed_criterion_panel["enabled"] and model_variant != "endpoint_early_fusion":
        raise ConfigurationError("signed criterion panel requires endpoint_early_fusion")
    if endpoint_layer_mixed["enabled"] and model_variant != "endpoint_early_fusion":
        raise ConfigurationError("layer-mixed endpoint requires endpoint_early_fusion")
    if long_cot_windowing is not None and model_variant != "endpoint_early_fusion":
        raise ConfigurationError("long_cot_windowing requires endpoint_early_fusion")
    if long_cot_mil is not None and model_variant != "endpoint_early_fusion":
        raise ConfigurationError("long_cot_mil requires endpoint_early_fusion")
    if joint_serialized:
        allowed_contracts = (
            (FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,)
            if causal_choice
            else (
                ENDPOINT_CONTRACT_IDS
                if model_variant == "endpoint_early_fusion"
                else (ENDPOINT_CONTRACT_ID,)
            )
        )
        if serialization_contract not in allowed_contracts:
            raise ConfigurationError(
                f"{model_variant} requires {option_prefix}_encoding_contract_id in "
                f"{allowed_contracts}"
            )
        if model_variant == "endpoint_early_fusion":
            derived_decision_mode = endpoint_decision_mode(serialization_contract)
            if (
                endpoint_decision_mode_value is not None
                and endpoint_decision_mode_value != derived_decision_mode
            ):
                raise ConfigurationError(
                    "endpoint_decision_mode disagrees with endpoint encoding contract"
                )
            endpoint_decision_mode_value = derived_decision_mode
        elif endpoint_decision_mode_value is not None:
            raise ConfigurationError(
                "endpoint_decision_mode is valid only for endpoint_early_fusion"
            )
        if (
            endpoint_role_embedding["enabled"]
            and serialization_contract != ENDPOINT_CONTRACT_ID
        ):
            raise ConfigurationError(
                "Role Embedding requires the original endpoint contract and cannot be "
                "combined with endpoint serialization ablations"
            )
        if endpoint_factor_guided_readout["enabled"]:
            if serialization_contract != ENDPOINT_CONTRACT_ID:
                raise ConfigurationError(
                    "guided readout requires the original endpoint contract"
                )
            if endpoint_decision_mode_value != SINGLE_ENDPOINT_DECISION_MODE:
                raise ConfigurationError(
                    "guided readout requires single-endpoint decision mode"
                )
            if endpoint_role_embedding["enabled"]:
                raise ConfigurationError(
                    "guided readout and Role Embedding are separate experiments"
                )
        if (
            endpoint_decision_mode_value == CRITERION_DIFFERENCE_DECISION_MODE
            and endpoint_role_embedding["enabled"]
        ):
            raise ConfigurationError(
                "Criterion-Difference and Role Embedding are separate experiments"
            )
        experimental_deltas = [
            name
            for name, enabled in (
                ("endpoint serialization", serialization_contract != ENDPOINT_CONTRACT_ID),
                ("Role Embedding", endpoint_role_embedding["enabled"]),
                ("Guided Readout", endpoint_factor_guided_readout["enabled"]),
                ("Signed Criterion Panel", endpoint_signed_criterion_panel["enabled"]),
                ("Layer-Mixed Endpoint", endpoint_layer_mixed["enabled"]),
            )
            if enabled
        ]
        if len(experimental_deltas) > 1:
            raise ConfigurationError(
                "endpoint architecture experiments are mutually exclusive: "
                + ", ".join(experimental_deltas)
            )
        if endpoint_signed_criterion_panel["enabled"]:
            if serialization_contract != ENDPOINT_CONTRACT_ID:
                raise ConfigurationError("signed criterion panel requires the original endpoint contract")
            if endpoint_decision_mode_value != SINGLE_ENDPOINT_DECISION_MODE:
                raise ConfigurationError("signed criterion panel requires single-endpoint mode")
        if endpoint_layer_mixed["enabled"]:
            if serialization_contract != ENDPOINT_CONTRACT_ID:
                raise ConfigurationError("layer-mixed endpoint requires the original endpoint contract")
            if endpoint_decision_mode_value != SINGLE_ENDPOINT_DECISION_MODE:
                raise ConfigurationError("layer-mixed endpoint requires single-endpoint mode")
        if serialization_audit_value is None:
            raise ConfigurationError(
                f"{model_variant} requires {option_prefix}_length_audit"
            )
        serialization_forward_budget = _integer(
            serialization_forward_budget,
            f"{option_prefix}_max_padded_tokens_per_backbone_forward",
        )
        if long_cot_windowing is not None:
            if model_variant != "endpoint_early_fusion":
                raise ConfigurationError(
                    "long_cot_windowing requires endpoint_early_fusion"
                )
            if serialization_contract != FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID:
                raise ConfigurationError(
                    "long_cot_windowing requires the Factor-Suffix endpoint contract"
                )
            if (
                long_cot_windowing["max_endpoint_tokens"]
                != int(effective["runtime"]["max_sequence_length"])
            ):
                raise ConfigurationError(
                    "long_cot_windowing max_endpoint_tokens must equal "
                    "runtime.max_sequence_length"
                )
            if serialization_forward_budget != long_cot_windowing["max_endpoint_tokens"]:
                raise ConfigurationError(
                    "long_cot_windowing max_endpoint_tokens must equal the endpoint "
                    "backbone forward budget"
                )
        if long_cot_mil is not None:
            if serialization_contract != FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID:
                raise ConfigurationError("long_cot_mil requires the Factor-Suffix contract")
            if long_cot_mil["max_endpoint_tokens"] != int(
                effective["runtime"]["max_sequence_length"]
            ):
                raise ConfigurationError(
                    "long_cot_mil max endpoint must equal runtime.max_sequence_length"
                )
            if serialization_forward_budget != long_cot_mil["max_endpoint_tokens"]:
                raise ConfigurationError(
                    "long_cot_mil max endpoint must equal the backbone forward budget"
                )
        if serialization_forward_budget < int(effective["runtime"]["max_sequence_length"]):
            raise ConfigurationError(
                "joint backbone forward token budget must fit one maximum-length "
                "untruncated query"
            )
        serialization_audit_path = resolve_path(
            serialization_audit_value, relative_to=config_path.parent
        )
        if not serialization_audit_path.is_file():
            raise ConfigurationError(
                f"{option_prefix}_length_audit does not exist: "
                f"{serialization_audit_path}"
            )
        serialization_audit = load_config(serialization_audit_path)
        expected_audit_encoding = (
            FACTOR_SUFFIX_CHOICE_CONTRACT_ID
            if causal_choice
            else serialization_contract
        )
        if serialization_audit.get("encoding") != expected_audit_encoding:
            raise ConfigurationError("joint length audit encoding contract differs")
        audit_tokenizer = serialization_audit.get("tokenizer", {})
        if not isinstance(audit_tokenizer, Mapping):
            raise ConfigurationError("joint length audit lacks tokenizer identity")
        audit_field_policy = audit_tokenizer.get(
            "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
        )
        if audit_field_policy != field_policy:
            raise ConfigurationError(
                "joint length audit field boundary policy differs from the run config"
            )
        if causal_choice and (
            serialization_audit.get("endpoint_encoding")
            != FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID
            or serialization_audit.get("chat_template") != "tokenizer_native"
            or serialization_audit.get("chat_template_kwargs")
            != {"enable_thinking": False}
            or serialization_audit.get("answer_tokens") != 1
            or serialization_audit.get("length_semantics")
            != "prompt_plus_answer_token"
        ):
            raise ConfigurationError(
                "causal choice length audit prompt or answer-token contract differs"
            )
        serialization_audited_maximum = (
            serialization_audit.get("totals", {}).get("lengths", {}).get("max")
        )
        if isinstance(serialization_audited_maximum, bool) or not isinstance(
            serialization_audited_maximum, int
        ):
            raise ConfigurationError("joint length audit lacks integer maximum")
        if (
            serialization_audited_maximum
            > int(effective["runtime"]["max_sequence_length"])
            and long_cot_windowing is None
            and long_cot_mil is None
        ):
            raise ConfigurationError(
                "max_sequence_length is below the audited joint maximum"
            )
        candidates = serialization_audit.get("candidate_limits", [])
        audit_limit = (
            long_cot_windowing["trigger_endpoint_tokens"]
            if long_cot_windowing is not None
            else (
                long_cot_mil["trigger_endpoint_tokens"]
                if long_cot_mil is not None
                else int(effective["runtime"]["max_sequence_length"])
            )
        )
        matching = [
            row
            for row in candidates
            if row.get("max_sequence_length")
            == audit_limit
        ]
        if len(matching) != 1:
            raise ConfigurationError(
                "joint length audit lacks the configured evaluation limit"
            )
        if (
            long_cot_windowing is None
            and long_cot_mil is None
            and matching[0].get("dropped_queries") != 0
        ):
            raise ConfigurationError(
                "joint length audit does not prove zero drops at max_sequence_length"
            )
        if long_cot_windowing is not None and not (
            isinstance(matching[0].get("dropped_queries"), int)
            and not isinstance(matching[0].get("dropped_queries"), bool)
            and matching[0]["dropped_queries"] > 0
        ):
            raise ConfigurationError(
                "long_cot_windowing requires an audit proving at least one query "
                "above trigger_endpoint_tokens"
            )
        if long_cot_mil is not None and not (
            isinstance(matching[0].get("dropped_queries"), int)
            and not isinstance(matching[0].get("dropped_queries"), bool)
            and matching[0]["dropped_queries"] > 0
        ):
            raise ConfigurationError(
                "long_cot_mil requires an audit proving overlength training queries"
            )
        audited_files = serialization_audit.get("artifact", {}).get("files", {})
        current_files = effective.get("dataset", {}).get("files", {})
        for file_value in current_files.values():
            basename = Path(str(file_value.get("path", ""))).name
            audited = audited_files.get(basename)
            if not isinstance(audited, Mapping) or audited.get("sha256") != file_value.get(
                "sha256"
            ):
                raise ConfigurationError(
                    f"joint length audit is not bound to current dataset file {basename}"
                )
    for prefix in ("endpoint", "joint"):
        if prefix == option_prefix and joint_serialized:
            continue
        for suffix in (
            "encoding_contract_id",
            "length_audit",
            "max_padded_tokens_per_backbone_forward",
        ):
            if options.get(f"{prefix}_{suffix}") is not None:
                raise ConfigurationError(
                    f"{model_variant} must not set {prefix}_{suffix}"
                )
    if model_variant != "endpoint_early_fusion" and endpoint_decision_mode_value is not None:
        raise ConfigurationError(
            "endpoint_decision_mode is valid only for endpoint_early_fusion"
        )
    if options["factor_encoding_contract_id"] != FACTOR_CONTRACT_ID:
        raise ConfigurationError("wrong factor encoding contract")
    if options["evidence_encoding_contract_id"] != EVIDENCE_CONTRACT_ID:
        raise ConfigurationError("wrong evidence encoding contract")
    model = require_mapping(effective.get("model"), "model")
    _text(model.get("checkpoint_revision"), "model.checkpoint_revision")
    _text(model.get("tokenizer_revision"), "model.tokenizer_revision")
    load_options = require_mapping(model.get("load_options"), "model.load_options")
    load_fields = {
        "trust_remote_code",
        "local_files_only",
        "attn_implementation",
        "torch_dtype",
        "device_map",
        # Some fast tokenizers ship a padding token in the vocabulary but do
        # not declare it in tokenizer_config.json.  Keep the override
        # explicit so the backend never silently chooses EOS or adds a token.
        "pad_token",
    }
    required_load_fields = load_fields - {"pad_token"}
    require_keys(load_options, required_load_fields, "model.load_options")
    reject_unknown(load_options, load_fields, "model.load_options")
    if not isinstance(load_options["trust_remote_code"], bool):
        raise ConfigurationError("model.load_options.trust_remote_code must be boolean")
    if not isinstance(load_options["local_files_only"], bool):
        raise ConfigurationError("model.load_options.local_files_only must be boolean")
    _text(load_options["attn_implementation"], "model.load_options.attn_implementation")
    if load_options["torch_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ConfigurationError("model.load_options.torch_dtype must be bfloat16/float16/float32")
    if load_options["device_map"] is not None:
        raise ConfigurationError("training/evaluation backend requires device_map: null")
    if load_options.get("pad_token") is not None:
        _text(load_options["pad_token"], "model.load_options.pad_token")

    architecture_path = resolve_path(options["architecture_config"], relative_to=config_path.parent)
    architecture = load_architecture(architecture_path)
    if options["evidence_routes"] != architecture.family_routes:
        raise ConfigurationError("run evidence_routes must exactly match architecture mapping")

    training_scope = None
    evaluation_scope = None
    if effective["kind"] == "train_run":
        training_scope = _required_mapping(
            options, "training_scope", {"mode", "families"}
        )
        mode = training_scope["mode"]
        if mode not in {"phase1_full", "engineering_full_smoke", "smoke_subset"}:
            raise ConfigurationError(
                "training_scope.mode must be phase1_full, engineering_full_smoke, "
                "or smoke_subset"
            )
        families_value = training_scope["families"]
        if not isinstance(families_value, list) or not families_value:
            raise ConfigurationError("training_scope.families must be a non-empty list")
        families = tuple(_text(item, "training_scope family") for item in families_value)
        if len(families) != len(set(families)):
            raise ConfigurationError("training_scope.families contains duplicates")
        unknown = set(families) - set(architecture.phase1_families)
        if unknown:
            raise ConfigurationError(
                f"training_scope contains unknown families: {sorted(unknown)}"
            )
        if mode in {"phase1_full", "engineering_full_smoke"} and set(families) != set(
            architecture.phase1_families
        ):
            raise ConfigurationError(
                f"{mode} training scope must contain exactly all Phase-1 families"
            )
        if mode == "smoke_subset" and set(families) == set(architecture.phase1_families):
            raise ConfigurationError("smoke_subset must be a proper Phase-1 family subset")
        training_scope = {"mode": mode, "families": list(families)}
        metric_weighting = None
        if options.get("metric_driven_weighting") is not None:
            metric_weighting = _metric_driven_weighting(
                options["metric_driven_weighting"],
                training_families=families,
            )
    else:
        evaluation_scope = require_mapping(
            options.get("evaluation_scope"), "evaluation_scope"
        )
        reject_unknown(
            evaluation_scope,
            {
                "mode",
                "trajectory_selection",
                "max_trajectories",
                "factor_interventions",
                "factor_path_interventions",
                "mixed_labels_only",
                "intervention_seed",
                "native_checkpoint_threshold",
                "long_cot_windowing",
            },
            "evaluation_scope",
        )
        require_keys(evaluation_scope, {"mode"}, "evaluation_scope")
        evaluation_mode = evaluation_scope["mode"]
        if evaluation_mode not in {"formal", "smoke_reload", "diagnostic"}:
            raise ConfigurationError(
                "evaluation_scope.mode must be formal, smoke_reload, or diagnostic"
            )
        selection = evaluation_scope.get("trajectory_selection")
        maximum = evaluation_scope.get("max_trajectories")
        if evaluation_mode == "formal" and (selection is not None or maximum is not None):
            raise ConfigurationError("formal evaluation cannot select a trajectory subset")
    if long_cot_windowing is not None and evaluation_mode not in {"formal", "diagnostic"}:
        raise ConfigurationError(
            "long_cot_windowing is available only for formal or diagnostic evaluation"
            )
        if (selection is None) != (maximum is None):
            raise ConfigurationError(
                "evaluation_scope trajectory_selection and max_trajectories must be set together"
            )
        if selection is not None:
            if selection != "longest_evidence":
                raise ConfigurationError(
                    "evaluation_scope.trajectory_selection must be longest_evidence"
                )
            if _integer(maximum, "evaluation_scope.max_trajectories") != 1:
                raise ConfigurationError(
                    "long-context evaluation gate requires max_trajectories: 1"
                )
        interventions = evaluation_scope.get("factor_interventions")
        path_interventions = evaluation_scope.get("factor_path_interventions")
        mixed_only = evaluation_scope.get("mixed_labels_only")
        intervention_seed = evaluation_scope.get("intervention_seed")
        native_checkpoint_threshold = evaluation_scope.get(
            "native_checkpoint_threshold"
        )
        if evaluation_mode == "diagnostic":
            if not isinstance(interventions, list) or not interventions:
                raise ConfigurationError(
                    "diagnostic evaluation requires non-empty factor_interventions"
                )
            parsed_interventions = [
                _text(value, "evaluation_scope.factor_interventions")
                for value in interventions
            ]
            if len(parsed_interventions) != len(set(parsed_interventions)) or set(
                parsed_interventions
            ) - {"matched", "shuffled", "constant", "no_factor"}:
                raise ConfigurationError(
                    "factor_interventions must be unique matched/shuffled/constant/no_factor values"
                )
            parsed_path_interventions = None
            if path_interventions is not None:
                if model_variant != "factor_late_fusion":
                    raise ConfigurationError(
                        "factor_path_interventions require factor_late_fusion"
                    )
                if tuple(parsed_interventions) != ("matched", "shuffled"):
                    raise ConfigurationError(
                        "factor_path_interventions require factor_interventions: "
                        "[matched, shuffled]"
                    )
                if not isinstance(path_interventions, list) or not path_interventions:
                    raise ConfigurationError(
                        "factor_path_interventions must be a non-empty list"
                    )
                parsed_path_interventions = [
                    _text(value, "evaluation_scope.factor_path_interventions")
                    for value in path_interventions
                ]
                if (
                    len(parsed_path_interventions)
                    != len(set(parsed_path_interventions))
                    or set(parsed_path_interventions) - set(FACTOR_PATH_INTERVENTIONS)
                ):
                    raise ConfigurationError(
                        "factor_path_interventions must be unique supported values"
                    )
                if parsed_path_interventions[0] != "matched":
                    raise ConfigurationError(
                        "factor_path_interventions must begin with matched"
                    )
            if not isinstance(mixed_only, bool):
                raise ConfigurationError(
                    "diagnostic evaluation requires boolean mixed_labels_only"
                )
            intervention_seed = _integer(
                intervention_seed, "evaluation_scope.intervention_seed", positive=False
            )
            native_checkpoint_threshold = _number(
                native_checkpoint_threshold,
                "evaluation_scope.native_checkpoint_threshold",
                nonnegative=True,
            )
            if native_checkpoint_threshold > 1:
                raise ConfigurationError(
                    "native_checkpoint_threshold must be in [0,1]"
                )
        elif any(
            value is not None
            for value in (
                interventions,
                path_interventions,
                mixed_only,
                intervention_seed,
                native_checkpoint_threshold,
            )
        ):
            raise ConfigurationError(
                "Factor intervention fields are valid only for diagnostic evaluation"
            )
        evaluation_scope = {
            "mode": evaluation_mode,
            **(
                {
                    "trajectory_selection": selection,
                    "max_trajectories": maximum,
                }
                if selection is not None
                else {}
            ),
            **(
                {
                    "factor_interventions": parsed_interventions,
                    **(
                        {"factor_path_interventions": parsed_path_interventions}
                        if parsed_path_interventions is not None
                        else {}
                    ),
                    "mixed_labels_only": mixed_only,
                    "intervention_seed": intervention_seed,
                    "native_checkpoint_threshold": native_checkpoint_threshold,
                }
                if evaluation_mode == "diagnostic"
                else {}
            ),
            **(
                {"long_cot_windowing": long_cot_windowing}
                if long_cot_windowing is not None
                else {}
            ),
        }

    sequence = _required_mapping(
        options,
        "sequence",
        {"max_factor_tokens", "length_buckets", "overlength_policy"},
    )
    _integer(sequence["max_factor_tokens"], "sequence.max_factor_tokens")
    buckets = sequence["length_buckets"]
    if not isinstance(buckets, list) or not buckets:
        raise ConfigurationError("sequence.length_buckets must be a non-empty list")
    parsed_buckets = [_integer(item, "sequence.length_buckets") for item in buckets]
    if parsed_buckets != sorted(set(parsed_buckets)):
        raise ConfigurationError("sequence.length_buckets must be strictly increasing")
    if parsed_buckets[-1] != effective["runtime"]["max_sequence_length"]:
        raise ConfigurationError("final length bucket must equal runtime.max_sequence_length")
    expected_overlength_policy = (
        "window_complete_no_truncation"
        if long_cot_mil is not None
        else "exclude_to_phase2"
    )
    if sequence["overlength_policy"] != expected_overlength_policy:
        raise ConfigurationError(
            f"overlength_policy must be {expected_overlength_policy}"
        )
    if effective["runtime"]["truncation_policy"] != "error":
        raise ConfigurationError("FactorMonitor never truncates Context or CoT")

    batching_value = require_mapping(options.get("batching"), "monitor.backend_options.batching")
    batching_fields = {
        "training_unit",
        "trajectories_per_microbatch",
        "max_queries_per_trajectory",
        "mixed_route_policy",
        "distributed_unused_route_policy",
    }
    required_batching_fields = batching_fields if effective["kind"] == "train_run" else (
        batching_fields - {"training_unit"}
    )
    require_keys(
        batching_value, required_batching_fields, "monitor.backend_options.batching"
    )
    reject_unknown(batching_value, batching_fields, "monitor.backend_options.batching")
    batching = {
        **batching_value,
        # Evaluation never samples training units. Preserve old evaluation
        # configs while still recording their non-operative canonical value.
        "training_unit": batching_value.get("training_unit", "trajectory"),
    }
    if batching["training_unit"] not in {
        "trajectory",
        "factor_instance_contrast",
        "long_cot_mil",
    }:
        raise ConfigurationError(
            "batching.training_unit must be trajectory, factor_instance_contrast, "
            "or long_cot_mil"
        )
    if (batching["training_unit"] == "long_cot_mil") != (long_cot_mil is not None):
        raise ConfigurationError(
            "batching.training_unit=long_cot_mil and long_cot_mil config must appear together"
        )
    _integer(batching["trajectories_per_microbatch"], "batching.trajectories_per_microbatch")
    maximum_queries = batching["max_queries_per_trajectory"]
    if maximum_queries is not None:
        _integer(maximum_queries, "batching.max_queries_per_trajectory")
    if batching["mixed_route_policy"] != "route_masked_vectorized":
        raise ConfigurationError("mixed_route_policy must be route_masked_vectorized")
    if batching["distributed_unused_route_policy"] not in {
        "find_unused_parameters_true",
        "sampler_guaranteed_per_synchronized_step",
    }:
        raise ConfigurationError("invalid distributed unused-route policy")

    policy = _text(options["repository_policy"], "repository_policy")
    if policy not in {"require_clean", "allow_dirty_with_recorded_diff"}:
        raise ConfigurationError("repository_policy must explicitly choose clean or recorded dirty")
    repository = require_mapping(effective.get("repository"), "repository")
    if policy == "require_clean" and repository.get("git_dirty") is not False:
        raise ConfigurationError("repository_policy=require_clean but repository is dirty")
    provenance_ready = (
        isinstance(repository.get("git_commit"), str)
        and bool(repository.get("git_commit"))
        and repository.get("git_dirty") in {True, False}
    )
    if policy == "allow_dirty_with_recorded_diff" and repository.get("git_dirty") is True:
        provenance_ready = (
            provenance_ready
            and isinstance(repository.get("git_diff_sha256"), str)
            and repository.get("git_untracked_entries") == 0
        )
    if repository.get("git_state") == "synthetic_snapshot":
        provenance_ready = (
            provenance_ready
            and repository.get("git_dirty") is False
            and isinstance(repository.get("source_bundle_path"), str)
            and bool(repository.get("source_bundle_path"))
            and isinstance(repository.get("source_bundle_sha256"), str)
            and bool(repository.get("source_bundle_sha256"))
        )
    formal_run = (
        training_scope is None or training_scope["mode"] == "phase1_full"
        if effective["kind"] == "train_run"
        else evaluation_scope is None or evaluation_scope["mode"] == "formal"
    )
    if require_repository_provenance and formal_run and not provenance_ready:
        raise ConfigurationError(
            "formal FactorMonitor runs require a committed repository and reproducible dirty "
            "provenance; commit/stage untracked source artifacts before launching"
        )
    if require_repository_provenance:
        runs_root = (Path(__file__).resolve().parents[3] / "runs").resolve()
        output_directory = Path(str(effective.get("output_directory"))).resolve()
        try:
            output_directory.relative_to(runs_root)
        except ValueError as exc:
            raise ConfigurationError(
                f"formal FactorMonitor output_directory must be inside {runs_root}"
            ) from exc

    if effective["kind"] == "train_run":
        if causal_choice and effective.get("initialization") is not None:
            raise ConfigurationError(
                "causal_lm_choice screen must initialize from the raw base model"
            )
        lora = _required_mapping(options, "lora", {"target_modules", "rank", "alpha", "dropout", "bias"})
        targets = lora["target_modules"]
        if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item for item in targets):
            raise ConfigurationError("lora.target_modules must be a non-empty string list")
        _integer(lora["rank"], "lora.rank")
        _integer(lora["alpha"], "lora.alpha")
        dropout = _number(lora["dropout"], "lora.dropout", nonnegative=True)
        if dropout >= 1:
            raise ConfigurationError("lora.dropout must be less than one")
        if lora["bias"] not in {"none", "all", "lora_only"}:
            raise ConfigurationError("invalid lora.bias")
        if causal_choice:
            if options["new_module_learning_rate"] is not None:
                raise ConfigurationError(
                    "causal_lm_choice has no custom module and requires "
                    "new_module_learning_rate: null"
                )
        else:
            _number(
                options["new_module_learning_rate"],
                "new_module_learning_rate",
                nonnegative=True,
            )
            if options["new_module_learning_rate"] <= 0:
                raise ConfigurationError("new_module_learning_rate must be positive")
        if options["label_weighting"] not in {"none", "record_loss_weight"}:
            raise ConfigurationError(
                "label_weighting must be none or record_loss_weight"
            )
        if metric_weighting is not None and options["label_weighting"] != "record_loss_weight":
            raise ConfigurationError(
                "metric-driven weighting requires label_weighting=record_loss_weight"
            )
        if options["loss_aggregation"] not in {
            "trajectory_hierarchical",
            "factor_instance_uniform",
            "mil_factor_instance_uniform",
        }:
            raise ConfigurationError("invalid loss_aggregation")
        expected_loss = {
            "trajectory": "trajectory_hierarchical",
            "factor_instance_contrast": "factor_instance_uniform",
            "long_cot_mil": "mil_factor_instance_uniform",
        }[batching["training_unit"]]
        if options["loss_aggregation"] != expected_loss:
            raise ConfigurationError(
                "loss_aggregation disagrees with batching.training_unit; expected "
                f"{expected_loss!r}"
            )
        options = {
            **options,
            "batching": batching,
            "model_variant": model_variant,
            "endpoint_role_embedding": endpoint_role_embedding,
            **(
                {
                    "endpoint_factor_guided_readout": endpoint_factor_guided_readout
                }
                if guided_readout_present
                else {}
            ),
            **(
                {"endpoint_signed_criterion_panel": endpoint_signed_criterion_panel}
                if signed_panel_present
                else {}
            ),
            **(
                {"endpoint_layer_mixed": endpoint_layer_mixed}
                if layer_mixed_present
                else {}
            ),
            **(
                {"endpoint_decision_mode": endpoint_decision_mode_value}
                if model_variant == "endpoint_early_fusion"
                else {}
            ),
            **(
                {
                    f"{option_prefix}_length_audit": {
                        "path": str(serialization_audit_path),
                        "sha256": file_sha256(serialization_audit_path),
                        "audited_max_tokens": serialization_audited_maximum,
                    }
                }
                if joint_serialized and serialization_audit_path is not None
                else {}
            ),
            "training_scope": training_scope,
            **(
                {"metric_driven_weighting": metric_weighting}
                if metric_weighting is not None
                else {}
            ),
            **({"long_cot_mil": long_cot_mil} if long_cot_mil is not None else {}),
        }
        optimization = require_mapping(effective.get("optimization"), "optimization")
        evaluation = require_mapping(effective.get("evaluation"), "evaluation")
        validation_only_value = evaluation.get("validation_only")
        validation_only: Mapping[str, Any] | None = None
        if validation_only_value is not None:
            validation_only = require_mapping(
                validation_only_value, "evaluation.validation_only"
            )
            validation_only_fields = {
                "checkpoint",
                "global_step",
                "source_config_hash",
                "trainer_state_sha256",
                "factor_monitor_sha256",
                "adapter_model_sha256",
                "source_policy",
                "training_steps",
            }
            require_keys(
                validation_only,
                validation_only_fields,
                "evaluation.validation_only",
            )
            reject_unknown(
                validation_only,
                validation_only_fields,
                "evaluation.validation_only",
            )
            checkpoint = Path(str(validation_only["checkpoint"]))
            if not checkpoint.is_absolute() or not checkpoint.is_dir():
                raise ConfigurationError(
                    "evaluation.validation_only.checkpoint must be an existing absolute directory"
                )
            global_step = validation_only["global_step"]
            if (
                isinstance(global_step, bool)
                or not isinstance(global_step, int)
                or global_step <= 0
                or checkpoint.name != f"checkpoint-{global_step}"
            ):
                raise ConfigurationError(
                    "evaluation.validation_only global_step must match the checkpoint directory"
                )
            if validation_only["source_policy"] != "immutable":
                raise ConfigurationError(
                    "evaluation.validation_only source checkpoint must be immutable"
                )
            if validation_only["training_steps"] != 0:
                raise ConfigurationError(
                    "evaluation.validation_only must execute zero training steps"
                )
            if optimization.get("resume_from_checkpoint") is not None:
                raise ConfigurationError(
                    "validation-only checkpoint cannot be combined with training resume"
                )
            if effective.get("runtime", {}).get("expected_world_size") != 1:
                raise ConfigurationError("validation-only checkpoint requires one rank")
            if evaluation.get("during_training", True):
                raise ConfigurationError(
                    "validation-only checkpoint must disable during-training evaluation"
                )
        if optimization.get("strategy") != "lora":
            raise ConfigurationError("FactorMonitor train strategy must be lora")
        expected_objective = (
            "factor_choice_sft" if causal_choice else "factor_binary"
        )
        if optimization.get("objective") != expected_objective:
            raise ConfigurationError(
                f"{model_variant} train objective must be {expected_objective}"
            )
        if optimization.get("batch_size") != batching["trajectories_per_microbatch"]:
            raise ConfigurationError(
                "optimization.batch_size must equal trajectories_per_microbatch"
            )
        if long_cot_mil is not None:
            initialization = effective.get("initialization")
            if optimization.get("batch_size") != 1:
                raise ConfigurationError("long_cot_mil requires per-rank microbatch one")
            if optimization.get("gradient_accumulation_steps") != 2:
                raise ConfigurationError("long_cot_mil requires gradient accumulation two")
            world_size = effective.get("runtime", {}).get("expected_world_size")
            if world_size not in {1, 2}:
                raise ConfigurationError(
                    "long_cot_mil requires one or two distributed ranks"
                )
            if (
                world_size == 1
                and optimization.get("resume_from_checkpoint") is None
                and validation_only is None
            ):
                raise ConfigurationError(
                    "single-rank long_cot_mil is validation-recovery only"
                )
            if long_cot_mil["training_schedule"] == "full_epoch_joint":
                if initialization is not None:
                    raise ConfigurationError(
                        "full_epoch_joint must initialize fresh LoRA/head from the base model"
                    )
                if optimization.get("learning_rate") != 5e-5:
                    raise ConfigurationError(
                        "full_epoch_joint LoRA learning rate is frozen at 5e-5"
                    )
                if options.get("new_module_learning_rate") != 1e-4:
                    raise ConfigurationError(
                        "full_epoch_joint head learning rate is frozen at 1e-4"
                    )
                if validation_only is None:
                    gate = optimization.get("max_steps") == 10
                    screen = optimization.get("max_steps") == 1054
                    formal = (
                        optimization.get("max_steps") is None
                        and optimization.get("epochs") == 3
                    )
                    if not (gate or screen or formal):
                        raise ConfigurationError(
                            "full_epoch_joint requires a 10-step gate, a 1054-step "
                            "screen, or three epochs"
                        )
                    expected_cadence = (
                        10 if gate else long_cot_mil["optimizer_steps_per_epoch"]
                    )
                    if screen:
                        expected_cadence = 1054
                    if optimization.get("save_steps") != expected_cadence:
                        raise ConfigurationError(
                            "full_epoch_joint save cadence must match gate or epoch cadence"
                        )
                    if evaluation.get("cadence") != expected_cadence:
                        raise ConfigurationError(
                            "full_epoch_joint evaluation cadence must match save cadence"
                        )
            else:
                if not isinstance(initialization, Mapping):
                    raise ConfigurationError(
                        "continuation long_cot_mil requires weights-only initialization"
                    )
                if initialization.get("policy") != "weights_only_fresh_optimizer":
                    raise ConfigurationError("long_cot_mil initialization policy drift")
                if optimization.get("learning_rate") != 1e-5:
                    raise ConfigurationError(
                        "continuation MIL LoRA learning rate is frozen at 1e-5"
                    )
                if options.get("new_module_learning_rate") != 1e-5:
                    raise ConfigurationError(
                        "continuation MIL head learning rate is frozen at 1e-5"
                    )
                expected_steps = 144 * int(long_cot_mil["long_exposures"])
                if optimization.get("max_steps") not in {10, expected_steps}:
                    raise ConfigurationError(
                        f"long_cot_mil max_steps must be 10 gate steps or {expected_steps}"
                    )
        if not isinstance(evaluation.get("during_training", True), bool):
            raise ConfigurationError("evaluation.during_training must be boolean")
        if evaluation.get("metric") not in SUPPORTED_METRICS:
            raise ConfigurationError(
                f"unsupported checkpoint metric: {evaluation.get('metric')!r}"
            )
        if optimization.get("checkpoint_policy") != "steps":
            raise ConfigurationError("FactorMonitor checkpoint_policy must be steps")
        expected_threshold_policy = (
            "fixed_choice_0.5" if causal_choice else "select_on_validation"
        )
        if evaluation.get("threshold_policy") != expected_threshold_policy:
            raise ConfigurationError(
                f"{model_variant} training threshold_policy must be "
                f"{expected_threshold_policy}"
            )
        if causal_choice and evaluation.get("metric") != "gmean2":
            raise ConfigurationError(
                "causal_lm_choice checkpoint metric must be fixed-threshold gmean2"
            )
        if evaluation.get("early_stopping_patience") is not None:
            raise ConfigurationError(
                "early stopping callback is not implemented; set early_stopping_patience: null"
            )
        auto_mil_cadence = (
            long_cot_mil is not None
            and long_cot_mil["training_schedule"] == "full_epoch_joint"
            and optimization.get("max_steps") is None
            and optimization.get("epochs") == 3
        )
        cadence = evaluation.get("cadence")
        if cadence == "auto" and auto_mil_cadence:
            cadence = "auto"
        elif isinstance(cadence, bool) or not isinstance(cadence, int) or cadence <= 0:
            raise ConfigurationError("evaluation.cadence must be a positive integer")
        save_steps = optimization.get("save_steps")
        if save_steps == "auto" and auto_mil_cadence:
            save_steps = "auto"
        elif isinstance(save_steps, bool) or not isinstance(save_steps, int) or save_steps <= 0:
            raise ConfigurationError("optimization.save_steps must be a positive integer")
        if save_steps != "auto" and cadence != "auto" and save_steps % cadence != 0:
            raise ConfigurationError(
                "optimization.save_steps must be a multiple of evaluation.cadence when "
                "load_best_model_at_end is enabled"
            )
    else:
        model_bundle = resolve_path(options["model_bundle"], relative_to=config_path.parent)
        if not (model_bundle / "factor_monitor.safetensors").is_file():
            raise ConfigurationError(f"model_bundle is incomplete: {model_bundle}")
        options = {
            **options,
            "batching": batching,
            "model_variant": model_variant,
            "endpoint_role_embedding": endpoint_role_embedding,
            **(
                {
                    "endpoint_factor_guided_readout": endpoint_factor_guided_readout
                }
                if guided_readout_present
                else {}
            ),
            **(
                {"endpoint_signed_criterion_panel": endpoint_signed_criterion_panel}
                if signed_panel_present
                else {}
            ),
            **(
                {"endpoint_layer_mixed": endpoint_layer_mixed}
                if layer_mixed_present
                else {}
            ),
            **(
                {"endpoint_decision_mode": endpoint_decision_mode_value}
                if model_variant == "endpoint_early_fusion"
                else {}
            ),
            **(
                {
                    f"{option_prefix}_length_audit": {
                        "path": str(serialization_audit_path),
                        "sha256": file_sha256(serialization_audit_path),
                        "audited_max_tokens": serialization_audited_maximum,
                    }
                }
                if joint_serialized and serialization_audit_path is not None
                else {}
            ),
            "evaluation_scope": evaluation_scope,
            "model_bundle": str(model_bundle),
        }
        evaluation = require_mapping(effective.get("evaluation"), "evaluation")
        if evaluation.get("metric") not in (
            SUPPORTED_METRICS - {"loss"}
        ) | {PREDICTION_ONLY_METRIC}:
            raise ConfigurationError(
                f"unsupported formal evaluation metric: {evaluation.get('metric')!r}"
            )
        if evaluation.get("threshold_policy") != "frozen_validation":
            raise ConfigurationError(
                "formal evaluation threshold_policy must be frozen_validation"
            )
        reporting_dimensions = evaluation.get("reporting_dimensions", {})
        if not isinstance(reporting_dimensions, Mapping):
            raise ConfigurationError("evaluation.reporting_dimensions must be a mapping")
        normalized_dimensions: dict[str, Any] = {}
        for dimension, specification_value in reporting_dimensions.items():
            if dimension not in {"factor_scope", "negative_type"}:
                raise ConfigurationError(
                    f"unsupported evaluation reporting dimension: {dimension!r}"
                )
            specification = require_mapping(
                specification_value,
                f"evaluation.reporting_dimensions.{dimension}",
            )
            require_keys(
                specification,
                {"required", "values"},
                f"evaluation.reporting_dimensions.{dimension}",
            )
            reject_unknown(
                specification,
                {"required", "values"},
                f"evaluation.reporting_dimensions.{dimension}",
            )
            if not isinstance(specification["required"], bool):
                raise ConfigurationError(
                    f"evaluation.reporting_dimensions.{dimension}.required must be boolean"
                )
            values = specification["values"]
            if not isinstance(values, list) or not values:
                raise ConfigurationError(
                    f"evaluation.reporting_dimensions.{dimension}.values must be non-empty"
                )
            parsed_values = [
                _text(value, f"evaluation.reporting_dimensions.{dimension}.values")
                for value in values
            ]
            if len(parsed_values) != len(set(parsed_values)):
                raise ConfigurationError(
                    f"evaluation.reporting_dimensions.{dimension}.values contains duplicates"
                )
            normalized_dimensions[dimension] = {
                "required": specification["required"],
                "values": parsed_values,
            }
        threshold_path = Path(str(evaluation.get("threshold_source"))).resolve()
        try:
            threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot read frozen validation threshold: {threshold_path}"
            ) from exc
        threshold_value = threshold_payload.get("threshold")
        if (
            isinstance(threshold_value, bool)
            or not isinstance(threshold_value, (int, float))
            or not 0 <= float(threshold_value) <= 1
        ):
            raise ConfigurationError(
                "evaluation.threshold_source must contain numeric threshold in [0,1]"
            )
        evaluation = {
            **evaluation,
            "threshold_value": float(threshold_value),
            "reporting_dimensions": normalized_dimensions,
        }

    enriched = dict(effective)
    enriched["monitor"] = {
        **monitor,
        "backend_options": options,
        "architecture": architecture.to_dict(),
        "architecture_hash": architecture.source_hash,
        "route_mapping_hash": stable_hash(architecture.family_routes),
    }
    enriched["repository"] = {
        **repository,
        "policy": policy,
        "provenance_ready_for_formal_run": provenance_ready,
    }
    if effective["kind"] == "eval_run":
        enriched["evaluation"] = evaluation
    return enriched
