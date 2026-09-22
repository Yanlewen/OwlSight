"""Lossless raw renderers and token-span masks for FactorMonitor inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..config import stable_hash
from ..schema import FactorSpec


FACTOR_CONTRACT_ID = "factor_encoding_contract_v1"
EVIDENCE_CONTRACT_ID = "evidence_encoding_contract_v1"
ENDPOINT_CONTRACT_ID = "endpoint_early_fusion_contract_v1"
FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID = "endpoint_factor_suffix_contract_v1"
FACTOR_SUFFIX_CHOICE_CONTRACT_ID = "factor_suffix_ab_forced_choice_v1"
FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID = (
    "endpoint_factor_suffix_relay_contract_v1"
)
FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID = "endpoint_factor_bookends_contract_v1"
CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID = (
    "endpoint_factor_suffix_criterion_difference_contract_v1"
)
SINGLE_ENDPOINT_DECISION_MODE = "single_endpoint"
CRITERION_DIFFERENCE_DECISION_MODE = "criterion_difference"
ENDPOINT_ROLE_FIELDS = (
    "factor",
    "relation",
    "applied",
    "not_applied",
    "context",
    "cot",
)
FACTOR_SUFFIX_DECISION_QUERY = (
    "Using only the Task Context and Model CoT, determine whether the Factor "
    "is applied according to the stated criteria."
)
CRITERION_DIFFERENCE_DECISION_QUERY = (
    "Using only the Task Context and Model CoT, determine how strongly the "
    "evidence supports that the Factor satisfies the Candidate Criterion."
)
RENDERER_VERSION = "factor_monitor_raw_renderer_1"
SPAN_ALGORITHM_VERSION = "offset_mapping_contained_token_1"
RELAY_TOKEN_TRANSFORM_VERSION = "field_periodic_single_token_relay_1"
FACTOR_SUFFIX_RELAY_TOKEN = "<|file_sep|>"
FACTOR_SUFFIX_RELAY_INTERVAL = 4096
FACTOR_SUFFIX_RELAY_FIELDS = ("context", "cot")
FIELD_BOUNDARY_POLICIES = ("strict", "space_before_closing_tag")
DEFAULT_FIELD_BOUNDARY_POLICY = "strict"
FIELD_BOUNDARY_POLICY_ATTRIBUTE = "_factor_monitor_field_boundary_policy"


@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid character span: {self}")


@dataclass(frozen=True)
class RenderedInput:
    contract_id: str
    text: str
    spans: Mapping[str, CharacterSpan]


@dataclass(frozen=True)
class TokenizedInput:
    contract_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    span_masks: Mapping[str, tuple[bool, ...]]


class FieldBoundaryError(ValueError):
    """A tokenizer token crosses one or both edges of a rendered field."""

    def __init__(self, *, field: str, token_index: int, side: str) -> None:
        self.field = field
        self.token_index = token_index
        self.side = side
        super().__init__(
            f"token {token_index} crosses the {field} field {side} boundary; "
            "refusing tag leakage"
        )


def validate_field_boundary_policy(value: Any) -> str:
    """Validate the tokenizer-specific structural boundary policy."""

    if value not in FIELD_BOUNDARY_POLICIES:
        raise ValueError(
            "field boundary policy must be one of "
            f"{FIELD_BOUNDARY_POLICIES}, got {value!r}"
        )
    return str(value)


def set_field_boundary_policy(tokenizer: Any, policy: str) -> Any:
    """Attach an explicit boundary policy to a tokenizer instance.

    Rendered inputs remain lossless and unchanged.  The policy is applied only
    at the tokenizer boundary, which keeps the raw renderer contract stable
    while allowing Llama-family tokenizers to separate field content from a
    closing tag.
    """

    validated = validate_field_boundary_policy(policy)
    setattr(tokenizer, FIELD_BOUNDARY_POLICY_ATTRIBUTE, validated)
    return tokenizer


def field_boundary_policy(tokenizer: Any) -> str:
    """Return a tokenizer's explicit policy, defaulting legacy callers to strict."""

    return validate_field_boundary_policy(
        getattr(tokenizer, FIELD_BOUNDARY_POLICY_ATTRIBUTE, DEFAULT_FIELD_BOUNDARY_POLICY)
    )


def boundary_safe_rendered(rendered: RenderedInput, policy: str) -> RenderedInput:
    """Insert structural spaces before every closing field tag when requested.

    The raw renderer deliberately preserves user-provided field text.  This
    derived representation is used only for tokenization and shifts all
    character spans so masks still refer exclusively to the original field
    content.  A space is inserted at each field end, yielding e.g.
    ``value + " \\n</TASK_CONTEXT>"`` for the Llama-safe policy.
    """

    validated = validate_field_boundary_policy(policy)
    if validated == DEFAULT_FIELD_BOUNDARY_POLICY or not rendered.spans:
        return rendered

    insertion_points = sorted(span.end for span in rendered.spans.values())
    pieces: list[str] = []
    cursor = 0
    for point in insertion_points:
        pieces.append(rendered.text[cursor:point])
        pieces.append(" ")
        cursor = point
    pieces.append(rendered.text[cursor:])

    def shift_start(position: int) -> int:
        return position + sum(point < position for point in insertion_points)

    shifted_spans = {
        name: CharacterSpan(
            shift_start(span.start),
            shift_start(span.end),
        )
        for name, span in rendered.spans.items()
    }
    return RenderedInput(
        contract_id=rendered.contract_id,
        text="".join(pieces),
        spans=shifted_spans,
    )


def rendered_for_tokenizer(tokenizer: Any, rendered: RenderedInput) -> RenderedInput:
    """Apply the tokenizer's boundary policy exactly once at each call site."""

    return boundary_safe_rendered(rendered, field_boundary_policy(tokenizer))


def factor_suffix_relay_token_id(tokenizer: Any) -> int:
    """Resolve the frozen Relay marker without adding or resizing vocabulary."""

    convert_token = getattr(tokenizer, "convert_tokens_to_ids", None)
    convert_id = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_token) or not callable(convert_id):
        raise ValueError("relay contract requires tokenizer token/id conversion methods")
    relay_token_id = convert_token(FACTOR_SUFFIX_RELAY_TOKEN)
    if (
        isinstance(relay_token_id, bool)
        or not isinstance(relay_token_id, int)
        or relay_token_id < 0
        or convert_id(relay_token_id) != FACTOR_SUFFIX_RELAY_TOKEN
    ):
        raise ValueError(
            f"relay marker {FACTOR_SUFFIX_RELAY_TOKEN!r} must be exactly one "
            "registered tokenizer token"
        )
    return relay_token_id


def factor_suffix_relay_insertion_count(
    rendered: RenderedInput,
    offsets: list[tuple[int, int]],
    attention_mask: list[int] | None = None,
) -> int:
    """Count Relay insertions from exact offsets without materializing masks."""

    if rendered.contract_id != FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID:
        return 0
    for name, span in rendered.spans.items():
        if FACTOR_SUFFIX_RELAY_TOKEN in rendered.text[span.start : span.end]:
            raise ValueError(
                f"relay token literal occurs in visible field {name!r}; "
                "refusing ambiguous token identity"
            )
    if attention_mask is not None and len(attention_mask) != len(offsets):
        raise ValueError("relay offset and attention lengths differ")
    counts = {name: 0 for name in FACTOR_SUFFIX_RELAY_FIELDS}
    for index, (start, end) in enumerate(offsets):
        if attention_mask is not None and not attention_mask[index]:
            continue
        if start == end:
            continue
        for name in FACTOR_SUFFIX_RELAY_FIELDS:
            span = rendered.spans[name]
            overlaps = start < span.end and end > span.start
            contained = start >= span.start and end <= span.end
            if overlaps and not contained:
                crosses_start = start < span.start and end > span.start
                crosses_end = start < span.end and end > span.end
                side = (
                    "both"
                    if crosses_start and crosses_end
                    else "start"
                    if crosses_start
                    else "end"
                )
                raise FieldBoundaryError(field=name, token_index=index, side=side)
            if contained:
                counts[name] += 1
                break
    return sum(value // FACTOR_SUFFIX_RELAY_INTERVAL for value in counts.values())


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _render_fields(contract_id: str, fields: tuple[tuple[str, str, str], ...]) -> RenderedInput:
    chunks: list[str] = []
    spans: dict[str, CharacterSpan] = {}
    cursor = 0
    for index, (name, tag, raw_value) in enumerate(fields):
        value = _require_string(raw_value, name)
        prefix = f"<{tag}>\n"
        suffix = f"\n</{tag}>"
        chunks.append(prefix)
        cursor += len(prefix)
        start = cursor
        chunks.append(value)
        cursor += len(value)
        spans[name] = CharacterSpan(start, cursor)
        chunks.append(suffix)
        cursor += len(suffix)
        if index + 1 < len(fields):
            chunks.append("\n\n")
            cursor += 2
    return RenderedInput(contract_id=contract_id, text="".join(chunks), spans=spans)


def render_factor(factor_spec: FactorSpec | Mapping[str, Any]) -> RenderedInput:
    if isinstance(factor_spec, FactorSpec):
        values = {
            "factor": factor_spec.factor,
            "applied_criterion": factor_spec.applied_criterion,
            "not_applied_criterion": factor_spec.not_applied_criterion,
        }
    else:
        required = {"factor", "applied_criterion", "not_applied_criterion", "relation"}
        if set(factor_spec) != required:
            raise ValueError(
                f"factor_spec fields differ: missing={sorted(required-set(factor_spec))}, "
                f"unexpected={sorted(set(factor_spec)-required)}"
            )
        values = {
            key: _require_string(factor_spec[key], f"factor_spec.{key}")
            for key in ("factor", "applied_criterion", "not_applied_criterion")
        }
    return _render_fields(
        FACTOR_CONTRACT_ID,
        (
            ("factor", "FACTOR", values["factor"]),
            ("applied", "APPLIED_CRITERION", values["applied_criterion"]),
            ("not_applied", "NOT_APPLIED_CRITERION", values["not_applied_criterion"]),
        ),
    )


def render_evidence(*, task_context: str, model_cot: str) -> RenderedInput:
    return _render_fields(
        EVIDENCE_CONTRACT_ID,
        (
            ("context", "TASK_CONTEXT", task_context),
            ("cot", "MODEL_COT", model_cot),
        ),
    )


def render_endpoint_input(
    factor_spec: FactorSpec | Mapping[str, Any],
    *,
    task_context: str,
    model_cot: str,
    contract_id: str = ENDPOINT_CONTRACT_ID,
    omit_factor: bool = False,
) -> RenderedInput:
    """Render one registered information-equivalent early-fusion endpoint input."""

    if isinstance(factor_spec, FactorSpec):
        values = {
            "factor": factor_spec.factor,
            "relation": factor_spec.relation,
            "applied_criterion": factor_spec.applied_criterion,
            "not_applied_criterion": factor_spec.not_applied_criterion,
        }
    else:
        required = {"factor", "relation", "applied_criterion", "not_applied_criterion"}
        if set(factor_spec) != required:
            raise ValueError(
                f"factor_spec fields differ: missing={sorted(required-set(factor_spec))}, "
                f"unexpected={sorted(set(factor_spec)-required)}"
            )
        values = {
            key: _require_string(factor_spec[key], f"factor_spec.{key}")
            for key in required
        }
    if omit_factor:
        # This is an evaluation-only ablation.  Keep the structural tags and
        # relation field, but remove all Factor/criterion content.  Do not
        # manufacture a sentinel such as ``NO FACTOR``: that would itself be
        # a new learned token pattern.
        values = {
            **values,
            "factor": "",
            "applied_criterion": "",
            "not_applied_criterion": "",
        }
    common_fields = {
        "factor": ("factor", "FACTOR", values["factor"]),
        "relation": ("relation", "RELATION", values["relation"]),
        "applied": ("applied", "APPLIED_CRITERION", values["applied_criterion"]),
        "not_applied": (
            "not_applied",
            "NOT_APPLIED_CRITERION",
            values["not_applied_criterion"],
        ),
        "context": (
            "context",
            "TASK_CONTEXT",
            _require_string(task_context, "task_context"),
        ),
        "cot": ("cot", "MODEL_COT", _require_string(model_cot, "model_cot")),
    }
    if contract_id == ENDPOINT_CONTRACT_ID:
        field_order = ENDPOINT_ROLE_FIELDS
        marker = "\n\n<END_MONITOR_INPUT>"
    elif contract_id in {
        FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
        FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    }:
        field_order = ("context", "cot", "factor", "relation", "applied", "not_applied")
        marker = (
            "\n\n<DECISION_QUERY>\n"
            f"{FACTOR_SUFFIX_DECISION_QUERY}\n"
            "</DECISION_QUERY>\n\n<END_MONITOR_INPUT>"
        )
    elif contract_id == FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID:
        rendered = _render_fields(
            contract_id,
            (
                ("factor_prefix", "FACTOR", values["factor"]),
                ("relation_prefix", "RELATION", values["relation"]),
                ("applied_prefix", "APPLIED_CRITERION", values["applied_criterion"]),
                (
                    "not_applied_prefix",
                    "NOT_APPLIED_CRITERION",
                    values["not_applied_criterion"],
                ),
                common_fields["context"],
                common_fields["cot"],
                ("factor_suffix", "FACTOR", values["factor"]),
                ("relation_suffix", "RELATION", values["relation"]),
                ("applied_suffix", "APPLIED_CRITERION", values["applied_criterion"]),
                (
                    "not_applied_suffix",
                    "NOT_APPLIED_CRITERION",
                    values["not_applied_criterion"],
                ),
            ),
        )
        marker = (
            "\n\n<DECISION_QUERY>\n"
            f"{FACTOR_SUFFIX_DECISION_QUERY}\n"
            "</DECISION_QUERY>\n\n<END_MONITOR_INPUT>"
        )
        return RenderedInput(
            contract_id=rendered.contract_id,
            text=rendered.text + marker,
            spans=rendered.spans,
        )
    else:
        raise ValueError(f"unknown endpoint encoding contract: {contract_id}")
    rendered = _render_fields(
        contract_id,
        tuple(common_fields[field] for field in field_order),
    )
    return RenderedInput(
        contract_id=rendered.contract_id,
        text=rendered.text + marker,
        spans=rendered.spans,
    )


def render_criterion_difference_endpoint_inputs(
    factor_spec: FactorSpec | Mapping[str, Any],
    *,
    task_context: str,
    model_cot: str,
    omit_factor: bool = False,
) -> tuple[RenderedInput, RenderedInput]:
    """Render neutral applied/not-applied candidate branches for shared scoring."""

    if isinstance(factor_spec, FactorSpec):
        values = {
            "factor": factor_spec.factor,
            "relation": factor_spec.relation,
            "applied_criterion": factor_spec.applied_criterion,
            "not_applied_criterion": factor_spec.not_applied_criterion,
        }
    else:
        required = {"factor", "relation", "applied_criterion", "not_applied_criterion"}
        if set(factor_spec) != required:
            raise ValueError(
                f"factor_spec fields differ: missing={sorted(required-set(factor_spec))}, "
                f"unexpected={sorted(set(factor_spec)-required)}"
            )
        values = {
            key: _require_string(factor_spec[key], f"factor_spec.{key}")
            for key in required
        }
    if omit_factor:
        values = {**values, "factor": ""}
    common_fields = (
        ("context", "TASK_CONTEXT", _require_string(task_context, "task_context")),
        ("cot", "MODEL_COT", _require_string(model_cot, "model_cot")),
        ("factor", "FACTOR", values["factor"]),
        ("relation", "RELATION", values["relation"]),
    )
    marker = (
        "\n\n<DECISION_QUERY>\n"
        f"{CRITERION_DIFFERENCE_DECISION_QUERY}\n"
        "</DECISION_QUERY>\n\n<END_MONITOR_INPUT>"
    )

    def render(candidate: str) -> RenderedInput:
        rendered = _render_fields(
            CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID,
            (*common_fields, ("candidate", "CANDIDATE_CRITERION", "" if omit_factor else candidate)),
        )
        return RenderedInput(
            contract_id=rendered.contract_id,
            text=rendered.text + marker,
            spans=rendered.spans,
        )

    return (
        render(values["applied_criterion"]),
        render(values["not_applied_criterion"]),
    )


def render_endpoint_inputs(
    factor_spec: FactorSpec | Mapping[str, Any],
    *,
    task_context: str,
    model_cot: str,
    contract_id: str = ENDPOINT_CONTRACT_ID,
    omit_factor: bool = False,
) -> tuple[RenderedInput, ...]:
    """Render the one or two sequences registered by an endpoint contract."""

    if contract_id == CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID:
        return render_criterion_difference_endpoint_inputs(
            factor_spec,
            task_context=task_context,
            model_cot=model_cot,
            omit_factor=omit_factor,
        )
    return (
        render_endpoint_input(
            factor_spec,
            task_context=task_context,
            model_cot=model_cot,
            contract_id=contract_id,
            omit_factor=omit_factor,
        ),
    )


def endpoint_decision_mode(contract_id: str) -> str:
    if contract_id == CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID:
        return CRITERION_DIFFERENCE_DECISION_MODE
    if contract_id in {
        ENDPOINT_CONTRACT_ID,
        FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
        FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
        FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID,
    }:
        return SINGLE_ENDPOINT_DECISION_MODE
    raise ValueError(f"unknown endpoint encoding contract: {contract_id}")


def contract_spec(contract_id: str) -> dict[str, Any]:
    common = {
        "renderer_version": RENDERER_VERSION,
        "span_algorithm_version": SPAN_ALGORITHM_VERSION,
        "text_policy": {
            "strip": False,
            "unicode_normalization": False,
            "preserve_field_whitespace": True,
            "fixed_newline": "\\n",
            "literal_boundary_tags_allowed": True,
        },
        "tokenizer_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "return_offsets_mapping": True,
        },
        "manual_eos": False,
    }
    if contract_id == FACTOR_CONTRACT_ID:
        return {
            "contract_id": contract_id,
            "fields": ["factor", "applied", "not_applied"],
            "template": (
                "<FACTOR>\\n{factor}\\n</FACTOR>\\n\\n"
                "<APPLIED_CRITERION>\\n{applied}\\n</APPLIED_CRITERION>\\n\\n"
                "<NOT_APPLIED_CRITERION>\\n{not_applied}\\n</NOT_APPLIED_CRITERION>"
            ),
            **common,
        }
    if contract_id == EVIDENCE_CONTRACT_ID:
        return {
            "contract_id": contract_id,
            "fields": ["context", "cot"],
            "template": (
                "<TASK_CONTEXT>\\n{context}\\n</TASK_CONTEXT>\\n\\n"
                "<MODEL_COT>\\n{cot}\\n</MODEL_COT>"
            ),
            "visible_fields": ["task_context", "model_cot"],
            "forbidden_fields": ["final_answer", "tool_call", "all_messages"],
            **common,
        }
    if contract_id == ENDPOINT_CONTRACT_ID:
        return {
            "contract_id": contract_id,
            "fields": [
                "factor",
                "relation",
                "applied",
                "not_applied",
                "context",
                "cot",
            ],
            "template": (
                "<FACTOR>\\n{factor}\\n</FACTOR>\\n\\n"
                "<RELATION>\\n{relation}\\n</RELATION>\\n\\n"
                "<APPLIED_CRITERION>\\n{applied}\\n</APPLIED_CRITERION>\\n\\n"
                "<NOT_APPLIED_CRITERION>\\n{not_applied}\\n</NOT_APPLIED_CRITERION>\\n\\n"
                "<TASK_CONTEXT>\\n{context}\\n</TASK_CONTEXT>\\n\\n"
                "<MODEL_COT>\\n{cot}\\n</MODEL_COT>\\n\\n"
                "<END_MONITOR_INPUT>"
            ),
            "visible_fields": [
                "factor_spec.factor",
                "factor_spec.relation",
                "factor_spec.applied_criterion",
                "factor_spec.not_applied_criterion",
                "task_context",
                "model_cot",
            ],
            "forbidden_fields": ["final_answer", "tool_call", "all_messages"],
            "endpoint_marker": "<END_MONITOR_INPUT>",
            **common,
        }
    if contract_id in {
        FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
        FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    }:
        spec = {
            "contract_id": contract_id,
            "fields": [
                "context",
                "cot",
                "factor",
                "relation",
                "applied",
                "not_applied",
            ],
            "template": (
                "<TASK_CONTEXT>\\n{context}\\n</TASK_CONTEXT>\\n\\n"
                "<MODEL_COT>\\n{cot}\\n</MODEL_COT>\\n\\n"
                "<FACTOR>\\n{factor}\\n</FACTOR>\\n\\n"
                "<RELATION>\\n{relation}\\n</RELATION>\\n\\n"
                "<APPLIED_CRITERION>\\n{applied}\\n</APPLIED_CRITERION>\\n\\n"
                "<NOT_APPLIED_CRITERION>\\n{not_applied}\\n</NOT_APPLIED_CRITERION>\\n\\n"
                "<DECISION_QUERY>\\n"
                f"{FACTOR_SUFFIX_DECISION_QUERY}\\n"
                "</DECISION_QUERY>\\n\\n<END_MONITOR_INPUT>"
            ),
            "visible_fields": [
                "task_context",
                "model_cot",
                "factor_spec.factor",
                "factor_spec.relation",
                "factor_spec.applied_criterion",
                "factor_spec.not_applied_criterion",
            ],
            "forbidden_fields": ["final_answer", "tool_call", "all_messages"],
            "fixed_decision_query": FACTOR_SUFFIX_DECISION_QUERY,
            "endpoint_marker": "<END_MONITOR_INPUT>",
            **common,
        }
        if contract_id == FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID:
            spec["token_transform"] = {
                "version": RELAY_TOKEN_TRANSFORM_VERSION,
                "operation": "insert_after_each_complete_field_token_block",
                "token": FACTOR_SUFFIX_RELAY_TOKEN,
                "require_single_token": True,
                "fields": list(FACTOR_SUFFIX_RELAY_FIELDS),
                "field_counters": "independent_reset_per_field",
                "interval_tokens": FACTOR_SUFFIX_RELAY_INTERVAL,
                "attention_mask": 1,
                "field_span_masks": "false",
                "literal_collision_policy": "error_in_any_visible_field",
                "truncation": False,
            }
        return spec
    if contract_id == FACTOR_BOOKENDS_ENDPOINT_CONTRACT_ID:
        factor_fields = (
            "<FACTOR>\\n{factor}\\n</FACTOR>\\n\\n"
            "<RELATION>\\n{relation}\\n</RELATION>\\n\\n"
            "<APPLIED_CRITERION>\\n{applied}\\n</APPLIED_CRITERION>\\n\\n"
            "<NOT_APPLIED_CRITERION>\\n{not_applied}\\n</NOT_APPLIED_CRITERION>"
        )
        return {
            "contract_id": contract_id,
            "fields": [
                "factor_prefix",
                "relation_prefix",
                "applied_prefix",
                "not_applied_prefix",
                "context",
                "cot",
                "factor_suffix",
                "relation_suffix",
                "applied_suffix",
                "not_applied_suffix",
            ],
            "template": (
                f"{factor_fields}\\n\\n"
                "<TASK_CONTEXT>\\n{context}\\n</TASK_CONTEXT>\\n\\n"
                "<MODEL_COT>\\n{cot}\\n</MODEL_COT>\\n\\n"
                f"{factor_fields}\\n\\n"
                "<DECISION_QUERY>\\n"
                f"{FACTOR_SUFFIX_DECISION_QUERY}\\n"
                "</DECISION_QUERY>\\n\\n<END_MONITOR_INPUT>"
            ),
            "visible_fields": [
                "factor_spec.factor_repeated_exactly_twice",
                "factor_spec.relation_repeated_exactly_twice",
                "factor_spec.applied_criterion_repeated_exactly_twice",
                "factor_spec.not_applied_criterion_repeated_exactly_twice",
                "task_context",
                "model_cot",
            ],
            "forbidden_fields": ["final_answer", "tool_call", "all_messages"],
            "fixed_decision_query": FACTOR_SUFFIX_DECISION_QUERY,
            "endpoint_marker": "<END_MONITOR_INPUT>",
            **common,
        }
    if contract_id == CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID:
        return {
            "contract_id": contract_id,
            "fields": ["context", "cot", "factor", "relation", "candidate"],
            "branches": ["applied_criterion", "not_applied_criterion"],
            "branch_template": (
                "<TASK_CONTEXT>\\n{context}\\n</TASK_CONTEXT>\\n\\n"
                "<MODEL_COT>\\n{cot}\\n</MODEL_COT>\\n\\n"
                "<FACTOR>\\n{factor}\\n</FACTOR>\\n\\n"
                "<RELATION>\\n{relation}\\n</RELATION>\\n\\n"
                "<CANDIDATE_CRITERION>\\n{candidate}\\n</CANDIDATE_CRITERION>\\n\\n"
                "<DECISION_QUERY>\\n"
                f"{CRITERION_DIFFERENCE_DECISION_QUERY}\\n"
                "</DECISION_QUERY>\\n\\n<END_MONITOR_INPUT>"
            ),
            "visible_fields": [
                "task_context",
                "model_cot",
                "factor_spec.factor",
                "factor_spec.relation",
                "one_candidate_criterion_per_branch",
            ],
            "forbidden_fields": [
                "other_candidate_criterion",
                "branch_role_label",
                "final_answer",
                "tool_call",
                "all_messages",
            ],
            "fixed_decision_query": CRITERION_DIFFERENCE_DECISION_QUERY,
            "endpoint_marker": "<END_MONITOR_INPUT>",
            "decision_mode": CRITERION_DIFFERENCE_DECISION_MODE,
            **common,
        }
    raise ValueError(f"unknown encoding contract: {contract_id}")


def contract_hash(contract_id: str, tokenizer_identity: Mapping[str, Any]) -> str:
    if not isinstance(tokenizer_identity, Mapping) or not tokenizer_identity:
        raise ValueError("tokenizer_identity must be a non-empty mapping")
    return stable_hash(
        {"contract": contract_spec(contract_id), "tokenizer_identity": dict(tokenizer_identity)}
    )


def _flat_sequence(value: Any, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"tokenizer {name} must be a sequence")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"tokenizer {name} unexpectedly contains a batch")
        value = value[0]
    return list(value)


def _flat_offsets(value: Any) -> list[Any]:
    """Accept [tokens, 2] offsets while rejecting a real batch dimension."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("tokenizer offset_mapping must be a sequence")
    if value and isinstance(value[0], (list, tuple)):
        first = value[0]
        if len(first) == 2 and not any(isinstance(item, (list, tuple)) for item in first):
            return list(value)
        if len(value) == 1:
            nested = value[0]
            if isinstance(nested, (list, tuple)):
                return list(nested)
    raise ValueError("tokenizer offset_mapping has an invalid shape or contains a batch")


def tokenize_rendered(tokenizer: Any, rendered: RenderedInput) -> TokenizedInput:
    """Tokenize once and derive masks only from authoritative character offsets."""

    rendered = rendered_for_tokenizer(tokenizer, rendered)

    try:
        encoded = tokenizer(
            rendered.text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "tokenizer must provide reliable offset_mapping for the frozen raw contract"
        ) from exc
    if "offset_mapping" not in encoded:
        raise ValueError("tokenizer did not return offset_mapping; refusing ambiguous spans")
    input_ids = [int(item) for item in _flat_sequence(encoded["input_ids"], "input_ids")]
    attention = [int(item) for item in _flat_sequence(encoded["attention_mask"], "attention_mask")]
    offsets_raw = _flat_offsets(encoded["offset_mapping"])
    offsets: list[tuple[int, int]] = []
    for item in offsets_raw:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"invalid tokenizer offset: {item!r}")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start or end > len(rendered.text):
            raise ValueError(f"tokenizer offset is outside rendered text: {(start, end)}")
        offsets.append((start, end))
    if not (len(input_ids) == len(attention) == len(offsets)):
        raise ValueError("tokenizer ids/mask/offset lengths differ")
    if any(item not in (0, 1) for item in attention):
        raise ValueError("tokenizer attention_mask must be binary")

    masks: dict[str, tuple[bool, ...]] = {}
    claimed = [False] * len(input_ids)
    for name, span in rendered.spans.items():
        values: list[bool] = []
        for index, ((start, end), attended) in enumerate(zip(offsets, attention, strict=True)):
            if not attended or start == end:
                values.append(False)
                continue
            # Empty ablation fields intentionally contain no content.  Some
            # tokenizers assign the newline/tag boundary token spanning both
            # sides of an empty field; it must remain structural rather than
            # being attributed to Factor/criterion content.
            if span.start == span.end:
                values.append(False)
                continue
            overlaps = start < span.end and end > span.start
            contained = start >= span.start and end <= span.end
            if overlaps and not contained:
                crosses_start = start < span.start and end > span.start
                crosses_end = start < span.end and end > span.end
                side = (
                    "both"
                    if crosses_start and crosses_end
                    else "start"
                    if crosses_start
                    else "end"
                )
                raise FieldBoundaryError(
                    field=name,
                    token_index=index,
                    side=side,
                )
            values.append(contained)
            if contained:
                if claimed[index]:
                    raise ValueError(f"token {index} belongs to multiple field spans")
                claimed[index] = True
        masks[name] = tuple(values)
    if rendered.contract_id == FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID:
        factor_suffix_relay_insertion_count(rendered, offsets, attention)
        relay_token_id = factor_suffix_relay_token_id(tokenizer)

        transformed_ids: list[int] = []
        transformed_attention: list[int] = []
        transformed_masks = {name: [] for name in masks}
        field_counts = {name: 0 for name in FACTOR_SUFFIX_RELAY_FIELDS}
        for index, (token_id, attended) in enumerate(
            zip(input_ids, attention, strict=True)
        ):
            transformed_ids.append(token_id)
            transformed_attention.append(attended)
            for name, values in masks.items():
                transformed_masks[name].append(values[index])

            insert_relay = False
            for name in FACTOR_SUFFIX_RELAY_FIELDS:
                if masks[name][index]:
                    field_counts[name] += 1
                    insert_relay = (
                        field_counts[name] % FACTOR_SUFFIX_RELAY_INTERVAL == 0
                    )
                    break
            if insert_relay:
                transformed_ids.append(relay_token_id)
                transformed_attention.append(1)
                for values in transformed_masks.values():
                    values.append(False)

        input_ids = transformed_ids
        attention = transformed_attention
        masks = {
            name: tuple(values) for name, values in transformed_masks.items()
        }

    return TokenizedInput(
        contract_id=rendered.contract_id,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention),
        span_masks=masks,
    )
