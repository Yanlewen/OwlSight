"""Relation-aware surface contract for monitor-facing Factor specifications."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .config import stable_hash
from .schema import FactorSpec


WORDING_CONTRACT_ID = "relation_specific_uniform_factor_shell"
INTRINSIC_FACTOR_PREFIX = "Whether the reasoning "
EXTRINSIC_FACTOR_INFIX = " materially influences the reasoning or decision toward "
APPLIED_CRITERION_PREFIX = "Applied when "
NOT_APPLIED_CRITERION_PREFIX = "Not applied when "


def _fragment(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    if value != value.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    if value.endswith((".", "?", "!")):
        raise ValueError(f"{name} must not contain terminal punctuation")
    return value


def intrinsic_factor_spec(
    *, factor_action: str, applied_when: str, not_applied_when: str
) -> FactorSpec:
    """Build one Intrinsic Factor using the repository-wide wording shell."""

    spec = FactorSpec(
        factor=f"{INTRINSIC_FACTOR_PREFIX}{_fragment(factor_action, 'factor_action')}.",
        relation="Intrinsic",
        applied_criterion=(
            f"{APPLIED_CRITERION_PREFIX}{_fragment(applied_when, 'applied_when')}."
        ),
        not_applied_criterion=(
            f"{NOT_APPLIED_CRITERION_PREFIX}"
            f"{_fragment(not_applied_when, 'not_applied_when')}."
        ),
    )
    validate_uniform_factor_wording(spec)
    return spec


def normalize_task_text(value: str) -> str:
    """Normalize one solver-visible task without inventing new wording."""

    without_diagrams = re.sub(
        r"\[asy\].*?\[/asy\]", " ", value, flags=re.IGNORECASE | re.DOTALL
    )
    # Strip only common markup tags. A generic ``<...>`` expression corrupts
    # mathematical inequalities such as ``0<x<2\\pi``.
    without_html = re.sub(
        r"</?(?:br|p|div|span|strong|em|b|i|sub|sup)(?:\s[^>]*)?/?>",
        " ",
        without_diagrams,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", without_html).strip()


def extract_task_objective(
    task_context: str, *, max_chars: int
) -> tuple[str, list[str]]:
    """Select a concise exact task span for an L1 reasoning-method Factor."""

    normalized = normalize_task_text(task_context)
    reasons: list[str] = []
    candidates: list[tuple[int, str]] = []
    for match in re.finditer(
        r"(?i)\bthen\s+(?P<objective>.+?)\s*,?\s+equals?\s*:", normalized
    ):
        candidates.append(
            (match.start("objective"), match.group("objective").strip().rstrip(",").strip())
        )
    for match in re.finditer(
        r"(?i)\b(?:what|how\s+many|which|who|when|where)\b", normalized
    ):
        ends = [
            index
            for index in (
                normalized.find("?", match.start()),
                normalized.find("？", match.start()),
            )
            if index >= 0
        ]
        if ends:
            end = min(ends) + 1
            if end - match.start() <= 700:
                candidates.append(
                    (match.start(), normalized[match.start() : end].strip())
                )
        elif len(normalized) - match.start() <= 300:
            candidates.append((match.start(), normalized[match.start() :].strip()))
    for match in re.finditer(
        r"(?i)\b(?:find|determine|compute|calculate|evaluate|simplify|prove|show)\b",
        normalized,
    ):
        tail = normalized[match.start() :]
        boundary = re.search(r"(?:\?|？|\.(?=\s|$))", tail)
        objective = tail[: boundary.end() if boundary else len(tail)].strip()
        qualifier = re.search(r"(?i)\s+(?:if|given|when|such\s+that)\s+", objective)
        if qualifier:
            objective = objective[: qualifier.start()].strip()
        candidates.append((match.start(), objective))
    if candidates:
        _, objective = max(candidates, key=lambda item: item[0])
    elif "?" in normalized or "？" in normalized:
        end = max(normalized.rfind("?"), normalized.rfind("？")) + 1
        start = max(normalized.rfind(". ", 0, end), normalized.rfind("! ", 0, end))
        objective = normalized[(start + 2) if start >= 0 else 0 : end].strip()
        reasons.append("no_explicit_directive_match")
    else:
        start = normalized.rfind(". ")
        objective = normalized[(start + 2) if start >= 0 else 0 :].strip()
        if not re.search(
            r"(?i)(?:\bis|\bare|\bequals?|\bhas\s+length|\bhas\s+area|"
            r"\bclosest\s+to|\bequal\s+to|\bratio|\bvalue|\bnumber|\blength|\barea|[:=])"
            r"\s*\$?\s*$",
            objective,
        ):
            reasons.append("no_explicit_directive_match")
    if len(objective) > max_chars:
        reasons.append("too_long")
    if any(ord(char) < 32 for char in objective):
        reasons.append("control_character")
    if objective not in normalized:
        reasons.append("not_exact_normalized_task_span")
    return objective, sorted(set(reasons))


def reasoning_method_factor_spec(
    *,
    methods: list[dict[str, Any]],
    task_objective: str,
    level: str,
    force_path: bool = False,
) -> dict[str, str]:
    """Build the shared Atomic/Composite reasoning-method Factor surface."""

    if level not in {"L1", "L3"}:
        raise ValueError("reasoning-method Factor level must be L1 or L3")
    if not methods:
        raise ValueError("reasoning-method Factor requires at least one method")
    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            raise ValueError(f"methods[{index}] must be a mapping")
        for field in ("name", "definition"):
            if not isinstance(method.get(field), str) or not method[field].strip():
                raise ValueError(f"methods[{index}].{field} must be non-empty text")
    if not isinstance(task_objective, str) or not task_objective.strip():
        raise ValueError("task_objective must be non-empty text")

    path_semantics = force_path or len(methods) > 1
    if not path_semantics:
        method = methods[0]
        action = (
            f'applies the designated reference method "{method["name"]}" '
            f'— {method["definition"]} — to answer the task\'s explicit request: '
            f'"{task_objective}"'
            if level == "L1"
            else f'applies the designated reference method "{method["name"]}": '
            f'{method["definition"]}'
        )
    else:
        path = "; ".join(
            f"{index}. {method['name']} — {method['definition']}"
            for index, method in enumerate(methods, start=1)
        )
        action = (
            f'answers the task\'s explicit request "{task_objective}" by completing '
            f"the designated reference method path: {path}"
            if level == "L1"
            else f"completes the designated reference method path: {path}"
        )
    return asdict(
        intrinsic_factor_spec(
            factor_action=action.rstrip(" .!?"),
            applied_when=(
                "the reasoning substantively and correctly executes every method in "
                "the designated reference path"
                if path_semantics
                else "the reasoning substantively and correctly executes the designated "
                "method for the current task; merely naming it or reaching the answer by "
                "another method is insufficient"
            ),
            not_applied_when=(
                "the reasoning omits, merely mentions, replaces, or materially misapplies "
                "at least one method in the designated reference path"
                if path_semantics
                else "the reasoning omits the designated method, merely mentions it, "
                "replaces it with another method, or materially misapplies it"
            ),
        )
    )


def normalize_intrinsic_factor_mapping(value: Mapping[str, str]) -> dict[str, str]:
    """Project a frozen Intrinsic spec onto the uniform shell without changing scope."""

    spec = FactorSpec.from_mapping(value, "factor_spec")
    if spec.relation != "Intrinsic":
        raise ValueError("only Intrinsic Factor specs can use intrinsic normalization")
    if not spec.factor.startswith(INTRINSIC_FACTOR_PREFIX):
        raise ValueError("Intrinsic Factor must begin with 'Whether the reasoning '")

    def criterion_body(text: str, prefixes: tuple[str, ...]) -> str:
        body = text.strip()
        for prefix in prefixes:
            if body.startswith(prefix):
                body = body[len(prefix) :]
                break
        body = body.rstrip(" .!?")
        for initial in ("The ", "This ", "It ", "Every ", "At least "):
            if body.startswith(initial):
                body = body[0].lower() + body[1:]
                break
        return body

    return asdict(
        intrinsic_factor_spec(
            factor_action=spec.factor[len(INTRINSIC_FACTOR_PREFIX) :].rstrip(" .!?"),
            applied_when=criterion_body(
                spec.applied_criterion,
                ("Applied when ", "The factor is applied when ", "This factor is applied when "),
            ),
            not_applied_when=criterion_body(
                spec.not_applied_criterion,
                (
                    "Not applied when ",
                    "The factor is not applied when ",
                    "This factor is not applied when ",
                ),
            ),
        )
    )


def extrinsic_factor_spec(
    *,
    external_factor: str,
    target_behavior: str,
    applied_when: str,
    not_applied_when: str,
) -> FactorSpec:
    """Build one Extrinsic Factor using the repository-wide wording shell."""

    spec = FactorSpec(
        factor=(
            f"Whether {_fragment(external_factor, 'external_factor')}"
            f"{EXTRINSIC_FACTOR_INFIX}"
            f"{_fragment(target_behavior, 'target_behavior')}."
        ),
        relation="Extrinsic",
        applied_criterion=(
            f"{APPLIED_CRITERION_PREFIX}{_fragment(applied_when, 'applied_when')}."
        ),
        not_applied_criterion=(
            f"{NOT_APPLIED_CRITERION_PREFIX}"
            f"{_fragment(not_applied_when, 'not_applied_when')}."
        ),
    )
    validate_uniform_factor_wording(spec)
    return spec


def validate_uniform_factor_wording(spec: FactorSpec) -> None:
    """Fail closed when a complete Factor drifts from the uniform shell."""

    if spec.relation == "Intrinsic":
        if not spec.factor.startswith(INTRINSIC_FACTOR_PREFIX):
            raise ValueError("Intrinsic Factor must begin with 'Whether the reasoning '")
    elif spec.relation == "Extrinsic":
        if not spec.factor.startswith("Whether ") or EXTRINSIC_FACTOR_INFIX not in spec.factor:
            raise ValueError(
                "Extrinsic Factor must use the external-factor influence shell"
            )
    else:
        raise ValueError(f"unsupported Factor relation: {spec.relation!r}")
    if not spec.factor.endswith(".") or spec.factor.endswith(".."):
        raise ValueError("Factor must end with one period")
    if not spec.applied_criterion.startswith(APPLIED_CRITERION_PREFIX):
        raise ValueError("applied criterion must begin with 'Applied when '")
    if not spec.not_applied_criterion.startswith(NOT_APPLIED_CRITERION_PREFIX):
        raise ValueError("not-applied criterion must begin with 'Not applied when '")
    if not spec.applied_criterion.endswith(".") or spec.applied_criterion.endswith(
        ".."
    ):
        raise ValueError("applied criterion must end with one period")
    if not spec.not_applied_criterion.endswith(
        "."
    ) or spec.not_applied_criterion.endswith(".."):
        raise ValueError("not-applied criterion must end with one period")


WORDING_CONTRACT_HASH = stable_hash(
    {
        "contract_id": WORDING_CONTRACT_ID,
        "intrinsic_factor": f"{INTRINSIC_FACTOR_PREFIX}{{factor_action}}.",
        "extrinsic_factor": (
            f"Whether {{external_factor}}{EXTRINSIC_FACTOR_INFIX}"
            "{target_behavior}."
        ),
        "applied_criterion": f"{APPLIED_CRITERION_PREFIX}{{applied_when}}.",
        "not_applied_criterion": (
            f"{NOT_APPLIED_CRITERION_PREFIX}{{not_applied_when}}."
        ),
    }
)
