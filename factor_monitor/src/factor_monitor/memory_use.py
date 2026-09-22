"""Shared monitor-facing Factor contract for correct memory application."""

from __future__ import annotations

from .config import stable_hash
from .factor_wording import intrinsic_factor_spec


MEMORY_USE_FACTOR_CONTRACT_ID = "memory_use_monitor_factor_uniform_shell"
MEMORY_USE_RELATION = "Intrinsic"
MEMORY_USE_TASK_SCOPE = "task_complete"
MEMORY_USE_ATOMIC_SCOPE = "dependency_atomic"
MEMORY_USE_L3_FACTOR = (
    "Whether the reasoning correctly applies material information from "
    "available memory to the current task."
)
MEMORY_USE_APPLIED_CRITERION = (
    "Applied when the reasoning correctly identifies, interprets, transforms when "
    "needed, and operationalizes all material memory required by the task."
)
MEMORY_USE_NOT_APPLIED_CRITERION = (
    "Not applied when the reasoning omits, guesses, merely mentions, misunderstands, "
    "or incorrectly applies any material required memory."
)
MEMORY_USE_ATOMIC_APPLIED_CRITERION = (
    "Applied when the reasoning correctly identifies, interprets, transforms when "
    "needed, and operationalizes this memory dependency."
)
MEMORY_USE_ATOMIC_NOT_APPLIED_CRITERION = (
    "Not applied when the reasoning omits, guesses, merely mentions, misunderstands, "
    "or incorrectly applies this memory dependency."
)


def memory_use_factor_spec(factor: str) -> dict[str, str]:
    """Build one canonical monitor-facing memory-use Factor specification."""

    if not isinstance(factor, str) or not factor.strip():
        raise ValueError("memory-use factor must be non-empty text")
    factor = factor.strip()
    prefix = "Whether the reasoning "
    if not factor.startswith(prefix) or not factor.endswith("."):
        raise ValueError("memory-use factor must use the Intrinsic wording shell")
    spec = intrinsic_factor_spec(
        factor_action=factor[len(prefix) : -1],
        applied_when=(
            "the reasoning correctly identifies, interprets, transforms when needed, "
            "and operationalizes all material memory required by the task"
        ),
        not_applied_when=(
            "the reasoning omits, guesses, merely mentions, misunderstands, or "
            "incorrectly applies any material required memory"
        ),
    )
    return {
        "factor": spec.factor,
        "relation": spec.relation,
        "applied_criterion": spec.applied_criterion,
        "not_applied_criterion": spec.not_applied_criterion,
    }


def atomic_memory_use_factor_spec(factor: str) -> dict[str, str]:
    """Build a Factor specification for exactly one named memory dependency."""

    if not isinstance(factor, str) or not factor.strip():
        raise ValueError("atomic memory-use factor must be non-empty text")
    factor = factor.strip()
    prefix = "Whether the reasoning "
    if not factor.startswith(prefix) or not factor.endswith("."):
        raise ValueError("atomic memory-use factor must use the Intrinsic wording shell")
    spec = intrinsic_factor_spec(
        factor_action=factor[len(prefix) : -1],
        applied_when=(
            "the reasoning correctly identifies, interprets, transforms when needed, "
            "and operationalizes this memory dependency"
        ),
        not_applied_when=(
            "the reasoning omits, guesses, merely mentions, misunderstands, or "
            "incorrectly applies this memory dependency"
        ),
    )
    return {
        "factor": spec.factor,
        "relation": spec.relation,
        "applied_criterion": spec.applied_criterion,
        "not_applied_criterion": spec.not_applied_criterion,
    }


MEMORY_USE_L3_SPEC = memory_use_factor_spec(MEMORY_USE_L3_FACTOR)
MEMORY_USE_FACTOR_CONTRACT_HASH = stable_hash(
    {
        "contract_id": MEMORY_USE_FACTOR_CONTRACT_ID,
        "l3": MEMORY_USE_L3_SPEC,
    }
)
