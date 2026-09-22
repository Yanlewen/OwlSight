"""Phase-1 factor-aware monitor architecture and training framework.

The package keeps ML dependencies optional at import time. Import concrete model
or backend modules only in environments that provide the ``training`` extra.
"""

from .config import (
    BACKEND_NAME,
    FactorMonitorArchitecture,
    load_architecture,
    validate_factor_monitor_effective_config,
)
from .encoding import (
    CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID,
    DEFAULT_FIELD_BOUNDARY_POLICY,
    ENDPOINT_CONTRACT_ID,
    FIELD_BOUNDARY_POLICIES,
    FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
    EVIDENCE_CONTRACT_ID,
    FACTOR_CONTRACT_ID,
    render_criterion_difference_endpoint_inputs,
    render_evidence,
    render_endpoint_inputs,
    render_endpoint_input,
    render_factor,
    boundary_safe_rendered,
    set_field_boundary_policy,
)

__all__ = [
    "BACKEND_NAME",
    "CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID",
    "DEFAULT_FIELD_BOUNDARY_POLICY",
    "EVIDENCE_CONTRACT_ID",
    "ENDPOINT_CONTRACT_ID",
    "FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID",
    "FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID",
    "FACTOR_CONTRACT_ID",
    "FIELD_BOUNDARY_POLICIES",
    "FactorMonitorArchitecture",
    "load_architecture",
    "render_criterion_difference_endpoint_inputs",
    "render_evidence",
    "render_endpoint_inputs",
    "render_endpoint_input",
    "render_factor",
    "boundary_safe_rendered",
    "set_field_boundary_policy",
    "validate_factor_monitor_effective_config",
]
