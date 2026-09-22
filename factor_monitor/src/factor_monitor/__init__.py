"""Unified contracts and pipelines for factor-conditioned CoT monitors."""

from .prompt_contract import CONTRACT_ID, contract_hash, render_evidence_blocks, render_messages

__all__ = ["CONTRACT_ID", "contract_hash", "render_evidence_blocks", "render_messages"]
__version__ = "0.1.0"
