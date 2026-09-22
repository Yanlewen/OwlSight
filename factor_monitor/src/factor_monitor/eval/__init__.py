"""Standalone, CPU-only evaluation adapters.

The adapters consume frozen evaluation artifacts and persisted predictions.  They
never load a model or run inference, which keeps scoring reproducible and easy to
share with collaborators.
"""

from .binary import score_binary_artifact

__all__ = ["score_binary_artifact"]
