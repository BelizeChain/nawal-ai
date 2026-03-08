"""
Cortex — Canonical brain Cerebrum module (Phase 0 alias).

Provides the same public API as nawal.architecture.
Existing imports of the form:
    from nawal.architecture import NawalTransformer, NawalConfig
continue to work unchanged.

New canonical path (Phase 1+):
    from nawal.cortex import NawalTransformer, NawalConfig
"""

# Re-export everything from the underlying architecture package
from architecture.transformer import NawalTransformer
from architecture.config import NawalConfig, NawalModelConfig
from architecture.attention import MultiHeadAttention
from architecture.feedforward import FeedForward
from architecture.embeddings import NawalEmbeddings

__all__ = [
    "NawalTransformer",
    "NawalConfig",
    "NawalModelConfig",
    "MultiHeadAttention",
    "FeedForward",
    "NawalEmbeddings",
]
