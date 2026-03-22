"""
Cortex — Canonical brain Cerebrum module (Phase 0 alias).

Provides the same public API as nawal.architecture.
Existing imports of the form:
    from nawal.architecture import NawalTransformer, NawalModelConfig
continue to work unchanged.

New canonical path (Phase 1+):
    from nawal.cortex import NawalTransformer, NawalModelConfig
"""

# Re-export everything from the underlying architecture package
from architecture.attention import MultiHeadAttention
from architecture.config import NawalModelConfig
from architecture.embeddings import NawalEmbeddings
from architecture.feedforward import FeedForward
from architecture.transformer import NawalTransformer

__all__ = [
    "FeedForward",
    "MultiHeadAttention",
    "NawalEmbeddings",
    "NawalModelConfig",
    "NawalTransformer",
]
