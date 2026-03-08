"""
Nawal Pure Transformer Architecture

100% sovereign implementation with NO pretrained weights or external dependencies.
All components built from scratch for Belizean AI sovereignty.
"""

from .config import NawalModelConfig
# Backward-compatible alias
NawalConfig = NawalModelConfig
from .transformer import NawalTransformer
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .embeddings import NawalEmbeddings

__all__ = [
    "NawalModelConfig",
    "NawalConfig",  # backward-compatible alias
    "NawalTransformer",
    "MultiHeadAttention",
    "FeedForward",
    "NawalEmbeddings",
]
