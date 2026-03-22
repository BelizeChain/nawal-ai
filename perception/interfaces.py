"""
Perception interfaces — Abstract Base Classes for Sensory Cortices.

Each sensory modality (text, image, audio) is an AbstractCortex that
produces a WorldState — the unified representation of "what is happening"
passed downstream to the Core Cortex and Memory systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerceptionWorldState:
    """
    Unified world-state representation produced by the Perception layer.

    All modalities contribute embeddings + metadata into a single object
    that is passed to the Core Cortex and Memory systems.

    Note: This is distinct from ``nawal_types.WorldState`` which carries
    higher-level text/audio content.  ``PerceptionWorldState`` holds the
    dense embedding vectors produced by sensory cortices.

    Attributes:
        text_embedding   : Dense vector from the textual cortex.
        image_embedding  : Dense vector from the visual cortex (None if absent).
        audio_embedding  : Dense vector from the auditory cortex (None if absent).
        raw_text         : Original text input (for debugging).
        metadata         : Modality-specific annotations (confidence, lang, …).
        fused_embedding  : Multimodal fusion vector (set by SensoryHub).
    """

    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    audio_embedding: list[float] | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fused_embedding: list[float] | None = None


# Backward-compatible alias — prefer PerceptionWorldState in new code.
WorldState = PerceptionWorldState


class AbstractCortex(ABC):
    """
    Abstract interface for a single sensory cortex (text / image / audio).

    A SensoryHub (Phase 3) aggregates multiple AbstractCortex instances
    and merges their outputs into a single WorldState via a fusion step.
    """

    @abstractmethod
    def encode(self, raw_input: Any) -> list[float]:
        """
        Convert raw modality input into a dense embedding vector.

        Args:
            raw_input : Modality-specific input (str, PIL.Image, np.ndarray …).

        Returns:
            Dense embedding as a flat list of floats.
        """

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        Apply modality-specific preprocessing (tokenize, resize, resample, …).

        Returns preprocessed input ready for ``encode``.
        """

    def perceive(self, raw_input: Any) -> WorldState:
        """
        High-level convenience: preprocess → encode → wrap in WorldState.

        Subclasses may override for more complex fusion logic.
        """
        preprocessed = self.preprocess(raw_input)
        embedding = self.encode(preprocessed)
        # Subclass sets the right field (text/image/audio) in __init__
        return self._to_world_state(embedding, raw_input)

    @abstractmethod
    def _to_world_state(self, embedding: list[float], raw_input: Any) -> WorldState:
        """Wrap an embedding in the correct WorldState field for this modality."""
