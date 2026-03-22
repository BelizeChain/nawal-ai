"""
nawal_types.py — Canonical shared data types for the Nawal Brain Architecture.

These types flow between subsystems (Perception → Cortex → Memory →
Control → Action → Maintenance) and give every layer a common language.

Usage::

    from nawal_types import WorldState, GenerationResult, FeedbackSignal

    ws = WorldState(text="Hello", modalities=["text"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Perception types                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class WorldState:
    """
    Unified world-state representation produced by the Perception Layer.

    Attributes:
        text             : Raw text extracted from the current context.
        image_embedding  : Dense vector from the Visual Cortex (optional).
        audio_transcript : Text transcribed from audio (optional).
        fused_embedding  : Multimodal fused vector (optional).
        modalities       : List of active modalities (e.g. ``["text", "image"]``).
        timestamp        : ISO-8601 creation time (empty string = not set).
        metadata         : Arbitrary extra data.
    """

    text: str | None = None
    image_embedding: list[float] | None = None
    audio_transcript: str | None = None
    fused_embedding: list[float] | None = None
    modalities: list[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_text(self) -> bool:
        return bool(self.text)

    def has_image(self) -> bool:
        return self.image_embedding is not None

    def has_audio(self) -> bool:
        return bool(self.audio_transcript)

    def is_multimodal(self) -> bool:
        return len(self.modalities) > 1


# --------------------------------------------------------------------------- #
# Generation / inference types                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class GenerationResult:
    """
    Result produced by the Core Cortex / LLM backbone.

    Attributes:
        prompt               : Input prompt sent to the model.
        response             : Generated response text.
        model_used           : Model identifier (e.g. ``"nawal-v1"``).
        confidence           : Scalar confidence in [0, 1].
        latency_ms           : Wall-clock generation time in milliseconds.
        token_count          : Number of tokens generated.
        memory_context_used  : Whether the response drew on memory retrieval.
        critique_applied     : Whether the MetacognitionLayer revised the output.
        safety_passed        : Whether the MaintenanceLayer approved the output.
        metadata             : Arbitrary extra data.
    """

    prompt: str = ""
    response: str = ""
    model_used: str = ""
    confidence: float = 1.0
    latency_ms: float = 0.0
    token_count: int = 0
    memory_context_used: bool = False
    critique_applied: bool = False
    safety_passed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.response)

    def summary(self) -> str:
        return (
            f"GenerationResult(model={self.model_used!r}, "
            f"confidence={self.confidence:.2f}, "
            f"latency_ms={self.latency_ms:.1f}, "
            f"safety_passed={self.safety_passed})"
        )


# --------------------------------------------------------------------------- #
# Feedback / RLHF types                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class FeedbackSignal:
    """
    Feedback signal used by the Valuation Layer and RLHF training loop.

    Attributes:
        prompt              : Original prompt that produced the response.
        response            : Response being evaluated.
        reward_score        : Composite reward in [-1, 1] (higher = better).
        human_rating        : Optional human preference score (0–1).
        safety_score        : Safety evaluation in [0, 1] (1 = fully safe).
        consistency_score   : Self-consistency measure in [0, 1].
        novelty_score       : Novelty / curiosity bonus in [0, 1].
        memory_utilized     : Whether memory retrieval improved the response.
        metadata            : Arbitrary extra labels or debug info.
    """

    prompt: str = ""
    response: str = ""
    reward_score: float = 0.0
    human_rating: float | None = None
    safety_score: float = 1.0
    consistency_score: float = 1.0
    novelty_score: float = 0.0
    memory_utilized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def composite_reward(
        self,
        w_reward: float = 0.5,
        w_safety: float = 0.3,
        w_novelty: float = 0.2,
    ) -> float:
        """Weighted composite of reward, safety, and novelty."""
        return (
            w_reward * self.reward_score
            + w_safety * self.safety_score
            + w_novelty * self.novelty_score
        )

    def is_positive(self) -> bool:
        return self.reward_score > 0.0


# --------------------------------------------------------------------------- #
# Convenience re-export                                                        #
# --------------------------------------------------------------------------- #

__all__ = [
    "FeedbackSignal",
    "GenerationResult",
    "WorldState",
]
