"""
Valuation interfaces — Abstract Base Classes for the Limbic System.

The Valuation layer assigns scalar scores to candidate actions / plans,
balancing multiple internal "drives" (curiosity, safety, alignment, novelty)
and enforcing hard safety constraints before any action is dispatched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DriveSignal:
    """
    An internal motivational scalar for a single "drive".

    Drives are the system's internal reward signals (not exposed to users).
    They bias action selection without constituting real emotion.

    Attributes:
        name   : Drive name (e.g. "curiosity", "safety", "novelty").
        value  : Current drive intensity (0.0–1.0).
        weight : Relative importance in composite scoring.
    """

    name: str
    value: float  # 0.0 = unsatisfied, 1.0 = fully satisfied
    weight: float = 1.0  # relative importance


class AbstractRewardModel(ABC):
    """
    Scores candidate actions / plans against internal drives.

    Classical implementation (Phase 2): RLHF preference model.
    Quantum override (Phase 5): multi-objective QAOA evaluator.
    """

    @abstractmethod
    def score(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
        drives: list[DriveSignal] | None = None,
    ) -> list[float]:
        """
        Assign a composite score to each candidate.

        Args:
            candidates : List of action / plan descriptors.
            context    : Current world state snapshot.
            drives     : Active internal drives with weights.

        Returns:
            List of float scores, parallel to *candidates*.
            Higher = more preferred.
        """

    @abstractmethod
    def ranked(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
        drives: list[DriveSignal] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return *candidates* sorted by descending score.
        """


class SafetyFilter(ABC):
    """
    Hard gate that rejects or rewrites unsafe outputs/actions.

    Wraps the existing ``security/`` modules (differential privacy,
    Byzantine detection) and RLHF safety classifiers.
    """

    @abstractmethod
    def is_safe(self, candidate: Any) -> bool:
        """
        Return True if *candidate* passes all safety checks.

        Args:
            candidate : Text, plan dict, or action descriptor to check.
        """

    @abstractmethod
    def filter(self, candidates: list[Any]) -> list[Any]:
        """
        Remove or rewrite unsafe entries.

        Returns only the candidates that pass ``is_safe``.
        """
