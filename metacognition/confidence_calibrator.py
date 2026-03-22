"""
Confidence Calibrator — estimates an agent's calibrated confidence.

Aggregates multiple heterogeneous *evidence signals* into a single
scalar confidence in [0, 1] attached to one candidate response or plan.

Design
------
Phase 3: all signals are heuristic (keyword, score-based, text-length).
Phase 5: can slot in actual log-probability signals from the transformer
         and semantic similarity from the embedding layer.

Built-in signal sources (all optional):
  - plan_score        : score field from a Plan object (float in [0,1])
  - memory_relevance  : highest cosine similarity from memory retrieval
  - critic_score      : ConfidenceScore.value from SelfCritic
  - consistency_score : ConsistencyResult.score from ConsistencyChecker
  - safety_score      : ValuationLayer-style safety signal
  - verbal_certainty  : naive text analysis (hedging phrases lower score)

Each signal has a configurable weight.
The final value is a weighted, clipped average.

Usage::

    cal = ConfidenceCalibrator()
    conf = cal.calibrate({
        "plan_score":     0.85,
        "critic_score":   0.90,
        "consistency":    0.75,
        "verbal_hedging": "I think this might be correct.",
    })
    print(conf.value)       # 0.827
    print(conf.method)      # "weighted_aggregate"
    print(conf.explanation) # "plan:0.85 critic:0.90 consistency:0.75 verbal:0.50"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from metacognition.interfaces import ConfidenceScore

# --------------------------------------------------------------------------- #
# Default signal weights                                                       #
# --------------------------------------------------------------------------- #

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "plan_score": 1.5,
    "critic_score": 1.5,
    "consistency": 1.2,
    "memory": 1.0,
    "safety": 1.0,
    "verbal": 0.8,
}

# Hedging phrases that reduce verbal confidence
_HEDGE_PATTERNS = [
    r"\bI (think|believe|suppose|guess)\b",
    r"\b(maybe|perhaps|possibly|probably)\b",
    r"\b(might|may|could) (be|not)\b",
    r"\b(I'?m not sure|I'?m uncertain|I'?m unsure)\b",
    r"\b(approximately|roughly|around|about)\b",
    r"\b(as far as I know|to my knowledge)\b",
]
_HEDGE_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGE_PATTERNS]

# Certainty boosters
_CERTAIN_PATTERNS = [
    r"\bdefin(itely|itively|ite)\b",
    r"\bcer(tainly|tain|tainly)\b",
    r"\bwithout (a )?doubt\b",
    r"\baccording to (the|official|verified)\b",
]
_CERTAIN_RE = [re.compile(p, re.IGNORECASE) for p in _CERTAIN_PATTERNS]


# --------------------------------------------------------------------------- #
# ConfidenceCalibrator                                                         #
# --------------------------------------------------------------------------- #


class ConfidenceCalibrator:
    """
    Combines multiple signal sources into a calibrated confidence score.

    Args:
        weights : Dict mapping signal name → weight.  Any signal present
                  in ``signals`` that is not in ``weights`` is ignored.
                  Defaults to ``_DEFAULT_WEIGHTS``.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self._weights = weights if weights is not None else dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        signals: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceScore:
        """
        Compute a weighted-average confidence from heterogeneous signals.

        Recognised keys in *signals*:
          - ``"plan_score"``   : float [0,1] from ClassicalPlanner
          - ``"critic_score"`` : float [0,1] from SelfCritic
          - ``"consistency"``  : float [0,1] from ConsistencyChecker
          - ``"memory"``       : float [0,1] top retrieval similarity
          - ``"safety"``       : float [0,1] from ValuationLayer
          - ``"verbal"``       : str — response text (parsed for hedging)

        Returns ConfidenceScore with method="weighted_aggregate".
        """
        parts: List[Tuple[str, float, float]] = []  # (name, value, weight)

        for name, weight in self._weights.items():
            val = self._extract_signal(name, signals)
            if val is not None:
                val = max(0.0, min(1.0, val))
                parts.append((name, val, weight))

        if not parts:
            logger.debug("ConfidenceCalibrator: no signals — returning 0.5")
            return ConfidenceScore(
                value=0.5,
                method="weighted_aggregate",
                explanation="no signals provided",
            )

        total_weight = sum(w for _, _, w in parts)
        weighted_sum = sum(v * w for _, v, w in parts)
        value = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.5

        explanation = " ".join(f"{n}:{v:.2f}" for n, v, _ in parts)
        logger.debug(f"ConfidenceCalibrator: value={value:.4f} signals={explanation}")

        return ConfidenceScore(
            value=value,
            method="weighted_aggregate",
            explanation=explanation,
            metadata={n: round(v, 4) for n, v, _ in parts},
        )

    def add_signal(self, name: str, weight: float) -> None:
        """Add/update a signal weight."""
        self._weights[name] = weight

    def remove_signal(self, name: str) -> None:
        self._weights.pop(name, None)

    def calibrate_text(self, text: str) -> float:
        """
        Extract a verbal-certainty score from raw text.

        Returns a value in [0, 1] — higher = more certain.
        """
        return self._verbal_certainty(text)

    # ------------------------------------------------------------------ #
    # Signal extraction                                                    #
    # ------------------------------------------------------------------ #

    def _extract_signal(self, name: str, signals: Dict[str, Any]) -> Optional[float]:
        """Find the value for *name* in *signals*, apply type conversion."""
        raw = signals.get(name)
        if raw is None:
            return None

        if name == "verbal":
            return self._verbal_certainty(str(raw))

        if isinstance(raw, (int, float)):
            return float(raw)

        # Coerce from ConfidenceScore
        if hasattr(raw, "value"):
            return float(raw.value)

        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.debug(
                f"ConfidenceCalibrator: cannot convert signal {name!r}={raw!r}"
            )
            return None

    def _verbal_certainty(self, text: str) -> float:
        """
        Score [0.5, 1.0] based on presence of hedging vs certainty markers.
        Starts at 0.75 (neutral), nudged down by hedges, up by certainties.
        """
        score = 0.75
        hedges = sum(1 for pat in _HEDGE_RE if pat.search(text))
        certains = sum(1 for pat in _CERTAIN_RE if pat.search(text))
        score -= hedges * 0.08
        score += certains * 0.05
        return round(max(0.1, min(1.0, score)), 4)
