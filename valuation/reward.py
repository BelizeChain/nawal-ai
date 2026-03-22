"""
Drive-Based Reward Model — the valuation/limbic layer of the nawal brain.

Maps candidate responses / actions to scalar scores by aggregating a set of
named *drives* (internal motivational signals).  Each drive has a weight and
a pluggable evaluator callable.

Built-in drives (configurable at init):
  - safety      (weight 2.0) : penalises unsafe or risky content
  - alignment   (weight 1.5) : rewards responses that match the stated goal
  - novelty     (weight 1.0) : rewards new information or creative outputs
  - curiosity   (weight 0.8) : rewards information-seeking actions

Custom drives can be added with ``add_drive()``.

Usage::

    model = DriveBasedRewardModel()
    candidates = [
        {"text": "Paris is the capital of France.", "source": "knowledge"},
        {"text": "I don't know.", "source": "fallback"},
    ]
    ranked = model.ranked(candidates, context={"goal": "geography question"})
    best   = ranked[0]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from loguru import logger

try:
    from valuation.interfaces import AbstractRewardModel, DriveSignal
except ImportError:  # graceful fallback during testing
    from interfaces import AbstractRewardModel, DriveSignal  # type: ignore


# --------------------------------------------------------------------------- #
# Drive evaluator functions                                                    #
# --------------------------------------------------------------------------- #


def _safety_evaluator(candidate: dict[str, Any], context: dict[str, Any]) -> float:
    """
    Returns 0.0–1.0.  1.0 = perfectly safe, 0.0 = flagged as unsafe.

    Basic heuristic: penalise if the candidate's text matches known unsafe
    patterns.  A real implementation wires into BasicSafetyFilter / security
    modules.
    """
    text = str(candidate.get("text", candidate.get("content", "")))
    unsafe_patterns = [
        r"\bharm\b",
        r"\bdamage\b",
        r"\bdestroy\b",
        r"\battack\b",
        r"\binjure\b",
        r"\bkill\b",
        r"\bexploit\b",
        r"\bbypass\b",
        r"\billegal\b",
    ]
    for pat in unsafe_patterns:
        if re.search(pat, text, re.IGNORECASE):
            logger.debug(f"safety_evaluator: matched unsafe pattern {pat!r}")
            return 0.2
    return 1.0


def _alignment_evaluator(candidate: dict[str, Any], context: dict[str, Any]) -> float:
    """
    Returns 0.0–1.0.  Simple keyword overlap between candidate text and goal.

    Phase 3 will replace this with embedding cosine similarity.
    """
    goal = str(context.get("goal", context.get("description", "")))
    text = str(candidate.get("text", candidate.get("content", "")))
    if not goal or not text:
        return 0.5  # neutral when no goal is set
    goal_tokens = set(re.findall(r"\w+", goal.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not goal_tokens:
        return 0.5
    overlap = len(goal_tokens & text_tokens) / len(goal_tokens)
    return min(1.0, overlap)


def _novelty_evaluator(candidate: dict[str, Any], context: dict[str, Any]) -> float:
    """
    Returns 0.0–1.0.  Rewards candidates not already present in context history.

    Phase 3 replaces with embedding-based deduplication against SemanticMemory.
    """
    history = context.get("history", [])
    text = str(candidate.get("text", candidate.get("content", "")))
    for prev in history:
        prev_text = str(prev.get("text", prev.get("content", "")))
        if prev_text.strip() and text.strip() == prev_text.strip():
            return 0.1  # identical to previous — low novelty
    return 0.8


def _curiosity_evaluator(candidate: dict[str, Any], context: dict[str, Any]) -> float:
    """
    Returns 0.0–1.0.  Rewards action/search candidates that gather new info.

    Heuristic: information-seeking verbs or question marks.
    """
    text = str(candidate.get("text", candidate.get("content", "")))
    search_signals = [
        r"\?",
        r"\bsearch\b",
        r"\bquery\b",
        r"\blook up\b",
        r"\bexplore\b",
    ]
    for pat in search_signals:
        if re.search(pat, text, re.IGNORECASE):
            return 0.9
    if candidate.get("type") in ("search", "retrieve"):
        return 0.9
    return 0.4


# Default drive definitions
_DEFAULT_DRIVES: list[tuple[str, float, Callable]] = [
    ("safety", 2.0, _safety_evaluator),
    ("alignment", 1.5, _alignment_evaluator),
    ("novelty", 1.0, _novelty_evaluator),
    ("curiosity", 0.8, _curiosity_evaluator),
]


# --------------------------------------------------------------------------- #
# DriveBasedRewardModel                                                        #
# --------------------------------------------------------------------------- #


class DriveBasedRewardModel(AbstractRewardModel):
    """
    Reward model driven by a weighted sum of named drive evaluators.

    Args:
        drives : List of (name, weight, evaluator_fn) tuples.  Defaults to
                 the four built-in drives if None.
    """

    def __init__(
        self,
        drives: list[tuple[str, float, Callable]] | None = None,
    ) -> None:
        raw = drives if drives is not None else _DEFAULT_DRIVES
        self._drives: dict[str, tuple[float, Callable]] = {
            name: (weight, fn) for name, weight, fn in raw
        }

    # ------------------------------------------------------------------ #
    # Drive management                                                     #
    # ------------------------------------------------------------------ #

    def add_drive(
        self,
        name: str,
        weight: float,
        evaluator: Callable[[dict[str, Any], dict[str, Any]], float],
    ) -> None:
        """Add or replace a drive."""
        self._drives[name] = (weight, evaluator)
        logger.debug(f"DriveBasedRewardModel: drive {name!r} registered (w={weight})")

    def remove_drive(self, name: str) -> None:
        """Remove a drive by name."""
        self._drives.pop(name, None)

    def drive_names(self) -> list[str]:
        return list(self._drives.keys())

    # ------------------------------------------------------------------ #
    # AbstractRewardModel implementation                                   #
    # ------------------------------------------------------------------ #

    def score(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        drives: list[DriveSignal] | None = None,
    ) -> list[float]:
        """
        Score each candidate.

        Args:
            candidates : List of dicts with at least a ``text`` or ``content`` key.
            context    : Dict with optional keys: ``goal``, ``history``.
            drives     : Optional list of DriveSignal overrides.  If provided,
                         only these drives are used with their specified weights.

        Returns:
            List of float scores aligned with *candidates*.
        """
        ctx = context or {}
        effective_drives = self._resolve_drives(drives)
        total_weight = sum(w for _, (w, _) in effective_drives.items()) or 1.0

        scores: list[float] = []
        for candidate in candidates:
            weighted_total = 0.0
            for drive_name, (weight, evaluator) in effective_drives.items():
                try:
                    v = float(evaluator(candidate, ctx))
                    v = max(0.0, min(1.0, v))
                except Exception as exc:
                    logger.warning(f"Drive {drive_name!r} raised: {exc}; using 0.5")
                    v = 0.5
                weighted_total += weight * v
            scores.append(round(weighted_total / total_weight, 6))
        return scores

    def ranked(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        drives: list[DriveSignal] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return candidates sorted by descending score, each augmented with a
        ``_score`` key.
        """
        scores = self.score(candidates, context, drives)
        annotated = [{**c, "_score": s} for c, s in zip(candidates, scores, strict=False)]
        annotated.sort(key=lambda x: x["_score"], reverse=True)
        return annotated

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_drives(
        self,
        overrides: list[DriveSignal] | None,
    ) -> dict[str, tuple[float, Callable]]:
        """Merge override DriveSignals with registered drives."""
        if not overrides:
            return self._drives
        merged: dict[str, tuple[float, Callable]] = {}
        for sig in overrides:
            if sig.name in self._drives:
                _, fn = self._drives[sig.name]
                merged[sig.name] = (sig.weight, fn)
            else:
                logger.debug(f"DriveSignal {sig.name!r} has no registered evaluator; skipped.")
        return merged or self._drives  # fallback to all drives if none matched
