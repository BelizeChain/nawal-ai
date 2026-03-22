"""
Safety Filter and ValuationLayer — the nawal brain's immune / homeostasis tier.

BasicSafetyFilter
-----------------
Implements ``SafetyFilter`` ABC.  Two guard layers:

1. Keyword/pattern blocklist (configurable, sensible defaults).
2. Maximum output length guard.
3. Extensible hook for Byzantine-detection patterns (from
   ``security/byzantine_detection.py``).

ValuationLayer
--------------
A lightweight facade combining ``DriveBasedRewardModel`` and
``BasicSafetyFilter`` into a single callable layer.

Typical usage::

    vl = ValuationLayer()
    candidates = [
        {"text": "The answer is 42."},
        {"text": "I will exploit the system."},
    ]
    safe      = vl.filter_safe(candidates)
    best      = vl.best(safe, context={"goal": "answer arithmetic question"})
    all_ranked = vl.ranked(candidates, context={"goal": "..."})
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from valuation.interfaces import SafetyFilter, AbstractRewardModel, DriveSignal
    from valuation.reward import DriveBasedRewardModel
except ImportError:
    from interfaces import SafetyFilter, AbstractRewardModel, DriveSignal  # type: ignore
    from reward import DriveBasedRewardModel  # type: ignore


# --------------------------------------------------------------------------- #
# Default blocklist                                                             #
# --------------------------------------------------------------------------- #

_DEFAULT_BLOCKLIST: List[str] = [
    # Physical harm
    r"\bhow to (make|build|create|synthesize)\b.*\b(bomb|weapon|explosive|poison|virus)\b",
    r"\b(kill|murder|assault|torture)\b.{0,30}\b(person|people|human|child|user)\b",
    # Cybersecurity abuse
    r"\b(exploit|bypass|hack|crack|brute.?force)\b.{0,40}\b(password|auth|security|system)\b",
    r"\b(inject|payload|sql injection|xss|buffer overflow)\b",
    # Illegal activities
    r"\b(manufacture|synthesize|produce)\b.{0,30}\b(drug|narcotic|meth|heroin|cocaine)\b",
    r"\b(launder|laundering)\b.{0,20}\b(money|fund|asset)\b",
    # Sensitive / hate
    r"\b(hate speech|racial slur|ethnic cleansing)\b",
    # Self-harm (block generation, not detection)
    r"\bhow to (commit suicide|self.?harm|cut myself)\b",
]


# --------------------------------------------------------------------------- #
# BasicSafetyFilter                                                            #
# --------------------------------------------------------------------------- #


class BasicSafetyFilter(SafetyFilter):
    """
    Text-output safety filter.

    Args:
        blocklist    : Regex patterns to treat as unsafe.  Defaults to
                       ``_DEFAULT_BLOCKLIST`` if None.  Pass ``[]`` to
                       disable all keyword checks.
        max_length   : Maximum allowed character length.  0 = no limit.
        extra_checks : Optional list of ``(name, callable)`` pairs for
                       custom checks.  Each callable takes a string and
                       returns ``(is_safe: bool, reason: str)``.
    """

    def __init__(
        self,
        blocklist: Optional[List[str]] = None,
        max_length: int = 0,
        extra_checks: Optional[List[Tuple[str, Any]]] = None,
    ) -> None:
        raw_patterns = blocklist if blocklist is not None else _DEFAULT_BLOCKLIST
        self._patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in raw_patterns
        ]
        self._max_length = max_length
        self._extra_checks = extra_checks or []

    # ------------------------------------------------------------------
    # SafetyFilter ABC
    # ------------------------------------------------------------------

    def is_safe(self, candidate: Any) -> bool:
        """
        Return True iff *candidate* passes all safety checks.

        *candidate* may be a ``str`` or a ``dict`` with a ``text``/``content`` key.
        """
        text = self._extract_text(candidate)
        safe, _ = self._check(text)
        return safe

    def filter(self, candidates: List[Any]) -> List[Any]:
        """Return only the candidates that are safe."""
        result = []
        for c in candidates:
            text = self._extract_text(c)
            safe, reason = self._check(text)
            if safe:
                result.append(c)
            else:
                logger.info(f"BasicSafetyFilter blocked candidate: {reason}")
        return result

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def check_with_reason(self, candidate: Any) -> Tuple[bool, str]:
        """Return (is_safe, reason_string)."""
        text = self._extract_text(candidate)
        return self._check(text)

    def add_pattern(self, pattern: str) -> None:
        """Add a regex pattern to the blocklist at runtime."""
        self._patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))

    def add_check(self, name: str, fn: Any) -> None:
        """Add a custom check callable ``fn(text) -> (bool, str)``."""
        self._extra_checks.append((name, fn))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check(self, text: str) -> Tuple[bool, str]:
        if self._max_length > 0 and len(text) > self._max_length:
            reason = f"length {len(text)} exceeds max {self._max_length}"
            return False, reason

        for pat in self._patterns:
            m = pat.search(text)
            if m:
                return False, f"matched blocklist pattern: {pat.pattern!r}"

        for check_name, fn in self._extra_checks:
            try:
                ok, reason = fn(text)
                if not ok:
                    return False, f"{check_name}: {reason}"
            except Exception as exc:
                logger.warning(f"Safety check {check_name!r} raised: {exc}")

        return True, "ok"

    @staticmethod
    def _extract_text(candidate: Any) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            return str(candidate.get("text", candidate.get("content", "")))
        return str(candidate)


# --------------------------------------------------------------------------- #
# ValuationLayer                                                               #
# --------------------------------------------------------------------------- #


class ValuationLayer:
    """
    Unified facade for scoring + filtering candidates.

    Combines ``DriveBasedRewardModel`` and ``BasicSafetyFilter``.

    Usage::

        vl  = ValuationLayer()
        ctx = {"goal": "answer a maths question"}

        safe    = vl.filter_safe(candidates)
        ranked  = vl.ranked(safe, context=ctx)
        best    = vl.best(safe, context=ctx)

    Args:
        reward_model  : A ``DriveBasedRewardModel`` (or any ``AbstractRewardModel``).
                        Defaults to a new ``DriveBasedRewardModel`` with built-in drives.
        safety_filter : A ``BasicSafetyFilter`` (or any ``SafetyFilter``).
                        Defaults to a new ``BasicSafetyFilter`` with default blocklist.
    """

    def __init__(
        self,
        reward_model: Optional[AbstractRewardModel] = None,
        safety_filter: Optional[SafetyFilter] = None,
    ) -> None:
        self._reward = (
            reward_model if reward_model is not None else DriveBasedRewardModel()
        )
        self._safety = (
            safety_filter if safety_filter is not None else BasicSafetyFilter()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_safe(self, candidates: List[Any]) -> List[Any]:
        """Return only safe candidates."""
        return self._safety.filter(candidates)

    def is_safe(self, candidate: Any) -> bool:
        return self._safety.is_safe(candidate)

    def score(
        self,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        drives: Optional[List[DriveSignal]] = None,
    ) -> List[float]:
        """Score candidates (unsafe ones receive 0.0)."""
        safe_mask = [self._safety.is_safe(c) for c in candidates]
        raw_scores = self._reward.score(candidates, context, drives)
        return [s if safe else 0.0 for s, safe in zip(raw_scores, safe_mask)]

    def ranked(
        self,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        drives: Optional[List[DriveSignal]] = None,
        include_unsafe: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return candidates sorted by score (descending).

        Unsafe candidates are scored 0.0 and appear last (unless
        ``include_unsafe=False``, in which case they are omitted).
        """
        scored = []
        for c in candidates:
            safe = self._safety.is_safe(c)
            if not safe and not include_unsafe:
                continue
            scored.append(c)

        return self._reward.ranked(scored, context, drives)

    def best(
        self,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        drives: Optional[List[DriveSignal]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the single highest-scoring safe candidate, or None.
        """
        ranked = self.ranked(candidates, context, drives, include_unsafe=False)
        return ranked[0] if ranked else None

    @property
    def reward_model(self) -> AbstractRewardModel:
        return self._reward

    @property
    def safety_filter(self) -> SafetyFilter:
        return self._safety
