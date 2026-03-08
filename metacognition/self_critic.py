"""
Self-Critic — the nawal brain's internal quality reviewer.

SelfCritic implements AbstractCritic using a purely classical, rule-based
critique pipeline.  It evaluates a candidate response against a configurable
set of check functions and returns a CritiqueResult describing:

  - Whether to approve the response or flag it.
  - Specific issues detected.
  - An optional revised response (for auto-fixable problems).
  - A confidence score for the critique itself.

Architecture note (Phase 3 → Phase 5)
--------------------------------------
Phase 3: Each check is a deterministic function (length, pattern, alignment).
Phase 5: Checks can be replaced with or augmented by neural self-scoring
         (e.g. log-probability-based quality estimation).

Built-in checks (all configurable):
  - min_length        : response must exceed a minimum word count
  - max_length        : response must not exceed a maximum word count
  - no_empty_response : response must not be blank / whitespace-only
  - goal_alignment    : response text must overlap with goal keywords
  - no_hallucination_markers : detects known hedging phrases that indicate
                               the model is unsure of a claim it presents
                               as fact.
  - no_refusal        : response should not be a plain refusal without reason
  - response_coherence: basic sentence and punctuation structure

Usage::

    critic = SelfCritic()
    result = critic.critique(
        response="The capital of France is Lyon.",
        context={"goal": "What is the capital of France?"},
    )
    if not result.approved:
        print(result.issues)
        print(result.revised_response)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from metacognition.interfaces import (
    AbstractCritic,
    ConfidenceScore,
    CritiqueResult,
)


# --------------------------------------------------------------------------- #
# Check helpers                                                                #
# --------------------------------------------------------------------------- #

CheckFn = Callable[[str, Dict[str, Any]], Optional[str]]
"""
A check function takes (response_text, context) and returns:
  - None   → check passed
  - str    → issue description (check failed)
"""


def _check_min_length(min_words: int = 3) -> CheckFn:
    def _check(text: str, _ctx: Dict) -> Optional[str]:
        words = text.split()
        if len(words) < min_words:
            return f"Response too short ({len(words)} words, minimum {min_words})"
        return None
    return _check


def _check_max_length(max_words: int = 2048) -> CheckFn:
    def _check(text: str, _ctx: Dict) -> Optional[str]:
        words = text.split()
        if len(words) > max_words:
            return f"Response too long ({len(words)} words, maximum {max_words})"
        return None
    return _check


def _check_not_empty() -> CheckFn:
    def _check(text: str, _ctx: Dict) -> Optional[str]:
        if not text or not text.strip():
            return "Response is empty"
        return None
    return _check


def _check_goal_alignment(min_overlap: float = 0.0) -> CheckFn:
    """
    Warn (not fail) when text has zero keyword overlap with the stated goal.
    ``min_overlap=0.0`` disables this check by default — enable by passing > 0.
    """
    def _check(text: str, ctx: Dict) -> Optional[str]:
        if min_overlap <= 0.0:
            return None
        goal = str(ctx.get("goal", ctx.get("description", "")))
        if not goal:
            return None
        goal_words = set(re.findall(r"\w+", goal.lower())) - _STOPWORDS
        text_words = set(re.findall(r"\w+", text.lower()))
        if not goal_words:
            return None
        overlap = len(goal_words & text_words) / len(goal_words)
        if overlap < min_overlap:
            return (
                f"Response may not address the goal "
                f"(keyword overlap {overlap:.0%} < required {min_overlap:.0%})"
            )
        return None
    return _check


def _check_no_hallucination_markers() -> CheckFn:
    """Detect hedging phrases that suggest the model is fabricating facts."""
    _PATTERNS = [
        r"\bI (think|believe|suppose|guess) (the answer|it|that)\b",
        r"\b(as far as I know|to the best of my knowledge)\b",
        r"\bI('m| am) not (sure|certain) but\b",
        r"\b(may|might|could) be (wrong|incorrect|mistaken)\b",
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

    def _check(text: str, _ctx: Dict) -> Optional[str]:
        for pat in compiled:
            if pat.search(text):
                return (
                    f"Response contains possible hallucination marker: "
                    f"{pat.pattern!r}"
                )
        return None
    return _check


def _check_no_bare_refusal() -> CheckFn:
    """
    Flag responses that are pure one-line refusals without explanation.
    A legitimate refusal should include at least one sentence of context.
    """
    _REFUSALS = [
        r"^I('m| am) sorry, I can'?t (help|assist) with that\.?$",
        r"^(Sorry|I apologize), (I can'?t|that'?s not something I can) .{0,40}\.?$",
        r"^I('m| am) unable to (help|assist|answer) with this\.?$",
    ]
    compiled = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _REFUSALS]

    def _check(text: str, _ctx: Dict) -> Optional[str]:
        stripped = text.strip()
        for pat in compiled:
            if pat.match(stripped):
                return (
                    "Response is a bare refusal without context or alternatives"
                )
        return None
    return _check


def _check_coherence() -> CheckFn:
    """
    Minimal coherence: response should have at least one sentence-ending
    punctuation mark if it's longer than 10 words.
    """
    def _check(text: str, _ctx: Dict) -> Optional[str]:
        if len(text.split()) < 10:
            return None
        if not re.search(r"[.!?]", text):
            return "Response lacks sentence-terminating punctuation — may be truncated"
        return None
    return _check


# Common English stop-words (used in goal-alignment overlap computation)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "no", "with", "by", "from",
    "as", "be", "was", "are", "were", "has", "have", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "this", "that", "these", "those", "i", "me", "my", "we",
    "you", "he", "she", "they", "them", "their", "its",
})

# Default check registry
_DEFAULT_CHECKS: List[Tuple[str, CheckFn]] = [
    ("not_empty",              _check_not_empty()),
    ("min_length",             _check_min_length(min_words=2)),
    ("max_length",             _check_max_length(max_words=2048)),
    ("no_hallucination",       _check_no_hallucination_markers()),
    ("no_bare_refusal",        _check_no_bare_refusal()),
    ("coherence",              _check_coherence()),
    # goal alignment disabled by default (min_overlap=0)
    ("goal_alignment",         _check_goal_alignment(min_overlap=0.0)),
]


# --------------------------------------------------------------------------- #
# SelfCritic                                                                   #
# --------------------------------------------------------------------------- #

class SelfCritic(AbstractCritic):
    """
    Rule-based self-critic that applies configurable check functions to
    a candidate response.

    Args:
        checks         : List of (name, CheckFn) pairs.  Defaults to the
                         built-in suite if None.
        fail_threshold : Number of check failures that cause approved=False.
                         Default 1 (any failure → not approved).
        auto_revise    : If True, attempt simple auto-fixes (trim, add period).
    """

    def __init__(
        self,
        checks: Optional[List[Tuple[str, CheckFn]]] = None,
        fail_threshold: int = 1,
        auto_revise: bool = True,
    ) -> None:
        self._checks = checks if checks is not None else list(_DEFAULT_CHECKS)
        self._fail_threshold = fail_threshold
        self._auto_revise = auto_revise

    # ------------------------------------------------------------------ #
    # AbstractCritic implementation                                        #
    # ------------------------------------------------------------------ #

    def critique(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> CritiqueResult:
        """
        Run all registered checks against *response*.

        Returns CritiqueResult with:
          - approved=True if fewer than fail_threshold checks failed.
          - list of issue strings for each failed check.
          - optional revised_response if auto_revise is enabled.
        """
        issues: List[str] = []

        for check_name, check_fn in self._checks:
            try:
                issue = check_fn(response, context)
            except Exception as exc:
                logger.warning(f"SelfCritic check {check_name!r} raised: {exc}")
                issue = None
            if issue:
                issues.append(f"[{check_name}] {issue}")
                logger.debug(f"SelfCritic issue detected: {issue!r}")

        approved = len(issues) < self._fail_threshold

        revised: Optional[str] = None
        if not approved and self._auto_revise:
            revised = self._auto_fix(response, issues)

        confidence = self._compute_critique_confidence(response, issues)

        logger.info(
            f"SelfCritic: approved={approved} issues={len(issues)} "
            f"response_len={len(response.split())} words"
        )
        return CritiqueResult(
            approved=approved,
            issues=issues,
            revised_response=revised if not approved else None,
            confidence=confidence,
        )

    def estimate_confidence(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> ConfidenceScore:
        """
        Estimate confidence as a function of check pass-rate.

        1.0 = all checks pass, 0.0 = all checks fail.
        """
        n_checks = len(self._checks)
        if n_checks == 0:
            return ConfidenceScore(value=1.0, method="self_critic_passrate")

        n_fail = sum(
            1 for _, fn in self._checks
            if fn(response, context) is not None
        )
        value = round(1.0 - (n_fail / n_checks), 4)
        return ConfidenceScore(
            value=value,
            method="self_critic_passrate",
            explanation=f"{n_fail}/{n_checks} checks failed",
            metadata={"n_checks": n_checks, "n_fail": n_fail},
        )

    # ------------------------------------------------------------------ #
    # Check management                                                     #
    # ------------------------------------------------------------------ #

    def add_check(self, name: str, fn: CheckFn, prepend: bool = False) -> None:
        """Add a custom check at the end (or beginning) of the check list."""
        if prepend:
            self._checks.insert(0, (name, fn))
        else:
            self._checks.append((name, fn))

    def remove_check(self, name: str) -> None:
        """Remove a check by name."""
        self._checks = [(n, f) for n, f in self._checks if n != name]

    def check_names(self) -> List[str]:
        return [n for n, _ in self._checks]

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _auto_fix(self, response: str, issues: List[str]) -> str:
        """
        Apply simple deterministic fixes for common issues.
        Returns revised text (may still have issues — requires re-critique).
        """
        revised = response.strip()

        # Add terminal punctuation if coherence issue
        if any("punctuation" in iss for iss in issues):
            if revised and revised[-1] not in ".!?":
                revised += "."

        # Truncate if too long
        if any("too long" in iss for iss in issues):
            words = revised.split()
            revised = " ".join(words[:256]) + " [truncated]"

        return revised if revised != response.strip() else response

    def _compute_critique_confidence(
        self, response: str, issues: List[str]
    ) -> ConfidenceScore:
        n = max(len(self._checks), 1)
        ratio = max(0.0, 1.0 - len(issues) / n)
        return ConfidenceScore(
            value=round(ratio, 4),
            method="self_critic_issue_ratio",
            metadata={"issues": len(issues), "checks": n},
        )
