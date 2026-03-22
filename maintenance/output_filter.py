"""
OutputFilter — Safety filter applied to all model outputs.

Detection layers
----------------
1. **Harm classifier patterns** — regex-based detection of harmful content
   categories: violence, self-harm, illegal instructions, credentials.
2. **PII leak detector** — catch credentials / secrets in generated text.
3. **Refusal detector** — detect when the model has already self-refused
   (and not double-flag it).
4. **Hallucination heuristic** — detect common hallucination signals like
   fabricated citations (``[source: N/A]``, ``(Smith et al., 1984)`` style
   non-grounded claims) — flagged LOW only.

All layers produce a flag string and a severity.  The final ``RiskLevel``
is the maximum across all layers.

Design rationale
----------------
The filter is *conservative*: it prefers false positives over missing
genuinely harmful content.  Low-severity flags annotate the response for
downstream auditing but do not block it.
"""

from __future__ import annotations

import re

from loguru import logger

from maintenance.interfaces import (
    AbstractOutputFilter,
    FilterResult,
    RiskLevel,
)

# Re-use the _max_risk helper from input_screener to avoid duplication
try:
    from maintenance.input_screener import _max_risk
except ImportError:
    _RISK_ORDER = [
        RiskLevel.NONE,
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.BLOCKED,
    ]

    def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:  # type: ignore[misc]
        return a if _RISK_ORDER.index(a) >= _RISK_ORDER.index(b) else b


# --------------------------------------------------------------------------- #
# Default output filter patterns                                                #
# --------------------------------------------------------------------------- #

_HARM_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # Violence / gore
    (
        r"\b(step.by.step|instructions?|guide)\b.{0,60}" r"\b(murder|torture|mutilate|assault)\b",
        "violence_instructions",
        RiskLevel.BLOCKED,
    ),
    # Weapon construction in output — broad catch
    (
        r"\b(make|build|create|assemble|construct|wire\s+up|ignite|detonate)\b.{0,60}"
        r"\b(bomb|explosive|IED|charge|weapon|device)\b",
        "weapon_construction",
        RiskLevel.BLOCKED,
    ),
    # Drug synthesis in output
    (
        r"\b(add\s+the|combine\s+the|heat\s+the)\b.{0,60}" r"\b(precursor|reagent|synthesis)\b",
        "drug_synthesis_output",
        RiskLevel.BLOCKED,
    ),
    # Credentials / secrets leakage
    (
        r"\b(api[_\s]?key|secret[_\s]?key|password|passwd|token)\s*[:=]\s*\S{8,}",
        "credential_leak",
        RiskLevel.HIGH,
    ),
    # Private keys
    (
        r"-----BEGIN\s+(RSA|EC|PRIVATE)\s+KEY-----|" r"0x[a-fA-F0-9]{40,}",
        "private_key_leak",
        RiskLevel.BLOCKED,
    ),
    # Self-harm encouragement
    (
        r"\b(you\s+should|try|consider)\b.{0,30}"
        r"\b(hurt yourself|cut yourself|end your life|take your own life)\b",
        "self_harm_encouragement",
        RiskLevel.BLOCKED,
    ),
    # Hate speech generation
    (
        r"\b(all\s+(blacks?|whites?|jews?|muslims?|gays?)\s+(are|should|must|deserve))\b",
        "hate_speech",
        RiskLevel.BLOCKED,
    ),
]

_HALLUCINATION_HINTS: list[tuple[str, str, RiskLevel]] = [
    (r"\[source:\s*N/A\]", "hallucination_na_source", RiskLevel.LOW),
    (
        r"\(as of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\)",
        "stale_date_claim",
        RiskLevel.LOW,
    ),
    (
        r"according to [A-Z][a-z]+ et al\.,\s*\d{4}",
        "ungrounded_citation",
        RiskLevel.LOW,
    ),
]

_PII_OUTPUT: list[tuple[str, str, RiskLevel]] = [
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "ssn_in_output", RiskLevel.HIGH),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "cc_in_output", RiskLevel.HIGH),
]

# Common model self-refusal phrases (don't double-penalise these)
_REFUSAL_PHRASES = re.compile(
    r"\b(I'?m\s+(sorry|unable|not\s+able|not\s+going\s+to|can'?t)|"
    r"I\s+cannot|I\s+won'?t|As\s+an\s+AI|This\s+request\s+(violates?|is\s+against))\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# OutputFilter                                                                  #
# --------------------------------------------------------------------------- #


class OutputFilter(AbstractOutputFilter):
    """
    Screens generated model outputs before delivery to the user.

    Args:
        extra_patterns : Additional (pattern, label, RiskLevel) triples.
        max_output_len : Flag outputs longer than this many chars (default 20 000).
        pii_check      : Enable PII leak detection (default True).
        hallucination_hints : Enable (low-severity) hallucination flagging.
    """

    def __init__(
        self,
        extra_patterns: list[tuple[str, str, RiskLevel]] | None = None,
        max_output_len: int = 20_000,
        pii_check: bool = True,
        hallucination_hints: bool = True,
    ) -> None:
        self._max_len = max_output_len
        self._pii_check = pii_check
        self._hallucination_hints = hallucination_hints

        self._harm: list[tuple[re.Pattern, str, RiskLevel]] = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), lbl, lvl)
            for p, lbl, lvl in _HARM_PATTERNS + (extra_patterns or [])
        ]
        self._pii: list[tuple[re.Pattern, str, RiskLevel]] = [
            (re.compile(p, re.IGNORECASE), lbl, lvl) for p, lbl, lvl in _PII_OUTPUT
        ]
        self._hints: list[tuple[re.Pattern, str, RiskLevel]] = [
            (re.compile(p, re.IGNORECASE), lbl, lvl) for p, lbl, lvl in _HALLUCINATION_HINTS
        ]

        logger.info(
            f"OutputFilter ready: {len(self._harm)} harm patterns "
            f"pii={pii_check} hallucination={hallucination_hints}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def filter(self, prompt: str, response: str) -> FilterResult:
        """
        Screen *response* (with *prompt* context) for safety violations.

        Returns:
            FilterResult — is_safe=False means the response must not be delivered.
        """
        flags: list[str] = []
        max_level = RiskLevel.NONE
        filtered = response

        # Skip further checks if output is a model self-refusal
        if _REFUSAL_PHRASES.search(response):
            return FilterResult(
                is_safe=True,
                risk_level=RiskLevel.NONE,
                flags=["model_self_refused"],
                filtered=response,
            )

        # 1. Harm patterns
        for compiled, label, level in self._harm:
            if compiled.search(response):
                flags.append(label)
                max_level = _max_risk(max_level, level)

        # 2. PII leak detection
        if self._pii_check:
            for compiled, label, level in self._pii:
                if compiled.search(response):
                    flags.append(label)
                    max_level = _max_risk(max_level, level)

        # 3. Length guard — treat as HIGH to ensure is_safe=False
        if len(response) > self._max_len:
            flags.append("excessive_output_length")
            max_level = _max_risk(max_level, RiskLevel.HIGH)
            filtered = response[: self._max_len] + " [output truncated]"

        # 4. Hallucination hints (LOW only — doesn't block)
        if self._hallucination_hints:
            for compiled, label, level in self._hints:
                if compiled.search(response):
                    flags.append(label)
                    max_level = _max_risk(max_level, level)

        is_safe = max_level not in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.BLOCKED)

        if not is_safe:
            logger.warning(f"OutputFilter blocked response: risk={max_level} flags={flags}")
            filtered = "[Content blocked by safety filter]"

        return FilterResult(
            is_safe=is_safe,
            risk_level=max_level,
            flags=flags,
            filtered=filtered,
        )

    def is_safe(self, response: str) -> bool:
        """Quick boolean safety check (passes an empty prompt)."""
        return self.filter("", response).is_safe

    def add_pattern(self, pattern, label: str, level: RiskLevel = RiskLevel.HIGH) -> None:
        """Register an additional harm pattern at runtime.

        *pattern* may be a raw string or a pre-compiled ``re.Pattern``.
        """
        if isinstance(pattern, re.Pattern):
            compiled = pattern
        else:
            compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        self._harm.append((compiled, label, level))
