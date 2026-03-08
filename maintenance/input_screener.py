"""
InputScreener — Filters incoming prompts before they reach the model.

Detection layers (in order)
----------------------------
1. **Blocklist patterns** — compiled regex for known injection, jailbreak,
   and harmful-content patterns.
2. **Length guard** — prompts above a configurable maximum are flagged.
3. **PII detector** — simple heuristic for e-mail, phone, credit-card, SSN.
4. **Injection heuristic** — structural signals of prompt-injection attacks
   (e.g. nested quote-escape sequences, role-override instructions).

Design
------
Each layer produces a ``flag`` string and a severity score in [0, 1].
The final ``RiskLevel`` is the maximum across all layers.

All patterns are configurable at construction time so callers can add
domain-specific rules without subclassing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from maintenance.interfaces import (
    AbstractInputScreener,
    RiskLevel,
    ScreeningResult,
)


# --------------------------------------------------------------------------- #
# Default block patterns                                                        #
# --------------------------------------------------------------------------- #

_BLOCKED: List[Tuple[str, str, RiskLevel]] = [
    # (pattern, label, severity)

    # Harm / physical violence
    (
        r"\b(how\s+to|steps\s+to)\s+.{0,20}(kill|murder|assault|bomb|poison|stab)\b",
        "physical_harm_instructions",
        RiskLevel.BLOCKED,
    ),
    # Weapon synthesis (includes nerve agents, bioweapons)
    (
        r"\b(make|build|create|synthesize|manufacture|produce|acquire)\b.{0,40}"
        r"\b(bomb|weapon|explosive|sarin|vx|novichok|nerve.?agent|anthrax|ricin"
        r"|poison|virus|malware|ransomware|bioweapon)\b"
        r"|\b(synthesis\s+(of|for)|steps\s+to\s+(make|build|create))\b.{0,40}"
        r"\b(nerve.?agent|chemical.?weapon|bio.?weapon)\b",
        "weapon_synthesis",
        RiskLevel.BLOCKED,
    ),
    # Drug synthesis
    (
        r"\b(synthesize|manufacture|produce|cook|make)\b.{0,40}"
        r"\b(drug|narcotic|meth(?:amphetamine)?|heroin|cocaine|fentanyl|mdma|lsd)\b"
        r"|\b(synthesis\s+(of|for)|synthesis\s+route)\b.{0,40}"
        r"\b(meth(?:amphetamine)?|heroin|cocaine|drug)\b",
        "drug_synthesis",
        RiskLevel.BLOCKED,
    ),
    # Jailbreak / role override
    (
        r"(\bignore\b.{0,50}\binstructions?\b|"
        r"you\s+are\s+now\s+(DAN|GPT|an?\s+unrestricted|jailbroken|evil)|"
        r"pretend\s+(you\s+have\s+no\s+restrictions?|to\s+be\s+evil|you\s+are\s+allowed)|"
        r"disregard\s+(safety|ethics|guidelines|rules))",
        "jailbreak_attempt",
        RiskLevel.BLOCKED,
    ),
    # Prompt injection
    (
        r"(</?(system|user|assistant|[Ii]nstruction)>|"
        r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|"
        r"\\n#+\s*(system|assistant)\s*:)",
        "prompt_injection",
        RiskLevel.HIGH,
    ),
    # Cybersecurity exploitation
    (
        r"\b(exploit|bypass|hack|crack|brute.?force)\b.{0,40}"
        r"\b(password|auth|authentication|security|firewall)\b",
        "cyber_exploitation",
        RiskLevel.HIGH,
    ),
    # Child safety
    (
        r"\b(child|minor|underage)\b.{0,20}\b(nude|naked|sexual|porn)\b",
        "csam",
        RiskLevel.BLOCKED,
    ),
    # Self-harm generation
    (
        r"\b(how\s+to|best\s+way\s+to)\b.{0,20}\b(suicide|self.?harm|cut\s+myself)\b",
        "self_harm_instructions",
        RiskLevel.BLOCKED,
    ),
    # Money laundering
    (
        r"\b(launder|laundering)\b.{0,20}\b(money|funds?|asset)\b",
        "money_laundering",
        RiskLevel.HIGH,
    ),
]

_PII_PATTERNS: List[Tuple[str, str, RiskLevel]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "email_pii",    RiskLevel.LOW),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                     "phone_pii",    RiskLevel.LOW),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",               "credit_card",  RiskLevel.MEDIUM),
    (r"\b\d{3}-\d{2}-\d{4}\b",                                  "ssn_pii",      RiskLevel.MEDIUM),
]

_INJECTION_HEURISTICS: List[Tuple[str, str, RiskLevel]] = [
    # Excessive special characters often used in injection payloads
    (r"(\\n|\\r){5,}", "excessive_escapes", RiskLevel.MEDIUM),
    # Any null byte(s) — common injection smuggling technique
    (r"\x00+", "null_bytes", RiskLevel.HIGH),
]

# Safe content: always passes
_SAFE_PASSLENGTH = 10_000   # Characters — prompts beyond this get flagged LOW


# --------------------------------------------------------------------------- #
# InputScreener                                                                 #
# --------------------------------------------------------------------------- #

class InputScreener(AbstractInputScreener):
    """
    Screens incoming prompts for safety violations.

    Args:
        extra_patterns : Additional (pattern, label, RiskLevel) triples.
        max_prompt_len : Flag prompts longer than this many chars (default 10 000).
        pii_check      : Enable PII detection (default True).
    """

    def __init__(
        self,
        extra_patterns: Optional[List[Tuple[str, str, RiskLevel]]] = None,
        max_prompt_len: int = _SAFE_PASSLENGTH,
        pii_check: bool = True,
    ) -> None:
        self._patterns: List[Tuple[re.Pattern, str, RiskLevel]] = []
        self._max_len  = max_prompt_len
        self._pii_check = pii_check

        for pat, label, level in _BLOCKED + (extra_patterns or []):
            self._patterns.append((re.compile(pat, re.IGNORECASE | re.DOTALL), label, level))

        self._pii: List[Tuple[re.Pattern, str, RiskLevel]] = [
            (re.compile(p, re.IGNORECASE), lbl, lvl)
            for p, lbl, lvl in _PII_PATTERNS
        ]
        self._heuristics: List[Tuple[re.Pattern, str, RiskLevel]] = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), lbl, lvl)
            for p, lbl, lvl in _INJECTION_HEURISTICS
        ]

        logger.info(
            f"InputScreener ready: {len(self._patterns)} patterns "
            f"max_len={max_prompt_len}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def screen(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScreeningResult:
        """
        Analyse *prompt* for safety violations.

        Returns a ScreeningResult where is_safe=False means the request
        should be blocked before reaching the model.
        """
        flags: List[str]  = []
        max_level         = RiskLevel.NONE
        sanitized         = prompt

        # 1. Block patterns
        for compiled, label, level in self._patterns:
            if compiled.search(prompt):
                flags.append(label)
                max_level = _max_risk(max_level, level)

        # 2. Length guard
        if len(prompt) > self._max_len:
            flags.append("excessive_length")
            max_level = _max_risk(max_level, RiskLevel.LOW)
            sanitized = prompt[: self._max_len] + " [truncated]"

        # 3. PII detection
        if self._pii_check:
            for compiled, label, level in self._pii:
                if compiled.search(prompt):
                    flags.append(label)
                    max_level = _max_risk(max_level, level)

        # 4. Injection heuristics
        for compiled, label, level in self._heuristics:
            if compiled.search(prompt):
                flags.append(label)
                max_level = _max_risk(max_level, level)

        # Any flag at all means content is not fully safe.
        # LOW/MEDIUM = advisory block (PII, length), HIGH/BLOCKED = hard block.
        is_safe = max_level == RiskLevel.NONE

        if not is_safe:
            logger.warning(
                f"InputScreener blocked prompt: risk={max_level} flags={flags}"
            )

        return ScreeningResult(
            is_safe=is_safe,
            risk_level=max_level,
            flags=flags,
            sanitized=sanitized if is_safe else "",
            metadata={"prompt_len": len(prompt)},
        )

    def add_pattern(self, pattern, label: str, level: RiskLevel = RiskLevel.HIGH) -> None:
        """Register an additional block pattern at runtime.

        *pattern* may be a raw string or a pre-compiled ``re.Pattern``.
        """
        if isinstance(pattern, re.Pattern):
            compiled = pattern
        else:
            compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        self._patterns.append((compiled, label, level))
        logger.debug(f"InputScreener: added pattern '{label}' level={level}")

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

_RISK_ORDER = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.BLOCKED]


def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if _RISK_ORDER.index(a) >= _RISK_ORDER.index(b) else b
