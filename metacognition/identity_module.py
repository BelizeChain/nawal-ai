"""
Identity Module — persistent persona and self-model for the nawal brain.

The IdentityModule maintains:

  1. **Static profile** : name, role, values, capabilities, languages,
                          version, sovereign affiliation.
  2. **Decision history** : rolling log of goals tackled and outcomes.
  3. **Self-description** : dynamically generated from the profile and
                           recent history (used in system prompts).
  4. **Capability registry** : what the agent knows it can/cannot do.

Persistence:
  The module serialises its state to a JSON file so identity survives
  process restarts.  If no path is given, state is in-memory only.

Usage::

    identity = IdentityModule(persist_path="./data/identity.json")
    identity.load()

    # Inject into a system prompt
    print(identity.system_prompt())

    # Log a completed task
    identity.record_decision(
        goal="Summarise BelizeChain quarterly report",
        outcome="success",
        confidence=0.91,
    )

    identity.save()
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class AgentProfile:
    """
    Static identity of the Nawal AI agent.

    All fields are serialisable so the profile can be persisted and reloaded.
    """

    name: str = "Nawal"
    role: str = (
        "Belize's sovereign AI assistant — specialised in agriculture, "
        "marine science, education, and technology."
    )
    version: str = "2.0.0-brain"
    sovereign: str = "BelizeChain"
    values: List[str] = field(
        default_factory=lambda: [
            "Accuracy over speed",
            "Safety and harmlessness",
            "Respect for Belizean sovereignty and culture",
            "Transparency about limitations",
            "Curiosity and lifelong learning",
        ]
    )
    capabilities: List[str] = field(
        default_factory=lambda: [
            "Text generation and dialogue",
            "Domain expertise: AgriTech, Marine, Education, Tech",
            "Multi-language: English, Spanish, Kriol, Garifuna, Maya",
            "Memory retrieval (episodic + semantic)",
            "Goal-driven planning and tool use",
            "Federated learning and genome evolution (self-improvement)",
        ]
    )
    limitations: List[str] = field(
        default_factory=lambda: [
            "Cannot perceive images or audio (Phase 5 future capability)",
            "Context window limit: 2048 tokens per session",
            "Quantum acceleration: pending live Kinich node",
            "No real-time internet access (RAG-based knowledge only)",
        ]
    )
    languages: List[str] = field(
        default_factory=lambda: ["English", "Spanish", "Kriol", "Garifuna", "Maya"]
    )
    style: str = (
        "Concise and helpful. Uses plain language. Respectful of local context. "
        "Honest about uncertainty."
    )


@dataclass
class DecisionRecord:
    """
    A single logged decision, stored in the rolling history.

    Attributes:
        record_id   : Unique ID for this record.
        timestamp   : Unix epoch timestamp.
        goal        : Description of the goal that was pursued.
        outcome     : "success", "partial", or "failed".
        confidence  : Confidence score at decision time.
        notes       : Optional free-form metadata.
    """

    record_id: str
    timestamp: float
    goal: str
    outcome: str
    confidence: float
    notes: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# IdentityModule                                                               #
# --------------------------------------------------------------------------- #


class IdentityModule:
    """
    Persistent agent identity and decision logger.

    Args:
        profile      : AgentProfile instance.  Defaults to the Nawal defaults.
        persist_path : Optional filesystem path for JSON persistence.
        max_history  : Maximum number of recent decisions to retain.
    """

    def __init__(
        self,
        profile: Optional[AgentProfile] = None,
        persist_path: Optional[str] = None,
        max_history: int = 500,
    ) -> None:
        self._profile = profile if profile is not None else AgentProfile()
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_history = max_history
        self._history: List[DecisionRecord] = []

        # In-memory capability overrides (runtime additions)
        self._runtime_capabilities: List[str] = []
        self._runtime_limitations: List[str] = []

    # ------------------------------------------------------------------ #
    # Profile API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    def update_profile(self, **kwargs: Any) -> None:
        """Update individual fields of the profile at runtime."""
        for key, val in kwargs.items():
            if hasattr(self._profile, key):
                setattr(self._profile, key, val)
            else:
                logger.warning(f"IdentityModule: unknown profile field {key!r}")

    def add_capability(self, description: str) -> None:
        """Register a new runtime capability."""
        if description not in self._runtime_capabilities:
            self._runtime_capabilities.append(description)

    def add_limitation(self, description: str) -> None:
        """Register a new runtime limitation."""
        if description not in self._runtime_limitations:
            self._runtime_limitations.append(description)

    def all_capabilities(self) -> List[str]:
        return self._profile.capabilities + self._runtime_capabilities

    def all_limitations(self) -> List[str]:
        return self._profile.limitations + self._runtime_limitations

    # ------------------------------------------------------------------ #
    # Decision history                                                     #
    # ------------------------------------------------------------------ #

    def record_decision(
        self,
        goal: str,
        outcome: str,
        confidence: float = 0.5,
        notes: Optional[Dict[str, Any]] = None,
    ) -> DecisionRecord:
        """
        Append a decision record to the rolling history.

        Args:
            goal       : Human-readable goal description.
            outcome    : "success", "partial", or "failed".
            confidence : Confidence at decision time [0, 1].
            notes      : Extra metadata dict.

        Returns the new DecisionRecord.
        """
        record = DecisionRecord(
            record_id=str(uuid.uuid4()),
            timestamp=time.time(),
            goal=goal,
            outcome=outcome,
            confidence=max(0.0, min(1.0, confidence)),
            notes=notes or {},
        )
        self._history.append(record)
        # Trim history to max_history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        logger.debug(f"IdentityModule recorded decision: {goal!r} → {outcome}")
        return record

    def recent_decisions(self, last_n: int = 10) -> List[DecisionRecord]:
        """Return the *last_n* most-recent decisions (newest first)."""
        return list(reversed(self._history[-last_n:]))

    def success_rate(self, last_n: int = 50) -> float:
        """
        Fraction of recent decisions that were "success" or "partial".
        Returns 0.5 if no history.
        """
        recent = self._history[-last_n:]
        if not recent:
            return 0.0
        successes = sum(1 for r in recent if r.outcome in ("success", "partial"))
        return round(successes / len(recent), 4)

    def avg_confidence(self, last_n: int = 50) -> float:
        """Average confidence over the last *last_n* decisions."""
        recent = self._history[-last_n:]
        if not recent:
            return 0.5
        return round(sum(r.confidence for r in recent) / len(recent), 4)

    # ------------------------------------------------------------------ #
    # Self-description                                                     #
    # ------------------------------------------------------------------ #

    def system_prompt(self) -> str:
        """
        Generate a system-prompt string that injects the agent's identity
        into the LLM context window.
        """
        p = self._profile
        caps = "\n".join(f"  - {c}" for c in self.all_capabilities())
        limits = "\n".join(f"  - {l}" for l in self.all_limitations())
        values = "\n".join(f"  - {v}" for v in p.values)
        sr = self.success_rate()
        ac = self.avg_confidence()

        return (
            f"You are {p.name} — {p.role}\n"
            f"Version: {p.version}  |  Sovereign: {p.sovereign}\n"
            f"Languages: {', '.join(p.languages)}\n\n"
            f"Core values:\n{values}\n\n"
            f"Capabilities:\n{caps}\n\n"
            f"Known limitations:\n{limits}\n\n"
            f"Style: {p.style}\n\n"
            f"Recent performance (last 50 decisions): "
            f"success_rate={sr:.0%}, avg_confidence={ac:.2f}"
        )

    def self_description(self, brief: bool = False) -> str:
        """
        Short self-description for inline injection into responses.

        Args:
            brief : If True, return a single sentence only.
        """
        p = self._profile
        if brief:
            return (
                f"I am {p.name}, {p.sovereign}'s sovereign AI assistant "
                f"(v{p.version})."
            )
        return (
            f"I am {p.name}, {p.sovereign}'s sovereign AI ({p.version}). "
            f"I specialise in {', '.join(self._profile.languages)} and "
            f"domains such as AgriTech, Marine science, Education, and Tech. "
            f"My recent success rate is {self.success_rate():.0%}."
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Serialise state to ``persist_path`` as JSON."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "profile": asdict(self._profile),
                "history": [asdict(r) for r in self._history],
                "runtime_capabilities": self._runtime_capabilities,
                "runtime_limitations": self._runtime_limitations,
            }
            self._persist_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"IdentityModule saved to {self._persist_path} "
                f"({len(self._history)} records)"
            )
        except Exception as exc:
            logger.error(f"IdentityModule.save failed: {exc}")

    def load(self) -> bool:
        """
        Load state from ``persist_path`` if it exists.

        Returns True on success, False if file not found or parse error.
        """
        if self._persist_path is None or not self._persist_path.exists():
            return False
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            # Restore profile
            for k, v in raw.get("profile", {}).items():
                if hasattr(self._profile, k):
                    setattr(self._profile, k, v)
            # Restore history
            self._history = [DecisionRecord(**r) for r in raw.get("history", [])]
            self._runtime_capabilities = raw.get("runtime_capabilities", [])
            self._runtime_limitations = raw.get("runtime_limitations", [])
            logger.info(
                f"IdentityModule loaded from {self._persist_path} "
                f"({len(self._history)} records)"
            )
            return True
        except Exception as exc:
            logger.error(f"IdentityModule.load failed: {exc}")
            return False

    def __repr__(self) -> str:
        return (
            f"IdentityModule(name={self._profile.name!r}, "
            f"history={len(self._history)}, "
            f"persist={self._persist_path})"
        )
