"""
Maintenance interfaces — ABCs and result data-classes for the Immune System.

All concrete implementations must satisfy these contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Enumerations                                                                  #
# --------------------------------------------------------------------------- #

class RiskLevel(str, Enum):
    NONE    = "none"
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    BLOCKED = "blocked"


class RepairStrategy(str, Enum):
    ROLLBACK = "rollback"   # Restore last known-good checkpoint
    ALERT    = "alert"      # Notify operators, no auto-change
    ISOLATE  = "isolate"    # Quarantine the offending node/model
    RESET    = "reset"      # Full state reset to factory defaults


# --------------------------------------------------------------------------- #
# Result data-classes                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class ScreeningResult:
    """Result returned by InputScreener.screen()."""
    is_safe:    bool
    risk_level: RiskLevel           = RiskLevel.NONE
    flags:      List[str]           = field(default_factory=list)
    sanitized:  str                 = ""
    metadata:   Dict[str, Any]      = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result returned by OutputFilter.filter()."""
    is_safe:    bool
    risk_level: RiskLevel           = RiskLevel.NONE
    flags:      List[str]           = field(default_factory=list)
    filtered:   str                 = ""   # Possibly redacted output
    metadata:   Dict[str, Any]      = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report returned by DriftDetector.check()."""
    is_drifted:  bool
    drift_score: float              = 0.0  # 0.0 = stable, 1.0 = fully drifted
    alerts:      List[str]          = field(default_factory=list)
    metrics:     Dict[str, float]   = field(default_factory=dict)
    checkpoint_id: Optional[str]    = None


@dataclass
class RepairResult:
    """Result returned by SelfRepair.repair()."""
    success:     bool
    strategy:    RepairStrategy     = RepairStrategy.ALERT
    checkpoint_restored: Optional[str] = None
    message:     str                = ""
    metadata:    Dict[str, Any]     = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Abstract base classes                                                         #
# --------------------------------------------------------------------------- #

class AbstractInputScreener(ABC):
    """Screens incoming prompts before they reach the model."""

    @abstractmethod
    def screen(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ScreeningResult:
        """
        Analyse *prompt* for safety violations.

        Returns:
            ScreeningResult — is_safe=False blocks the request.
        """

    @abstractmethod
    def add_pattern(self, pattern: str, label: str) -> None:
        """Register an additional block pattern."""


class AbstractOutputFilter(ABC):
    """Screens generated outputs before returning them to the user."""

    @abstractmethod
    def filter(self, prompt: str, response: str) -> FilterResult:
        """
        Analyse *response* in the context of *prompt*.

        Returns:
            FilterResult — is_safe=False suppresses the response.
        """

    @abstractmethod
    def is_safe(self, response: str) -> bool:
        """Quick boolean check."""


class AbstractDriftDetector(ABC):
    """Monitors model behaviour over time for deviation from a baseline."""

    @abstractmethod
    def record_baseline(self, checkpoint_id: str, metrics: Dict[str, float]) -> None:
        """Establish a reference point."""

    @abstractmethod
    def record_observation(self, metrics: Dict[str, float]) -> None:
        """Record a new operational observation."""

    @abstractmethod
    def check(self) -> DriftReport:
        """Compare recent observations against baseline."""

    @abstractmethod
    def is_drifted(self) -> bool:
        """Return True if drift exceeds configured threshold."""


class AbstractSelfRepair(ABC):
    """Triggered when DriftDetector.is_drifted() returns True."""

    @abstractmethod
    def repair(
        self,
        strategy: RepairStrategy = RepairStrategy.ROLLBACK,
        drift_report: Optional[DriftReport] = None,
    ) -> RepairResult:
        """Execute the repair strategy."""

    @abstractmethod
    def rollback(self, checkpoint_id: str) -> bool:
        """Restore a specific checkpoint."""
