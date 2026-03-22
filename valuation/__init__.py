"""
Valuation Module — Nawal Brain Architecture (Limbic System / Basal Ganglia)

Sub-systems:
    reward_model   — RLHF-style preference scoring
    safety_filter  — adversarial / policy enforcement layer
    drives         — internal signals: curiosity, consistency, novelty

Phase 0: skeleton + interfaces only.
Phase 2: classical reward model + safety filter wired to existing
         security/ and training/ modules.
Phase 5: Quantum multi-objective value evaluator.

Canonical import:
    from nawal.valuation.interfaces import AbstractRewardModel, SafetyFilter
"""

from valuation.interfaces import AbstractRewardModel, DriveSignal, SafetyFilter
from valuation.reward import DriveBasedRewardModel
from valuation.safety import BasicSafetyFilter, ValuationLayer

__all__ = [
    # Interfaces
    "AbstractRewardModel",
    "BasicSafetyFilter",
    # Implementations (Phase 2)
    "DriveBasedRewardModel",
    "DriveSignal",
    "SafetyFilter",
    "ValuationLayer",
]
