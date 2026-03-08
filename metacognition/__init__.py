"""
Metacognition Module — Nawal Brain Architecture (Default Mode Network)

Sub-systems:
    critic          — self-critique loop (model reviews own output)
    confidence      — uncertainty estimation per response
    simulator       — imagined future rollouts before acting
    identity        — persistent agent persona and decision history
    layer           — unified facade (MetacognitionLayer)

Phase 0: skeleton + interfaces only.
Phase 3: classical critic + confidence estimation + identity + facade.
Phase 5: Quantum Imagination Engine (quantum rollout sampler).

Canonical import:
    from nawal.metacognition import MetacognitionLayer
"""

from metacognition.interfaces import (
    AbstractCritic,
    AbstractSimulator,
    ConfidenceScore,
    CritiqueResult,
)
from metacognition.self_critic import SelfCritic
from metacognition.consistency_checker import ConsistencyChecker, ConsistencyResult
from metacognition.confidence_calibrator import ConfidenceCalibrator
from metacognition.internal_simulator import InternalSimulator
from metacognition.identity_module import IdentityModule, AgentProfile, DecisionRecord
from metacognition.layer import MetacognitionLayer, ReflectionResult

__all__ = [
    # interfaces
    "AbstractCritic",
    "AbstractSimulator",
    "ConfidenceScore",
    "CritiqueResult",
    # components
    "SelfCritic",
    "ConsistencyChecker",
    "ConsistencyResult",
    "ConfidenceCalibrator",
    "InternalSimulator",
    "IdentityModule",
    "AgentProfile",
    "DecisionRecord",
    # facade
    "MetacognitionLayer",
    "ReflectionResult",
]
