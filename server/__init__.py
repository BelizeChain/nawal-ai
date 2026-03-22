"""
Federated Learning Server for Nawal AI

Coordinates federated training across BelizeChain validators:
- Model aggregation (FedAvg, Byzantine-robust)
- Participant management
- Metrics tracking
- Integration with blockchain staking

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

from server.aggregator import (
    AggregationRound,
    AggregationStrategy,
    ByzantineRobustStrategy,
    FedAvgStrategy,
    FederatedAggregator,
    ModelUpdate,
)
from server.metrics_tracker import (
    AggregatedMetrics,
    MetricsTracker,
    TrainingMetrics,
)
from server.participant_manager import (
    Participant,
    ParticipantManager,
    ParticipantStatus,
)

__version__ = "1.0.0"

__all__ = [
    "AggregatedMetrics",
    "AggregationRound",
    "AggregationStrategy",
    "ByzantineRobustStrategy",
    "FedAvgStrategy",
    # Aggregation
    "FederatedAggregator",
    # Metrics
    "MetricsTracker",
    "ModelUpdate",
    "Participant",
    # Participants
    "ParticipantManager",
    "ParticipantStatus",
    "TrainingMetrics",
]
