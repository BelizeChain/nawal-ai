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
    FederatedAggregator,
    AggregationStrategy,
    FedAvgStrategy,
    ByzantineRobustStrategy,
    ModelUpdate,
    AggregationRound,
)

from server.participant_manager import (
    ParticipantManager,
    Participant,
    ParticipantStatus,
)

from server.metrics_tracker import (
    MetricsTracker,
    TrainingMetrics,
    AggregatedMetrics,
)

__version__ = "1.0.0"

__all__ = [
    # Aggregation
    "FederatedAggregator",
    "AggregationStrategy",
    "FedAvgStrategy",
    "ByzantineRobustStrategy",
    "ModelUpdate",
    "AggregationRound",
    # Participants
    "ParticipantManager",
    "Participant",
    "ParticipantStatus",
    # Metrics
    "MetricsTracker",
    "TrainingMetrics",
    "AggregatedMetrics",
]
