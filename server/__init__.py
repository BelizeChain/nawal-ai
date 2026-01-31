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

from nawal.server.aggregator import (
    FederatedAggregator,
    AggregationStrategy,
    FedAvgStrategy,
    ByzantineRobustStrategy,
    ModelUpdate,
    AggregationRound,
)

from nawal.server.participant_manager import (
    ParticipantManager,
    Participant,
    ParticipantStatus,
)

from nawal.server.metrics_tracker import (
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

from .aggregator import (
    FederatedAggregator,
    AggregationStrategy,
    FedAvgStrategy,
)

from .participant_manager import (
    ParticipantManager,
    Participant,
    ParticipantStatus,
)

from .metrics_tracker import (
    MetricsTracker,
    TrainingMetrics,
    AggregatedMetrics,
)

__version__ = "1.0.0"

__all__ = [
    # Aggregator
    "FederatedAggregator",
    "AggregationStrategy",
    "FedAvgStrategy",
    
    # Participant Management
    "ParticipantManager",
    "Participant",
    "ParticipantStatus",
    
    # Metrics
    "MetricsTracker",
    "TrainingMetrics",
    "AggregatedMetrics",
]
