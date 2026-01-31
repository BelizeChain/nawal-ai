"""
Federated Aggregation Server

Implements federated learning aggregation strategies:
- FedAvg (Federated Averaging)
- Weighted aggregation based on fitness scores
- Byzantine-robust aggregation
- Genome-aware aggregation

Integrates with Nawal's genome evolution system.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from loguru import logger
import torch
import torch.nn as nn

from ..genome import Genome
from ..storage.metrics_db import MetricsStore


# =============================================================================
# Model Update Types
# =============================================================================


@dataclass
class ModelUpdate:
    """
    Model update from a participant.
    
    Contains:
    - Participant information
    - Updated model weights
    - Training metrics
    - Fitness scores
    """
    
    participant_id: str
    genome_id: str
    round_number: int
    
    # Model weights (state dict)
    weights: dict[str, torch.Tensor]
    
    # Training info
    samples_trained: int
    training_time: float  # seconds
    
    # Fitness components
    quality_score: float | None = None
    timeliness_score: float | None = None
    honesty_score: float | None = None
    fitness_score: float | None = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def calculate_weight(self, strategy: str = "samples") -> float:
        """
        Calculate aggregation weight for this update.
        
        Args:
            strategy: Weighting strategy ("samples", "fitness", "hybrid")
        
        Returns:
            Weight value (0.0-1.0)
        """
        match strategy:
            case "samples":
                # Weight by number of samples
                return float(self.samples_trained)
            
            case "fitness":
                # Weight by fitness score
                if self.fitness_score is not None:
                    return max(0.0, self.fitness_score / 100.0)
                return 0.0
            
            case "hybrid":
                # Combine samples and fitness
                sample_weight = float(self.samples_trained)
                fitness_weight = (self.fitness_score or 0.0) / 100.0
                return (sample_weight * 0.7 + fitness_weight * 0.3)
            
            case _:
                return 1.0


# =============================================================================
# Aggregation Strategy Protocol
# =============================================================================


class AggregationStrategy(Protocol):
    """
    Protocol for aggregation strategies.
    
    Implementations must provide aggregate() method.
    """
    
    async def aggregate(
        self,
        updates: list[ModelUpdate],
        current_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate model updates into new global model.
        
        Args:
            updates: List of model updates from participants
            current_weights: Current global model weights
        
        Returns:
            Aggregated model weights
        """
        ...


# =============================================================================
# FedAvg Strategy
# =============================================================================


class FedAvgStrategy:
    """
    Federated Averaging (FedAvg) aggregation strategy.
    
    The classic federated learning algorithm:
    1. Weight updates by number of samples
    2. Compute weighted average
    3. Update global model
    
    Reference: McMahan et al., "Communication-Efficient Learning of
    Deep Networks from Decentralized Data" (2017)
    """
    
    def __init__(self, weighting: str = "samples"):
        """
        Initialize FedAvg strategy.
        
        Args:
            weighting: Weighting strategy ("samples", "fitness", "hybrid")
        """
        self.weighting = weighting
        logger.info(f"Initialized FedAvgStrategy with weighting={weighting}")
    
    async def aggregate(
        self,
        updates: list[ModelUpdate],
        current_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate updates using weighted averaging.
        
        Args:
            updates: Model updates from participants
            current_weights: Current global model weights
        
        Returns:
            Aggregated model weights
        """
        if not updates:
            logger.warning("No updates to aggregate, returning current weights")
            return current_weights
        
        # Calculate weights for each update
        weights = [update.calculate_weight(self.weighting) for update in updates]
        total_weight = sum(weights)
        
        if total_weight == 0:
            logger.warning("Total weight is 0, returning current weights")
            return current_weights
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]
        
        # Aggregate weights
        aggregated = {}
        
        for key in current_weights.keys():
            # Skip if any update is missing this key
            if not all(key in update.weights for update in updates):
                logger.warning(f"Key {key} missing in some updates, using current weight")
                aggregated[key] = current_weights[key]
                continue
            
            # Weighted average
            weighted_sum = sum(
                update.weights[key] * weight
                for update, weight in zip(updates, normalized_weights)
            )
            
            aggregated[key] = weighted_sum
        
        logger.info(
            "FedAvg aggregation complete",
            num_updates=len(updates),
            total_samples=sum(u.samples_trained for u in updates),
            avg_fitness=sum(u.fitness_score or 0 for u in updates) / len(updates),
        )
        
        return aggregated


# =============================================================================
# Byzantine-Robust Strategy
# =============================================================================


class ByzantineRobustStrategy:
    """
    Byzantine-robust aggregation using coordinate-wise median.
    
    Protects against malicious participants by using median
    instead of mean for each model parameter.
    
    More robust but slower than FedAvg.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize Byzantine-robust strategy.
        
        Args:
            trim_ratio: Fraction of extreme values to trim (0.0-0.5)
        """
        self.trim_ratio = trim_ratio
        logger.info(f"Initialized ByzantineRobustStrategy with trim_ratio={trim_ratio}")
    
    async def aggregate(
        self,
        updates: list[ModelUpdate],
        current_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate using trimmed median.
        
        Args:
            updates: Model updates from participants
            current_weights: Current global model weights
        
        Returns:
            Aggregated model weights (median)
        """
        if not updates:
            return current_weights
        
        if len(updates) < 3:
            # Fall back to FedAvg for small number of updates
            logger.warning("Too few updates for Byzantine-robust aggregation, using FedAvg")
            fedavg = FedAvgStrategy()
            return await fedavg.aggregate(updates, current_weights)
        
        aggregated = {}
        num_trim = int(len(updates) * self.trim_ratio)
        
        for key in current_weights.keys():
            if not all(key in update.weights for update in updates):
                aggregated[key] = current_weights[key]
                continue
            
            # Stack all values for this parameter
            values = torch.stack([update.weights[key] for update in updates])
            
            # Compute trimmed median along participant dimension
            if num_trim > 0:
                values_sorted, _ = torch.sort(values, dim=0)
                values_trimmed = values_sorted[num_trim:-num_trim]
                aggregated[key] = torch.median(values_trimmed, dim=0)[0]
            else:
                aggregated[key] = torch.median(values, dim=0)[0]
        
        logger.info(
            "Byzantine-robust aggregation complete",
            num_updates=len(updates),
            trim_ratio=self.trim_ratio,
        )
        
        return aggregated


# =============================================================================
# Federated Aggregator
# =============================================================================


@dataclass
class AggregationRound:
    """Record of an aggregation round."""
    
    round_number: int
    genome_id: str
    num_participants: int
    total_samples: int
    avg_fitness: float
    strategy_used: str
    timestamp: str
    aggregation_time: float  # seconds


class FederatedAggregator:
    """
    Main federated learning aggregator.
    
    Coordinates:
    - Receiving model updates from participants
    - Aggregating updates using chosen strategy
    - Distributing updated global model
    - Tracking aggregation history
    """
    
    def __init__(
        self,
        strategy: AggregationStrategy | None = None,
        min_participants: int = 3,
        max_wait_time: float = 300.0,  # 5 minutes
        config: "FederatedConfig | None" = None,  # Backward compatibility
        metrics_store: MetricsStore | None = None,  # Metrics persistence
    ):
        """
        Initialize federated aggregator.
        
        Args:
            strategy: Aggregation strategy (FedAvg if None)
            min_participants: Minimum participants before aggregation
            max_wait_time: Maximum wait time for updates (seconds)
            config: FederatedConfig for backward compatibility (takes precedence)
            metrics_store: MetricsStore for persisting training metrics
        """
        # Support backward compatibility with config parameter
        if config is not None:
            self.config = config
            self.min_participants = config.min_participants
            self.max_wait_time = getattr(config, 'max_wait_time', max_wait_time)
            # Create strategy based on config
            if config.aggregation_strategy == "fedavg":
                self.strategy = strategy or FedAvgStrategy()
            else:
                self.strategy = strategy or FedAvgStrategy()  # Default to FedAvg
        else:
            self.config = None
            self.strategy = strategy or FedAvgStrategy()
            self.min_participants = min_participants
            self.max_wait_time = max_wait_time
        
        # Metrics persistence
        self.metrics_store = metrics_store
        
        # Current state
        self.current_genome: Genome | None = None
        self.current_weights: dict[str, torch.Tensor] = {}
        self.round_number: int = 0
        
        # Pending updates
        self.pending_updates: dict[int, list[ModelUpdate]] = defaultdict(list)
        
        # History
        self.aggregation_history: list[AggregationRound] = []
        
        logger.info(
            "Initialized FederatedAggregator",
            strategy=type(self.strategy).__name__,
            min_participants=self.min_participants,
            max_wait_time=self.max_wait_time,
            metrics_enabled=metrics_store is not None,
        )
    
    def set_genome(self, genome: Genome, initial_weights: dict[str, torch.Tensor]) -> None:
        """
        Set current genome and initial weights.
        
        Args:
            genome: Genome to train
            initial_weights: Initial model weights
        """
        self.current_genome = genome
        self.current_weights = initial_weights
        self.round_number = 0
        self.pending_updates.clear()
        
        logger.info(
            "Genome set for federated training",
            genome_id=genome.genome_id,
            generation=genome.generation,
        )
    
    def fedavg_aggregate(self, client_params: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Synchronous FedAvg aggregation for backward compatibility.
        
        Args:
            client_params: List of client state dicts (model parameters)
        
        Returns:
            Aggregated global parameters
        """
        if not client_params:
            raise ValueError("Cannot aggregate empty client parameters list")
        
        # Simple FedAvg: average all parameters
        aggregated = {}
        num_clients = len(client_params)
        
        # Get all parameter keys from first client
        for key in client_params[0].keys():
            # Stack all client parameters for this key
            stacked = torch.stack([params[key] for params in client_params])
            # Average across clients
            aggregated[key] = stacked.mean(dim=0)
        
        return aggregated
    
    def weighted_aggregate(
        self, 
        client_params: list[dict[str, torch.Tensor]], 
        weights: list[float]
    ) -> dict[str, torch.Tensor]:
        """
        Weighted aggregation for backward compatibility.
        
        Args:
            client_params: List of client state dicts
            weights: Weight for each client (should sum to 1.0)
        
        Returns:
            Weighted aggregated parameters
        """
        if not client_params:
            raise ValueError("Cannot aggregate empty client parameters list")
        
        if len(client_params) != len(weights):
            raise ValueError("Number of clients must match number of weights")
        
        # Weighted aggregation
        aggregated = {}
        
        for key in client_params[0].keys():
            # Weighted sum of all client parameters
            weighted_sum = sum(
                params[key] * weight 
                for params, weight in zip(client_params, weights)
            )
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def select_clients(self, total_clients: int, num_to_select: int | None = None) -> list[int]:
        """
        Select clients for federated round (backward compatibility).
        
        Args:
            total_clients: Total number of available clients
            num_to_select: Number of clients to select (optional)
        
        Returns:
            List of selected client indices
        """
        import random
        
        # Use minimum participants if not specified
        n: int = num_to_select if num_to_select is not None else self.min_participants
        
        # Enforce minimum if config has it
        min_clients: int = (
            getattr(self.config, 'min_clients', self.min_participants) 
            if self.config else self.min_participants
        )
        n = max(n, min_clients)
        
        # Don't select more than available
        final_num = min(n, total_clients)
        
        # Random selection without replacement
        return random.sample(range(total_clients), final_num)
    
    def select_clients_by_fraction(self, total_clients: int) -> list[int]:
        """
        Select clients by fraction (backward compatibility).
        
        Args:
            total_clients: Total number of available clients
        
        Returns:
            List of selected client indices
        """
        # Get client fraction from config
        client_fraction = getattr(self.config, 'client_fraction', 1.0) if self.config else 1.0
        num_to_select = max(1, int(total_clients * client_fraction))
        
        return self.select_clients(total_clients, num_to_select)
    
    async def submit_update(self, update: ModelUpdate) -> bool:
        """
        Submit model update from participant.
        
        Args:
            update: Model update from participant
        
        Returns:
            True if accepted, False otherwise
        """
        # Validate update
        if self.current_genome is None:
            logger.warning("No genome set, rejecting update")
            return False
        
        if update.genome_id != self.current_genome.genome_id:
            logger.warning(
                "Update genome mismatch",
                expected=self.current_genome.genome_id,
                received=update.genome_id,
            )
            return False
        
        # Add to pending updates
        self.pending_updates[update.round_number].append(update)
        
        logger.info(
            "Update submitted",
            participant=update.participant_id,
            round=update.round_number,
            samples=update.samples_trained,
            fitness=update.fitness_score,
        )
        
        # Check if we can aggregate
        if len(self.pending_updates[update.round_number]) >= self.min_participants:
            # Trigger aggregation
            asyncio.create_task(self._aggregate_round(update.round_number))
        
        return True
    
    async def _aggregate_round(self, round_number: int) -> None:
        """
        Aggregate updates for a specific round.
        
        Args:
            round_number: Round number to aggregate
        """
        updates = self.pending_updates.get(round_number, [])
        
        if len(updates) < self.min_participants:
            logger.warning(
                f"Not enough updates for round {round_number}",
                received=len(updates),
                required=self.min_participants,
            )
            return
        
        logger.info(f"Starting aggregation for round {round_number}")
        start_time = asyncio.get_event_loop().time()
        
        # Perform aggregation
        try:
            aggregated_weights = await self.strategy.aggregate(
                updates,
                self.current_weights,
            )
            
            # Update global model
            self.current_weights = aggregated_weights
            self.round_number = round_number + 1
            
            # Record aggregation
            total_samples = sum(u.samples_trained for u in updates)
            avg_fitness = sum(u.fitness_score or 0 for u in updates) / len(updates)
            
            aggregation_time = asyncio.get_event_loop().time() - start_time
            
            record = AggregationRound(
                round_number=round_number,
                genome_id=self.current_genome.genome_id if self.current_genome else "unknown",
                num_participants=len(updates),
                total_samples=total_samples,
                avg_fitness=avg_fitness,
                strategy_used=type(self.strategy).__name__,
                timestamp=datetime.now(timezone.utc).isoformat(),
                aggregation_time=aggregation_time,
            )
            
            self.aggregation_history.append(record)
            
            # Persist metrics to database
            if self.metrics_store:
                try:
                    await self.metrics_store.log_training_round(
                        round_id=round_number,
                        metrics={
                            "avg_fitness": avg_fitness,
                            "total_samples": total_samples,
                            "num_participants": len(updates),
                            "aggregation_time": aggregation_time,
                            "strategy": type(self.strategy).__name__,
                        },
                        participating_clients=len(updates),
                        aggregated_weights_hash=str(hash(frozenset(aggregated_weights.keys()))),
                    )
                    logger.debug(f"Metrics persisted for round {round_number}")
                except Exception as e:
                    logger.error(f"Failed to persist metrics: {e}")
            
            # Clear pending updates for this round
            del self.pending_updates[round_number]
            
            logger.info(
                "Aggregation complete",
                round=round_number,
                participants=len(updates),
                samples=total_samples,
                avg_fitness=f"{avg_fitness:.2f}",
                time=f"{aggregation_time:.2f}s",
            )
        
        except Exception as e:
            logger.error(
                f"Aggregation failed for round {round_number}",
                error=str(e),
            )
    
    def get_global_weights(self) -> dict[str, torch.Tensor]:
        """Get current global model weights."""
        return self.current_weights
    
    def get_aggregation_history(self) -> list[AggregationRound]:
        """Get complete aggregation history."""
        return self.aggregation_history
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get aggregation statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.aggregation_history:
            return {
                "total_rounds": 0,
                "total_participants": 0,
                "total_samples": 0,
                "avg_fitness": 0.0,
            }
        
        return {
            "total_rounds": len(self.aggregation_history),
            "total_participants": sum(r.num_participants for r in self.aggregation_history),
            "total_samples": sum(r.total_samples for r in self.aggregation_history),
            "avg_fitness": sum(r.avg_fitness for r in self.aggregation_history) / len(self.aggregation_history),
            "avg_aggregation_time": sum(r.aggregation_time for r in self.aggregation_history) / len(self.aggregation_history),
            "current_round": self.round_number,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ModelUpdate",
    "AggregationStrategy",
    "FedAvgStrategy",
    "ByzantineRobustStrategy",
    "AggregationRound",
    "FederatedAggregator",
]
