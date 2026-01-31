"""
Training Metrics Tracking for Federated Learning

Tracks and aggregates training metrics:
- Loss, accuracy, throughput
- Training times and resource usage
- Population-wide statistics
- Time-series data for visualization
- Export formats (JSON, Prometheus)

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from loguru import logger
import json


# =============================================================================
# Training Metrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """
    Metrics from a single training session.
    
    Tracks:
    - Loss and accuracy
    - Training performance
    - Resource usage
    """
    
    # Identity
    participant_id: str
    genome_id: str
    round_number: int
    
    # Performance metrics
    train_loss: float
    train_accuracy: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None
    
    # Training stats
    samples_trained: int = 0
    batches_processed: int = 0
    epochs: int = 1
    
    # Timing
    training_time: float = 0.0  # seconds
    throughput: float = 0.0     # samples/sec
    
    # Resource usage
    peak_memory_mb: float | None = None
    avg_gpu_utilization: float | None = None
    
    # Fitness scores
    quality_score: float | None = None
    timeliness_score: float | None = None
    honesty_score: float | None = None
    fitness_score: float | None = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def calculate_throughput(self) -> None:
        """Calculate training throughput."""
        if self.training_time > 0:
            self.throughput = self.samples_trained / self.training_time
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participant_id": self.participant_id,
            "genome_id": self.genome_id,
            "round_number": self.round_number,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "samples_trained": self.samples_trained,
            "batches_processed": self.batches_processed,
            "epochs": self.epochs,
            "training_time": self.training_time,
            "throughput": self.throughput,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "quality_score": self.quality_score,
            "timeliness_score": self.timeliness_score,
            "honesty_score": self.honesty_score,
            "fitness_score": self.fitness_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# Aggregated Metrics
# =============================================================================


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across all participants for a round.
    
    Provides population-wide statistics.
    """
    
    round_number: int
    genome_id: str
    
    # Participant stats
    num_participants: int = 0
    total_samples: int = 0
    
    # Loss metrics
    avg_train_loss: float = 0.0
    min_train_loss: float = float('inf')
    max_train_loss: float = 0.0
    std_train_loss: float = 0.0
    
    # Accuracy metrics
    avg_train_accuracy: float | None = None
    min_train_accuracy: float | None = None
    max_train_accuracy: float | None = None
    
    # Validation metrics
    avg_val_loss: float | None = None
    avg_val_accuracy: float | None = None
    
    # Performance metrics
    avg_training_time: float = 0.0
    total_training_time: float = 0.0
    avg_throughput: float = 0.0
    
    # Fitness metrics
    avg_quality: float = 0.0
    avg_timeliness: float = 0.0
    avg_honesty: float = 0.0
    avg_fitness: float = 0.0
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round_number": self.round_number,
            "genome_id": self.genome_id,
            "num_participants": self.num_participants,
            "total_samples": self.total_samples,
            "avg_train_loss": self.avg_train_loss,
            "min_train_loss": self.min_train_loss if self.min_train_loss != float('inf') else None,
            "max_train_loss": self.max_train_loss if self.max_train_loss != 0.0 else None,
            "std_train_loss": self.std_train_loss,
            "avg_train_accuracy": self.avg_train_accuracy,
            "min_train_accuracy": self.min_train_accuracy,
            "max_train_accuracy": self.max_train_accuracy,
            "avg_val_loss": self.avg_val_loss,
            "avg_val_accuracy": self.avg_val_accuracy,
            "avg_training_time": self.avg_training_time,
            "total_training_time": self.total_training_time,
            "avg_throughput": self.avg_throughput,
            "avg_quality": self.avg_quality,
            "avg_timeliness": self.avg_timeliness,
            "avg_honesty": self.avg_honesty,
            "avg_fitness": self.avg_fitness,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Metrics Tracker
# =============================================================================


class MetricsTracker:
    """
    Tracks and aggregates training metrics.
    
    Features:
    - Real-time metric collection
    - Population-wide aggregation
    - Time-series storage
    - Export to multiple formats
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum metrics to store per round
        """
        self.max_history = max_history
        
        # Per-round metrics
        self.round_metrics: dict[int, list[TrainingMetrics]] = {}
        
        # Aggregated metrics per round
        self.aggregated_metrics: dict[int, AggregatedMetrics] = {}
        
        # Time-series data for visualization
        self.loss_history: list[tuple[int, float]] = []  # (round, avg_loss)
        self.accuracy_history: list[tuple[int, float]] = []  # (round, avg_acc)
        self.fitness_history: list[tuple[int, float]] = []  # (round, avg_fitness)
        
        logger.info("Initialized MetricsTracker", max_history=max_history)
    
    def record_metrics(self, metrics: TrainingMetrics) -> None:
        """
        Record training metrics from a participant.
        
        Args:
            metrics: Training metrics to record
        """
        round_num = metrics.round_number
        
        # Initialize round if needed
        if round_num not in self.round_metrics:
            self.round_metrics[round_num] = []
        
        # Add metrics
        self.round_metrics[round_num].append(metrics)
        
        # Enforce max history
        if len(self.round_metrics[round_num]) > self.max_history:
            self.round_metrics[round_num] = self.round_metrics[round_num][-self.max_history:]
        
        logger.debug(
            "Metrics recorded",
            participant=metrics.participant_id,
            round=round_num,
            loss=f"{metrics.train_loss:.4f}",
            accuracy=f"{metrics.train_accuracy:.2f}%" if metrics.train_accuracy else "N/A",
        )
    
    def aggregate_round_metrics(self, round_number: int) -> AggregatedMetrics:
        """
        Aggregate metrics for a specific round.
        
        Args:
            round_number: Round number to aggregate
        
        Returns:
            Aggregated metrics
        """
        if round_number not in self.round_metrics:
            logger.warning(f"No metrics found for round {round_number}")
            return AggregatedMetrics(round_number=round_number, genome_id="unknown")
        
        metrics_list = self.round_metrics[round_number]
        
        if not metrics_list:
            return AggregatedMetrics(round_number=round_number, genome_id="unknown")
        
        # Basic stats
        num_participants = len(metrics_list)
        total_samples = sum(m.samples_trained for m in metrics_list)
        genome_id = metrics_list[0].genome_id
        
        # Loss stats
        train_losses = [m.train_loss for m in metrics_list]
        avg_train_loss = sum(train_losses) / len(train_losses)
        min_train_loss = min(train_losses)
        max_train_loss = max(train_losses)
        
        # Calculate standard deviation
        mean_loss = avg_train_loss
        variance = sum((x - mean_loss) ** 2 for x in train_losses) / len(train_losses)
        std_train_loss = variance ** 0.5
        
        # Accuracy stats
        train_accuracies = [m.train_accuracy for m in metrics_list if m.train_accuracy is not None]
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else None
        min_train_accuracy = min(train_accuracies) if train_accuracies else None
        max_train_accuracy = max(train_accuracies) if train_accuracies else None
        
        # Validation stats
        val_losses = [m.val_loss for m in metrics_list if m.val_loss is not None]
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None
        
        val_accuracies = [m.val_accuracy for m in metrics_list if m.val_accuracy is not None]
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else None
        
        # Performance stats
        training_times = [m.training_time for m in metrics_list]
        avg_training_time = sum(training_times) / len(training_times)
        total_training_time = sum(training_times)
        
        throughputs = [m.throughput for m in metrics_list if m.throughput > 0]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0
        
        # Fitness stats
        quality_scores = [m.quality_score for m in metrics_list if m.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        timeliness_scores = [m.timeliness_score for m in metrics_list if m.timeliness_score is not None]
        avg_timeliness = sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.0
        
        honesty_scores = [m.honesty_score for m in metrics_list if m.honesty_score is not None]
        avg_honesty = sum(honesty_scores) / len(honesty_scores) if honesty_scores else 0.0
        
        fitness_scores = [m.fitness_score for m in metrics_list if m.fitness_score is not None]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        
        # Create aggregated metrics
        aggregated = AggregatedMetrics(
            round_number=round_number,
            genome_id=genome_id,
            num_participants=num_participants,
            total_samples=total_samples,
            avg_train_loss=avg_train_loss,
            min_train_loss=min_train_loss,
            max_train_loss=max_train_loss,
            std_train_loss=std_train_loss,
            avg_train_accuracy=avg_train_accuracy,
            min_train_accuracy=min_train_accuracy,
            max_train_accuracy=max_train_accuracy,
            avg_val_loss=avg_val_loss,
            avg_val_accuracy=avg_val_accuracy,
            avg_training_time=avg_training_time,
            total_training_time=total_training_time,
            avg_throughput=avg_throughput,
            avg_quality=avg_quality,
            avg_timeliness=avg_timeliness,
            avg_honesty=avg_honesty,
            avg_fitness=avg_fitness,
        )
        
        # Store aggregated metrics
        self.aggregated_metrics[round_number] = aggregated
        
        # Update time-series
        self.loss_history.append((round_number, avg_train_loss))
        if avg_train_accuracy:
            self.accuracy_history.append((round_number, avg_train_accuracy))
        if avg_fitness > 0:
            self.fitness_history.append((round_number, avg_fitness))
        
        logger.info(
            "Round metrics aggregated",
            round=round_number,
            participants=num_participants,
            avg_loss=f"{avg_train_loss:.4f}",
            avg_fitness=f"{avg_fitness:.2f}",
        )
        
        return aggregated
    
    def get_round_metrics(self, round_number: int) -> list[TrainingMetrics]:
        """
        Get individual metrics for a round.
        
        Args:
            round_number: Round number
        
        Returns:
            List of training metrics
        """
        return self.round_metrics.get(round_number, [])
    
    def get_aggregated_metrics(self, round_number: int) -> AggregatedMetrics | None:
        """
        Get aggregated metrics for a round.
        
        Args:
            round_number: Round number
        
        Returns:
            Aggregated metrics if available
        """
        return self.aggregated_metrics.get(round_number)
    
    def get_latest_aggregated(self) -> AggregatedMetrics | None:
        """Get most recent aggregated metrics."""
        if not self.aggregated_metrics:
            return None
        latest_round = max(self.aggregated_metrics.keys())
        return self.aggregated_metrics[latest_round]
    
    def get_loss_history(self, last_n: int | None = None) -> list[tuple[int, float]]:
        """
        Get loss history.
        
        Args:
            last_n: Number of recent entries (None for all)
        
        Returns:
            List of (round_number, avg_loss) tuples
        """
        if last_n is None:
            return self.loss_history
        return self.loss_history[-last_n:]
    
    def get_accuracy_history(self, last_n: int | None = None) -> list[tuple[int, float]]:
        """
        Get accuracy history.
        
        Args:
            last_n: Number of recent entries (None for all)
        
        Returns:
            List of (round_number, avg_accuracy) tuples
        """
        if last_n is None:
            return self.accuracy_history
        return self.accuracy_history[-last_n:]
    
    def get_fitness_history(self, last_n: int | None = None) -> list[tuple[int, float]]:
        """
        Get fitness history.
        
        Args:
            last_n: Number of recent entries (None for all)
        
        Returns:
            List of (round_number, avg_fitness) tuples
        """
        if last_n is None:
            return self.fitness_history
        return self.fitness_history[-last_n:]
    
    def export_to_json(self, filepath: str, round_number: int | None = None) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Output file path
            round_number: Specific round (None for all)
        """
        if round_number is not None:
            # Export specific round
            metrics = self.get_round_metrics(round_number)
            aggregated = self.get_aggregated_metrics(round_number)
            
            data = {
                "round_number": round_number,
                "individual_metrics": [m.to_dict() for m in metrics],
                "aggregated_metrics": aggregated.to_dict() if aggregated else None,
            }
        else:
            # Export all rounds
            data = {
                "rounds": sorted(self.round_metrics.keys()),
                "aggregated_metrics": [
                    agg.to_dict() for agg in self.aggregated_metrics.values()
                ],
                "loss_history": self.loss_history,
                "accuracy_history": self.accuracy_history,
                "fitness_history": self.fitness_history,
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def export_to_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        # Get latest aggregated metrics
        latest = self.get_latest_aggregated()
        if not latest:
            return ""
        
        # Training loss
        lines.append(f"# HELP nawal_train_loss Average training loss")
        lines.append(f"# TYPE nawal_train_loss gauge")
        lines.append(f'nawal_train_loss{{round="{latest.round_number}"}} {latest.avg_train_loss}')
        
        # Training accuracy
        if latest.avg_train_accuracy:
            lines.append(f"# HELP nawal_train_accuracy Average training accuracy")
            lines.append(f"# TYPE nawal_train_accuracy gauge")
            lines.append(f'nawal_train_accuracy{{round="{latest.round_number}"}} {latest.avg_train_accuracy}')
        
        # Fitness
        lines.append(f"# HELP nawal_fitness Average fitness score")
        lines.append(f"# TYPE nawal_fitness gauge")
        lines.append(f'nawal_fitness{{round="{latest.round_number}"}} {latest.avg_fitness}')
        
        # Participants
        lines.append(f"# HELP nawal_participants Number of participants")
        lines.append(f"# TYPE nawal_participants gauge")
        lines.append(f'nawal_participants{{round="{latest.round_number}"}} {latest.num_participants}')
        
        # Samples
        lines.append(f"# HELP nawal_samples Total samples trained")
        lines.append(f"# TYPE nawal_samples counter")
        lines.append(f'nawal_samples{{round="{latest.round_number}"}} {latest.total_samples}')
        
        return "\n".join(lines) + "\n"
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.aggregated_metrics:
            return {
                "total_rounds": 0,
                "total_participants": 0,
                "total_samples": 0,
            }
        
        all_rounds = list(self.aggregated_metrics.values())
        
        return {
            "total_rounds": len(all_rounds),
            "total_participants": sum(r.num_participants for r in all_rounds),
            "total_samples": sum(r.total_samples for r in all_rounds),
            "avg_loss_overall": sum(r.avg_train_loss for r in all_rounds) / len(all_rounds),
            "avg_fitness_overall": sum(r.avg_fitness for r in all_rounds) / len(all_rounds),
            "total_training_time": sum(r.total_training_time for r in all_rounds),
            "latest_round": max(self.aggregated_metrics.keys()),
        }
    
    # =========================================================================
    # Backward Compatibility Methods
    # =========================================================================
    
    def record(self, metric_name: str, value: float, round_num: int) -> None:
        """
        Record a single metric value (backward compatibility).
        
        Args:
            metric_name: Name of the metric (e.g., "loss", "accuracy")
            value: Metric value
            round_num: Round number
        """
        # Store in a simple dict for backward compatibility
        if not hasattr(self, '_simple_metrics'):
            self._simple_metrics: dict[str, dict[int, float]] = {}
        
        if metric_name not in self._simple_metrics:
            self._simple_metrics[metric_name] = {}
        
        self._simple_metrics[metric_name][round_num] = value
    
    def get(self, metric_name: str, round_num: int) -> float | None:
        """
        Get a single metric value (backward compatibility).
        
        Args:
            metric_name: Name of the metric
            round_num: Round number
        
        Returns:
            Metric value or None if not found
        """
        if not hasattr(self, '_simple_metrics'):
            return None
        
        return self._simple_metrics.get(metric_name, {}).get(round_num)
    
    def get_history(self, metric_name: str) -> list[float]:
        """
        Get history of a metric across all rounds (backward compatibility).
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            List of metric values ordered by round number
        """
        if not hasattr(self, '_simple_metrics'):
            return []
        
        metric_dict = self._simple_metrics.get(metric_name, {})
        return [metric_dict[r] for r in sorted(metric_dict.keys())]
    
    def record_client_metric(
        self,
        client_id: int | str,
        metric_name: str,
        value: float,
        round_num: int
    ) -> None:
        """
        Record a client-specific metric (backward compatibility).
        
        Args:
            client_id: Client identifier
            metric_name: Name of the metric
            value: Metric value
            round_num: Round number
        """
        if not hasattr(self, '_client_metrics'):
            self._client_metrics: dict[int, dict[str, dict[int | str, float]]] = {}
        
        if round_num not in self._client_metrics:
            self._client_metrics[round_num] = {}
        
        if metric_name not in self._client_metrics[round_num]:
            self._client_metrics[round_num][metric_name] = {}
        
        self._client_metrics[round_num][metric_name][client_id] = value
    
    def get_client_metrics(self, metric_name: str, round_num: int) -> dict[int | str, float]:
        """
        Get all client metrics for a specific metric and round (backward compatibility).
        
        Args:
            metric_name: Name of the metric
            round_num: Round number
        
        Returns:
            Dictionary mapping client IDs to metric values
        """
        if not hasattr(self, '_client_metrics'):
            return {}
        
        return self._client_metrics.get(round_num, {}).get(metric_name, {})
    
    def aggregate_client_metrics(
        self,
        metric_name: str,
        round_num: int,
        method: str = "mean"
    ) -> float:
        """
        Aggregate client metrics using specified method (backward compatibility).
        
        Args:
            metric_name: Name of the metric
            round_num: Round number
            method: Aggregation method ("mean", "median", "min", "max")
        
        Returns:
            Aggregated value
        """
        client_values = list(self.get_client_metrics(metric_name, round_num).values())
        
        if not client_values:
            return 0.0
        
        if method == "mean":
            return sum(client_values) / len(client_values)
        elif method == "median":
            sorted_values = sorted(client_values)
            mid = len(sorted_values) // 2
            if len(sorted_values) % 2 == 0:
                return (sorted_values[mid - 1] + sorted_values[mid]) / 2
            return sorted_values[mid]
        elif method == "min":
            return min(client_values)
        elif method == "max":
            return max(client_values)
        else:
            return sum(client_values) / len(client_values)
    
    def export(self, filepath: str | Path) -> None:
        """
        Export metrics to JSON file (backward compatibility).
        
        Args:
            filepath: Path to export file
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        data = {
            'simple_metrics': getattr(self, '_simple_metrics', {}),
            'client_metrics': getattr(self, '_client_metrics', {}),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str | Path) -> None:
        """
        Load metrics from JSON file (backward compatibility).
        
        Args:
            filepath: Path to import file
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"Metrics file not found: {filepath}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to integers for round numbers
        simple_metrics = data.get('simple_metrics', {})
        self._simple_metrics = {}
        for metric_name, rounds_dict in simple_metrics.items():
            self._simple_metrics[metric_name] = {
                int(round_num): value 
                for round_num, value in rounds_dict.items()
            }
        
        # Convert client metrics
        client_metrics = data.get('client_metrics', {})
        self._client_metrics = {}
        for round_num, metrics_dict in client_metrics.items():
            self._client_metrics[int(round_num)] = metrics_dict


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TrainingMetrics",
    "AggregatedMetrics",
    "MetricsTracker",
]
