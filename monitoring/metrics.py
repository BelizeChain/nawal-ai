"""
Metrics Collection for Nawal AI.

Collects and aggregates metrics from training, evolution,
federated learning, and blockchain operations.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import psutil
from pathlib import Path

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics collected."""
    
    # Training metrics
    TRAINING_LOSS = "training_loss"
    TRAINING_ACCURACY = "training_accuracy"
    VALIDATION_LOSS = "validation_loss"
    VALIDATION_ACCURACY = "validation_accuracy"
    EPOCH_TIME = "epoch_time"
    BATCH_TIME = "batch_time"
    
    # Evolution metrics
    FITNESS_SCORE = "fitness_score"
    BEST_FITNESS = "best_fitness"
    AVERAGE_FITNESS = "average_fitness"
    GENERATION_TIME = "generation_time"
    
    # Federated learning metrics
    ROUND_TIME = "round_time"
    CLIENT_COUNT = "client_count"
    AGGREGATION_TIME = "aggregation_time"
    COMMUNICATION_COST = "communication_cost"
    
    # Blockchain metrics
    TRANSACTION_TIME = "transaction_time"
    TRANSACTION_SUCCESS = "transaction_success"
    TRANSACTION_FAILURE = "transaction_failure"
    BLOCK_NUMBER = "block_number"
    FINALIZATION_TIME = "finalization_time"
    
    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    GPU_MEMORY = "gpu_memory"
    DISK_USAGE = "disk_usage"
    NETWORK_SENT = "network_sent"
    NETWORK_RECV = "network_recv"


@dataclass
class Metric:
    """Individual metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics from various sources.
    
    Thread-safe metrics collection with buffering and export.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            buffer_size: Maximum number of metrics to buffer
        """
        self.buffer_size = buffer_size
        self.metrics: List[Metric] = []
        self._start_time = time.time()
        self._system_metrics_enabled = True
        
        # Track metric history
        self.history: Dict[str, List[float]] = {}
        
        # Performance tracking
        self._epoch_start: Optional[float] = None
        self._batch_start: Optional[float] = None
        self._round_start: Optional[float] = None
    
    def record(self, metric_type: MetricType, value: float, 
               labels: Optional[Dict[str, str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            labels: Optional labels for metric
            metadata: Optional metadata
        """
        metric = Metric(
            name=metric_type.value,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metadata=metadata or {},
        )
        
        # Add to buffer
        self.metrics.append(metric)
        
        # Add to history
        if metric_type.value not in self.history:
            self.history[metric_type.value] = []
        self.history[metric_type.value].append(value)
        
        # Trim buffer if needed
        if len(self.metrics) > self.buffer_size:
            self.metrics = self.metrics[-self.buffer_size:]
        
        # Trim history
        if len(self.history[metric_type.value]) > 1000:
            self.history[metric_type.value] = self.history[metric_type.value][-1000:]
    
    def record_training_epoch(self, epoch: int, train_loss: float, 
                             train_acc: float, val_loss: float, 
                             val_acc: float) -> None:
        """
        Record training epoch metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        labels = {"epoch": str(epoch)}
        
        self.record(MetricType.TRAINING_LOSS, train_loss, labels)
        self.record(MetricType.TRAINING_ACCURACY, train_acc, labels)
        self.record(MetricType.VALIDATION_LOSS, val_loss, labels)
        self.record(MetricType.VALIDATION_ACCURACY, val_acc, labels)
        
        # Record epoch time if started
        if self._epoch_start is not None:
            epoch_time = time.time() - self._epoch_start
            self.record(MetricType.EPOCH_TIME, epoch_time, labels)
            self._epoch_start = None
    
    def start_epoch(self) -> None:
        """Mark the start of a training epoch."""
        self._epoch_start = time.time()
    
    def start_batch(self) -> None:
        """Mark the start of a training batch."""
        self._batch_start = time.time()
    
    def end_batch(self) -> None:
        """Mark the end of a training batch."""
        if self._batch_start is not None:
            batch_time = time.time() - self._batch_start
            self.record(MetricType.BATCH_TIME, batch_time)
            self._batch_start = None
    
    def record_fitness(self, generation: int, best: float, 
                      average: float, individual: Optional[float] = None) -> None:
        """
        Record evolutionary fitness metrics.
        
        Args:
            generation: Generation number
            best: Best fitness in generation
            average: Average fitness in generation
            individual: Individual fitness (optional)
        """
        labels = {"generation": str(generation)}
        
        self.record(MetricType.BEST_FITNESS, best, labels)
        self.record(MetricType.AVERAGE_FITNESS, average, labels)
        
        if individual is not None:
            self.record(MetricType.FITNESS_SCORE, individual, labels)
    
    def start_federated_round(self) -> None:
        """Mark the start of a federated learning round."""
        self._round_start = time.time()
    
    def record_federated_round(self, round_num: int, num_clients: int, 
                              aggregation_time: float,
                              communication_cost: float) -> None:
        """
        Record federated learning round metrics.
        
        Args:
            round_num: Round number
            num_clients: Number of participating clients
            aggregation_time: Time for aggregation
            communication_cost: Communication cost in bytes
        """
        labels = {"round": str(round_num)}
        
        self.record(MetricType.CLIENT_COUNT, num_clients, labels)
        self.record(MetricType.AGGREGATION_TIME, aggregation_time, labels)
        self.record(MetricType.COMMUNICATION_COST, communication_cost, labels)
        
        # Record round time if started
        if self._round_start is not None:
            round_time = time.time() - self._round_start
            self.record(MetricType.ROUND_TIME, round_time, labels)
            self._round_start = None
    
    def record_blockchain_transaction(self, success: bool, 
                                     tx_time: float,
                                     block_number: Optional[int] = None,
                                     finalization_time: Optional[float] = None) -> None:
        """
        Record blockchain transaction metrics.
        
        Args:
            success: Whether transaction succeeded
            tx_time: Transaction submission time
            block_number: Block number (if available)
            finalization_time: Finalization time (if available)
        """
        if success:
            self.record(MetricType.TRANSACTION_SUCCESS, 1.0)
        else:
            self.record(MetricType.TRANSACTION_FAILURE, 1.0)
        
        self.record(MetricType.TRANSACTION_TIME, tx_time)
        
        if block_number is not None:
            self.record(MetricType.BLOCK_NUMBER, float(block_number))
        
        if finalization_time is not None:
            self.record(MetricType.FINALIZATION_TIME, finalization_time)
    
    def collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        if not self._system_metrics_enabled:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.record(MetricType.CPU_USAGE, cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record(MetricType.MEMORY_USAGE, memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.record(MetricType.DISK_USAGE, disk.percent)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.record(MetricType.NETWORK_SENT, float(net_io.bytes_sent))
        self.record(MetricType.NETWORK_RECV, float(net_io.bytes_recv))
        
        # GPU metrics (if available)
        if GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.record(MetricType.GPU_USAGE, float(util.gpu), 
                              {"device": str(i)})
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    self.record(MetricType.GPU_MEMORY, mem_percent, 
                              {"device": str(i)})
            except Exception:
                pass  # GPU metrics not available
    
    def get_metrics(self, metric_type: Optional[MetricType] = None,
                   since: Optional[datetime] = None) -> List[Metric]:
        """
        Get collected metrics.
        
        Args:
            metric_type: Filter by metric type
            since: Filter by timestamp
        
        Returns:
            List of metrics
        """
        metrics = self.metrics
        
        if metric_type is not None:
            metrics = [m for m in metrics if m.name == metric_type.value]
        
        if since is not None:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_latest(self, metric_type: MetricType) -> Optional[Metric]:
        """
        Get latest metric value.
        
        Args:
            metric_type: Type of metric
        
        Returns:
            Latest metric or None
        """
        metrics = self.get_metrics(metric_type)
        return metrics[-1] if metrics else None
    
    def get_average(self, metric_type: MetricType, 
                   window: int = 10) -> Optional[float]:
        """
        Get average metric value over window.
        
        Args:
            metric_type: Type of metric
            window: Number of recent values to average
        
        Returns:
            Average value or None
        """
        if metric_type.value not in self.history:
            return None
        
        values = self.history[metric_type.value][-window:]
        return sum(values) / len(values) if values else None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Summary dictionary
        """
        uptime = time.time() - self._start_time
        
        summary = {
            "uptime_seconds": uptime,
            "total_metrics": len(self.metrics),
            "metric_types": len(self.history),
            "buffer_usage": f"{len(self.metrics)}/{self.buffer_size}",
        }
        
        # Add latest values for key metrics
        for metric_type in [MetricType.TRAINING_LOSS, MetricType.FITNESS_SCORE,
                           MetricType.CPU_USAGE, MetricType.MEMORY_USAGE]:
            latest = self.get_latest(metric_type)
            if latest:
                summary[f"latest_{metric_type.value}"] = latest.value
        
        return summary
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.history.clear()
    
    def export_csv(self, filepath: Path) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to CSV file
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['timestamp', 'name', 'value', 'labels', 'metadata'])
            
            # Data
            for metric in self.metrics:
                writer.writerow([
                    metric.timestamp.isoformat(),
                    metric.name,
                    metric.value,
                    str(metric.labels),
                    str(metric.metadata),
                ])
