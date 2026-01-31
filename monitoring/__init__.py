"""
Monitoring and Observability for Nawal AI.

Provides comprehensive metrics collection, Prometheus integration,
and structured logging for production deployments.

Features:
- Training metrics (loss, accuracy, epoch time)
- Fitness score tracking
- Blockchain interaction monitoring
- System resource metrics (CPU, memory, GPU)
- Federated learning round metrics
- Validator performance tracking

Author: BelizeChain Team
License: MIT
"""

from .metrics import MetricsCollector, MetricType, Metric
from .prometheus_exporter import PrometheusExporter
from .logging_config import configure_logging, get_logger

__all__ = [
    "MetricsCollector",
    "MetricType",
    "Metric",
    "PrometheusExporter",
    "configure_logging",
    "get_logger",
]

__version__ = "0.1.0"
