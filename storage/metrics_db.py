"""
Metrics Database Stub.

Provides MetricsStore interface for aggregator metrics persistence.
Currently a stub — to be replaced with actual database backend.
"""

from typing import Any

from loguru import logger


class MetricsStore:
    """
    Stub metrics store for persisting training metrics.

    Replace with actual database (SQLite, PostgreSQL, etc.) in production.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path
        self._metrics: list[dict[str, Any]] = []
        logger.info("MetricsStore initialized (in-memory stub)", db_path=db_path)

    def record(self, metric_name: str, value: float, **labels: str) -> None:
        """Record a metric data point."""
        self._metrics.append({"name": metric_name, "value": value, **labels})

    def get_metrics(self, metric_name: str | None = None) -> list[dict[str, Any]]:
        """Retrieve stored metrics, optionally filtered by name."""
        if metric_name is None:
            return list(self._metrics)
        return [m for m in self._metrics if m["name"] == metric_name]
