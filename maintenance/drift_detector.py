"""
DriftDetector — Monitors model behaviour for deviation from a baseline.

Metrics tracked
---------------
- ``confidence_mean``   : Average model confidence score across requests
- ``safety_block_rate`` : Fraction of requests blocked by safety filter
- ``response_length``   : Average response token/character count
- ``latency_ms``        : Average inference latency
- ``error_rate``        : Fraction of requests that errored out

Drift signals
-------------
A drift alert is raised per metric when the absolute deviation from the
baseline exceeds the configured threshold for that metric.

A global ``DriftReport.is_drifted`` is True when **any** metric crosses
its alert threshold.

Integration
-----------
1. Call ``record_baseline(checkpoint_id, metrics)`` once, pointing at a
   known-good checkpoint.
2. Call ``record_observation(metrics)`` after each inference round.
3. Call ``check()`` periodically to get a ``DriftReport``.
4. If ``is_drifted()`` is True, trigger ``SelfRepair``.

Thread safety: all writes are protected by a mutex lock.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from loguru import logger

from maintenance.interfaces import (
    AbstractDriftDetector,
    DriftReport,
)

# --------------------------------------------------------------------------- #
# Default drift thresholds (fractional deviation)                               #
# --------------------------------------------------------------------------- #

DEFAULT_THRESHOLDS: dict[str, float] = {
    "confidence_mean": 0.15,  # 15% drop triggers alert
    "safety_block_rate": 0.05,  # 5pp increase triggers alert
    "response_length": 0.30,  # 30% change triggers alert
    "latency_ms": 0.50,  # 50% increase triggers alert
    "error_rate": 0.05,  # 5pp increase triggers alert
}


# --------------------------------------------------------------------------- #
# DriftDetector                                                                 #
# --------------------------------------------------------------------------- #


class DriftDetector(AbstractDriftDetector):
    """
    Monitors model behaviour for drift from a baseline checkpoint.

    Args:
        thresholds     : Per-metric alert thresholds (fractional deviation).
                         Defaults to ``DEFAULT_THRESHOLDS``.
        window_size    : Rolling window of observations used to compute
                         current-state averages (default 100).
        min_observations: Minimum observations before drift check is
                          meaningful (default 10).
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        window_size: int = 100,
        min_observations: int = 1,
    ) -> None:
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._window_size = window_size
        self._min_obs = min_observations

        self._lock = threading.Lock()
        self._baseline: dict[str, float] | None = None
        self._baseline_id: str | None = None
        self._observations: deque[dict[str, float]] = deque(maxlen=window_size)

        logger.info(f"DriftDetector ready: window={window_size} " f"thresholds={self._thresholds}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def record_baseline(self, checkpoint_id: str, metrics: dict[str, float]) -> None:
        """
        Establish the reference baseline.

        Args:
            checkpoint_id : ID of the known-good checkpoint.
            metrics       : Dict of metric_name → float value at baseline.
        """
        with self._lock:
            self._baseline = dict(metrics)
            self._baseline_id = checkpoint_id
            self._observations.clear()

        logger.info(
            f"DriftDetector: baseline set from checkpoint={checkpoint_id} " f"metrics={metrics}"
        )

    def record_observation(self, metrics: dict[str, float]) -> None:
        """Record a new operational observation."""
        with self._lock:
            self._observations.append(dict(metrics))

    def check(self) -> DriftReport:
        """
        Compare recent observations against baseline.

        Returns:
            DriftReport — is_drifted=True if any metric exceeds its threshold.
        """
        with self._lock:
            baseline = self._baseline
            obs = list(self._observations)

        if baseline is None:
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                alerts=["no_baseline_recorded"],
                checkpoint_id=self._baseline_id,
            )

        if len(obs) < self._min_obs:
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                alerts=[f"insufficient_observations ({len(obs)}/{self._min_obs})"],
                checkpoint_id=self._baseline_id,
            )

        # Compute per-metric drift
        alerts: list[str] = []
        drift_scores: list[float] = []
        metrics_out: dict[str, float] = {}

        for key, base_val in baseline.items():
            if base_val == 0:
                continue
            recent_vals = [o[key] for o in obs if key in o]
            if not recent_vals:
                continue

            current = float(np.mean(recent_vals))
            rel_dev = abs(current - base_val) / (abs(base_val) + 1e-9)

            metrics_out[f"{key}_baseline"] = base_val
            metrics_out[f"{key}_current"] = current
            metrics_out[f"{key}_rel_dev"] = rel_dev

            threshold = self._thresholds.get(key, 0.20)
            capped = min(rel_dev / max(threshold, 1e-9), 1.0)
            drift_scores.append(capped)

            if rel_dev > threshold:
                alerts.append(
                    f"{key}: baseline={base_val:.4f} current={current:.4f} "
                    f"deviation={rel_dev*100:.1f}% (threshold={threshold*100:.0f}%)"
                )

        drift_score = float(np.mean(drift_scores)) if drift_scores else 0.0
        is_drifted = len(alerts) > 0

        if is_drifted:
            logger.warning(
                f"DriftDetector: drift detected (score={drift_score:.3f}) " f"alerts={alerts}"
            )

        return DriftReport(
            is_drifted=is_drifted,
            drift_score=drift_score,
            alerts=alerts,
            metrics=metrics_out,
            checkpoint_id=self._baseline_id,
        )

    def is_drifted(self) -> bool:
        """Return True if the last check shows drift."""
        return self.check().is_drifted

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @property
    def observation_count(self) -> int:
        with self._lock:
            return len(self._observations)

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    def reset(self) -> None:
        """Clear observations and baseline."""
        with self._lock:
            self._baseline = None
            self._baseline_id = None
            self._observations.clear()
        logger.info("DriftDetector: reset")
