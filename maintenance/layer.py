"""
MaintenanceLayer — Immune System / Homeostasis facade.

Single entry-point for the maintenance sub-system.  Combines:

    InputScreener   → screen incoming prompts
    OutputFilter    → filter generated outputs
    DriftDetector   → detect model drift from baseline
    SelfRepair      → auto-rollback / alert on detected drift
    QuantumAnomalyDetector (optional) → quantum-enhanced telemetry anomaly

Typical usage in api_server.py or orchestrator.py::

    from nawal.maintenance import MaintenanceLayer

    ml = MaintenanceLayer(checkpoint_path="checkpoints/")
    ml.drift_detector.record_baseline("final_checkpoint", baseline_metrics)

    # At inference time:
    screen_result = ml.screen_input(prompt)
    if not screen_result.is_safe:
        return {"error": "Request blocked"}

    response = model.generate(prompt)
    filter_result = ml.filter_output(prompt, response)
    if not filter_result.is_safe:
        return {"error": "Response blocked"}

    # After each round:
    ml.record_metrics({
        "confidence_mean": conf,
        "safety_block_rate": block_rate,
    })
    ml.check_and_repair()   # auto-repairs if drifted
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from maintenance.interfaces import (
    DriftReport,
    FilterResult,
    RepairResult,
    RepairStrategy,
    RiskLevel,
    ScreeningResult,
)
from maintenance.input_screener import InputScreener
from maintenance.output_filter import OutputFilter
from maintenance.drift_detector import DriftDetector
from maintenance.self_repair import SelfRepair


# --------------------------------------------------------------------------- #
# MaintenanceLayer                                                              #
# --------------------------------------------------------------------------- #

class MaintenanceLayer:
    """
    Unified maintenance facade (Immune System / Homeostasis).

    Args:
        checkpoint_path          : Directory of checkpoint files for rollback.
        episodic_memory          : Optional EpisodicMemory for repair logs.
        alert_callback           : Optional callable(DriftReport) for alerting.
        auto_repair              : Auto-repair on drift (default True).
        quantum_anomaly_detector : Optional QuantumAnomalyDetector for
                                   telemetry anomaly detection.
        drift_thresholds         : Per-metric drift thresholds.
        drift_window             : Rolling window size for drift detector.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        episodic_memory: Optional[Any] = None,
        alert_callback: Optional[Callable[[DriftReport], None]] = None,
        auto_repair: bool = True,
        quantum_anomaly_detector: Optional[Any] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        drift_window: int = 100,
    ) -> None:
        self.input_screener = InputScreener()
        self.output_filter  = OutputFilter()
        self.drift_detector = DriftDetector(
            thresholds=drift_thresholds,
            window_size=drift_window,
        )
        self.self_repair = SelfRepair(
            checkpoint_path=checkpoint_path,
            episodic_memory=episodic_memory,
            alert_callback=alert_callback,
            auto_repair=auto_repair,
        )
        self._quantum_anomaly = quantum_anomaly_detector

        # Telemetry for anomaly detector (lazily fitted)
        self._telemetry_buffer: List[np.ndarray] = []
        self._anomaly_fitted   = False

        logger.info(
            f"MaintenanceLayer ready: checkpoint_path={checkpoint_path} "
            f"auto_repair={auto_repair} "
            f"quantum_anomaly={'yes' if quantum_anomaly_detector else 'no'}"
        )

    # ------------------------------------------------------------------ #
    # Primary pipeline methods                                             #
    # ------------------------------------------------------------------ #

    def screen_input(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScreeningResult:
        """Screen an incoming prompt.  Returns is_safe=False to block."""
        return self.input_screener.screen(prompt, context)

    def filter_output(self, prompt: str, response: str) -> FilterResult:
        """Screen a generated response.  Returns is_safe=False to suppress."""
        return self.output_filter.filter(prompt, response)

    # ------------------------------------------------------------------ #
    # Drift monitoring                                                     #
    # ------------------------------------------------------------------ #

    def record_baseline(self, checkpoint_id: str, metrics: Dict[str, float]) -> None:
        """Delegate to DriftDetector.record_baseline."""
        self.drift_detector.record_baseline(checkpoint_id, metrics)

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record an operational observation to the DriftDetector."""
        self.drift_detector.record_observation(metrics)

    def check_and_repair(self) -> Optional[RepairResult]:
        """
        Check for drift and trigger SelfRepair if drifted.

        Returns:
            RepairResult if repair was triggered, else None.
        """
        report = self.drift_detector.check()
        if report.is_drifted:
            repair_count = self.self_repair.repair_count
            logger.warning(
                f"MaintenanceLayer: drift detected — triggering repair "
                f"(attempt #{repair_count + 1})"
            )
            result = self.self_repair.repair(
                strategy=RepairStrategy.ROLLBACK,
                drift_report=report,
            )
            if not result.success:
                logger.error(
                    "MaintenanceLayer: repair attempt failed — "
                    "operator intervention may be required"
                )
            return result
        return None

    # ------------------------------------------------------------------ #
    # Quantum anomaly detection                                            #
    # ------------------------------------------------------------------ #

    def record_telemetry(self, telemetry_vector: np.ndarray) -> None:
        """
        Feed a telemetry vector to the quantum anomaly detector buffer.
        The detector is auto-fitted once ≥ 50 vectors have been collected.
        """
        if self._quantum_anomaly is None:
            return
        self._telemetry_buffer.append(np.asarray(telemetry_vector, dtype=np.float64))
        if not self._anomaly_fitted and len(self._telemetry_buffer) >= 50:
            X = np.stack(self._telemetry_buffer)
            self._quantum_anomaly.fit(X)
            self._anomaly_fitted = True
            logger.info("MaintenanceLayer: QuantumAnomalyDetector auto-fitted")

    def check_telemetry_anomaly(self, telemetry_vector: np.ndarray) -> bool:
        """
        Return True if the telemetry vector is anomalous (quantum or classical).
        Returns False (not anomalous) if detector is not yet fitted.
        """
        if self._quantum_anomaly is None or not self._anomaly_fitted:
            return False
        predictions = self._quantum_anomaly.predict(telemetry_vector)
        return bool(predictions[0]) if predictions else False

    # ------------------------------------------------------------------ #
    # Convenience introspection                                            #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary dict for monitoring dashboards."""
        drift_report = self.drift_detector.check()
        return {
            "drift_score":        drift_report.drift_score,
            "is_drifted":         drift_report.is_drifted,
            "drift_alerts":       drift_report.alerts,
            "repair_count":       self.self_repair.repair_count,
            "observation_count":  self.drift_detector.observation_count,
            "anomaly_detector":   "fitted" if self._anomaly_fitted else "pending",
            "input_patterns":     self.input_screener.pattern_count,
        }
