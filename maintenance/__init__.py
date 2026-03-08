"""
Maintenance Module — Nawal Brain Architecture (Immune System / Homeostasis)

Sub-systems
-----------
InputScreener         — detect prompt injection, jailbreaks, unsafe inputs
OutputFilter          — post-generation safety classifier and PII guard
DriftDetector         — detect model behaviour change from baseline
SelfRepair            — rollback to last safe checkpoint on anomaly
MaintenanceLayer      — unified facade combining all sub-systems

Quantum extension (Phase 5)
---------------------------
QuantumAnomalyDetector from ``nawal.quantum`` can be passed to
``MaintenanceLayer(quantum_anomaly_detector=...)`` for quantum-enhanced
telemetry anomaly detection.

Canonical import:
    from nawal.maintenance import MaintenanceLayer
    from nawal.maintenance import InputScreener, OutputFilter
    from nawal.maintenance import DriftDetector, SelfRepair
    from nawal.maintenance.interfaces import RiskLevel, ScreeningResult
"""

from maintenance.interfaces import (
    RiskLevel,
    RepairStrategy,
    ScreeningResult,
    FilterResult,
    DriftReport,
    RepairResult,
)
from maintenance.input_screener import InputScreener
from maintenance.output_filter import OutputFilter
from maintenance.drift_detector import DriftDetector
from maintenance.self_repair import SelfRepair
from maintenance.layer import MaintenanceLayer

__all__ = [
    # Data classes / enums
    "RiskLevel",
    "RepairStrategy",
    "ScreeningResult",
    "FilterResult",
    "DriftReport",
    "RepairResult",
    # Components
    "InputScreener",
    "OutputFilter",
    "DriftDetector",
    "SelfRepair",
    # Facade
    "MaintenanceLayer",
]
