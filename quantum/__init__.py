"""
Quantum Module — Nawal Brain Architecture

Houses all quantum-computing integration components:

Core connectors
---------------
- KinichQuantumConnector    classical→quantum bridge via Kinich HTTP API
- QuantumEnhancedLayer      PyTorch layer wrapping the connector
- HybridQuantumClassicalLLM full hybrid transformer

Phase 5 quantum subsystems
--------------------------
- QuantumMemory          Grover-inspired episodic memory search
- QuantumPlanOptimizer   QAOA-based plan selection from candidate Plans
- QuantumAnomalyDetector Quantum kernel SVM for telemetry anomaly detection
- QuantumImagination     Quantum parallel future-trajectory sampling
- SimulatedState         Data-class returned by QuantumImagination

Canonical import paths:
    from nawal.quantum import QuantumMemory, QuantumPlanOptimizer
    from nawal.quantum import QuantumAnomalyDetector, QuantumImagination

Backward-compat paths (still work via shims):
    from nawal.integration.kinich_connector import QuantumEnhancedLayer
    from nawal.models.hybrid_llm import HybridQuantumClassicalLLM
"""

from quantum.hybrid_llm import HybridQuantumClassicalLLM
from quantum.kinich_connector import KinichQuantumConnector, QuantumEnhancedLayer
from quantum.quantum_anomaly import QuantumAnomalyDetector
from quantum.quantum_imagination import QuantumImagination, SimulatedState

# Phase 5 — quantum subsystems
from quantum.quantum_memory import QuantumMemory
from quantum.quantum_optimizer import QuantumPlanOptimizer

__all__ = [
    "HybridQuantumClassicalLLM",
    # Core
    "KinichQuantumConnector",
    "QuantumAnomalyDetector",
    "QuantumEnhancedLayer",
    "QuantumImagination",
    # Phase 5
    "QuantumMemory",
    "QuantumPlanOptimizer",
    "SimulatedState",
]
