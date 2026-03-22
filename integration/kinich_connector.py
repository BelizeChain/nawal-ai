"""
Kinich Quantum Connector  [DEPRECATED LOCATION — Phase 0]

.. deprecated::
    This file remains for backward compatibility.
    The canonical location is now ``nawal.quantum.kinich_connector``.

    Old (still works)::
        from nawal.integration.kinich_connector import QuantumEnhancedLayer

    New (preferred)::
        from nawal.quantum.kinich_connector import QuantumEnhancedLayer
"""

# Re-export everything from the canonical module to avoid code duplication.
from quantum.kinich_connector import (
    TORCH_AVAILABLE,
    KinichQuantumConnector,
    QuantumEnhancedLayer,
)

__all__ = ["TORCH_AVAILABLE", "KinichQuantumConnector", "QuantumEnhancedLayer"]
