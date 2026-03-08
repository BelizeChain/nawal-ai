"""
Nawal AI Security Module.

Provides security and privacy features for federated learning:
- Differential Privacy (DP-SGD)
- Secure Aggregation (encrypted gradients)
- Byzantine Detection (robust aggregation)
- Model Poisoning Detection (anomaly detection)

Author: BelizeChain Team
License: MIT
"""

from .differential_privacy import DifferentialPrivacy, PrivacyBudget, PrivacyAccountant
from .secure_aggregation import SecureAggregator
from .byzantine_detection import ByzantineDetector, AggregationMethod
from .dp_inference import DPInferenceGuard

__all__ = [
    "DifferentialPrivacy",
    "PrivacyBudget",
    "PrivacyAccountant",
    "SecureAggregator",
    "ByzantineDetector",
    "AggregationMethod",
    "DPInferenceGuard",
]

__version__ = "0.1.0"
