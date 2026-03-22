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

from .byzantine_detection import AggregationMethod, ByzantineDetector
from .differential_privacy import DifferentialPrivacy, PrivacyAccountant, PrivacyBudget
from .dp_inference import DPInferenceGuard
from .secure_aggregation import SecureAggregator

__all__ = [
    "AggregationMethod",
    "ByzantineDetector",
    "DPInferenceGuard",
    "DifferentialPrivacy",
    "PrivacyAccountant",
    "PrivacyBudget",
    "SecureAggregator",
]

__version__ = "0.1.0"
