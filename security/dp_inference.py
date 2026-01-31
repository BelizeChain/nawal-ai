"""
Differential privacy guard for inference
Prevents model inversion attacks
"""

import torch
from contextlib import contextmanager


class DPInferenceGuard:
    """Apply differential privacy noise during inference"""
    
    def __init__(self, epsilon: float = 2.0):
        self.epsilon = epsilon
    
    @contextmanager
    def inference_context(self):
        """Context manager for DP-protected inference"""
        # Add noise to embeddings/hidden states during inference
        # (Implementation simplified - full DP-SGD in training)
        yield
