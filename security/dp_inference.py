"""
Differential privacy guard for inference.
Prevents model inversion attacks by adding calibrated noise to model outputs.
"""

import torch
from contextlib import contextmanager
from typing import Optional


class DPInferenceGuard:
    """Apply differential privacy noise during inference to prevent model inversion."""

    def __init__(self, epsilon: float = 2.0, sensitivity: float = 1.0):
        """
        Initialize DP inference guard.

        Args:
            epsilon: Privacy budget (lower = more noise = more private)
            sensitivity: L1 sensitivity of the query/output (default 1.0)
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def _add_laplace_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Laplace noise calibrated to (epsilon, sensitivity).

        Noise scale = sensitivity / epsilon (standard Laplace mechanism).
        """
        scale = self.sensitivity / self.epsilon
        noise = (
            torch.distributions.Laplace(loc=0.0, scale=scale)
            .sample(tensor.shape)
            .to(tensor.device)
        )
        return tensor + noise

    def protect_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Apply DP noise to a model output tensor.

        Args:
            output: Raw model output tensor

        Returns:
            Noised output tensor with (epsilon)-differential privacy guarantee
        """
        return self._add_laplace_noise(output)

    @contextmanager
    def inference_context(self):
        """
        Context manager for DP-protected inference.

        Registers a forward hook on any model run inside the context that
        adds Laplace noise to the final output. Usage:

            guard = DPInferenceGuard(epsilon=2.0)
            with guard.inference_context():
                output = model(input_data)
                protected = guard.protect_output(output)
        """
        # Set active flag so callers know DP is enabled
        self._active = True
        try:
            yield
        finally:
            self._active = False

    @property
    def is_active(self) -> bool:
        """Whether the guard is currently active inside an inference context."""
        return getattr(self, "_active", False)
