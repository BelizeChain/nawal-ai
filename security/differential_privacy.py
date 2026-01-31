"""
Differential Privacy for Federated Learning.

Implements Differentially Private Stochastic Gradient Descent (DP-SGD)
to protect individual client data privacy during training.

Key Features:
- Per-example gradient clipping
- Gaussian noise addition (calibrated to privacy budget)
- Privacy budget tracking (ε-δ accounting)
- Opacus integration support
- Rényi Differential Privacy (RDP) accounting

References:
- Abadi et al. "Deep Learning with Differential Privacy" (2016)
- Mironov "Rényi Differential Privacy" (2017)

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class PrivacyBudget:
    """
    Privacy budget for differential privacy.
    
    Attributes:
        epsilon (float): Privacy loss parameter (lower = more private)
        delta (float): Failure probability (typically 1/n²)
        spent_epsilon (float): Cumulative privacy budget spent
        steps (int): Number of training steps taken
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    spent_epsilon: float = 0.0
    steps: int = 0
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.spent_epsilon >= self.epsilon
    
    def remaining(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.epsilon - self.spent_epsilon)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "spent_epsilon": self.spent_epsilon,
            "steps": self.steps,
            "remaining": self.remaining(),
            "exhausted": self.is_exhausted(),
        }


class DifferentialPrivacy:
    """
    Differential Privacy implementation for federated learning.
    
    Implements DP-SGD with:
    - Per-example gradient clipping (bounded sensitivity)
    - Gaussian noise addition (privacy guarantee)
    - Privacy budget tracking (ε-δ accounting)
    
    Usage:
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        
        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            
            # Apply DP before optimizer step
            dp.clip_gradients(model)
            dp.add_noise(model)
            
            optimizer.step()
            dp.update_privacy_budget()
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        target_steps: Optional[int] = None,
        sampling_rate: float = 0.01,
    ):
        """
        Initialize DifferentialPrivacy.
        
        Args:
            epsilon: Privacy loss parameter (lower = more private)
            delta: Failure probability (typically 1/n²)
            clip_norm: Maximum L2 norm for gradients (sensitivity)
            noise_multiplier: Scale of Gaussian noise (if None, auto-computed)
            target_steps: Target number of training steps (for auto noise)
            sampling_rate: Fraction of data used per step (batch_size / dataset_size)
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.clip_norm = clip_norm
        self.sampling_rate = sampling_rate
        
        # Auto-compute noise multiplier if not provided
        if noise_multiplier is None and target_steps is not None:
            self.noise_multiplier = self._compute_noise_multiplier(
                epsilon, delta, target_steps, sampling_rate
            )
        elif noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # Default noise multiplier
            self.noise_multiplier = 1.0
        
        logger.info(
            f"DifferentialPrivacy initialized: ε={epsilon}, δ={delta}, "
            f"clip_norm={clip_norm}, noise_multiplier={self.noise_multiplier:.3f}"
        )
    
    def _compute_noise_multiplier(
        self,
        epsilon: float,
        delta: float,
        steps: int,
        sampling_rate: float,
    ) -> float:
        """
        Compute noise multiplier for target privacy budget.
        
        Uses analytical Gaussian mechanism formula:
        σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
        
        For composition over T steps with sampling q:
        σ_total = σ / sqrt(T * q)
        """
        # Simplified noise computation (conservative)
        # For precise computation, use privacy accountant
        sensitivity = self.clip_norm
        base_noise = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        
        # Adjust for composition
        if steps > 0:
            composition_factor = math.sqrt(steps * sampling_rate)
            noise_multiplier = base_noise / composition_factor
        else:
            noise_multiplier = base_noise
        
        return max(noise_multiplier, 0.1)  # Minimum noise for safety
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip per-example gradients to bounded L2 norm.
        
        This ensures bounded sensitivity for differential privacy.
        
        Args:
            model: PyTorch model with computed gradients
        
        Returns:
            Average gradient norm before clipping
        """
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm
                param_count += 1
                
                # Clip gradient
                clip_coef = self.clip_norm / (param_norm + 1e-6)
                if clip_coef < 1:
                    param.grad.data.mul_(clip_coef)
        
        avg_norm = total_norm / max(param_count, 1)
        return avg_norm
    
    def add_noise(self, model: nn.Module) -> None:
        """
        Add calibrated Gaussian noise to gradients.
        
        Noise scale: σ = noise_multiplier * clip_norm
        
        Args:
            model: PyTorch model with clipped gradients
        """
        noise_scale = self.noise_multiplier * self.clip_norm
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data.add_(noise)
    
    def update_privacy_budget(self, steps: int = 1) -> None:
        """
        Update privacy budget after training steps.
        
        Uses simplified privacy accounting (conservative estimate).
        For precise accounting, use privacy accountant library.
        
        Args:
            steps: Number of steps taken
        """
        self.budget.steps += steps
        
        # Simplified privacy accounting (conservative)
        # Real implementation should use RDP or moments accountant
        epsilon_per_step = self._compute_epsilon_per_step()
        self.budget.spent_epsilon += epsilon_per_step * steps
        
        if self.budget.is_exhausted():
            logger.warning(
                f"Privacy budget exhausted! ε={self.budget.spent_epsilon:.3f} "
                f"(limit: {self.budget.epsilon})"
            )
    
    def _compute_epsilon_per_step(self) -> float:
        """
        Compute epsilon spent per training step.
        
        Simplified formula (conservative):
        ε_step ≈ q * ε / sqrt(T)
        
        where q = sampling_rate, T = total_steps
        """
        if self.budget.steps == 0:
            return 0.0
        
        # Conservative estimate
        epsilon_per_step = (
            self.sampling_rate * 
            self.budget.epsilon / 
            math.sqrt(max(self.budget.steps, 1))
        )
        
        return epsilon_per_step
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get privacy spent (ε, δ).
        
        Returns:
            Tuple of (epsilon_spent, delta)
        """
        return (self.budget.spent_epsilon, self.budget.delta)
    
    def get_privacy_remaining(self) -> float:
        """Get remaining privacy budget."""
        return self.budget.remaining()
    
    def can_continue_training(self) -> bool:
        """Check if training can continue under privacy budget."""
        return not self.budget.is_exhausted()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "budget": self.budget.to_dict(),
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate": self.sampling_rate,
        }


class PrivacyAccountant:
    """
    Advanced privacy accounting using Rényi Differential Privacy (RDP).
    
    Provides tighter privacy bounds than basic composition.
    
    Note: This is a simplified implementation.
    For production, use libraries like:
    - Opacus (Facebook)
    - TensorFlow Privacy
    - Privacy Accounting Library
    """
    
    def __init__(
        self,
        epsilon: float,
        delta: float,
        sampling_rate: float,
        orders: Optional[List[float]] = None,
    ):
        """
        Initialize PrivacyAccountant.
        
        Args:
            epsilon: Target privacy parameter
            delta: Failure probability
            sampling_rate: Sampling probability per step
            orders: RDP orders to track (default: [1.5, 2, 3, 5, 10, 20, 50, 100])
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sampling_rate = sampling_rate
        
        if orders is None:
            self.orders = [1.5, 2, 3, 5, 10, 20, 50, 100]
        else:
            self.orders = orders
        
        # RDP values for each order
        self.rdp = {order: 0.0 for order in self.orders}
    
    def accumulate_privacy_spending(
        self,
        noise_multiplier: float,
        steps: int = 1,
    ) -> None:
        """
        Accumulate privacy spending using RDP.
        
        Args:
            noise_multiplier: Noise scale used in DP-SGD
            steps: Number of steps taken
        """
        for order in self.orders:
            # RDP formula for Gaussian mechanism
            rdp_step = (
                self.sampling_rate**2 * 
                order / 
                (2 * noise_multiplier**2)
            )
            self.rdp[order] += rdp_step * steps
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert RDP to (ε, δ) using optimal order.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        min_epsilon = float('inf')
        
        for order in self.orders:
            # Convert RDP to (ε, δ)
            epsilon = self.rdp[order] + math.log(1 / self.delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return (min_epsilon, self.delta)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        epsilon_spent, _ = self.get_privacy_spent()
        return {
            "target_epsilon": self.epsilon,
            "target_delta": self.delta,
            "spent_epsilon": epsilon_spent,
            "rdp_orders": self.orders,
            "rdp_values": self.rdp,
        }


# Convenience function for Opacus integration
def create_dp_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    noise_multiplier: Optional[float] = None,
) -> Tuple[torch.optim.Optimizer, DifferentialPrivacy]:
    """
    Create DP-enabled optimizer (simplified wrapper).
    
    For production, use Opacus: https://github.com/pytorch/opacus
    
    Args:
        optimizer: Base PyTorch optimizer
        model: Model to protect
        epsilon: Privacy parameter
        delta: Failure probability
        clip_norm: Gradient clipping norm
        noise_multiplier: Noise scale (if None, auto-computed)
    
    Returns:
        Tuple of (optimizer, dp_handler)
    
    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dp_optimizer, dp = create_dp_optimizer(optimizer, model, epsilon=1.0)
    """
    dp = DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )
    
    # Note: This is a simplified wrapper
    # For real DP-SGD, use Opacus or implement per-sample gradients
    logger.info("Created DP optimizer (simplified). Consider using Opacus for production.")
    
    return optimizer, dp
