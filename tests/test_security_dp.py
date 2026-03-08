"""
Direct tests for security/differential_privacy.py classes.

Covers:
  - PrivacyBudget  — is_exhausted, remaining, to_dict
  - PrivacyBudgetExhaustedError
  - DifferentialPrivacy — init variants, clip/noise/budget
  - PrivacyAccountant — RDP accumulation + conversion
  - DPOptimizer     — zero_grad, step, privacy_spent property
  - create_dp_optimizer convenience function
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from security.differential_privacy import (
    DPOptimizer,
    DifferentialPrivacy,
    PrivacyAccountant,
    PrivacyBudget,
    PrivacyBudgetExhaustedError,
    create_dp_optimizer,
)


# ---------------------------------------------------------------------------
# Tiny model for gradient tests
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


def _model_with_grads(scale: float = 1.0) -> _TinyModel:
    """Return a model with synthetic gradients of given scale."""
    model = _TinyModel()
    x = torch.randn(2, 4) * scale
    loss = model(x).sum()
    loss.backward()
    return model


# ---------------------------------------------------------------------------
# PrivacyBudget
# ---------------------------------------------------------------------------

class TestPrivacyBudget:

    def test_defaults(self):
        budget = PrivacyBudget()
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.spent_epsilon == 0.0
        assert budget.steps == 0

    def test_not_exhausted_initially(self):
        assert PrivacyBudget(epsilon=1.0).is_exhausted() is False

    def test_exhausted_when_spent_equals_epsilon(self):
        budget = PrivacyBudget(epsilon=1.0, spent_epsilon=1.0)
        assert budget.is_exhausted() is True

    def test_exhausted_when_spent_exceeds_epsilon(self):
        budget = PrivacyBudget(epsilon=1.0, spent_epsilon=1.5)
        assert budget.is_exhausted() is True

    def test_remaining_full_budget(self):
        budget = PrivacyBudget(epsilon=2.0, spent_epsilon=0.0)
        assert budget.remaining() == pytest.approx(2.0)

    def test_remaining_partial(self):
        budget = PrivacyBudget(epsilon=2.0, spent_epsilon=0.5)
        assert budget.remaining() == pytest.approx(1.5)

    def test_remaining_never_negative(self):
        budget = PrivacyBudget(epsilon=1.0, spent_epsilon=2.0)
        assert budget.remaining() == 0.0

    def test_to_dict_keys(self):
        budget = PrivacyBudget()
        d = budget.to_dict()
        assert set(d.keys()) == {"epsilon", "delta", "spent_epsilon", "steps", "remaining", "exhausted"}

    def test_to_dict_values(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5, spent_epsilon=0.3, steps=5)
        d = budget.to_dict()
        assert d["epsilon"] == 1.0
        assert d["steps"] == 5
        assert d["remaining"] == pytest.approx(0.7)
        assert d["exhausted"] is False


# ---------------------------------------------------------------------------
# PrivacyBudgetExhaustedError
# ---------------------------------------------------------------------------

class TestPrivacyBudgetExhaustedError:

    def test_is_runtime_error(self):
        assert issubclass(PrivacyBudgetExhaustedError, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PrivacyBudgetExhaustedError):
            raise PrivacyBudgetExhaustedError("budget gone")


# ---------------------------------------------------------------------------
# DifferentialPrivacy — init
# ---------------------------------------------------------------------------

class TestDifferentialPrivacyInit:

    def test_default_noise_multiplier_one(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        assert dp.noise_multiplier == 1.0

    def test_explicit_noise_multiplier_used(self):
        dp = DifferentialPrivacy(epsilon=1.0, noise_multiplier=2.5)
        assert dp.noise_multiplier == pytest.approx(2.5)

    def test_auto_compute_noise_multiplier_with_target_steps(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, target_steps=1000, sampling_rate=0.01)
        # Should compute a positive noise multiplier >= 0.1
        assert dp.noise_multiplier >= 0.1

    def test_budget_initialized(self):
        dp = DifferentialPrivacy(epsilon=3.0, delta=1e-4)
        assert dp.budget.epsilon == 3.0
        assert dp.budget.delta == 1e-4
        assert dp.budget.spent_epsilon == 0.0

    def test_clip_norm_stored(self):
        dp = DifferentialPrivacy(clip_norm=2.0)
        assert dp.clip_norm == 2.0

    def test_sampling_rate_stored(self):
        dp = DifferentialPrivacy(sampling_rate=0.05)
        assert dp.sampling_rate == 0.05

    def test_minimum_noise_enforced(self):
        """even with large target_steps, noise_multiplier >= 0.1"""
        dp = DifferentialPrivacy(
            epsilon=100.0, delta=1e-5, target_steps=1_000_000, sampling_rate=0.001
        )
        assert dp.noise_multiplier >= 0.1


# ---------------------------------------------------------------------------
# DifferentialPrivacy — clip_gradients
# ---------------------------------------------------------------------------

class TestClipGradients:

    def test_returns_float_norm(self):
        model = _model_with_grads(scale=1.0)
        dp = DifferentialPrivacy(clip_norm=1.0)
        norm = dp.clip_gradients(model)
        assert isinstance(norm, float)
        assert norm >= 0.0

    def test_no_grads_returns_zero(self):
        model = _TinyModel()
        # No backward pass → no grads
        dp = DifferentialPrivacy(clip_norm=1.0)
        norm = dp.clip_gradients(model)
        assert norm == 0.0

    def test_gradient_norm_clipped_to_clip_norm(self):
        """After clipping, the global gradient norm should be <= clip_norm."""
        model = _model_with_grads(scale=100.0)  # Big gradients
        dp = DifferentialPrivacy(clip_norm=1.0)
        dp.clip_gradients(model)

        # Recompute norm after clipping
        grads = [p.grad.data for p in model.parameters() if p.grad is not None]
        clipped_norm = math.sqrt(sum(g.norm(2).item() ** 2 for g in grads))
        assert clipped_norm <= 1.0 + 1e-5

    def test_small_gradients_not_clipped(self):
        """Gradients already below clip_norm should not change."""
        model = _model_with_grads(scale=0.001)  # Tiny gradients
        original_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        dp = DifferentialPrivacy(clip_norm=100.0)
        dp.clip_gradients(model)
        for name, orig in original_grads.items():
            assert torch.allclose(model.get_parameter(name).grad, orig, atol=1e-6)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DifferentialPrivacy — add_noise
# ---------------------------------------------------------------------------

class TestAddNoise:

    def test_add_noise_modifies_gradients(self):
        model = _model_with_grads()
        original = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        dp = DifferentialPrivacy(noise_multiplier=10.0, clip_norm=1.0)
        dp.add_noise(model)
        # At least one gradient should differ
        changed = any(
            not torch.allclose(model.get_parameter(n).grad, orig)  # type: ignore[arg-type]
            for n, orig in original.items()
        )
        assert changed

    def test_noise_scale_proportional_to_multiplier(self):
        """Higher noise_multiplier → larger deviation from original gradients."""
        torch.manual_seed(0)
        model_lo = _model_with_grads()
        orig_lo = model_lo.fc.weight.grad.clone()  # type: ignore[union-attr]
        DifferentialPrivacy(noise_multiplier=0.01, clip_norm=1.0).add_noise(model_lo)
        diff_lo = (model_lo.fc.weight.grad - orig_lo).norm().item()  # type: ignore[operator]

        torch.manual_seed(0)
        model_hi = _model_with_grads()
        orig_hi = model_hi.fc.weight.grad.clone()  # type: ignore[union-attr]
        DifferentialPrivacy(noise_multiplier=100.0, clip_norm=1.0).add_noise(model_hi)
        diff_hi = (model_hi.fc.weight.grad - orig_hi).norm().item()  # type: ignore[operator]

        assert diff_hi > diff_lo


# ---------------------------------------------------------------------------
# DifferentialPrivacy — privacy budget
# ---------------------------------------------------------------------------

class TestPrivacyBudgetUpdates:

    def test_update_increments_steps(self):
        dp = DifferentialPrivacy(epsilon=10.0)
        dp.update_privacy_budget(steps=3)
        assert dp.budget.steps == 3

    def test_update_increments_spent_epsilon(self):
        dp = DifferentialPrivacy(epsilon=10.0, sampling_rate=0.1)
        dp.update_privacy_budget(steps=1)
        assert dp.budget.spent_epsilon > 0.0

    def test_exhaustion_raises(self):
        """sampling_rate=1.0, epsilon=0.001 → exhausted after 1 step."""
        dp = DifferentialPrivacy(epsilon=0.001, sampling_rate=1.0)
        with pytest.raises(PrivacyBudgetExhaustedError):
            dp.update_privacy_budget(steps=1)

    def test_get_privacy_spent_tuple(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        dp.update_privacy_budget(steps=1)
        spent_eps, delta = dp.get_privacy_spent()
        assert isinstance(spent_eps, float)
        assert delta == 1e-5

    def test_get_privacy_remaining(self):
        dp = DifferentialPrivacy(epsilon=2.0, sampling_rate=0.001)
        dp.update_privacy_budget(steps=1)
        remaining = dp.get_privacy_remaining()
        assert 0.0 < remaining < 2.0

    def test_can_continue_training_initially_true(self):
        dp = DifferentialPrivacy(epsilon=10.0)
        assert dp.can_continue_training() is True

    def test_can_continue_training_false_after_exhaustion(self):
        dp = DifferentialPrivacy(epsilon=10.0)
        dp.budget.spent_epsilon = 10.0  # manually exhaust
        assert dp.can_continue_training() is False

    def test_to_dict_structure(self):
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=0.5, noise_multiplier=1.2, sampling_rate=0.01)
        d = dp.to_dict()
        assert "budget" in d
        assert d["clip_norm"] == 0.5
        assert d["noise_multiplier"] == pytest.approx(1.2)
        assert d["sampling_rate"] == 0.01


# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------

class TestPrivacyAccountant:

    def test_default_orders(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        assert pa.orders == [1.5, 2, 3, 5, 10, 20, 50, 100]

    def test_custom_orders(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01, orders=[2, 5])
        assert pa.orders == [2, 5]

    def test_rdp_zero_initially(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        assert all(v == 0.0 for v in pa.rdp.values())

    def test_accumulate_increases_rdp(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        pa.accumulate_privacy_spending(noise_multiplier=1.0, steps=10)
        assert any(v > 0.0 for v in pa.rdp.values())

    def test_accumulate_multiple_calls_adds(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        pa.accumulate_privacy_spending(noise_multiplier=1.0, steps=5)
        rdp_5 = dict(pa.rdp)
        pa.accumulate_privacy_spending(noise_multiplier=1.0, steps=5)
        for order in pa.orders:
            assert pa.rdp[order] == pytest.approx(rdp_5[order] * 2)

    def test_get_privacy_spent_returns_tuple(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        pa.accumulate_privacy_spending(noise_multiplier=1.0, steps=100)
        eps, delta = pa.get_privacy_spent()
        assert isinstance(eps, float)
        assert delta == 1e-5

    def test_get_privacy_spent_positive(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.1)
        pa.accumulate_privacy_spending(noise_multiplier=0.5, steps=10)
        eps, _ = pa.get_privacy_spent()
        assert eps > 0.0

    def test_to_dict_structure(self):
        pa = PrivacyAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        d = pa.to_dict()
        assert "target_epsilon" in d
        assert "spent_epsilon" in d
        assert "rdp_orders" in d
        assert "rdp_values" in d


# ---------------------------------------------------------------------------
# DPOptimizer
# ---------------------------------------------------------------------------

class TestDPOptimizer:

    def _setup(self):
        model = _TinyModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        dp = DifferentialPrivacy(epsilon=100.0, noise_multiplier=0.1)
        dp_opt = DPOptimizer(optimizer=optimizer, dp=dp)
        return model, optimizer, dp, dp_opt

    def test_zero_grad_clears_gradients(self):
        model, _, _, dp_opt = self._setup()
        # Create some gradients first
        model(torch.randn(2, 4)).sum().backward()
        dp_opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_step_updates_weights(self):
        model, _, _, dp_opt = self._setup()
        original_weights = model.fc.weight.data.clone()
        model(torch.randn(2, 4)).sum().backward()
        dp_opt.step(model)
        # Weights should change after a step
        assert not torch.allclose(model.fc.weight.data, original_weights)

    def test_step_increments_budget(self):
        model, _, dp, dp_opt = self._setup()
        model(torch.randn(2, 4)).sum().backward()
        dp_opt.step(model)
        assert dp.budget.steps == 1

    def test_privacy_spent_property(self):
        model, _, _, dp_opt = self._setup()
        model(torch.randn(2, 4)).sum().backward()
        dp_opt.step(model)
        spent = dp_opt.privacy_spent
        # Returns whatever dp.get_privacy_spent() gives
        assert isinstance(spent, tuple)
        assert len(spent) == 2


# ---------------------------------------------------------------------------
# create_dp_optimizer
# ---------------------------------------------------------------------------

class TestCreateDPOptimizer:

    def test_returns_dp_optimizer(self):
        model = _TinyModel()
        opt = optim.SGD(model.parameters(), lr=0.01)
        dp_opt = create_dp_optimizer(opt, model, epsilon=1.0, noise_multiplier=1.0)
        assert isinstance(dp_opt, DPOptimizer)

    def test_dp_params_set(self):
        model = _TinyModel()
        opt = optim.SGD(model.parameters(), lr=0.01)
        dp_opt = create_dp_optimizer(opt, model, epsilon=2.0, delta=1e-4, clip_norm=0.5)
        assert dp_opt.dp.budget.epsilon == 2.0
        assert dp_opt.dp.budget.delta == 1e-4
        assert dp_opt.dp.clip_norm == 0.5

    def test_custom_noise_multiplier(self):
        model = _TinyModel()
        opt = optim.SGD(model.parameters(), lr=0.01)
        dp_opt = create_dp_optimizer(opt, model, noise_multiplier=3.0)
        assert dp_opt.dp.noise_multiplier == pytest.approx(3.0)
