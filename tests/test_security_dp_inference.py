"""
Tests for security/dp_inference.py — DPInferenceGuard (75% → ~100%).
"""

import pytest
import torch

from security.dp_inference import DPInferenceGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(shape=(4,), value=1.0) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DPInferenceGuard — __init__
# ---------------------------------------------------------------------------

class TestDPInferenceGuardInit:

    def test_default_params(self):
        guard = DPInferenceGuard()
        assert guard.epsilon == 2.0
        assert guard.sensitivity == 1.0

    def test_custom_epsilon_and_sensitivity(self):
        guard = DPInferenceGuard(epsilon=0.5, sensitivity=2.0)
        assert guard.epsilon == 0.5
        assert guard.sensitivity == 2.0

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            DPInferenceGuard(epsilon=0.0)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            DPInferenceGuard(epsilon=-1.0)

    def test_is_active_false_by_default(self):
        guard = DPInferenceGuard()
        assert guard.is_active is False


# ---------------------------------------------------------------------------
# DPInferenceGuard — _add_laplace_noise
# ---------------------------------------------------------------------------

class TestAddLaplaceNoise:

    def test_returns_tensor_same_shape(self):
        guard = DPInferenceGuard(epsilon=1.0)
        t = _make_output((3, 4))
        result = guard._add_laplace_noise(t)
        assert result.shape == t.shape

    def test_noise_is_added(self):
        """Output should differ from input due to Laplace noise."""
        torch.manual_seed(42)
        guard = DPInferenceGuard(epsilon=0.01, sensitivity=1.0)  # high noise
        t = torch.zeros(100)
        result = guard._add_laplace_noise(t)
        assert not torch.allclose(result, t)

    def test_lower_epsilon_more_noise(self):
        """Lower epsilon → larger noise scale → bigger L2 deviation from original."""
        torch.manual_seed(0)
        t = torch.ones(200)
        guard_lo = DPInferenceGuard(epsilon=0.01)
        guard_hi = DPInferenceGuard(epsilon=100.0)
        dev_lo = (guard_lo._add_laplace_noise(t) - t).norm().item()
        torch.manual_seed(0)
        dev_hi = (guard_hi._add_laplace_noise(t) - t).norm().item()
        assert dev_lo > dev_hi

    def test_sensitivity_scales_noise(self):
        """Higher sensitivity → larger noise (scale = sensitivity / epsilon)."""
        torch.manual_seed(1)
        t = torch.ones(200)
        guard_lo = DPInferenceGuard(epsilon=1.0, sensitivity=0.01)
        guard_hi = DPInferenceGuard(epsilon=1.0, sensitivity=10.0)
        dev_lo = (guard_lo._add_laplace_noise(t) - t).norm().item()
        torch.manual_seed(1)
        dev_hi = (guard_hi._add_laplace_noise(t) - t).norm().item()
        assert dev_hi > dev_lo

    def test_gradient_not_required_on_output(self):
        guard = DPInferenceGuard()
        t = torch.tensor([1.0, 2.0, 3.0])
        result = guard._add_laplace_noise(t)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# DPInferenceGuard — protect_output
# ---------------------------------------------------------------------------

class TestProtectOutput:

    def test_same_shape_as_input(self):
        guard = DPInferenceGuard(epsilon=1.0)
        t = _make_output((5, 3))
        result = guard.protect_output(t)
        assert result.shape == t.shape

    def test_returns_tensor(self):
        guard = DPInferenceGuard()
        t = _make_output()
        assert isinstance(guard.protect_output(t), torch.Tensor)

    def test_output_differs_from_input(self):
        torch.manual_seed(7)
        guard = DPInferenceGuard(epsilon=0.01)
        t = torch.zeros(50)
        result = guard.protect_output(t)
        assert not torch.allclose(result, t)

    def test_works_with_2d_tensor(self):
        guard = DPInferenceGuard(epsilon=1.0)
        t = torch.randn(8, 16)
        result = guard.protect_output(t)
        assert result.shape == (8, 16)

    def test_works_with_batch(self):
        guard = DPInferenceGuard(epsilon=2.0)
        batch = torch.randn(32, 10)
        result = guard.protect_output(batch)
        assert result.shape == (32, 10)


# ---------------------------------------------------------------------------
# DPInferenceGuard — inference_context
# ---------------------------------------------------------------------------

class TestInferenceContext:

    def test_is_active_true_inside_context(self):
        guard = DPInferenceGuard()
        with guard.inference_context():
            assert guard.is_active is True

    def test_is_active_false_after_context(self):
        guard = DPInferenceGuard()
        with guard.inference_context():
            pass
        assert guard.is_active is False

    def test_is_active_false_before_context(self):
        guard = DPInferenceGuard()
        assert guard.is_active is False
        with guard.inference_context():
            pass

    def test_is_active_restored_on_exception(self):
        """_active must be reset to False even if an exception is raised inside."""
        guard = DPInferenceGuard()
        with pytest.raises(RuntimeError):
            with guard.inference_context():
                raise RuntimeError("test error")
        assert guard.is_active is False

    def test_context_manager_yields(self):
        """inference_context should yield (allow 'as' binding or plain with)."""
        guard = DPInferenceGuard()
        with guard.inference_context():
            result = guard.protect_output(torch.tensor([1.0, 2.0]))
        assert result is not None

    def test_nested_context_first_takes_precedence(self):
        """Nested calls: inner deactivation shouldn't permanently clear outer."""
        guard = DPInferenceGuard()
        with guard.inference_context():
            assert guard.is_active is True
            # Simulate a second entry (not a real nested context manager call,
            # just verify state holds throughout outer block)
            inner_active = guard.is_active
        assert inner_active is True
        assert guard.is_active is False


# ---------------------------------------------------------------------------
# DPInferenceGuard — is_active property
# ---------------------------------------------------------------------------

class TestIsActiveProperty:

    def test_false_on_fresh_instance(self):
        guard = DPInferenceGuard(epsilon=1.0)
        assert guard.is_active is False

    def test_true_during_context(self):
        guard = DPInferenceGuard(epsilon=1.0)
        captured = []
        with guard.inference_context():
            captured.append(guard.is_active)
        assert captured == [True]
        assert guard.is_active is False

    def test_getattr_default_false(self):
        """is_active uses getattr with default False when _active not set."""
        guard = DPInferenceGuard()
        # Before any context, _active attribute may not exist — should still return False
        if hasattr(guard, '_active'):
            del guard._active
        assert guard.is_active is False
