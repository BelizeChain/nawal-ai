"""
B12 — Quantum & Kinich Integration audit tests.

C12.1  Feature extraction correctness (input validation, NaN/Inf guards, deterministic fallback)
C12.2  Kinich response handling (missing key, shape mismatch, integration shim re-exports)
C12.3  Timeout & circuit breaker (configurable timeout, breaker open/close/reset)
"""
from __future__ import annotations

import asyncio
import urllib.error
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from quantum.kinich_connector import (
    KinichQuantumConnector,
    QuantumEnhancedLayer,
    TORCH_AVAILABLE,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _connector(**kwargs) -> KinichQuantumConnector:
    """Build a KinichQuantumConnector with the health-check mocked out."""
    with patch("urllib.request.urlopen") as m:
        m.side_effect = urllib.error.URLError("no kinich")
        return KinichQuantumConnector(fallback_to_classical=True, **kwargs)


def _quantum_connector(**kwargs) -> KinichQuantumConnector:
    """Build a connector that believes Kinich is healthy."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    with patch("urllib.request.urlopen", return_value=mock_resp):
        return KinichQuantumConnector(fallback_to_classical=True, **kwargs)


def _mock_kinich_session(response_json: dict, status: int = 200):
    """Return an aiohttp.ClientSession mock that returns *response_json*."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_json)
    mock_resp.text = AsyncMock(return_value="error")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


# ══════════════════════════════════════════════════════════════════════════════
# C12.1  Feature extraction correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestC12_1_FeatureValidation:
    """Input validation, NaN/Inf guards, deterministic fallback."""

    # ── dimension checks ──────────────────────────────────────────────────

    def test_3d_input_raises_valueerror(self):
        conn = _connector(classical_dim=8)
        features = np.ones((2, 3, 8))
        with pytest.raises(ValueError, match="2D"):
            asyncio.run(conn.quantum_process(features))

    def test_wrong_feature_dim_raises(self):
        conn = _connector(classical_dim=8)
        features = np.ones((2, 16))
        with pytest.raises(ValueError, match="classical_dim"):
            asyncio.run(conn.quantum_process(features))

    def test_1d_input_auto_reshaped(self):
        conn = _connector(classical_dim=8)
        features = np.ones(8)
        result = asyncio.run(conn.quantum_process(features))
        assert result.shape == (1, 8)

    # ── NaN / Inf guards ──────────────────────────────────────────────────

    def test_nan_features_raise_valueerror(self):
        conn = _connector(classical_dim=8)
        features = np.array([[1.0, float("nan")] + [0.0] * 6])
        with pytest.raises(ValueError, match="NaN"):
            asyncio.run(conn.quantum_process(features))

    def test_inf_features_raise_valueerror(self):
        conn = _connector(classical_dim=8)
        features = np.array([[float("inf")] + [0.0] * 7])
        with pytest.raises(ValueError, match="NaN or Inf"):
            asyncio.run(conn.quantum_process(features))

    # ── deterministic fallback ────────────────────────────────────────────

    def test_classical_fallback_is_deterministic(self):
        """Two fresh connectors with same dims must produce identical fallback."""
        a = _connector(classical_dim=32)
        b = _connector(classical_dim=32)
        features = np.ones((2, 32))
        np.testing.assert_array_equal(
            a._classical_fallback(features),
            b._classical_fallback(features),
        )

    def test_fallback_shape_matches_input(self):
        conn = _connector(classical_dim=16)
        features = np.random.default_rng(0).standard_normal((5, 16))
        result = conn._classical_fallback(features)
        assert result.shape == features.shape


# ══════════════════════════════════════════════════════════════════════════════
# C12.2  Kinich response handling
# ══════════════════════════════════════════════════════════════════════════════

class TestC12_2_ResponseHandling:
    """Missing key, shape mismatch, integration shim correctness."""

    # ── missing key ───────────────────────────────────────────────────────

    def test_missing_key_raises_runtime_error(self):
        conn = _quantum_connector(classical_dim=8)
        features = np.ones((2, 8))
        resp_json = {"wrong_key": [[0.0] * 8, [0.0] * 8]}
        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            with pytest.raises(RuntimeError, match="quantum_enhanced_features"):
                asyncio.run(conn._quantum_forward(features, "vqc"))

    # ── shape mismatch ────────────────────────────────────────────────────

    def test_shape_mismatch_raises_runtime_error(self):
        conn = _quantum_connector(classical_dim=8)
        features = np.ones((2, 8))
        # Return 3 rows instead of 2
        resp_json = {"quantum_enhanced_features": [[0.0] * 8] * 3}
        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            with pytest.raises(RuntimeError, match="shape"):
                asyncio.run(conn._quantum_forward(features, "vqc"))

    def test_correct_response_returns_ndarray(self):
        conn = _quantum_connector(classical_dim=8)
        features = np.ones((2, 8))
        resp_json = {"quantum_enhanced_features": np.ones((2, 8)).tolist()}
        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            result = asyncio.run(conn._quantum_forward(features, "vqc"))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 8)

    # ── quantum_process wires into quantum path when available ────────────

    def test_quantum_process_uses_quantum_path_when_available(self):
        conn = _quantum_connector(classical_dim=8)
        features = np.ones((2, 8))
        resp_json = {"quantum_enhanced_features": np.ones((2, 8)).tolist()}
        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            result = asyncio.run(conn.quantum_process(features))
        assert conn.stats["quantum_calls"] == 1
        assert conn.stats["fallback_calls"] == 0
        assert result.shape == (2, 8)

    # ── integration shim re-exports ───────────────────────────────────────

    def test_integration_shim_exports_match_canonical(self):
        """integration.kinich_connector re-exports are the exact same objects."""
        from integration.kinich_connector import (
            KinichQuantumConnector as IntConn,
            QuantumEnhancedLayer as IntLayer,
            TORCH_AVAILABLE as IntTorch,
        )
        assert IntConn is KinichQuantumConnector
        assert IntLayer is QuantumEnhancedLayer
        assert IntTorch is TORCH_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# C12.3  Timeout & circuit breaker
# ══════════════════════════════════════════════════════════════════════════════

class TestC12_3_TimeoutAndCircuitBreaker:
    """Configurable timeout, circuit breaker open / close / reset."""

    # ── configurable timeout ──────────────────────────────────────────────

    def test_default_timeout(self):
        conn = _connector()
        assert conn.request_timeout == 30.0

    def test_custom_timeout_stored(self):
        conn = _connector(request_timeout=10.0)
        assert conn.request_timeout == 10.0

    def test_timeout_used_in_quantum_forward(self):
        """The aiohttp timeout kwarg receives self.request_timeout."""
        conn = _quantum_connector(classical_dim=8, request_timeout=7.5)
        features = np.ones((2, 8))
        resp_json = {"quantum_enhanced_features": np.ones((2, 8)).tolist()}

        import aiohttp

        captured_timeout = None
        original_client_timeout = aiohttp.ClientTimeout

        def capture_timeout(**kwargs):
            nonlocal captured_timeout
            captured_timeout = kwargs.get("total")
            return original_client_timeout(**kwargs)

        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            with patch("aiohttp.ClientTimeout", side_effect=capture_timeout):
                asyncio.run(conn._quantum_forward(features, "vqc"))

        assert captured_timeout == 7.5

    # ── circuit breaker threshold ─────────────────────────────────────────

    def test_default_circuit_breaker_threshold(self):
        conn = _connector()
        assert conn._circuit_breaker_threshold == 5
        assert conn._circuit_open is False
        assert conn._consecutive_failures == 0

    def test_custom_circuit_breaker_threshold(self):
        conn = _connector(circuit_breaker_threshold=3)
        assert conn._circuit_breaker_threshold == 3

    # ── circuit breaker opens after N failures ────────────────────────────

    def test_circuit_breaker_opens_after_threshold_failures(self):
        conn = _quantum_connector(classical_dim=8, circuit_breaker_threshold=3, enable_caching=False)
        rng = np.random.default_rng(0)

        # Mock aiohttp to always fail
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(side_effect=RuntimeError("timeout"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            for i in range(3):
                features = rng.standard_normal((2, 8))
                asyncio.run(conn.quantum_process(features))

        assert conn._circuit_open is True
        assert conn._consecutive_failures == 3

    def test_open_circuit_skips_quantum(self):
        """Once open, quantum_process goes straight to fallback."""
        conn = _quantum_connector(classical_dim=8)
        conn._circuit_open = True

        features = np.ones((2, 8))
        result = asyncio.run(conn.quantum_process(features))
        assert conn.stats["fallback_calls"] == 1
        assert conn.stats["quantum_calls"] == 0
        assert result.shape == (2, 8)

    # ── circuit breaker closes on success ─────────────────────────────────

    def test_circuit_closes_on_success(self):
        conn = _quantum_connector(classical_dim=8)
        conn._consecutive_failures = 2
        assert conn._circuit_open is False

        features = np.ones((2, 8))
        resp_json = {"quantum_enhanced_features": np.ones((2, 8)).tolist()}
        with patch("aiohttp.ClientSession", return_value=_mock_kinich_session(resp_json)):
            asyncio.run(conn.quantum_process(features))

        assert conn._consecutive_failures == 0

    def test_circuit_breaker_reopens_correctly(self):
        """After recovery (close), breaker can re-open on new failure streak."""
        conn = _quantum_connector(classical_dim=8, circuit_breaker_threshold=2, enable_caching=False)
        # Simulate a recovered state
        conn._consecutive_failures = 0
        conn._circuit_open = False

        rng = np.random.default_rng(42)
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(side_effect=RuntimeError("fail"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            asyncio.run(conn.quantum_process(rng.standard_normal((2, 8))))
            asyncio.run(conn.quantum_process(rng.standard_normal((2, 8))))

        assert conn._circuit_open is True

    # ── manual reset ──────────────────────────────────────────────────────

    def test_manual_reset_clears_breaker(self):
        conn = _connector()
        conn._circuit_open = True
        conn._consecutive_failures = 99
        conn.reset_circuit_breaker()
        assert conn._circuit_open is False
        assert conn._consecutive_failures == 0

    # ── statistics include circuit breaker state ──────────────────────────

    def test_statistics_include_circuit_breaker_fields(self):
        conn = _connector()
        stats = conn.get_statistics()
        assert "circuit_open" in stats
        assert "consecutive_failures" in stats
        assert stats["circuit_open"] is False
        assert stats["consecutive_failures"] == 0

    def test_statistics_reflect_open_breaker(self):
        conn = _connector()
        conn._circuit_open = True
        conn._consecutive_failures = 5
        stats = conn.get_statistics()
        assert stats["circuit_open"] is True
        assert stats["consecutive_failures"] == 5

    # ── _is_quantum_available ─────────────────────────────────────────────

    def test_is_quantum_available_false_when_circuit_open(self):
        conn = _quantum_connector()
        conn._circuit_open = True
        assert conn._is_quantum_available() is False

    def test_is_quantum_available_true_normal(self):
        conn = _quantum_connector()
        assert conn._is_quantum_available() is True

    def test_is_quantum_available_false_when_kinich_down(self):
        conn = _connector()  # health check fails → kinich_available=False
        assert conn._is_quantum_available() is False
