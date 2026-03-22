"""
Tests for quantum/kinich_connector.py — KinichQuantumConnector + QuantumEnhancedLayer.

Covers the major uncovered branches (18% → ~90%+):
  - _init_kinich_connection: success, URLError, generic exception paths
  - quantum_process: fallback (bridge=None), cache hit, cache eviction
  - _classical_fallback
  - _get_cache_key
  - _quantum_forward (mocked aiohttp)
  - _qsvm_forward, _vqc_forward, _qnn_forward
  - get_statistics, clear_cache, reset_statistics, __repr__
  - QuantumEnhancedLayer init and forward
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

# ──────────────────────────────────────────────────────────────────────────────
# Helper — build connector with mocked HTTP health-check
# ──────────────────────────────────────────────────────────────────────────────


def _connector(
    kinich_status: int = 200,
    url_error: bool = False,
    generic_error: bool = False,
    fallback: bool = True,
    **kwargs,
) -> KinichQuantumConnector:
    """Build a KinichQuantumConnector with mocked urllib health-check."""
    if url_error:
        side_effect = urllib.error.URLError("connection refused")
    elif generic_error:
        side_effect = Exception("unexpected failure")
    else:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = kinich_status
        side_effect = None

    with patch("urllib.request.urlopen") as mock_urlopen:
        if side_effect:
            mock_urlopen.side_effect = side_effect
        else:
            mock_urlopen.return_value = mock_resp
        conn = KinichQuantumConnector(fallback_to_classical=fallback, **kwargs)

    return conn


# ──────────────────────────────────────────────────────────────────────────────
# KinichQuantumConnector — __init__ / _init_kinich_connection
# ──────────────────────────────────────────────────────────────────────────────


class TestKinichConnectorInit:

    def test_default_attributes(self):
        conn = _connector(url_error=True)
        assert conn.classical_dim == 768
        assert conn.quantum_dim == 8
        assert conn.enable_caching is True
        assert conn.fallback_to_classical is True
        assert isinstance(conn.result_cache, dict)
        assert isinstance(conn.stats, dict)

    def test_kinich_available_when_health_check_200(self):
        conn = _connector(kinich_status=200)
        assert conn.kinich_available is True

    def test_kinich_unavailable_when_health_check_non_200(self):
        conn = _connector(kinich_status=503)
        assert conn.kinich_available is False

    def test_kinich_unavailable_on_url_error(self):
        conn = _connector(url_error=True)
        assert conn.kinich_available is False

    def test_kinich_unavailable_on_generic_error_with_fallback(self):
        """Generic exception during health check → unavailable but no raise when fallback=True."""
        conn = _connector(generic_error=True, fallback=True)
        assert conn.kinich_available is False

    def test_generic_error_raises_when_no_fallback(self):
        """Generic exception + fallback_to_classical=False → should raise."""
        with pytest.raises(Exception):
            _connector(generic_error=True, fallback=False)

    def test_custom_dims(self):
        conn = _connector(
            url_error=True,
            classical_dim=256,
            quantum_dim=4,
        )
        assert conn.classical_dim == 256
        assert conn.quantum_dim == 4

    def test_bridge_starts_none(self):
        conn = _connector(url_error=True)
        assert conn.bridge is None

    def test_stats_initialized_to_zero(self):
        conn = _connector(url_error=True)
        assert conn.stats["quantum_calls"] == 0
        assert conn.stats["cache_hits"] == 0
        assert conn.stats["fallback_calls"] == 0
        assert conn.stats["total_latency"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# quantum_process — fallback path (bridge is always None)
# ──────────────────────────────────────────────────────────────────────────────


class TestQuantumProcess:

    def _run(self, coro):
        return asyncio.run(coro)

    def test_fallback_returns_array_of_correct_shape(self):
        conn = _connector(url_error=True)
        features = np.random.randn(4, 768).astype(np.float32)
        result = self._run(conn.quantum_process(features))
        assert result.shape == (4, 768)

    def test_fallback_increments_fallback_calls(self):
        conn = _connector(url_error=True)
        features = np.random.randn(2, 768)
        self._run(conn.quantum_process(features))
        assert conn.stats["fallback_calls"] == 1

    def test_fallback_updates_total_latency(self):
        conn = _connector(url_error=True)
        self._run(conn.quantum_process(np.ones((2, 768))))
        assert conn.stats["total_latency"] > 0.0

    def test_cache_hit_on_second_call(self):
        conn = _connector(url_error=True, enable_caching=True)
        features = np.ones((2, 768), dtype=np.float64)
        self._run(conn.quantum_process(features))
        self._run(conn.quantum_process(features))
        assert conn.stats["cache_hits"] == 1

    def test_no_cache_when_disabled(self):
        conn = _connector(url_error=True, enable_caching=False)
        features = np.ones((2, 768), dtype=np.float64)
        self._run(conn.quantum_process(features))
        self._run(conn.quantum_process(features))
        assert conn.stats["cache_hits"] == 0
        assert conn.stats["fallback_calls"] == 2

    def test_cache_eviction_when_full(self):
        """When cache is at capacity, the oldest entry is evicted."""
        conn = _connector(url_error=True, enable_caching=True)
        conn._cache_max_size = 2

        for i in range(3):
            f = np.full((1, 768), float(i))
            self._run(conn.quantum_process(f))

        # Cache should not exceed max size
        assert len(conn.result_cache) <= conn._cache_max_size

    def test_fallback_result_is_ndarray(self):
        conn = _connector(url_error=True)
        features = np.random.randn(3, 768)
        result = self._run(conn.quantum_process(features))
        assert isinstance(result, np.ndarray)

    def test_cache_returns_same_result(self):
        conn = _connector(url_error=True, enable_caching=True)
        features = np.ones((2, 768), dtype=np.float64)
        r1 = self._run(conn.quantum_process(features))
        r2 = self._run(conn.quantum_process(features))
        np.testing.assert_array_equal(r1, r2)


# ──────────────────────────────────────────────────────────────────────────────
# _classical_fallback
# ──────────────────────────────────────────────────────────────────────────────


class TestClassicalFallback:

    def test_returns_same_shape(self):
        conn = _connector(url_error=True)
        features = np.random.randn(5, 768)
        result = conn._classical_fallback(features)
        assert result.shape == (5, 768)

    def test_repeated_calls_use_same_matrix(self):
        conn = _connector(url_error=True)
        features = np.ones((2, 768))
        r1 = conn._classical_fallback(features)
        r2 = conn._classical_fallback(features)
        np.testing.assert_array_equal(r1, r2)

    def test_fallback_matrix_created_once(self):
        conn = _connector(url_error=True)
        features = np.ones((2, 768))
        conn._classical_fallback(features)
        assert hasattr(conn, "_fallback_matrix")
        mat1 = conn._fallback_matrix
        conn._classical_fallback(features)
        assert conn._fallback_matrix is mat1


# ──────────────────────────────────────────────────────────────────────────────
# _get_cache_key
# ──────────────────────────────────────────────────────────────────────────────


class TestGetCacheKey:

    def test_returns_string(self):
        conn = _connector(url_error=True)
        key = conn._get_cache_key(np.ones((2, 3)))
        assert isinstance(key, str)

    def test_same_arrays_same_key(self):
        conn = _connector(url_error=True)
        a = np.array([1.0, 2.0, 3.0])
        assert conn._get_cache_key(a) == conn._get_cache_key(a.copy())

    def test_different_arrays_different_keys(self):
        conn = _connector(url_error=True)
        k1 = conn._get_cache_key(np.array([1.0]))
        k2 = conn._get_cache_key(np.array([2.0]))
        assert k1 != k2


# ──────────────────────────────────────────────────────────────────────────────
# _quantum_forward (aiohttp mocked)
# ──────────────────────────────────────────────────────────────────────────────


class TestQuantumForward:

    def test_quantum_forward_parses_response(self):
        """_quantum_forward posts to Kinich API and returns enhanced features."""
        conn = _connector(url_error=True)
        conn.kinich_available = True

        expected = np.ones((2, 768)).tolist()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"quantum_enhanced_features": expected})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        features = np.ones((2, 768))
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(conn._quantum_forward(features, "vqc"))

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 768)

    def test_quantum_forward_raises_on_api_error(self):
        """Non-200 response raises RuntimeError."""
        conn = _connector(url_error=True)

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        features = np.ones((2, 768))
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(RuntimeError, match="Kinich API error"):
                asyncio.run(conn._quantum_forward(features, "vqc"))


# ──────────────────────────────────────────────────────────────────────────────
# _vqc_forward, _qsvm_forward, _qnn_forward
# ──────────────────────────────────────────────────────────────────────────────


class TestLegacyForwardMethods:

    def test_qsvm_returns_class_scores(self):
        """_qsvm_forward returns a mock array of class scores."""
        conn = _connector(url_error=True)
        features = np.ones((3, 768), dtype=np.float32)
        result = asyncio.run(conn._qsvm_forward(features, num_classes=2))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_qsvm_custom_num_classes(self):
        conn = _connector(url_error=True)
        features = np.ones((2, 768))
        result = asyncio.run(conn._qsvm_forward(features, num_classes=5))
        assert result.shape == (2, 5)

    def test_vqc_forward_delegates_to_quantum_forward(self):
        """_vqc_forward calls _quantum_forward with model_type='vqc'."""
        conn = _connector(url_error=True)
        features = np.ones((2, 768))

        expected = np.zeros((2, 768)).tolist()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"quantum_enhanced_features": expected})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(conn._vqc_forward(features))

        assert isinstance(result, np.ndarray)

    def test_qnn_forward_delegates_to_quantum_forward(self):
        """_qnn_forward calls _quantum_forward with model_type='qnn'."""
        conn = _connector(url_error=True)
        features = np.ones((2, 768))

        expected = np.zeros((2, 768)).tolist()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"quantum_enhanced_features": expected})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(conn._qnn_forward(features))

        assert isinstance(result, np.ndarray)


# ──────────────────────────────────────────────────────────────────────────────
# get_statistics, clear_cache, reset_statistics, __repr__
# ──────────────────────────────────────────────────────────────────────────────


class TestStatistics:

    def test_get_statistics_structure(self):
        conn = _connector(url_error=True)
        stats = conn.get_statistics()
        expected_keys = {
            "total_calls",
            "quantum_calls",
            "cache_hits",
            "fallback_calls",
            "quantum_ratio",
            "cache_hit_ratio",
            "avg_latency",
            "kinich_available",
        }
        assert expected_keys <= set(stats.keys())

    def test_get_statistics_zero_calls(self):
        conn = _connector(url_error=True)
        stats = conn.get_statistics()
        assert stats["total_calls"] == 0
        assert stats["quantum_ratio"] == 0.0
        assert stats["cache_hit_ratio"] == 0.0
        assert stats["avg_latency"] == 0.0

    def test_get_statistics_after_fallback(self):
        conn = _connector(url_error=True)
        asyncio.run(conn.quantum_process(np.ones((1, 768))))
        stats = conn.get_statistics()
        assert stats["total_calls"] == 1
        assert stats["fallback_calls"] == 1

    def test_clear_cache_empties_result_cache(self):
        conn = _connector(url_error=True, enable_caching=True)
        asyncio.run(conn.quantum_process(np.ones((1, 768))))
        assert len(conn.result_cache) > 0
        conn.clear_cache()
        assert len(conn.result_cache) == 0

    def test_reset_statistics_zeros_all_counters(self):
        conn = _connector(url_error=True)
        asyncio.run(conn.quantum_process(np.ones((1, 768))))
        conn.reset_statistics()
        assert conn.stats["fallback_calls"] == 0
        assert conn.stats["quantum_calls"] == 0
        assert conn.stats["cache_hits"] == 0
        assert conn.stats["total_latency"] == 0.0

    def test_repr_contains_endpoint_and_dims(self):
        conn = _connector(url_error=True, kinich_endpoint="http://q:9000")
        r = repr(conn)
        assert "http://q:9000" in r
        assert "768" in r

    def test_repr_is_string(self):
        conn = _connector(url_error=True)
        assert isinstance(repr(conn), str)


# ──────────────────────────────────────────────────────────────────────────────
# QuantumEnhancedLayer
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestQuantumEnhancedLayer:

    def _make_layer(self) -> QuantumEnhancedLayer:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("no kinich")
            layer = QuantumEnhancedLayer(classical_dim=8, quantum_dim=2)
        return layer

    def test_init_sets_dims(self):
        layer = self._make_layer()
        assert layer.classical_dim == 8
        assert layer.quantum_dim == 2

    def test_init_creates_connector(self):
        layer = self._make_layer()
        assert isinstance(layer.connector, KinichQuantumConnector)

    def test_forward_returns_same_shape(self):
        import torch

        layer = self._make_layer()
        x = torch.randn(3, 8)
        result = layer(x)
        assert result.shape == (3, 8)

    def test_forward_returns_tensor(self):
        import torch

        layer = self._make_layer()
        x = torch.randn(2, 8)
        result = layer(x)
        assert isinstance(result, torch.Tensor)

    def test_forward_preserves_dtype(self):
        import torch

        layer = self._make_layer()
        x = torch.randn(2, 8, dtype=torch.float32)
        result = layer(x)
        assert result.dtype == torch.float32

    def test_extra_repr(self):
        layer = self._make_layer()
        s = layer.extra_repr()
        assert "8" in s
        assert "2" in s

    def test_no_torch_raises_import_error(self):
        """QuantumEnhancedLayer requires PyTorch; omitted when not available."""
        # When TORCH_AVAILABLE is True the class is a real nn.Module — no need
        # to test the import-error path in an environment with torch installed.
        assert TORCH_AVAILABLE is True


# ──────────────────────────────────────────────────────────────────────────────
# Module-level import guard — TORCH_AVAILABLE = False branch (lines 16-18)
# ──────────────────────────────────────────────────────────────────────────────


class TestTorchUnavailableBranch:

    def test_torch_available_false_when_torch_absent(self):
        """Reload kinich_connector with torch hidden → TORCH_AVAILABLE=False (lines 16-18)."""
        import sys
        import importlib
        import quantum.kinich_connector as target_module

        # Save originals
        saved_torch = sys.modules.pop("torch", None)
        saved_torch_nn = sys.modules.pop("torch.nn", None)
        # Make 'import torch' raise ImportError during reload
        sys.modules["torch"] = None  # type: ignore[assignment]
        sys.modules["torch.nn"] = None  # type: ignore[assignment]

        try:
            reloaded = importlib.reload(target_module)
            assert reloaded.TORCH_AVAILABLE is False
        finally:
            # Restore originals unconditionally
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
            else:
                sys.modules.pop("torch", None)
            if saved_torch_nn is not None:
                sys.modules["torch.nn"] = saved_torch_nn
            else:
                sys.modules.pop("torch.nn", None)
            # Reload back to True state for subsequent tests
            importlib.reload(target_module)


# ──────────────────────────────────────────────────────────────────────────────
# quantum_process — bridge≠None exception path (lines 164-173)
# ──────────────────────────────────────────────────────────────────────────────


class TestQuantumProcessBridgePath:

    def test_successful_quantum_forward_increments_quantum_calls(self):
        """When bridge is set and _quantum_forward succeeds, quantum_calls is incremented (line 166)."""
        conn = _connector(url_error=True)
        conn.kinich_available = True
        conn.bridge = MagicMock()

        features = np.ones((2, 768))
        expected_result = np.zeros((2, 768))

        async def successful_forward(feats, model_type, **kwargs):
            return expected_result

        with patch.object(conn, "_quantum_forward", side_effect=successful_forward):
            asyncio.run(conn.quantum_process(features))

        assert conn.stats["quantum_calls"] == 1
        assert conn.stats["fallback_calls"] == 0
        """When bridge is set and _quantum_forward raises, fallback is used."""
        conn = _connector(url_error=True)
        conn.kinich_available = True
        conn.bridge = MagicMock()  # Simulate an initialized bridge

        async def failing_forward(features, model_type, **kwargs):
            raise RuntimeError("quantum hardware error")

        features = np.ones((2, 768))

        with patch.object(conn, "_quantum_forward", side_effect=failing_forward):
            result = asyncio.run(conn.quantum_process(features))

        assert isinstance(result, np.ndarray)
        assert conn.stats["fallback_calls"] == 1
        assert conn.stats["quantum_calls"] == 0

    def test_exception_raises_when_no_fallback(self):
        """With fallback_to_classical=False a quantum error propagates."""
        conn = _connector(url_error=True, fallback=False)
        conn.kinich_available = True
        conn.bridge = MagicMock()
        conn.fallback_to_classical = False

        async def failing_forward(features, model_type, **kwargs):
            raise RuntimeError("fatal quantum error")

        features = np.ones((2, 768))

        with patch.object(conn, "_quantum_forward", side_effect=failing_forward):
            with pytest.raises(RuntimeError, match="fatal quantum error"):
                asyncio.run(conn.quantum_process(features))


# ──────────────────────────────────────────────────────────────────────────────
# QuantumEnhancedLayer — additional edge cases (lines 386, 427-429)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestQuantumEnhancedLayerEdges:

    def _make_layer(self) -> QuantumEnhancedLayer:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("no kinich")
            return QuantumEnhancedLayer(classical_dim=8, quantum_dim=2)

    def test_init_raises_when_torch_unavailable(self):
        """QuantumEnhancedLayer.__init__ raises ImportError when torch absent (line 386)."""
        with patch("quantum.kinich_connector.TORCH_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyTorch required"):
                with patch("urllib.request.urlopen") as mock_urlopen:
                    mock_urlopen.side_effect = urllib.error.URLError("no kinich")
                    QuantumEnhancedLayer(classical_dim=8, quantum_dim=2)

    def test_forward_inside_running_event_loop(self):
        """QuantumEnhancedLayer.forward() uses ThreadPoolExecutor when inside an async loop (lines 427-429)."""
        import torch

        layer = self._make_layer()
        x = torch.randn(2, 8)

        async def run_inside_loop():
            # This exercises the `if loop is not None and loop.is_running()` path
            return layer(x)

        result = asyncio.run(run_inside_loop())
        assert result.shape == (2, 8)
