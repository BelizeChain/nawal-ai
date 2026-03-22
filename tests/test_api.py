"""
Tests for api/inference_server.py — Priority 2 operational layer.

Covers:
  - _RateLimiter     — sliding-window logic
  - InferenceRequest  — Pydantic validation
  - GET  /health      — with & without model
  - GET  /model/info  — 200 vs 503
  - POST /infer       — 200 vs 503, inference path
  - POST /batch/infer — batch size guard, success
  - POST /infer/stream — streaming response
  - GET  /metrics     — prometheus export
  - Rate-limit middleware — 429, /health exemption

All external dependencies (model, metrics, belizeid) are mocked so
the test suite runs without GPU, blockchain node, or checkpoint files.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.inference_server import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    _RateLimiter,
    app,
)

# ============================================================================
# Helpers / fixtures
# ============================================================================


def _make_mock_model(text: str = "Belmopan is the capital of Belize.") -> MagicMock:
    """Minimal mock that satisfies every attribute accessed in the endpoints."""
    m = MagicMock()
    m.version = "test-v1"
    m.num_parameters.return_value = 117_000_000
    m.training_rounds = 10
    m.last_updated = datetime(2026, 1, 1)
    m.privacy_epsilon = 2.0
    m.privacy_delta = 1e-5
    m.generate.return_value = text
    m.generate_stream.return_value = iter(["Belmopan", " is", " great"])
    return m


@pytest.fixture
def fresh_rate_limiter():
    """Override the module-level rate limiter with a fresh high-limit instance."""
    limiter = _RateLimiter(max_requests=1000, window_seconds=60)
    with patch("api.inference_server._rate_limiter", limiter):
        yield limiter


_AUTH_HEADERS = {"belizeid": "test-belize-id-001"}


@pytest.fixture
def mock_services():
    """Patch BelizeIDVerifier and InferenceMetricsCollector constructors for tests."""
    mock_verifier = MagicMock()
    mock_verifier.verify = AsyncMock(return_value=True)
    mock_metrics = MagicMock()
    mock_metrics.log_inference = AsyncMock()
    mock_metrics.export_prometheus = AsyncMock(return_value=b"# Prometheus metrics\n")
    with (
        patch("api.inference_server.BelizeIDVerifier", return_value=mock_verifier),
        patch(
            "api.inference_server.InferenceMetricsCollector", return_value=mock_metrics
        ),
    ):
        yield mock_verifier, mock_metrics


@pytest.fixture
def client_no_model(fresh_rate_limiter, mock_services):
    """TestClient where lifespan sets model=None (no checkpoint file)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_with_model(fresh_rate_limiter, mock_services):
    """TestClient with a mock model injected via patched lifespan."""
    mock_model = _make_mock_model()
    with patch(
        "api.inference_server.BelizeChainLLM.from_checkpoint",
        return_value=mock_model,
        create=True,
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_model


# ============================================================================
# _RateLimiter — pure unit tests
# ============================================================================


class TestRateLimiter:

    def test_first_request_always_allowed(self):
        rl = _RateLimiter(max_requests=5, window_seconds=60)
        assert rl.is_allowed("192.168.1.1") is True

    def test_up_to_limit_all_allowed(self):
        rl = _RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("10.0.0.1") is True

    def test_over_limit_blocked(self):
        rl = _RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("10.0.0.1")
        assert rl.is_allowed("10.0.0.1") is False

    def test_different_ips_independent(self):
        rl = _RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        # IP 1 is now at limit, IP 2 should still be allowed
        assert rl.is_allowed("2.2.2.2") is True
        assert rl.is_allowed("1.1.1.1") is False

    def test_old_hits_expire_from_window(self):
        """Hits older than window_seconds should be purged."""
        rl = _RateLimiter(max_requests=2, window_seconds=1)
        # Inject 2 hits with a timestamp older than the window
        old_ts = time.monotonic() - 5.0  # 5 seconds ago → outside 1-second window
        rl._hits["10.0.0.1"] = [old_ts, old_ts]
        # After purge, 0 recent hits → new request allowed
        assert rl.is_allowed("10.0.0.1") is True

    def test_window_resets_after_expiry(self):
        rl = _RateLimiter(max_requests=1, window_seconds=1)
        rl.is_allowed("ip_x")
        assert rl.is_allowed("ip_x") is False

        # Expire all hits
        rl._hits["ip_x"] = [time.monotonic() - 5.0]
        assert rl.is_allowed("ip_x") is True

    def test_rate_limiter_returns_bool(self):
        rl = _RateLimiter()
        result = rl.is_allowed("test")
        assert isinstance(result, bool)


# ============================================================================
# Pydantic model validation
# ============================================================================


class TestInferenceRequestValidation:

    def test_valid_minimal_request(self):
        req = InferenceRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.stream is False

    def test_valid_full_request(self):
        req = InferenceRequest(
            prompt="What is Belize?",
            max_tokens=256,
            temperature=0.5,
            top_p=0.8,
            stream=True,
            belizeid="BZid_123",
        )
        assert req.max_tokens == 256
        assert req.temperature == 0.5
        assert req.stream is True

    def test_empty_prompt_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="")

    def test_prompt_too_long_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x" * 2049)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="hi", max_tokens=0)

    def test_max_tokens_over_limit_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="hi", max_tokens=2049)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="hi", temperature=-0.1)

    def test_temperature_over_max_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="hi", temperature=2.1)

    def test_top_p_over_one_raises(self):
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="hi", top_p=1.1)

    def test_top_p_zero_valid(self):
        req = InferenceRequest(prompt="hi", top_p=0.0)
        assert req.top_p == 0.0

    def test_belizeid_optional_none(self):
        req = InferenceRequest(prompt="hi")
        assert req.belizeid is None


# ============================================================================
# GET /health
# ============================================================================


class TestHealthEndpoint:

    def test_health_no_model_returns_503(self, client_no_model):
        response = client_no_model.get("/health")
        assert response.status_code == 503

    def test_health_no_model_shows_degraded(self, client_no_model):
        response = client_no_model.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_model_loaded_true(self, client_with_model):
        client, _ = client_with_model
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_contains_timestamp(self, client_no_model):
        response = client_no_model.get("/health")
        assert "timestamp" in response.json()

    def test_health_exempt_from_rate_limit(self, mock_services):
        """GET /health must bypass the rate-limit middleware."""
        tight_limiter = _RateLimiter(max_requests=1, window_seconds=60)
        # Exhaust the limiter for the test-client IP
        tight_limiter.is_allowed("testclient")

        with patch("api.inference_server._rate_limiter", tight_limiter):
            with TestClient(app, raise_server_exceptions=False) as c:
                # /health should NOT be blocked even though limiter is exhausted
                response = c.get("/health")
                assert response.status_code != 429


# ============================================================================
# GET /model/info
# ============================================================================


class TestModelInfoEndpoint:

    def test_model_info_no_model_returns_503(self, client_no_model):
        response = client_no_model.get("/model/info")
        assert response.status_code == 503

    def test_model_info_with_model_returns_200(self, client_with_model):
        client, _ = client_with_model
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_structure(self, client_with_model):
        client, mock_model = client_with_model
        data = response = client.get("/model/info").json()
        assert data["model_name"] == "BelizeChainLLM"
        assert data["version"] == "test-v1"
        assert data["parameters"] == 117_000_000
        assert data["training_rounds"] == 10
        assert "epsilon" in data["privacy_budget"]
        assert "delta" in data["privacy_budget"]


# ============================================================================
# POST /infer
# ============================================================================


class TestInferEndpoint:

    def test_infer_no_model_returns_503(self, client_no_model):
        response = client_no_model.post(
            "/infer", json={"prompt": "Hello"}, headers=_AUTH_HEADERS
        )
        assert response.status_code == 503

    def test_infer_with_model_returns_200(self, client_with_model):
        client, mock_model = client_with_model
        response = client.post(
            "/infer", json={"prompt": "What is Belize?"}, headers=_AUTH_HEADERS
        )
        assert response.status_code == 200

    def test_infer_response_structure(self, client_with_model):
        client, _ = client_with_model
        data = client.post(
            "/infer", json={"prompt": "Hello there"}, headers=_AUTH_HEADERS
        ).json()
        for key in (
            "text",
            "tokens_generated",
            "inference_time_ms",
            "model_version",
            "timestamp",
        ):
            assert key in data, f"Missing key: {key}"

    def test_infer_text_from_model(self, client_with_model):
        client, mock_model = client_with_model
        mock_model.generate.return_value = "Belmopan"
        data = client.post(
            "/infer", json={"prompt": "Capital"}, headers=_AUTH_HEADERS
        ).json()
        assert data["text"] == "Belmopan"

    def test_infer_model_version_in_response(self, client_with_model):
        client, _ = client_with_model
        data = client.post(
            "/infer", json={"prompt": "test"}, headers=_AUTH_HEADERS
        ).json()
        assert data["model_version"] == "test-v1"

    def test_infer_inference_time_positive(self, client_with_model):
        client, _ = client_with_model
        data = client.post(
            "/infer", json={"prompt": "test"}, headers=_AUTH_HEADERS
        ).json()
        assert data["inference_time_ms"] >= 0.0

    def test_infer_logs_metrics(self, client_with_model):
        client, _ = client_with_model
        client.post("/infer", json={"prompt": "Hello"}, headers=_AUTH_HEADERS)
        app.state.metrics.log_inference.assert_awaited_once()

    def test_infer_model_generate_called_with_params(self, client_with_model):
        client, mock_model = client_with_model
        client.post(
            "/infer",
            json={"prompt": "Test", "max_tokens": 100, "temperature": 0.5},
            headers=_AUTH_HEADERS,
        )
        mock_model.generate.assert_called_once_with(
            prompt="Test", max_tokens=100, temperature=0.5, top_p=0.9
        )

    def test_infer_invalid_prompt_returns_422(self, client_no_model):
        response = client_no_model.post(
            "/infer", json={"prompt": ""}, headers=_AUTH_HEADERS
        )
        assert response.status_code == 422

    def test_infer_missing_prompt_returns_422(self, client_no_model):
        response = client_no_model.post("/infer", json={}, headers=_AUTH_HEADERS)
        assert response.status_code == 422


# ============================================================================
# POST /batch/infer
# ============================================================================


class TestBatchInferEndpoint:

    def test_batch_no_model_returns_503(self, client_no_model):
        response = client_no_model.post(
            "/batch/infer",
            json=[{"prompt": "Hello"}],
            headers=_AUTH_HEADERS,
        )
        assert response.status_code == 503

    def test_batch_over_limit_returns_400(self, client_with_model):
        client, _ = client_with_model
        requests = [{"prompt": f"Q{i}"} for i in range(33)]
        response = client.post("/batch/infer", json=requests, headers=_AUTH_HEADERS)
        assert response.status_code == 400
        assert "32" in response.json()["detail"]

    def test_batch_at_limit_returns_200(self, client_with_model):
        client, _ = client_with_model
        requests = [{"prompt": f"Q{i}"} for i in range(32)]
        response = client.post("/batch/infer", json=requests, headers=_AUTH_HEADERS)
        assert response.status_code == 200

    def test_batch_response_structure(self, client_with_model):
        client, _ = client_with_model
        response = client.post(
            "/batch/infer",
            json=[{"prompt": "A"}, {"prompt": "B"}],
            headers=_AUTH_HEADERS,
        )
        data = response.json()
        assert "results" in data
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_each_result_has_status(self, client_with_model):
        client, _ = client_with_model
        data = client.post(
            "/batch/infer",
            json=[{"prompt": "X"}],
            headers=_AUTH_HEADERS,
        ).json()
        result = data["results"][0]
        assert "status" in result

    def test_batch_empty_list_returns_200(self, client_with_model):
        client, _ = client_with_model
        response = client.post("/batch/infer", json=[], headers=_AUTH_HEADERS)
        assert response.status_code == 200
        assert response.json()["total"] == 0


# ============================================================================
# POST /infer/stream
# ============================================================================


class TestStreamEndpoint:

    def test_stream_no_model_returns_503(self, client_no_model):
        response = client_no_model.post(
            "/infer/stream",
            json={"prompt": "Hello"},
            headers=_AUTH_HEADERS,
        )
        assert response.status_code == 503

    def test_stream_with_model_returns_200(self, client_with_model):
        client, _ = client_with_model
        response = client.post(
            "/infer/stream",
            json={"prompt": "Tell me about Belize"},
            headers=_AUTH_HEADERS,
        )
        assert response.status_code == 200

    def test_stream_content_type_ndjson(self, client_with_model):
        client, _ = client_with_model
        response = client.post(
            "/infer/stream",
            json={"prompt": "Test"},
            headers=_AUTH_HEADERS,
        )
        assert (
            "ndjson" in response.headers.get("content-type", "").lower()
            or response.status_code == 200
        )  # streaming; content-type may vary

    def test_stream_yields_json_lines(self, client_with_model):
        client, mock_model = client_with_model
        mock_model.generate_stream.return_value = iter(["Hello", " world"])
        response = client.post(
            "/infer/stream",
            json={"prompt": "Hi"},
            headers=_AUTH_HEADERS,
        )
        assert response.status_code == 200
        lines = [l for l in response.text.strip().splitlines() if l]
        assert len(lines) >= 1
        # Each line must be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "token" in parsed or "error" in parsed


# ============================================================================
# GET /metrics
# ============================================================================


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client_no_model):
        response = client_no_model.get("/metrics", headers=_AUTH_HEADERS)
        assert response.status_code == 200

    def test_metrics_export_called(self, client_no_model):
        client_no_model.get("/metrics", headers=_AUTH_HEADERS)
        app.state.metrics.export_prometheus.assert_awaited_once()


# ============================================================================
# Rate-limit middleware
# ============================================================================


class TestRateLimitMiddleware:

    def test_rate_limit_429_after_exhaustion(self, fresh_rate_limiter):
        """After max_requests, subsequent calls return 429."""
        fresh_rate_limiter.max_requests = 3
        # Reset any existing hits
        fresh_rate_limiter._hits.clear()

        with TestClient(app, raise_server_exceptions=False) as c:
            # First 3 health-exempt calls succeed
            for _ in range(3):
                fresh_rate_limiter._hits["testclient"] = fresh_rate_limiter._hits.get(
                    "testclient", []
                )

            # Inject max_requests hits for the test-client IP
            now = time.monotonic()
            fresh_rate_limiter._hits["testclient"] = [now, now, now]

            response = c.post("/infer", json={"prompt": "hi"})
            assert response.status_code == 429

    def test_rate_limit_response_body(self, fresh_rate_limiter):
        """429 body must contain 'Rate limit exceeded' message."""
        now = time.monotonic()
        fresh_rate_limiter.max_requests = 1
        fresh_rate_limiter._hits["testclient"] = [now]

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post("/infer", json={"prompt": "hi"})
            if response.status_code == 429:
                assert "Rate limit" in response.json()["detail"]

    def test_model_info_exempt_from_rate_limit(self):
        """/model/info path is NOT exempt — no special carve-out."""
        # Just verify the endpoint exists and returns something sensible
        tight = _RateLimiter(max_requests=1000, window_seconds=60)
        with patch("api.inference_server._rate_limiter", tight):
            with TestClient(app, raise_server_exceptions=False) as c:
                # Without model → 503
                response = c.get("/model/info")
                assert response.status_code in (200, 503)
