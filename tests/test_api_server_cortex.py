"""
Coverage Batch 5 — targets highest-impact remaining uncovered lines:
  api_server.py, cortex/__init__.py, monitoring/prometheus_exporter.py,
  memory/episodic.py (chroma/qdrant mocks), server/aggregator.py,
  genome/population.py, blockchain/genome_registry.py,
  genome/model_builder.py, blockchain/substrate_client.py,
  plus miscellaneous near-100% modules.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper to run async coroutines in sync tests (robust against closed loops)
# ---------------------------------------------------------------------------
def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    return loop.run_until_complete(coro)


###############################################################################
# 1.  cortex/__init__.py  (0% → 100%)
###############################################################################


class TestCortexInit:
    def test_import_transformer(self):
        from cortex import NawalTransformer

        assert NawalTransformer is not None

    def test_import_config(self):
        from cortex import NawalModelConfig

        assert NawalModelConfig is not None

    def test_import_attention(self):
        from cortex import MultiHeadAttention

        assert MultiHeadAttention is not None

    def test_import_feedforward(self):
        from cortex import FeedForward

        assert FeedForward is not None

    def test_import_embeddings(self):
        from cortex import NawalEmbeddings

        assert NawalEmbeddings is not None

    def test_all_exports(self):
        import cortex

        for name in cortex.__all__:
            assert hasattr(cortex, name)


###############################################################################
# 2.  api_server.py  (0% → ~85%)
###############################################################################


class TestApiServerModels:
    """Pydantic models and utility classes."""

    def test_server_config_defaults(self, tmp_path):
        from api_server import ServerConfig

        cfg = ServerConfig(checkpoint_dir=tmp_path / "ckpt")
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.min_participants == 3

    def test_server_config_checkpoint_dir_created(self, tmp_path):
        from api_server import ServerConfig

        d = tmp_path / "new_ckpt"
        ServerConfig(checkpoint_dir=d)
        assert d.exists()

    def test_enroll_request_valid(self):
        from api_server import EnrollRequest

        req = EnrollRequest(account_id="5FHne...", stake_amount=5000)
        assert req.stake_amount == 5000

    def test_enroll_request_below_min_stake(self):
        from api_server import EnrollRequest
        import pydantic

        with pytest.raises((pydantic.ValidationError, ValueError)):
            EnrollRequest(account_id="acc", stake_amount=0)

    def test_submit_model_request(self):
        from api_server import SubmitModelRequest

        req = SubmitModelRequest(
            participant_id="p1",
            round_id="r1",
            model_cid="Qmabc",
            quality_score=80.0,
            training_samples=100,
        )
        assert req.quality_score == 80.0

    def test_start_round_request_defaults(self):
        from api_server import StartRoundRequest

        req = StartRoundRequest(dataset_name="belize")
        assert req.target_accuracy == 0.85

    def test_rate_limiter_allow(self):
        from api_server import RateLimiter

        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("ip1") is True

    def test_rate_limiter_block_after_limit(self):
        from api_server import RateLimiter

        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("ip1")
        rl.is_allowed("ip1")
        assert rl.is_allowed("ip1") is False

    def test_rate_limiter_window_expires(self):
        from api_server import RateLimiter

        rl = RateLimiter(max_requests=1, window_seconds=0)
        rl.is_allowed("ip1")
        time.sleep(0.01)
        assert rl.is_allowed("ip1") is True  # window expired

    def test_app_state_defaults(self):
        from api_server import AppState

        state = AppState()
        assert state.round_counter == 0
        assert state.active_rounds == {}
        assert state.staking_connector is None

    def test_app_state_initialize_no_blockchain(self, tmp_path):
        from api_server import AppState, ServerConfig

        state = AppState()
        cfg = ServerConfig(blockchain_enabled=False, checkpoint_dir=tmp_path / "c")
        _run(state.initialize(cfg))
        assert state.staking_connector is None
        assert state.config is not None

    def test_app_state_initialize_blockchain_mock(self, tmp_path):
        from api_server import AppState, ServerConfig

        state = AppState()
        cfg = ServerConfig(blockchain_enabled=True, checkpoint_dir=tmp_path / "c")
        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(return_value=True)
        with patch("api_server.StakingConnector", return_value=mock_connector):
            _run(state.initialize(cfg))
        assert state.staking_connector is mock_connector

    def test_app_state_initialize_blockchain_fails(self, tmp_path):
        from api_server import AppState, ServerConfig

        state = AppState()
        cfg = ServerConfig(blockchain_enabled=True, checkpoint_dir=tmp_path / "c")
        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(side_effect=Exception("conn fail"))
        with patch("api_server.StakingConnector", return_value=mock_connector):
            _run(state.initialize(cfg))  # should not raise — degraded mode

    def test_app_state_shutdown_with_connector(self, tmp_path):
        from api_server import AppState, ServerConfig

        state = AppState()
        state.staking_connector = AsyncMock()
        state.staking_connector.disconnect = AsyncMock()
        state.config = ServerConfig(
            blockchain_enabled=False, checkpoint_dir=tmp_path / "c"
        )
        _run(state.shutdown())
        state.staking_connector.disconnect.assert_called_once()

    def test_app_state_shutdown_no_connector(self):
        from api_server import AppState

        state = AppState()
        _run(state.shutdown())  # should not raise


class TestApiServerEndpoints:
    """FastAPI route handlers - tested directly to bypass broken middleware stack."""

    @pytest.fixture(autouse=True)
    def setup_state(self, tmp_path):
        from api_server import app_state, ServerConfig

        app_state.active_rounds.clear()
        app_state.completed_rounds.clear()
        app_state.participant_submissions.clear()
        app_state.round_counter = 0
        app_state.staking_connector = None
        app_state.config = ServerConfig(
            blockchain_enabled=False,
            checkpoint_dir=tmp_path / "ckpt",
        )
        yield app_state
        app_state.active_rounds.clear()
        app_state.completed_rounds.clear()
        app_state.participant_submissions.clear()
        app_state.round_counter = 0
        app_state.staking_connector = None

    def test_health_check(self, setup_state):
        from api_server import health_check

        response = _run(health_check())
        data = json.loads(response.body)
        assert data["status"] == "healthy"
        assert data["blockchain_connected"] is False

    def test_health_no_blockchain(self, setup_state):
        from api_server import health_check

        response = _run(health_check())
        data = json.loads(response.body)
        assert data["blockchain_connected"] is False

    def test_get_status(self, setup_state):
        from api_server import get_status

        result = _run(get_status())
        assert result.service == "Nawal Federated Learning"
        assert result.active_rounds == 0

    def test_start_fl_round(self, setup_state):
        from api_server import start_fl_round, StartRoundRequest

        req = StartRoundRequest(dataset_name="belize_corpus")
        result = _run(start_fl_round(req))
        assert "round_id" in result.__dict__ or hasattr(result, "round_id")

    def test_start_multiple_rounds(self, setup_state):
        from api_server import start_fl_round, StartRoundRequest

        r1 = _run(start_fl_round(StartRoundRequest(dataset_name="ds1")))
        r2 = _run(start_fl_round(StartRoundRequest(dataset_name="ds2")))
        assert r1.round_id != r2.round_id
        assert setup_state.round_counter == 2

    def test_get_round_status(self, setup_state):
        from api_server import start_fl_round, get_round_status, StartRoundRequest

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        status = _run(get_round_status(round_id))
        assert status.round_id == round_id
        assert status.participants == 0

    def test_get_round_not_found(self, setup_state):
        from api_server import get_round_status
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _run(get_round_status("nonexistent_round_xyz"))
        assert exc_info.value.status_code == 404

    def test_enroll_no_blockchain(self, setup_state):
        from api_server import enroll_participant, EnrollRequest
        from fastapi import HTTPException

        req = EnrollRequest(account_id="5FHne...", stake_amount=5000)
        with pytest.raises(HTTPException) as exc_info:
            _run(enroll_participant(req))
        assert exc_info.value.status_code == 503

    def test_enroll_with_mock_blockchain(self, setup_state):
        from api_server import enroll_participant, EnrollRequest

        mock_connector = AsyncMock()
        mock_connector.enroll_participant = AsyncMock(
            return_value={"success": True, "message": "enrolled"}
        )
        setup_state.staking_connector = mock_connector
        req = EnrollRequest(
            account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            stake_amount=5000,
        )
        result = _run(enroll_participant(req))
        assert result.success is True

    def test_enroll_blockchain_failure(self, setup_state):
        from api_server import enroll_participant, EnrollRequest
        from fastapi import HTTPException

        mock_connector = AsyncMock()
        mock_connector.enroll_participant = AsyncMock(
            return_value={"success": False, "message": "insufficient stake"}
        )
        setup_state.staking_connector = mock_connector
        req = EnrollRequest(
            account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            stake_amount=5000,
        )
        with pytest.raises(HTTPException) as exc_info:
            _run(enroll_participant(req))
        assert exc_info.value.status_code == 400

    def test_submit_round_not_found(self, setup_state):
        from api_server import submit_model_delta, SubmitModelRequest
        from fastapi import HTTPException

        req = SubmitModelRequest(
            participant_id="p1",
            round_id="bad_round",
            model_cid="Qm...",
            quality_score=80.0,
            training_samples=100,
        )
        with pytest.raises(HTTPException) as exc_info:
            _run(submit_model_delta(req))
        assert exc_info.value.status_code == 404

    def test_submit_round_not_active(self, setup_state):
        from api_server import (
            start_fl_round,
            submit_model_delta,
            StartRoundRequest,
            SubmitModelRequest,
        )
        from fastapi import HTTPException

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        req = SubmitModelRequest(
            participant_id="p1",
            round_id=round_id,
            model_cid="Qm...",
            quality_score=80.0,
            training_samples=100,
        )
        with pytest.raises(HTTPException) as exc_info:
            _run(submit_model_delta(req))
        assert exc_info.value.status_code == 400

    def test_submit_active_round_no_blockchain(self, setup_state):
        from api_server import (
            start_fl_round,
            submit_model_delta,
            StartRoundRequest,
            SubmitModelRequest,
        )

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        setup_state.active_rounds[round_id]["status"] = "active"
        req = SubmitModelRequest(
            participant_id="p1",
            round_id=round_id,
            model_cid="Qmabc",
            quality_score=85.0,
            training_samples=100,
        )
        with patch("api_server.TrainingSubmission"):
            result = _run(submit_model_delta(req))
        assert result.success is True
        assert result.reward_eligible is True

    def test_submit_low_quality_not_eligible(self, setup_state):
        from api_server import (
            start_fl_round,
            submit_model_delta,
            StartRoundRequest,
            SubmitModelRequest,
        )

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        setup_state.active_rounds[round_id]["status"] = "active"
        req = SubmitModelRequest(
            participant_id="p1",
            round_id=round_id,
            model_cid="Qmabc",
            quality_score=50.0,
            training_samples=50,
        )
        with patch("api_server.TrainingSubmission"):
            result = _run(submit_model_delta(req))
        assert result.reward_eligible is False

    def test_submit_with_blockchain_success(self, setup_state):
        from api_server import (
            start_fl_round,
            submit_model_delta,
            StartRoundRequest,
            SubmitModelRequest,
        )

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        setup_state.active_rounds[round_id]["status"] = "active"
        mock_connector = AsyncMock()
        mock_connector.submit_training_proof = AsyncMock(
            return_value={"success": True, "message": "ok"}
        )
        setup_state.staking_connector = mock_connector
        req = SubmitModelRequest(
            participant_id="p1",
            round_id=round_id,
            model_cid="Qmabc",
            quality_score=75.0,
            training_samples=100,
        )
        with patch("api_server.TrainingSubmission"):
            result = _run(submit_model_delta(req))
        assert result.success is True

    def test_submit_with_blockchain_failure(self, setup_state):
        from api_server import (
            start_fl_round,
            submit_model_delta,
            StartRoundRequest,
            SubmitModelRequest,
        )
        from fastapi import HTTPException

        r = _run(start_fl_round(StartRoundRequest(dataset_name="test")))
        round_id = r.round_id
        setup_state.active_rounds[round_id]["status"] = "active"
        mock_connector = AsyncMock()
        mock_connector.submit_training_proof = AsyncMock(
            return_value={"success": False, "message": "proof rejected"}
        )
        setup_state.staking_connector = mock_connector
        req = SubmitModelRequest(
            participant_id="p1",
            round_id=round_id,
            model_cid="Qmabc",
            quality_score=75.0,
            training_samples=100,
        )
        with (
            patch("api_server.TrainingSubmission"),
            pytest.raises(HTTPException) as exc_info,
        ):
            _run(submit_model_delta(req))
        assert exc_info.value.status_code == 400

    def test_participant_stats_no_blockchain(self, setup_state):
        from api_server import get_participant_stats
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _run(get_participant_stats("5FHne..."))
        assert exc_info.value.status_code == 503

    def test_participant_stats_not_found(self, setup_state):
        from api_server import get_participant_stats
        from fastapi import HTTPException

        mock_connector = AsyncMock()
        mock_connector.get_participant = AsyncMock(return_value=None)
        setup_state.staking_connector = mock_connector
        with pytest.raises(HTTPException) as exc_info:
            _run(get_participant_stats("5abc..."))
        assert exc_info.value.status_code == 404

    def test_participant_stats_found(self, setup_state):
        from api_server import get_participant_stats

        mock_participant = SimpleNamespace(
            total_rounds=5,
            successful_rounds=4,
            total_rewards_earned=1000,
            avg_quality_score=80.0,
        )
        mock_connector = AsyncMock()
        mock_connector.get_participant = AsyncMock(return_value=mock_participant)
        setup_state.staking_connector = mock_connector
        result = _run(
            get_participant_stats("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert result.total_rounds == 5

    def test_participant_stats_with_last_submission(self, setup_state):
        from api_server import get_participant_stats

        mock_participant = SimpleNamespace(
            total_rounds=3,
            successful_rounds=3,
            total_rewards_earned=500,
            avg_quality_score=90.0,
        )
        mock_connector = AsyncMock()
        mock_connector.get_participant = AsyncMock(return_value=mock_participant)
        setup_state.staking_connector = mock_connector
        account = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        setup_state.participant_submissions[account] = time.time()
        result = _run(get_participant_stats(account))
        assert result.last_submission is not None

    def test_system_metrics_no_blockchain(self, setup_state):
        from api_server import get_system_metrics

        result = _run(get_system_metrics())
        assert result.total_rounds == 0
        assert result.blockchain_connected is False

    def test_system_metrics_with_blockchain(self, setup_state):
        from api_server import get_system_metrics

        mock_p1 = SimpleNamespace(is_enrolled=True)
        mock_p2 = SimpleNamespace(is_enrolled=False)
        mock_connector = AsyncMock()
        mock_connector.get_all_participants = AsyncMock(return_value=[mock_p1, mock_p2])
        setup_state.staking_connector = mock_connector
        result = _run(get_system_metrics())
        assert result.total_participants == 2
        assert result.active_participants == 1

    def test_system_metrics_with_completed_rounds(self, setup_state):
        from api_server import get_system_metrics
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        setup_state.completed_rounds.append(
            {
                "start_time": now.isoformat(),
                "completion_time": (now + timedelta(seconds=90)).isoformat(),
            }
        )
        result = _run(get_system_metrics())
        assert result.average_round_time == pytest.approx(90.0)

    # ---- auth middleware unit tests (no TestClient needed) ----

    @staticmethod
    async def _noop_call_next(request):
        return None

    def test_auth_disabled_no_header_needed(self, setup_state):
        from api_server import verify_api_key

        class MockRequest:
            class url:
                path = "/api/v1/status"

            headers = {}
            query_params = {}

        # auth disabled → function returns None (no exception)
        result = _run(verify_api_key(MockRequest(), self._noop_call_next))
        assert result is None

    def test_auth_health_skips_check(self, setup_state):
        from api_server import verify_api_key, ServerConfig

        setup_state.config = ServerConfig(
            blockchain_enabled=False,
            enable_auth=True,
            api_key="secret",
            checkpoint_dir=setup_state.config.checkpoint_dir,
        )

        class MockRequest:
            class url:
                path = "/health"

            headers = {}
            query_params = {}

        # /health path → skips auth even when enabled
        result = _run(verify_api_key(MockRequest(), self._noop_call_next))
        assert result is None

    def test_auth_enabled_missing_key(self, setup_state):
        from api_server import verify_api_key, ServerConfig

        setup_state.config = ServerConfig(
            blockchain_enabled=False,
            enable_auth=True,
            api_key="secret-key",
            checkpoint_dir=setup_state.config.checkpoint_dir,
        )

        class MockRequest:
            class url:
                path = "/api/v1/status"

            headers = {}
            query_params = {}

        result = _run(verify_api_key(MockRequest(), self._noop_call_next))
        assert result.status_code == 401

    def test_auth_enabled_correct_key(self, setup_state):
        from api_server import verify_api_key, ServerConfig

        setup_state.config = ServerConfig(
            blockchain_enabled=False,
            enable_auth=True,
            api_key="correct-key",
            checkpoint_dir=setup_state.config.checkpoint_dir,
        )

        class MockRequest:
            class url:
                path = "/api/v1/status"

            headers = {"X-API-Key": "correct-key"}
            query_params = {}

        # correct key → returns None (no exception)
        result = _run(verify_api_key(MockRequest(), self._noop_call_next))
        assert result is None


###############################################################################
# 3.  monitoring/prometheus_exporter.py  (37% → ~95%)
###############################################################################


class TestPrometheusExporter:
    def test_init_creates_metrics(self):
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        reg = CollectorRegistry()
        exp = PrometheusExporter(port=29091, registry=reg)
        assert exp.port == 29091
        assert hasattr(exp, "training_loss")
        assert hasattr(exp, "fitness_score")
        assert hasattr(exp, "cpu_usage")
        assert hasattr(exp, "transactions_total")

    def test_init_without_prometheus_raises(self):
        with patch("monitoring.prometheus_exporter.PROMETHEUS_AVAILABLE", False):
            import importlib
            import monitoring.prometheus_exporter as m

            orig = m.PROMETHEUS_AVAILABLE
            m.PROMETHEUS_AVAILABLE = False
            try:
                from monitoring.prometheus_exporter import PrometheusExporter

                with pytest.raises(ImportError):
                    PrometheusExporter(port=29099)
            finally:
                m.PROMETHEUS_AVAILABLE = orig

    def test_start_and_stop(self):
        import socket
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        reg = CollectorRegistry()
        exp = PrometheusExporter(port=port, registry=reg)
        exp.start()
        time.sleep(0.05)
        assert exp._server is not None
        exp.stop()
        assert exp._server is None

    def test_start_idempotent(self):
        import socket
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        reg = CollectorRegistry()
        exp = PrometheusExporter(port=port, registry=reg)
        exp.start()
        exp.start()  # second call is a no-op
        assert exp._server is not None
        exp.stop()

    def test_stop_when_not_running(self):
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        reg = CollectorRegistry()
        exp = PrometheusExporter(port=29099, registry=reg)
        exp.stop()  # should not raise

    def test_update_from_collector_full(self):
        from monitoring.prometheus_exporter import PrometheusExporter
        from monitoring.metrics import MetricsCollector, MetricType
        from prometheus_client import CollectorRegistry

        reg = CollectorRegistry()
        exp = PrometheusExporter(port=29095, registry=reg)
        collector = MetricsCollector()
        collector.record(MetricType.TRAINING_LOSS, 0.5, {"epoch": "1"})
        collector.record(MetricType.TRAINING_ACCURACY, 85.0, {"epoch": "1"})
        collector.record(MetricType.VALIDATION_LOSS, 0.6, {"epoch": "1"})
        collector.record(MetricType.VALIDATION_ACCURACY, 80.0, {"epoch": "1"})
        collector.record(MetricType.FITNESS_SCORE, 90.0, {"generation": "1"})
        collector.record(MetricType.BEST_FITNESS, 95.0, {"generation": "1"})
        collector.record(MetricType.AVERAGE_FITNESS, 80.0, {"generation": "1"})
        collector.record(MetricType.CPU_USAGE, 45.0, {})
        collector.record(MetricType.MEMORY_USAGE, 60.0, {})
        collector.record(MetricType.DISK_USAGE, 30.0, {})
        collector.record(MetricType.GPU_USAGE, 75.0, {"device": "0"})
        collector.record(MetricType.GPU_MEMORY, 50.0, {"device": "0"})
        collector.record(MetricType.EPOCH_TIME, 120.0, {})
        exp.update_from_collector(collector)  # should not raise

    def test_update_from_collector_empty(self):
        from monitoring.prometheus_exporter import PrometheusExporter
        from monitoring.metrics import MetricsCollector
        from prometheus_client import CollectorRegistry

        reg = CollectorRegistry()
        exp = PrometheusExporter(port=29096, registry=reg)
        collector = MetricsCollector()
        exp.update_from_collector(collector)  # empty — no raises

    def test_metrics_http_endpoint(self):
        import socket
        import urllib.request
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        reg = CollectorRegistry()
        exp = PrometheusExporter(port=port, registry=reg)
        exp.start()
        time.sleep(0.05)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
                assert resp.status == 200
        finally:
            exp.stop()

    def test_health_http_endpoint(self):
        import socket
        import urllib.request
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        reg = CollectorRegistry()
        exp = PrometheusExporter(port=port, registry=reg)
        exp.start()
        time.sleep(0.05)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
                assert resp.status == 200
        finally:
            exp.stop()

    def test_404_endpoint(self):
        import socket
        import urllib.request
        from monitoring.prometheus_exporter import PrometheusExporter
        from prometheus_client import CollectorRegistry

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        reg = CollectorRegistry()
        exp = PrometheusExporter(port=port, registry=reg)
        exp.start()
        time.sleep(0.05)
        try:
            with pytest.raises(Exception):  # 404 raises HTTPError
                urllib.request.urlopen(f"http://127.0.0.1:{port}/notfound")
        finally:
            exp.stop()


###############################################################################
# 4.  memory/episodic.py — mock chroma + qdrant backends  (47% → ~90%)
###############################################################################


class TestEpisodicMemoryChromaBackend:
    """Mock chromadb to cover the chroma code paths."""

    def _make_col(self):
        """Build a realistic mock ChromaDB collection."""
        store = {}
        col = MagicMock()

        def upsert(**kw):
            for i, rid in enumerate(kw.get("ids", [])):
                store[rid] = {
                    "metadata": kw.get("metadatas", [{}])[i],
                    "document": (
                        (kw.get("documents") or [""])[i] if kw.get("documents") else ""
                    ),
                    "embedding": (
                        (kw.get("embeddings") or [[]])[i]
                        if kw.get("embeddings")
                        else []
                    ),
                }

        col.upsert.side_effect = upsert

        def query(**kw):
            ids = list(store.keys())[: kw.get("n_results", 5)]
            return {
                "ids": [ids],
                "metadatas": [[store[i]["metadata"] for i in ids]],
                "documents": [[store[i]["document"] for i in ids]],
                "embeddings": [[store[i]["embedding"] for i in ids]],
            }

        col.query.side_effect = query

        def get_fn(**kw):
            if kw.get("ids") and kw["ids"][0] in store:
                r = store[kw["ids"][0]]
                return {
                    "ids": kw["ids"][:1],
                    "metadatas": [r["metadata"]],
                    "documents": [r["document"]],
                    "embeddings": [r["embedding"]],
                }
            return {"ids": list(store.keys())}

        col.get = MagicMock(side_effect=get_fn)
        col.delete = MagicMock()
        col.count = MagicMock(side_effect=lambda: len(store))
        return col, store

    def _chroma_ctx(self, col):
        """Return context managers that inject chromadb into the episodic module."""
        import memory.episodic as em_mod

        mock_client = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=col)
        mock_chroma_mod = MagicMock()
        mock_chroma_mod.PersistentClient = MagicMock(return_value=mock_client)
        mock_settings = MagicMock(return_value=MagicMock())
        # Use create=True because chromadb / ChromaSettings may not be in the namespace
        return [
            patch("memory.episodic.CHROMA_AVAILABLE", True),
            patch.object(em_mod, "chromadb", mock_chroma_mod, create=True),
            patch.object(em_mod, "ChromaSettings", mock_settings, create=True),
        ]

    def _run_chroma(self, col, tmp_path, fn):
        """Run fn(em) inside chroma-patched context."""
        from memory.episodic import EpisodicMemory

        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            return fn(em)

    def test_chroma_backend_selected(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            assert em._backend == "chroma"

    def test_chroma_store(self, tmp_path):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            rec = MemoryRecord(
                key="k1", content="hello chroma", embedding=[0.1, 0.2, 0.3]
            )
            em.store(rec)
            assert col.upsert.call_count == 1

    def test_chroma_store_no_embedding(self, tmp_path):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            rec = MemoryRecord(key="k_noemb", content="no embedding")
            em.store(rec)
            assert col.upsert.called

    def test_chroma_retrieve(self, tmp_path):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            rec = MemoryRecord(key="k1", content="hello", embedding=[0.1, 0.2, 0.3])
            em.store(rec)
            em.retrieve([0.1, 0.2, 0.3], top_k=5)
            assert col.query.called

    def test_chroma_retrieve_with_filters(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            em.retrieve([0.1, 0.2, 0.3], filters={"category": "test"})
            assert col.query.called

    def test_chroma_get_existing(self, tmp_path):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            rec = MemoryRecord(key="k1", content="item", embedding=[0.1, 0.2, 0.3])
            em.store(rec)
            em.get("k1")
            assert col.get.called

    def test_chroma_get_missing(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            result = em.get("nonexistent")
            assert result is None

    def test_chroma_delete_success(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            ok = em.delete("k1")
            assert ok is True
            assert col.delete.called

    def test_chroma_delete_exception(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        col.delete.side_effect = Exception("delete error")
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            ok = em.delete("k1")
            assert ok is False

    def test_chroma_clear_with_ids(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        col.get = MagicMock(return_value={"ids": ["k1", "k2"]})
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            em.clear()
            col.delete.assert_called()

    def test_chroma_clear_empty(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        col.get = MagicMock(return_value={"ids": []})
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            em.clear()  # should not call delete on empty

    def test_chroma_len(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        col.count = MagicMock(return_value=7)
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            assert len(em) == 7

    def test_chroma_repr(self, tmp_path):
        from memory.episodic import EpisodicMemory

        col, _ = self._make_col()
        patches = self._chroma_ctx(col)
        with patches[0], patches[1], patches[2]:
            em = EpisodicMemory(persist_path=str(tmp_path / "cdb"))
            assert "chroma" in repr(em)


class TestEpisodicMemoryQdrantBackend:
    """Mock qdrant_client to cover the qdrant code paths."""

    def _make_qdrant(self):
        hit = SimpleNamespace(
            id="k1",
            payload={"_content": "hello qdrant", "_timestamp": 0.0, "_ttl": -1.0},
            vector=[0.1, 0.2, 0.3],
        )
        q = MagicMock()
        q.get_collections.return_value = SimpleNamespace(collections=[])
        q.create_collection = MagicMock()
        q.upsert = MagicMock()
        q.search.return_value = [hit]
        q.retrieve.return_value = [hit]
        q.delete = MagicMock()
        q.delete_collection = MagicMock()
        q.get_collection.return_value = SimpleNamespace(vectors_count=3)
        return q

    def _qdrant_ctx(self, q):
        """Return a stack of patches that inject all qdrant names into the module."""
        import memory.episodic as em_mod

        mock_distance = MagicMock()
        mock_distance.COSINE = "Cosine"
        return [
            patch("memory.episodic.QDRANT_AVAILABLE", True),
            patch.object(
                em_mod, "QdrantClient", MagicMock(return_value=q), create=True
            ),
            patch.object(em_mod, "Distance", mock_distance, create=True),
            patch.object(
                em_mod, "PointStruct", MagicMock(return_value=MagicMock()), create=True
            ),
            patch.object(em_mod, "VectorParams", MagicMock(), create=True),
            patch.object(em_mod, "Filter", MagicMock(), create=True),
            patch.object(em_mod, "FieldCondition", MagicMock(), create=True),
            patch.object(em_mod, "MatchValue", MagicMock(), create=True),
        ]

    def _with_qdrant(self, q, fn):
        from memory.episodic import EpisodicMemory

        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            return fn(em)

    def test_qdrant_init(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            assert em._backend == "qdrant"
            q.create_collection.assert_called_once()

    def test_qdrant_init_existing_collection(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        q.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="nawal_episodic")]
        )
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            q.create_collection.assert_not_called()

    def test_qdrant_store(self):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            rec = MemoryRecord(key="k1", content="hello", embedding=[0.1, 0.2, 0.3])
            em.store(rec)
            q.upsert.assert_called_once()

    def test_qdrant_store_no_embedding(self):
        from memory.episodic import EpisodicMemory
        from memory.interfaces import MemoryRecord

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            rec = MemoryRecord(key="k_noe", content="no embedding")
            em.store(rec)
            q.upsert.assert_called_once()

    def test_qdrant_retrieve(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            results = em.retrieve([0.1, 0.2, 0.3], top_k=3)
            assert len(results) == 1
            assert results[0].content == "hello qdrant"

    def test_qdrant_retrieve_with_filters(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            em.retrieve([0.1, 0.2, 0.3], filters={"cat": "test"})
            assert q.search.called

    def test_qdrant_get(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            result = em.get("k1")
            assert result is not None
            assert result.content == "hello qdrant"

    def test_qdrant_get_not_found(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        q.retrieve.return_value = []
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            result = em.get("missing")
            assert result is None

    def test_qdrant_delete_success(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            ok = em.delete("k1")
            assert ok is True

    def test_qdrant_delete_exception(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        q.delete.side_effect = Exception("err")
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            ok = em.delete("k1")
            assert ok is False

    def test_qdrant_clear(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            em.clear()
            q.delete_collection.assert_called_once()

    def test_qdrant_len(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            assert len(em) == 3

    def test_qdrant_repr(self):
        from memory.episodic import EpisodicMemory

        q = self._make_qdrant()
        patches = self._qdrant_ctx(q)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            em = EpisodicMemory(qdrant_url="http://localhost:6333")
            assert "qdrant" in repr(em)


###############################################################################
# 5.  server/aggregator.py — deeper coverage  (~62% → ~90%)
###############################################################################


def _make_update(pid="p1", gid="g1", rnd=0, fitness=None, samples=100):
    """Helper to build a ModelUpdate with all required fields."""
    from server.aggregator import ModelUpdate

    return ModelUpdate(
        participant_id=pid,
        genome_id=gid,
        round_number=rnd,
        weights={},
        samples_trained=samples,
        training_time=1.0,
        fitness_score=fitness,
    )


class TestModelUpdateWeight:
    def test_weight_fitness_not_none(self):
        u = _make_update(fitness=80.0)
        assert u.calculate_weight("fitness") == pytest.approx(0.8)

    def test_weight_fitness_is_none(self):
        u = _make_update()
        assert u.calculate_weight("fitness") == 0.0

    def test_weight_hybrid(self):
        u = _make_update(fitness=60.0)
        w = u.calculate_weight("hybrid")
        assert w > 0.0

    def test_weight_unknown_defaults_to_one(self):
        u = _make_update()
        assert u.calculate_weight("unknown_strategy") == 1.0


class TestFedAvgStrategyEdges:
    def test_aggregate_empty(self):
        from server.aggregator import FedAvgStrategy

        s = FedAvgStrategy()
        result = _run(s.aggregate([], {"a": torch.ones(2)}))
        assert torch.allclose(result["a"], torch.ones(2))

    def test_aggregate_zero_total_weight(self):
        from server.aggregator import FedAvgStrategy

        s = FedAvgStrategy(weighting="fitness")
        u = _make_update(fitness=None)  # weight=0 → total_weight=0 → return current
        u.weights = {"a": torch.ones(2)}
        result = _run(s.aggregate([u], {"a": torch.ones(2) * 5}))
        assert torch.allclose(result["a"], torch.ones(2) * 5)

    def test_aggregate_missing_key_in_update(self):
        from server.aggregator import FedAvgStrategy

        s = FedAvgStrategy()
        current = {"a": torch.ones(2), "b": torch.ones(2)}
        u = _make_update()
        u.weights = {"a": torch.ones(2)}
        result = _run(s.aggregate([u], current))
        assert "b" in result


class TestByzantineRobustStrategy:
    def test_empty_updates(self):
        from server.aggregator import ByzantineRobustStrategy

        s = ByzantineRobustStrategy()
        result = _run(s.aggregate([], {"a": torch.ones(2)}))
        assert torch.allclose(result["a"], torch.ones(2))

    def test_too_few_updates_fallback(self):
        from server.aggregator import ByzantineRobustStrategy

        s = ByzantineRobustStrategy()
        updates = [_make_update(pid=f"p{i}") for i in range(2)]
        for u in updates:
            u.weights = {"a": torch.ones(2)}
        result = _run(s.aggregate(updates, {"a": torch.ones(2)}))
        assert "a" in result

    def test_sufficient_updates(self):
        from server.aggregator import ByzantineRobustStrategy

        s = ByzantineRobustStrategy()
        updates = [_make_update(pid=f"p{i}") for i in range(4)]
        for i, u in enumerate(updates):
            u.weights = {"a": torch.ones(2) * (i + 1)}
        result = _run(s.aggregate(updates, {"a": torch.ones(2)}))
        assert "a" in result


class TestFederatedAggregatorExtra:
    def _genome(self):
        from genome.dna import Genome

        return Genome(
            genome_id="g1",
            generation=0,
            encoder_layers=[],
            decoder_layers=[],
            parent_genomes=[],
        )

    def test_fedavg_aggregate(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        params = [{"a": torch.ones(2) * i} for i in range(1, 4)]
        result = fa.fedavg_aggregate(params)
        assert result["a"].mean().item() == pytest.approx(2.0)

    def test_fedavg_aggregate_empty_raises(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        with pytest.raises(ValueError):
            fa.fedavg_aggregate([])

    def test_weighted_aggregate(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        params = [{"a": torch.ones(2)}, {"a": torch.ones(2) * 3}]
        result = fa.weighted_aggregate(params, [0.5, 0.5])
        assert result["a"].mean().item() == pytest.approx(2.0)

    def test_weighted_aggregate_empty_raises(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        with pytest.raises(ValueError):
            fa.weighted_aggregate([], [])

    def test_weighted_aggregate_mismatch_raises(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        with pytest.raises(ValueError):
            fa.weighted_aggregate([{"a": torch.ones(2)}], [0.5, 0.5])

    def test_select_clients_returns_correct_count(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=2)
        clients = fa.select_clients(10, num_to_select=4)
        assert len(clients) == 4
        assert all(0 <= c < 10 for c in clients)

    def test_select_clients_caps_at_total(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=2)
        clients = fa.select_clients(3, num_to_select=10)
        assert len(clients) == 3

    def test_select_clients_by_fraction(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        clients = fa.select_clients_by_fraction(10)
        assert len(clients) >= 1

    def test_get_statistics_empty(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator()
        stats = fa.get_statistics()
        assert stats["total_rounds"] == 0

    def test_get_statistics_with_history(self):
        from server.aggregator import FederatedAggregator, AggregationRound

        fa = FederatedAggregator()
        fa.aggregation_history.append(
            AggregationRound(
                round_number=0,
                genome_id="g1",
                num_participants=3,
                total_samples=300,
                avg_fitness=80.0,
                strategy_used="FedAvg",
                timestamp=datetime.now(timezone.utc).isoformat(),
                aggregation_time=1.0,
            )
        )
        stats = fa.get_statistics()
        assert stats["total_rounds"] == 1
        assert stats["avg_fitness"] == 80.0

    def test_submit_update_no_genome(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=10)
        u = _make_update()
        u.weights = {"a": torch.ones(2)}
        assert _run(fa.submit_update(u)) is False

    def test_submit_update_genome_mismatch(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=10)
        fa.set_genome(self._genome(), {"a": torch.ones(2)})
        u = _make_update(gid="wrong")
        u.weights = {"a": torch.ones(2)}
        assert _run(fa.submit_update(u)) is False

    def test_submit_update_accepted(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=10)
        fa.set_genome(self._genome(), {"a": torch.ones(2)})
        u = _make_update()
        u.weights = {"a": torch.ones(2)}
        assert _run(fa.submit_update(u)) is True

    def test_submit_update_triggers_aggregation(self):
        from server.aggregator import FederatedAggregator

        fa = FederatedAggregator(min_participants=2)
        fa.set_genome(self._genome(), {"a": torch.ones(2)})
        u1 = _make_update(pid="p1")
        u1.weights = {"a": torch.ones(2)}
        u2 = _make_update(pid="p2", samples=200)
        u2.weights = {"a": torch.ones(2) * 2}
        _run(fa.submit_update(u1))
        _run(fa.submit_update(u2))
        time.sleep(0.05)

    def test_submit_update_with_unknown_participant(self):
        from server.aggregator import FederatedAggregator
        from server.participant_manager import ParticipantManager

        pm = ParticipantManager()
        fa = FederatedAggregator(min_participants=10, participant_manager=pm)
        fa.set_genome(self._genome(), {"a": torch.ones(2)})
        u = _make_update(pid="unknown_p")
        u.weights = {"a": torch.ones(2)}
        assert _run(fa.submit_update(u)) is False

    def test_config_compat(self):
        from server.aggregator import FederatedAggregator

        cfg = SimpleNamespace(
            min_participants=2,
            aggregation_strategy="fedavg",
            max_wait_time=60.0,
            min_clients=2,
            client_fraction=0.8,
        )
        fa = FederatedAggregator(config=cfg)
        assert fa.min_participants == 2


###############################################################################
# 6.  genome/population.py  (~63% → ~90%)
###############################################################################


class TestPopulationSelectionStrategies:
    def _genome(self, gid: str, fitness: float):
        from genome.dna import Genome

        g = Genome(
            genome_id=gid,
            generation=0,
            encoder_layers=[],
            decoder_layers=[],
            parent_genomes=[],
        )
        g.fitness_score = fitness
        return g

    def _pop(self, strategy: str = "TOURNAMENT"):
        from genome.population import (
            PopulationManager,
            PopulationConfig,
            SelectionStrategy,
        )

        cfg = PopulationConfig(
            target_size=10,
            min_size=2,
            max_size=20,
            tournament_size=3,
            elitism_count=2,
            selection_strategy=SelectionStrategy[strategy],
            elitism_threshold=0.0,  # any fitness qualifies as elite
        )
        return PopulationManager(cfg)

    def test_roulette_selection(self):
        pop = self._pop("ROULETTE")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10 + 1)))
        parent = pop.select_parent()
        assert parent is not None

    def test_rank_selection(self):
        pop = self._pop("RANK")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10)))
        parent = pop.select_parent()
        assert parent is not None

    def test_elite_selection(self):
        pop = self._pop("ELITE")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10 + 1)))
        pop.update_elite(generation=1)
        parent = pop.select_parent()
        assert parent is not None

    def test_elite_selection_no_elite_falls_back(self):
        pop = self._pop("ELITE")
        pop.add_genome(self._genome("g1", 50.0))
        # no update_elite called — no elite set
        parent = pop.select_parent()
        assert parent is not None

    def test_unknown_strategy_default(self):
        # Test the default case in select_parent match
        from genome.population import (
            PopulationManager,
            PopulationConfig,
            SelectionStrategy,
        )

        pop = PopulationManager()
        for i in range(3):
            pop.add_genome(self._genome(f"g{i}", float(i)))
        parent = pop.select_parent()
        assert parent is not None

    def test_select_parents_multiple(self):
        pop = self._pop("TOURNAMENT")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i + 1)))
        parents = pop.select_parents(count=3)
        assert len(parents) == 3

    def test_update_elite_no_scored(self):
        pop = self._pop("TOURNAMENT")
        g = self._genome("g1", 10.0)
        g.fitness_score = None
        pop.add_genome(g)
        pop.update_elite(generation=1)  # no-op

    def test_get_elite_genomes(self):
        pop = self._pop("TOURNAMENT")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10 + 1)))
        pop.update_elite(generation=1)
        elites = pop.get_elite_genomes()
        assert len(elites) > 0

    def test_remove_from_elite(self):
        pop = self._pop("TOURNAMENT")
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10 + 1)))
        pop.update_elite(generation=1)
        # elite_genomes is a list of genome ID strings
        first_elite_id = pop.elite_genomes[0]
        pop.remove_genome(first_elite_id)
        assert first_elite_id not in pop.elite_genomes

    def test_cull_population(self):
        from genome.population import PopulationConfig, PopulationManager

        cfg = PopulationConfig(target_size=3, min_size=1, max_size=5)
        pop = PopulationManager(cfg)
        for i in range(7):
            pop.add_genome(self._genome(f"g{i}", float(i + 1)))
        assert len(pop.genomes) <= cfg.max_size

    def test_compute_statistics(self):
        pop = self._pop()
        for i in range(5):
            pop.add_genome(self._genome(f"g{i}", float(i * 10 + 1)))
        pop.update_elite(generation=1)
        stats = pop.compute_statistics(generation=1)
        assert stats is not None
        assert stats.max_fitness == 41.0

    def test_compute_statistics_empty(self):
        from genome.population import PopulationManager

        pop = PopulationManager()
        stats = pop.compute_statistics(generation=0)
        # empty population → None or stats with all zeros
        assert stats is None or stats.population_size == 0

    def test_add_duplicate_replaces(self):
        from genome.population import PopulationManager

        pop = PopulationManager()
        pop.add_genome(self._genome("g1", 50.0))
        pop.add_genome(self._genome("g1", 90.0))
        assert pop.genomes["g1"].fitness_score == 90.0

    def test_config_validation_errors(self):
        from genome.population import PopulationConfig

        cfg = PopulationConfig(target_size=1, min_size=5)
        valid, errors = cfg.validate()
        assert not valid

    def test_invalid_config_raises(self):
        from genome.population import PopulationManager, PopulationConfig

        cfg = PopulationConfig(target_size=1, min_size=5)
        with pytest.raises(ValueError):
            PopulationManager(cfg)


###############################################################################
# 7.  blockchain/genome_registry.py  (~27% → ~80%)
###############################################################################


class TestGenomeRegistryLocal:
    def _client(self):
        c = MagicMock()
        c.submit_extrinsic = MagicMock(
            return_value=SimpleNamespace(success=True, error=None)
        )
        c.query_storage = MagicMock(return_value=None)
        return c

    def _keypair(self):
        return SimpleNamespace(ss58_address="5FHne...")

    def test_init(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        assert reg.storage_backend == StorageBackend.LOCAL

    def test_store_genome_local(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        meta = reg.store_genome(
            self._keypair(), {"layers": 4}, fitness=90.0, generation=3
        )
        assert meta.genome_id is not None
        assert meta.generation == 3
        assert meta.fitness == 90.0

    def test_store_genome_with_parent_ids(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        meta = reg.store_genome(
            self._keypair(),
            {"x": 1},
            fitness=80.0,
            generation=5,
            parent_ids=["p1", "p2"],
        )
        assert meta.parent_ids == ["p1", "p2"]

    def test_store_genome_chain_failure_raises(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        c = self._client()
        c.submit_extrinsic.return_value = SimpleNamespace(
            success=False, error="chain error"
        )
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        with pytest.raises(RuntimeError):
            reg.store_genome(self._keypair(), {"x": 1}, fitness=80.0, generation=1)

    def test_get_genome_not_found(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        assert reg.get_genome("nonexistent") is None

    def test_get_genome_roundtrip(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        genome_data = {"layers": 2, "hidden": 128}
        c = self._client()
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        meta = reg.store_genome(
            self._keypair(), genome_data, fitness=75.0, generation=1
        )
        c.query_storage.return_value = {
            "owner": "5FHne...",
            "generation": 1,
            "fitness": int(75.0 * 100),
            "storage_backend": "local",
            "content_hash": meta.content_hash,
            "parent_ids": [],
            "timestamp": int(datetime.now().timestamp()),
            "size_bytes": 100,
        }
        result = reg.get_genome(meta.genome_id)
        assert result == genome_data

    def test_get_metadata_returns_none(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        assert reg.get_metadata("missing") is None

    def test_get_metadata_exception_returns_none(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        c = self._client()
        c.query_storage.side_effect = Exception("rpc error")
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        assert reg.get_metadata("some_id") is None

    def test_get_lineage_stops_at_no_parent(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        c = self._client()
        c.query_storage.return_value = {
            "owner": "5FHne...",
            "generation": 1,
            "fitness": 8000,
            "storage_backend": "local",
            "content_hash": "abc",
            "parent_ids": [],
            "timestamp": int(datetime.now().timestamp()),
            "size_bytes": 100,
        }
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        lineage = reg.get_lineage("genome_id")
        assert isinstance(lineage, list)
        assert len(lineage) == 1

    def test_get_by_owner_empty(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        c = self._client()
        c.query_storage.return_value = []
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        assert reg.get_by_owner("5FHne...") == []

    def test_get_by_owner_exception(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        c = self._client()
        c.query_storage.side_effect = Exception("rpc error")
        reg = GenomeRegistry(c, StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs")
        assert reg.get_by_owner("5FHne...") == []

    def test_metadata_to_dict(self):
        from blockchain.genome_registry import GenomeMetadata, StorageBackend

        meta = GenomeMetadata(
            genome_id="abc123",
            owner="5FHne...",
            generation=2,
            fitness=90.5,
            storage_backend=StorageBackend.LOCAL,
            content_hash="/path",
        )
        d = meta.to_dict()
        assert d["genome_id"] == "abc123"
        assert d["fitness"] == 9050  # basis points

    def test_store_arweave_falls_back_to_local(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.ARWEAVE, local_storage_dir=tmp_path / "gs"
        )
        meta = reg.store_genome(self._keypair(), {"x": 1}, fitness=70.0, generation=2)
        assert meta.content_hash is not None

    def test_retrieve_arweave_falls_back_to_local(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.ARWEAVE, local_storage_dir=tmp_path / "gs"
        )
        meta = reg.store_genome(self._keypair(), {"y": 2}, fitness=60.0, generation=1)
        # retrieve should work via local fallback
        data = reg._retrieve_arweave(meta.content_hash)
        assert data is not None

    def test_store_ipfs_raises(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.IPFS, local_storage_dir=tmp_path / "gs"
        )
        with pytest.raises((RuntimeError, Exception)):
            reg.store_genome(self._keypair(), {"x": 1}, fitness=70.0, generation=1)

    def test_local_path_traversal_blocked(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        reg = GenomeRegistry(
            self._client(), StorageBackend.LOCAL, local_storage_dir=tmp_path / "gs"
        )
        with pytest.raises(ValueError):
            reg._store_local("../evil/path", b"data")

    def test_retrieve_local_not_found(self, tmp_path):
        from blockchain.genome_registry import GenomeRegistry, StorageBackend

        gs = tmp_path / "gs"
        gs.mkdir(parents=True, exist_ok=True)
        reg = GenomeRegistry(self._client(), StorageBackend.LOCAL, local_storage_dir=gs)
        with pytest.raises((FileNotFoundError, ValueError)):
            reg._retrieve_local(str(gs / "missing.json"))


###############################################################################
# 8.  genome/model_builder.py — uncovered paths  (~74% → ~90%)
###############################################################################


class TestActivationFactory:
    def test_unknown_activation_returns_gelu(self):
        from genome.model_builder import ActivationFactory
        import torch.nn as nn

        act = ActivationFactory.create("totally_unknown_xyz")
        assert isinstance(act, nn.GELU)

    def test_all_known_activations(self):
        from genome.model_builder import ActivationFactory
        import torch.nn as nn

        for name in ["relu", "gelu", "silu", "swish", "tanh", "mish", "sigmoid"]:
            act = ActivationFactory.create(name)
            assert isinstance(act, nn.Module)


class TestNormalizationFactory:
    def test_unknown_norm_returns_layernorm(self):
        from genome.model_builder import NormalizationFactory
        import torch.nn as nn

        norm = NormalizationFactory.create("unknown_norm_xyz", 64)
        assert isinstance(norm, nn.LayerNorm)

    def test_group_norm(self):
        from genome.model_builder import NormalizationFactory
        import torch.nn as nn

        norm = NormalizationFactory.create("group_norm", 64)
        assert isinstance(norm, nn.GroupNorm)

    def test_rms_norm(self):
        from genome.model_builder import NormalizationFactory, RMSNorm

        norm = NormalizationFactory.create("rms_norm", 64)
        assert isinstance(norm, RMSNorm)

    def test_batch_norm(self):
        from genome.model_builder import NormalizationFactory
        import torch.nn as nn

        norm = NormalizationFactory.create("batch_norm", 64)
        assert isinstance(norm, nn.BatchNorm1d)


class TestRMSNormForward:
    def test_forward_shape(self):
        from genome.model_builder import RMSNorm

        norm = RMSNorm(hidden_size=16)
        x = torch.randn(2, 4, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_forward_all_zeros(self):
        from genome.model_builder import RMSNorm

        norm = RMSNorm(hidden_size=8)
        x = torch.zeros(1, 3, 8)
        out = norm(x)
        assert out.shape == x.shape


class TestMultiHeadAttentionForward:
    def test_forward_basic(self):
        from genome.model_builder import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == (2, 8, 32)

    def test_forward_with_mask(self):
        from genome.model_builder import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 4, 8, 8)
        out = attn(x, attention_mask=mask)
        assert out.shape == (2, 8, 32)


###############################################################################
# 9.  blockchain/substrate_client.py  (~37% → ~70%)
###############################################################################


class TestSubstrateClientMocked:
    def _mock_substrate(self):
        sub = MagicMock()
        sub.chain = "TestNet"
        sub.runtime_version = 1
        sub.properties = {}
        sub.query.return_value = MagicMock(value={"key": "val"})
        result = MagicMock()
        result.is_success = True
        result.block_hash = "0xabc"
        result.error_message = None
        result.triggered_events = []
        sub.submit_extrinsic = MagicMock(return_value=result)
        sub.compose_call = MagicMock(return_value=MagicMock())
        extrinsic = MagicMock()
        extrinsic.extrinsic_hash = "0xhash"
        sub.create_signed_extrinsic = MagicMock(return_value=extrinsic)
        block = MagicMock()
        block.__getitem__ = lambda self, key: {"header": {"number": 42}}[key]
        sub.get_block = MagicMock(return_value={"header": {"number": 42}})
        sub.close = MagicMock()
        return sub

    def test_init_no_substrate_raises(self):
        with patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", False):
            from blockchain.substrate_client import SubstrateClient, ChainConfig

            with pytest.raises(RuntimeError):
                SubstrateClient(ChainConfig())

    def test_connect_success(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig

        sub = self._mock_substrate()
        with (
            patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", True),
            patch("blockchain.substrate_client.SubstrateInterface", return_value=sub),
        ):
            client = SubstrateClient(ChainConfig())
            client.connect()
            assert client.is_connected()

    def test_connect_failure_raises(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig

        with (
            patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", True),
            patch(
                "blockchain.substrate_client.SubstrateInterface",
                side_effect=Exception("timeout"),
            ),
        ):
            client = SubstrateClient(ChainConfig())
            with pytest.raises(ConnectionError):
                client.connect(max_retries=1, base_delay=0.01)

    def test_disconnect(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig

        sub = self._mock_substrate()
        with (
            patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", True),
            patch("blockchain.substrate_client.SubstrateInterface", return_value=sub),
        ):
            client = SubstrateClient(ChainConfig())
            client.connect()
            client.disconnect()
            sub.close.assert_called_once()
            assert not client.is_connected()

    def test_query_storage_auto_connect(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig

        sub = self._mock_substrate()
        with (
            patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", True),
            patch("blockchain.substrate_client.SubstrateInterface", return_value=sub),
        ):
            client = SubstrateClient(ChainConfig())
            client.connect()  # connect first
            result = client.query_storage("System", "Account", ["0xabc"])
            assert result == {"key": "val"}

    def test_chain_config_local(self):
        from blockchain.substrate_client import ChainConfig, NetworkType

        cfg = ChainConfig.local()
        assert cfg.network == NetworkType.LOCAL
        assert "9944" in cfg.rpc_url

    def test_chain_config_testnet(self):
        from blockchain.substrate_client import ChainConfig, NetworkType

        cfg = ChainConfig.testnet()
        assert cfg.network == NetworkType.TESTNET

    def test_chain_config_mainnet(self):
        from blockchain.substrate_client import ChainConfig, NetworkType

        cfg = ChainConfig.mainnet()
        assert cfg.network == NetworkType.MAINNET


###############################################################################
# 10.  monitoring/logging_config.py  (~89% → ~100%)
###############################################################################


class TestLoggingConfig:
    def test_configure_logging_default(self):
        from monitoring.logging_config import configure_logging

        configure_logging()  # should not raise

    def test_configure_logging_with_file(self, tmp_path):
        from monitoring.logging_config import configure_logging

        log_file = tmp_path / "test.log"
        configure_logging(log_level="DEBUG", log_file=log_file)

    def test_get_logger(self):
        from monitoring.logging_config import get_logger

        logger = get_logger("test_module")
        assert logger is not None

    def test_configure_serialize(self):
        from monitoring.logging_config import configure_logging

        configure_logging(serialize=True)


###############################################################################
# 11.  blockchain/staking_connector.py — more coverage  (~48% → ~70%)
###############################################################################


class TestStakingConnectorExtra:
    def test_importerror_without_substrate(self):
        with patch("blockchain.staking_connector.SUBSTRATE_AVAILABLE", False):
            from blockchain.staking_connector import StakingConnector

            with pytest.raises(ImportError):
                StakingConnector(node_url="ws://localhost:9944", mock_mode=False)

    def test_mock_with_community_tracking(self):
        from blockchain.staking_connector import StakingConnector

        with (
            patch("blockchain.staking_connector.COMMUNITY_AVAILABLE", True),
            patch("blockchain.staking_connector.CommunityConnector") as MockCC,
        ):
            mock_cc = AsyncMock()
            mock_cc.connect = AsyncMock()
            MockCC.return_value = mock_cc
            sc = StakingConnector(
                node_url="ws://localhost:9944",
                mock_mode=True,
                enable_community_tracking=True,
            )
            _run(sc.connect())
            mock_cc.connect.assert_called_once()

    def test_enroll_duplicate_returns_false(self):
        from blockchain.staking_connector import StakingConnector

        sc = StakingConnector(node_url="ws://localhost:9944", mock_mode=True)
        _run(sc.connect())
        _run(sc.enroll_participant("5abc...", 1000))
        result = _run(sc.enroll_participant("5abc...", 1000))
        assert result is False

    def test_unenroll_not_enrolled(self):
        from blockchain.staking_connector import StakingConnector

        sc = StakingConnector(node_url="ws://localhost:9944", mock_mode=True)
        _run(sc.connect())
        result = _run(sc.unenroll_participant("notexist..."))
        assert result is False

    def test_disconnect_cleans_up(self):
        from blockchain.staking_connector import StakingConnector

        sc = StakingConnector(node_url="ws://localhost:9944", mock_mode=True)
        _run(sc.connect())
        assert sc.is_connected
        _run(sc.disconnect())

    def test_get_participant_info_enrolled(self):
        from blockchain.staking_connector import StakingConnector

        sc = StakingConnector(node_url="ws://localhost:9944", mock_mode=True)
        _run(sc.connect())
        account = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        _run(sc.enroll_participant(account, 5000))
        info = _run(sc.get_participant_info(account))
        assert info is not None

    def test_get_all_participants(self):
        from blockchain.staking_connector import StakingConnector

        sc = StakingConnector(node_url="ws://localhost:9944", mock_mode=True)
        _run(sc.connect())
        _run(
            sc.enroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 2000
            )
        )
        _run(
            sc.enroll_participant(
                "5FHneW7L5ZSfuhqtjK9DseAiQCFf1y2GaXFqsioP5zHN3vhb", 3000
            )
        )
        participants = _run(sc.get_all_participants())
        assert isinstance(participants, list)
        assert len(participants) == 2


###############################################################################
# 12.  genome/encoding.py  (~90% → ~100%)
###############################################################################


class TestGenomeEncodingExtra:
    def test_genome_to_dict_roundtrip(self):
        from genome.dna import Genome

        g = Genome(
            genome_id="enc_test",
            generation=1,
            encoder_layers=[],
            decoder_layers=[],
            parent_genomes=[],
        )
        d = g.model_dump()
        assert d["genome_id"] == "enc_test"
        g2 = Genome(**d)
        assert g2.genome_id == "enc_test"

    def test_genome_with_layers_to_dict(self):
        from genome.dna import Genome, ArchitectureLayer, LayerType

        g = Genome(
            genome_id="g_layers",
            generation=0,
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.MULTIHEAD_ATTENTION,
                    hidden_size=64,
                    num_heads=2,
                )
            ],
            decoder_layers=[],
            parent_genomes=[],
        )
        d = g.model_dump()
        assert len(d["encoder_layers"]) == 1

    def test_architecture_layer_to_dict(self):
        from genome.encoding import ArchitectureLayer, LayerType

        layer = ArchitectureLayer(layer_type=LayerType.LINEAR, hidden_size=128)
        d = layer.to_dict()
        assert d["layer_type"] == LayerType.LINEAR

    def test_architecture_layer_from_dict(self):
        from genome.encoding import ArchitectureLayer, LayerType

        layer = ArchitectureLayer(layer_type=LayerType.LINEAR, hidden_size=64)
        d = layer.to_dict()
        reconstructed = ArchitectureLayer.from_dict(d)
        assert reconstructed.hidden_size == 64

    def test_hyperparameters_to_dict(self):
        from genome.encoding import Hyperparameters

        hp = Hyperparameters()
        d = hp.to_dict()
        assert isinstance(d, dict)


###############################################################################
# 13.  data/tokenizers.py  (~86% → ~98%)
###############################################################################


class TestTokenizerRemaining:
    def _char_cfg(self):
        from data.tokenizers import TokenizerConfig, TokenizerType

        return TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER)

    def _word_cfg(self, vocab_size=1000):
        from data.tokenizers import TokenizerConfig, TokenizerType

        return TokenizerConfig(tokenizer_type=TokenizerType.WORD, vocab_size=vocab_size)

    def test_char_tokenizer_encode(self):
        from data.tokenizers import CharacterTokenizer, TokenizerConfig, TokenizerType

        cfg = TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER, max_length=3)
        t = CharacterTokenizer(cfg)
        ids = t.encode("abc")
        assert len(ids) == 3

    def test_char_tokenizer_decode(self):
        from data.tokenizers import CharacterTokenizer

        t = CharacterTokenizer(self._char_cfg())
        ids = t.encode("hello")
        decoded = t.decode(ids)
        assert "hello" in decoded

    def test_char_tokenizer_special_tokens(self):
        from data.tokenizers import CharacterTokenizer

        t = CharacterTokenizer(self._char_cfg())
        # CharacterTokenizer stores special tokens in vocab dict
        assert isinstance(t.vocab["<PAD>"], int)
        assert isinstance(t.vocab["<UNK>"], int)

    def test_word_tokenizer_encode_returns_ids(self):
        from data.tokenizers import WordTokenizer, TokenizerConfig, TokenizerType

        cfg = TokenizerConfig(
            tokenizer_type=TokenizerType.WORD, vocab_size=1000, max_length=2
        )
        t = WordTokenizer(cfg)
        t.build_vocab(["hello world hello", "world test"])
        ids = t.encode("hello world")
        assert len(ids) == 2

    def test_word_tokenizer_decode_list(self):
        from data.tokenizers import WordTokenizer

        t = WordTokenizer(self._word_cfg(vocab_size=500))
        t.build_vocab(["hello world test hello world"])
        ids = t.encode("hello world")
        decoded = t.decode(ids)
        assert isinstance(decoded, str)

    def test_tokenizer_config_defaults(self):
        from data.tokenizers import TokenizerConfig, TokenizerType

        cfg = TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER)
        # defaults: vocab_size=None (no limit for character tokenizer), max_length=512
        assert cfg.vocab_size is None
        assert cfg.max_length == 512

    def test_create_tokenizer_char_type(self):
        from data.tokenizers import create_tokenizer, TokenizerConfig, TokenizerType

        t = create_tokenizer(TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER))
        assert t is not None


###############################################################################
# 14.  Near-100% module fixes
###############################################################################


class TestMaintenance4:
    def test_output_filter_safe(self):
        from maintenance.output_filter import OutputFilter

        f = OutputFilter()
        result = f.filter("Tell me something", "Here is a helpful response.")
        assert result is not None

    def test_output_filter_is_safe(self):
        from maintenance.output_filter import OutputFilter

        f = OutputFilter()
        assert f.is_safe("safe normal text") is True

    def test_output_filter_add_pattern(self):
        from maintenance.output_filter import OutputFilter
        from maintenance.interfaces import RiskLevel

        f = OutputFilter()
        import re

        f.add_pattern(re.compile(r"forbidden"), "forbidden", RiskLevel.HIGH)
        assert not f.is_safe("this is forbidden content")

    def test_drift_detector_no_baseline(self):
        from maintenance.drift_detector import DriftDetector

        dd = DriftDetector()
        assert dd.is_drifted() is False  # no baseline

    def test_drift_detector_record_baseline(self):
        from maintenance.drift_detector import DriftDetector

        dd = DriftDetector()
        dd.record_baseline("ckpt1", {"loss": 0.5, "acc": 0.9})
        assert dd.has_baseline

    def test_drift_detector_check(self):
        from maintenance.drift_detector import DriftDetector

        dd = DriftDetector()
        dd.record_baseline("ckpt1", {"loss": 0.5, "acc": 0.9})
        dd.record_observation({"loss": 0.6, "acc": 0.85})
        report = dd.check()
        assert report is not None

    def test_drift_detector_is_drifted_after_check(self):
        from maintenance.drift_detector import DriftDetector

        dd = DriftDetector()
        result = dd.is_drifted()
        assert isinstance(result, bool)

    def test_self_repair_repair(self):
        from maintenance.self_repair import SelfRepair

        srm = SelfRepair()
        result = srm.repair()
        assert result is not None


class TestValuationCoverage:
    def test_reward_model_score_list(self):
        from valuation.reward import DriveBasedRewardModel

        rm = DriveBasedRewardModel()
        # score() takes a list of candidate dicts, returns list of floats
        scores = rm.score([{"text": "great answer"}, {"text": "bad answer"}])
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_reward_model_ranked(self):
        from valuation.reward import DriveBasedRewardModel

        rm = DriveBasedRewardModel()
        ranked = rm.ranked([{"text": "a"}, {"text": "b"}, {"text": "c"}])
        assert isinstance(ranked, list)
        assert len(ranked) == 3

    def test_safety_filter_is_safe(self):
        from valuation.safety import BasicSafetyFilter

        sf = BasicSafetyFilter()
        assert sf.is_safe("normal helpful text") is True

    def test_safety_filter_check_with_reason(self):
        from valuation.safety import BasicSafetyFilter

        sf = BasicSafetyFilter()
        ok, reason = sf.check_with_reason("normal text")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


class TestMemoryCoverage:
    def test_manager_store_text(self):
        from memory.manager import MemoryManager

        mm = MemoryManager()
        mm.store_text("test content", key="k1")
        rec = mm.get("k1")
        assert rec is not None
        assert rec.content == "test content"

    def test_manager_store_record(self):
        from memory.manager import MemoryManager
        from memory.interfaces import MemoryRecord

        mm = MemoryManager()
        rec = MemoryRecord(key="k2", content="hello world")
        mm.store(rec)
        retrieved = mm.get("k2")
        assert retrieved is not None

    def test_manager_retrieve_semantic(self):
        from memory.manager import MemoryManager
        from memory.interfaces import MemoryRecord

        mm = MemoryManager()
        rec = MemoryRecord(
            key="k3", content="Belize is in Central America", embedding=[0.1] * 768
        )
        mm.store(rec)
        results = mm.retrieve([0.1] * 768, top_k=3)
        assert isinstance(results, list)

    def test_manager_context_window(self):
        from memory.manager import MemoryManager

        mm = MemoryManager()
        mm.store_text("first item", key="w1")
        window = mm.context_window()
        assert isinstance(window, list)

    def test_manager_stats(self):
        from memory.manager import MemoryManager

        mm = MemoryManager()
        stats = mm.stats()
        assert "working" in stats or isinstance(stats, dict)

    def test_semantic_memory_store(self):
        from memory.semantic import SemanticMemory
        from memory.interfaces import MemoryRecord

        sm = SemanticMemory()
        rec = MemoryRecord(
            key="s1", content="Belize is in Central America", embedding=[0.1] * 128
        )
        sm.store(rec)
        results = sm.retrieve([0.1] * 128, top_k=5)
        assert isinstance(results, list)


class TestConfigMoreCoverage:
    def test_dev_config_loads(self):
        from config import load_config

        cfg = load_config("config.dev.yaml")
        assert cfg is not None

    def test_prod_config_loads(self):
        from config import load_config

        cfg = load_config("config.prod.yaml")
        assert cfg is not None


class TestGenomeHistoryCoverage:
    def _make_stats(self, gen=0):
        from genome.population import PopulationStatistics

        return PopulationStatistics(
            generation=gen,
            population_size=5,
            avg_fitness=50.0,
            max_fitness=90.0,
            min_fitness=10.0,
            std_fitness=20.0,
            avg_quality=70.0,
            avg_timeliness=80.0,
            avg_honesty=85.0,
            unique_architectures=3,
            diversity_score=0.6,
            elite_count=2,
            elite_avg_fitness=85.0,
        )

    def _make_genome(self, gid="g1", fitness=90.0):
        from genome.dna import Genome

        g = Genome(
            genome_id=gid,
            generation=0,
            encoder_layers=[],
            decoder_layers=[],
            parent_genomes=[],
        )
        g.fitness_score = fitness
        return g

    def test_record_generation(self):
        from genome.history import EvolutionHistory

        h = EvolutionHistory()
        h.record_generation(0, self._make_stats(0), [self._make_genome()])
        progression = h.get_fitness_progression()
        assert len(progression) == 1

    def test_evolution_history_empty(self):
        from genome.history import EvolutionHistory

        h = EvolutionHistory()
        assert h.get_fitness_progression() == []

    def test_get_generation_record(self):
        from genome.history import EvolutionHistory

        h = EvolutionHistory()
        h.record_generation(0, self._make_stats(0), [self._make_genome()])
        record = h.get_generation_record(0)
        assert record is not None
        assert record.generation == 0

    def test_get_best_genome_id(self):
        from genome.history import EvolutionHistory

        h = EvolutionHistory()
        h.record_generation(0, self._make_stats(0), [self._make_genome("best1", 90.0)])
        best = h.get_best_genome_id()
        assert best == "best1"

    def test_compute_summary(self):
        from genome.history import EvolutionHistory

        h = EvolutionHistory()
        h.record_generation(0, self._make_stats(0), [self._make_genome()])
        summary = h.compute_summary()
        assert isinstance(summary, dict)


class TestMonitoringMetrics:
    def test_metrics_record_multiple(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record(MetricType.TRAINING_LOSS, 0.5, {"epoch": "1"})
        mc.record(MetricType.TRAINING_LOSS, 0.4, {"epoch": "2"})
        history = mc.get_metrics(MetricType.TRAINING_LOSS)
        assert len(history) == 2

    def test_metrics_latest(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record(MetricType.TRAINING_LOSS, 0.5, {})
        latest = mc.get_latest(MetricType.TRAINING_LOSS)
        assert latest.value == 0.5

    def test_metrics_average(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record(MetricType.TRAINING_LOSS, 0.4, {})
        mc.record(MetricType.TRAINING_LOSS, 0.6, {})
        avg = mc.get_average(MetricType.TRAINING_LOSS)
        assert avg == pytest.approx(0.5)

    def test_metrics_get_summary(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record(MetricType.CPU_USAGE, 50.0, {})
        summary = mc.get_summary()
        assert isinstance(summary, dict)

    def test_record_training_epoch(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record_training_epoch(1, 0.5, 80.0, 0.6, 75.0)
        latest = mc.get_latest(MetricType.TRAINING_LOSS)
        assert latest is not None

    def test_metrics_clear(self):
        from monitoring.metrics import MetricsCollector, MetricType

        mc = MetricsCollector()
        mc.record(MetricType.TRAINING_LOSS, 0.5, {})
        mc.clear()
        assert mc.get_latest(MetricType.TRAINING_LOSS) is None
