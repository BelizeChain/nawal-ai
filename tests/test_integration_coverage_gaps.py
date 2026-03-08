"""
Coverage gap tests for integration/ module.

Covers:
- integration/oracle_pipeline.py (85 miss lines):
  - OracleDataFetcher.get_device_info (parsing path)
  - OracleDataFetcher.get_pending_submissions (full flow)
  - OracleDataFetcher.get_operator_stats
  - DataPreprocessor.__init__, get_model, preprocess
  - ModelInferenceRunner.run_inference (success + error), get_stats
  - ResultSubmitter.__init__, submit_prediction (success/failure), claim_rewards
  - OraclePipeline.__init__, process_submission, process_pending_submissions
  - OraclePipeline.process_loop

- integration/kinich_connector.py (30 miss lines):
  - KinichQuantumConnector._init_kinich_connection (health OK + error)
  - KinichQuantumConnector.quantum_process (cache hit, bridge, fallback)
  - KinichQuantumConnector._quantum_forward (aiohttp path)
  - KinichQuantumConnector._vqc_forward, _qsvm_forward, _qnn_forward
  - QuantumEnhancedLayer.forward (sync + running loop)
  - QuantumEnhancedLayer.extra_repr
"""

import asyncio
import time
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Oracle pipeline imports
# ---------------------------------------------------------------------------
from integration.oracle_pipeline import (
    OracleDataFetcher,
    DataPreprocessor,
    ModelInferenceRunner,
    ResultSubmitter,
    OraclePipeline,
    IoTDataSubmission,
    IoTDeviceInfo,
    DeviceType,
)

# ModelDomain used in oracle_pipeline
from nawal.client.domain_models import ModelDomain

# Kinich imports
from integration.kinich_connector import (
    KinichQuantumConnector,
    QuantumEnhancedLayer,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_submission() -> IoTDataSubmission:
    return IoTDataSubmission(
        device_id=b'\x01' * 32,
        data=b'\xDE\xAD' * 10,
        feed_type="sensor",
        location=(17.5, -88.2),
        timestamp=1700000000,
        quality_metrics={"accuracy": 95},
        metadata={"version": "1.0"},
    )


def _make_device_info() -> IoTDeviceInfo:
    return IoTDeviceInfo(
        device_id=b'\x01' * 32,
        device_type=DeviceType.SENSOR,
        domain=ModelDomain.AGRITECH,
        operator="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        location=(17.5, -88.2),
        reputation_score=90,
        total_submissions=100,
        is_verified=True,
        registration_block=1000,
    )


# ===========================================================================
# OracleDataFetcher
# ===========================================================================


class TestOracleDataFetcher:
    """Tests for OracleDataFetcher with mocked SubstrateInterface."""

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_init(self, mock_substrate_cls):
        fetcher = OracleDataFetcher("ws://test:9944")
        mock_substrate_cls.assert_called_once()
        assert fetcher.substrate is not None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_device_info_found(self, mock_substrate_cls):
        """Full success path for get_device_info -> IoTDeviceInfo."""
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        # Mock the query result
        mock_result = MagicMock()
        mock_result.value = {
            "device_type": 2,  # SENSOR
            "domain_index": 0,  # AGRITECH
            "operator": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "location": {"lat": 17.5, "lon": -88.2},
            "reputation_score": 90,
            "total_submissions": 100,
            "is_verified": True,
            "registration_block": 1000,
        }
        mock_sub.query.return_value = mock_result

        fetcher = OracleDataFetcher()
        info = fetcher.get_device_info(b'\x01' * 32)

        assert info is not None
        assert info.device_type == DeviceType.SENSOR
        assert info.reputation_score == 90

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_device_info_none(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_result = MagicMock()
        mock_result.value = None
        mock_sub.query.return_value = mock_result

        fetcher = OracleDataFetcher()
        info = fetcher.get_device_info(b'\x01' * 32)
        assert info is None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_device_info_exception(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_sub.query.side_effect = Exception("connection refused")

        fetcher = OracleDataFetcher()
        info = fetcher.get_device_info(b'\x01' * 32)
        assert info is None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_pending_submissions(self, mock_substrate_cls):
        """Full get_pending_submissions flow: returns parsed submissions list."""
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        device_id_hex = ("01" * 32)

        # Mock PendingSubmissions query
        pending_result = MagicMock()
        pending_result.value = [
            (device_id_hex, {
                "data": [0xDE, 0xAD],
                "feed_type": "sensor",
                "location": {"lat": 17.5, "lon": -88.2},
                "timestamp": 1700000000,
                "quality_metrics": {"accuracy": 95},
                "metadata": {"v": 1},
            }),
        ]

        # Mock IoTDevices query for get_device_info
        device_result = MagicMock()
        device_result.value = {
            "device_type": 2,
            "domain_index": 1,  # AGRITECH
            "operator": "5Abc",
            "location": None,
            "reputation_score": 80,
            "total_submissions": 50,
            "is_verified": True,
            "registration_block": 500,
        }

        # First query = PendingSubmissions, second = IoTDevices
        mock_sub.query.side_effect = [pending_result, device_result]

        fetcher = OracleDataFetcher()
        subs = fetcher.get_pending_submissions(domain=ModelDomain.AGRITECH, limit=10)
        assert len(subs) == 1
        assert subs[0].feed_type == "sensor"

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_pending_submissions_empty(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_result = MagicMock()
        mock_result.value = None
        mock_sub.query.return_value = mock_result

        fetcher = OracleDataFetcher()
        subs = fetcher.get_pending_submissions()
        assert subs == []

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_pending_submissions_exception(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_sub.query.side_effect = Exception("timeout")

        fetcher = OracleDataFetcher()
        subs = fetcher.get_pending_submissions()
        assert subs == []

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_operator_stats_found(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_result = MagicMock()
        mock_result.value = {"total_devices": 5, "active_devices": 3}
        mock_sub.query.return_value = mock_result

        fetcher = OracleDataFetcher()
        stats = fetcher.get_operator_stats("5Abc")
        assert stats["total_devices"] == 5

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_operator_stats_none(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_result = MagicMock()
        mock_result.value = None
        mock_sub.query.return_value = mock_result

        fetcher = OracleDataFetcher()
        stats = fetcher.get_operator_stats("5Abc")
        assert stats["total_devices"] == 0

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_get_operator_stats_exception(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_sub.query.side_effect = Exception("err")

        fetcher = OracleDataFetcher()
        stats = fetcher.get_operator_stats("5Abc")
        assert stats == {}


# ===========================================================================
# DataPreprocessor
# ===========================================================================


class TestDataPreprocessor:
    def test_init(self):
        prep = DataPreprocessor(device="cpu")
        assert prep.device == "cpu"
        assert prep.models == {}

    @patch("integration.oracle_pipeline.DomainModelFactory")
    def test_get_model_caches(self, mock_factory):
        mock_model = MagicMock()
        mock_factory.create_model.return_value = mock_model

        prep = DataPreprocessor()
        m1 = prep.get_model(ModelDomain.AGRITECH)
        m2 = prep.get_model(ModelDomain.AGRITECH)
        assert m1 is m2
        mock_factory.create_model.assert_called_once()

    @patch("integration.oracle_pipeline.DomainModelFactory")
    def test_preprocess(self, mock_factory):
        mock_model = MagicMock()
        mock_model.preprocess_data.return_value = torch.randn(1, 64)
        mock_factory.create_model.return_value = mock_model

        prep = DataPreprocessor()
        submission = _make_submission()
        device_info = _make_device_info()

        tensor, model = prep.preprocess(submission, device_info)
        assert isinstance(tensor, torch.Tensor)
        assert model is mock_model
        mock_model.preprocess_data.assert_called_once()


# ===========================================================================
# ModelInferenceRunner
# ===========================================================================


class TestModelInferenceRunner:
    def test_init(self):
        runner = ModelInferenceRunner()
        assert runner.inference_stats["total_inferences"] == 0

    def test_run_inference_success(self):
        runner = ModelInferenceRunner()
        mock_model = MagicMock()
        predictions = {"predictions": torch.ones(1)}
        mock_model.forward.return_value = predictions

        input_tensor = torch.randn(1, 64)
        result = runner.run_inference(mock_model, input_tensor, ModelDomain.AGRITECH)

        assert result is predictions
        assert runner.inference_stats["total_inferences"] == 1
        assert "AGRITECH" in runner.inference_stats["domain_breakdown"]

    def test_run_inference_error(self):
        runner = ModelInferenceRunner()
        mock_model = MagicMock()
        mock_model.forward.side_effect = RuntimeError("GPU OOM")

        input_tensor = torch.randn(1, 64)
        result = runner.run_inference(mock_model, input_tensor, ModelDomain.MARINE)

        assert "error" in result
        assert "GPU OOM" in result["error"]

    def test_get_stats_with_average(self):
        runner = ModelInferenceRunner()
        runner.inference_stats["total_inferences"] = 10
        runner.inference_stats["total_time_ms"] = 100.0
        stats = runner.get_stats()
        assert stats["average_time_ms"] == 10.0

    def test_get_stats_zero_inferences(self):
        runner = ModelInferenceRunner()
        stats = runner.get_stats()
        assert "average_time_ms" not in stats


# ===========================================================================
# ResultSubmitter
# ===========================================================================


class TestResultSubmitter:
    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_init_no_keypair_raises(self, mock_substrate_cls):
        with pytest.raises(ValueError, match="Keypair is required"):
            ResultSubmitter(keypair=None)

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_submit_prediction_success(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        mock_receipt = MagicMock()
        mock_receipt.is_success = True
        mock_receipt.extrinsic_hash = "0xABCDEF"
        mock_sub.submit_extrinsic.return_value = mock_receipt

        mock_keypair = MagicMock()
        submitter = ResultSubmitter(keypair=mock_keypair)

        submission = _make_submission()
        device_info = _make_device_info()
        predictions = {"predictions": torch.ones(1)}

        with patch("integration.oracle_pipeline.prepare_oracle_submission") as mock_prep:
            mock_prep.return_value = {
                "device_id": b'\x01' * 32,
                "data": b'\xDE\xAD',
                "feed_type": "sensor",
                "quality_score": 95,
                "location": None,
            }
            tx_hash = submitter.submit_prediction(submission, predictions, device_info)

        assert tx_hash == "0xABCDEF"

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_submit_prediction_failure(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        mock_receipt = MagicMock()
        mock_receipt.is_success = False
        mock_receipt.error_message = "nonce error"
        mock_sub.submit_extrinsic.return_value = mock_receipt

        mock_keypair = MagicMock()
        submitter = ResultSubmitter(keypair=mock_keypair)

        with patch("integration.oracle_pipeline.prepare_oracle_submission") as mock_prep:
            mock_prep.return_value = {
                "device_id": b'\x01', "data": b'', "feed_type": "s",
                "quality_score": 0
            }
            result = submitter.submit_prediction(
                _make_submission(), {"p": torch.zeros(1)}, _make_device_info()
            )

        assert result is None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_submit_prediction_exception(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_sub.compose_call.side_effect = Exception("network error")

        mock_keypair = MagicMock()
        submitter = ResultSubmitter(keypair=mock_keypair)
        result = submitter.submit_prediction(
            _make_submission(), {"p": torch.zeros(1)}, _make_device_info()
        )
        assert result is None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_claim_rewards_success(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        mock_receipt = MagicMock()
        mock_receipt.is_success = True
        mock_receipt.extrinsic_hash = "0x1234"
        mock_sub.submit_extrinsic.return_value = mock_receipt

        mock_keypair = MagicMock()
        mock_keypair.ss58_address = "5Grw"
        submitter = ResultSubmitter(keypair=mock_keypair)

        tx = submitter.claim_rewards()
        assert tx == "0x1234"

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_claim_rewards_failure(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub

        mock_receipt = MagicMock()
        mock_receipt.is_success = False
        mock_receipt.error_message = "insufficient balance"
        mock_sub.submit_extrinsic.return_value = mock_receipt

        mock_keypair = MagicMock()
        mock_keypair.ss58_address = "5Grw"
        submitter = ResultSubmitter(keypair=mock_keypair)

        tx = submitter.claim_rewards(operator="5Other")
        assert tx is None

    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_claim_rewards_exception(self, mock_substrate_cls):
        mock_sub = MagicMock()
        mock_substrate_cls.return_value = mock_sub
        mock_sub.compose_call.side_effect = Exception("timeout")

        mock_keypair = MagicMock()
        mock_keypair.ss58_address = "5Grw"
        submitter = ResultSubmitter(keypair=mock_keypair)

        tx = submitter.claim_rewards()
        assert tx is None


# ===========================================================================
# OraclePipeline
# ===========================================================================


class TestOraclePipeline:
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    def test_init(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(
            substrate_url="ws://test:9944",
            device="cpu",
            keypair=mock_keypair,
        )
        assert pipeline.fetcher is not None
        assert pipeline.preprocessor is not None
        assert pipeline.runner is not None

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_submission_success(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)

        # Mock fetcher
        pipeline.fetcher = MagicMock()
        pipeline.fetcher.get_device_info.return_value = _make_device_info()

        # Mock preprocessor
        pipeline.preprocessor = MagicMock()
        pipeline.preprocessor.preprocess.return_value = (torch.randn(1, 64), MagicMock())

        # Mock runner
        pipeline.runner = MagicMock()
        pipeline.runner.run_inference.return_value = {"predictions": torch.ones(1)}

        # Mock submitter
        pipeline.submitter = MagicMock()
        pipeline.submitter.submit_prediction.return_value = "0xABC"

        result = await pipeline.process_submission(_make_submission())
        assert result is True

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_submission_device_not_found(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)
        pipeline.fetcher = MagicMock()
        pipeline.fetcher.get_device_info.return_value = None

        result = await pipeline.process_submission(_make_submission())
        assert result is False

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_submission_preprocess_error(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)

        pipeline.fetcher = MagicMock()
        pipeline.fetcher.get_device_info.return_value = _make_device_info()
        pipeline.preprocessor = MagicMock()
        pipeline.preprocessor.preprocess.side_effect = Exception("decode error")

        result = await pipeline.process_submission(_make_submission())
        assert result is False

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_pending_submissions(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)

        # Mock internals
        pipeline.fetcher = MagicMock()
        pipeline.fetcher.get_pending_submissions.return_value = [
            _make_submission(), _make_submission(),
        ]
        pipeline.fetcher.get_device_info.return_value = _make_device_info()

        pipeline.preprocessor = MagicMock()
        pipeline.preprocessor.preprocess.return_value = (torch.randn(1, 64), MagicMock())

        pipeline.runner = MagicMock()
        pipeline.runner.run_inference.return_value = {"p": torch.ones(1)}
        pipeline.runner.get_stats.return_value = {"total_inferences": 2}

        pipeline.submitter = MagicMock()
        pipeline.submitter.submit_prediction.return_value = "0x1"

        results = await pipeline.process_pending_submissions(domain=ModelDomain.AGRITECH, limit=10)
        assert results["total"] == 2
        assert results["success"] == 2
        assert results["failed"] == 0

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_loop_stops_on_keyboard_interrupt(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)

        pipeline.fetcher = MagicMock()
        pipeline.fetcher.get_pending_submissions.return_value = []
        pipeline.runner = MagicMock()
        pipeline.runner.get_stats.return_value = {}

        # Simulate KeyboardInterrupt on second iteration
        call_count = 0

        async def mock_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await pipeline.process_loop(interval_seconds=1)

    @pytest.mark.asyncio
    @patch("integration.oracle_pipeline.ResultSubmitter")
    @patch("integration.oracle_pipeline.SubstrateInterface")
    async def test_process_loop_handles_exception(self, mock_substrate_cls, mock_submitter_cls):
        mock_keypair = MagicMock()
        pipeline = OraclePipeline(keypair=mock_keypair)

        call_count = 0

        # First call raises exception, second raises KeyboardInterrupt
        original_method = pipeline.process_pending_submissions

        async def mock_process(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            raise KeyboardInterrupt()

        pipeline.process_pending_submissions = mock_process

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await pipeline.process_loop(interval_seconds=1)


# ===========================================================================
# KinichQuantumConnector
# ===========================================================================


class TestKinichQuantumConnector:
    @patch("urllib.request.urlopen")
    def test_init_health_check_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        conn = KinichQuantumConnector(kinich_endpoint="http://test:8002")
        assert conn.kinich_available is True

    @patch("urllib.request.urlopen")
    def test_init_health_check_url_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(kinich_endpoint="http://test:8002")
        assert conn.kinich_available is False

    @patch("urllib.request.urlopen")
    def test_init_fallback_disabled_raises(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("fatal")

        with pytest.raises(Exception, match="fatal"):
            KinichQuantumConnector(
                kinich_endpoint="http://test:8002",
                fallback_to_classical=False,
            )

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_quantum_process_classical_fallback(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=16, quantum_dim=4)
        features = np.random.randn(2, 16).astype(np.float64)
        result = await conn.quantum_process(features)
        assert result.shape == (2, 16)
        assert conn.stats["fallback_calls"] >= 1

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_quantum_process_cache_hit(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=16, quantum_dim=4, enable_caching=True)
        features = np.ones((2, 16))

        # First call -> no cache
        result1 = await conn.quantum_process(features)
        # Second call -> cache hit
        result2 = await conn.quantum_process(features)

        assert conn.stats["cache_hits"] >= 1
        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_quantum_process_cache_eviction(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2, enable_caching=True)
        conn._cache_max_size = 2  # Tiny cache

        # Fill cache
        for i in range(3):
            features = np.full((1, 4), float(i))
            await conn.quantum_process(features)

        # Cache should have evicted oldest
        assert len(conn.result_cache) <= 2

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_quantum_process_with_bridge_and_fallback(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=16, quantum_dim=4)
        conn.kinich_available = True
        conn.bridge = MagicMock()  # Force bridge path

        # _quantum_forward will fail (aiohttp not connecting)
        # Then falls back to classical
        features = np.random.randn(2, 16)
        with patch.object(conn, "_quantum_forward", side_effect=Exception("aiohttp error")):
            result = await conn.quantum_process(features)
        assert result.shape == (2, 16)
        assert conn.stats["fallback_calls"] >= 1

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_quantum_process_bridge_no_fallback_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(
            classical_dim=16, quantum_dim=4, fallback_to_classical=False, enable_caching=False
        )
        conn.kinich_available = True
        conn.bridge = MagicMock()

        features = np.random.randn(2, 16)
        with patch.object(conn, "_quantum_forward", side_effect=RuntimeError("quantum error")):
            with pytest.raises(RuntimeError, match="quantum error"):
                await conn.quantum_process(features)

    @pytest.mark.asyncio
    async def test_quantum_forward_aiohttp(self):
        """Test _quantum_forward via mocked aiohttp."""
        conn = KinichQuantumConnector.__new__(KinichQuantumConnector)
        conn.kinich_endpoint = "http://test:8002"
        conn.classical_dim = 16
        conn.quantum_dim = 4

        features = np.random.randn(2, 16)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "quantum_enhanced_features": features.tolist()
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await conn._quantum_forward(features, "vqc")

        assert result.shape == (2, 16)

    @pytest.mark.asyncio
    async def test_quantum_forward_aiohttp_error(self):
        """Test _quantum_forward raises on non-200."""
        conn = KinichQuantumConnector.__new__(KinichQuantumConnector)
        conn.kinich_endpoint = "http://test:8002"
        conn.classical_dim = 16
        conn.quantum_dim = 4

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            with pytest.raises(RuntimeError, match="Kinich API error"):
                await conn._quantum_forward(np.random.randn(2, 16), "vqc")

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_vqc_forward(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        features = np.random.randn(2, 4)

        with patch.object(conn, "_quantum_forward", new_callable=AsyncMock,
                          return_value=features) as mock_qf:
            result = await conn._vqc_forward(features)
            mock_qf.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_qsvm_forward(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        features = np.random.randn(2, 4)

        result = await conn._qsvm_forward(features, num_classes=3)
        assert result.shape == (2, 3)

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_qnn_forward(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        features = np.random.randn(2, 4)

        with patch.object(conn, "_quantum_forward", new_callable=AsyncMock,
                          return_value=features) as mock_qf:
            result = await conn._qnn_forward(features)
            mock_qf.assert_awaited_once()

    @patch("urllib.request.urlopen")
    def test_get_statistics(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        stats = conn.get_statistics()
        assert "total_calls" in stats
        assert "kinich_available" in stats
        assert stats["kinich_available"] is False

    @patch("urllib.request.urlopen")
    def test_clear_cache(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        conn.result_cache["test"] = np.array([1])
        conn.clear_cache()
        assert len(conn.result_cache) == 0

    @patch("urllib.request.urlopen")
    def test_reset_statistics(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        conn.stats["quantum_calls"] = 100
        conn.reset_statistics()
        assert conn.stats["quantum_calls"] == 0

    @patch("urllib.request.urlopen")
    def test_repr(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        conn = KinichQuantumConnector(classical_dim=4, quantum_dim=2)
        r = repr(conn)
        assert "KinichQuantumConnector" in r
        assert "4" in r and "2" in r


# ===========================================================================
# QuantumEnhancedLayer
# ===========================================================================


class TestQuantumEnhancedLayer:
    @patch("urllib.request.urlopen")
    def test_init(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        layer = QuantumEnhancedLayer(classical_dim=16, quantum_dim=4)
        assert layer.classical_dim == 16
        assert layer.quantum_dim == 4

    @patch("urllib.request.urlopen")
    def test_forward_no_running_loop(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        layer = QuantumEnhancedLayer(classical_dim=16, quantum_dim=4)

        # Mock quantum_process to return numpy array
        async def mock_qp(x_np, model_type="vqc"):
            return x_np  # Identity transform

        layer.connector.quantum_process = mock_qp

        x = torch.randn(2, 16)
        result = layer.forward(x)
        assert result.shape == (2, 16)

    @patch("urllib.request.urlopen")
    def test_forward_with_running_loop(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        layer = QuantumEnhancedLayer(classical_dim=16, quantum_dim=4)

        async def mock_qp(x_np, model_type="vqc"):
            return x_np

        layer.connector.quantum_process = mock_qp

        # Run from inside a running event loop
        async def run_in_loop():
            x = torch.randn(2, 16)
            return layer.forward(x)

        result = asyncio.run(run_in_loop())
        assert result.shape == (2, 16)

    @patch("urllib.request.urlopen")
    def test_extra_repr(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        layer = QuantumEnhancedLayer(classical_dim=16, quantum_dim=4)
        s = layer.extra_repr()
        assert "classical_dim=16" in s
        assert "quantum_dim=4" in s
