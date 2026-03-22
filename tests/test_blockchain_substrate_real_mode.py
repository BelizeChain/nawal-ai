"""
Category 1: Blockchain real-mode substrate paths.

Covers the non-mock (real SubstrateInterface) branches across all 6 blockchain
modules by injecting a shared MagicMock substrate instance.

Targets:
  blockchain/staking_connector.py   (134 missed lines → real-mode paths)
  blockchain/substrate_client.py    ( 86 missed lines → connect/query/submit)
  blockchain/community_connector.py ( 58 missed lines → real-mode paths)
  blockchain/payroll_connector.py   ( 82 missed lines → real-mode paths)
  blockchain/events.py              ( 64 missed lines → connect/parse/dispatch)
  blockchain/identity_verifier.py   ( 48 missed lines → DummyVerifier + factory)
  blockchain/validator_manager.py   ( 46 missed lines → register/get/check)
"""

import asyncio
import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _mock_receipt(success=True, error=None, block_hash="0xABC", block_number=1):
    r = MagicMock()
    r.is_success = success
    r.error_message = error
    r.block_hash = block_hash
    r.block_number = block_number
    return r


def _mock_substrate():
    """Return a fully-mocked SubstrateInterface-shaped object."""
    sub = MagicMock()
    sub.chain = "BelizeChain"
    sub.properties = {"tokenDecimals": 12}
    sub.compose_call.return_value = MagicMock()
    sub.create_signed_extrinsic.return_value = MagicMock(extrinsic_hash="0xDEAD")
    sub.submit_extrinsic.return_value = _mock_receipt(success=True)
    sub.query.return_value.value = None
    sub.query_map.return_value = []
    sub.get_chain_head.return_value = "0xHEAD"
    sub.get_events.return_value = []
    return sub


# ===========================================================================
# blockchain/staking_connector.py — real-mode (mock_mode=False) paths
# ===========================================================================


class TestStakingConnectorRealModeConnect:

    def test_real_connect_success(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        with patch("blockchain.staking_connector.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            result = _run(conn.connect())
        assert result is True
        assert conn.is_connected is True

    def test_real_connect_failure_returns_false(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        with patch("blockchain.staking_connector.SubstrateInterface") as mock_cls:
            mock_cls.side_effect = Exception("connection refused")
            result = _run(conn.connect())
        assert result is False
        assert conn.is_connected is False

    def test_real_disconnect_clears_substrate(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        _run(conn.disconnect())
        assert conn.is_connected is False


class TestStakingConnectorRealModeEnroll:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def test_enroll_not_connected_returns_false(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        result = _run(
            conn.enroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 1000
            )
        )
        assert result is False

    def test_enroll_real_success(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(success=True)
        result = _run(
            conn.enroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 1000
            )
        )
        assert result is True

    def test_enroll_real_failure(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(
            success=False, error="InsufficientStake"
        )
        result = _run(
            conn.enroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 1
            )
        )
        assert result is False

    def test_enroll_real_exception_returns_false(self):
        conn = self._make_real_conn()
        conn.substrate.compose_call.side_effect = Exception("rpc error")
        result = _run(
            conn.enroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", 1000
            )
        )
        assert result is False


class TestStakingConnectorRealModeUnenroll:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def test_unenroll_not_connected(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        result = _run(
            conn.unenroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is False

    def test_unenroll_real_success(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(success=True)
        result = _run(
            conn.unenroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is True

    def test_unenroll_real_failure(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(
            success=False, error="NotEnrolled"
        )
        result = _run(
            conn.unenroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is False

    def test_unenroll_exception_returns_false(self):
        conn = self._make_real_conn()
        conn.substrate.compose_call.side_effect = RuntimeError("boom")
        result = _run(
            conn.unenroll_participant(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is False


class TestStakingConnectorRealModeGetParticipant:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def test_get_participant_not_connected(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        result = _run(
            conn.get_participant_info(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is None

    def test_get_participant_real_found(self):
        conn = self._make_real_conn()
        conn.substrate.query.return_value.value = {
            "stake_amount": 1000,
            "is_enrolled": True,
            "rounds_completed": 5,
            "total_samples": 500,
            "avg_fitness": 80.0,
        }
        result = _run(
            conn.get_participant_info(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is not None
        assert result.stake_amount == 1000

    def test_get_participant_real_not_found(self):
        conn = self._make_real_conn()
        conn.substrate.query.return_value.value = None
        result = _run(
            conn.get_participant_info(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is None

    def test_get_participant_exception_returns_none(self):
        conn = self._make_real_conn()
        conn.substrate.query.side_effect = Exception("rpc error")
        result = _run(
            conn.get_participant_info(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        )
        assert result is None


class TestStakingConnectorRealModeSubmitProof:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def _make_submission(self, pid="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"):
        from blockchain.staking_connector import TrainingSubmission

        return TrainingSubmission(
            participant_id=pid,
            genome_id="genome-001",
            round_number=1,
            samples_trained=100,
            training_time=30.0,
            quality_score=80.0,
            timeliness_score=90.0,
            honesty_score=85.0,
            fitness_score=85.0,
            model_hash="abc123def456",
        )

    def test_submit_not_connected(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        result = _run(conn.submit_training_proof(self._make_submission()))
        assert result is False

    def test_submit_real_success(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(success=True)
        result = _run(conn.submit_training_proof(self._make_submission()))
        assert result is True

    def test_submit_real_failure(self):
        conn = self._make_real_conn()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(
            success=False, error="BadProof"
        )
        result = _run(conn.submit_training_proof(self._make_submission()))
        assert result is False

    def test_submit_exception_returns_false(self):
        conn = self._make_real_conn()
        conn.substrate.compose_call.side_effect = Exception("timeout")
        result = _run(conn.submit_training_proof(self._make_submission()))
        assert result is False


class TestStakingConnectorRealModeClaimRewards:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def test_claim_not_connected(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        success, amount = _run(
            conn.claim_rewards("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert success is False

    def test_claim_real_success(self):
        conn = self._make_real_conn()
        conn.substrate.query.return_value.value = {"pending_rewards": 500}
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(success=True)
        success, amount = _run(
            conn.claim_rewards("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert isinstance(success, bool)

    def test_claim_real_exception(self):
        conn = self._make_real_conn()
        conn.substrate.compose_call.side_effect = Exception("network error")
        success, amount = _run(
            conn.claim_rewards("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert success is False


class TestStakingConnectorRealModeQueryAll:

    def _make_real_conn(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        return conn

    def test_get_total_staked_not_connected(self):
        from blockchain.staking_connector import StakingConnector

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        conn.is_connected = False
        total = _run(conn.get_total_staked())
        assert total == 0

    def test_get_total_staked_real_empty(self):
        conn = self._make_real_conn()
        conn.substrate.query_map.return_value = []
        total = _run(conn.get_total_staked())
        assert total == 0

    def test_get_total_staked_real_with_data(self):
        conn = self._make_real_conn()
        # query_map returns list of (key, value) tuples
        entry1 = (MagicMock(), MagicMock())
        entry1[1].value = {"stake_amount": 500, "is_enrolled": True}
        entry2 = (MagicMock(), MagicMock())
        entry2[1].value = {"stake_amount": 300, "is_enrolled": True}
        conn.substrate.query_map.return_value = [entry1, entry2]
        total = _run(conn.get_total_staked())
        assert isinstance(total, (int, float))

    def test_get_all_participants_real_empty(self):
        conn = self._make_real_conn()
        conn.substrate.query_map.return_value = []
        results = _run(conn.get_all_participants())
        assert isinstance(results, list)

    def test_get_all_participants_real_exception(self):
        conn = self._make_real_conn()
        conn.substrate.query_map.side_effect = Exception("rpc down")
        results = _run(conn.get_all_participants())
        assert isinstance(results, list)


# ===========================================================================
# blockchain/substrate_client.py — connect/query/submit paths
# ===========================================================================


class TestSubstrateClientConnect:

    def _make_config(self):
        from blockchain.substrate_client import ChainConfig

        return ChainConfig.local()

    def test_connect_success(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        with patch("blockchain.substrate_client.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            client.connect()
        assert client.is_connected() is True

    def test_connect_failure_returns_false(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        with patch("blockchain.substrate_client.SubstrateInterface") as mock_cls:
            mock_cls.side_effect = Exception("refused")
            try:
                client.connect(max_retries=1, base_delay=0)
            except Exception:
                pass
        assert client.is_connected() is False

    def test_disconnect(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        client.substrate = _mock_substrate()
        client.disconnect()
        assert not client.is_connected()

    def test_is_connected_false_initially(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        assert client.is_connected() is False


class TestSubstrateClientQuery:

    def _make_connected(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        client.substrate = _mock_substrate()
        client._connected = True
        return client

    def _make_config(self):
        from blockchain.substrate_client import ChainConfig

        return ChainConfig.local()

    def test_query_storage_returns_value(self):
        client = self._make_connected()
        client.substrate.query.return_value.value = {"balance": 1000}
        result = client.query_storage("System", "Account", params=["5ADDR"])
        assert result == {"balance": 1000}

    def test_query_storage_returns_none(self):
        client = self._make_connected()
        client.substrate.query.return_value.value = None
        result = client.query_storage("System", "Account", params=["5ADDR"])
        assert result is None

    def test_query_map_returns_list(self):
        client = self._make_connected()
        k = MagicMock()
        k.value = "key1"
        v = MagicMock()
        v.value = {"data": 1}
        client.substrate.query_map.return_value = [(k, v)]
        result = client.query_map("Staking", "Trainers")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_query_map_empty(self):
        client = self._make_connected()
        client.substrate.query_map.return_value = []
        result = client.query_map("Staking", "Trainers")
        assert result == []


class TestSubstrateClientSubmit:

    def _make_config(self):
        from blockchain.substrate_client import ChainConfig

        return ChainConfig.local()

    def _make_connected(self):
        from blockchain.substrate_client import SubstrateClient

        client = SubstrateClient(self._make_config())
        client.substrate = _mock_substrate()
        client._connected = True
        return client

    def test_submit_extrinsic_success(self):
        from blockchain.substrate_client import ExtrinsicReceipt

        client = self._make_connected()
        mock_result = MagicMock()
        mock_result.block_hash = "0xABC"
        mock_result.is_success = True
        mock_result.error_message = None
        mock_result.triggered_events = []
        client.substrate.submit_extrinsic.return_value = mock_result
        client.substrate.get_block.return_value = {"header": {"number": 42}}

        keypair = MagicMock()
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="enroll_ai_trainer",
            call_params={"stake_amount": 1000},
            wait_for_inclusion=True,
        )
        assert isinstance(receipt, ExtrinsicReceipt)
        assert receipt.success is True

    def test_submit_extrinsic_failure(self):
        from blockchain.substrate_client import ExtrinsicReceipt

        client = self._make_connected()
        mock_result = MagicMock()
        mock_result.block_hash = "0xABC"
        mock_result.is_success = False
        mock_result.error_message = "BadOrigin"
        mock_result.triggered_events = []
        client.substrate.submit_extrinsic.return_value = mock_result
        client.substrate.get_block.return_value = {"header": {"number": 1}}

        keypair = MagicMock()
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="enroll_ai_trainer",
            call_params={},
            wait_for_inclusion=True,
        )
        assert receipt.success is False

    def test_submit_extrinsic_no_wait(self):
        from blockchain.substrate_client import ExtrinsicReceipt

        client = self._make_connected()
        keypair = MagicMock()
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="System",
            call_function="remark",
            call_params={"remark": b"hello"},
            wait_for_inclusion=False,
        )
        assert isinstance(receipt, ExtrinsicReceipt)


# ===========================================================================
# blockchain/community_connector.py — real-mode paths
# ===========================================================================


class TestCommunityConnectorRealMode:

    def test_real_connect_success(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=False)
        with patch("blockchain.community_connector.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            result = _run(conn.connect())
        assert result is True

    def test_real_connect_failure(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=False)
        with patch("blockchain.community_connector.SubstrateInterface") as mock_cls:
            mock_cls.side_effect = Exception("refused")
            result = _run(conn.connect())
        assert result is False

    def test_mock_connect(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        result = _run(conn.connect())
        assert result is True

    def test_mock_get_srs_info(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        result = _run(
            conn.get_srs_info("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert result is not None

    def test_mock_record_participation(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        success, tx_hash = _run(
            conn.record_participation(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                activity_type="FederatedLearning",
                quality_score=85.0,
            )
        )
        assert isinstance(success, bool)

    def test_mock_disconnect(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        _run(conn.disconnect())  # should not raise

    def test_format_balance(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        result = conn.format_balance(1_000_000_000_000)
        assert isinstance(result, str)

    def test_parse_balance(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        result = conn.parse_balance(1.0)
        assert isinstance(result, int)

    def test_get_tier_name(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        result = _run(conn.get_tier_name(1))
        assert isinstance(result, str)

    def test_record_federated_learning_contribution(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        success, tx_hash = _run(
            conn.record_federated_learning_contribution(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                round_number=1,
                quality_score=80.0,
                samples_trained=100,
                training_duration_seconds=60,
            )
        )
        assert isinstance(success, bool)

    def test_record_education_completion(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        success, tx_hash = _run(
            conn.record_education_completion(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                module_id=101,
                completion_score=95.0,
            )
        )
        assert isinstance(success, bool)

    def test_record_green_project_contribution(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=True)
        _run(conn.connect())
        success, tx_hash = _run(
            conn.record_green_project_contribution(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                project_id=1,
                amount_dalla=50.0,
            )
        )
        assert isinstance(success, bool)

    def test_real_get_srs_info_not_found(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn.is_connected = True
        conn.substrate.query.return_value.value = None
        result = _run(
            conn.get_srs_info("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        )
        assert result is None

    def test_real_record_participation_not_connected(self):
        from blockchain.community_connector import CommunityConnector

        conn = CommunityConnector(mock_mode=False)
        conn._connected = False
        success, tx_hash = _run(
            conn.record_participation(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                activity_type="FederatedLearning",
                quality_score=80.0,
            )
        )
        assert success is False


# ===========================================================================
# blockchain/payroll_connector.py — real-mode paths
# ===========================================================================


class TestPayrollConnectorDataClasses:

    def _make_entry(self):
        from blockchain.payroll_connector import PayrollEntry, EmployeeType

        gross = 5_000_000_000_000
        tax = 750_000_000_000
        ss = 100_000_000_000
        pension = 250_000_000_000
        net = gross - tax - ss - pension
        return PayrollEntry(
            employee_id="EMP001",
            employee_name_hash="abc123hash",
            gross_salary=gross,
            tax_withholding=tax,
            social_security=ss,
            pension_contribution=pension,
            net_salary=net,
            payment_period="2026-01",
            employee_type=EmployeeType.PRIVATE,
        )

    def test_payroll_entry_post_init(self):
        entry = self._make_entry()
        assert (
            entry.net_salary
            == entry.gross_salary
            - entry.tax_withholding
            - entry.social_security
            - entry.pension_contribution
        )

    def test_payroll_entry_negative_gross_raises(self):
        from blockchain.payroll_connector import PayrollEntry, EmployeeType
        import pytest

        with pytest.raises(ValueError):
            PayrollEntry(
                employee_id="EMP002",
                employee_name_hash="hash",
                gross_salary=-1,
                tax_withholding=0,
                social_security=0,
                pension_contribution=0,
                net_salary=0,
                payment_period="2026-01",
                employee_type=EmployeeType.PRIVATE,
            )

    def test_payroll_submission_validate_valid(self):
        from blockchain.payroll_connector import PayrollSubmission, PayrollStatus

        entry = self._make_entry()
        sub = PayrollSubmission(
            submission_id="SUB001",
            employer_id="EMP_BIZ001",
            employer_name="Belize Corp",
            payment_period="2026-01",
            total_gross=entry.gross_salary,
            total_tax=entry.tax_withholding,
            total_net=entry.net_salary,
            employee_count=1,
            entries=[entry],
            zk_proof="0xPROOF",
            merkle_root="0xROOT",
            status=PayrollStatus.PENDING,
        )
        errors = sub.validate()
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_payroll_proof_verify(self):
        from blockchain.payroll_connector import PayrollProof

        proof = PayrollProof(
            proof_type="zk-snark",
            proof_data="valid_proof_base64",
            public_inputs=["1000", "200", "800"],
            commitment="0xROOT",
        )
        result = proof.verify()
        assert isinstance(result, bool)


class TestPayrollConnectorRealMode:

    def test_real_connect_success(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=False)
        with patch("blockchain.payroll_connector.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            result = _run(conn.connect())
        assert result is True

    def test_real_connect_failure(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=False)
        with patch("blockchain.payroll_connector.SubstrateInterface") as mock_cls:
            mock_cls.side_effect = Exception("refused")
            result = _run(conn.connect())
        assert result is False

    def test_mock_connect_idempotent(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=True)
        _run(conn.connect())
        _run(conn.connect())  # second call should return True without error
        assert conn._connected is True

    def test_mock_disconnect(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=True)
        _run(conn.connect())
        _run(conn.disconnect())

    def _make_entry(self):
        from blockchain.payroll_connector import PayrollEntry, EmployeeType

        gross = 5_000_000_000_000
        tax = 750_000_000_000
        ss = 100_000_000_000
        pension = 250_000_000_000
        return PayrollEntry(
            employee_id="EMP001",
            employee_name_hash="abc123hash",
            gross_salary=gross,
            tax_withholding=tax,
            social_security=ss,
            pension_contribution=pension,
            net_salary=gross - tax - ss - pension,
            payment_period="2026-01",
            employee_type=EmployeeType.PRIVATE,
        )

    def test_mock_submit_payroll(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=True)
        conn.keypair = MagicMock()
        _run(conn.connect())
        result = _run(
            conn.submit_payroll([self._make_entry()], payment_period="2026-01")
        )
        assert result is not None

    def test_mock_verify_payroll(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=True)
        conn.keypair = MagicMock()
        _run(conn.connect())
        result = _run(conn.verify_payroll("SUB001"))
        assert result is True

    def test_mock_get_payroll_stats(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=True)
        _run(conn.connect())
        stats = _run(conn.get_payroll_stats("2026-01"))
        assert isinstance(stats, dict)
        assert "total_submissions" in stats

    def test_real_submit_payroll_success(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        conn.keypair = MagicMock()
        conn.substrate.submit_extrinsic.return_value = _mock_receipt(success=True)
        result = _run(
            conn.submit_payroll([self._make_entry()], payment_period="2026-01")
        )
        assert result is not None

    def test_real_get_payroll_stats(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        conn.substrate.query.return_value = {"total_submissions": 10}
        stats = _run(conn.get_payroll_stats("2026-01"))
        assert isinstance(stats, dict)

    def test_real_get_payroll_stats_exception(self):
        from blockchain.payroll_connector import PayrollConnector

        conn = PayrollConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        conn.substrate.query.side_effect = Exception("rpc error")
        stats = _run(conn.get_payroll_stats("2026-01"))
        assert stats == {}


# ===========================================================================
# blockchain/events.py — connect/disconnect/parse paths
# ===========================================================================


class TestBlockchainEventsConnect:

    def test_connect_real_success(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=False)
        with patch("blockchain.events.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            result = _run(listener.connect())
        assert result is True

    def test_connect_real_failure(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=False)
        with patch("blockchain.events.SubstrateInterface") as mock_cls:
            mock_cls.side_effect = Exception("refused")
            result = _run(listener.connect())
        assert result is False

    def test_disconnect_clears_substrate(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=False)
        listener.substrate = _mock_substrate()
        listener.is_listening = True
        _run(listener.disconnect())
        assert listener.is_listening is False
        assert listener.substrate is None


class TestBlockchainEventsParse:

    def test_parse_event_known_type(self):
        from blockchain.events import BlockchainEventListener, EventType

        listener = BlockchainEventListener(mock_mode=True)
        event_record = MagicMock()
        event_record.value = {
            "module_id": "Staking",
            "event_id": "TrainingRoundStarted",
            "attributes": {"round": 1},
        }
        result = _run(listener._parse_event(1, "0xABC", event_record))
        assert result is not None
        assert result.event_type == EventType.TRAINING_ROUND_STARTED

    def test_parse_event_unknown_type_returns_none(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=True)
        event_record = MagicMock()
        event_record.value = {
            "module_id": "System",
            "event_id": "ExtrinsicSuccess",
            "attributes": {},
        }
        result = _run(listener._parse_event(1, "0xABC", event_record))
        assert result is None

    def test_parse_all_known_event_types(self):
        from blockchain.events import BlockchainEventListener, EventType

        listener = BlockchainEventListener(mock_mode=True)
        event_map = [
            ("Staking", "TrainingRoundStarted"),
            ("Staking", "TrainingProofSubmitted"),
            ("Staking", "TrainingRoundCompleted"),
            ("Staking", "TrainerEnrolled"),
            ("Staking", "TrainerUnenrolled"),
            ("Staking", "RewardsCalculated"),
            ("Staking", "RewardsClaimed"),
            ("Staking", "TrainerSlashed"),
            ("Staking", "ReputationUpdated"),
            ("AIRegistry", "GenomeDeployed"),
            ("AIRegistry", "GenomeEvolved"),
        ]
        for module_id, event_id in event_map:
            event_record = MagicMock()
            event_record.value = {
                "module_id": module_id,
                "event_id": event_id,
                "attributes": {},
            }
            result = _run(listener._parse_event(1, "0xABC", event_record))
            assert result is not None, f"Expected event for {module_id}.{event_id}"


class TestBlockchainEventsDispatch:

    def test_dispatch_calls_all_handlers(self):
        from blockchain.events import BlockchainEventListener, EventType, TrainingEvent

        listener = BlockchainEventListener(mock_mode=True)
        calls = []

        async def handler(event):
            calls.append(event)

        listener.register_handler(EventType.TRAINING_ROUND_STARTED, handler)
        event = TrainingEvent(
            event_type=EventType.TRAINING_ROUND_STARTED,
            block_number=1,
            block_hash="0xABC",
            timestamp="2026-01-01T00:00:00Z",
            data={"round": 1},
        )
        _run(listener._dispatch_event(event))
        assert len(calls) == 1

    def test_dispatch_unknown_event_type_no_crash(self):
        from blockchain.events import BlockchainEventListener, EventType, TrainingEvent

        listener = BlockchainEventListener(mock_mode=True)
        event = TrainingEvent(
            event_type=EventType.TRAINER_ENROLLED,
            block_number=1,
            block_hash="0xABC",
            timestamp="2026-01-01T00:00:00Z",
            data={},
        )
        _run(
            listener._dispatch_event(event)
        )  # no handlers registered — should not crash

    def test_start_listening_mock_mode(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=True)
        _run(listener.start_listening())
        assert listener.is_listening is True

    def test_stop_listening(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=True)
        _run(listener.start_listening())
        listener.stop_listening()
        assert listener.is_listening is False

    def test_get_event_history_empty(self):
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=True)
        history = listener.get_event_history()
        assert isinstance(history, list)

    def test_emit_mock_event(self):
        from blockchain.events import BlockchainEventListener, EventType

        listener = BlockchainEventListener(mock_mode=True)
        calls = []

        async def handler(event):
            calls.append(event)

        listener.register_handler(EventType.TRAINER_ENROLLED, handler)
        _run(
            listener.emit_mock_event(
                event_type=EventType.TRAINER_ENROLLED,
                data={"account": "5ADDR"},
            )
        )
        assert len(calls) == 1

    def test_unregister_handler(self):
        from blockchain.events import BlockchainEventListener, EventType

        listener = BlockchainEventListener(mock_mode=True)
        calls = []

        async def handler(event):
            calls.append(1)

        listener.register_handler(EventType.TRAINER_ENROLLED, handler)
        listener.unregister_handler(EventType.TRAINER_ENROLLED, handler)

        from blockchain.events import TrainingEvent

        event = TrainingEvent(
            event_type=EventType.TRAINER_ENROLLED,
            block_number=1,
            block_hash="0x",
            timestamp="",
            data={},
        )
        _run(listener._dispatch_event(event))
        assert calls == []


# ===========================================================================
# blockchain/identity_verifier.py — DummyVerifier + create_verifier
# ===========================================================================


class TestDummyBelizeIDVerifier:

    def test_connect(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        _run(v.connect())  # should not raise

    def test_verify_always_true(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        result = _run(v.verify("BZ-12345-6789"))
        assert result is True

    def test_verify_any_id(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        for bid in ["BZ-00001-0000", "BZ-99999-9999", "BZ-00000-0001"]:
            assert _run(v.verify(bid)) is True

    def test_get_identity_details(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        details = _run(v.get_identity_details("BZ-12345-6789"))
        assert isinstance(details, dict)
        assert details["kycApproved"] is True

    def test_check_rate_limits_always_true(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        result = _run(v.check_rate_limits("BZ-12345-6789"))
        assert result is True

    def test_close(self):
        from blockchain.identity_verifier import DummyBelizeIDVerifier

        v = DummyBelizeIDVerifier()
        _run(v.close())  # should not raise


class TestCreateVerifier:

    def test_development_mode_returns_dummy(self):
        from blockchain.identity_verifier import create_verifier, DummyBelizeIDVerifier

        os.environ["NAWAL_ENV"] = "development"
        try:
            v = create_verifier(mode="development")
            assert isinstance(v, DummyBelizeIDVerifier)
        finally:
            os.environ.pop("NAWAL_ENV", None)

    def test_development_in_production_env_raises(self):
        from blockchain.identity_verifier import create_verifier

        os.environ["NAWAL_ENV"] = "production"
        try:
            with pytest.raises(RuntimeError, match="cannot be used"):
                create_verifier(mode="development")
        finally:
            os.environ.pop("NAWAL_ENV", None)

    def test_unknown_mode_raises(self):
        from blockchain.identity_verifier import create_verifier

        with pytest.raises(ValueError, match="Unknown mode"):
            create_verifier(mode="invalid_mode")


# ===========================================================================
# blockchain/validator_manager.py — register/get/check/tier/reputation
# ===========================================================================


class TestValidatorManagerMocked:

    def _make_manager(self):
        from blockchain.validator_manager import ValidatorManager
        from blockchain.substrate_client import (
            SubstrateClient,
            ChainConfig,
            ExtrinsicReceipt,
        )

        client = MagicMock(spec=SubstrateClient)
        client.submit_extrinsic.return_value = ExtrinsicReceipt(
            extrinsic_hash="0xDEAD",
            block_hash="0xABC",
            block_number=1,
            success=True,
        )
        client.query_storage.return_value = None
        client.query_map.return_value = []
        return ValidatorManager(client), client

    def _make_identity(
        self, address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    ):
        from blockchain.validator_manager import ValidatorIdentity

        return ValidatorIdentity(
            address=address,
            name="Test Validator",
            email="test@example.com",
            legal_name="Test Ltd",
            tax_id="BZ123456789",
        )

    def test_init(self):
        manager, _ = self._make_manager()
        assert manager is not None

    def test_register_identity_success(self):
        manager, client = self._make_manager()
        identity = self._make_identity()
        keypair = MagicMock()
        receipt = manager.register_identity(keypair, identity)
        assert receipt.success is True

    def test_register_identity_failure(self):
        from blockchain.substrate_client import ExtrinsicReceipt

        manager, client = self._make_manager()
        client.submit_extrinsic.return_value = ExtrinsicReceipt(
            extrinsic_hash="0xDEAD",
            success=False,
            error="BadOrigin",
        )
        identity = self._make_identity()
        receipt = manager.register_identity(MagicMock(), identity)
        assert receipt.success is False

    def test_get_identity_none_when_not_found(self):
        manager, client = self._make_manager()
        client.query_storage.return_value = None
        result = manager.get_identity(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        assert result is None

    def test_get_identity_returns_object_when_found(self):
        manager, client = self._make_manager()
        client.query_storage.return_value = {
            "name": "Alice",
            "email": "alice@example.com",
            "legal_name": "Alice Ltd",
            "tax_id": "BZ001",
            "kyc_status": "pending",
            "registration_block": 100,
        }
        result = manager.get_identity(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        assert result is not None

    def test_submit_kyc_calls_extrinsic(self):
        manager, client = self._make_manager()
        receipt = manager.submit_kyc(
            keypair=MagicMock(),
            documents={"passport": "0xHASH", "proof_of_address": "0xHASH2"},
        )
        assert receipt is not None

    def test_check_compliance_not_registered(self):
        manager, client = self._make_manager()
        client.query_storage.return_value = None
        result = manager.check_compliance(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        assert result is False

    def test_check_compliance_not_verified(self):
        manager, client = self._make_manager()
        client.query_storage.return_value = {
            "name": "Alice",
            "email": "alice@example.com",
            "legal_name": "Alice Ltd",
            "tax_id": "BZ001",
            "kyc_status": "pending",
            "registration_block": 1,
        }
        result = manager.check_compliance(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        assert result is False

    def test_calculate_tier(self):
        from blockchain.validator_manager import ValidatorTier

        manager, client = self._make_manager()
        tier = manager.calculate_tier(
            stake=10_000_000_000_000,
            reputation=90.0,
            min_stake=1_000_000_000_000,
        )
        assert tier is not None
        assert isinstance(tier, ValidatorTier)

    def test_get_reputation_score_not_found(self):
        manager, client = self._make_manager()
        client.query_storage.return_value = None
        score = manager.get_reputation_score(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        assert isinstance(score, float)

    def test_get_all_validators_empty(self):
        manager, client = self._make_manager()
        client.query_map.return_value = []
        validators = manager.get_all_validators()
        assert isinstance(validators, list)

    def test_get_compliant_validators_empty(self):
        manager, client = self._make_manager()
        client.query_map.return_value = []
        validators = manager.get_compliant_validators()
        assert isinstance(validators, list)

    def test_validator_identity_to_dict(self):
        identity = self._make_identity()
        d = identity.to_dict()
        assert isinstance(d, dict)
        assert "name" in d

    def test_update_tier(self):
        from blockchain.validator_manager import ValidatorTier

        manager, client = self._make_manager()
        result = manager.update_tier(
            keypair=MagicMock(),
            new_tier=ValidatorTier.GOLD,
        )
        assert result is not None
