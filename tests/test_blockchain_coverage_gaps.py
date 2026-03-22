"""
Blockchain coverage gaps — targets all uncovered lines across blockchain modules
that were not hit by test_blockchain_substrate_real_mode.py.

Covers:
  - blockchain/substrate_client.py  — auto-connect, exception paths, get_events,
    subscribe_events, get_runtime_constant, get_metadata, get_balance,
    context-manager, create_keypair
  - blockchain/staking_connector.py — community-tracking connect path,
    claim-rewards event parsing, enable_community_tracking flag
  - blockchain/community_connector.py — real-mode disconnect, get_srs_info found,
    record_participation with keypair, example_usage module function
  - blockchain/events.py            — start_listening loop, stop_listening,
    emit_mock_event, create_training_round_handler helpers
  - blockchain/payroll_connector.py — disconnect, not-connected/no-keypair raises,
    real submit success, verify, get_paystub mock+real, stats real mode
  - blockchain/validator_manager.py — get_identity exception, submit_kyc receipt
    log, check_compliance stake exception, update_tier success, reputation
    exception, get_all_validators, get_compliant_validators
  - blockchain/identity_verifier.py — BelizeIDVerifier full mock,
    verify_belizeid cache + blockchain, clear_cache, close, unknown mode
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import timedelta


# ---------------------------------------------------------------------------
# Shared async runner (no asyncio.run to avoid closing the event loop)
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


def _mock_substrate():
    sub = MagicMock()
    sub.chain = "BelizeChain"
    sub.properties = {"tokenSymbol": "DALLA"}
    sub.compose_call.return_value = MagicMock()
    sub.create_signed_extrinsic.return_value = MagicMock(extrinsic_hash="0xDEAD")
    result = MagicMock()
    result.is_success = True
    result.error_message = None
    result.block_hash = "0xABC"
    result.triggered_events = []
    sub.submit_extrinsic.return_value = result
    sub.query.return_value.value = None
    sub.query_map.return_value = []
    return sub


def _make_connected_substrate_client():
    from blockchain.substrate_client import SubstrateClient, ChainConfig
    client = SubstrateClient(ChainConfig.local())
    client.substrate = _mock_substrate()
    client._connected = True
    return client


# ===========================================================================
# blockchain/substrate_client.py — gaps
# ===========================================================================

import blockchain.substrate_client as _sc_mod

_FakeSubstrateExc = type("SubstrateRequestException", (Exception,), {})


class TestSubstrateClientAutoConnect:
    """Line 249 / 288 / 302 / 330 — auto-connect when is_connected is False."""

    def test_query_storage_auto_connects(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        mock_sub = _mock_substrate()
        mock_sub.query.return_value.value = {"balance": 42}

        def _fake_connect(**_kw):
            client._connected = True
            client.substrate = mock_sub

        with patch.object(client, "connect", side_effect=_fake_connect):
            result = client.query_storage("System", "Account", ["5ADDR"])
        assert result == {"balance": 42}

    def test_query_map_auto_connects(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        mock_sub = _mock_substrate()
        mock_sub.query_map.return_value = []

        def _fake_connect(**_kw):
            client._connected = True
            client.substrate = mock_sub

        with patch.object(client, "connect", side_effect=_fake_connect):
            result = client.query_map("Staking", "Trainers")
        assert result == []

    def test_query_storage_no_value_attr_returns_raw(self):
        """Covers the `else result` branch when result has no .value."""
        client = _make_connected_substrate_client()
        raw = "raw_plain_string"  # plain string — no .value attribute
        client.substrate.query.return_value = raw
        result = client.query_storage("System", "Events")
        assert result == "raw_plain_string"

    def test_get_runtime_constant_no_value_attr(self):
        """Covers `else result` in get_runtime_constant."""
        client = _make_connected_substrate_client()
        raw = 9999
        client.substrate.get_constant.return_value = raw
        result = client.get_runtime_constant("System", "MaxWeight")
        assert result == 9999


class TestSubstrateClientExceptionPaths:
    """Covers except SubstrateRequestException lines."""

    def setup_method(self):
        _sc_mod.SubstrateRequestException = _FakeSubstrateExc

    def teardown_method(self):
        try:
            del _sc_mod.SubstrateRequestException
        except AttributeError:
            pass

    def test_query_storage_exception_reraises(self):
        client = _make_connected_substrate_client()
        client.substrate.query.side_effect = _FakeSubstrateExc("storage fail")
        with pytest.raises(_FakeSubstrateExc):
            client.query_storage("X", "Y")

    def test_query_map_exception_reraises(self):
        client = _make_connected_substrate_client()
        client.substrate.query_map.side_effect = _FakeSubstrateExc("map fail")
        with pytest.raises(_FakeSubstrateExc):
            client.query_map("X", "Y")

    def test_submit_extrinsic_exception_reraises(self):
        client = _make_connected_substrate_client()
        client.substrate.submit_extrinsic.side_effect = _FakeSubstrateExc("tx fail")
        keypair = MagicMock()
        with pytest.raises(_FakeSubstrateExc):
            client.submit_extrinsic(
                keypair=keypair,
                call_module="Staking",
                call_function="enroll",
                call_params={},
                wait_for_inclusion=True,
            )


class TestSubstrateClientGetEvents:
    """Covers get_events, subscribe_events lines."""

    def test_get_events_returns_list(self):
        client = _make_connected_substrate_client()
        ev = MagicMock()
        ev.value = {"module_id": "Staking", "event_id": "Enrolled", "attributes": {"who": "5A"}}
        client.substrate.get_events.return_value = [ev]
        events = client.get_events("0xDEAD")
        assert len(events) == 1
        assert events[0]["module"] == "Staking"
        assert events[0]["event"] == "Enrolled"

    def test_get_events_empty(self):
        client = _make_connected_substrate_client()
        client.substrate.get_events.return_value = []
        assert client.get_events() == []

    def test_get_events_auto_connect(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        mock_sub = _mock_substrate()
        mock_sub.get_events.return_value = []

        def _fake_connect(**_kw):
            client._connected = True
            client.substrate = mock_sub

        with patch.object(client, "connect", side_effect=_fake_connect):
            client.get_events()
        assert client.is_connected()

    def test_subscribe_events_calls_subscribe_block_headers(self):
        client = _make_connected_substrate_client()
        received = []
        client.subscribe_events(lambda e: received.append(e))
        client.substrate.subscribe_block_headers.assert_called_once()

    def test_subscribe_events_with_filters(self):
        client = _make_connected_substrate_client()
        client.subscribe_events(
            callback=lambda e: None,
            module_filter="Staking",
            event_filter="Enrolled",
        )
        client.substrate.subscribe_block_headers.assert_called_once()


class TestSubstrateClientGetExtrinsicEvents:
    """Covers _get_extrinsic_events with populated triggered_events."""

    def test_extrinsic_events_populated(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        client.substrate = _mock_substrate()
        client._connected = True

        ev = MagicMock()
        ev.value = {"module_id": "Staking", "event_id": "Enrolled", "attributes": {"who": "5A"}}
        mock_result = MagicMock()
        mock_result.block_hash = "0xABC"
        mock_result.is_success = True
        mock_result.error_message = None
        mock_result.triggered_events = [ev]
        client.substrate.submit_extrinsic.return_value = mock_result
        client.substrate.get_block.return_value = {"header": {"number": 10}}

        keypair = MagicMock()
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="enroll",
            call_params={"amount": 100},
            wait_for_inclusion=True,
        )
        assert receipt.success is True
        assert len(receipt.events) == 1
        assert receipt.events[0]["module"] == "Staking"

    def test_extrinsic_no_triggered_events_attr(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        client.substrate = _mock_substrate()
        client._connected = True

        # result without triggered_events attribute
        mock_result = MagicMock(spec=["block_hash", "is_success", "error_message"])
        mock_result.block_hash = "0xBBB"
        mock_result.is_success = True
        mock_result.error_message = None
        client.substrate.submit_extrinsic.return_value = mock_result
        client.substrate.get_block.return_value = {"header": {"number": 5}}

        keypair = MagicMock()
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="System",
            call_function="remark",
            call_params={},
            wait_for_inclusion=True,
        )
        assert receipt.events == []


class TestSubstrateClientHighLevel:
    """get_runtime_constant, get_metadata, get_account_info, get_balance,
       context manager, create_keypair."""

    def test_get_runtime_constant_with_value(self):
        client = _make_connected_substrate_client()
        client.substrate.get_constant.return_value.value = 4096
        result = client.get_runtime_constant("System", "BlockLength")
        assert result == 4096

    def test_get_metadata(self):
        client = _make_connected_substrate_client()
        client.substrate.get_metadata.return_value = {"version": 14}
        assert client.get_metadata() == {"version": 14}

    def test_get_account_info(self):
        client = _make_connected_substrate_client()
        info = {"nonce": 1, "data": {"free": 1000}}
        client.substrate.query.return_value.value = info
        result = client.get_account_info("5ADDR")
        assert result == info

    def test_get_balance(self):
        client = _make_connected_substrate_client()
        client.substrate.query.return_value.value = {"data": {"free": 5000}}
        balance = client.get_balance("5ADDR")
        assert balance == 5000

    def test_context_manager(self):
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        client = SubstrateClient(ChainConfig.local())
        with patch("blockchain.substrate_client.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            with client:
                assert client.is_connected()
        assert not client.is_connected()

    def test_create_keypair_uri(self):
        orig_kp = getattr(_sc_mod, "Keypair", None)
        mock_kp_cls = MagicMock()
        mock_kp_cls.create_from_uri.return_value = MagicMock(ss58_address="5ALICE")
        _sc_mod.Keypair = mock_kp_cls
        try:
            from blockchain.substrate_client import SubstrateClient
            kp = SubstrateClient.create_keypair(uri="//Alice")
            assert kp is not None
        finally:
            if orig_kp is not None:
                _sc_mod.Keypair = orig_kp
            else:
                try:
                    del _sc_mod.Keypair
                except AttributeError:
                    pass

    def test_create_keypair_mnemonic(self):
        orig_kp = getattr(_sc_mod, "Keypair", None)
        mock_kp_cls = MagicMock()
        mock_kp_cls.create_from_mnemonic.return_value = MagicMock(ss58_address="5BOB")
        _sc_mod.Keypair = mock_kp_cls
        try:
            from blockchain.substrate_client import SubstrateClient
            kp = SubstrateClient.create_keypair(mnemonic="word1 word2 word3")
            assert kp is not None
        finally:
            if orig_kp is not None:
                _sc_mod.Keypair = orig_kp
            else:
                try:
                    del _sc_mod.Keypair
                except AttributeError:
                    pass

    def test_create_keypair_seed(self):
        orig_kp = getattr(_sc_mod, "Keypair", None)
        mock_kp_cls = MagicMock()
        mock_kp_cls.create_from_seed.return_value = MagicMock(ss58_address="5CHARLIE")
        _sc_mod.Keypair = mock_kp_cls
        try:
            from blockchain.substrate_client import SubstrateClient
            kp = SubstrateClient.create_keypair(seed="0xdeadbeef")
            assert kp is not None
        finally:
            if orig_kp is not None:
                _sc_mod.Keypair = orig_kp
            else:
                try:
                    del _sc_mod.Keypair
                except AttributeError:
                    pass

    def test_create_keypair_generates_random(self):
        orig_kp = getattr(_sc_mod, "Keypair", None)
        mock_kp_cls = MagicMock()
        mock_kp_cls.generate_mnemonic.return_value = "abandon word1 word2"
        mock_kp_cls.create_from_mnemonic.return_value = MagicMock(ss58_address="5RAND")
        _sc_mod.Keypair = mock_kp_cls
        try:
            from blockchain.substrate_client import SubstrateClient
            kp = SubstrateClient.create_keypair()  # no args → generates random
            assert kp is not None
        finally:
            if orig_kp is not None:
                _sc_mod.Keypair = orig_kp
            else:
                try:
                    del _sc_mod.Keypair
                except AttributeError:
                    pass


# ===========================================================================
# blockchain/staking_connector.py — community-tracking paths + claim events
# ===========================================================================

class TestStakingConnectorCommunityTracking:
    """Lines 211-214 — community connector initialized during real connect."""

    def test_mock_connect_with_community_tracking(self):
        """Mock-mode connect when community_tracking is enabled."""
        from blockchain.staking_connector import StakingConnector
        conn = StakingConnector(mock_mode=True, enable_community_tracking=True)
        # CommunityConnector created; patch its connect()
        if conn.community_connector:
            conn.community_connector.connect = AsyncMock(return_value=True)
        result = _run(conn.connect())
        assert result is True
        assert conn.is_connected

    def test_real_connect_with_community_tracking(self):
        """Real-mode connect: SubstrateInterface mock + community connector mock."""
        from blockchain.staking_connector import StakingConnector
        conn = StakingConnector(mock_mode=False, enable_community_tracking=True)
        if conn.community_connector:
            conn.community_connector.connect = AsyncMock(return_value=True)

        with patch("blockchain.staking_connector.SubstrateInterface") as mock_cls:
            mock_cls.return_value = _mock_substrate()
            result = _run(conn.connect())
        assert result is True


class TestStakingConnectorClaimRewardsEventParsing:
    """Lines 622-640 — event loop in claim_rewards extracts reward_amount."""

    def test_claim_rewards_extracts_reward_from_event(self):
        """Covers the triggered_events loop in claim_rewards success path."""
        from blockchain.staking_connector import StakingConnector
        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()

        # Build a receipt with a RewardsClaimed event
        reward_event = MagicMock()
        reward_event.value = {
            "module_id": "Staking",
            "event_id": "RewardsClaimed",
            "attributes": {"amount": 5_000_000_000_000},
        }
        receipt = MagicMock()
        receipt.is_success = True
        receipt.error_message = None
        receipt.block_hash = "0xRWD"
        receipt.triggered_events = [reward_event]
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        success, amount = _run(conn.claim_rewards("5ADDR"))
        assert success is True
        assert amount == 5_000_000_000_000

    def test_claim_rewards_event_not_matching_skipped(self):
        """Non-matching event in loop — reward_amount stays 0."""
        from blockchain.staking_connector import StakingConnector
        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()

        other_event = MagicMock()
        other_event.value = {
            "module_id": "System",
            "event_id": "ExtrinsicSuccess",
            "attributes": {},
        }
        receipt = MagicMock()
        receipt.is_success = True
        receipt.error_message = None
        receipt.block_hash = "0xRWD"
        receipt.triggered_events = [other_event]
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        success, amount = _run(conn.claim_rewards("5ADDR"))
        assert success is True
        assert amount == 0  # no RewardsClaimed event found

    def test_claim_rewards_failure_path(self):
        """is_success=False — returns (False, 0)."""
        from blockchain.staking_connector import StakingConnector
        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = False
        receipt.error_message = "Module error"
        receipt.block_hash = "0xFAIL"
        receipt.triggered_events = []
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        success, amount = _run(conn.claim_rewards("5ADDR"))
        assert success is False
        assert amount == 0


class TestStakingConnectorCommunityRecordOnSubmit:
    """Lines 529-544 — community tracking in submit_training_proof."""

    def test_submit_proof_records_community_participation(self):
        """Covers community tracking success path after proof submission."""
        from blockchain.staking_connector import StakingConnector, TrainingSubmission
        from datetime import datetime, timezone

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = True
        receipt.extrinsic_hash = "0xPROOF"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        # Enable community tracking after init so we can control the mock
        from blockchain.community_connector import CommunityConnector
        mock_cc = MagicMock()
        mock_cc.record_federated_learning_contribution = AsyncMock(
            return_value=(True, "0xCC_TX")
        )
        conn.enable_community_tracking = True
        conn.community_connector = mock_cc

        submission = TrainingSubmission(
            participant_id="5ADDR",
            round_number=1,
            genome_id="0xGENOME",
            model_hash="0xMODEL",
            fitness_score=85.0,
            quality_score=90.0,
            timeliness_score=80.0,
            honesty_score=95.0,
            samples_trained=500,
            training_time=3600.0,
        )
        result = _run(conn.submit_training_proof(submission))
        assert result is True
        mock_cc.record_federated_learning_contribution.assert_called_once()

    def test_submit_proof_community_tracking_failure_continues(self):
        """Community tracking failure doesn't break submit."""
        from blockchain.staking_connector import StakingConnector, TrainingSubmission
        from datetime import datetime, timezone

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = True
        receipt.extrinsic_hash = "0xPROOF"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        mock_cc = MagicMock()
        mock_cc.record_federated_learning_contribution = AsyncMock(
            side_effect=Exception("community error")
        )
        conn.enable_community_tracking = True
        conn.community_connector = mock_cc

        submission = TrainingSubmission(
            participant_id="5ADDR",
            round_number=2,
            genome_id="0xGENOME",
            model_hash="0xMODEL",
            fitness_score=75.0,
            quality_score=80.0,
            timeliness_score=70.0,
            honesty_score=90.0,
            samples_trained=300,
            training_time=1800.0,
        )
        result = _run(conn.submit_training_proof(submission))
        assert result is True  # still succeeds despite community error

    def test_submit_proof_community_tracking_returns_false(self):
        """Community tracking returns (False, '') — logs warning but continues."""
        from blockchain.staking_connector import StakingConnector, TrainingSubmission
        from datetime import datetime, timezone

        conn = StakingConnector(mock_mode=False, enable_community_tracking=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = True
        receipt.extrinsic_hash = "0xPROOF"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn.is_connected = True
        conn.keypair = MagicMock(ss58_address="5ADDR")

        mock_cc = MagicMock()
        mock_cc.record_federated_learning_contribution = AsyncMock(
            return_value=(False, "")
        )
        conn.enable_community_tracking = True
        conn.community_connector = mock_cc

        submission = TrainingSubmission(
            participant_id="5ADDR",
            round_number=3,
            genome_id="0xGENOME",
            model_hash="0xMODEL",
            fitness_score=70.0,
            quality_score=75.0,
            timeliness_score=65.0,
            honesty_score=85.0,
            samples_trained=200,
            training_time=900.0,
        )
        result = _run(conn.submit_training_proof(submission))
        assert result is True


# ===========================================================================
# blockchain/community_connector.py — gaps
# ===========================================================================

class TestCommunityConnectorDisconnect:
    """Line 128-130 — disconnect when _connected=True."""

    def test_disconnect_clears_state(self):
        from blockchain.community_connector import CommunityConnector
        conn = CommunityConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        _run(conn.disconnect())
        assert conn._connected is False


class TestCommunityConnectorRealGetSRS:
    """Lines 166-192 — real-mode get_srs_info."""

    def _make_real_conn(self):
        from blockchain.community_connector import CommunityConnector
        conn = CommunityConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        return conn

    def test_real_get_srs_info_returns_none_when_no_data(self):
        conn = self._make_real_conn()
        conn.substrate.query.return_value.value = None
        result = _run(conn.get_srs_info("5ADDR"))
        assert result is None

    def test_real_get_srs_info_returns_srs_object(self):
        from blockchain.community_connector import SRSInfo
        conn = self._make_real_conn()
        conn.substrate.query.return_value.value = {
            "score": 750,
            "tier": 3,
            "participation_count": 20,
            "volunteer_hours": 100,
            "education_modules_completed": 5,
            "green_project_contributions": 2,
            "monthly_fee_exemption": 0,
            "last_updated": 12345,
        }
        result = _run(conn.get_srs_info("5ALICE"))
        assert isinstance(result, SRSInfo)
        assert result.score == 750
        assert result.tier == 3

    def test_real_get_srs_info_exception_returns_none(self):
        conn = self._make_real_conn()
        conn.substrate.query.side_effect = Exception("rpc error")
        result = _run(conn.get_srs_info("5ADDR"))
        assert result is None

    def test_real_get_srs_info_result_is_falsy_returns_none(self):
        conn = self._make_real_conn()
        conn.substrate.query.return_value = None
        result = _run(conn.get_srs_info("5ADDR"))
        assert result is None


class TestCommunityConnectorRealRecordParticipation:
    """Lines 232-271 — real-mode record_participation with keypair."""

    def _make_real_conn(self):
        from blockchain.community_connector import CommunityConnector
        conn = CommunityConnector(mock_mode=False)
        conn.substrate = _mock_substrate()
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5SIGNER")
        return conn

    def test_real_record_participation_success(self):
        conn = self._make_real_conn()
        receipt = MagicMock()
        receipt.is_success = True
        receipt.extrinsic_hash = "0xPARTIC"
        conn.substrate.submit_extrinsic.return_value = receipt
        success, tx_hash = _run(conn.record_participation(
            account_id="5ALICE",
            activity_type="FederatedLearning",
            quality_score=85.0,
        ))
        assert success is True
        assert tx_hash == "0xPARTIC"

    def test_real_record_participation_receipt_failure(self):
        conn = self._make_real_conn()
        receipt = MagicMock()
        receipt.is_success = False
        receipt.error_message = "BadOrigin"
        conn.substrate.submit_extrinsic.return_value = receipt
        success, tx_hash = _run(conn.record_participation(
            account_id="5ALICE",
            activity_type="FederatedLearning",
        ))
        assert success is False
        assert tx_hash == ""

    def test_real_record_participation_exception_returns_false(self):
        conn = self._make_real_conn()
        conn.substrate.compose_call.side_effect = Exception("tx error")
        success, tx_hash = _run(conn.record_participation(
            account_id="5ALICE",
            activity_type="EducationModule",
        ))
        assert success is False

    def test_real_record_participation_not_connected_with_keypair(self):
        from blockchain.community_connector import CommunityConnector
        conn = CommunityConnector(mock_mode=False)
        conn._connected = False
        conn.keypair = MagicMock(ss58_address="5SIGNER")
        success, tx_hash = _run(conn.record_participation(
            account_id="5ALICE",
            activity_type="FederatedLearning",
        ))
        assert success is False


class TestCommunityConnectorExampleUsage:
    """Lines 389-419 — the module-level example_usage() async function."""

    def test_example_usage_runs_in_mock_mode(self):
        """example_usage() creates a mock-mode connector and runs through it."""
        from blockchain.community_connector import example_usage
        # Should complete without errors in mock mode (default in the function)
        _run(example_usage())


# ===========================================================================
# blockchain/events.py — gaps
# ===========================================================================

class TestBlockchainEventsStartListening:
    """Lines 276-340 — start_listening real-mode loop, _subscribe_new_heads."""

    def test_start_listening_mock_mode_sets_flag(self):
        from blockchain.events import BlockchainEventListener
        listener = BlockchainEventListener(mock_mode=True)
        _run(listener.connect())
        _run(listener.start_listening())
        assert listener.is_listening is True
        _run(listener.disconnect())

    def test_start_listening_already_listening_returns_early(self):
        from blockchain.events import BlockchainEventListener
        listener = BlockchainEventListener(mock_mode=True)
        listener.is_listening = True
        _run(listener.start_listening())  # should return immediately
        assert listener.is_listening is True

    def test_start_listening_not_connected_auto_connects(self):
        """Triggers auto-connect path: substrate is None + connect() fails."""
        from blockchain.events import BlockchainEventListener
        listener = BlockchainEventListener(mock_mode=False)
        listener.substrate = None
        listener.is_listening = False
        # connect() is an AsyncMock that returns False → start_listening returns early
        with patch.object(listener, "connect", new=AsyncMock(return_value=False)):
            _run(listener.start_listening())
        assert listener.is_listening is False

    def test_subscribe_new_heads_yields_block(self):
        """Exercise _subscribe_new_heads single iteration."""
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=False)
        mock_sub = _mock_substrate()
        mock_sub.get_chain_head.return_value = "0xHEAD"
        mock_sub.get_block.return_value = {"header": {"hash": "0xHEAD", "number": 1}}
        listener.substrate = mock_sub
        listener.is_listening = True

        async def _consume_one():
            async for block in listener._subscribe_new_heads():
                listener.is_listening = False  # stop after first
                return block

        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            block = _run(_consume_one())
        assert block is not None

    def test_subscribe_new_heads_exception_retries(self):
        """_subscribe_new_heads handles exceptions and keeps going."""
        from blockchain.events import BlockchainEventListener

        listener = BlockchainEventListener(mock_mode=False)
        mock_sub = _mock_substrate()
        call_count = {"n": 0}

        def _raise_then_stop():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("transient error")
            listener.is_listening = False
            return "0xHEAD"

        mock_sub.get_chain_head.side_effect = _raise_then_stop
        mock_sub.get_block.return_value = {"header": {"hash": "0xHEAD", "number": 2}}
        listener.substrate = mock_sub
        listener.is_listening = True

        async def _drain():
            async for _ in listener._subscribe_new_heads():
                pass

        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            _run(_drain())

    def test_stop_listening_sets_flag_false(self):
        from blockchain.events import BlockchainEventListener
        listener = BlockchainEventListener(mock_mode=True)
        listener.is_listening = True
        listener.stop_listening()
        assert listener.is_listening is False


class TestBlockchainEventsEmitMock:
    """Lines 429-431 — emit_mock_event in mock mode."""

    def test_emit_mock_event_dispatches_to_handler(self):
        from blockchain.events import BlockchainEventListener, EventType
        listener = BlockchainEventListener(mock_mode=True)
        received = []

        async def _handler(event):
            received.append(event)

        listener.register_handler(EventType.TRAINING_ROUND_STARTED, _handler)
        _run(listener.emit_mock_event(
            EventType.TRAINING_ROUND_STARTED,
            {"round_number": 1, "genome_id": "0xG"},
        ))
        assert len(received) == 1
        assert received[0].data["round_number"] == 1

    def test_emit_mock_event_outside_mock_mode_logs_warning(self):
        from blockchain.events import BlockchainEventListener, EventType
        listener = BlockchainEventListener(mock_mode=False)
        # Should log warning and return without dispatching
        _run(listener.emit_mock_event(EventType.TRAINING_ROUND_STARTED, {}))


class TestCreateTrainingRoundHandler:
    """Lines 436-437 / 450-470 — create_training_round_handler convenience fn."""

    def test_all_callbacks(self):
        from blockchain.events import create_training_round_handler, EventType, TrainingEvent
        from datetime import datetime, timezone

        started_calls, proof_calls, completed_calls = [], [], []

        async def on_started(round_num, genome_id):
            started_calls.append((round_num, genome_id))

        async def on_proof(participant_id, round_num):
            proof_calls.append((participant_id, round_num))

        async def on_completed(round_num):
            completed_calls.append(round_num)

        handlers = _run(create_training_round_handler(
            on_round_started=on_started,
            on_proof_submitted=on_proof,
            on_round_completed=on_completed,
        ))
        assert EventType.TRAINING_ROUND_STARTED in handlers
        assert EventType.TRAINING_PROOF_SUBMITTED in handlers
        assert EventType.TRAINING_ROUND_COMPLETED in handlers

        now = datetime.now(timezone.utc).isoformat()

        # Drive handler for started
        _run(handlers[EventType.TRAINING_ROUND_STARTED](
            TrainingEvent(EventType.TRAINING_ROUND_STARTED, 1, "0xH", now,
                          {"round_number": 5, "genome_id": "0xGG"})
        ))
        assert started_calls == [(5, "0xGG")]

        # Drive handler for proof
        _run(handlers[EventType.TRAINING_PROOF_SUBMITTED](
            TrainingEvent(EventType.TRAINING_PROOF_SUBMITTED, 1, "0xH", now,
                          {"participant": "5ALICE", "round_number": 5})
        ))
        assert proof_calls == [("5ALICE", 5)]

        # Drive handler for completed
        _run(handlers[EventType.TRAINING_ROUND_COMPLETED](
            TrainingEvent(EventType.TRAINING_ROUND_COMPLETED, 1, "0xH", now,
                          {"round_number": 5})
        ))
        assert completed_calls == [5]

    def test_no_callbacks_empty_handlers(self):
        from blockchain.events import create_training_round_handler
        handlers = _run(create_training_round_handler())
        assert handlers == {}


# ===========================================================================
# blockchain/payroll_connector.py — gaps
# ===========================================================================

class TestPayrollConnectorDisconnect:
    """Lines 286-287 — disconnect body."""

    def test_disconnect_closes_substrate(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        mock_sub = _mock_substrate()
        conn.substrate = mock_sub
        conn._connected = True
        _run(conn.disconnect())
        mock_sub.close.assert_called_once()
        assert conn._connected is False


class TestPayrollConnectorNotConnectedRaises:
    """Lines 311, 314 — RuntimeError when not connected / no keypair."""

    def _make_entry(self):
        from blockchain.payroll_connector import PayrollEntry, EmployeeType
        import hashlib
        return PayrollEntry(
            employee_id="5E001",
            employee_name_hash=hashlib.sha256(b"Alice").hexdigest(),
            gross_salary=1_000_000_000_000,
            tax_withholding=200_000_000_000,
            social_security=50_000_000_000,
            pension_contribution=25_000_000_000,
            net_salary=725_000_000_000,
            payment_period="2026-01",
            employee_type=EmployeeType.PRIVATE,
        )

    def test_submit_payroll_not_connected_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            _run(conn.submit_payroll([self._make_entry()], "2026-01"))

    def test_submit_payroll_no_keypair_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = True
        conn.keypair = None
        with pytest.raises(RuntimeError, match="No keypair"):
            _run(conn.submit_payroll([self._make_entry()], "2026-01"))

    def test_verify_payroll_not_connected_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            _run(conn.verify_payroll("sub-001"))

    def test_verify_payroll_no_keypair_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = True
        conn.keypair = None
        with pytest.raises(RuntimeError, match="No keypair"):
            _run(conn.verify_payroll("sub-001"))

    def test_get_payroll_stats_not_connected_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            _run(conn.get_payroll_stats("2026-01"))


class TestPayrollConnectorRealSubmit:
    """Lines 349, 382-388 — real-mode submit_payroll success."""

    def _make_entry(self, eid="5E001"):
        from blockchain.payroll_connector import PayrollEntry, EmployeeType
        import hashlib
        return PayrollEntry(
            employee_id=eid,
            employee_name_hash=hashlib.sha256(b"Alice").hexdigest(),
            gross_salary=1_000_000_000_000,
            tax_withholding=200_000_000_000,
            social_security=50_000_000_000,
            pension_contribution=25_000_000_000,
            net_salary=725_000_000_000,
            payment_period="2026-01",
            employee_type=EmployeeType.PRIVATE,
        )

    def test_real_submit_payroll_success(self):
        from blockchain.payroll_connector import PayrollConnector, PayrollStatus
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = True
        receipt.extrinsic_hash = "0xPAY"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5EMPLOYER")

        submission = _run(conn.submit_payroll([self._make_entry()], "2026-01"))
        assert submission.status == PayrollStatus.VERIFIED

    def test_real_submit_payroll_blockchain_failure(self):
        from blockchain.payroll_connector import PayrollConnector, PayrollStatus
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        receipt = MagicMock()
        receipt.is_success = False
        receipt.error_message = "BadOrigin"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5EMPLOYER")

        with pytest.raises(RuntimeError, match="Payroll submission failed"):
            _run(conn.submit_payroll([self._make_entry()], "2026-01"))

    def test_real_submit_payroll_substrate_exception(self):
        """Covers SubstrateRequestException handler in submit_payroll."""
        import blockchain.payroll_connector as _pc_mod
        orig_exc = getattr(_pc_mod, "SubstrateRequestException", None)
        FakeExc = type("SubstrateRequestException", (Exception,), {})
        _pc_mod.SubstrateRequestException = FakeExc

        try:
            from blockchain.payroll_connector import PayrollConnector
            conn = PayrollConnector(mock_mode=False)
            sub = _mock_substrate()
            sub.submit_extrinsic.side_effect = FakeExc("chain error")
            conn.substrate = sub
            conn._connected = True
            conn.keypair = MagicMock(ss58_address="5EMPLOYER")

            with pytest.raises(RuntimeError, match="Blockchain submission failed"):
                _run(conn.submit_payroll([self._make_entry()], "2026-01"))
        finally:
            if orig_exc is not None:
                _pc_mod.SubstrateRequestException = orig_exc
            else:
                try:
                    del _pc_mod.SubstrateRequestException
                except AttributeError:
                    pass


class TestPayrollConnectorVerifyReal:
    """Lines 411, 414 — real-mode verify_payroll."""

    def _submission_data(self):
        """Return dict with valid PayrollProof commitment data."""
        import json, hashlib, secrets, hmac as hmac_mod
        merkle_root = "0xdeadbeef1234"
        nonce = secrets.token_hex(32)
        entry_commit = hmac_mod.new(
            nonce.encode(), b"EMP|1000|800|2025-01", hashlib.sha256
        ).hexdigest()
        aggregate = hashlib.sha256(entry_commit.encode()).hexdigest()
        merkle_binding = hashlib.sha256(
            (aggregate + merkle_root).encode()
        ).hexdigest()
        proof_json = json.dumps({
            "version": 1, "scheme": "hmac-sha256-commitment",
            "nonce": nonce, "entry_commitments": [entry_commit],
            "aggregate_commitment": aggregate, "entry_count": 1,
            "merkle_binding": merkle_binding, "timestamp": 0.0,
        })
        return {
            "zk_proof": proof_json,
            "total_gross": "1000000000000",
            "total_tax": "200000000000",
            "merkle_root": merkle_root,
        }

    def test_real_verify_payroll_success(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.return_value = self._submission_data()
        receipt = MagicMock()
        receipt.is_success = True
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5VALIDATOR")

        result = _run(conn.verify_payroll("sub-0001"))
        assert result is True

    def test_real_verify_payroll_failure(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.return_value = self._submission_data()
        receipt = MagicMock()
        receipt.is_success = False
        receipt.error_message = "NotAuthorized"
        sub.submit_extrinsic.return_value = receipt
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5VALIDATOR")

        result = _run(conn.verify_payroll("sub-0001"))
        assert result is False

    def test_real_verify_payroll_not_found(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.return_value = None  # submission not found
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5VALIDATOR")

        result = _run(conn.verify_payroll("sub-missing"))
        assert result is False

    def test_real_verify_payroll_exception(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.side_effect = Exception("network error")
        conn.substrate = sub
        conn._connected = True
        conn.keypair = MagicMock(ss58_address="5VALIDATOR")

        result = _run(conn.verify_payroll("sub-0001"))
        assert result is False


class TestPayrollConnectorGetPaystub:
    """Lines 420-474 — get_employee_paystub mock and real paths."""

    def test_mock_paystub_not_connected_raises(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        conn._connected = False
        with pytest.raises(RuntimeError, match="Not connected"):
            _run(conn.get_employee_paystub("E001", "2026-01"))

    def test_mock_paystub_returns_object(self):
        from blockchain.payroll_connector import PayrollConnector, EmployeePaystub
        conn = PayrollConnector(mock_mode=True)
        conn._connected = True
        paystub = _run(conn.get_employee_paystub("E001", "2026-01"))
        assert isinstance(paystub, EmployeePaystub)
        assert paystub.employee_id == "E001"

    def test_real_paystub_returns_none_when_no_data(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.return_value = None  # no paystub data
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_employee_paystub("E001", "2026-01"))
        assert result is None

    def test_real_paystub_returns_object_when_data_found(self):
        from blockchain.payroll_connector import PayrollConnector, EmployeePaystub
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        paystub_data = {
            "gross_salary": 1_000_000_000_000,
            "tax_withholding": 200_000_000_000,
            "social_security": 50_000_000_000,
            "pension_contribution": 25_000_000_000,
            "net_salary": 725_000_000_000,
            "payment_date": "2026-01-28",
            "employer_name": "Test Corp",
            "status": "paid",
        }
        sub.query.return_value = paystub_data
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_employee_paystub("E001", "2026-01"))
        assert isinstance(result, EmployeePaystub)
        assert result.gross_salary == 1_000_000_000_000

    def test_real_paystub_exception_returns_none(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.side_effect = Exception("rpc timeout")
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_employee_paystub("E001", "2026-01"))
        assert result is None


class TestPayrollConnectorGetStatsReal:
    """Lines 509-537 — real-mode get_payroll_stats paths."""

    def test_real_stats_returns_dict(self):
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        # Return a real dict — `return stats or {}` → stats dict (truthy)
        sub.query.return_value = {
            "total_submissions": 10,
            "total_gross": 5_000_000_000_000,
            "total_tax": 1_000_000_000_000,
        }
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_payroll_stats("2026-01"))
        assert isinstance(result, dict)
        assert result["total_submissions"] == 10

    def test_real_stats_returns_empty_when_no_data(self):
        """None from substrate → None or {} → returns {} not None."""
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.return_value = None
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_payroll_stats("2026-01"))
        assert result == {}

    def test_real_stats_exception_returns_empty_dict(self):
        """Exception in stats → returns {} (not None)."""
        from blockchain.payroll_connector import PayrollConnector
        conn = PayrollConnector(mock_mode=False)
        sub = _mock_substrate()
        sub.query.side_effect = Exception("query failed")
        conn.substrate = sub
        conn._connected = True
        result = _run(conn.get_payroll_stats("2026-01"))
        assert result == {}


# ===========================================================================
# blockchain/validator_manager.py — gaps
# ===========================================================================

def _make_validator_manager():
    from blockchain.substrate_client import SubstrateClient, ChainConfig
    from blockchain.validator_manager import ValidatorManager
    client = SubstrateClient(ChainConfig.local())
    client.substrate = _mock_substrate()
    client._connected = True
    return ValidatorManager(client)


class TestValidatorManagerGaps:
    """Lines 212-214, 249, 286-290, 318-321, 356, 383-385, 403-418, 423-425."""

    def test_get_identity_exception_returns_none(self):
        """Line 212-214 — exception path in get_identity."""
        manager = _make_validator_manager()
        manager.client.substrate.query.side_effect = Exception("rpc error")
        result = manager.get_identity("5ADDR")
        assert result is None

    def test_submit_kyc_receipt_success_logs(self):
        """Line 249 — submit_kyc receipt success logging."""
        from blockchain.substrate_client import ExtrinsicReceipt
        manager = _make_validator_manager()
        receipt = ExtrinsicReceipt("0xKYC")
        receipt.success = True
        receipt.error = None
        with patch.object(manager.client, "submit_extrinsic", return_value=receipt):
            result = manager.submit_kyc(
                keypair=MagicMock(ss58_address="5KYC"),
                documents={"id_front": "hash1", "id_back": "hash2"},
            )
        assert result.success is True

    def test_submit_kyc_receipt_failure_logs(self):
        """submit_kyc receipt failure logging."""
        from blockchain.substrate_client import ExtrinsicReceipt
        manager = _make_validator_manager()
        receipt = ExtrinsicReceipt("0xKYC")
        receipt.success = False
        receipt.error = "BadFormat"
        with patch.object(manager.client, "submit_extrinsic", return_value=receipt):
            result = manager.submit_kyc(
                keypair=MagicMock(ss58_address="5KYC"),
                documents={"passport": "hash_xyz"},
            )
        assert result.success is False

    def test_check_compliance_stake_exception_returns_false(self):
        """Lines 286-290 — check_compliance when staking raises."""
        manager = _make_validator_manager()
        with patch("blockchain.validator_manager.StakingInterface") as mock_si:
            mock_si.return_value.get_stake_info.side_effect = Exception("stake error")
            result = manager.check_compliance("5ADDR")
        assert result is False

    def test_update_tier_success_path(self):
        """Line 356 — update_tier receipt success logging."""
        from blockchain.substrate_client import ExtrinsicReceipt
        from blockchain.validator_manager import ValidatorTier
        manager = _make_validator_manager()
        receipt = ExtrinsicReceipt("0xTIER")
        receipt.success = True
        receipt.error = None
        with patch.object(manager.client, "submit_extrinsic", return_value=receipt):
            result = manager.update_tier(
                keypair=MagicMock(ss58_address="5VAL"),
                new_tier=ValidatorTier.GOLD,
            )
        assert result.success is True

    def test_update_tier_failure_path(self):
        """update_tier receipt failure logging."""
        from blockchain.substrate_client import ExtrinsicReceipt
        from blockchain.validator_manager import ValidatorTier
        manager = _make_validator_manager()
        receipt = ExtrinsicReceipt("0xTIER")
        receipt.success = False
        receipt.error = "Unauthorized"
        with patch.object(manager.client, "submit_extrinsic", return_value=receipt):
            result = manager.update_tier(
                keypair=MagicMock(ss58_address="5VAL"),
                new_tier=ValidatorTier.SILVER,
            )
        assert result.success is False

    def test_get_reputation_score_exception_returns_zero(self):
        """Lines 383-385 — reputation query exception → 0.0."""
        manager = _make_validator_manager()
        manager.client.substrate.query.side_effect = Exception("timeout")
        score = manager.get_reputation_score("5ADDR")
        assert score == 0.0

    def test_get_all_validators_returns_list(self):
        """Lines 403-418 — get_all_validators with data."""
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        from blockchain.validator_manager import ValidatorManager
        client = SubstrateClient(ChainConfig.local())
        sub = _mock_substrate()
        # query_map in SubstrateClient returns [(k.value, v.value) for k, v in result]
        # So we give it objects with .value attributes
        k1 = MagicMock()
        k1.value = "5ADDR1"
        v1 = MagicMock()
        v1.value = {
            "name": "Alice",
            "email": "alice@bz.com",
            "website": "https://alice.bz",
            "legal_name": "Alice Smith",
            "jurisdiction": "BZ",
            "tax_id": "12345",
            "kyc_status": "verified",
            "tier": "gold",
        }
        sub.query_map.return_value = [(k1, v1)]
        client.substrate = sub
        client._connected = True
        manager = ValidatorManager(client)
        validators = manager.get_all_validators()
        assert len(validators) == 1
        assert validators[0].address == "5ADDR1"
        assert validators[0].name == "Alice"

    def test_get_all_validators_empty(self):
        manager = _make_validator_manager()
        manager.client.substrate.query_map.return_value = []
        validators = manager.get_all_validators()
        assert validators == []

    def test_get_all_validators_exception_returns_empty(self):
        """get_all_validators exception → returns []."""
        manager = _make_validator_manager()
        manager.client.substrate.query_map.side_effect = Exception("rpc fail")
        validators = manager.get_all_validators()
        assert validators == []

    def test_get_all_validators_partial_parse_error(self):
        """One entry fails to parse — warning logged, others continue."""
        from blockchain.substrate_client import SubstrateClient, ChainConfig
        from blockchain.validator_manager import ValidatorManager
        client = SubstrateClient(ChainConfig.local())
        sub = _mock_substrate()
        # Bad entry: invalid kyc_status enum value
        k_bad = MagicMock()
        k_bad.value = "5BAD"
        v_bad = MagicMock()
        v_bad.value = {
            "name": "Bad",
            "email": "b@bz.com",
            "kyc_status": "INVALID_STATUS_XYZ",
            "tier": "bronze",
        }
        k_good = MagicMock()
        k_good.value = "5GOOD"
        v_good = MagicMock()
        v_good.value = {
            "name": "Good",
            "email": "g@bz.com",
            "kyc_status": "verified",
            "tier": "bronze",
            "jurisdiction": "BZ",
        }
        sub.query_map.return_value = [(k_bad, v_bad), (k_good, v_good)]
        client.substrate = sub
        client._connected = True
        manager = ValidatorManager(client)
        validators = manager.get_all_validators()
        # Bad entry skipped, good one included
        assert len(validators) == 1
        assert validators[0].name == "Good"

    def test_get_compliant_validators(self):
        """Lines 423-425 — get_compliant_validators returns subset."""
        manager = _make_validator_manager()
        manager.client.substrate.query_map.return_value = []
        compliant = manager.get_compliant_validators()
        assert isinstance(compliant, list)


# ===========================================================================
# blockchain/identity_verifier.py — gaps
# ===========================================================================

import blockchain.identity_verifier as _iv_mod


class TestBelizeIDVerifierMocked:
    """Tests for BelizeIDVerifier with SUBSTRATE_AVAILABLE mocked to True."""

    def _make_verifier(self):
        """Bypass the ImportError guard using object.__new__."""
        from blockchain.identity_verifier import BelizeIDVerifier
        verifier = BelizeIDVerifier.__new__(BelizeIDVerifier)
        verifier.rpc_url = "ws://127.0.0.1:9944"
        verifier.cache = {}
        verifier.cache_ttl = timedelta(seconds=3600)
        verifier.substrate = None
        return verifier

    def test_belizeid_to_identity_id_is_int(self):
        verifier = self._make_verifier()
        result = verifier._belizeid_to_identity_id("BZ123456789")
        assert isinstance(result, int)
        assert result > 0

    def test_belizeid_to_identity_id_consistent(self):
        verifier = self._make_verifier()
        id1 = verifier._belizeid_to_identity_id("BZ-SAME")
        id2 = verifier._belizeid_to_identity_id("BZ-SAME")
        assert id1 == id2

    def test_verify_belizeid_uses_cache(self):
        from datetime import datetime, timezone
        verifier = self._make_verifier()
        verifier.cache["BZ-CACHED"] = (True, datetime.now(timezone.utc))
        result = _run(verifier.verify("BZ-CACHED"))
        assert result is True

    def test_verify_belizeid_cache_expired_re_queries(self):
        from datetime import datetime, timezone
        verifier = self._make_verifier()
        # Put stale entry in cache
        verifier.cache["BZ-OLD"] = (
            True, datetime(2000, 1, 1, tzinfo=timezone.utc)
        )
        with patch.object(verifier, "_query_blockchain", new=AsyncMock(return_value=False)):
            result = _run(verifier.verify("BZ-OLD"))
        assert result is False

    def test_verify_belizeid_no_cache_queries_blockchain(self):
        verifier = self._make_verifier()
        with patch.object(verifier, "_query_blockchain", new=AsyncMock(return_value=True)):
            result = _run(verifier.verify("BZ-NEW"))
        assert result is True
        assert "BZ-NEW" in verifier.cache

    def test_verify_belizeid_exception_returns_false(self):
        verifier = self._make_verifier()
        with patch.object(verifier, "_query_blockchain", side_effect=Exception("error")):
            result = _run(verifier.verify("BZ-ERR"))
        assert result is False

    def test_query_blockchain_returns_false_when_none(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        sub.query.return_value = None
        verifier.substrate = sub

        async def _run_in_executor_side_effect(executor, func):
            return func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_run_in_executor_side_effect)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = _run(verifier._query_blockchain("BZ-NORESULT"))
        assert result is False

    def test_query_blockchain_kyc_not_approved(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        result_mock = MagicMock()
        result_mock.value = {"kycApproved": False}
        sub.query.return_value = result_mock
        verifier.substrate = sub

        async def _run_in_executor_side_effect(executor, func):
            return func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_run_in_executor_side_effect)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = _run(verifier._query_blockchain("BZ-KYC"))
        assert result is False

    def test_query_blockchain_kyc_approved(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        result_mock = MagicMock()
        result_mock.value = {"kycApproved": True}
        sub.query.return_value = result_mock
        verifier.substrate = sub

        async def _run_in_executor_side_effect(executor, func):
            return func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_run_in_executor_side_effect)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = _run(verifier._query_blockchain("BZ-APPR"))
        assert result is True

    def test_get_identity_details_none_result(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        sub.query.return_value = None
        verifier.substrate = sub

        async def _rexec(executor, func):
            return func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_rexec)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = _run(verifier.get_identity_details("BZ-NONE"))
        assert result is None

    def test_get_identity_details_with_data(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        data_mock = MagicMock()
        data_mock.value = {"kycApproved": True, "name": "Alice"}
        sub.query.return_value = data_mock
        verifier.substrate = sub

        async def _rexec(executor, func):
            return func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_rexec)

        with patch("asyncio.get_event_loop", return_value=loop):
            result = _run(verifier.get_identity_details("BZ-DATA"))
        assert result is not None

    def test_check_rate_limits_returns_true(self):
        verifier = self._make_verifier()
        result = _run(verifier.check_rate_limits("BZ-123"))
        assert result is True

    def test_clear_cache(self):
        from datetime import datetime, timezone
        verifier = self._make_verifier()
        verifier.cache["BZ-X"] = (True, datetime.now(timezone.utc))
        verifier.clear_cache()
        assert verifier.cache == {}

    def test_close_with_substrate(self):
        verifier = self._make_verifier()
        sub = MagicMock()
        sub.close = MagicMock()
        verifier.substrate = sub

        async def _rexec(executor, func):
            func()

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=_rexec)

        with patch("asyncio.get_event_loop", return_value=loop):
            _run(verifier.close())
        sub.close.assert_called_once()

    def test_close_without_substrate(self):
        verifier = self._make_verifier()
        verifier.substrate = None
        _run(verifier.close())  # should not raise


class TestCreateVerifierUnknownMode:
    """Line 234 — create_verifier with unknown mode raises ValueError."""

    def test_unknown_mode_raises(self):
        from blockchain.identity_verifier import create_verifier
        with pytest.raises(ValueError, match="Unknown mode"):
            create_verifier(mode="staging")
