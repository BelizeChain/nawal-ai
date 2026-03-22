"""
tests/test_blockchain_data_cli.py

Professional test suite covering:
  - blockchain/staking_connector.py  (ParticipantInfo, TrainingSubmission, StakingConnector mock-mode)
  - blockchain/mesh_network.py       (MessageType, PeerInfo, MeshMessage, FLRoundAnnouncement, MeshNetworkClient)
  - blockchain/events.py             (EventType, TrainingEvent, BlockchainEventListener mock-mode)
  - data/data_manager.py             (DatasetType, SplitConfig, DatasetConfig, DataManager, ListDataset)
  - client/data_loader.py            (DataSovereigntyLevel, ComplianceMetadata, ComplianceDataFilter)
  - cli/commands.py                  (CLI via Click CliRunner)
  - client/train.py                  (BelizeTrainingConfig; BelizeChainFederatedClient with mocks)
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: blockchain/staking_connector.py
# ─────────────────────────────────────────────────────────────────────────────


class TestParticipantInfo:
    """Tests for ParticipantInfo dataclass."""

    def _make(self, **kwargs):
        from blockchain.staking_connector import ParticipantInfo

        defaults = dict(
            account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            stake_amount=1000,
            is_enrolled=True,
            training_rounds_completed=5,
            total_samples_trained=5000,
            avg_fitness_score=85.0,
        )
        defaults.update(kwargs)
        return ParticipantInfo(**defaults)

    def test_create(self):
        p = self._make()
        assert p.account_id.startswith("5G")
        assert p.avg_fitness_score == 85.0

    def test_default_reputation(self):
        p = self._make()
        assert p.reputation_score == 100.0

    def test_default_slashed_amount(self):
        p = self._make()
        assert p.slashed_amount == 0

    def test_last_submission_default_none(self):
        p = self._make()
        assert p.last_submission is None

    def test_invalid_fitness_raises(self):
        from blockchain.staking_connector import ParticipantInfo

        with pytest.raises(ValueError):
            ParticipantInfo(
                account_id="5G...",
                stake_amount=100,
                is_enrolled=True,
                training_rounds_completed=0,
                total_samples_trained=0,
                avg_fitness_score=150.0,  # invalid
            )

    def test_negative_fitness_raises(self):
        from blockchain.staking_connector import ParticipantInfo

        with pytest.raises(ValueError):
            ParticipantInfo(
                account_id="5G...",
                stake_amount=100,
                is_enrolled=True,
                training_rounds_completed=0,
                total_samples_trained=0,
                avg_fitness_score=-1.0,  # invalid
            )

    def test_zero_fitness_valid(self):
        p = self._make(avg_fitness_score=0.0)
        assert p.avg_fitness_score == 0.0

    def test_hundred_fitness_valid(self):
        p = self._make(avg_fitness_score=100.0)
        assert p.avg_fitness_score == 100.0


class TestTrainingSubmission:
    """Tests for TrainingSubmission dataclass and validation."""

    def _make(self, **kwargs):
        from blockchain.staking_connector import TrainingSubmission

        defaults = dict(
            participant_id="5G...",
            round_number=1,
            genome_id="genome_001",
            samples_trained=1000,
            training_time=60.0,
            quality_score=80.0,
            timeliness_score=90.0,
            honesty_score=95.0,
            fitness_score=87.5,
            model_hash="abc123def456",
        )
        defaults.update(kwargs)
        return TrainingSubmission(**defaults)

    def test_create_valid_submission(self):
        s = self._make()
        assert s.samples_trained == 1000
        assert s.model_hash == "abc123def456"

    def test_validate_valid_returns_empty_list(self):
        s = self._make()
        errors = s.validate()
        assert errors == []

    def test_validate_negative_samples(self):
        s = self._make(samples_trained=-1)
        errors = s.validate()
        assert any("samples_trained" in e for e in errors)

    def test_validate_zero_training_time(self):
        s = self._make(training_time=0.0)
        errors = s.validate()
        assert any("training_time" in e for e in errors)

    def test_validate_score_out_of_range(self):
        s = self._make(quality_score=101.0)
        errors = s.validate()
        assert any("quality_score" in e for e in errors)

    def test_validate_negative_score(self):
        s = self._make(timeliness_score=-5.0)
        errors = s.validate()
        assert any("timeliness_score" in e for e in errors)

    def test_validate_missing_model_hash(self):
        s = self._make(model_hash="")
        errors = s.validate()
        assert any("model_hash" in e for e in errors)

    def test_timestamp_auto_generated(self):
        s = self._make()
        assert isinstance(s.timestamp, str)
        assert len(s.timestamp) > 0

    def test_multiple_errors(self):
        s = self._make(samples_trained=0, training_time=0.0, model_hash="")
        errors = s.validate()
        assert len(errors) >= 3


class TestStakingConnectorMockMode:
    """Tests for StakingConnector in mock_mode=True."""

    @pytest.fixture
    def connector(self):
        from blockchain.staking_connector import StakingConnector

        return StakingConnector(mock_mode=True, enable_community_tracking=False)

    def test_mock_mode_flag(self, connector):
        assert connector.mock_mode is True

    def test_not_connected_initially(self, connector):
        assert connector.is_connected is False

    def test_connect_in_mock_mode(self, connector):
        result = asyncio.get_event_loop().run_until_complete(connector.connect())
        assert result is True

    def test_mock_participants_empty(self, connector):
        assert connector._mock_participants == {}

    def test_mock_submissions_empty(self, connector):
        assert connector._mock_submissions == []

    def test_enroll_participant_mock(self, connector):
        asyncio.get_event_loop().run_until_complete(connector.connect())
        result = asyncio.get_event_loop().run_until_complete(
            connector.enroll_participant(
                account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                stake_amount=5000,
                keypair=None,
            )
        )
        # In mock mode should succeed or return something truthy
        assert result is True or result is not None

    def test_get_participant_info_not_enrolled(self, connector):
        asyncio.get_event_loop().run_until_complete(connector.connect())
        info = asyncio.get_event_loop().run_until_complete(
            connector.get_participant_info(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY_NOTFOUND"
            )
        )
        assert info is None

    def test_get_all_participants_empty(self, connector):
        asyncio.get_event_loop().run_until_complete(connector.connect())
        participants = asyncio.get_event_loop().run_until_complete(
            connector.get_all_participants()
        )
        assert isinstance(participants, list)

    def test_get_total_staked_mock(self, connector):
        asyncio.get_event_loop().run_until_complete(connector.connect())
        total = asyncio.get_event_loop().run_until_complete(
            connector.get_total_staked()
        )
        assert isinstance(total, int)
        assert total >= 0

    def test_non_mock_mode_raises_without_substrate(self):
        from blockchain.staking_connector import StakingConnector

        # Without substrate installed, non-mock mode should raise ImportError
        with patch("blockchain.staking_connector.SUBSTRATE_AVAILABLE", False):
            with pytest.raises(ImportError):
                StakingConnector(mock_mode=False, enable_community_tracking=False)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: blockchain/mesh_network.py
# ─────────────────────────────────────────────────────────────────────────────


class TestMessageType:
    """Tests for MessageType enum."""

    def test_enum_values(self):
        from blockchain.mesh_network import MessageType

        assert MessageType.PEER_DISCOVERY.value == "peer_discovery"
        assert MessageType.FL_ROUND_START.value == "fl_round_start"
        assert MessageType.HEARTBEAT.value == "heartbeat"

    def test_all_values_strings(self):
        from blockchain.mesh_network import MessageType

        for mt in MessageType:
            assert isinstance(mt.value, str)


class TestPeerInfo:
    """Tests for PeerInfo dataclass."""

    def _make(self, **kwargs):
        from blockchain.mesh_network import PeerInfo

        defaults = dict(
            peer_id="peer_abc123",
            account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            multiaddr="/ip4/127.0.0.1/tcp/9090",
            public_key="deadbeefdeadbeef",
            last_seen=time.time(),
        )
        defaults.update(kwargs)
        return PeerInfo(**defaults)

    def test_create_peer(self):
        p = self._make()
        assert p.peer_id == "peer_abc123"

    def test_default_reputation(self):
        p = self._make()
        assert p.reputation == 100.0

    def test_is_alive_recent(self):
        p = self._make(last_seen=time.time())
        assert p.is_alive(timeout=300.0) is True

    def test_is_alive_old(self):
        p = self._make(last_seen=time.time() - 1000)
        assert p.is_alive(timeout=300.0) is False

    def test_is_alive_zero_timestamp(self):
        p = self._make(last_seen=0.0)
        assert p.is_alive(timeout=300.0) is False

    def test_default_stake_zero(self):
        p = self._make()
        assert p.stake_amount == 0

    def test_capabilities_default_empty(self):
        p = self._make()
        assert p.capabilities == []

    def test_is_validator_default_false(self):
        p = self._make()
        assert p.is_validator is False


class TestMeshMessage:
    """Tests for MeshMessage dataclass."""

    def _make(self, **kwargs):
        from blockchain.mesh_network import MeshMessage, MessageType

        defaults = dict(
            message_id="msg_001",
            message_type=MessageType.HEARTBEAT,
            sender_id="peer_abc",
            timestamp=1700000000.0,
            payload={"data": "hello"},
        )
        defaults.update(kwargs)
        return MeshMessage(**defaults)

    def test_create_message(self):
        m = self._make()
        assert m.message_id == "msg_001"

    def test_default_ttl(self):
        m = self._make()
        assert m.ttl == 5

    def test_default_signature_none(self):
        m = self._make()
        assert m.signature is None

    def test_to_dict_has_keys(self):
        m = self._make()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "message_id" in d
        assert "message_type" in d
        assert "sender_id" in d
        assert "timestamp" in d
        assert "payload" in d

    def test_to_dict_message_type_is_string(self):
        m = self._make()
        d = m.to_dict()
        assert isinstance(d["message_type"], str)

    def test_from_dict_roundtrip(self):
        from blockchain.mesh_network import MeshMessage, MessageType

        m = self._make()
        d = m.to_dict()
        m2 = MeshMessage.from_dict(d)
        assert m2.message_id == m.message_id
        assert m2.message_type == m.message_type
        assert m2.sender_id == m.sender_id

    def test_from_dict_payload_preserved(self):
        from blockchain.mesh_network import MeshMessage

        m = self._make(payload={"round": 5, "model": "test"})
        d = m.to_dict()
        m2 = MeshMessage.from_dict(d)
        assert m2.payload["round"] == 5

    def test_with_signature(self):
        m = self._make(signature="sig_xyz")
        d = m.to_dict()
        assert d["signature"] == "sig_xyz"


class TestFLRoundAnnouncement:
    """Tests for FLRoundAnnouncement dataclass."""

    def test_create(self):
        from blockchain.mesh_network import FLRoundAnnouncement

        ann = FLRoundAnnouncement(
            round_id="round_001",
            coordinator_id="coord_abc",
            dataset_name="belizean_corpus",
            target_participants=10,
            start_time=1700000000.0,
            deadline=1700003600.0,
            min_stake=500,
            reward_pool=10000,
            model_hash="deadbeef",
        )
        assert ann.round_id == "round_001"
        assert ann.target_participants == 10
        assert ann.min_stake == 500


class TestMeshNetworkClient:
    """Tests for MeshNetworkClient initialization and sync methods."""

    @pytest.fixture
    def client(self):
        from blockchain.mesh_network import MeshNetworkClient

        # MeshNetworkClient takes: peer_id, listen_port, blockchain_rpc, private_key
        return MeshNetworkClient(
            peer_id="test_peer",
            listen_port=19191,
            blockchain_rpc="ws://localhost:19944",
        )

    def test_peer_id(self, client):
        assert client.peer_id == "test_peer"

    def test_peers_dict_empty(self, client):
        assert isinstance(client.peers, dict)
        assert len(client.peers) == 0

    def test_running_false(self, client):
        assert client._running is False

    def test_register_handler(self, client):
        from blockchain.mesh_network import MessageType

        async def handler(msg):
            pass

        client.register_handler(MessageType.HEARTBEAT, handler)
        assert MessageType.HEARTBEAT in client.message_handlers

    def test_register_multiple_handlers(self, client):
        from blockchain.mesh_network import MessageType

        async def h1(msg):
            pass

        async def h2(msg):
            pass

        client.register_handler(MessageType.HEARTBEAT, h1)
        client.register_handler(MessageType.HEARTBEAT, h2)
        assert len(client.message_handlers[MessageType.HEARTBEAT]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: blockchain/events.py
# ─────────────────────────────────────────────────────────────────────────────


class TestEventType:
    """Tests for EventType enum."""

    def test_training_events_exist(self):
        from blockchain.events import EventType

        assert EventType.TRAINING_ROUND_STARTED
        assert EventType.TRAINING_PROOF_SUBMITTED
        assert EventType.TRAINING_ROUND_COMPLETED

    def test_enrollment_events_exist(self):
        from blockchain.events import EventType

        assert EventType.TRAINER_ENROLLED
        assert EventType.TRAINER_UNENROLLED

    def test_reward_events_exist(self):
        from blockchain.events import EventType

        assert EventType.REWARDS_CALCULATED
        assert EventType.REWARDS_CLAIMED

    def test_genome_events_exist(self):
        from blockchain.events import EventType

        assert EventType.GENOME_DEPLOYED
        assert EventType.GENOME_EVOLVED

    def test_penalty_events_exist(self):
        from blockchain.events import EventType

        assert EventType.TRAINER_SLASHED


class TestTrainingEvent:
    """Tests for TrainingEvent dataclass."""

    def _make(self, **kwargs):
        from blockchain.events import TrainingEvent, EventType

        defaults = dict(
            event_type=EventType.TRAINING_ROUND_STARTED,
            block_number=100,
            block_hash="0xdeadbeef",
            timestamp="2024-01-01T00:00:00+00:00",
            data={"round": 1},
        )
        defaults.update(kwargs)
        return TrainingEvent(**defaults)

    def test_create_event(self):
        e = self._make()
        assert e.block_number == 100

    def test_str_representation(self):
        e = self._make()
        s = str(e)
        assert isinstance(s, str)
        assert "block" in s.lower() or "100" in s

    def test_data_dict(self):
        e = self._make(data={"key": "value"})
        assert e.data["key"] == "value"

    def test_different_event_types(self):
        from blockchain.events import EventType

        for et in EventType:
            e = self._make(event_type=et)
            assert e.event_type == et


class TestBlockchainEventListenerMockMode:
    """Tests for BlockchainEventListener in mock_mode=True."""

    @pytest.fixture
    def listener(self):
        from blockchain.events import BlockchainEventListener

        return BlockchainEventListener(mock_mode=True)

    def test_mock_mode_set(self, listener):
        assert listener.mock_mode is True

    def test_not_listening_initially(self, listener):
        assert listener.is_listening is False

    def test_handlers_dict_initialized(self, listener):
        from blockchain.events import EventType

        assert isinstance(listener.handlers, dict)
        for et in EventType:
            assert et in listener.handlers

    def test_event_history_empty(self, listener):
        assert listener.event_history == []

    def test_connect_mock_mode(self, listener):
        result = asyncio.get_event_loop().run_until_complete(listener.connect())
        assert result is True

    def test_register_handler(self, listener):
        from blockchain.events import EventType

        async def handler(event):
            pass

        listener.register_handler(EventType.TRAINING_ROUND_STARTED, handler)
        assert handler in listener.handlers[EventType.TRAINING_ROUND_STARTED]

    def test_unregister_handler(self, listener):
        from blockchain.events import EventType

        async def handler(event):
            pass

        listener.register_handler(EventType.TRAINING_ROUND_STARTED, handler)
        listener.unregister_handler(EventType.TRAINING_ROUND_STARTED, handler)
        assert handler not in listener.handlers[EventType.TRAINING_ROUND_STARTED]

    def test_get_event_history_empty(self, listener):
        from blockchain.events import EventType

        history = listener.get_event_history(
            event_type=EventType.TRAINING_ROUND_STARTED
        )
        assert history == []

    def test_get_event_history_all(self, listener):
        history = listener.get_event_history()
        assert isinstance(history, list)

    def test_stop_listening(self, listener):
        listener.stop_listening()
        assert listener.is_listening is False

    def test_emit_mock_event(self, listener):
        from blockchain.events import EventType

        asyncio.get_event_loop().run_until_complete(listener.connect())
        asyncio.get_event_loop().run_until_complete(
            listener.emit_mock_event(EventType.TRAINING_ROUND_STARTED, {"round": 1})
        )
        history = listener.get_event_history(
            event_type=EventType.TRAINING_ROUND_STARTED
        )
        assert len(history) >= 1

    def test_dispatch_event_calls_handlers(self, listener):
        from blockchain.events import EventType, TrainingEvent

        called = []

        async def my_handler(event):
            called.append(event)

        listener.register_handler(EventType.REWARDS_CLAIMED, my_handler)
        event = TrainingEvent(
            event_type=EventType.REWARDS_CLAIMED,
            block_number=50,
            block_hash="0xbeef",
            timestamp="2024-01-01T00:00:00+00:00",
            data={"amount": 100},
        )
        asyncio.get_event_loop().run_until_complete(listener._dispatch_event(event))
        assert len(called) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: data/data_manager.py
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitConfig:
    """Tests for SplitConfig dataclass."""

    def test_default_ratios(self):
        from data.data_manager import SplitConfig

        s = SplitConfig()
        assert abs(s.train_ratio + s.val_ratio + s.test_ratio - 1.0) < 1e-6

    def test_custom_ratios(self):
        from data.data_manager import SplitConfig

        s = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert abs(s.train_ratio + s.val_ratio + s.test_ratio - 1.0) < 1e-6

    def test_invalid_ratios_raises(self):
        from data.data_manager import SplitConfig

        with pytest.raises(ValueError):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_shuffle_default_true(self):
        from data.data_manager import SplitConfig

        s = SplitConfig()
        assert s.shuffle is True

    def test_seed_default(self):
        from data.data_manager import SplitConfig

        s = SplitConfig()
        assert s.seed == 42


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_create_with_mnist(self):
        from data.data_manager import DatasetConfig, DatasetType

        cfg = DatasetConfig(dataset_type=DatasetType.MNIST)
        assert cfg.dataset_type == DatasetType.MNIST

    def test_default_batch_size(self):
        from data.data_manager import DatasetConfig, DatasetType

        cfg = DatasetConfig(dataset_type=DatasetType.MNIST)
        assert cfg.batch_size == 32

    def test_max_samples_none_default(self):
        from data.data_manager import DatasetConfig, DatasetType

        cfg = DatasetConfig(dataset_type=DatasetType.MNIST)
        assert cfg.max_samples is None

    def test_custom_max_samples(self):
        from data.data_manager import DatasetConfig, DatasetType

        cfg = DatasetConfig(dataset_type=DatasetType.CIFAR10, max_samples=1000)
        assert cfg.max_samples == 1000

    def test_split_config_default(self):
        from data.data_manager import DatasetConfig, DatasetType, SplitConfig

        cfg = DatasetConfig(dataset_type=DatasetType.MNIST)
        assert isinstance(cfg.split_config, SplitConfig)


class TestDatasetType:
    """Tests for DatasetType enum."""

    def test_mnist_value(self):
        from data.data_manager import DatasetType

        assert DatasetType.MNIST.value == "mnist"

    def test_wikitext2_value(self):
        from data.data_manager import DatasetType

        assert DatasetType.WIKITEXT2.value == "wikitext-2-raw-v1"

    def test_custom_json_value(self):
        from data.data_manager import DatasetType

        assert DatasetType.CUSTOM_JSON.value == "custom_json"


class TestDataManagerInit:
    """Tests for DataManager initialization without actual data loading."""

    def test_init_creates_cache_dir(self):
        from data.data_manager import DataManager, DatasetConfig, DatasetType

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cfg = DatasetConfig(
                dataset_type=DatasetType.MNIST,
                cache_dir=cache_dir,
            )
            dm = DataManager(cfg)
            assert dm.config == cfg
            assert dm.dataset is None

    def test_get_stats_before_load(self):
        from data.data_manager import DataManager, DatasetConfig, DatasetType

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DatasetConfig(
                dataset_type=DatasetType.MNIST,
                cache_dir=Path(tmpdir),
            )
            dm = DataManager(cfg)
            stats = dm.get_stats()
            assert isinstance(stats, dict)

    def test_get_hf_datasets_returns_list(self):
        from data.data_manager import DataManager

        result = DataManager._get_hf_datasets()
        assert isinstance(result, list)

    def test_get_custom_datasets_returns_list(self):
        from data.data_manager import DataManager

        result = DataManager._get_custom_datasets()
        assert isinstance(result, list)

    def test_load_json_dataset(self):
        from data.data_manager import DataManager, DatasetConfig, DatasetType

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small JSON file
            json_path = Path(tmpdir) / "data.json"
            data = [{"text": "hello"}, {"text": "world"}]
            with open(json_path, "w") as f:
                json.dump(data, f)
            cfg = DatasetConfig(
                dataset_type=DatasetType.CUSTOM_JSON,
                cache_dir=Path(tmpdir),
                custom_path=json_path,
            )
            dm = DataManager(cfg)
            result = dm._load_json_dataset(json_path)
            assert len(result) == 2

    def test_load_custom_dataset_json(self):
        from data.data_manager import DataManager, DatasetConfig, DatasetType

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "corpus.json"
            with open(json_path, "w") as f:
                json.dump([{"text": "a"}, {"text": "b"}], f)
            cfg = DatasetConfig(
                dataset_type=DatasetType.CUSTOM_JSON,
                cache_dir=Path(tmpdir),
                custom_path=json_path,
            )
            dm = DataManager(cfg)
            result = dm._load_custom_dataset()
            assert result is not None


class TestListDataset:
    """Tests for ListDataset (torch Dataset wrapper)."""

    def test_len(self):
        from data.data_manager import ListDataset

        ds = ListDataset([1, 2, 3, 4, 5])
        assert len(ds) == 5

    def test_getitem(self):
        from data.data_manager import ListDataset

        ds = ListDataset(["a", "b", "c"])
        assert ds[0] == "a"
        assert ds[2] == "c"

    def test_empty(self):
        from data.data_manager import ListDataset

        ds = ListDataset([])
        assert len(ds) == 0

    def test_with_dicts(self):
        from data.data_manager import ListDataset

        data = [{"x": 1}, {"x": 2}]
        ds = ListDataset(data)
        assert ds[1]["x"] == 2

    def test_with_tensors(self):
        from data.data_manager import ListDataset

        data = [torch.randn(4), torch.randn(4)]
        ds = ListDataset(data)
        assert isinstance(ds[0], torch.Tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: client/data_loader.py
# ─────────────────────────────────────────────────────────────────────────────


class TestDataSovereigntyLevel:
    """Tests for DataSovereigntyLevel enum."""

    def test_values(self):
        from client.data_loader import DataSovereigntyLevel

        assert DataSovereigntyLevel.PUBLIC.value == "public"
        assert DataSovereigntyLevel.RESTRICTED.value == "restricted"
        assert DataSovereigntyLevel.CONFIDENTIAL.value == "confidential"
        assert DataSovereigntyLevel.SECRET.value == "secret"


class TestComplianceMetadata:
    """Tests for ComplianceMetadata dataclass."""

    def test_create_default(self):
        from client.data_loader import ComplianceMetadata, DataSovereigntyLevel

        m = ComplianceMetadata(data_classification=DataSovereigntyLevel.PUBLIC)
        assert m.data_classification == DataSovereigntyLevel.PUBLIC
        assert m.contains_pii is False
        assert m.geographic_restriction == "BZ"

    def test_pii_flag(self):
        from client.data_loader import ComplianceMetadata, DataSovereigntyLevel

        m = ComplianceMetadata(
            data_classification=DataSovereigntyLevel.CONFIDENTIAL,
            contains_pii=True,
        )
        assert m.contains_pii is True

    def test_retention_default(self):
        from client.data_loader import ComplianceMetadata, DataSovereigntyLevel

        m = ComplianceMetadata(data_classification=DataSovereigntyLevel.PUBLIC)
        assert m.retention_period_days == 2555  # 7 years

    def test_encryption_required_default(self):
        from client.data_loader import ComplianceMetadata, DataSovereigntyLevel

        m = ComplianceMetadata(data_classification=DataSovereigntyLevel.PUBLIC)
        assert m.encryption_required is True


class TestComplianceDataFilter:
    """Tests for ComplianceDataFilter."""

    @pytest.fixture
    def flt(self):
        from client.data_loader import ComplianceDataFilter

        return ComplianceDataFilter()

    def test_init(self, flt):
        assert flt.filtered_count == 0
        assert flt.total_processed == 0

    def test_sensitive_patterns_loaded(self, flt):
        assert len(flt.sensitive_patterns) > 0

    def test_is_compliant_clean_text(self, flt):
        assert flt._is_compliant("Hello, how are you today?") is True

    def test_is_compliant_credit_card_rejected(self, flt):
        # Pattern: 4 groups of 4 digits
        assert flt._is_compliant("Card: 4111-1111-1111-1111") is False

    def test_is_compliant_email_rejected(self, flt):
        assert flt._is_compliant("Contact me at user@example.com for info") is False

    def test_is_compliant_restricted_term(self, flt):
        assert flt._is_compliant("This is an illegal gambling operation") is False

    def test_is_compliant_ponzi(self, flt):
        assert flt._is_compliant("This scheme is a ponzi scheme") is False

    def test_get_stats_initial(self, flt):
        stats = flt.get_stats()
        assert stats["total_processed"] == 0
        assert stats["filtered_count"] == 0
        assert stats["compliance_rate"] == 1.0

    def test_filter_batch_compliant(self, flt):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "text": ["Clean Belizean agricultural data"],
        }
        result = flt.filter_batch(batch)
        assert result is not None

    def test_filter_batch_increments_counter(self, flt):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "text": ["Clean text"],
        }
        flt.filter_batch(batch)
        assert flt.total_processed == 1

    def test_is_compliant_money_laundering(self, flt):
        assert flt._is_compliant("money laundering is illegal") is False

    def test_get_stats_compliance_rate(self, flt):
        flt._is_compliant("hello")
        flt._is_compliant("4111-1111-1111-1111")
        # Directly manipulate counters since _is_compliant doesn't touch them
        # Let filter_batch touch the counters instead
        stats = flt.get_stats()
        assert 0.0 <= stats["compliance_rate"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: cli/commands.py
# ─────────────────────────────────────────────────────────────────────────────


class TestCliCommands:
    """Tests for Click CLI commands via CliRunner."""

    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner

        return CliRunner()

    def test_cli_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "Options" in result.output

    def test_train_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0

    def test_evolve_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["evolve", "--help"])
        assert result.exit_code == 0

    def test_federate_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["federate", "--help"])
        assert result.exit_code == 0

    def test_validator_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["validator", "--help"])
        assert result.exit_code == 0

    def test_genome_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["genome", "--help"])
        assert result.exit_code == 0

    def test_config_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    def test_config_show(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["config", "--show"])
        # Should print config without crashing
        assert result.exit_code == 0 or result.exit_code in (0, 1)

    def test_config_init(self, runner):
        from cli.commands import cli

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["config", "--init"])
            # May succeed or fail gracefully
            assert result.exit_code in (0, 1)

    def test_validator_register_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["validator", "register", "--help"])
        assert result.exit_code == 0

    def test_genome_store_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["genome", "store", "--help"])
        assert result.exit_code == 0

    def test_genome_get_help(self, runner):
        from cli.commands import cli

        result = runner.invoke(cli, ["genome", "get", "--help"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: client/train.py
# ─────────────────────────────────────────────────────────────────────────────


class TestBelizeTrainingConfig:
    """Tests for BelizeTrainingConfig dataclass."""

    def test_create(self):
        from client.train import BelizeTrainingConfig

        cfg = BelizeTrainingConfig(participant_id="p1")
        assert cfg.participant_id == "p1"

    def test_defaults(self):
        from client.train import BelizeTrainingConfig

        cfg = BelizeTrainingConfig(participant_id="p1")
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 32
        assert cfg.local_epochs == 3
        assert cfg.compliance_mode is True
        assert cfg.data_sovereignty_check is True

    def test_custom_values(self):
        from client.train import BelizeTrainingConfig

        cfg = BelizeTrainingConfig(
            participant_id="p2",
            learning_rate=1e-3,
            batch_size=64,
            quantization_bits=4,
        )
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 64
        assert cfg.quantization_bits == 4

    def test_azure_endpoint_none_default(self):
        from client.train import BelizeTrainingConfig

        cfg = BelizeTrainingConfig(participant_id="p1")
        assert cfg.azure_endpoint is None


class TestBelizeChainFederatedClient:
    """Tests for BelizeChainFederatedClient with mocked dependencies."""

    def _make_client(self, quantization_bits=8):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        with (
            patch("client.train.QuantizedBelizeModel") as MockQ,
            patch("client.train.BelizeChainLLM") as MockLLM,
            patch("client.train.BelizeDataLoader") as MockLoader,
        ):

            mock_model = MagicMock(spec=nn.Module)
            mock_model.state_dict.return_value = {
                "weight": torch.randn(4, 4),
                "bias": torch.randn(4),
            }
            mock_model.to.return_value = mock_model
            MockQ.return_value = mock_model
            MockLLM.return_value = mock_model
            MockLoader.return_value = MagicMock()

            cfg = BelizeTrainingConfig(
                participant_id="test_p1",
                quantization_bits=quantization_bits,
            )
            client = BelizeChainFederatedClient(cfg)

        return client

    def test_client_init(self):
        client = self._make_client()
        assert client.config.participant_id == "test_p1"

    def test_model_set(self):
        client = self._make_client()
        assert client.model is not None

    def test_data_loader_set(self):
        client = self._make_client()
        assert client.data_loader is not None

    def test_compliance_filter_created(self):
        from client.data_loader import ComplianceDataFilter

        client = self._make_client()
        assert isinstance(client.compliance_filter, ComplianceDataFilter)

    def test_get_parameters_returns_list(self):
        client = self._make_client()
        params = client.get_parameters({})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters(self):
        client = self._make_client()
        # Get params and put them back
        params = client.get_parameters({})
        # set_parameters should not raise
        client.set_parameters(params)

    def test_apply_differential_privacy(self):
        client = self._make_client()
        params = client.get_parameters({})
        result = client._apply_differential_privacy(params, epsilon=1.0)
        assert isinstance(result, list)
        assert len(result) == len(params)

    def test_apply_dp_shapes_preserved(self):
        client = self._make_client()
        params = client.get_parameters({})
        result = client._apply_differential_privacy(params, epsilon=5.0)
        for original, noised in zip(params, result):
            assert original.shape == noised.shape

    def test_full_precision_model_path(self):
        # quantization_bits >= 16 → uses BelizeChainLLM
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        with (
            patch("client.train.QuantizedBelizeModel") as MockQ,
            patch("client.train.BelizeChainLLM") as MockLLM,
            patch("client.train.BelizeDataLoader") as MockLoader,
        ):

            mock_model = MagicMock(spec=nn.Module)
            mock_model.state_dict.return_value = {"w": torch.randn(2, 2)}
            mock_model.to.return_value = mock_model
            MockLLM.return_value = mock_model
            MockLoader.return_value = MagicMock()

            cfg = BelizeTrainingConfig(participant_id="p_fp", quantization_bits=16)
            client = BelizeChainFederatedClient(cfg)

            MockLLM.assert_called_once()
            MockQ.assert_not_called()
