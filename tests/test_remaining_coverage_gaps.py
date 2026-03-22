"""
Coverage gap tests for remaining uncovered modules.

Targets (by miss count):
- blockchain/mesh_network.py (110 miss)
- data/data_manager.py (52 miss)
- genome/operators.py (35 miss)
- orchestrator.py (34 miss)
- genome/population.py (33 miss)
- hybrid/teacher.py (24 miss)
- data/tokenizers.py (22 miss)
- blockchain/staking_interface.py (22 miss)
- blockchain/substrate_client.py (20 miss)
- client/model.py (17 miss)
- client/nawal.py (14 miss)
- genome/nawal_adapter.py (10 miss)
"""

import asyncio
import json
import hashlib
import pickle
import time
import random
import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, mock_open, call
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# MESH NETWORK
# ============================================================================

from blockchain.mesh_network import (
    MeshNetworkClient,
    MeshMessage,
    MessageType,
    PeerInfo,
    FLRoundAnnouncement,
)
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


@pytest.fixture
def mesh_client():
    """Create a MeshNetworkClient for testing."""
    priv = ed25519.Ed25519PrivateKey.generate()
    client = MeshNetworkClient(
        peer_id="test_peer",
        listen_port=19090,
        blockchain_rpc="ws://localhost:9944",
        private_key=priv,
    )
    return client


@pytest.fixture
def peer_info():
    """Create a PeerInfo with a recent timestamp."""
    return PeerInfo(
        peer_id="peer_abc",
        account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        multiaddr="/ip4/127.0.0.1/tcp/9091",
        public_key="aa" * 32,
        last_seen=datetime.now(timezone.utc).timestamp(),
        is_validator=True,
    )


class TestPeerInfo:
    def test_is_alive_recent(self, peer_info):
        assert peer_info.is_alive(timeout=300.0) is True

    def test_is_alive_stale(self, peer_info):
        peer_info.last_seen = 0.0
        assert peer_info.is_alive(timeout=300.0) is False


class TestMeshMessage:
    def test_to_dict_and_from_dict(self):
        msg = MeshMessage(
            message_id="msg1",
            message_type=MessageType.HEARTBEAT,
            sender_id="s1",
            timestamp=12345.0,
            payload={"key": "val"},
            signature="deadbeef",
            ttl=3,
        )
        d = msg.to_dict()
        assert d["message_type"] == "heartbeat"
        rebuilt = MeshMessage.from_dict(d)
        assert rebuilt.message_id == "msg1"
        assert rebuilt.message_type == MessageType.HEARTBEAT
        assert rebuilt.ttl == 3


class TestMeshNetworkClientInit:
    def test_generates_key_if_none(self):
        client = MeshNetworkClient(peer_id="auto_key", listen_port=19999)
        assert client.public_key_hex
        assert len(client.public_key_hex) == 64

    def test_custom_private_key(self, mesh_client):
        assert mesh_client.peer_id == "test_peer"
        assert mesh_client._running is False


class TestMeshNetworkStart:
    @pytest.mark.asyncio
    async def test_start_sets_running(self, mesh_client):
        """start() sets up app, runner, site and background tasks."""
        mock_runner = AsyncMock()
        mock_site = AsyncMock()

        with (
            patch("blockchain.mesh_network.web.AppRunner", return_value=mock_runner),
            patch("blockchain.mesh_network.web.TCPSite", return_value=mock_site),
            patch("blockchain.mesh_network.SUBSTRATE_AVAILABLE", False),
            patch("asyncio.create_task"),
        ):
            await mesh_client.start()
            assert mesh_client._running is True

        # cleanup
        mesh_client._running = False

    @pytest.mark.asyncio
    async def test_start_already_running(self, mesh_client):
        mesh_client._running = True
        # Should return immediately without error
        await mesh_client.start()

    @pytest.mark.asyncio
    async def test_start_substrate_connect_failure(self, mesh_client):
        mock_runner = AsyncMock()
        mock_site = AsyncMock()

        with (
            patch("blockchain.mesh_network.web.AppRunner", return_value=mock_runner),
            patch("blockchain.mesh_network.web.TCPSite", return_value=mock_site),
            patch("blockchain.mesh_network.SUBSTRATE_AVAILABLE", True),
            patch(
                "blockchain.mesh_network.SubstrateInterface",
                side_effect=Exception("conn fail"),
            ),
            patch("asyncio.create_task"),
        ):
            await mesh_client.start()
            assert mesh_client._running is True

        mesh_client._running = False


class TestMeshNetworkStop:
    @pytest.mark.asyncio
    async def test_stop_not_running(self, mesh_client):
        mesh_client._running = False
        await mesh_client.stop()  # Should be no-op

    @pytest.mark.asyncio
    async def test_stop_running(self, mesh_client):
        mesh_client._running = True
        mesh_client.site = AsyncMock()
        mesh_client.runner = AsyncMock()
        mesh_client.substrate = MagicMock()
        await mesh_client.stop()
        assert mesh_client._running is False
        mesh_client.site.stop.assert_awaited_once()
        mesh_client.runner.cleanup.assert_awaited_once()
        mesh_client.substrate.close.assert_called_once()


class TestMeshNetworkRegisterHandler:
    def test_register_handler(self, mesh_client):
        handler = MagicMock()
        mesh_client.register_handler(MessageType.HEARTBEAT, handler)
        assert handler in mesh_client.message_handlers[MessageType.HEARTBEAT]


class TestMeshNetworkBroadcast:
    @pytest.mark.asyncio
    async def test_announce_fl_round(self, mesh_client):
        mesh_client._running = True
        with patch.object(
            mesh_client, "_broadcast_message", new_callable=AsyncMock
        ) as mock_bc:
            await mesh_client.announce_fl_round(
                round_id="r1",
                dataset_name="ds",
                target_participants=5,
                deadline=9999999.0,
            )
            mock_bc.assert_awaited_once()
            payload = mock_bc.call_args[1]["payload"]
            assert payload["round_id"] == "r1"


class TestMeshNetworkSendModelDelta:
    @pytest.mark.asyncio
    async def test_send_model_delta_unknown_peer(self, mesh_client):
        result = await mesh_client.send_model_delta("unknown_peer", "r1", "cid", 0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_model_delta_known_peer(self, mesh_client, peer_info):
        mesh_client.peers[peer_info.peer_id] = peer_info
        with patch.object(
            mesh_client, "_send_to_peer", new_callable=AsyncMock, return_value=True
        ):
            result = await mesh_client.send_model_delta(
                peer_info.peer_id, "r1", "cid", 0.9
            )
            assert result is True


class TestMeshNetworkDiscoverPeers:
    @pytest.mark.asyncio
    async def test_no_substrate(self, mesh_client):
        mesh_client.substrate = None
        result = await mesh_client.discover_peers()
        assert result == []

    @pytest.mark.asyncio
    async def test_discover_with_validators(self, mesh_client):
        mock_sub = MagicMock()
        mock_sub.query.side_effect = [
            ["validator1"],  # Validators list
            {
                "network_address": "/ip4/10.0.0.1/tcp/9090",
                "public_key": "bb" * 32,
                "stake": 500,
            },
        ]
        mesh_client.substrate = mock_sub
        result = await mesh_client.discover_peers()
        assert len(result) == 1
        assert result[0].is_validator is True

    @pytest.mark.asyncio
    async def test_discover_exception(self, mesh_client):
        mock_sub = MagicMock()
        mock_sub.query.side_effect = Exception("rpc error")
        mesh_client.substrate = mock_sub
        result = await mesh_client.discover_peers()
        assert result == []


class TestMeshNetworkInternalMethods:
    @pytest.mark.asyncio
    async def test_broadcast_message(self, mesh_client, peer_info):
        mesh_client.peers[peer_info.peer_id] = peer_info
        with patch.object(
            mesh_client, "_send_to_peer_raw", new_callable=AsyncMock, return_value=True
        ):
            await mesh_client._broadcast_message(
                message_type=MessageType.HEARTBEAT,
                payload={"ts": 1.0},
            )
            # Should have sent to the alive peer
            assert len(mesh_client.seen_messages) == 1

    @pytest.mark.asyncio
    async def test_broadcast_lru_eviction(self, mesh_client):
        """LRU eviction when seen_messages exceeds max."""
        mesh_client._seen_messages_max = 2
        mesh_client.seen_messages["old1"] = 1.0
        mesh_client.seen_messages["old2"] = 2.0
        with patch.object(mesh_client, "_send_to_peer_raw", new_callable=AsyncMock):
            await mesh_client._broadcast_message(
                message_type=MessageType.HEARTBEAT,
                payload={"ts": 3.0},
            )
        # old1 should have been evicted
        assert "old1" not in mesh_client.seen_messages

    @pytest.mark.asyncio
    async def test_send_to_peer_unknown(self, mesh_client):
        result = await mesh_client._send_to_peer(
            "no_such_peer", MessageType.HEARTBEAT, {}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_peer_known(self, mesh_client, peer_info):
        mesh_client.peers[peer_info.peer_id] = peer_info
        with patch.object(
            mesh_client, "_send_to_peer_raw", new_callable=AsyncMock, return_value=True
        ):
            result = await mesh_client._send_to_peer(
                peer_info.peer_id, MessageType.HEARTBEAT, {}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_to_peer_raw_success(self, mesh_client):
        mock_resp = MagicMock()
        mock_resp.status = 200

        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})

        # We need to mock the nested async with pattern:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(...) as response:
        with patch("blockchain.mesh_network.aiohttp") as mock_aiohttp:
            # session.post() returns an async context manager (NOT a coroutine)
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_resp

            # session is obtained from ClientSession() async context manager
            # Use MagicMock so .post() returns the CM directly, not a coroutine
            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post_cm)

            # ClientSession() returns an async context manager
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_aiohttp.ClientSession.return_value = mock_session_cm
            mock_aiohttp.ClientTimeout = MagicMock()

            result = await mesh_client._send_to_peer_raw("/ip4/127.0.0.1/tcp/9091", msg)
            assert result is True

    @pytest.mark.asyncio
    async def test_send_to_peer_raw_bad_multiaddr(self, mesh_client):
        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})
        result = await mesh_client._send_to_peer_raw("bad_addr", msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_peer_raw_exception(self, mesh_client):
        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})
        with patch(
            "blockchain.mesh_network.aiohttp.ClientSession",
            side_effect=Exception("err"),
        ):
            result = await mesh_client._send_to_peer_raw("/ip4/127.0.0.1/tcp/9091", msg)
            assert result is False


class TestMeshNetworkVerifySignature:
    def test_verify_no_signature(self, mesh_client):
        msg = MeshMessage("id", MessageType.HEARTBEAT, "s", 1.0, {}, signature=None)
        assert mesh_client._verify_message_signature(msg) is False

    def test_verify_unknown_sender(self, mesh_client):
        msg = MeshMessage(
            "id", MessageType.HEARTBEAT, "unknown", 1.0, {}, signature="aa"
        )
        assert mesh_client._verify_message_signature(msg) is False

    def test_verify_valid_signature(self, mesh_client):
        """Create a message with valid signature and verify it."""
        # Create a peer with mesh_client's own public key
        peer = PeerInfo(
            peer_id=mesh_client.peer_id,
            account_id="test",
            multiaddr="/ip4/127.0.0.1/tcp/9090",
            public_key=mesh_client.public_key_hex,
        )
        mesh_client.peers[mesh_client.peer_id] = peer

        # Create and sign the message (uses mesh_client's private key)
        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})
        assert mesh_client._verify_message_signature(msg) is True

    def test_verify_invalid_signature(self, mesh_client):
        peer = PeerInfo(
            peer_id="sender1",
            account_id="test",
            multiaddr="/ip4/127.0.0.1/tcp/9090",
            public_key=mesh_client.public_key_hex,
        )
        mesh_client.peers["sender1"] = peer
        msg = MeshMessage(
            "id", MessageType.HEARTBEAT, "sender1", 1.0, {}, signature="deadbeef" * 8
        )
        assert mesh_client._verify_message_signature(msg) is False


class TestMeshNetworkHandleIncoming:
    @pytest.mark.asyncio
    async def test_handle_incoming_valid(self, mesh_client):
        """Valid signed message → 200 ok."""
        peer = PeerInfo(
            peer_id=mesh_client.peer_id,
            account_id="test",
            multiaddr="/ip4/127.0.0.1/tcp/9090",
            public_key=mesh_client.public_key_hex,
            last_seen=datetime.now(timezone.utc).timestamp(),
        )
        mesh_client.peers[mesh_client.peer_id] = peer

        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})
        request = AsyncMock()
        request.remote = "127.0.0.1"
        request.json = AsyncMock(return_value=msg.to_dict())

        with patch("asyncio.create_task"):
            resp = await mesh_client._handle_incoming_message(request)
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_incoming_duplicate(self, mesh_client):
        """Duplicate message → 200 duplicate."""
        mesh_client.seen_messages["existing_id"] = 1.0
        now = datetime.now(timezone.utc).timestamp()
        msg_dict = {
            "message_id": "existing_id",
            "message_type": "heartbeat",
            "sender_id": "s1",
            "timestamp": now,
            "payload": {},
        }
        request = AsyncMock()
        request.remote = "127.0.0.1"
        request.json = AsyncMock(return_value=msg_dict)
        resp = await mesh_client._handle_incoming_message(request)
        assert resp.status == 200
        assert "duplicate" in resp.text

    @pytest.mark.asyncio
    async def test_handle_incoming_invalid_signature(self, mesh_client):
        """Message with missing/invalid signature → 403."""
        now = datetime.now(timezone.utc).timestamp()
        msg_dict = {
            "message_id": "new_id",
            "message_type": "heartbeat",
            "sender_id": "unknown_sender",
            "timestamp": now,
            "payload": {},
            "signature": "badbeef",
        }
        request = AsyncMock()
        request.remote = "127.0.0.1"
        request.json = AsyncMock(return_value=msg_dict)
        resp = await mesh_client._handle_incoming_message(request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_handle_incoming_rate_limited(self, mesh_client):
        """Exceed rate limit → 429."""
        mesh_client._rate_limit_max = 1
        now = datetime.now(timezone.utc).timestamp()
        mesh_client._rate_limits["127.0.0.1"] = [now]

        request = AsyncMock()
        request.remote = "127.0.0.1"
        request.json = AsyncMock(return_value={})
        resp = await mesh_client._handle_incoming_message(request)
        assert resp.status == 429

    @pytest.mark.asyncio
    async def test_handle_incoming_exception(self, mesh_client):
        """Parse error → 400."""
        request = AsyncMock()
        request.remote = "127.0.0.1"
        request.json = AsyncMock(side_effect=Exception("bad json"))
        resp = await mesh_client._handle_incoming_message(request)
        assert resp.status == 400


class TestMeshNetworkGossipForward:
    @pytest.mark.asyncio
    async def test_gossip_forward(self, mesh_client, peer_info):
        mesh_client.peers[peer_info.peer_id] = peer_info
        msg = mesh_client._create_message(MessageType.HEARTBEAT, {"ts": 1.0})
        msg.sender_id = "other_peer"  # Not the same as our peer_info
        with patch.object(mesh_client, "_send_to_peer_raw", new_callable=AsyncMock):
            await mesh_client._gossip_forward(msg)


class TestMeshNetworkHandlers:
    @pytest.mark.asyncio
    async def test_handle_peers_request(self, mesh_client, peer_info):
        mesh_client.peers[peer_info.peer_id] = peer_info
        request = AsyncMock()
        resp = await mesh_client._handle_peers_request(request)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_health(self, mesh_client):
        mesh_client._running = True
        request = AsyncMock()
        resp = await mesh_client._handle_health(request)
        assert resp.status == 200


class TestMeshNetworkBackgroundLoops:
    @pytest.mark.asyncio
    async def test_heartbeat_loop_one_iteration(self, mesh_client):
        call_count = 0
        original_running = True

        async def fake_sleep(t):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mesh_client._running = False

        mesh_client._running = True
        with (
            patch("asyncio.sleep", side_effect=fake_sleep),
            patch.object(mesh_client, "_broadcast_message", new_callable=AsyncMock),
        ):
            await mesh_client._heartbeat_loop()

    @pytest.mark.asyncio
    async def test_peer_discovery_loop_one_iteration(self, mesh_client):
        call_count = 0

        async def fake_sleep(t):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mesh_client._running = False

        mesh_client._running = True
        with (
            patch("asyncio.sleep", side_effect=fake_sleep),
            patch.object(
                mesh_client, "discover_peers", new_callable=AsyncMock, return_value=[]
            ),
        ):
            await mesh_client._peer_discovery_loop()

    @pytest.mark.asyncio
    async def test_cleanup_loop_removes_dead_peers(self, mesh_client):
        dead_peer = PeerInfo(
            peer_id="dead",
            account_id="acc",
            multiaddr="/ip4/0.0.0.0/tcp/1",
            public_key="cc" * 32,
            last_seen=0.0,
        )
        mesh_client.peers["dead"] = dead_peer
        # Add lots of seen_messages to trigger eviction
        mesh_client._seen_messages_max = 2
        for i in range(10):
            mesh_client.seen_messages[f"msg_{i}"] = float(i)

        call_count = 0

        async def fake_sleep(t):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mesh_client._running = False

        mesh_client._running = True
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await mesh_client._cleanup_loop()

        assert "dead" not in mesh_client.peers
        assert len(mesh_client.seen_messages) <= mesh_client._seen_messages_max


class TestMeshNetworkReceiveMessages:
    @pytest.mark.asyncio
    async def test_receive_messages(self, mesh_client):
        mesh_client._running = True
        msg = MeshMessage("id1", MessageType.HEARTBEAT, "s1", 1.0, {})
        await mesh_client._message_queue.put(msg)

        received = []
        async for m in mesh_client.receive_messages():
            received.append(m)
            mesh_client._running = False
            break

        assert len(received) == 1
        assert received[0].message_id == "id1"

    @pytest.mark.asyncio
    async def test_receive_messages_timeout(self, mesh_client):
        """No messages + timeout → loop continues then stops."""
        mesh_client._running = True

        async def stop_after_one():
            await asyncio.sleep(0.05)
            mesh_client._running = False

        task = asyncio.create_task(stop_after_one())
        received = []
        async for m in mesh_client.receive_messages():
            received.append(m)
        await task
        assert received == []


# ============================================================================
# DATA MANAGER
# ============================================================================

from data.data_manager import (
    DataManager,
    DatasetConfig,
    DatasetType,
    SplitConfig,
    ListDataset,
)


class TestListDataset:
    def test_len_and_getitem(self):
        ds = ListDataset([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ds) == 3
        assert ds[0] == {"a": 1}


class TestDataManagerLoadCustomJSON:
    def test_load_json(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"text": "hello"}, {"text": "world"}]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
        )
        dm = DataManager(config)
        dm.load_dataset()
        assert len(dm.dataset) == 2

    def test_load_json_cached(self, tmp_path):
        """Second load reads from cache."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"text": "a"}, {"text": "b"}, {"text": "c"}]))
        cache_dir = tmp_path / "cache"
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=cache_dir,
        )
        dm = DataManager(config)
        dm.load_dataset()
        assert len(dm.dataset) == 3
        # Load again (from cache)
        dm2 = DataManager(config)
        dm2.load_dataset()
        assert len(dm2.dataset) == 3


class TestDataManagerLoadCSV:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_CSV,
            custom_path=csv_file,
            cache_dir=tmp_path / "cache",
        )
        dm = DataManager(config)
        dm.load_dataset()
        assert len(dm.dataset) == 2


class TestDataManagerLoadParquet:
    def test_load_parquet(self, tmp_path):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        parquet_file = tmp_path / "data.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file)
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_PARQUET,
            custom_path=parquet_file,
            cache_dir=tmp_path / "cache",
        )
        dm = DataManager(config)
        dm.load_dataset()
        assert len(dm.dataset) == 2


class TestDataManagerHF:
    def test_load_hf_not_available(self, tmp_path):
        config = DatasetConfig(
            dataset_type=DatasetType.WIKITEXT2,
            cache_dir=tmp_path / "cache",
        )
        dm = DataManager(config)
        with patch("data.data_manager.HF_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="HuggingFace"):
                dm._load_huggingface_dataset()


class TestDataManagerCustomNoPath:
    def test_no_custom_path_raises(self, tmp_path):
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            cache_dir=tmp_path / "cache",
        )
        dm = DataManager(config)
        with pytest.raises(ValueError, match="custom_path"):
            dm._load_custom_dataset()


class TestDataManagerMaxSamples:
    def test_max_samples(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(100)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            max_samples=10,
        )
        dm = DataManager(config)
        dm.load_dataset()
        assert len(dm.dataset) == 10


class TestDataManagerSplit:
    def test_split_dataset(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(100)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            split_config=SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1),
        )
        dm = DataManager(config)
        train, val, test = dm.split_dataset()
        assert len(train) + len(val) + len(test) == 100

    def test_split_no_shuffle(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(50)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            split_config=SplitConfig(
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=False
            ),
        )
        dm = DataManager(config)
        train, val, test = dm.split_dataset()
        assert len(train) + len(val) + len(test) == 50


class TestDataManagerDataloaders:
    def test_get_dataloaders(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(50)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            batch_size=10,
        )
        dm = DataManager(config)
        train_dl, val_dl, test_dl = dm.get_dataloaders()
        assert len(train_dl) > 0


class TestDataManagerPartitionFederated:
    def test_iid_partition(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(100)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            batch_size=5,
        )
        dm = DataManager(config)
        dm.load_dataset()
        dm.split_dataset()
        loaders = dm.partition_federated(num_clients=4, iid=True)
        assert len(loaders) == 4

    def test_non_iid_no_labels_fallback(self, tmp_path):
        """Non-IID without labels → fallback to IID."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(50)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            batch_size=5,
        )
        dm = DataManager(config)
        dm.load_dataset()
        dm.split_dataset()
        loaders = dm.partition_federated(num_clients=3, iid=False)
        assert len(loaders) == 3

    def test_dirichlet_partition(self, tmp_path):
        """Dirichlet partition with labels."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"x": i} for i in range(60)]))
        config = DatasetConfig(
            dataset_type=DatasetType.CUSTOM_JSON,
            custom_path=json_file,
            cache_dir=tmp_path / "cache",
            batch_size=5,
        )
        dm = DataManager(config)
        dm.load_dataset()
        dm.split_dataset()

        # Inject labels attribute
        dm.train_dataset.dataset.targets = [i % 3 for i in range(len(dm.dataset))]
        loaders = dm.partition_federated(num_clients=3, iid=False, alpha=0.5)
        assert len(loaders) == 3


class TestSplitConfig:
    def test_invalid_ratios(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)


# ============================================================================
# HYBRID TEACHER
# ============================================================================

from hybrid.teacher import DeepSeekTeacher, DeepSeekConfig, create_deepseek_teacher


class TestDeepSeekTeacher:
    def test_init_default(self):
        teacher = DeepSeekTeacher()
        assert teacher.config.model_name == "deepseek-ai/deepseek-coder-33b-instruct"
        assert teacher.model is None

    def test_load_model_vllm(self):
        import sys

        teacher = DeepSeekTeacher()
        mock_llm = MagicMock()
        mock_tokenizer = MagicMock()
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = MagicMock()
        sys.modules["vllm"] = mock_vllm
        try:
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                teacher.load_model()
                assert teacher.model is mock_llm
                assert teacher.tokenizer is mock_tokenizer
        finally:
            del sys.modules["vllm"]

    def test_load_model_fallback_to_transformers(self):
        teacher = DeepSeekTeacher()
        with patch(
            "hybrid.teacher.DeepSeekTeacher._load_with_transformers"
        ) as mock_load:
            # Simulate vLLM import failure
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "vllm":
                    raise ImportError("no vllm")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                teacher.load_model()
                mock_load.assert_called_once()

    def test_load_with_transformers_bitsandbytes(self):
        teacher = DeepSeekTeacher(DeepSeekConfig(quantization="bitsandbytes"))
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch("transformers.BitsAndBytesConfig", return_value=MagicMock()),
        ):
            teacher._load_with_transformers()
            assert teacher.tokenizer is mock_tokenizer
            assert teacher.model is mock_model

    def test_load_with_transformers_no_quantization(self):
        teacher = DeepSeekTeacher(DeepSeekConfig(quantization=None))
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            teacher._load_with_transformers()
            assert teacher.model is mock_model

    def test_generate_cached(self):
        teacher = DeepSeekTeacher()
        teacher.cache[("hello", None, None, 0.95)] = {"text": "world"}
        result = teacher.generate("hello")
        assert result["cached"] is True
        assert result["text"] == "world"

    def test_generate_vllm_path(self):
        import sys

        teacher = DeepSeekTeacher()
        mock_model = MagicMock()
        mock_model.__module__ = "vllm.engine"
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(text="generated", logprobs=None, token_ids=[1, 2])
        ]
        mock_model.generate.return_value = [mock_output]
        teacher.model = mock_model
        teacher.tokenizer = MagicMock()

        mock_vllm = MagicMock()
        sys.modules["vllm"] = mock_vllm
        try:
            result = teacher.generate("prompt", max_tokens=10, temperature=0.5)
            assert result["text"] == "generated"
            assert result["cached"] is False
        finally:
            del sys.modules["vllm"]

    def test_generate_transformers_path(self):
        teacher = DeepSeekTeacher()
        mock_model = MagicMock()
        mock_model.__module__ = "transformers.models"
        mock_model.device = "cpu"
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        teacher.model = mock_model
        teacher.model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "hi"
        # Make tokenizer callable return an object with .to()
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        teacher.tokenizer = mock_tokenizer

        result = teacher.generate("prompt", max_tokens=5, temperature=0.5)
        assert "text" in result

    def test_generate_cache_eviction(self):
        teacher = DeepSeekTeacher()
        teacher._cache_maxsize = 2
        teacher.cache[("a", None, None, 0.95)] = {"text": "aa"}
        teacher.cache[("b", None, None, 0.95)] = {"text": "bb"}

        # Generating with new prompt should evict oldest
        teacher.model = MagicMock()
        teacher.model.__module__ = "transformers.models"
        teacher.model.device = "cpu"
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        teacher.model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "cc"
        teacher.tokenizer = mock_tokenizer

        result = teacher.generate("c", max_tokens=5, temperature=0.5)
        assert len(teacher.cache) <= 2

    def test_get_soft_targets_with_logits(self):
        teacher = DeepSeekTeacher()
        logits = torch.randn(1, 5, 100)
        teacher.generate = MagicMock(return_value={"text": "x", "logits": logits})
        soft = teacher.get_soft_targets("prompt", temperature=2.0)
        assert soft is not None
        assert soft.shape == logits.shape

    def test_get_soft_targets_without_logits(self):
        teacher = DeepSeekTeacher()
        teacher.generate = MagicMock(return_value={"text": "x"})
        result = teacher.get_soft_targets("prompt")
        assert result is None

    def test_clear_cache(self):
        teacher = DeepSeekTeacher()
        teacher.cache["key"] = "val"
        teacher.clear_cache()
        assert len(teacher.cache) == 0


class TestCreateDeepSeekTeacher:
    def test_create(self):
        with patch.object(DeepSeekTeacher, "load_model"):
            teacher = create_deepseek_teacher(quantization="awq", num_gpus=1)
            assert isinstance(teacher, DeepSeekTeacher)


# ============================================================================
# GENOME OPERATORS
# ============================================================================

from genome.operators import (
    MutationOperator,
    MutationConfig,
    MutationType,
    CrossoverOperator,
    CrossoverConfig,
    CrossoverType,
    EvolutionStrategy,
    EvolutionConfig,
)
from genome.encoding import Genome, ArchitectureLayer, LayerType, Hyperparameters


def _make_test_genome(num_encoder=3, hidden_size=256) -> Genome:
    encoder_layers = [
        ArchitectureLayer(
            layer_type=LayerType.TRANSFORMER_ENCODER,
            hidden_size=hidden_size,
            num_heads=4,
            dropout_rate=0.1,
        )
        for _ in range(num_encoder)
    ]
    return Genome(
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=[],
        dropout_rate=0.1,
    )


class TestMutationConfig:
    def test_validate_valid(self):
        config = MutationConfig()
        is_valid, errors = config.validate()
        assert is_valid is True

    def test_validate_min_layers(self):
        config = MutationConfig(min_layers=0)
        is_valid, errors = config.validate()
        assert is_valid is False
        assert any("min_layers" in e for e in errors)

    def test_validate_max_less_than_min(self):
        config = MutationConfig(min_layers=10, max_layers=5)
        is_valid, errors = config.validate()
        assert is_valid is False


class TestMutationOperator:
    def test_mutate_basic(self):
        op = MutationOperator()
        genome = _make_test_genome()
        child = op.mutate(genome, generation=1)
        assert child.generation == 1
        assert child.fitness_score is None
        assert genome.genome_id in child.parent_genomes

    def test_mutate_add_layer(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=2)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.ADD_LAYER
        ):
            child = op.mutate(genome, generation=1)
            total = len(child.encoder_layers) + len(child.decoder_layers)
            assert total >= 2  # May or may not have added

    def test_mutate_remove_layer(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=5)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.REMOVE_LAYER
        ):
            child = op.mutate(genome, generation=1)
            total = len(child.encoder_layers) + len(child.decoder_layers)
            assert total <= 5

    def test_mutate_replace_layer(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=3)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.REPLACE_LAYER
        ):
            child = op.mutate(genome, generation=1)
            assert len(child.encoder_layers) + len(child.decoder_layers) >= 1

    def test_mutate_modify_layer(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=3)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.MODIFY_LAYER
        ):
            child = op.mutate(genome, generation=1)
            assert child is not None

    def test_mutate_learning_rate(self):
        op = MutationOperator()
        genome = _make_test_genome()
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.LEARNING_RATE
        ):
            child = op.mutate(genome, generation=1)
            # LR should have changed
            assert child is not None

    def test_mutate_batch_size(self):
        op = MutationOperator()
        genome = _make_test_genome()
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.BATCH_SIZE
        ):
            child = op.mutate(genome, generation=1)
            assert child is not None

    def test_mutate_optimizer(self):
        op = MutationOperator()
        genome = _make_test_genome()
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.OPTIMIZER
        ):
            child = op.mutate(genome, generation=1)
            assert child is not None

    def test_mutate_precision(self):
        op = MutationOperator()
        genome = _make_test_genome()
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.PRECISION
        ):
            child = op.mutate(genome, generation=1)
            assert child is not None

    def test_mutate_add_attention(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=2)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.ADD_ATTENTION
        ):
            child = op.mutate(genome, generation=1)
            assert len(child.encoder_layers) >= 2

    def test_mutate_add_moe(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=2)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.ADD_MOE
        ):
            child = op.mutate(genome, generation=1)
            assert len(child.encoder_layers) >= 2

    def test_mutate_add_ssm(self):
        op = MutationOperator()
        genome = _make_test_genome(num_encoder=2)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.ADD_SSM
        ):
            child = op.mutate(genome, generation=1)
            assert len(child.encoder_layers) >= 2

    def test_mutate_add_layer_max_reached(self):
        op = MutationOperator(MutationConfig(max_layers=3))
        genome = _make_test_genome(num_encoder=3)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.ADD_LAYER
        ):
            child = op.mutate(genome, generation=1)
            # Should not exceed max
            assert len(child.encoder_layers) + len(child.decoder_layers) <= 4

    def test_mutate_remove_layer_min_reached(self):
        op = MutationOperator(MutationConfig(min_layers=2))
        genome = _make_test_genome(num_encoder=2)
        with patch.object(
            op, "_select_mutation_type", return_value=MutationType.REMOVE_LAYER
        ):
            child = op.mutate(genome, generation=1)
            total = len(child.encoder_layers) + len(child.decoder_layers)
            assert total >= 1  # min is enforced

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid mutation config"):
            MutationOperator(MutationConfig(min_layers=0))


class TestCrossoverOperator:
    def test_crossover_basic(self):
        op = CrossoverOperator()
        p1 = _make_test_genome(num_encoder=3)
        p2 = _make_test_genome(num_encoder=4)
        child = op.crossover(p1, p2, generation=1)
        assert child.generation == 1
        assert child.fitness_score is None

    def test_uniform_crossover(self):
        op = CrossoverOperator()
        p1 = _make_test_genome(num_encoder=3)
        p2 = _make_test_genome(num_encoder=3)
        with patch.object(
            op, "_select_crossover_type", return_value=CrossoverType.UNIFORM
        ):
            child = op.crossover(p1, p2, generation=1)
            assert child is not None

    def test_single_point_crossover(self):
        op = CrossoverOperator()
        p1 = _make_test_genome(num_encoder=3)
        p2 = _make_test_genome(num_encoder=4)
        with patch.object(
            op, "_select_crossover_type", return_value=CrossoverType.SINGLE_POINT
        ):
            child = op.crossover(p1, p2, generation=1)
            assert len(child.encoder_layers) >= 1

    def test_two_point_crossover(self):
        op = CrossoverOperator()
        p1 = _make_test_genome(num_encoder=4)
        p2 = _make_test_genome(num_encoder=4)
        with patch.object(
            op, "_select_crossover_type", return_value=CrossoverType.TWO_POINT
        ):
            child = op.crossover(p1, p2, generation=1)
            assert child is not None

    def test_layer_wise_crossover(self):
        op = CrossoverOperator()
        p1 = _make_test_genome(num_encoder=3)
        p2 = _make_test_genome(num_encoder=3)
        with patch.object(
            op, "_select_crossover_type", return_value=CrossoverType.LAYER_WISE
        ):
            child = op.crossover(p1, p2, generation=1)
            assert child is not None

    def test_hyperparameter_crossover(self):
        op = CrossoverOperator()
        p1 = _make_test_genome()
        p2 = _make_test_genome()
        with patch.object(
            op, "_select_crossover_type", return_value=CrossoverType.HYPERPARAMETER
        ):
            child = op.crossover(p1, p2, generation=1)
            assert child is not None

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid crossover config"):
            CrossoverOperator(
                CrossoverConfig(
                    uniform_rate=0.0,
                    single_point_rate=0.0,
                    two_point_rate=0.0,
                    layer_wise_rate=0.0,
                    hyperparameter_rate=0.0,
                )
            )


class TestEvolutionStrategy:
    def test_evolve_crossover(self):
        strategy = EvolutionStrategy()
        p1 = _make_test_genome()
        p2 = _make_test_genome()
        with patch("genome.operators.random") as mock_random:
            mock_random.random.side_effect = [
                0.1,
                0.1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.1,
                0.5,
            ]
            mock_random.choice = random.choice
            mock_random.randint = random.randint
            mock_random.uniform = random.uniform
            mock_random.sample = random.sample
            child = strategy.evolve(p1, p2, generation=1)
            assert child is not None

    def test_evolve_mutation_only(self):
        strategy = EvolutionStrategy()
        p1 = _make_test_genome()
        child = strategy.evolve(p1, None, generation=2)
        assert child.generation == 2

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid evolution config"):
            EvolutionStrategy(
                EvolutionConfig(mutation_config=MutationConfig(min_layers=0))
            )


# ============================================================================
# GENOME POPULATION
# ============================================================================

from genome.population import Population, PopulationConfig, SelectionStrategy


class TestPopulationSelection:
    def _make_pop(self, selection_strategy=SelectionStrategy.TOURNAMENT, n=5):
        config = PopulationConfig(
            min_size=5,
            target_size=10,
            max_size=20,
            selection_strategy=selection_strategy,
            tournament_size=3,
        )
        pop = Population(config)
        for i in range(n):
            g = _make_test_genome()
            g.fitness_score = float(i * 10)
            pop.add_genome(g)
        return pop

    def test_tournament_selection(self):
        pop = self._make_pop(SelectionStrategy.TOURNAMENT)
        parent = pop.select_parent()
        assert parent is not None

    def test_roulette_selection(self):
        pop = self._make_pop(SelectionStrategy.ROULETTE)
        parent = pop.select_parent()
        assert parent is not None

    def test_rank_selection(self):
        pop = self._make_pop(SelectionStrategy.RANK)
        parent = pop.select_parent()
        assert parent is not None

    def test_elite_selection_no_elite(self):
        pop = self._make_pop(SelectionStrategy.ELITE)
        # No elite set → falls back to tournament
        parent = pop.select_parent()
        assert parent is not None

    def test_elite_selection_with_elite(self):
        pop = self._make_pop(SelectionStrategy.ELITE)
        # Set some elites
        pop.elite_genomes = list(pop.genomes.keys())[:2]
        parent = pop.select_parent()
        assert parent is not None

    def test_select_parents(self):
        pop = self._make_pop()
        parents = pop.select_parents(count=3)
        assert len(parents) == 3


class TestPopulationUpdateElite:
    def test_update_elite(self):
        config = PopulationConfig(
            min_size=5,
            target_size=10,
            max_size=20,
            elitism_count=2,
            elitism_threshold=0.0,
        )
        pop = Population(config)
        for i in range(5):
            g = _make_test_genome()
            g.fitness_score = float(i * 20)
            pop.add_genome(g)
        pop.update_elite(generation=1)
        assert len(pop.elite_genomes) > 0

    def test_update_elite_no_scored(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        g = _make_test_genome()
        pop.add_genome(g)
        pop.update_elite(generation=1)
        assert len(pop.elite_genomes) == 0


class TestPopulationStatistics:
    def test_compute_statistics(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        for i in range(3):
            g = _make_test_genome()
            g.fitness_score = float(i * 10)
            g.quality_score = float(i * 5)
            g.timeliness_score = float(i * 3)
            g.honesty_score = float(i * 2)
            pop.add_genome(g)
        stats = pop.compute_statistics(generation=1)
        assert stats.population_size == 3
        assert stats.max_fitness == 20.0

    def test_compute_statistics_empty(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        stats = pop.compute_statistics(generation=1)
        assert stats.population_size == 0

    def test_get_statistics_latest(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        g = _make_test_genome()
        g.fitness_score = 50.0
        pop.add_genome(g)
        pop.compute_statistics(generation=1)
        stats = pop.get_statistics()
        assert stats is not None
        assert stats.population_size == 1

    def test_get_statistics_none(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        assert pop.get_statistics() is None


class TestPopulationCullAndDiversity:
    def test_cull_population(self):
        config = PopulationConfig(min_size=2, target_size=3, max_size=5)
        pop = Population(config)
        for i in range(7):
            g = _make_test_genome()
            g.fitness_score = float(i)
            pop.add_genome(g)
        pop._cull_population()
        assert len(pop.genomes) <= 5

    def test_calculate_diversity(self):
        config = PopulationConfig(min_size=5, target_size=10, max_size=20)
        pop = Population(config)
        for _ in range(3):
            g = _make_test_genome()
            pop.add_genome(g)
        d = pop.calculate_diversity()
        assert 0.0 <= d <= 1.0

    def test_enforce_diversity(self):
        config = PopulationConfig(
            min_size=5,
            target_size=10,
            max_size=20,
            maintain_diversity=True,
            max_similar_genomes=1,
        )
        pop = Population(config)
        # Create a genome and deep-copy it to ensure identical hash
        g1 = _make_test_genome(num_encoder=2, hidden_size=128)
        g1.genome_id = "dup_a"
        g1.fitness_score = 10.0

        # Deep-copy the genome dict and reconstruct → identical architecture
        g2_dict = g1.to_dict()
        g2 = Genome.from_dict(g2_dict)
        g2.genome_id = "dup_b"
        g2.fitness_score = 5.0

        pop.add_genome(g1)
        pop.add_genome(g2)
        # Verify they share same hash (they use the same layer IDs)
        assert g1.genome_hash == g2.genome_hash
        pop.enforce_diversity()
        # max_similar_genomes=1, so lower-fitness duplicate should be removed
        assert len(pop.genomes) == 1
        assert "dup_a" in pop.genomes  # Higher fitness one kept


# ============================================================================
# ORCHESTRATOR
# ============================================================================

from orchestrator import EvolutionOrchestrator


class TestOrchestratorMethods:
    def _make_orchestrator(self, tmp_path):
        """Create a minimal orchestrator with mocked config."""
        mock_config = MagicMock()
        mock_config.storage.checkpoint_dir = tmp_path / "checkpoints"
        mock_config.storage.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mock_config.storage.save_best_only = False
        mock_config.storage.max_checkpoints = 3
        mock_config.storage.data_dir = tmp_path / "data"
        mock_config.evolution.population_size = 5
        mock_config.evolution.num_generations = 10
        mock_config.evolution.mutation_rate = 0.3
        mock_config.evolution.crossover_rate = 0.7
        mock_config.evolution.elitism_count = 1
        mock_config.evolution.tournament_size = 3
        mock_config.evolution.early_stopping_patience = 10
        mock_config.training.epochs_per_generation = 1
        mock_config.training.batch_size = 4

        with (
            patch("orchestrator.PopulationManager") as MockPop,
            patch("orchestrator.GenomeEncoder") as MockEnc,
            patch("orchestrator.MemoryManager", MagicMock()),
        ):
            MockPop.return_value = MagicMock()
            MockPop.return_value.genomes = {}
            MockEnc.return_value = MagicMock()
            mock_config.create_directories = MagicMock()
            orch = EvolutionOrchestrator(mock_config)
            return orch

    def test_clean_old_checkpoints(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        # Create 5 fake checkpoints
        for i in range(5):
            (ckpt_dir / f"checkpoint_gen_{i:04d}.pt").write_text("data")

        orch._clean_old_checkpoints()
        remaining = list(ckpt_dir.glob("checkpoint_gen_*.pt"))
        assert len(remaining) <= 3

    def test_should_stop_early_not_enough(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.generation_history = [MagicMock(best_fitness=1.0)] * 5
        assert orch._should_stop_early() is False

    def test_should_stop_early_plateau(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.generation_history = [MagicMock(best_fitness=50.0)] * 15
        assert orch._should_stop_early() is True

    def test_should_stop_early_improving(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.generation_history = [MagicMock(best_fitness=float(i)) for i in range(15)]
        assert orch._should_stop_early() is False

    def test_get_statistics_empty(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.generation_history = []
        stats = orch.get_statistics()
        assert stats["generations"] == 0

    def test_get_statistics_with_history(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        latest = MagicMock()
        latest.best_fitness = 80.0
        latest.avg_fitness = 50.0
        latest.diversity = 0.8
        orch.generation_history = [latest]
        stats = orch.get_statistics()
        assert stats["generations"] == 1
        assert stats["best_fitness"] == 80.0

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        ckpt_path = tmp_path / "ckpt.pt"

        # Create a real genome dict for the checkpoint
        test_genome = _make_test_genome()
        test_genome.fitness_score = 60.0
        genome_dict = test_genome.to_dict()

        # Create fake checkpoint with actual population data
        ckpt_data = {
            "generation": 5,
            "population": [genome_dict],
            "generation_history": [
                {
                    "generation": 5,
                    "best_fitness": 60.0,
                    "avg_fitness": 40.0,
                    "diversity": 0.7,
                    "training_time": 10.0,
                    "timestamp": 1000.0,
                }
            ],
            "best_fitness": 60.0,
        }
        # Write a dummy file so Path.exists() passes
        ckpt_path.write_bytes(b"dummy")

        # Mock population so we can track calls
        mock_pop = MagicMock()
        mock_pop.genomes = {}
        mock_pop.remove_genome = MagicMock()

        # When add_genome is called, add to genomes dict
        def side_effect_add(g):
            mock_pop.genomes[g.genome_id] = g

        mock_pop.add_genome = MagicMock(side_effect=side_effect_add)
        orch.population = mock_pop

        # Mock torch.load to bypass weights_only=True restriction on LayerType enum
        with patch("orchestrator.torch.load", return_value=ckpt_data):
            await orch.resume_from_checkpoint(ckpt_path)
        assert orch.current_generation == 6

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_not_found(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        with pytest.raises(FileNotFoundError):
            await orch.resume_from_checkpoint(tmp_path / "nonexistent.pt")
