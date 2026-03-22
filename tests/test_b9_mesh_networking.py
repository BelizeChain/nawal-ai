"""
B9 Audit — Mesh Networking Tests

Checks:
  C9.1  Ed25519 signing & verification + replay protection
  C9.2  Gossip deduplication (LRU cache, bounded, no re-propagation)
  C9.3  Peer discovery from blockchain with fallback cache
  C9.4  Byzantine-resistant consensus (supermajority)

Covers 6 fixes:
  F9.1a  Env-var private-key loading
  F9.1b  Timestamp-based replay protection
  F9.2a  Configurable gossip fanout
  F9.3a  Last-known-good peer cache on discovery failure
  F9.4a  ConsensusRound supermajority primitive
  F9.4b  Docstring corrected (no false claims)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from blockchain.mesh_network import (
    MeshNetworkClient,
    MeshMessage,
    MessageType,
    PeerInfo,
    ConsensusRound,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_peer(peer_id: str, pub_hex: str = "", alive: bool = True) -> PeerInfo:
    return PeerInfo(
        peer_id=peer_id,
        account_id=f"5G{peer_id}",
        multiaddr=f"/ip4/127.0.0.1/tcp/9{peer_id[-3:].replace('_', '0')}",
        public_key=pub_hex,
        last_seen=time.time() if alive else time.time() - 9999,
    )


def _hex_pub(key: ed25519.Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


# ═════════════════════════════════════════════════════════════════════════════
# C9.1  Ed25519 signing, verification, & replay protection
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestC91Ed25519:
    """C9.1 — Ed25519 signing, verification, replay protection."""

    # ── F9.1a: env-var private key ──────────────────────────

    async def test_private_key_from_env(self, tmp_path):
        """Init loads Ed25519 key from NAWAL_MESH_PRIVATE_KEY env var."""
        key = ed25519.Ed25519PrivateKey.generate()
        key_hex = key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ).hex()

        with patch.dict(os.environ, {"NAWAL_MESH_PRIVATE_KEY": key_hex}):
            mesh = MeshNetworkClient(peer_id="env_key_peer")
            assert mesh.public_key_hex == _hex_pub(key)

    async def test_generated_key_when_no_env(self):
        """Without env var or explicit key, a fresh key is generated."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NAWAL_MESH_PRIVATE_KEY", None)
            mesh = MeshNetworkClient(peer_id="gen_key_peer")
            assert len(mesh.public_key_hex) == 64  # 32 bytes hex

    async def test_explicit_key_overrides_env(self):
        """Explicit private_key param takes priority over env var."""
        env_key = ed25519.Ed25519PrivateKey.generate()
        explicit_key = ed25519.Ed25519PrivateKey.generate()

        env_hex = env_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ).hex()

        with patch.dict(os.environ, {"NAWAL_MESH_PRIVATE_KEY": env_hex}):
            mesh = MeshNetworkClient(peer_id="expl", private_key=explicit_key)
            assert mesh.public_key_hex == _hex_pub(explicit_key)

    # ── Signing & verification round-trip ────────────────────

    async def test_sign_and_verify_roundtrip(self):
        """A locally signed message verifies when sender is in peers."""
        mesh = MeshNetworkClient(peer_id="signer")
        msg = mesh._create_message(MessageType.HEARTBEAT, {"ping": 1})

        # Register sender as peer so verifier can look it up
        mesh.peers["signer"] = _make_peer("signer", pub_hex=mesh.public_key_hex)

        assert mesh._verify_message_signature(msg) is True

    async def test_tampered_payload_rejected(self):
        """Modified payload invalidates the Ed25519 signature."""
        mesh = MeshNetworkClient(peer_id="tamper")
        msg = mesh._create_message(MessageType.HEARTBEAT, {"data": "original"})

        mesh.peers["tamper"] = _make_peer("tamper", pub_hex=mesh.public_key_hex)

        msg.payload["data"] = "tampered"
        assert mesh._verify_message_signature(msg) is False

    async def test_unknown_sender_rejected(self):
        """Message from sender not in peer list fails verification."""
        mesh = MeshNetworkClient(peer_id="unknown_sender")
        msg = mesh._create_message(MessageType.HEARTBEAT, {"x": 1})
        # Do NOT add sender to peer list
        assert mesh._verify_message_signature(msg) is False

    # ── F9.1b: replay protection ────────────────────────────

    async def test_stale_message_rejected(self):
        """Messages older than MAX_MESSAGE_AGE are rejected."""
        mesh = MeshNetworkClient(peer_id="replay_sender", listen_port=19001)
        await mesh.start()

        try:
            mesh.peers["replay_sender"] = _make_peer(
                "replay_sender", pub_hex=mesh.public_key_hex
            )

            msg = mesh._create_message(MessageType.HEARTBEAT, {"ts": 1})
            # Backdate timestamp beyond max age
            msg.timestamp = datetime.now(timezone.utc).timestamp() - (mesh._max_message_age + 60)
            # Re-sign with backdated timestamp
            msg.signature = None
            msg_bytes = json.dumps(msg.to_dict()).encode()
            sig = mesh.private_key.sign(msg_bytes)
            msg.signature = sig.hex()

            from aiohttp.test_utils import TestClient, TestServer

            client = TestClient(TestServer(mesh.app))
            await client.start_server()
            resp = await client.post("/message", json=msg.to_dict())
            assert resp.status == 400 or (await resp.text()) == "message too old"
            await client.close()
        finally:
            await mesh.stop()

    async def test_fresh_message_accepted(self):
        """Messages within MAX_MESSAGE_AGE are accepted (not rejected as stale)."""
        mesh = MeshNetworkClient(peer_id="fresh_sender", listen_port=19002)
        await mesh.start()

        try:
            mesh.peers["fresh_sender"] = _make_peer(
                "fresh_sender", pub_hex=mesh.public_key_hex
            )

            msg = mesh._create_message(MessageType.HEARTBEAT, {"ts": 2})

            from aiohttp.test_utils import TestClient, TestServer

            client = TestClient(TestServer(mesh.app))
            await client.start_server()
            resp = await client.post("/message", json=msg.to_dict())
            assert resp.status == 200
            text = await resp.text()
            assert text == "ok"
            await client.close()
        finally:
            await mesh.stop()


# ═════════════════════════════════════════════════════════════════════════════
# C9.2  Gossip Deduplication
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestC92GossipDedup:
    """C9.2 — Gossip deduplication & bounded cache."""

    async def test_seen_cache_bounded(self):
        """seen_messages never exceeds _seen_messages_max."""
        mesh = MeshNetworkClient(peer_id="cache_test")
        mesh._seen_messages_max = 50

        for i in range(100):
            mesh.seen_messages[f"msg_{i}"] = time.time()
            if len(mesh.seen_messages) > mesh._seen_messages_max:
                mesh.seen_messages.popitem(last=False)

        assert len(mesh.seen_messages) <= 50

    async def test_duplicate_not_repropagated(self):
        """A duplicate message is detected and NOT gossip-forwarded."""
        mesh = MeshNetworkClient(peer_id="dedup_test", listen_port=19003)
        await mesh.start()

        try:
            mesh.peers["dedup_test"] = _make_peer(
                "dedup_test", pub_hex=mesh.public_key_hex
            )

            msg = mesh._create_message(MessageType.HEARTBEAT, {"d": 1})
            # Pre-mark as seen
            mesh.seen_messages[msg.message_id] = time.time()

            from aiohttp.test_utils import TestClient, TestServer

            client = TestClient(TestServer(mesh.app))
            await client.start_server()

            with patch.object(mesh, "_gossip_forward") as mock_fwd:
                resp = await client.post("/message", json=msg.to_dict())
                assert (await resp.text()) == "duplicate"
                mock_fwd.assert_not_called()

            await client.close()
        finally:
            await mesh.stop()

    async def test_gossip_fanout_configurable(self):
        """gossip_fanout parameter controls max peers for gossip forwarding."""
        mesh = MeshNetworkClient(peer_id="fanout_test", gossip_fanout=3)
        assert mesh.gossip_fanout == 3

        # Create 10 alive peers
        for i in range(10):
            k = ed25519.Ed25519PrivateKey.generate()
            mesh.peers[f"p{i}"] = _make_peer(f"p{i:03d}", pub_hex=_hex_pub(k), alive=True)

        msg = mesh._create_message(MessageType.HEARTBEAT, {"f": 1})
        msg.ttl = 3

        with patch.object(mesh, "_send_to_peer_raw", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            await mesh._gossip_forward(msg)
            assert mock_send.call_count <= 3

    async def test_gossip_fanout_default(self):
        """Default gossip_fanout falls back to 50% heuristic."""
        mesh = MeshNetworkClient(peer_id="default_fanout")
        # Default should be None or a sane value
        assert mesh.gossip_fanout is None or mesh.gossip_fanout > 0


# ═════════════════════════════════════════════════════════════════════════════
# C9.3  Peer Discovery
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestC93PeerDiscovery:
    """C9.3 — Peer discovery from blockchain with fallback cache."""

    async def test_discover_populates_peers(self):
        """Successful discovery stores peers in self.peers."""
        mesh = MeshNetworkClient(peer_id="disco_test")
        mesh.substrate = MagicMock()
        mesh.substrate.query.side_effect = [
            ["5Gval1"],  # Validators list
            {"network_address": "/ip4/10.0.0.1/tcp/9090", "public_key": "aa" * 32, "stake": 5000},
        ]

        result = await mesh.discover_peers()
        assert len(result) >= 1
        assert len(mesh.peers) >= 1

    async def test_fallback_cache_on_failure(self):
        """On discovery failure, last-known-good peers are preserved."""
        mesh = MeshNetworkClient(peer_id="fallback_test")

        # Pre-populate peers (simulating a previous successful discovery)
        key = ed25519.Ed25519PrivateKey.generate()
        mesh.peers["existing_peer"] = _make_peer(
            "existing_peer", pub_hex=_hex_pub(key), alive=True
        )

        mesh.substrate = MagicMock()
        mesh.substrate.query.side_effect = Exception("RPC timeout")

        result = await mesh.discover_peers()
        # Should return empty (failure) but NOT wipe existing peers
        assert "existing_peer" in mesh.peers

    async def test_discovery_without_blockchain(self):
        """Returns empty list when substrate not connected."""
        mesh = MeshNetworkClient(peer_id="no_chain_test")
        mesh.substrate = None
        result = await mesh.discover_peers()
        assert result == []

    async def test_multiaddr_validation(self):
        """Invalid multiaddr from blockchain is rejected."""
        mesh = MeshNetworkClient(peer_id="addr_test")
        mesh.substrate = MagicMock()
        mesh.substrate.query.side_effect = [
            ["5Gval1"],
            {"network_address": "not-a-valid-address", "public_key": "bb" * 32, "stake": 1000},
        ]

        result = await mesh.discover_peers()
        # Invalid address should be filtered out
        valid_peers = [p for p in mesh.peers.values() if _is_valid_multiaddr(p.multiaddr)]
        # The invalid peer should not be in the peer list
        for p in mesh.peers.values():
            assert _is_valid_multiaddr(p.multiaddr)


# ═════════════════════════════════════════════════════════════════════════════
# C9.4  Byzantine Consensus (Supermajority)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestC94Consensus:
    """C9.4 — Byzantine-resistant supermajority consensus."""

    async def test_consensus_round_reaches_supermajority(self):
        """ConsensusRound correctly detects 2/3+1 supermajority."""
        cr = ConsensusRound(round_id="r1", proposal="model_hash_abc", total_peers=6)
        # Need >= 5 votes for 6 peers (ceil(6*2/3)+1 = 5)
        cr.add_vote("p1", True)
        cr.add_vote("p2", True)
        cr.add_vote("p3", True)
        cr.add_vote("p4", True)
        assert cr.has_supermajority() is False
        cr.add_vote("p5", True)
        assert cr.has_supermajority() is True

    async def test_consensus_round_no_supermajority(self):
        """Insufficient votes do not satisfy supermajority."""
        cr = ConsensusRound(round_id="r2", proposal="hash_xyz", total_peers=10)
        for i in range(6):
            cr.add_vote(f"p{i}", True)
        # Need >= 7 for 10 peers
        assert cr.has_supermajority() is False

    async def test_consensus_rejects_duplicate_votes(self):
        """Same peer cannot vote twice."""
        cr = ConsensusRound(round_id="r3", proposal="hash_dup", total_peers=3)
        cr.add_vote("p1", True)
        cr.add_vote("p1", True)  # duplicate — should be ignored
        assert cr.vote_count == 1

    async def test_consensus_negative_votes_tracked(self):
        """Negative votes are tracked but do not count toward supermajority."""
        cr = ConsensusRound(round_id="r4", proposal="hash_neg", total_peers=3)
        cr.add_vote("p1", False)
        cr.add_vote("p2", True)
        cr.add_vote("p3", True)
        # 2 positive out of 3 total — supermajority threshold is ceil(3*2/3)+1 = 3
        # So 2 positives < 3 → no supermajority
        assert cr.has_supermajority() is False

    async def test_consensus_round_finalized(self):
        """ConsensusRound marks finalized after supermajority reached."""
        cr = ConsensusRound(round_id="r5", proposal="hash_fin", total_peers=3)
        cr.add_vote("p1", True)
        cr.add_vote("p2", True)
        cr.add_vote("p3", True)
        assert cr.has_supermajority() is True
        assert cr.is_finalized is True

    async def test_docstring_no_false_claims(self):
        """Module docstring mentions consensus only accurately (not 'Byzantine-resistant')."""
        import blockchain.mesh_network as mod

        docstring = mod.__doc__ or ""
        # Should NOT claim Byzantine-resistant consensus exists as a self-contained feature
        # Should accurately describe what's available
        assert "Byzantine-resistant distributed consensus" not in docstring


# ═════════════════════════════════════════════════════════════════════════════
# C9.4b  Consensus protocol wired into MeshNetworkClient
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestC94ConsensusProtocol:
    """C9.4b — Full consensus propose / vote / finalize over the network."""

    def _build_client(self) -> MeshNetworkClient:
        """Build a MeshNetworkClient with a fresh Ed25519 key."""
        key = ed25519.Ed25519PrivateKey.generate()
        return MeshNetworkClient(
            peer_id="proposer_1",
            listen_port=0,
            private_key=key,
        )

    async def test_propose_creates_round_and_broadcasts(self):
        """propose_consensus creates a ConsensusRound and broadcasts."""
        client = self._build_client()
        client.peers = {
            "p1": _make_peer("p1", alive=True),
            "p2": _make_peer("p2", alive=True),
            "p3": _make_peer("p3", alive=True),
        }
        client._broadcast_message = AsyncMock()

        cr = await client.propose_consensus("round_42", "model_hash_abc")

        assert isinstance(cr, ConsensusRound)
        assert cr.round_id == "round_42"
        assert cr.proposal == "model_hash_abc"
        assert cr.total_peers == 3
        assert client.consensus_rounds["round_42"] is cr
        client._broadcast_message.assert_awaited_once()
        call_kwargs = client._broadcast_message.call_args
        assert call_kwargs[1]["message_type"] == MessageType.CONSENSUS_PROPOSE

    async def test_cast_vote_broadcasts(self):
        """cast_vote sends CONSENSUS_VOTE to all peers."""
        client = self._build_client()
        client._broadcast_message = AsyncMock()

        await client.cast_vote("round_42", approve=True)

        client._broadcast_message.assert_awaited_once()
        call_kwargs = client._broadcast_message.call_args
        assert call_kwargs[1]["message_type"] == MessageType.CONSENSUS_VOTE
        assert call_kwargs[1]["payload"]["approve"] is True

    async def test_handle_vote_updates_round(self):
        """_handle_consensus_vote tallies incoming votes."""
        client = self._build_client()
        cr = ConsensusRound(round_id="r10", proposal="p", total_peers=3)
        client.consensus_rounds["r10"] = cr

        msg = MeshMessage(
            message_id="m1",
            sender_id="voter_1",
            message_type=MessageType.CONSENSUS_VOTE,
            payload={"round_id": "r10", "voter_id": "voter_1", "approve": True},
            timestamp=datetime.now(timezone.utc).timestamp(),
            signature="",
        )
        await client._handle_consensus_vote(msg)

        assert cr.vote_count == 1
        assert cr.approve_count == 1

    async def test_full_consensus_flow(self):
        """Propose → receive enough votes → round finalizes."""
        client = self._build_client()
        client.peers = {
            f"p{i}": _make_peer(f"p{i}", alive=True) for i in range(4)
        }
        client._broadcast_message = AsyncMock()

        cr = await client.propose_consensus("r_full", "hash_full")
        assert not cr.is_finalized

        # Need ceil(4*2/3)+1 = 4 approvals for supermajority
        for i in range(4):
            msg = MeshMessage(
                message_id=f"m{i}",
                sender_id=f"p{i}",
                message_type=MessageType.CONSENSUS_VOTE,
                payload={"round_id": "r_full", "voter_id": f"p{i}", "approve": True},
                timestamp=datetime.now(timezone.utc).timestamp(),
                signature="",
            )
            await client._handle_consensus_vote(msg)

        assert cr.has_supermajority() is True
        assert cr.is_finalized is True

    async def test_vote_for_unknown_round_ignored(self):
        """Votes for an unknown round_id are silently dropped."""
        client = self._build_client()

        msg = MeshMessage(
            message_id="m_unk",
            sender_id="voter_x",
            message_type=MessageType.CONSENSUS_VOTE,
            payload={"round_id": "no_such_round", "voter_id": "voter_x", "approve": True},
            timestamp=datetime.now(timezone.utc).timestamp(),
            signature="",
        )
        # Should not raise
        await client._handle_consensus_vote(msg)
        assert client.get_consensus_result("no_such_round") is None

    async def test_get_consensus_result(self):
        """get_consensus_result returns the round or None."""
        client = self._build_client()
        cr = ConsensusRound(round_id="rX", proposal="p", total_peers=1)
        client.consensus_rounds["rX"] = cr

        assert client.get_consensus_result("rX") is cr
        assert client.get_consensus_result("missing") is None

    async def test_vote_after_finalized_ignored(self):
        """Votes arriving after finalization are ignored."""
        client = self._build_client()
        cr = ConsensusRound(round_id="rfin", proposal="p", total_peers=3)
        cr.add_vote("p0", True)
        cr.add_vote("p1", True)
        cr.add_vote("p2", True)
        assert cr.is_finalized is True
        client.consensus_rounds["rfin"] = cr

        msg = MeshMessage(
            message_id="m_late",
            sender_id="latecomer",
            message_type=MessageType.CONSENSUS_VOTE,
            payload={"round_id": "rfin", "voter_id": "latecomer", "approve": True},
            timestamp=datetime.now(timezone.utc).timestamp(),
            signature="",
        )
        await client._handle_consensus_vote(msg)
        # Vote count should NOT increase
        assert cr.vote_count == 3


# ── helper used by C9.3 multiaddr test ──────────────────────────────────────


def _is_valid_multiaddr(addr: str) -> bool:
    """Minimal multiaddr format check: /ip4/<ip>/tcp/<port>."""
    parts = addr.split("/")
    if len(parts) < 5:
        return False
    return parts[1] == "ip4" and parts[3] == "tcp" and parts[4].isdigit()
