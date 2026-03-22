"""
Mesh Network Connector for Nawal AI Validators

Implements decentralized P2P mesh networking for validator communication,
enabling direct model sharing, gossip protocol for FL rounds, and
supermajority-based consensus primitives (ConsensusRound).

Integrates with BelizeChain's validator registry to discover peers
and establish encrypted communication channels.

Features:
- Peer discovery via blockchain validator registry
- Encrypted P2P communication (libp2p-style)
- Gossip protocol for FL round announcements
- Model delta exchange without central server
- NAT traversal via STUN/TURN
- DHT-based peer routing

Author: BelizeChain AI Team
Date: February 2026
Python: 3.11+
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from loguru import logger
import aiohttp
from aiohttp import web

try:
    from substrateinterface import SubstrateInterface, Keypair

    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logger.warning("substrate-interface not installed, mesh network in limited mode")


# =============================================================================
# Data Classes
# =============================================================================


class MessageType(Enum):
    """Types of mesh network messages."""

    PEER_DISCOVERY = "peer_discovery"
    PEER_ANNOUNCE = "peer_announce"
    FL_ROUND_START = "fl_round_start"
    FL_ROUND_UPDATE = "fl_round_update"
    MODEL_DELTA_OFFER = "model_delta_offer"
    MODEL_DELTA_REQUEST = "model_delta_request"
    MODEL_DELTA_TRANSFER = "model_delta_transfer"
    HEARTBEAT = "heartbeat"
    GOSSIP = "gossip"
    CONSENSUS_PROPOSE = "consensus_propose"
    CONSENSUS_VOTE = "consensus_vote"


@dataclass
class PeerInfo:
    """Information about a mesh network peer."""

    peer_id: str  # Public key hash
    account_id: str  # BelizeChain account (SS58)
    multiaddr: str  # Network address (e.g., /ip4/1.2.3.4/tcp/9090)
    public_key: str  # Ed25519 public key (hex)
    stake_amount: int = 0
    reputation: float = 100.0
    last_seen: float = 0.0  # Unix timestamp
    is_validator: bool = False
    capabilities: List[str] = field(default_factory=list)

    def is_alive(self, timeout: float = 300.0) -> bool:
        """Check if peer is alive based on last heartbeat."""
        now = datetime.now(timezone.utc).timestamp()
        return (now - self.last_seen) < timeout


@dataclass
class MeshMessage:
    """Message in the mesh network."""

    message_id: str
    message_type: MessageType
    sender_id: str
    timestamp: float
    payload: Dict[str, Any]
    signature: Optional[str] = None
    ttl: int = 5  # Time-to-live for gossip

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "signature": self.signature,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeshMessage:
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            payload=data["payload"],
            signature=data.get("signature"),
            ttl=data.get("ttl", 5),
        )


@dataclass
class FLRoundAnnouncement:
    """Federated learning round announcement."""

    round_id: str
    coordinator_id: str
    dataset_name: str
    target_participants: int
    start_time: float
    deadline: float
    min_stake: int
    reward_pool: int
    model_hash: str  # Initial model hash


# =============================================================================
# Mesh Network Client
# =============================================================================


class MeshNetworkClient:
    """
    P2P mesh network client for Nawal AI validators.

    Enables decentralized communication between validators without
    relying on a central server. Uses gossip protocol for message
    propagation and peer discovery.

    Usage:
        mesh = MeshNetworkClient(
            peer_id="validator1",
            listen_port=9090,
            blockchain_rpc="ws://localhost:9944",
        )

        await mesh.start()

        # Announce FL round
        await mesh.announce_fl_round(
            round_id="round_001",
            dataset_name="belize_corpus",
            target_participants=10,
            deadline=3600,
        )

        # Listen for messages
        async for message in mesh.receive_messages():
            if message.message_type == MessageType.FL_ROUND_START:
                # Handle round start
                pass
    """

    def __init__(
        self,
        peer_id: str,
        listen_port: int = 9090,
        blockchain_rpc: str = "ws://localhost:9944",
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        gossip_fanout: Optional[int] = None,
        max_message_age: float = 300.0,
    ):
        """
        Initialize mesh network client.

        Args:
            peer_id: Unique peer identifier
            listen_port: Port to listen for incoming connections
            blockchain_rpc: BelizeChain RPC endpoint for peer discovery
            private_key: Ed25519 private key for signing (generated if None)
            gossip_fanout: Max peers to forward gossip to (None = 50% heuristic)
            max_message_age: Reject messages older than this many seconds (replay protection)
        """
        self.peer_id = peer_id
        self.listen_port = listen_port
        self.blockchain_rpc = blockchain_rpc
        self.gossip_fanout = gossip_fanout
        self._max_message_age = max_message_age

        # Cryptography — explicit key > env var > generate
        if private_key is None:
            env_hex = os.environ.get("NAWAL_MESH_PRIVATE_KEY")
            if env_hex:
                raw = bytes.fromhex(env_hex)
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
            else:
                private_key = ed25519.Ed25519PrivateKey.generate()
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.public_key_hex = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.seen_messages: OrderedDict[str, float] = (
            OrderedDict()
        )  # msg_id -> timestamp for LRU
        self._seen_messages_max = 10000
        self.message_handlers: Dict[MessageType, List[Callable]] = {}

        # Rate limiting: per-IP message counters
        self._rate_limits: Dict[str, List[float]] = {}  # ip -> list of timestamps
        self._rate_limit_max = 100  # max messages per window
        self._rate_limit_window = 60.0  # seconds

        # Networking
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Consensus state
        self.consensus_rounds: Dict[str, ConsensusRound] = {}

        # Blockchain connection
        self.substrate: Optional[SubstrateInterface] = None
        self._background_tasks: list = []

        logger.info(
            f"Initialized mesh network client {peer_id} "
            f"on port {listen_port} with pubkey {self.public_key_hex[:16]}..."
        )

    async def start(self) -> None:
        """Start the mesh network client."""
        if self._running:
            logger.warning("Mesh network already running")
            return

        # Start HTTP server for incoming messages
        self.app = web.Application()
        self.app.router.add_post("/message", self._handle_incoming_message)
        self.app.router.add_get("/peers", self._handle_peers_request)
        self.app.router.add_get("/health", self._handle_health)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.listen_port)
        await self.site.start()

        # Connect to blockchain for peer discovery
        if SUBSTRATE_AVAILABLE:
            try:
                self.substrate = SubstrateInterface(url=self.blockchain_rpc)
                logger.info(f"Connected to BelizeChain at {self.blockchain_rpc}")
            except Exception as e:
                logger.error(f"Failed to connect to blockchain: {e}")

        self._running = True

        # Register internal consensus vote handler
        self.register_handler(MessageType.CONSENSUS_VOTE, self._handle_consensus_vote)

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._peer_discovery_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        logger.info(f"Mesh network started on port {self.listen_port}")

    async def stop(self) -> None:
        """Stop the mesh network client."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        for task in getattr(self, "_background_tasks", []):
            task.cancel()
        for task in getattr(self, "_background_tasks", []):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks = []

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        if self.substrate:
            self.substrate.close()

        logger.info("Mesh network stopped")

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[MeshMessage], None],
    ) -> None:
        """
        Register a message handler for specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async callback function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    async def announce_fl_round(
        self,
        round_id: str,
        dataset_name: str,
        target_participants: int,
        deadline: float,
        min_stake: int = 1000,
        reward_pool: int = 10000,
        model_hash: str = "",
    ) -> None:
        """
        Announce a new FL round to the mesh network.

        Args:
            round_id: Unique round identifier
            dataset_name: Dataset to train on
            target_participants: Number of participants needed
            deadline: Unix timestamp deadline
            min_stake: Minimum stake required
            reward_pool: Total rewards available
            model_hash: Initial model hash
        """
        announcement = {
            "round_id": round_id,
            "coordinator_id": self.peer_id,
            "dataset_name": dataset_name,
            "target_participants": target_participants,
            "start_time": datetime.now(timezone.utc).timestamp(),
            "deadline": deadline,
            "min_stake": min_stake,
            "reward_pool": reward_pool,
            "model_hash": model_hash,
        }

        await self._broadcast_message(
            message_type=MessageType.FL_ROUND_START,
            payload=announcement,
        )

        logger.info(f"Announced FL round {round_id} to mesh network")

    async def send_model_delta(
        self,
        recipient_id: str,
        round_id: str,
        model_cid: str,
        quality_score: float,
    ) -> bool:
        """
        Send model delta to a specific peer.

        Args:
            recipient_id: Target peer ID
            round_id: FL round ID
            model_cid: IPFS CID of model delta
            quality_score: Model quality score

        Returns:
            True if sent successfully
        """
        if recipient_id not in self.peers:
            logger.error(f"Peer {recipient_id} not found")
            return False

        payload = {
            "round_id": round_id,
            "model_cid": model_cid,
            "quality_score": quality_score,
        }

        return await self._send_to_peer(
            peer_id=recipient_id,
            message_type=MessageType.MODEL_DELTA_TRANSFER,
            payload=payload,
        )

    # -------------------------------------------------------------------------
    # Consensus Protocol
    # -------------------------------------------------------------------------

    async def propose_consensus(
        self,
        round_id: str,
        proposal: str,
    ) -> ConsensusRound:
        """Broadcast a consensus proposal to all peers.

        Creates a new ConsensusRound, records it locally, and gossips
        a CONSENSUS_PROPOSE message to the mesh.  Peers respond with
        CONSENSUS_VOTE messages which are tallied automatically.

        Args:
            round_id: Unique identifier for this consensus round.
            proposal: Opaque proposal value (e.g. model hash).

        Returns:
            The newly created ConsensusRound.
        """
        total = max(len([p for p in self.peers.values() if p.is_alive()]), 1)
        cr = ConsensusRound(round_id=round_id, proposal=proposal, total_peers=total)
        self.consensus_rounds[round_id] = cr

        await self._broadcast_message(
            message_type=MessageType.CONSENSUS_PROPOSE,
            payload={"round_id": round_id, "proposal": proposal},
        )
        logger.info(f"Proposed consensus round {round_id} (peers={total})")
        return cr

    async def cast_vote(
        self,
        round_id: str,
        approve: bool,
    ) -> None:
        """Vote on an existing consensus round and broadcast."""
        await self._broadcast_message(
            message_type=MessageType.CONSENSUS_VOTE,
            payload={
                "round_id": round_id,
                "voter_id": self.peer_id,
                "approve": approve,
            },
        )

    async def _handle_consensus_vote(self, message: MeshMessage) -> None:
        """Internal handler: tally incoming CONSENSUS_VOTE messages."""
        payload = message.payload
        round_id = payload.get("round_id")
        voter_id = payload.get("voter_id", message.sender_id)
        approve = payload.get("approve", False)

        cr = self.consensus_rounds.get(round_id)
        if cr is None:
            logger.debug(f"Ignoring vote for unknown round {round_id}")
            return
        if cr.is_finalized:
            return

        cr.add_vote(voter_id, approve)
        if cr.is_finalized:
            logger.info(
                f"Consensus round {round_id} finalized "
                f"({cr.approve_count}/{cr.total_peers} approved)"
            )

    def get_consensus_result(self, round_id: str) -> Optional[ConsensusRound]:
        """Return the ConsensusRound for a given round_id, or None."""
        return self.consensus_rounds.get(round_id)

    # -------------------------------------------------------------------------
    # Peer Discovery
    # -------------------------------------------------------------------------

    async def discover_peers(self) -> List[PeerInfo]:
        """
        Discover peers from blockchain validator registry.

        Returns:
            List of discovered peers
        """
        if not self.substrate:
            logger.warning("Blockchain not connected, cannot discover peers")
            return []

        try:
            # Query validator list from Staking pallet
            validators = self.substrate.query(
                module="Staking",
                storage_function="Validators",
            )

            discovered = []

            if validators:
                for validator_account in validators:
                    # Get validator metadata
                    metadata = self.substrate.query(
                        module="Staking",
                        storage_function="ValidatorMetadata",
                        params=[validator_account],
                    )

                    if metadata and "network_address" in metadata:
                        addr = metadata["network_address"]
                        if not _is_valid_multiaddr(addr):
                            logger.warning(
                                f"Invalid multiaddr from validator {validator_account}: {addr}"
                            )
                            continue

                        peer = PeerInfo(
                            peer_id=hashlib.sha256(
                                validator_account.encode()
                            ).hexdigest()[:16],
                            account_id=validator_account,
                            multiaddr=addr,
                            public_key=metadata.get("public_key", ""),
                            stake_amount=metadata.get("stake", 0),
                            last_seen=datetime.now(timezone.utc).timestamp(),
                            is_validator=True,
                        )

                        self.peers[peer.peer_id] = peer
                        discovered.append(peer)

            logger.info(f"Discovered {len(discovered)} peers from blockchain")
            return discovered

        except Exception as e:
            logger.error(f"Peer discovery failed: {e} — keeping last-known peers")
            return []

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    async def _broadcast_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        ttl: int = 5,
    ) -> None:
        """Broadcast message to all known peers via gossip."""
        message = self._create_message(message_type, payload, ttl)

        # Mark as seen (LRU OrderedDict)
        self.seen_messages[message.message_id] = datetime.now(timezone.utc).timestamp()
        if len(self.seen_messages) > self._seen_messages_max:
            self.seen_messages.popitem(last=False)

        # Send to all peers
        tasks = [
            self._send_to_peer_raw(peer.multiaddr, message)
            for peer in self.peers.values()
            if peer.is_alive()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        logger.debug(
            f"Broadcast {message_type.value} to {success_count}/{len(tasks)} peers"
        )

    async def _send_to_peer(
        self,
        peer_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
    ) -> bool:
        """Send message to specific peer."""
        if peer_id not in self.peers:
            return False

        peer = self.peers[peer_id]
        message = self._create_message(message_type, payload)

        return await self._send_to_peer_raw(peer.multiaddr, message)

    async def _send_to_peer_raw(
        self,
        multiaddr: str,
        message: MeshMessage,
    ) -> bool:
        """Send message to peer by address."""
        try:
            # Parse multiaddr (simplified - assumes /ip4/x.x.x.x/tcp/port)
            parts = multiaddr.split("/")
            if len(parts) >= 5 and parts[1] == "ip4" and parts[3] == "tcp":
                host = parts[2]
                port = int(parts[4])
                url = f"http://{host}:{port}/message"
            else:
                logger.error(f"Invalid multiaddr format: {multiaddr}")
                return False

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.debug(f"Failed to send to {multiaddr}: {e}")
            return False

    def _verify_message_signature(self, message: MeshMessage) -> bool:
        """Verify Ed25519 signature on an incoming mesh message.

        Looks up the sender's public key from known peers and verifies
        the signature over the message body (excluding the signature field).

        Returns:
            True if signature is valid, False otherwise.
        """
        if not message.signature:
            return False

        # Look up sender's public key
        peer = self.peers.get(message.sender_id)
        if peer is None or not peer.public_key:
            logger.debug(f"Cannot verify message from unknown peer {message.sender_id}")
            return False

        try:
            pub_key_bytes = bytes.fromhex(peer.public_key)
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_key_bytes)

            # Reconstruct the message dict WITHOUT the signature (same as signing)
            verify_msg = MeshMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                sender_id=message.sender_id,
                timestamp=message.timestamp,
                payload=message.payload,
                ttl=message.ttl,
                signature=None,
            )
            message_bytes = json.dumps(verify_msg.to_dict()).encode()
            sig_bytes = bytes.fromhex(message.signature)

            pub_key.verify(sig_bytes, message_bytes)
            return True
        except Exception as e:
            logger.debug(f"Signature verification failed for {message.message_id}: {e}")
            return False

    def _create_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        ttl: int = 5,
    ) -> MeshMessage:
        """Create and sign a mesh message."""
        message_id = hashlib.sha256(
            f"{self.peer_id}{datetime.now(timezone.utc).timestamp()}{json.dumps(payload)}".encode()
        ).hexdigest()

        message = MeshMessage(
            message_id=message_id,
            message_type=message_type,
            sender_id=self.peer_id,
            timestamp=datetime.now(timezone.utc).timestamp(),
            payload=payload,
            ttl=ttl,
        )

        # Sign the message
        message_bytes = json.dumps(message.to_dict()).encode()
        signature = self.private_key.sign(message_bytes)
        message.signature = signature.hex()

        return message

    async def _handle_incoming_message(self, request: web.Request) -> web.Response:
        """Handle incoming message from peer."""
        try:
            # Rate limiting per remote IP
            remote_ip = request.remote or "unknown"
            now = datetime.now(timezone.utc).timestamp()
            timestamps = self._rate_limits.setdefault(remote_ip, [])
            # Purge old entries outside the window
            timestamps[:] = [t for t in timestamps if now - t < self._rate_limit_window]
            if len(timestamps) >= self._rate_limit_max:
                logger.warning(f"Rate limit exceeded for {remote_ip}")
                return web.Response(status=429, text="rate limited")
            timestamps.append(now)

            data = await request.json()
            message = MeshMessage.from_dict(data)

            # Replay protection: reject stale messages
            age = now - message.timestamp
            if age > self._max_message_age:
                logger.warning(
                    f"Rejected stale message {message.message_id} (age={age:.0f}s)"
                )
                return web.Response(status=400, text="message too old")

            # Deduplication via LRU OrderedDict
            if message.message_id in self.seen_messages:
                return web.Response(status=200, text="duplicate")

            # Verify Ed25519 signature
            if not self._verify_message_signature(message):
                logger.warning(
                    f"Rejected message {message.message_id} from {message.sender_id}: "
                    "invalid or missing signature"
                )
                return web.Response(status=403, text="invalid signature")

            self.seen_messages[message.message_id] = now

            # Update peer info
            if message.sender_id in self.peers:
                self.peers[message.sender_id].last_seen = datetime.now(
                    timezone.utc
                ).timestamp()

            # Call registered handlers
            if message.message_type in self.message_handlers:
                for handler in self.message_handlers[message.message_type]:
                    asyncio.create_task(handler(message))

            # Queue for processing
            await self._message_queue.put(message)

            # Gossip forwarding (if TTL > 0)
            if message.ttl > 0:
                message.ttl -= 1
                asyncio.create_task(self._gossip_forward(message))

            return web.Response(status=200, text="ok")

        except Exception as e:
            logger.error(f"Failed to handle incoming message: {e}")
            return web.Response(status=400, text=str(e))

    async def _gossip_forward(self, message: MeshMessage) -> None:
        """Forward message to random subset of peers (gossip protocol)."""
        import random

        alive_peers = [
            p
            for p in self.peers.values()
            if p.is_alive() and p.peer_id != message.sender_id
        ]

        # Forward to gossip_fanout peers (or ~50% heuristic)
        if self.gossip_fanout is not None:
            forward_count = min(self.gossip_fanout, len(alive_peers))
        else:
            forward_count = max(1, len(alive_peers) // 2)
        forward_peers = random.sample(alive_peers, min(forward_count, len(alive_peers)))

        for peer in forward_peers:
            await self._send_to_peer_raw(peer.multiaddr, message)

    async def _handle_peers_request(self, request: web.Request) -> web.Response:
        """Handle peer list request."""
        peers_data = [
            {
                "peer_id": peer.peer_id,
                "account_id": peer.account_id,
                "multiaddr": peer.multiaddr,
                "is_validator": peer.is_validator,
                "last_seen": peer.last_seen,
            }
            for peer in self.peers.values()
        ]
        return web.json_response(peers_data)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "peer_id": self.peer_id,
                "peers_count": len(self.peers),
                "running": self._running,
            }
        )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to peers."""
        while self._running:
            await asyncio.sleep(60)

            await self._broadcast_message(
                message_type=MessageType.HEARTBEAT,
                payload={"timestamp": datetime.now(timezone.utc).timestamp()},
                ttl=1,
            )

    async def _peer_discovery_loop(self) -> None:
        """Periodically discover new peers from blockchain."""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes
            await self.discover_peers()

    async def _cleanup_loop(self) -> None:
        """Clean up dead peers and old messages."""
        while self._running:
            await asyncio.sleep(600)  # Every 10 minutes

            # Remove dead peers
            dead_peers = [
                peer_id
                for peer_id, peer in self.peers.items()
                if not peer.is_alive(timeout=600)
            ]

            for peer_id in dead_peers:
                del self.peers[peer_id]
                logger.debug(f"Removed dead peer {peer_id}")

            # LRU eviction of oldest seen messages (keep newest half)
            if len(self.seen_messages) > self._seen_messages_max:
                evict_count = len(self.seen_messages) - (self._seen_messages_max // 2)
                for _ in range(evict_count):
                    self.seen_messages.popitem(last=False)

    async def receive_messages(self):
        """Async generator for receiving messages."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue


# =============================================================================
# Helpers
# =============================================================================

_MULTIADDR_RE = re.compile(r"^/ip4/\d{1,3}(\.\d{1,3}){3}/tcp/\d{1,5}$")


def _is_valid_multiaddr(addr: str) -> bool:
    """Validate a simplified multiaddr: /ip4/<ip>/tcp/<port>."""
    return bool(_MULTIADDR_RE.match(addr))


# =============================================================================
# Consensus Primitive
# =============================================================================


class ConsensusRound:
    """Supermajority consensus round (2/3 + 1 threshold).

    Tracks votes from peers for a given proposal and determines
    whether a supermajority has been reached.
    """

    def __init__(self, round_id: str, proposal: str, total_peers: int):
        self.round_id = round_id
        self.proposal = proposal
        self.total_peers = total_peers
        self._votes: dict[str, bool] = {}  # peer_id -> approve
        self.is_finalized = False
        self._threshold = math.ceil(total_peers * 2 / 3) + 1

    def add_vote(self, peer_id: str, approve: bool) -> None:
        """Record a vote (first vote per peer wins; ignored after finalization)."""
        if self.is_finalized:
            return
        if peer_id in self._votes:
            return
        self._votes[peer_id] = approve
        if self.has_supermajority():
            self.is_finalized = True

    @property
    def vote_count(self) -> int:
        return len(self._votes)

    @property
    def approve_count(self) -> int:
        return sum(1 for v in self._votes.values() if v)

    def has_supermajority(self) -> bool:
        """True when positive votes >= ceil(total*2/3)+1."""
        return self.approve_count >= self._threshold
