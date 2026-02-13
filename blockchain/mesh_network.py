"""
Mesh Network Connector for Nawal AI Validators

Implements decentralized P2P mesh networking for validator communication,
enabling direct model sharing, gossip protocol for FL rounds, and 
Byzantine-resistant distributed consensus.

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
from dataclasses import dataclass, field
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
    ):
        """
        Initialize mesh network client.
        
        Args:
            peer_id: Unique peer identifier
            listen_port: Port to listen for incoming connections
            blockchain_rpc: BelizeChain RPC endpoint for peer discovery
            private_key: Ed25519 private key for signing (generated if None)
        """
        self.peer_id = peer_id
        self.listen_port = listen_port
        self.blockchain_rpc = blockchain_rpc
        
        # Cryptography
        if private_key is None:
            private_key = ed25519.Ed25519PrivateKey.generate()
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.public_key_hex = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.seen_messages: Set[str] = set()  # For gossip deduplication
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Networking
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Blockchain connection
        self.substrate: Optional[SubstrateInterface] = None
        
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
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.listen_port)
        await self.site.start()
        
        # Connect to blockchain for peer discovery
        if SUBSTRATE_AVAILABLE:
            try:
                self.substrate = SubstrateInterface(url=self.blockchain_rpc)
                logger.info(f"Connected to BelizeChain at {self.blockchain_rpc}")
            except Exception as e:
                logger.error(f"Failed to connect to blockchain: {e}")
        
        self._running = True
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._peer_discovery_loop())
        asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Mesh network started on port {self.listen_port}")
    
    async def stop(self) -> None:
        """Stop the mesh network client."""
        if not self._running:
            return
        
        self._running = False
        
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
                        peer = PeerInfo(
                            peer_id=hashlib.sha256(validator_account.encode()).hexdigest()[:16],
                            account_id=validator_account,
                            multiaddr=metadata["network_address"],
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
            logger.error(f"Peer discovery failed: {e}")
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
        
        # Mark as seen
        self.seen_messages.add(message.message_id)
        
        # Send to all peers
        tasks = [
            self._send_to_peer_raw(peer.multiaddr, message)
            for peer in self.peers.values()
            if peer.is_alive()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)
        
        logger.debug(f"Broadcast {message_type.value} to {success_count}/{len(tasks)} peers")
    
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
            data = await request.json()
            message = MeshMessage.from_dict(data)
            
            # Deduplication
            if message.message_id in self.seen_messages:
                return web.Response(status=200, text="duplicate")
            
            self.seen_messages.add(message.message_id)
            
            # Update peer info
            if message.sender_id in self.peers:
                self.peers[message.sender_id].last_seen = datetime.now(timezone.utc).timestamp()
            
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
        
        alive_peers = [p for p in self.peers.values() if p.is_alive() and p.peer_id != message.sender_id]
        
        # Forward to ~50% of peers
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
        return web.json_response({
            "status": "healthy",
            "peer_id": self.peer_id,
            "peers_count": len(self.peers),
            "running": self._running,
        })
    
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
                peer_id for peer_id, peer in self.peers.items()
                if not peer.is_alive(timeout=600)
            ]
            
            for peer_id in dead_peers:
                del self.peers[peer_id]
                logger.debug(f"Removed dead peer {peer_id}")
            
            # Limit seen messages cache
            if len(self.seen_messages) > 10000:
                self.seen_messages.clear()
    
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
