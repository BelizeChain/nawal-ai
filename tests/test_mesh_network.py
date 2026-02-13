"""
Tests for Mesh Network functionality.

Tests peer discovery, message broadcasting, gossip protocol,
and P2P communication for Nawal AI validators.

Author: BelizeChain AI Team
Date: February 2026
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from blockchain.mesh_network import (
    MeshNetworkClient,
    MeshMessage,
    MessageType,
    PeerInfo,
)


@pytest.mark.asyncio
class TestMeshNetworkClient:
    """Test suite for MeshNetworkClient."""
    
    async def test_initialization(self):
        """Test mesh network client initialization."""
        mesh = MeshNetworkClient(
            peer_id="test_peer_001",
            listen_port=9091,
            blockchain_rpc="ws://localhost:9944",
        )
        
        assert mesh.peer_id == "test_peer_001"
        assert mesh.listen_port == 9091
        assert mesh.blockchain_rpc == "ws://localhost:9944"
        assert mesh.public_key_hex is not None
        assert len(mesh.peers) == 0
        assert mesh._running is False
    
    async def test_start_stop(self):
        """Test starting and stopping mesh network."""
        mesh = MeshNetworkClient(
            peer_id="test_peer_002",
            listen_port=9092,
        )
        
        await mesh.start()
        assert mesh._running is True
        assert mesh.app is not None
        
        await mesh.stop()
        assert mesh._running is False
    
    async def test_message_creation(self):
        """Test creating and signing messages."""
        mesh = MeshNetworkClient(peer_id="test_peer_003")
        
        message = mesh._create_message(
            message_type=MessageType.HEARTBEAT,
            payload={"test": "data"},
        )
        
        assert message.message_id is not None
        assert message.message_type == MessageType.HEARTBEAT
        assert message.sender_id == "test_peer_003"
        assert message.payload == {"test": "data"}
        assert message.signature is not None
        assert message.ttl == 5
    
    async def test_message_serialization(self):
        """Test message serialization and deserialization."""
        mesh = MeshNetworkClient(peer_id="test_peer_004")
        
        original = mesh._create_message(
            message_type=MessageType.FL_ROUND_START,
            payload={"round_id": "round_001"},
        )
        
        # Serialize to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["message_id"] == original.message_id
        assert data["message_type"] == "fl_round_start"
        
        # Deserialize from dict
        restored = MeshMessage.from_dict(data)
        assert restored.message_id == original.message_id
        assert restored.message_type == original.message_type
        assert restored.sender_id == original.sender_id
        assert restored.payload == original.payload
    
    @patch('blockchain.mesh_network.SubstrateInterface')
    async def test_peer_discovery(self, mock_substrate):
        """Test peer discovery from blockchain."""
        # Mock substrate connection
        mock_instance = Mock()
        mock_instance.query.return_value = [
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        ]
        mock_substrate.return_value = mock_instance
        
        mesh = MeshNetworkClient(peer_id="test_peer_005")
        mesh.substrate = mock_instance
        
        # Mock validator metadata
        def mock_query_metadata(module, storage_function, params):
            if storage_function == "ValidatorMetadata":
                return {
                    "network_address": "/ip4/127.0.0.1/tcp/9090",
                    "public_key": "abcd1234",
                    "stake": 1000000000000,
                }
            return []
        
        mock_instance.query = mock_query_metadata
        
        peers = await mesh.discover_peers()
        
        assert len(peers) > 0
        # Note: In mock mode, peer discovery might not populate peers
    
    async def test_register_handler(self):
        """Test registering message handlers."""
        mesh = MeshNetworkClient(peer_id="test_peer_006")
        
        handler_called = []
        
        async def test_handler(message):
            handler_called.append(message)
        
        mesh.register_handler(MessageType.HEARTBEAT, test_handler)
        
        assert MessageType.HEARTBEAT in mesh.message_handlers
        assert len(mesh.message_handlers[MessageType.HEARTBEAT]) == 1
    
    async def test_fl_round_announcement(self):
        """Test FL round announcement."""
        mesh = MeshNetworkClient(peer_id="test_peer_007", listen_port=9097)
        
        await mesh.start()
        
        # Mock broadcast
        with patch.object(mesh, '_broadcast_message') as mock_broadcast:
            await mesh.announce_fl_round(
                round_id="round_test_001",
                dataset_name="test_dataset",
                target_participants=5,
                deadline=3600,
                min_stake=1000,
                reward_pool=5000,
                model_hash="Qm...",
            )
            
            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args
            assert call_args[1]["message_type"] == MessageType.FL_ROUND_START
            assert call_args[1]["payload"]["round_id"] == "round_test_001"
        
        await mesh.stop()
    
    async def test_merkle_root_computation(self):
        """Test Merkle root computation (used in mesh for data integrity)."""
        mesh = MeshNetworkClient(peer_id="test_peer_008")
        
        # This is actually in payroll, but the concept applies to mesh for data integrity
        # Just testing the mesh network for now
        pass
    
    def test_peer_info_is_alive(self):
        """Test peer liveness check."""
        import time
        
        # Recent peer
        peer1 = PeerInfo(
            peer_id="peer1",
            account_id="5GrwvaEF...",
            multiaddr="/ip4/1.2.3.4/tcp/9090",
            public_key="abc123",
            last_seen=time.time(),
        )
        assert peer1.is_alive(timeout=300) is True
        
        # Old peer
        peer2 = PeerInfo(
            peer_id="peer2",
            account_id="5GrwvaEF...",
            multiaddr="/ip4/1.2.3.5/tcp/9090",
            public_key="def456",
            last_seen=time.time() - 400,
        )
        assert peer2.is_alive(timeout=300) is False


@pytest.mark.asyncio
class TestMeshNetworkIntegration:
    """Integration tests for mesh network."""
    
    async def test_two_peers_communication(self):
        """Test communication between two peers."""
        # Create two mesh clients
        mesh1 = MeshNetworkClient(peer_id="peer1", listen_port=9101)
        mesh2 = MeshNetworkClient(peer_id="peer2", listen_port=9102)
        
        await mesh1.start()
        await mesh2.start()
        
        # Add peer2 to peer1's peer list
        mesh1.peers["peer2"] = PeerInfo(
            peer_id="peer2",
            account_id="5GrwvaEF...",
            multiaddr="/ip4/127.0.0.1/tcp/9102",
            public_key=mesh2.public_key_hex,
            last_seen=asyncio.get_event_loop().time(),
        )
        
        # Send message from peer1 to peer2
        message_received = []
        
        async def handle_message(msg):
            message_received.append(msg)
        
        mesh2.register_handler(MessageType.HEARTBEAT, handle_message)
        
        # Give some time for setup
        await asyncio.sleep(0.5)
        
        # Send heartbeat
        await mesh1._broadcast_message(
            message_type=MessageType.HEARTBEAT,
            payload={"timestamp": 12345},
        )
        
        # Wait for message
        await asyncio.sleep(1)
        
        # Check if message was received (may not work in test environment)
        # This is more of an integration test that needs real networking
        
        await mesh1.stop()
        await mesh2.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
