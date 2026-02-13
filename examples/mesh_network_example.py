"""
Mesh Network Example - Validator Node with P2P Communication

Demonstrates how to set up a validator node with mesh networking
for decentralized federated learning coordination.

Author: BelizeChain AI Team
Date: February 2026
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blockchain.mesh_network import (
    MeshNetworkClient,
    MessageType,
    MeshMessage,
)
from blockchain.staking_connector import StakingConnector

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


async def main():
    """Run mesh network validator node."""
    
    logger.info("🌐 Starting Nawal AI Mesh Network Validator")
    logger.info("=" * 70)
    
    # Configuration
    PEER_ID = "validator_belizecity_01"
    LISTEN_PORT = 9090
    BLOCKCHAIN_RPC = "ws://127.0.0.1:9944"
    
    # Initialize mesh network
    mesh = MeshNetworkClient(
        peer_id=PEER_ID,
        listen_port=LISTEN_PORT,
        blockchain_rpc=BLOCKCHAIN_RPC,
    )
    
    logger.info(f"Peer ID: {PEER_ID}")
    logger.info(f"Listen Port: {LISTEN_PORT}")
    logger.info(f"Public Key: {mesh.public_key_hex[:32]}...")
    logger.info("")
    
    # Start mesh network
    logger.info("Starting mesh network server...")
    await mesh.start()
    logger.info(f"✅ Mesh network started on port {LISTEN_PORT}")
    logger.info("")
    
    # Discover peers from blockchain
    logger.info("Discovering validator peers from blockchain...")
    peers = await mesh.discover_peers()
    logger.info(f"✅ Discovered {len(peers)} validator peers")
    
    for peer in peers:
        logger.info(
            f"  - {peer.peer_id} @ {peer.multiaddr} "
            f"(stake: {peer.stake_amount / 100000000:.0f} DALLA)"
        )
    logger.info("")
    
    # Register message handlers
    logger.info("Registering message handlers...")
    
    async def handle_fl_round_start(message: MeshMessage):
        """Handle FL round start announcement."""
        payload = message.payload
        logger.info("📢 NEW FL ROUND ANNOUNCED")
        logger.info(f"  Round ID: {payload['round_id']}")
        logger.info(f"  Dataset: {payload['dataset_name']}")
        logger.info(f"  Coordinator: {payload['coordinator_id']}")
        logger.info(f"  Target Participants: {payload['target_participants']}")
        logger.info(f"  Reward Pool: {payload['reward_pool'] / 100000000:.0f} DALLA")
        logger.info(f"  Deadline: {payload['deadline']}")
        logger.info("")
        
        # TODO: Enroll in round if interested
        # await staking.enroll_validator(...)
    
    async def handle_heartbeat(message: MeshMessage):
        """Handle peer heartbeat."""
        logger.debug(f"💓 Heartbeat from {message.sender_id}")
    
    async def handle_model_delta(message: MeshMessage):
        """Handle model delta transfer."""
        payload = message.payload
        logger.info("📦 MODEL DELTA RECEIVED")
        logger.info(f"  Round ID: {payload['round_id']}")
        logger.info(f"  Model CID: {payload['model_cid']}")
        logger.info(f"  Quality Score: {payload['quality_score']:.2f}")
        logger.info("")
    
    mesh.register_handler(MessageType.FL_ROUND_START, handle_fl_round_start)
    mesh.register_handler(MessageType.HEARTBEAT, handle_heartbeat)
    mesh.register_handler(MessageType.MODEL_DELTA_TRANSFER, handle_model_delta)
    
    logger.info("✅ Message handlers registered")
    logger.info("")
    
    # Simulate announcing an FL round (if this is the coordinator)
    if len(sys.argv) > 1 and sys.argv[1] == "--coordinator":
        logger.info("🎯 COORDINATOR MODE - Announcing FL round...")
        await asyncio.sleep(5)  # Wait for peers to connect
        
        await mesh.announce_fl_round(
            round_id="round_20260213_001",
            dataset_name="belize_corpus",
            target_participants=5,
            deadline=3600,  # 1 hour
            min_stake=1000,
            reward_pool=5000000000000,  # 50,000 DALLA
            model_hash="Qm...",
        )
        
        logger.info("✅ FL round announced to mesh network")
        logger.info("")
    
    # Keep running and process messages
    logger.info("🔄 Listening for mesh network messages...")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    try:
        # Process messages from queue
        async for message in mesh.receive_messages():
            # Messages are handled by registered handlers
            pass
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("🛑 Shutting down...")
        await mesh.stop()
        logger.info("✅ Mesh network stopped")


if __name__ == "__main__":
    asyncio.run(main())
