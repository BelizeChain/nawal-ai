# 🌐 Mesh Networking Guide

**Version**: 1.1.0  
**Component**: `blockchain/mesh_network.py`  
**Status**: Production Ready

---

## Overview

Mesh networking enables Nawal AI validators to communicate peer-to-peer without relying on a central server. This decentralized approach provides:

- **Resilience**: No single point of failure
- **Scalability**: Direct peer communication reduces bottlenecks
- **Byzantine Resistance**: Reputation-based peer filtering
- **Privacy**: End-to-end encrypted communication

---

## Architecture

### Components

```
┌─────────────────┐         ┌─────────────────┐
│   Validator A   │◄───────►│   Validator B   │
│  (Mesh Client)  │         │  (Mesh Client)  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │      Gossip Protocol      │
         │   ◄─────────────────────► │
         │                           │
         ▼                           ▼
  ┌──────────────┐          ┌──────────────┐
  │ BelizeChain  │          │ BelizeChain  │
  │  (Registry)  │          │  (Registry)  │
  └──────────────┘          └──────────────┘
```

### Key Features

1. **Peer Discovery**: Automatic discovery via blockchain validator registry
2. **Message Types**: FL rounds, model deltas, heartbeats, gossip
3. **Cryptographic Security**: Ed25519 signing for all messages
4. **TTL-based Propagation**: Time-to-live prevents infinite message loops
5. **Deduplication**: Message IDs prevent replay attacks

---

## Quick Start

### Installation

```bash
# Already included in Nawal AI
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from blockchain import MeshNetworkClient, MessageType

async def main():
    # Initialize mesh client
    mesh = MeshNetworkClient(
        peer_id="validator_belizecity_01",
        listen_port=9090,
        blockchain_rpc="ws://localhost:9944",
    )
    
    # Start mesh network
    await mesh.start()
    print("✅ Mesh network started")
    
    # Discover peers
    peers = await mesh.discover_peers()
    print(f"Found {len(peers)} validators")
    
    # Register message handler
    async def handle_fl_round(message):
        print(f"New FL round: {message.payload['round_id']}")
    
    mesh.register_handler(MessageType.FL_ROUND_START, handle_fl_round)
    
    # Announce FL round (if coordinator)
    await mesh.announce_fl_round(
        round_id="round_001",
        dataset_name="belize_corpus",
        target_participants=10,
        deadline=3600,
        reward_pool=50000,
    )
    
    # Keep running
    await asyncio.sleep(float('inf'))

asyncio.run(main())
```

---

## Configuration

### Environment Variables

```bash
# Mesh Network Configuration
MESH_LISTEN_PORT=9090                          # Port to listen on
MESH_PEER_ID="validator_belizecity_01"         # Unique peer identifier
BLOCKCHAIN_RPC="ws://localhost:9944"           # BelizeChain RPC endpoint

# Optional
MESH_HEARTBEAT_INTERVAL=60                     # Heartbeat interval (seconds)
MESH_PEER_TIMEOUT=300                          # Peer timeout (seconds)
MESH_MAX_PEERS=100                             # Maximum peers
MESH_GOSSIP_FANOUT=5                           # Gossip propagation factor
```

### Python Configuration

```python
from blockchain import MeshNetworkClient

mesh = MeshNetworkClient(
    peer_id="validator_001",               # Required: Unique ID
    listen_port=9090,                      # Required: Listen port
    blockchain_rpc="ws://localhost:9944",  # Required: Blockchain RPC
    private_key=None,                      # Optional: Ed25519 key (auto-generated)
)
```

---

## Message Types

### 1. FL Round Announcement

**Purpose**: Broadcast new federated learning round to all validators

```python
await mesh.announce_fl_round(
    round_id="round_20260213_001",
    dataset_name="belize_corpus",
    target_participants=10,
    deadline=3600,              # Seconds from now
    min_stake=1000,             # Minimum stake in Mahogany
    reward_pool=50000,          # Total rewards in Mahogany
    model_hash="Qm...",         # IPFS hash of initial model
)
```

**Received by**: All connected validators via gossip protocol

### 2. Model Delta Transfer

**Purpose**: Send model updates directly to specific peer

```python
success = await mesh.send_model_delta(
    recipient_id="validator_002",
    round_id="round_001",
    model_cid="QmX...",         # IPFS CID of model delta
    quality_score=95.5,         # Model quality (0-100)
)
```

**Received by**: Specified peer only (direct P2P)

### 3. Heartbeat

**Purpose**: Keep-alive signal for peer liveness

```python
# Automatically sent every 60 seconds
# No manual invocation needed
```

**Received by**: All connected peers

### 4. Gossip Messages

**Purpose**: Generic gossip protocol messages

```python
# Automatically handled by mesh network
# Custom messages can use the gossip protocol:
await mesh._broadcast_message(
    message_type=MessageType.GOSSIP,
    payload={"custom": "data"},
    ttl=5,
)
```

**Received by**: All peers via gossip propagation

---

## Peer Discovery

### Automatic Discovery

Mesh network automatically discovers peers from the  blockchain validator registry:

```python
# Discover peers from blockchain
peers = await mesh.discover_peers()

for peer in peers:
    print(f"Peer: {peer.peer_id}")
    print(f"  Address: {peer.multiaddr}")
    print(f"  Stake: {peer.stake_amount / 100000000} DALLA")
    print(f"  Validator: {peer.is_validator}")
```

### Manual Peer Addition

```python
from blockchain.mesh_network import PeerInfo
import time

# Add peer manually
mesh.peers["peer_002"] = PeerInfo(
    peer_id="peer_002",
    account_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    multiaddr="/ip4/192.168.1.100/tcp/9090",
    public_key="abc123...",
    stake_amount=5000000000000,
    last_seen=time.time(),
    is_validator=True,
)
```

---

## Security

### Message Signing

All messages are signed with Ed25519:

```python
# Automatic signing when creating messages
message = mesh._create_message(
    message_type=MessageType.FL_ROUND_START,
    payload={"round_id": "001"},
)

# Signature is automatically included
print(f"Signature: {message.signature[:32]}...")
```

### Signature Verification

```python
# Automatic verification on message receipt
# Invalid signatures are rejected
```

### Byzantine Resistance

Peers with low reputation are filtered:

```python
# Check peer reputation
if peer.reputation < 50:
    print("⚠️ Low reputation peer detected")
    # Messages from this peer may be ignored
```

---

## Monitoring

### Peer Status

```python
# Get all peers
for peer_id, peer in mesh.peers.items():
    print(f"{peer_id}: {'🟢 Alive' if peer.is_alive() else '🔴 Dead'}")
```

### Message Statistics

```python
# Check seen messages
print(f"Messages processed: {len(mesh.seen_messages)}")

# Monitor peer count
print(f"Active peers: {len([p for p in mesh.peers.values() if p.is_alive()])}")
```

### Health Check Endpoint

```bash
# HTTP health check (when mesh is running)
curl http://localhost:9090/health

# Response:
# {
#   "status": "healthy",
#   "peer_id": "validator_001",
#   "peers_count": 5,
#   "running": true
# }
```

---

## Advanced Usage

### Custom Message Handlers

```python
async def handle_custom_message(message: MeshMessage):
    """Handle custom message type."""
    payload = message.payload
    
    # Process message
    if payload.get("action") == "request_model":
        model_cid = await upload_model()
        await mesh.send_model_delta(
            recipient_id=message.sender_id,
            round_id=payload["round_id"],
            model_cid=model_cid,
            quality_score=98.5,
        )

# Register handler
mesh.register_handler(MessageType.MODEL_DELTA_REQUEST, handle_custom_message)
```

### Gossip Forwarding

Messages are automatically forwarded to ~50% of peers:

```python
# Automatic gossip forwarding
# TTL decreases with each hop
# Message stops propagating when TTL reaches 0
```

### Peer Cleanup

Dead peers are automatically removed:

```python
# Automatic cleanup every 10 minutes
# Removes peers with no heartbeat for 10+ minutes
```

---

## Deployment

### Docker Compose

```yaml
services:
  validator:
    image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
    ports:
      - "8080:8080"   # API server
      - "9090:9090"   # Mesh network
    environment:
      - MESH_LISTEN_PORT=9090
      - MESH_PEER_ID=validator_001
      - BLOCKCHAIN_RPC=ws://blockchain:9944
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nawal-mesh
spec:
  type: LoadBalancer
  ports:
    - port: 9090
      targetPort: 9090
      name: mesh
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nawal-validator
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: nawal
          image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
          ports:
            - containerPort: 9090
              name: mesh
          env:
            - name: MESH_PEER_ID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: MESH_LISTEN_PORT
              value: "9090"
```

---

## Troubleshooting

### Peers Not Discovered

```bash
# Check blockchain connection
curl -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
     http://localhost:9944

# Check validator registry
python3 -c "
from blockchain import MeshNetworkClient
import asyncio

async def check():
    mesh = MeshNetworkClient('test', 9090)
    await mesh.start()
    peers = await mesh.discover_peers()
    print(f'Found {len(peers)} peers')
    await mesh.stop()

asyncio.run(check())
"
```

### Port Already in Use

```bash
# Find process using port 9090
lsof -i :9090

# Change port
export MESH_LISTEN_PORT=9091
```

### Messages Not Received

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check message queue
print(f"Queue size: {mesh._message_queue.qsize()}")

# Verify handler registration
print(f"Handlers: {mesh.message_handlers}")
```

---

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Message latency | < 100ms (local network) |
| Gossip propagation | ~500ms for 10 peers |
| Memory per peer | ~1KB |
| CPU usage | < 5% (idle), < 15% (active) |

### Optimization Tips

1. **Limit peer count**: Keep under 50 peers for best performance
2. **Adjust gossip fanout**: Lower fanout = less bandwidth, slower propagation
3. **Increase TTL**: For large networks (> 20 peers)
4. **Use local blockchain node**: Reduce peer discovery latency

---

## Example: Complete Validator Node

See [examples/mesh_network_example.py](../../examples/mesh_network_example.py) for a complete working example.

---

**Next Steps**:
- [Payroll Integration Guide](payroll-integration.md)
- [API Reference](../reference/api-reference.md)
- [Deployment Guide](deployment.md)
