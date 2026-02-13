# 🚀 Quick Start: Mesh Networking & ZK-Proof Payroll

**Nawal AI v1.1.0** - New BelizeChain Integration Features

---

## Overview

This guide walks you through the new features added to align Nawal AI with BelizeChain updates:

1. **Mesh Networking** - P2P communication for validators
2. **ZK-Proof Payroll** - Privacy-preserving payroll system

---

## Prerequisites

```bash
# 1. Clone Nawal AI
git clone https://github.com/BelizeChain/nawal-ai.git
cd nawal-ai

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure BelizeChain node is running
# (Required for blockchain features, optional for mock mode)
```

---

## Feature 1: Mesh Networking

### What is it?

Mesh networking enables validators to communicate peer-to-peer without a central server. This improves:
- **Decentralization**: No single point of failure
- **Scalability**: Direct peer communication reduces bottlenecks
- **Resilience**: Byzantine-resistant gossip protocol

### Quick Start

```python
# mesh_demo.py
import asyncio
from blockchain.mesh_network import MeshNetworkClient, MessageType

async def main():
    # Initialize mesh network
    mesh = MeshNetworkClient(
        peer_id="validator_001",
        listen_port=9090,
        blockchain_rpc="ws://localhost:9944",
    )
    
    # Start mesh network
    await mesh.start()
    print(f"✅ Mesh network started on port 9090")
    
    # Discover peers from blockchain
    peers = await mesh.discover_peers()
    print(f"✅ Discovered {len(peers)} validator peers")
    
    # Register message handler
    async def handle_fl_round(message):
        print(f"📢 New FL round: {message.payload['round_id']}")
    
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

### Run the Example

```bash
# Terminal 1 - Start first validator (coordinator)
python3 examples/mesh_network_example.py --coordinator

# Terminal 2 - Start second validator
python3 examples/mesh_network_example.py

# Terminal 3 - Start third validator
python3 examples/mesh_network_example.py
```

### Key Features

- ✅ Automatic peer discovery via blockchain
- ✅ Ed25519 cryptographic signing
- ✅ Gossip protocol for message propagation
- ✅ FL round announcements
- ✅ Direct model delta exchange
- ✅ Byzantine resistance via reputation

---

## Feature 2: ZK-Proof Payroll

### What is it?

Zero-knowledge proof payroll enables privacy-preserving payroll submissions where:
- **Employers** submit encrypted payroll data
- **Validators** verify correctness without seeing salaries
- **Employees** access their paystubs securely
- **Government** tracks aggregated tax revenue

### Quick Start

```python
# payroll_demo.py
import asyncio
import hashlib
from blockchain.payroll_connector import (
    PayrollConnector,
    PayrollEntry,
    EmployeeType,
)

async def main():
    # Initialize connector
    payroll = PayrollConnector(
        websocket_url="ws://localhost:9944",
        mock_mode=True,  # Use True for testing
    )
    
    await payroll.connect()
    print("✅ Connected to Payroll pallet")
    
    # Create payroll entry
    entry = PayrollEntry(
        employee_id="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        employee_name_hash=hashlib.sha256(b"Maria Garcia").hexdigest(),
        gross_salary=500000000000,  # 5,000 DALLA
        tax_withholding=payroll.calculate_tax_withholding(500000000000),
        social_security=40000000000,  # 8%
        pension_contribution=25000000000,  # 5%
        net_salary=335000000000,
        payment_period="2026-02",
        employee_type=EmployeeType.GOVERNMENT,
        department="Ministry of Health",
    )
    
    # Submit payroll with ZK-proof
    submission = await payroll.submit_payroll(
        entries=[entry],
        payment_period="2026-02",
        employer_name="Ministry of Health",
    )
    
    print(f"✅ Payroll submitted!")
    print(f"   Submission ID: {submission.submission_id}")
    print(f"   ZK Proof: {submission.zk_proof[:32]}...")
    print(f"   Merkle Root: {submission.merkle_root}")
    print(f"   Status: {submission.status.value}")
    
    # Get employee paystub
    paystub = await payroll.get_employee_paystub(
        employee_id=entry.employee_id,
        payment_period="2026-02",
    )
    
    if paystub:
        print(f"\n📄 Paystub for {paystub.payment_period}")
        print(f"   Gross: {paystub.gross_salary / 100000000:.2f} DALLA")
        print(f"   Net: {paystub.net_salary / 100000000:.2f} DALLA")

asyncio.run(main())
```

### Run the Example

```bash
# Run payroll submission example
python3 examples/payroll_example.py
```

### Key Features

- ✅ Zero-knowledge proofs hide individual salaries
- ✅ Merkle tree commitments for data integrity
- ✅ Automatic Belize tax calculations
- ✅ Encrypted employee paystubs
- ✅ Aggregated statistics for government
- ✅ Validator verification rewards

---

## Testing

### Run Unit Tests

```bash
# Test mesh networking
pytest tests/test_mesh_network.py -v

# Test payroll integration
pytest tests/test_payroll.py -v

# Run all tests
pytest tests/ -v
```

### Manual Testing

```bash
# 1. Start local BelizeChain node (in separate terminal)
cd ~/belizechain
./target/release/belizechain --dev --tmp

# 2. Test mesh network
python3 examples/mesh_network_example.py

# 3. Test payroll
python3 examples/payroll_example.py
```

---

## Integration with Existing Code

### Add Mesh Network to API Server

```python
# api_server.py
from nawal.blockchain import MeshNetworkClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with mesh network."""
    
    # Start mesh network
    mesh = MeshNetworkClient(
        peer_id=app_state.validator_id,
        listen_port=9090,
        blockchain_rpc=config.blockchain_rpc,
    )
    await mesh.start()
    app_state.mesh_network = mesh
    
    yield
    
    # Cleanup
    await mesh.stop()

app = FastAPI(lifespan=lifespan)
```

### Add Payroll to Validator

```python
# validator.py
from nawal.blockchain import PayrollConnector

async def verify_payroll_submissions():
    """Validate payroll submissions and earn rewards."""
    
    payroll = PayrollConnector(
        websocket_url="ws://localhost:9944",
        keypair=validator_keypair,
    )
    
    await payroll.connect()
    
    # Verify pending submissions
    verified = await payroll.verify_payroll(
        submission_id="sub_12345",
    )
    
    if verified:
        print("✅ Payroll verified - PoUW reward earned!")
```

---

## Configuration

### Environment Variables

```bash
# Mesh Network
export MESH_LISTEN_PORT=9090
export MESH_PEER_ID="validator_belizecity_01"

# Payroll
export PAYROLL_MOCK_MODE=false  # Set to true for testing

# Blockchain
export BLOCKCHAIN_RPC="ws://localhost:9944"
export VALIDATOR_KEYPAIR_URI="//Alice"  # For development only
```

### Docker Deployment

```yaml
# docker-compose.yml
services:
  nawal-validator:
    image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
    ports:
      - "8080:8080"  # API server
      - "9090:9090"  # Mesh network
    environment:
      - MESH_LISTEN_PORT=9090
      - BLOCKCHAIN_RPC=wss://rpc.belizechain.io
      - VALIDATOR_KEYPAIR_URI=${VALIDATOR_KEYPAIR}
```

---

## Troubleshooting

### Mesh Network Issues

**Problem**: Peers not discovered
```bash
# Check BelizeChain connection
curl -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
     http://localhost:9944

# Check mesh network health
curl http://localhost:9090/health
curl http://localhost:9090/peers
```

**Problem**: Port already in use
```bash
# Change listen port
export MESH_LISTEN_PORT=9091
```

### Payroll Issues

**Problem**: Tax calculation incorrect
```python
# Verify tax brackets
connector = PayrollConnector()
tax = connector.calculate_tax_withholding(gross_salary)
print(f"Tax for {gross_salary / 100000000:.2f} DALLA: {tax / 100000000:.2f}")
```

**Problem**: ZK-proof generation fails
```bash
# Check if substrate-interface is installed
pip install substrate-interface>=1.7.0

# Enable mock mode for testing
payroll = PayrollConnector(mock_mode=True)
```

---

## Next Steps

1. **Deploy to Testnet**
   ```bash
   docker build -t nawal-ai:1.1.0 .
   docker push belizechainregistry.azurecr.io/nawal-ai:1.1.0
   ```

2. **Monitor Metrics**
   - Mesh peer count: `/metrics/mesh/peers`
   - Payroll submissions: `/metrics/payroll/submissions`
   - FL rounds announced: `/metrics/mesh/rounds`

3. **Read Documentation**
   - [Mesh Networking Guide](docs/MESH_NETWORKING.md) (TODO)
   - [Payroll Integration](docs/PAYROLL_INTEGRATION.md) (TODO)
   - [BelizeChain Alignment Audit](BELIZECHAIN_ALIGNMENT_AUDIT_2026-02-13.md)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/BelizeChain/nawal-ai/issues)
- **Docs**: [Nawal AI Wiki](https://github.com/BelizeChain/nawal-ai/wiki)
- **BelizeChain**: [belizechain.org](https://belizechain.org)

---

**Version**: 1.1.0  
**Updated**: February 13, 2026  
**Status**: ✅ Production Ready
