# 🔗 Blockchain Integration

**Version**: 1.1.0  
**BelizeChain**: Substrate v1.0  
**Last Updated**: February 13, 2026

---

## Overview

Nawal AI integrates with BelizeChain (Substrate/Polkadot SDK) for identity, staking, governance, and payroll verification.

---

## BelizeChain Architecture

### Substrate Framework

```
┌─────────────────────────────────────────┐
│         BelizeChain Runtime             │
├─────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  │
│  │   Identity    │  │   Staking     │  │
│  │   Registry    │  │   System      │  │
│  └───────────────┘  └───────────────┘  │
│  ┌───────────────┐  ┌───────────────┐  │
│  │   Payroll     │  │   Community   │  │
│  │   Pallet      │  │   DAO         │  │
│  └───────────────┘  └───────────────┘  │
│  ┌───────────────┐  ┌───────────────┐  │
│  │   Genome      │  │   Rewards     │  │
│  │   Registry    │  │   System      │  │
│  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │   Substrate    │
        │   Node         │
        └────────────────┘
```

---

## Pallet Integration

### 1. Identity Pallet

**User identity verification and registration**

```python
from blockchain import IdentityVerifier

# Verify identity
verifier = IdentityVerifier(blockchain_rpc="ws://localhost:9944")
result = await verifier.verify_identity(account_id="5GrwvaEF...")

print(f"Verified: {result['is_verified']}")
print(f"Name: {result['display_name']}")
print(f"Reputation: {result['reputation_score']}")
```

**Identity Fields**:
- `display_name`: Human-readable name
- `legal_name`: Government ID name
- `nationality`: Country code
- `verification_level`: None, Basic, Full
- `reputation_score`: 0-100

**Storage**:
```rust
// Identity pallet storage
pub struct IdentityInfo {
    pub display_name: Vec<u8>,
    pub legal_name: Vec<u8>,
    pub nationality: Vec<u8>,
    pub verification_level: VerificationLevel,
    pub reputation_score: u32,
}

#[pallet::storage]
pub type Identities<T> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    IdentityInfo,
>;
```

---

### 2. Staking Pallet

**Validator staking and rewards**

```python
from blockchain import StakingConnector

staking = StakingConnector(
    blockchain_rpc="ws://localhost:9944",
    account_seed="//Alice"
)

# Stake DALLA to become validator
tx_hash = await staking.stake(
    amount=1000000000000,  # 10,000 DALLA (12 decimals)
    duration_blocks=28800,  # ~1 month (3s blocks)
)

# Check staking status
status = await staking.get_staking_status(account_id="5GrwvaEF...")
print(f"Staked: {status['staked_amount'] / 1e12} DALLA")
print(f"Is validator: {status['is_validator']}")
```

**Staking Requirements**:
- Minimum stake: 1,000 DALLA
- Lock period: 7 days (201,600 blocks)
- Slashing: Up to 10% for malicious behavior

**Rewards**:
- Block rewards: 5 DALLA per block
- PoUW rewards: Variable based on FL contribution
- Transaction fees: 20% to validators

---

### 3. Payroll Pallet

**ZK-proof payroll submissions**

```python
from blockchain import PayrollConnector

payroll = PayrollConnector(
    blockchain_rpc="ws://localhost:9944",
    account_seed="//Government"
)

# Submit payroll
submission = await payroll.submit_payroll(
    employees=employee_list,
    pay_period_id="2026-02",
    employer_name="Ministry of Health",
)

print(f"Submission ID: {submission['id']}")
print(f"Merkle root: {submission['merkle_root']}")
```

**Extrinsics**:
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    /// Submit payroll with ZK-proof
    #[pallet::weight(10_000)]
    pub fn submit_payroll(
        origin: OriginFor<T>,
        merkle_root: H256,
        total_employees: u32,
        proof: Vec<u8>,
    ) -> DispatchResult {
        // Verify proof
        ensure!(verify_zk_proof(&proof, &merkle_root), Error::<T>::InvalidProof);
        
        // Store commitment
        PayrollSubmissions::<T>::insert(submission_id, PayrollInfo {
            merkle_root,
            total_employees,
            employer: origin,
            timestamp: now(),
        });
        
        Ok(())
    }
    
    /// Verify payroll (for validators)
    #[pallet::weight(5_000)]
    pub fn verify_payroll(
        origin: OriginFor<T>,
        submission_id: H256,
        is_valid: bool,
    ) -> DispatchResult {
        // Ensure validator
        ensure!(is_validator(&origin), Error::<T>::NotValidator);
        
        // Record verification
        Verifications::<T>::insert((submission_id, origin), is_valid);
        
        // Distribute reward
        if is_valid {
            T::Currency::deposit_creating(&origin, VERIFICATION_REWARD);
        }
        
        Ok(())
    }
}
```

**Storage**:
```rust
#[pallet::storage]
pub type PayrollSubmissions<T> = StorageMap<
    _,
    Blake2_128Concat,
    H256,  // submission_id
    PayrollInfo<T>,
>;

#[pallet::storage]
pub type Verifications<T> = StorageDoubleMap<
    _,
    Blake2_128Concat, H256,  // submission_id
    Blake2_128Concat, T::AccountId,  // validator
    bool,  // is_valid
>;
```

---

### 4. Community DAO

**Decentralized governance**

```python
from blockchain import CommunityConnector

community = CommunityConnector(blockchain_rpc="ws://localhost:9944")

# Submit proposal
proposal = await community.submit_proposal(
    title="Increase FL Round Rewards",
    description="Increase PoUW rewards from 50 to 75 DALLA per round",
    action="set_reward_amount",
    params={"amount": 75_000_000_000_000},
)

# Vote on proposal
await community.vote(
    proposal_id=proposal["id"],
    vote="Aye",
    conviction=5,  # 5x voting power, 32-week lock
)
```

**Governance Parameters**:
- Proposal deposit: 100 DALLA
- Voting period: 7 days
- Execution delay: 2 days
- Quorum: 10% of total stake

---

### 5. Genome Registry

**AI model registration**

```python
from blockchain import GenomeRegistry

registry = GenomeRegistry(blockchain_rpc="ws://localhost:9944")

# Register model
model_id = await registry.register_model(
    name="Nawal-Medium-v1.1.0",
    description="Sovereign transformer, 350M parameters",
    cid="QmX5ZWqMq...",  # Pakit CID
    architecture="transformer",
    parameters=350_000_000,
    accuracy=0.87,
    license="MIT",
)

print(f"Model registered: {model_id}")
```

**Model Metadata**:
```rust
pub struct ModelInfo {
    pub name: Vec<u8>,
    pub description: Vec<u8>,
    pub cid: Vec<u8>,  // Pakit CID
    pub architecture: Vec<u8>,
    pub parameters: u64,
    pub accuracy: u32,  // FixedU32
    pub license: Vec<u8>,
    pub owner: AccountId,
    pub created_at: BlockNumber,
}
```

---

## Substrate Client

### Connection Management

```python
from blockchain import SubstrateClient
from substrateinterface import SubstrateInterface

# Initialize client
client = SubstrateInterface(
    url="ws://localhost:9944",
    type_registry_preset="substrate-node-template",
)

# Check connection
chain = client.query("System", "Chain")
print(f"Connected to: {chain.value}")

# Get block number
block_number = client.get_block_number(None)
print(f"Latest block: {block_number}")
```

### Query Storage

```python
# Query single value
validators = client.query("Staking", "Validators")
print(f"Validators: {validators.value}")

# Query map
identity = client.query(
    module="Identity",
    storage_function="IdentityOf",
    params=["5GrwvaEF..."]
)
print(f"Identity: {identity.value}")

# Query all entries
all_identities = client.query_map(
    module="Identity",
    storage_function="IdentityOf"
)
for account, identity in all_identities:
    print(f"{account}: {identity['display_name']}")
```

### Submit Extrinsics

```python
from substrateinterface import Keypair

# Create keypair
keypair = Keypair.create_from_uri("//Alice")

# Compose call
call = client.compose_call(
    call_module="Payroll",
    call_function="submit_payroll",
    call_params={
        "merkle_root": "0xabc123...",
        "total_employees": 150,
        "proof": proof_bytes,
    }
)

# Create signed extrinsic
extrinsic = client.create_signed_extrinsic(
    call=call,
    keypair=keypair
)

# Submit
receipt = client.submit_extrinsic(
    extrinsic,
    wait_for_inclusion=True
)

print(f"Extrinsic hash: {receipt.extrinsic_hash}")
print(f"Block hash: {receipt.block_hash}")
print(f"Finalized: {receipt.is_success}")
```

### Event Monitoring

```python
# Subscribe to events
def event_handler(obj, update_nr, subscription_id):
    print(f"Event #{update_nr}")
    
    for event in obj['params']['result']['events']:
        if event['event_id'] == 'PayrollSubmitted':
            print(f"New payroll: {event['attributes']}")

# Subscribe
result = client.subscribe_block_headers(event_handler)
```

---

## Token Economics

### DALLA Token

**Native token of BelizeChain**

- **Symbol**: DALLA
- **Decimals**: 12
- **Total Supply**: 1,000,000,000 DALLA (1 billion)
- **Block Time**: ~3 seconds
- **Block Reward**: 5 DALLA

### Token Distribution

| Allocation | Amount | Percentage |
|------------|--------|------------|
| Community Treasury | 500M DALLA | 50% |
| Validator Rewards | 300M DALLA | 30% |
| Development Fund | 150M DALLA | 15% |
| Initial Distribution | 50M DALLA | 5% |

### Transaction Fees

```python
# Calculate fee
fee = client.get_payment_info(call=call, keypair=keypair)
print(f"Transaction fee: {fee['partialFee'] / 1e12} DALLA")

# Fee breakdown:
# - Base fee: 0.001 DALLA
# - Length fee: 0.0001 DALLA per byte
# - Weight fee: Variable based on computation
```

---

## RPC Endpoints

### Production Endpoints

```bash
# Mainnet (coming soon)
wss://rpc.belizechain.org:9944

# Testnet
wss://testnet-rpc.belizechain.org:9944

# Local development
ws://localhost:9944
```

### RPC Methods

```python
# System methods
client.rpc_request("system_chain", [])
client.rpc_request("system_health", [])
client.rpc_request("system_version", [])

# Chain methods
client.rpc_request("chain_getBlock", [])
client.rpc_request("chain_getBlockHash", [block_number])
client.rpc_request("chain_getFinalizedHead", [])

# State methods
client.rpc_request("state_getStorage", [storage_key])
client.rpc_request("state_getMetadata", [])

# Author methods (requires --rpc-methods=Unsafe)
client.rpc_request("author_submitExtrinsic", [extrinsic])
```

---

## Error Handling

### Common Errors

```python
from substrateinterface.exceptions import SubstrateRequestException

try:
    result = await client.submit_extrinsic(extrinsic)
except SubstrateRequestException as e:
    if "insufficient balance" in str(e):
        print("❌ Not enough DALLA for transaction")
    elif "invalid signature" in str(e):
        print("❌ Invalid keypair signature")
    elif "already exists" in str(e):
        print("❌ Duplicate submission")
    else:
        print(f"❌ Error: {e}")
```

### Retry Logic

```python
import asyncio

async def submit_with_retry(client, extrinsic, max_retries=3):
    """Submit extrinsic with retry logic."""
    for attempt in range(max_retries):
        try:
            return await client.submit_extrinsic(extrinsic)
        except SubstrateRequestException as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise
```

---

## Testing

### Local Development Node

```bash
# Start local Substrate node
substrate-node-template \
  --dev \
  --tmp \
  --rpc-external \
  --rpc-cors=all \
  --ws-external

# Or with Docker
docker run -p 9944:9944 parity/substrate:latest \
  --dev \
  --rpc-external \
  --ws-external
```

### Test Accounts

```python
# Well-known development accounts
ALICE = "//Alice"  # 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY
BOB = "//Bob"      # 5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty
CHARLIE = "//Charlie"  # 5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y

# Create test keypair
from substrateinterface import Keypair
keypair = Keypair.create_from_uri(ALICE)
```

---

**Next Steps**:
- [Architecture Overview](overview.md)
- [Federated Learning Architecture](federated-learning.md)
- [API Reference](../reference/api-reference.md)
