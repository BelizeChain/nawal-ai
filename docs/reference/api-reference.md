# 📚 API Reference

**Version**: 1.1.0  
**Last Updated**: February 13, 2026

---

## Mesh Network API

### `MeshNetworkClient`

Decentralized P2P communication for validators.

```python
from blockchain import MeshNetworkClient
```

#### Constructor

```python
MeshNetworkClient(
    peer_id: str,
    listen_port: int = 9090,
    blockchain_rpc: str = "ws://localhost:9944",
    private_key: Optional[bytes] = None,
)
```

**Parameters:**
- `peer_id` (str): Unique identifier for this peer
- `listen_port` (int): Port to listen on for incoming connections
- `blockchain_rpc` (str): BelizeChain RPC endpoint
- `private_key` (bytes, optional): Ed25519 private key (auto-generated if None)

**Returns:** `MeshNetworkClient` instance

#### Methods

##### `start()`

Start the mesh network client.

```python
await mesh.start()
```

**Returns:** `None`

##### `stop()`

Stop the mesh network client.

```python
await mesh.stop()
```

**Returns:** `None`

##### `discover_peers()`

Discover peers from blockchain validator registry.

```python
peers = await mesh.discover_peers()
```

**Returns:** `List[PeerInfo]` - List of discovered peers

##### `announce_fl_round()`

Announce new federated learning round.

```python
await mesh.announce_fl_round(
    round_id: str,
    dataset_name: str,
    target_participants: int,
    deadline: int,
    min_stake: int = 1000,
    reward_pool: int = 50000,
    model_hash: Optional[str] = None,
)
```

**Parameters:**
- `round_id` (str): Unique round identifier
- `dataset_name` (str): Dataset name for training
- `target_participants` (int): Target number of validators
- `deadline` (int): Deadline in seconds from now
- `min_stake` (int): Minimum stake required (Mahogany)
- `reward_pool` (int): Total rewards (Mahogany)
- `model_hash` (str, optional): Initial model hash

**Returns:** `None`

##### `send_model_delta()`

Send model delta to specific peer.

```python
success = await mesh.send_model_delta(
    recipient_id: str,
    round_id: str,
    model_cid: str,
    quality_score: float,
)
```

**Parameters:**
- `recipient_id` (str): Recipient peer ID
- `round_id` (str): FL round ID
- `model_cid` (str): IPFS/Pakit CID of model
- `quality_score` (float): Model quality (0-100)

**Returns:** `bool` - True if sent successfully

##### `register_handler()`

Register message handler for specific message type.

```python
mesh.register_handler(
    message_type: MessageType,
    handler: Callable[[MeshMessage], Awaitable[None]],
)
```

**Parameters:**
- `message_type` (MessageType): Type of message to handle
- `handler` (async callable): Handler function

**Returns:** `None`

**Example:**
```python
async def handle_fl_round(message):
    print(f"New FL round: {message.payload['round_id']}")

mesh.register_handler(MessageType.FL_ROUND_START, handle_fl_round)
```

---

## Payroll API

### `PayrollConnector`

ZK-proof payroll system integration.

```python
from blockchain import PayrollConnector
```

#### Constructor

```python
PayrollConnector(
    blockchain_rpc: str = "ws://localhost:9944",
    account_seed: Optional[str] = None,
)
```

**Parameters:**
- `blockchain_rpc` (str): BelizeChain RPC endpoint
- `account_seed` (str, optional): Account seed phrase

**Returns:** `PayrollConnector` instance

#### Methods

##### `submit_payroll()`

Submit payroll batch with ZK-proofs.

```python
submission_id = await payroll.submit_payroll(
    employees: List[PayrollEntry],
    pay_period_id: str,
    employer_name: str,
    metadata: Optional[Dict] = None,
)
```

**Parameters:**
- `employees` (List[PayrollEntry]): List of employee payroll entries
- `pay_period_id` (str): Pay period identifier (e.g., "2026-02")
- `employer_name` (str): Employer organization name
- `metadata` (dict, optional): Additional metadata

**Returns:** `str` - Submission ID

**Example:**
```python
employees = [
    PayrollEntry(
        employee_id="emp_001",
        account_id="5GrwvaEF...",
        gross_salary=500000,  # $5,000 in cents
        deductions=50000,     # $500 in cents
        pay_period_start=1707782400,
        pay_period_end=1708387200,
    ),
]

submission_id = await payroll.submit_payroll(
    employees=employees,
    pay_period_id="2026-02",
    employer_name="Ministry of Health",
)
```

##### `get_employee_paystub()`

Retrieve employee paystub (private query).

```python
paystub = await payroll.get_employee_paystub(
    submission_id: str,
    employee_id: str,
)
```

**Parameters:**
- `submission_id` (str): Payroll submission ID
- `employee_id` (str): Employee identifier

**Returns:** `dict` - Paystub information

**Response:**
```python
{
    "employee_id": "emp_001",
    "account_id": "5GrwvaEF...",
    "gross_salary": 500000,
    "tax_amount": 87500,
    "deductions": 50000,
    "net_salary": 362500,
    "pay_period_start": 1707782400,
    "pay_period_end": 1708387200,
    "employer_name": "Ministry of Health",
}
```

##### `verify_payroll_submission()`

Verify payroll submission (for validators).

```python
is_valid = await payroll.verify_payroll_submission(
    submission_id: str,
)
```

**Parameters:**
- `submission_id` (str): Submission to verify

**Returns:** `bool` - True if valid

##### `submit_verification()`

Submit verification result (for validators).

```python
reward = await payroll.submit_verification(
    submission_id: str,
    is_valid: bool,
)
```

**Parameters:**
- `submission_id` (str): Submission ID
- `is_valid` (bool): Verification result

**Returns:** `int` - Reward amount (Mahogany)

---

## Blockchain API

### `SubstrateClient`

BelizeChain Substrate client.

```python
from blockchain import SubstrateClient
```

#### Constructor

```python
client = SubstrateClient(
    rpc_url: str = "ws://localhost:9944",
)
```

**Parameters:**
- `rpc_url` (str): Substrate node RPC URL

**Returns:** `SubstrateClient` instance

#### Methods

##### `get_validators()`

Get list of active validators.

```python
validators = await client.get_validators()
```

**Returns:** `List[str]` - List of validator account IDs

##### `get_identity()`

Get identity information for account.

```python
identity = await client.get_identity(
    account_id: str,
)
```

**Parameters:**
- `account_id` (str): Account ID

**Returns:** `dict` - Identity information

##### `submit_extrinsic()`

Submit signed extrinsic to blockchain.

```python
receipt = await client.submit_extrinsic(
    call_module: str,
    call_function: str,
    call_params: dict,
    keypair: Keypair,
)
```

**Parameters:**
- `call_module` (str): Pallet name
- `call_function` (str): Function name
- `call_params` (dict): Function parameters
- `keypair` (Keypair): Signing keypair

**Returns:** `ExtrinsicReceipt`

---

## Model API

### `NawalTransformer`

Sovereign transformer model.

```python
from client.model import NawalTransformer
```

#### Constructor

```python
model = NawalTransformer(
    vocab_size: int = 50000,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    d_ff: int = 3072,
    max_seq_len: int = 2048,
    dropout: float = 0.1,
)
```

**Parameters:**
- `vocab_size` (int): Vocabulary size
- `d_model` (int): Model dimension
- `n_layers` (int): Number of transformer layers
- `n_heads` (int): Number of attention heads
- `d_ff` (int): Feedforward dimension
- `max_seq_len` (int): Maximum sequence length
- `dropout` (float): Dropout rate

**Returns:** `NawalTransformer` instance

#### Class Methods

##### `from_pretrained()`

Load pretrained model from checkpoint.

```python
model = NawalTransformer.from_pretrained(
    checkpoint_path: str,
)
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file

**Returns:** `NawalTransformer` instance

##### `from_config()`

Create model from configuration.

```python
model = NawalTransformer.from_config(
    config: dict,
)
```

**Parameters:**
- `config` (dict): Model configuration

**Returns:** `NawalTransformer` instance

#### Methods

##### `generate()`

Generate text from prompt.

```python
output = model.generate(
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
)
```

**Parameters:**
- `prompt` (str): Input prompt
- `max_length` (int): Maximum output length
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k sampling
- `top_p` (float): Nucleus sampling threshold

**Returns:** `str` - Generated text

##### `save_checkpoint()`

Save model checkpoint.

```python
model.save_checkpoint(
    path: str,
)
```

**Parameters:**
- `path` (str): Save path

**Returns:** `None`

---

## Hybrid API

### `HybridInferenceEngine`

Teacher-student inference routing.

```python
from hybrid import HybridInferenceEngine
```

#### Constructor

```python
engine = HybridInferenceEngine(
    nawal_model: NawalTransformer,
    teacher_api_key: str,
    confidence_threshold: float = 0.85,
)
```

**Parameters:**
- `nawal_model` (NawalTransformer): Student model
- `teacher_api_key` (str): Teacher API key
- `confidence_threshold` (float): Confidence threshold for routing

**Returns:** `HybridInferenceEngine` instance

#### Methods

##### `infer()`

Run inference with automatic routing.

```python
response = await engine.infer(
    prompt: str,
    max_length: int = 100,
)
```

**Parameters:**
- `prompt` (str): Input prompt
- `max_length` (int): Maximum response length

**Returns:** `str` - Generated response

**Example:**
```python
engine = HybridInferenceEngine(
    nawal_model=model,
    teacher_api_key="sk-...",
)

response = await engine.infer("What is the capital of Belize?")
print(response)  # "The capital of Belize is Belmopan."
```

---

## Storage API

### `PakitClient`

Content-addressable storage client.

```python
from storage.pakit_client import PakitClient
```

#### Constructor

```python
pakit = PakitClient(
    api_endpoint: str = "http://localhost:5001",
)
```

**Parameters:**
- `api_endpoint` (str): Pakit API endpoint

**Returns:** `PakitClient` instance

#### Methods

##### `store()`

Store data and get CID.

```python
cid = await pakit.store(
    data: bytes,
)
```

**Parameters:**
- `data` (bytes): Data to store

**Returns:** `str` - Content ID (CID)

##### `retrieve()`

Retrieve data by CID.

```python
data = await pakit.retrieve(
    cid: str,
)
```

**Parameters:**
- `cid` (str): Content ID

**Returns:** `bytes` - Retrieved data

---

## Data Types

### `PayrollEntry`

Individual employee payroll entry.

```python
from blockchain import PayrollEntry

entry = PayrollEntry(
    employee_id: str,
    account_id: str,
    gross_salary: int,
    deductions: int,
    pay_period_start: int,
    pay_period_end: int,
)
```

**Fields:**
- `employee_id` (str): Unique employee identifier
- `account_id` (str): BelizeChain account ID
- `gross_salary` (int): Gross salary in cents
- `deductions` (int): Total deductions in cents
- `pay_period_start` (int): Period start (Unix timestamp)
- `pay_period_end` (int): Period end (Unix timestamp)
- `tax_amount` (int, auto): Calculated tax
- `net_salary` (int, auto): Calculated net salary

### `PeerInfo`

Mesh network peer information.

```python
from blockchain.mesh_network import PeerInfo

peer = PeerInfo(
    peer_id: str,
    account_id: str,
    multiaddr: str,
    public_key: str,
    stake_amount: int,
    last_seen: float,
    is_validator: bool,
)
```

**Fields:**
- `peer_id` (str): Unique peer identifier
- `account_id` (str): BelizeChain account
- `multiaddr` (str): Multiaddress
- `public_key` (str): Ed25519 public key
- `stake_amount` (int): Staked amount
- `last_seen` (float): Last seen timestamp
- `is_validator` (bool): Validator status

### `MessageType`

Mesh message types.

```python
from blockchain.mesh_network import MessageType

MessageType.FL_ROUND_START      # FL round announcement
MessageType.FL_ROUND_END        # FL round completion
MessageType.MODEL_DELTA_SEND    # Model delta transfer
MessageType.HEARTBEAT           # Keepalive
MessageType.GOSSIP              # Generic gossip
```

---

## Configuration

### `Config`

Global configuration object.

```python
from config import Config

config = Config.from_yaml("config.prod.yaml")

# Access configuration
print(config.blockchain.rpc_url)
print(config.mesh_network.listen_port)
print(config.model.size)
```

**Configuration sections:**
- `blockchain`: Blockchain connection settings
- `mesh_network`: Mesh network configuration
- `api`: API server settings
- `model`: Model configuration
- `training`: Training parameters
- `logging`: Logging configuration

---

**Next Steps**:
- [Configuration Reference](configuration.md)
- [Architecture Overview](../architecture/overview.md)
- [Deployment Guide](../guides/deployment.md)
