# 🏗️ Architecture Overview

**Version**: 1.1.0  
**Last Updated**: February 13, 2026

---

## System Architecture

Nawal AI is a sovereign federated learning platform built on BelizeChain, combining blockchain consensus with decentralized machine learning.

```
┌─────────────────────────────────────────────────────────────┐
│                     Nawal AI Platform                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   Validator  │   │   Validator  │   │   Validator  │   │
│  │   Node 1     │◄─►│   Node 2     │◄─►│   Node 3     │   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
│         │  Mesh Network     │                  │           │
│         └───────────────────┴──────────────────┘           │
│                          │                                  │
│                          ▼                                  │
│              ┌────────────────────────┐                     │
│              │   BelizeChain Layer    │                     │
│              │  - Identity Registry   │                     │
│              │  - Staking System      │                     │
│              │  - Payroll Pallet      │                     │
│              │  - Community DAO       │                     │
│              └────────────────────────┘                     │
│                          │                                  │
│                          ▼                                  │
│              ┌────────────────────────┐                     │
│              │   Storage Layer        │                     │
│              │  - Pakit DAG           │                     │
│              │  - Model Checkpoints   │                     │
│              └────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Transformer Architecture

**Pure sovereign transformer** - no GPT-2 dependencies

```python
# architecture/transformer.py

class NawalTransformer:
    """
    Pure transformer architecture for Nawal AI.
    
    Sizes:
    - Small: 117M parameters (12 layers, 768 hidden, 12 heads)
    - Medium: 350M parameters (24 layers, 1024 hidden, 16 heads)
    - Large: 1.3B parameters (36 layers, 1536 hidden, 24 heads)
    """
    
    def __init__(self, config):
        self.embedding = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.head = LanguageModelHead(config)
```

**Features**:
- Multi-head self-attention
- Layer normalization
- Feedforward neural networks
- Positional embeddings
- Multilingual tokenization (English, Spanish, Kriol)

**Training**:
- Corpus: Belize legislative documents, news, literature
- Tokenizer: BPE with 50,000 vocabulary
- Context length: 2048 tokens
- Training data: ~500M tokens

---

### 2. Federated Learning System

**Flower (flwr) integration** for decentralized training

```python
# client/genome_trainer.py

class GenomeTrainer(fl.client.NumPyClient):
    """Federated learning client for Nawal AI."""
    
    def get_parameters(self):
        """Return model parameters to aggregator."""
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """Train model on local data."""
        set_parameters(self.model, parameters)
        train(self.model, self.trainloader)
        return get_parameters(self.model), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": accuracy}
```

**FL Round Flow**:
1. **Coordinator announces** FL round via mesh network
2. **Validators join** by staking DALLA tokens
3. **Local training** on Belize corpus data
4. **Model deltas** uploaded to Pakit DAG
5. **Aggregation** using FedAvg algorithm
6. **Rewards distributed** based on contribution quality

**Proof-of-Useful-Work (PoUW)**:
- Validators earn DALLA for training
- Rewards proportional to model quality
- Byzantine-resistant aggregation
- Slashing for malicious updates

---

### 3. Hybrid Teacher-Student System

**Progressive sovereignty** through knowledge distillation

```python
# hybrid/engine.py

class HybridInferenceEngine:
    """
    Routes queries to Nawal (student) or DeepSeek-Coder (teacher).
    
    Strategy:
    - 95% of queries → Nawal (sovereign)
    - 5% of queries → DeepSeek (fallback + teaching)
    """
    
    async def infer(self, prompt: str):
        # Check confidence
        confidence = self.confidence_estimator.predict(prompt)
        
        if confidence > 0.85:
            # High confidence - use Nawal
            return await self.nawal_model.generate(prompt)
        else:
            # Low confidence - use teacher
            teacher_response = await self.teacher_model.generate(prompt)
            
            # Log for distillation
            self.training_queue.append({
                "prompt": prompt,
                "response": teacher_response,
                "confidence": confidence
            })
            
            return teacher_response
```

**Knowledge Distillation**:
- Teacher responses → training data
- Periodic fine-tuning on hard cases
- Confidence threshold increases over time
- Goal: 100% sovereignty by Q4 2026

---

### 4. Mesh Networking

**P2P communication** for validators

```python
# blockchain/mesh_network.py

class MeshNetworkClient:
    """
    Decentralized mesh network for validator communication.
    
    Features:
    - Automatic peer discovery
    - Gossip protocol for message propagation
    - Ed25519 cryptographic signing
    - Byzantine-resistant consensus
    """
    
    async def announce_fl_round(self, round_id, dataset, participants):
        """Broadcast FL round to all validators."""
        message = self._create_message(MessageType.FL_ROUND_START, {
            "round_id": round_id,
            "dataset": dataset,
            "target_participants": participants,
        })
        await self._broadcast_message(message)
```

**Message Types**:
- FL round announcements
- Model delta transfers
- Heartbeats
- Gossip messages

**Security**:
- Ed25519 signatures
- Reputation-based filtering
- TTL-based loop prevention
- Message deduplication

---

### 5. ZK-Proof Payroll System

**Privacy-preserving payroll** integration

```python
# blockchain/payroll_connector.py

class PayrollConnector:
    """
    ZK-proof payroll system for government entities.
    
    Features:
    - Merkle tree commitments
    - Zero-knowledge proofs
    - Belize tax calculations
    - Private paystub queries
    """
    
    async def submit_payroll(self, employees, pay_period):
        """Submit payroll with ZK-proofs."""
        # Build Merkle tree
        merkle_root = self._build_merkle_tree(employees)
        
        # Generate ZK-proof
        proof = self._generate_zk_proof(employees)
        
        # Submit to blockchain
        tx = await self.client.submit_payroll(
            merkle_root=merkle_root,
            proof=proof,
            pay_period=pay_period,
        )
        
        return tx.hash
```

**Privacy Guarantees**:
- Salaries hidden via Merkle commitments
- ZK-proofs verify correctness
- Only employees can query their paystubs
- Validators verify without seeing data

---

## Data Flow

### Training Flow

```
┌──────────────┐
│ Belize       │
│ Corpus Data  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Preprocessing    │
│ - Tokenization   │
│ - Normalization  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Local Training   │
│ (Validator Node) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Model Delta      │
│ Upload to Pakit  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Aggregation      │
│ (FedAvg)         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Global Model     │
│ Checkpoint       │
└──────────────────┘
```

### Inference Flow

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Confidence Check │
└──────┬───────────┘
       │
       ├───────► High Confidence (>85%)
       │         │
       │         ▼
       │    ┌────────────┐
       │    │ Nawal AI   │
       │    │ (Sovereign)│
       │    └────────────┘
       │
       └───────► Low Confidence (<85%)
                │
                ▼
           ┌────────────┐
           │ DeepSeek   │
           │ (Teacher)  │
           └──────┬─────┘
                  │
                  ▼
           ┌────────────┐
           │ Log for    │
           │ Distill.   │
           └────────────┘
```

---

## Storage Architecture

### Pakit DAG

**Content-addressable storage** for models and data

```python
# storage/pakit_client.py

class PakitClient:
    """
    DAG-based storage for Nawal AI.
    
    Features:
    - Content addressing (CID)
    - Deduplication
    - Versioning
    - Distributed replication
    """
    
    async def store(self, data):
        """Store data and return CID."""
        cid = compute_cid(data)
        await self._upload(cid, data)
        return cid
    
    async def retrieve(self, cid):
        """Retrieve data by CID."""
        return await self._download(cid)
```

**Storage Layout**:
```
/models
  /checkpoints
    - checkpoint_gen_0000.pt (CID: Qm...)
    - checkpoint_gen_0001.pt (CID: Qm...)
    - final_checkpoint.pt (CID: Qm...)
  /deltas
    - round_001_validator_001.pt (CID: Qm...)
    - round_001_validator_002.pt (CID: Qm...)
/data
  /belize_corpus
    - legislative_docs.jsonl (CID: Qm...)
    - news_articles.jsonl (CID: Qm...)
```

---

## Blockchain Integration

### BelizeChain Pallets

**13 custom pallets** for governance and AI integration

1. **Identity**: Verified identity registry
2. **GenomeRegistry**: AI model registration
3. **Payroll**: ZK-proof payroll system
4. **Staking**: Validator staking
5. **Community**: DAO governance
6. **Rewards**: PoUW reward distribution
7. **Oracle**: External data feeds
8. **Treasury**: Community treasury
9. **Democracy**: On-chain voting
10. **Council**: Elected council
11. **TechnicalCommittee**: Technical governance
12. **Scheduler**: Automated tasks
13. **Preimage**: Proposal storage

### Substrate Client

```python
# blockchain/substrate_client.py

class SubstrateClient:
    """Interface to BelizeChain Substrate node."""
    
    async def get_validators(self):
        """Get list of active validators."""
        return await self.client.query("Staking", "Validators")
    
    async def submit_payroll(self, merkle_root, proof):
        """Submit payroll to Payroll pallet."""
        call = self.client.compose_call(
            call_module="Payroll",
            call_function="submit_payroll",
            call_params={
                "merkle_root": merkle_root,
                "proof": proof,
            }
        )
        return await self.client.submit_extrinsic(call)
```

---

## Security Architecture

### 1. Byzantine Resistance

**Protections against malicious validators**:
- Reputation scoring
- Stake-weighted voting
- Outlier detection in model aggregation
- Slashing for malicious updates

### 2. Cryptographic Security

**Ed25519 signatures** for all messages:
```python
from cryptography.hazmat.primitives.asymmetric import ed25519

# Sign message
signature = private_key.sign(message)

# Verify signature
public_key.verify(signature, message)
```

### 3. Privacy Preservation

**Differential privacy** for training data:
```python
# security/differential_privacy.py

class DPMechanism:
    """Differential privacy for federated learning."""
    
    def add_noise(self, gradients, epsilon=1.0):
        """Add calibrated noise to gradients."""
        noise = np.random.laplace(0, 1/epsilon, gradients.shape)
        return gradients + noise
```

---

## Performance Characteristics

### Model Performance

| Model Size | Parameters | Inference (tokens/sec) | Memory (GB) |
|------------|------------|------------------------|-------------|
| Small      | 117M       | ~100                   | 2           |
| Medium     | 350M       | ~50                    | 4           |
| Large      | 1.3B       | ~20                    | 12          |

### Network Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Mesh message | <100ms | 1000 msg/sec |
| Blockchain query | ~200ms | 500 queries/sec |
| Pakit upload | ~1s | 10 MB/sec |
| Pakit download | ~500ms | 20 MB/sec |

---

## Scalability

### Horizontal Scaling

- **Validators**: Up to 1000 concurrent nodes
- **FL participants**: 10-100 per round
- **Mesh peers**: 50-100 per validator

### Vertical Scaling

- **GPU training**: NVIDIA T4, V100, A100
- **CPU inference**: 4-16 cores
- **Memory**: 8-64 GB RAM

---

**Next Steps**:
- [Blockchain Integration Details](blockchain-integration.md)
- [Federated Learning Architecture](federated-learning.md)
- [API Reference](../reference/api-reference.md)
