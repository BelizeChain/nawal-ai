# 🧠 Nawal AI: Sovereign Federated Learning for BelizeChain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Substrate](https://img.shields.io/badge/Substrate-Polkadot%20SDK%202512-E6007A)](https://substrate.io/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)

> **Nawal** (Mayan: "Wisdom") is a 100% sovereign AI platform combining pure transformer architecture, hybrid teacher-student learning, federated training, and evolutionary genome optimization for BelizeChain's national infrastructure.

## 🌟 Key Features

### 🇧🇿 **100% Sovereign AI**
- **Zero dependency** on GPT-2, DialoGPT, or Microsoft models
- Pure NawalTransformer architecture with **random weight initialization**
- Three sizes: **small** (117M params), **medium** (350M params), **large** (1.3B params)
- Complete control over model training, deployment, and governance

### 🧬 **Genome Evolution System**
- DNA-based architecture encoding: **5KB vs 500MB** model weights
- Genetic algorithms optimize layer count, hidden size, attention heads
- Stores evolved architectures in **Pakit** (IPFS/Arweave) for decentralized access
- Converts genome DNA → NawalConfig → NawalTransformer via `GenomeToNawalAdapter`

### 🎓 **Hybrid Teacher-Student Learning**
- **Nawal** (student, sovereign): Targets **95% of production traffic**
- **DeepSeek-Coder-33B** (teacher): Fallback for **5% complex queries**
- Multi-factor confidence scoring: entropy, perplexity, length, language detection
- Progressive sovereignty: start **50/50**, evolve to **95/5** Nawal/DeepSeek

### 🔐 **Privacy-Preserving Federated Learning**
- **Flower (flwr)** framework for distributed training across national institutions
- **Differential privacy** (Opacus) with ε=1.0 (high privacy), δ=1e-5
- **Byzantine fault tolerance** via Krum aggregation (rejects malicious updates)
- No centralized data collection — models train on local devices

### 🏆 **Proof of Useful Work (PoUW) Integration**
- Validators earn **DALLA tokens** for contributing federated learning updates
- Quality-based rewards: **40% accuracy**, **30% timeliness**, **30% privacy compliance**
- Blockchain consensus rewards computational contributions to national AI

### 🌍 **Multilingual Support**
- **English**, **Spanish**, **Kriol** (Maya, Garifuna, Mandarin in progress)
- Automatic language detection for Spanish-English code switching
- Culturally-aware responses for Belizean contexts

### 📦 **Pakit Storage Integration**
- Decentralized model hosting via **IPFS** (primary) and **Arweave** (permanent)
- Quantum compression reduces model sizes by **60-80%**
- Content-addressable deduplication for efficient storage
- Off-chain storage with on-chain proofs in `LandLedger` pallet

## � Integration Architecture

Nawal AI is part of the **BelizeChain ecosystem** with multi-repository architecture:

| Component | Protocol | Purpose |
|-----------|----------|---------|
| **BelizeChain** | Substrate RPC (ws:9944, http:9933) | Submit PoUW rewards, query staking pallet |
| **Kinich Quantum** | HTTP REST (8888) | Quantum-enhanced ML processing (VQC, QNN, QSVM) |
| **Pakit Storage** | HTTP REST (8080), DAG Gateway (8081) | Store trained models, FL aggregates, genome DNAs |

### Repository Information

- **Repository**: `github.com/BelizeChain/nawal-ai`
- **Role**: Federated learning + privacy-preserving ML component
- **Architecture**: Multi-repository, unified system
- **Integration**: Production requires blockchain + Pakit (Kinich optional)

### Deployment Modes

**Development (Standalone)**:
```bash
# Mock external integrations for local development
BLOCKCHAIN_ENABLED=false \
KINICH_ENABLED=false \
PAKIT_ENABLED=false \
python -m nawal.orchestrator server
```

**Integration Testing (Full Stack)**:
```bash
# Run all BelizeChain components
cd ../
docker-compose -f docker-compose.integrated.yml up
```

**Production (Kubernetes)**:
```bash
# Helm chart deployment
helm install nawal belizechain/nawal \
  --namespace belizechain \
  --set blockchain.wsUrl=ws://blockchain:9944 \
  --set kinich.apiUrl=http://kinich:8888 \
  --set pakit.apiUrl=http://pakit:8080
```

### Integration Workflows

**Proof of Useful Work (PoUW) Rewards**:
1. Nawal client completes federated learning round
2. Client submits model update + quality metrics to aggregator
3. Aggregator scores contribution (accuracy, timeliness, privacy)
4. Aggregator submits PoUW proof to blockchain Staking pallet
5. Blockchain issues DALLA rewards to client account

**Model Storage**:
1. Nawal trains model checkpoint (500MB)
2. Upload to Pakit with quantum compression (150MB)
3. Pakit returns Content ID (CID) + Merkle proof
4. Nawal submits storage proof to blockchain LandLedger pallet
5. Blockchain validates Merkle proof and records CID
6. Users retrieve model via CID from Pakit gateway

**Quantum-Enhanced Inference** (optional):
1. Nawal extracts classical features from input
2. Send features to Kinich API for quantum processing
3. Kinich processes via VQC/QNN (quantum neural network)
4. Kinich returns quantum-enhanced features
5. Nawal uses enhanced features for final prediction

## �🚀 Quick Start

### Installation

```bash
# Clone BelizeChain repository
git clone https://github.com/belizechain/belizechain.git
cd belizechain/nawal

# Create virtual environment (Python 3.11+ required)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Nawal with all dependencies
pip install -e ".[all]"

# Or install specific features:
pip install -e ".[dev]"        # Development tools (pytest, ruff, mypy)
pip install -e ".[server]"     # Federated learning server (FastAPI, Redis)
pip install -e ".[deepseek]"   # Hybrid engine with DeepSeek teacher (vLLM)
pip install -e ".[monitoring]" # TensorBoard, Weights & Biases tracking
```

### Basic Usage

#### 1️⃣ **Pure Nawal Transformer**

```python
from nawal.architecture import NawalTransformer, NawalConfig

# Create sovereign model (no external dependencies)
config = NawalConfig.nawal_small()  # 117M parameters, 768 hidden, 12 layers
model = NawalTransformer(config)

# Generate text
prompt = "The capital of Belize is"
output = model.generate(
    prompt=prompt,
    max_length=50,
    temperature=0.8,
    top_p=0.95
)
print(output)  # "The capital of Belize is Belmopan, located in the Cayo District..."

# Save/load locally
model.save_pretrained("/path/to/nawal-small")
loaded_model = NawalTransformer.load_pretrained("/path/to/nawal-small")

# Upload to Pakit (IPFS/Arweave)
from nawal.storage.pakit_client import PakitClient
pakit = PakitClient(ipfs_gateway="http://localhost:5001")
cid = model.save_to_pakit(pakit, metadata={"version": "1.0", "language": "en-es"})
print(f"Model uploaded to IPFS: {cid}")
```

#### 2️⃣ **Hybrid Engine with DeepSeek**

```python
from nawal.hybrid import HybridNawalEngine

# Initialize hybrid system
engine = HybridNawalEngine(
    nawal_model_path="/path/to/nawal-medium",  # 350M sovereign model
    teacher_model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    confidence_threshold=0.75,  # Route to DeepSeek if Nawal confidence < 0.75
    use_vllm=True  # Fast inference with vLLM
)

# Generate with intelligent routing
result = engine.generate(
    prompt="Write a Python function to validate SSN format for BelizeID",
    max_tokens=200,
    temperature=0.7
)

print(result.text)
print(f"Used: {result.source}")  # "nawal" or "deepseek"
print(f"Confidence: {result.confidence:.2f}")
print(f"Sovereignty Rate: {engine.get_sovereignty_rate():.1%}")  # Target: 95%+
```

#### 3️⃣ **Genome Evolution**

```python
from nawal.genome import create_baseline_nawal_genome, genome_to_nawal

# Create baseline genome (nawal-small architecture)
genome = create_baseline_nawal_genome(
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# Evolve architecture with genetic algorithm
from nawal.genome.evolution import GeneticEvolver
evolver = GeneticEvolver(
    population_size=20,
    mutation_rate=0.15,
    crossover_rate=0.7,
    num_generations=50
)

# Fitness function: minimize params while maintaining perplexity < 20
best_genome = evolver.evolve(
    fitness_fn=lambda g: evaluate_perplexity(g),
    constraint_fn=lambda g: g.num_parameters() < 200_000_000  # < 200M params
)

# Convert evolved genome to model
model = genome_to_nawal(best_genome)
print(f"Evolved model: {model.config.num_parameters():,} parameters")
```

#### 4️⃣ **Federated Learning Server**

```bash
# Start federated learning coordinator
nawal-server \
  --rounds 100 \
  --clients-per-round 5 \
  --min-clients 3 \
  --aggregation krum \
  --differential-privacy \
  --epsilon 1.0 \
  --delta 1e-5 \
  --blockchain-integration
```

```python
# Client training (runs on local hospital/school/government office)
from nawal.client import NawalFederatedClient

client = NawalFederatedClient(
    model_path="/path/to/nawal-small",
    server_address="grpc://fl-server.belizechain.gov:8080",
    client_id="HOSPITAL_KARL_HEUSNER",
    data_path="/secure/local/medical_records.csv"
)

# Train locally, submit encrypted updates to server
client.train(
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    differential_privacy=True
)
```

#### 5️⃣ **Blockchain Integration (PoUW)**

```python
from nawal.blockchain import StakingConnector

# Submit federated learning work to Substrate blockchain
connector = StakingConnector(
    node_url="ws://localhost:9944",
    validator_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
)

# Report training contribution
quality_score = 0.92  # 92% model improvement
timeliness_score = 1.0  # Submitted before deadline
honesty_score = 0.98  # 98% differential privacy compliance

reward_dalla = connector.report_training(
    quality=quality_score,
    timeliness=timeliness_score,
    honesty=honesty_score
)
print(f"PoUW Reward: {reward_dalla / 10**12:.2f} DALLA")
```

## 📁 Architecture

```
nawal/
├── architecture/          # Pure Nawal Transformer (zero external dependencies)
│   ├── config.py         # NawalConfig with nawal_small/medium/large presets
│   ├── embeddings.py     # Token + positional embeddings
│   ├── attention.py      # Multi-head self-attention with KV caching
│   ├── feedforward.py    # FFN with GELU/ReLU/SiLU activation
│   └── transformer.py    # NawalTransformer with generate() method
│
├── hybrid/               # Teacher-student hybrid system
│   ├── confidence.py     # Multi-factor confidence scoring
│   ├── teacher.py        # DeepSeek-Coder-33B wrapper with vLLM
│   ├── router.py         # Intelligent routing logic
│   └── engine.py         # HybridNawalEngine orchestrator
│
├── genome/               # Evolutionary architecture optimization
│   ├── encoding.py       # DNA-based architecture representation
│   ├── evolution.py      # Genetic algorithm with mutation/crossover
│   ├── nawal_adapter.py  # Genome DNA → NawalConfig → NawalTransformer
│   └── storage.py        # Pakit IPFS/Arweave integration
│
├── client/               # Federated learning client
│   ├── nawal.py          # Main Nawal model interface (renamed from nawal_gpt.py)
│   ├── model.py          # BelizeChainLLM wrapper for training
│   └── trainer.py        # Local training loop with DP-SGD
│
├── server/               # Federated learning server
│   ├── aggregator.py     # Krum aggregation with Byzantine tolerance
│   ├── strategy.py       # FedAvg, FedProx, FedNova strategies
│   └── orchestrator.py   # Flower server with client selection
│
├── security/             # Privacy and compliance
│   ├── differential_privacy.py  # Opacus DP-SGD wrapper
│   ├── byzantine.py      # Krum, Median aggregation
│   └── compliance.py     # KYC/AML checks, content filtering
│
├── blockchain/           # Substrate integration
│   ├── staking_connector.py  # PoUW reward submission
│   └── consensus_connector.py # PQW (quantum work) submission
│
├── storage/              # Decentralized model hosting
│   └── pakit_client.py   # IPFS/Arweave upload/download
│
└── training/             # Advanced training techniques (COMING SOON)
    └── distillation.py   # Knowledge distillation (DeepSeek → Nawal)
```

## 🎯 Use Cases

### 1. **National Healthcare AI**
Train models on patient data **without** sharing sensitive medical records:
- Hospitals contribute federated updates (encrypted, differentially private)
- Central model improves diagnosis accuracy across all institutions
- DALLA rewards validators for quality contributions

### 2. **Education Platform**
Multilingual tutoring for English, Spanish, Kriol students:
- Nawal handles **95%** of basic Q&A (sovereign, low-latency)
- DeepSeek assists with **5%** complex coding/math problems
- Genome evolution optimizes model size for mobile deployment

### 3. **Government Document Processing**
BelizeID, land titles, payroll automation:
- Compliance filtering ensures KYC/AML adherence
- Pakit stores document embeddings (quantum compressed)
- LandLedger pallet records on-chain storage proofs

### 4. **Tourism Chatbot**
5-8% cashback rewards for DALLA spending at verified merchants:
- Language detection auto-switches English/Spanish/Kriol
- Hybrid routing: Nawal for simple queries, DeepSeek for complex itineraries
- Oracle pallet verifies merchant authenticity

## 🔧 Development

### Running Tests

```bash
# Unit tests (fast)
pytest tests/ -v -m unit

# Integration tests (requires blockchain, IPFS, Redis)
pytest tests/ -v -m integration

# Specific module tests
pytest tests/test_architecture.py -v
pytest tests/test_genome_adapter.py -v
pytest tests/test_hybrid_engine.py -v

# Coverage report
pytest --cov=nawal --cov-report=html tests/
```

### Code Quality

```bash
# Linting
ruff check nawal/

# Type checking
mypy nawal/

# Auto-format
ruff format nawal/
```

### Build & Deployment

```bash
# Install in editable mode
pip install -e ".[all]"

# Build distribution packages
pip install build
python -m build

# Upload to PyPI (maintainers only)
pip install twine
twine upload dist/*
```

## 🛡️ Security

### Differential Privacy
All federated learning uses **Opacus** DP-SGD:
- **ε = 1.0** (strong privacy)
- **δ = 1e-5** (failure probability)
- **Max gradient norm = 1.0** (clipping)

### Byzantine Fault Tolerance
**Krum aggregation** rejects malicious updates:
- Selects **k** most consistent updates (k = ⌊clients / 2⌋ + 1)
- Discards outliers that could poison model

### Compliance
All models enforce:
- **KYC/AML checks** via `Compliance` pallet
- **Content filtering** for hate speech, illegal content
- **SSN/Passport validation** for BelizeID integration

## 📊 Performance

| Model | Parameters | Hidden Size | Layers | Perplexity | Inference Speed | Memory |
|-------|-----------|-------------|--------|-----------|-----------------|--------|
| **nawal-small** | 117M | 768 | 12 | ~25 | 120 tokens/s | 2.3 GB |
| **nawal-medium** | 350M | 1024 | 24 | ~18 | 80 tokens/s | 5.1 GB |
| **nawal-large** | 1.3B | 1536 | 36 | ~12 | 35 tokens/s | 12.4 GB |
| **DeepSeek-V3** | 33B | 4096 | 40 | ~8 | 15 tokens/s | 66 GB |

**Target Hybrid Performance**:
- **95%** queries served by Nawal (sovereign, fast)
- **5%** queries routed to DeepSeek (complex, accurate)
- **Average latency**: < 200ms (nawal-small on GPU)

## 🌐 Multilingual Support

| Language | Coverage | Status | Examples |
|----------|----------|--------|----------|
| **English** | 100% | ✅ Production | "What is BelizeID?" |
| **Spanish** | 95% | ✅ Production | "¿Qué es BelizeID?" |
| **Kriol** | 80% | 🚧 Beta | "Wa di BelizeID?" |
| **Maya** | 40% | 🔬 Research | "Bix a beel BelizeID?" |
| **Garifuna** | 35% | 🔬 Research | "Ka BelizeID?" |
| **Mandarin** | 20% | 🔬 Research | "什么是BelizeID?" |

**Code-Switching**: Automatically detects and responds to Spanish-English mixing (common in Belize).

## 🤝 Contributing

We welcome contributions from the Belizean developer community!

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** changes: `git commit -m "Add my feature"`
4. **Push** to branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flower (flwr)**: Federated learning framework
- **PyTorch**: Deep learning library
- **Substrate/Polkadot**: Blockchain runtime
- **DeepSeek AI**: Teacher model for hybrid system
- **IPFS/Arweave**: Decentralized storage via Pakit
- **Belizean developers**: For cultural and linguistic guidance

## 📞 Support

- **Documentation**: [docs.belizechain.org/nawal](https://docs.belizechain.org/nawal)
- **Discord**: [discord.gg/belizechain](https://discord.gg/belizechain)
- **Email**: [dev@belizechain.org](mailto:dev@belizechain.org)
- **GitHub Issues**: [github.com/belizechain/belizechain/issues](https://github.com/belizechain/belizechain/issues)

---

**🇧🇿 Built with pride for Belize's sovereign digital future**
