# ⚙️ Configuration Reference

**Version**: 1.1.0  
**Last Updated**: February 13, 2026

---

## Overview

Nawal AI supports configuration via YAML files, environment variables, and Python configuration objects.

**Priority** (highest to lowest):
1. Environment variables
2. Configuration file
3. Default values

---

## Configuration Files

### Development Configuration

**File**: `config.dev.yaml`

```yaml
# Development configuration for Nawal AI

blockchain:
  rpc_url: "ws://localhost:9944"
  account_seed: "//Alice"  # Development seed
  
mesh_network:
  peer_id: "${HOSTNAME}"
  listen_port: 9090
  max_peers: 10
  heartbeat_interval: 60
  gossip_fanout: 3
  peer_timeout: 300

api:
  host: "127.0.0.1"
  port: 8080
  workers: 1
  reload: true
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8080"

model:
  size: "small"
  checkpoint: "checkpoints/final_checkpoint.pt"
  device: "cpu"
  vocab_size: 50000
  max_seq_len: 2048

training:
  enabled: true
  batch_size: 16
  epochs: 3
  learning_rate: 0.0001
  gradient_accumulation: 4

hybrid:
  enabled: false
  teacher_api_key: ""
  confidence_threshold: 0.85
  teacher_usage_ratio: 0.05

storage:
  pakit_endpoint: "http://localhost:5001"
  checkpoint_dir: "checkpoints"
  data_dir: "data"

logging:
  level: "DEBUG"
  format: "text"
  file: "logs/nawal_dev.log"
  console: true

metrics:
  prometheus_enabled: false
  prometheus_port: 9100
```

### Production Configuration

**File**: `config.prod.yaml`

```yaml
# Production configuration for Nawal AI

blockchain:
  rpc_url: "${BLOCKCHAIN_RPC}"
  account_seed: "${BLOCKCHAIN_ACCOUNT_SEED}"
  
mesh_network:
  peer_id: "${MESH_PEER_ID}"
  listen_port: 9090
  max_peers: 50
  heartbeat_interval: 60
  gossip_fanout: 5
  peer_timeout: 300

api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  reload: false
  cors_origins:
    - "https://dashboard.belizechain.org"
    - "https://api.belizechain.org"

model:
  size: "medium"
  checkpoint: "/app/checkpoints/final_checkpoint.pt"
  device: "cuda"
  vocab_size: 50000
  max_seq_len: 2048

training:
  enabled: true
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001
  gradient_accumulation: 1

hybrid:
  enabled: true
  teacher_api_key: "${TEACHER_API_KEY}"
  confidence_threshold: 0.85
  teacher_usage_ratio: 0.05

storage:
  pakit_endpoint: "https://pakit.belizechain.org"
  checkpoint_dir: "/app/checkpoints"
  data_dir: "/app/data"

logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/nawal.log"
  console: true

metrics:
  prometheus_enabled: true
  prometheus_port: 9100

security:
  enable_tls: true
  tls_cert_path: "/app/certs/tls.crt"
  tls_key_path: "/app/certs/tls.key"
```

---

## Environment Variables

### Blockchain

```bash
# BelizeChain RPC endpoint
BLOCKCHAIN_RPC="ws://localhost:9944"

# Account seed phrase (keep secret!)
BLOCKCHAIN_ACCOUNT_SEED="//Alice"

# Network ID
BLOCKCHAIN_NETWORK_ID="belizechain_mainnet"
```

### Mesh Network

```bash
# Unique peer identifier
MESH_PEER_ID="validator_belizecity_01"

# Listen port
MESH_LISTEN_PORT=9090

# Maximum connected peers
MESH_MAX_PEERS=50

# Heartbeat interval (seconds)
MESH_HEARTBEAT_INTERVAL=60

# Peer timeout (seconds)
MESH_PEER_TIMEOUT=300

# Gossip fanout
MESH_GOSSIP_FANOUT=5
```

### API Server

```bash
# Bind address
API_HOST="0.0.0.0"

# Listen port
API_PORT=8080

# Number of workers
API_WORKERS=4

# Enable auto-reload (development)
API_RELOAD=false

# CORS origins (comma-separated)
API_CORS_ORIGINS="https://dashboard.belizechain.org,https://api.belizechain.org"
```

### Model

```bash
# Model size: small, medium, large
MODEL_SIZE="medium"

# Checkpoint path
MODEL_CHECKPOINT="/app/checkpoints/final_checkpoint.pt"

# Device: cpu, cuda, mps
MODEL_DEVICE="cuda"

# Vocabulary size
MODEL_VOCAB_SIZE=50000

# Maximum sequence length
MODEL_MAX_SEQ_LEN=2048
```

### Training

```bash
# Enable training
TRAINING_ENABLED=true

# Batch size
TRAINING_BATCH_SIZE=32

# Number of epochs
TRAINING_EPOCHS=10

# Learning rate
TRAINING_LEARNING_RATE=0.0001

# Gradient accumulation steps
TRAINING_GRADIENT_ACCUMULATION=1

# Enable differential privacy
TRAINING_ENABLE_DP=true

# DP epsilon (privacy budget)
TRAINING_DP_EPSILON=1.0

# DP delta
TRAINING_DP_DELTA=0.00001
```

### Hybrid System

```bash
# Enable hybrid routing
HYBRID_ENABLED=true

# Teacher API key
TEACHER_API_KEY="sk-..."

# Confidence threshold
HYBRID_CONFIDENCE_THRESHOLD=0.85

# Teacher usage ratio (0-1)
HYBRID_TEACHER_USAGE_RATIO=0.05
```

### Storage

```bash
# Pakit endpoint
PAKIT_ENDPOINT="https://pakit.belizechain.org"

# Checkpoint directory
CHECKPOINT_DIR="/app/checkpoints"

# Data directory
DATA_DIR="/app/data"
```

### Logging

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL="INFO"

# Log format: text, json
LOG_FORMAT="json"

# Log file path
LOG_FILE="/app/logs/nawal.log"

# Enable console logging
LOG_CONSOLE=true
```

### Metrics

```bash
# Enable Prometheus metrics
PROMETHEUS_ENABLED=true

# Prometheus port
PROMETHEUS_PORT=9100
```

### Security

```bash
# Enable TLS
ENABLE_TLS=true

# TLS certificate path
TLS_CERT_PATH="/app/certs/tls.crt"

# TLS key path
TLS_KEY_PATH="/app/certs/tls.key"
```

---

## Python Configuration

### Loading Configuration

```python
from config import Config

# Load from YAML
config = Config.from_yaml("config.prod.yaml")

# Load from environment
config = Config.from_env()

# Load with overrides
config = Config.from_yaml("config.prod.yaml", overrides={
    "api.port": 9000,
    "model.size": "large",
})
```

### Accessing Values

```python
# Nested access
rpc_url = config.blockchain.rpc_url
listen_port = config.mesh_network.listen_port
model_size = config.model.size

# Dictionary access
rpc_url = config["blockchain"]["rpc_url"]

# Get with default
device = config.get("model.device", default="cpu")
```

### Updating Configuration

```python
# Update values
config.model.size = "large"
config.api.port = 9000

# Update from dict
config.update({
    "model": {
        "size": "large",
        "device": "cuda",
    }
})

# Save to file
config.save("config.custom.yaml")
```

---

## Model Configurations

### Small Model

```yaml
model:
  size: "small"
  vocab_size: 50000
  d_model: 768
  n_layers: 12
  n_heads: 12
  d_ff: 3072
  max_seq_len: 2048
  dropout: 0.1
  parameters: 117_000_000
```

**Hardware Requirements:**
- GPU: NVIDIA T4 or better
- RAM: 4 GB
- VRAM: 4 GB

### Medium Model (Default)

```yaml
model:
  size: "medium"
  vocab_size: 50000
  d_model: 1024
  n_layers: 24
  n_heads: 16
  d_ff: 4096
  max_seq_len: 2048
  dropout: 0.1
  parameters: 350_000_000
```

**Hardware Requirements:**
- GPU: NVIDIA V100 or better
- RAM: 8 GB
- VRAM: 16 GB

### Large Model

```yaml
model:
  size: "large"
  vocab_size: 50000
  d_model: 1536
  n_layers: 36
  n_heads: 24
  d_ff: 6144
  max_seq_len: 2048
  dropout: 0.1
  parameters: 1_300_000_000
```

**Hardware Requirements:**
- GPU: NVIDIA A100 or better
- RAM: 32 GB
- VRAM: 40 GB

---

## Training Configurations

### Quick Training (Development)

```yaml
training:
  enabled: true
  batch_size: 8
  epochs: 1
  learning_rate: 0.0001
  gradient_accumulation: 8
  enable_dp: false
  checkpointing_interval: 100
```

### Full Training (Production)

```yaml
training:
  enabled: true
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001
  gradient_accumulation: 1
  enable_dp: true
  dp_epsilon: 1.0
  dp_delta: 0.00001
  checkpointing_interval: 1000
  early_stopping: true
  early_stopping_patience: 3
```

### Distributed Training

```yaml
training:
  enabled: true
  batch_size: 64
  epochs: 10
  learning_rate: 0.0001
  distributed: true
  world_size: 4
  backend: "nccl"
  gradient_accumulation: 1
```

---

## Network Configurations

### Conservative (Low Bandwidth)

```yaml
mesh_network:
  max_peers: 10
  heartbeat_interval: 120
  gossip_fanout: 3
  message_compression: true
  max_message_size: 1048576  # 1 MB
```

### Aggressive (High Bandwidth)

```yaml
mesh_network:
  max_peers: 100
  heartbeat_interval: 30
  gossip_fanout: 10
  message_compression: false
  max_message_size: 10485760  # 10 MB
```

---

## Security Configurations

### High Security

```yaml
security:
  enable_tls: true
  tls_cert_path: "/app/certs/tls.crt"
  tls_key_path: "/app/certs/tls.key"
  require_client_cert: true
  min_tls_version: "1.3"
  
  rate_limiting:
    enabled: true
    max_requests_per_minute: 60
  
  authentication:
    enabled: true
    jwt_secret: "${JWT_SECRET}"
    token_expiry: 3600
```

---

## Monitoring Configurations

### Full Monitoring

```yaml
metrics:
  prometheus_enabled: true
  prometheus_port: 9100
  
logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/nawal.log"
  console: true
  
  handlers:
    - type: "file"
      filename: "/app/logs/nawal.log"
      max_bytes: 10485760  # 10 MB
      backup_count: 5
    
    - type: "syslog"
      address: ["localhost", 514]
    
  filters:
    - name: "sensitive_data_filter"
      patterns:
        - "BLOCKCHAIN_ACCOUNT_SEED"
        - "API_KEY"
```

---

## Configuration Validation

### Validating Configuration

```python
from config import Config, ConfigValidator

# Load configuration
config = Config.from_yaml("config.prod.yaml")

# Validate
validator = ConfigValidator()
errors = validator.validate(config)

if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✅ Configuration is valid")
```

### Common Validation Errors

```
❌ blockchain.rpc_url: Invalid WebSocket URL
❌ mesh_network.listen_port: Port out of range (1-65535)
❌ model.size: Invalid size (must be: small, medium, large)
❌ training.batch_size: Must be positive integer
❌ api.workers: Must be at least 1
```

---

## Configuration Examples

### Validator Node

```yaml
# Optimized for validator operations
blockchain:
  rpc_url: "ws://localhost:9944"
  account_seed: "${VALIDATOR_SEED}"

mesh_network:
  peer_id: "${HOSTNAME}"
  listen_port: 9090
  max_peers: 50

training:
  enabled: true
  batch_size: 32
  epochs: 10

model:
  size: "medium"
  device: "cuda"
```

### Inference-Only Node

```yaml
# Optimized for inference
blockchain:
  rpc_url: "ws://blockchain.belizechain.org:9944"

training:
  enabled: false

model:
  size: "large"
  device: "cuda"

api:
  host: "0.0.0.0"
  port: 8080
  workers: 8
```

### Development Node

```yaml
# Optimized for local development
blockchain:
  rpc_url: "ws://localhost:9944"
  account_seed: "//Alice"

mesh_network:
  peer_id: "dev_node"
  listen_port: 9090
  max_peers: 5

model:
  size: "small"
  device: "cpu"

training:
  enabled: true
  batch_size: 4
  epochs: 1

logging:
  level: "DEBUG"
  format: "text"
  console: true
```

---

**Next Steps**:
- [API Reference](api-reference.md)
- [Deployment Guide](../guides/deployment.md)
- [Architecture Overview](../architecture/overview.md)
