# 🚀 Deployment Guide

**Version**: 1.1.0  
**Target**: Production Environments  
**Status**: Production Ready

---

## Overview

This guide covers deploying Nawal AI validators to production environments including Azure, Kubernetes, and Docker Compose.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 100 GB SSD | 500+ GB NVMe |
| Network | 10 Mbps | 100+ Mbps |
| GPU | None | NVIDIA T4+ (for training) |

### Software Requirements

- **OS**: Ubuntu 22.04 LTS, Debian 11+, or containerized
- **Python**: 3.11+
- **Docker**: 24.0+
- **Kubernetes**: 1.28+ (optional)
- **BelizeChain Node**: v1.0+ (local or remote)

---

## Quick Deploy

### Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/BelizeChain/nawal-ai.git
cd nawal-ai

# Create environment file
cat > .env <<EOF
BLOCKCHAIN_RPC=ws://blockchain:9944
MESH_PEER_ID=validator_$(hostname)
MESH_LISTEN_PORT=9090
API_PORT=8080
EOF

# Start services
docker compose up -d

# Check status
docker compose ps
docker compose logs -f validator
```

### Docker Compose File

```yaml
version: '3.9'

services:
  validator:
    image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
    container_name: nawal-validator
    restart: unless-stopped
    ports:
      - "${API_PORT:-8080}:8080"
      - "${MESH_LISTEN_PORT:-9090}:9090"
    environment:
      - BLOCKCHAIN_RPC=${BLOCKCHAIN_RPC}
      - MESH_PEER_ID=${MESH_PEER_ID}
      - MESH_LISTEN_PORT=9090
      - LOG_LEVEL=INFO
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - nawal-network

  blockchain:
    image: parity/substrate:latest
    container_name: belizechain-node
    restart: unless-stopped
    ports:
      - "9944:9944"
      - "9933:9933"
    command: >
      --dev
      --rpc-external
      --rpc-cors=all
      --ws-external
    networks:
      - nawal-network

networks:
  nawal-network:
    driver: bridge
```

---

## Azure Deployment

### Azure Container Instances (Simple)

```bash
# Login to Azure
az login

# Create resource group
az group create \
  --name nawal-rg \
  --location eastus

# Deploy container
az container create \
  --resource-group nawal-rg \
  --name nawal-validator \
  --image belizechainregistry.azurecr.io/nawal-ai:1.1.0 \
  --ports 8080 9090 \
  --cpu 4 \
  --memory 8 \
  --environment-variables \
    BLOCKCHAIN_RPC=ws://blockchain.belizechain.org:9944 \
    MESH_PEER_ID=validator_azure_001 \
  --registry-login-server belizechainregistry.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD

# Get public IP
az container show \
  --resource-group nawal-rg \
  --name nawal-validator \
  --query ipAddress.fqdn
```

### Azure Kubernetes Service (Scalable)

```yaml
# nawal-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nawal-validator
  namespace: nawal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nawal-validator
  template:
    metadata:
      labels:
        app: nawal-validator
    spec:
      containers:
        - name: validator
          image: belizechainregistry.azurecr.io/nawal-ai:1.1.0
          ports:
            - containerPort: 8080
              name: api
            - containerPort: 9090
              name: mesh
          env:
            - name: BLOCKCHAIN_RPC
              value: "ws://blockchain.belizechain.org:9944"
            - name: MESH_PEER_ID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: MESH_LISTEN_PORT
              value: "9090"
          resources:
            requests:
              memory: "4Gi"
              cpu: "2000m"
            limits:
              memory: "8Gi"
              cpu: "4000m"
          volumeMounts:
            - name: checkpoints
              mountPath: /app/checkpoints
      volumes:
        - name: checkpoints
          persistentVolumeClaim:
            claimName: nawal-checkpoints

---
apiVersion: v1
kind: Service
metadata:
  name: nawal-api
  namespace: nawal
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8080
      name: api
    - port: 9090
      targetPort: 9090
      name: mesh
  selector:
    app: nawal-validator

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nawal-checkpoints
  namespace: nawal
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: azurefile
  resources:
    requests:
      storage: 100Gi
```

Deploy to AKS:

```bash
# Create AKS cluster
az aks create \
  --resource-group nawal-rg \
  --name nawal-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group nawal-rg \
  --name nawal-cluster

# Create namespace
kubectl create namespace nawal

# Deploy
kubectl apply -f nawal-deployment.yaml

# Check status
kubectl get pods -n nawal
kubectl logs -f deployment/nawal-validator -n nawal
```

---

## Kubernetes Deployment

### Helm Chart (Recommended)

```bash
# Add Nawal Helm repository
helm repo add nawal https://charts.belizechain.org
helm repo update

# Install with default values
helm install nawal-validator nawal/nawal-ai \
  --namespace nawal \
  --create-namespace \
  --set blockchain.rpc=ws://blockchain.belizechain.org:9944 \
  --set replicaCount=3

# Custom values
cat > values.yaml <<EOF
replicaCount: 5

image:
  repository: belizechainregistry.azurecr.io/nawal-ai
  tag: "1.1.0"

blockchain:
  rpc: "ws://blockchain.belizechain.org:9944"

mesh:
  listenPort: 9090
  maxPeers: 50

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 2000m
    memory: 4Gi

persistence:
  enabled: true
  size: 100Gi
  storageClass: fast-ssd
EOF

helm install nawal-validator nawal/nawal-ai \
  --namespace nawal \
  --create-namespace \
  --values values.yaml
```

---

## Environment Configuration

### Environment Variables

```bash
# Blockchain
BLOCKCHAIN_RPC="ws://localhost:9944"           # BelizeChain RPC endpoint
BLOCKCHAIN_ACCOUNT_SEED="//Alice"              # Validator account seed

# Mesh Networking
MESH_PEER_ID="validator_001"                   # Unique peer ID
MESH_LISTEN_PORT=9090                          # Mesh network port
MESH_MAX_PEERS=50                              # Maximum peers
MESH_HEARTBEAT_INTERVAL=60                     # Heartbeat interval (seconds)

# API Server
API_HOST="0.0.0.0"                             # API bind address
API_PORT=8080                                  # API port
API_WORKERS=4                                  # Gunicorn workers

# Model Configuration
MODEL_SIZE="medium"                            # small, medium, large
MODEL_CHECKPOINT="/app/checkpoints/final_checkpoint.pt"

# Training
TRAINING_ENABLED=true                          # Enable FL training
TRAINING_BATCH_SIZE=32
TRAINING_EPOCHS=10

# Logging
LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT="json"                              # json, text
LOG_FILE="/app/logs/nawal.log"

# Metrics
PROMETHEUS_ENABLED=true                        # Enable Prometheus metrics
PROMETHEUS_PORT=9100

# Security
ENABLE_TLS=true                                # Enable TLS
TLS_CERT_PATH="/app/certs/tls.crt"
TLS_KEY_PATH="/app/certs/tls.key"
```

### Configuration File

```yaml
# config.prod.yaml
blockchain:
  rpc_url: "ws://blockchain.belizechain.org:9944"
  account_seed: "${BLOCKCHAIN_ACCOUNT_SEED}"

mesh_network:
  peer_id: "${HOSTNAME}"
  listen_port: 9090
  max_peers: 50
  heartbeat_interval: 60
  gossip_fanout: 5

api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  cors_origins:
    - "https://dashboard.belizechain.org"
    - "https://api.belizechain.org"

model:
  size: "medium"
  checkpoint: "/app/checkpoints/final_checkpoint.pt"
  device: "cuda"  # or "cpu"

training:
  enabled: true
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001

logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/nawal.log"

metrics:
  prometheus_enabled: true
  prometheus_port: 9100
```

---

## Monitoring

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'nawal-validators'
    static_configs:
      - targets:
          - 'validator-01:9100'
          - 'validator-02:9100'
          - 'validator-03:9100'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Nawal AI Validators",
    "panels": [
      {
        "title": "Active Peers",
        "targets": [
          {
            "expr": "nawal_mesh_peers_total"
          }
        ]
      },
      {
        "title": "FL Rounds Completed",
        "targets": [
          {
            "expr": "rate(nawal_fl_rounds_total[5m])"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "nawal_model_accuracy"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

```bash
# API health check
curl http://localhost:8080/health

# Mesh network health
curl http://localhost:9090/health

# Prometheus metrics
curl http://localhost:9100/metrics
```

---

## Security

### TLS Configuration

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 \
  -keyout tls.key -out tls.crt \
  -days 365 -nodes \
  -subj "/CN=validator.belizechain.org"

# Use Let's Encrypt (production)
certbot certonly --standalone \
  -d validator.belizechain.org \
  --email admin@belizechain.org
```

### Firewall Rules

```bash
# Allow API and mesh ports
sudo ufw allow 8080/tcp   # API
sudo ufw allow 9090/tcp   # Mesh network
sudo ufw allow 9100/tcp   # Prometheus

# Block everything else
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw enable
```

### Azure Network Security Group

```bash
az network nsg rule create \
  --resource-group nawal-rg \
  --nsg-name nawal-nsg \
  --name AllowAPI \
  --priority 100 \
  --source-address-prefixes '*' \
  --destination-port-ranges 8080 \
  --protocol Tcp \
  --access Allow

az network nsg rule create \
  --resource-group nawal-rg \
  --nsg-name nawal-nsg \
  --name AllowMesh \
  --priority 110 \
  --source-address-prefixes '*' \
  --destination-port-ranges 9090 \
  --protocol Tcp \
  --access Allow
```

---

## Backup & Recovery

### Checkpoint Backup

```bash
# Backup checkpoints to Azure Blob Storage
az storage blob upload-batch \
  --account-name nawalbackups \
  --destination checkpoints \
  --source ./checkpoints \
  --pattern "*.pt"

# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf checkpoints_${TIMESTAMP}.tar.gz ./checkpoints
az storage blob upload \
  --account-name nawalbackups \
  --container backups \
  --file checkpoints_${TIMESTAMP}.tar.gz \
  --name checkpoints_${TIMESTAMP}.tar.gz
```

### Disaster Recovery

```bash
# Restore from backup
az storage blob download \
  --account-name nawalbackups \
  --container backups \
  --name checkpoints_20260213_120000.tar.gz \
  --file checkpoints.tar.gz

tar -xzf checkpoints.tar.gz
```

---

## Scaling

### Horizontal Scaling (Kubernetes)

```bash
# Scale to 10 replicas
kubectl scale deployment/nawal-validator \
  --replicas=10 \
  --namespace nawal

# Autoscaling
kubectl autoscale deployment/nawal-validator \
  --min=3 \
  --max=20 \
  --cpu-percent=70 \
  --namespace nawal
```

### Vertical Scaling

```bash
# Update resource limits
kubectl set resources deployment/nawal-validator \
  --limits=cpu=8000m,memory=16Gi \
  --requests=cpu=4000m,memory=8Gi \
  --namespace nawal
```

---

## Troubleshooting

### Check Logs

```bash
# Docker
docker logs nawal-validator

# Kubernetes
kubectl logs deployment/nawal-validator -n nawal

# Follow logs
kubectl logs -f deployment/nawal-validator -n nawal
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/nawal-validator \
  LOG_LEVEL=DEBUG \
  --namespace nawal
```

### Common Issues

**Issue**: Cannot connect to blockchain

```bash
# Test connection
curl -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
     http://blockchain.belizechain.org:9944
```

**Issue**: Mesh peers not discovered

```bash
# Check mesh network
kubectl exec -it deployment/nawal-validator -n nawal -- \
  python3 -c "
from blockchain import MeshNetworkClient
import asyncio

async def check():
    mesh = MeshNetworkClient('test', 9090)
    await mesh.start()
    peers = await mesh.discover_peers()
    print(f'Found {len(peers)} peers')

asyncio.run(check())
"
```

---

**Next Steps**:
- [Architecture Overview](../architecture/overview.md)
- [API Reference](../reference/api-reference.md)
- [Monitoring Guide](../reference/monitoring.md)
