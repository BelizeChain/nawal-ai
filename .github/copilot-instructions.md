# Nawal AI — Sovereign Federated Learning & Privacy ML

## Project Identity
- **Repo**: `BelizeChain/nawal-ai`
- **Role**: Federated learning + privacy-preserving ML for BelizeChain
- **Language**: Python
- **Branch**: `main` (default)

## Features
- Federated learning orchestration (cross-node model training)
- Privacy-preserving ML (differential privacy, secure aggregation)
- BelizeChain pallet integration (AI oracle, compliance scoring)
- REST API server (`api_server.py`)
- Config-driven (config.dev.yaml, config.prod.yaml)

## Azure Deployment Target
- **ACR**: `belizechainacr.azurecr.io` → image: `belizechainacr.azurecr.io/nawal`
- **AKS**: `belizechain-aks` (Free tier, 1x Standard_D2s_v3, K8s v1.33.6)
- **Resource Group**: `BelizeChain` in `centralus`
- **Subscription**: `77e6d0a2-78d2-4568-9f5a-34bd62357c40`
- **Tenant**: `belizechain.org`

## Deployment Status: Phase 2 — TODO
### What needs to be done:
1. **Verify Dockerfile** — Ensure Python dependencies install, API server starts on correct port
2. **Update deploy.yml** — Migrate from VM/SSH to AKS deployment:
   - Use `azure/login@v2` with `${{ secrets.AZURE_CREDENTIALS }}`
   - Use `azure/aks-set-context@v4` with `${{ secrets.AKS_CLUSTER_NAME }}`
   - Push image to `belizechainacr.azurecr.io/nawal`
   - `kubectl apply` Deployment + Service (expose port 8001)
3. **Configure GitHub Secrets**:
   - `ACR_USERNAME` = `belizechainacr`
   - `ACR_PASSWORD` = (get from `az acr credential show --name belizechainacr`)
   - `AZURE_CREDENTIALS` = (service principal JSON)
   - `AZURE_RESOURCE_GROUP` = `BelizeChain`
   - `AKS_CLUSTER_NAME` = `belizechain-aks`
4. **K8s namespace**: Deploy into `belizechain` namespace
5. **Resource limits**: CPU-only mode, limit to 100m-500m CPU, 256Mi-1Gi RAM
6. **Use config.prod.yaml** for AKS deployment
7. **Connect to blockchain**: `ws://belizechain-node.belizechain.svc.cluster.local:9944`

## Sibling Services (same AKS cluster)
| Service | Image | Ports |
|---------|-------|-------|
| belizechain-node | `belizechainacr.azurecr.io/belizechain-node` | 30333, 9944, 9615 |
| ui | `belizechainacr.azurecr.io/ui` | 80 |
| kinich-quantum | `belizechainacr.azurecr.io/kinich` | 8000 |
| pakit-storage | `belizechainacr.azurecr.io/pakit` | 8002 |

## Dev Commands
```bash
pip install -r requirements.txt          # Install dependencies
python api_server.py                     # Run API server
docker build -t belizechainacr.azurecr.io/nawal .  # Docker image
pytest                                   # Run tests
```

## Constraints
- **CPU-only**: No GPU on AKS Free tier — ML models must use CPU inference
- **Shared node**: All services share 2 vCPU / 8GB RAM
- **Cost ceiling**: ~$75/mo total for ALL services
- **Privacy**: Federated learning data must never leave node boundaries
