# 🚨 Nawal AI - Immediate Azure Action Items
**Generated:** February 11, 2026  
**Priority:** HIGH  
**Status:** Nawal is running ✅ but needs optimizations

---

## Critical Fixes (This Week)

### 1. ✅ **FIXED: Host Binding Inconsistency**
**Issue:** API server was defaulting to `127.0.0.1` instead of `0.0.0.0`

**Fix Applied:**
```python
# api_server.py - Updated host binding priority
host = os.getenv("NAWAL_API_HOST", os.getenv("NAWAL_HOST", os.getenv("HOST", "0.0.0.0")))
```

**Test:**
```bash
# Verify the fix works
docker build -t nawal-test .
docker run -p 8080:8080 -e ENVIRONMENT=production nawal-test
curl http://localhost:8080/health  # Should return {"status":"healthy",...}
```

---

### 2. ⚠️ **TODO: Tighten CORS Policy**
**Current:** Allows all origins (`allow_origins=["*"]`)

**Action Required:**
```python
# api_server.py - Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://belizechain.org",
        "https://dashboard.belizechain.org",
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)
```

**Test:**
```bash
# After change, test CORS
curl -H "Origin: https://belizechain.org" -I http://localhost:8080/health
# Should include: Access-Control-Allow-Origin: https://belizechain.org
```

---

### 3. ⚡ **TODO: Add Application Insights**
**Why:** Currently no centralized logging/monitoring in Azure Portal

**Steps:**
```bash
# 1. Install SDK
pip install opencensus-ext-azure

# 2. Get connection string from Azure Portal
az monitor app-insights component create \
  --app nawal-ai-insights \
  --location eastus \
  --resource-group BelizeChain

# 3. Add to .env.example and deployment
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx;IngestionEndpoint=...

# 4. Update api_server.py (see full audit report for code)
```

**Resources:**
- [Application Insights Python Setup](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python)

---

## High Priority (Next Sprint - 1-2 Weeks)

### 4. 🔐 **Integrate Azure Key Vault**
**Current:** Secrets passed via GitHub Actions environment variables

**Action:**
```bash
# 1. Create Key Vault
az keyvault create \
  --name belizechain-kv \
  --resource-group BelizeChain \
  --location eastus

# 2. Add secrets
az keyvault secret set --vault-name belizechain-kv --name "postgres-password" --value "xxx"
az keyvault secret set --vault-name belizechain-kv --name "redis-password" --value "xxx"

# 3. Enable managed identity for container app
az containerapp identity assign --name nawal-ai --resource-group BelizeChain

# 4. Grant access
IDENTITY_ID=$(az containerapp identity show --name nawal-ai --resource-group BelizeChain --query principalId -o tsv)
az keyvault set-policy --name belizechain-kv --object-id $IDENTITY_ID --secret-permissions get list
```

---

### 5. 🚀 **Migrate to Azure Container Apps**
**Current:** VM with SSH-based deployment  
**Target:** Managed container service with auto-scaling

**Why?**
- ✅ Auto-scaling (1-10 instances based on load)
- ✅ Zero-downtime deployments
- ✅ Built-in load balancing and HTTPS
- ✅ Pay-per-use (scale to zero when idle)
- ✅ No VM management (no patching, no SSH keys)

**Migration Steps:**
```bash
# 1. Create Container Apps environment
az containerapp env create \
  --name belizechain-env \
  --resource-group BelizeChain \
  --location eastus

# 2. Create container app
az containerapp create \
  --name nawal-ai \
  --resource-group BelizeChain \
  --environment belizechain-env \
  --image belizechainregistry.azurecr.io/nawal-ai:latest \
  --target-port 8080 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10 \
  --cpu 2.0 \
  --memory 4Gi \
  --registry-server belizechainregistry.azurecr.io \
  --registry-username ${{ secrets.ACR_USERNAME }} \
  --registry-password ${{ secrets.ACR_PASSWORD }} \
  --env-vars \
    NAWAL_API_HOST=0.0.0.0 \
    BLOCKCHAIN_WS_URL=ws://blockchain:9944 \
    ENVIRONMENT=production \
    LOG_LEVEL=info

# 3. Update GitHub Actions deploy.yml to push to Container Apps instead of VM
```

**Cost Comparison:**
- VM: ~$80/month (always on)
- Container Apps: ~$50/month (1 instance) or ~$0/month when idle
- **Savings:** ~30-40% + better scalability

---

### 6. 🛡️ **Add API Authentication**
**Current:** No authentication (open API)

**Options:**
1. **API Key** (simple, good for internal services)
2. **Azure AD B2C** (enterprise, good for user authentication)
3. **JWT tokens** (flexible, good for microservices)

**Quick Fix - API Key:**
```python
# api_server.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    valid_key = os.getenv("NAWAL_API_KEY")
    if not valid_key or api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Protect endpoints
@app.post("/api/v1/fl/rounds", dependencies=[Security(verify_api_key)])
async def start_round(request: StartRoundRequest):
    ...
```

---

## Medium Priority (Next Month)

### 7. ⚡ **Enable Redis Caching**
**Issue:** Redis credentials passed but not used

**Action:**
```python
# pip install aioredis
import aioredis

@app.on_event("startup")
async def setup_redis():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        app.state.redis = await aioredis.create_redis_pool(redis_url)
        logger.info("✅ Connected to Redis")

# Use for caching participant stats, round status, etc.
```

---

### 8. 🎯 **Add Rate Limiting**
**Why:** Prevent abuse and DoS attacks

```bash
# pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/fl/participants/submit")
@limiter.limit("100/minute")
async def submit_model(request: Request, data: SubmitModelRequest):
    ...
```

---

### 9. 📦 **Optimize Docker Image**
**Current:** ~1.5GB  
**Target:** < 800MB

**Steps:**
1. Multi-stage build
2. Non-root user
3. Remove build dependencies from final image
4. Use `.dockerignore` properly

*See full audit report for implementation*

---

## Testing Checklist

Before deploying any changes:

```bash
# 1. Local test
python api_server.py
curl http://localhost:8080/health

# 2. Docker test
docker build -t nawal-test .
docker run -p 8080:8080 -e NAWAL_API_HOST=0.0.0.0 nawal-test
curl http://localhost:8080/health

# 3. Load test (optional)
ab -n 1000 -c 10 http://localhost:8080/health

# 4. Check logs
docker logs <container-id>

# 5. Verify environment variables
docker exec <container-id> env | grep NAWAL
```

---

## Quick Reference

### Environment Variables (Production)
```env
# Required
NAWAL_API_HOST=0.0.0.0
PORT=8080
ENVIRONMENT=production
BLOCKCHAIN_WS_URL=ws://blockchain:9944

# Optional (recommended)
APPLICATIONINSIGHTS_CONNECTION_STRING=xxx
AZURE_KEY_VAULT_URL=https://belizechain-kv.vault.azure.net/
NAWAL_API_KEY=<strong-random-key>
LOG_LEVEL=info
WORKERS=4
```

### Health Check
```bash
# Local
curl http://localhost:8080/health

# Production (VM)
curl http://<vm-ip>:8002/health

# Production (Container Apps - after migration)
curl https://nawal-ai.happyflower-12345.eastus.azurecontainerapps.io/health
```

### Deployment Commands
```bash
# Build and push to ACR
docker build -t belizechainregistry.azurecr.io/nawal-ai:latest .
docker push belizechainregistry.azurecr.io/nawal-ai:latest

# GitHub Actions (automatic)
git push origin main  # Triggers CI/CD pipeline
```

---

## Questions?

- **Full Audit Report:** [NAWAL_AZURE_AUDIT_2026-02-11.md](./NAWAL_AZURE_AUDIT_2026-02-11.md)
- **BelizeChain Docs:** `/docs` folder
- **Azure Support:** [Azure Portal](https://portal.azure.com)

---

**Last Updated:** February 11, 2026  
**Next Review:** February 18, 2026 (check progress on action items)
