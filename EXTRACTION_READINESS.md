# Nawal AI Component - Extraction Readiness Assessment

**Date**: January 31, 2026  
**Target Repository**: `github.com/BelizeChain/nawal-ai`  
**Current Status**: 🟡 PREPARATION NEEDED

---

## 📊 Component Statistics

### File Counts
- **Total Files**: 147
- **Python Files**: 90
- **Documentation**: ~10 files
- **Configuration Files**: 5 existing (pyproject.toml, requirements.txt, pytest.ini, .coveragerc, .gitignore)
- **Workflows**: 0 (need to create 4)

### Codebase Health
- **sys.path Hacks**: 4 instances found
- **Kinich Dependencies**: 3 direct Python imports
- **Pakit Dependencies**: 1 commented import
- **setup.py URLs**: Need update to standalone repo

---

## 🔴 Critical Issues Found

### 1. sys.path Hacks (4 instances)

#### A. `integration/kinich_connector.py` (Lines 110-111)
```python
# PROBLEM: Direct sys.path manipulation to import Kinich
kinich_path = os.path.join(os.path.dirname(__file__), '../../kinich')
if kinich_path not in sys.path:
    sys.path.insert(0, kinich_path)

from kinich.qml.hybrid.nawal_bridge import NawalQuantumBridge
```

**Solution**: Convert to HTTP REST API client (same as Pakit's approach)
- Add `KINICH_API_URL` environment variable
- Create HTTP client for Kinich quantum services
- Add fallback to classical processing when Kinich unavailable

#### B. `tests/test_blockchain.py` (Line 20)
```python
# PROBLEM: sys.path hack to import parent module
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Solution**: Remove - tests should use pip-installed package

#### C. `tests/conftest.py` (Line 27)
```python
# PROBLEM: sys.path hack for test fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Solution**: Remove - pytest automatically adds project root to path

---

### 2. Kinich Direct Imports (3 locations)

All in `integration/kinich_connector.py`:

```python
from kinich.qml.hybrid.nawal_bridge import NawalQuantumBridge  # Line 112
from kinich.qml.classifiers.vqc import VariationalQuantumClassifier  # Used later
from kinich.qml.models.circuit_qnn import CircuitQuantumNeuralNetwork  # Used later
```

**Solution**: Convert to HTTP API client
```python
# New approach: HTTP-based integration
class KinichQuantumClient:
    def __init__(self, api_url: str = "http://kinich:8888"):
        self.base_url = api_url
        self.session = aiohttp.ClientSession()
    
    async def quantum_process(
        self,
        features: np.ndarray,
        model_type: str = "vqc"
    ) -> Dict[str, Any]:
        """Process features via Kinich quantum API"""
        async with self.session.post(
            f"{self.base_url}/api/v1/qml/process",
            json={
                "features": features.tolist(),
                "model_type": model_type,
                "quantum_dim": self.quantum_dim
            }
        ) as resp:
            return await resp.json()
```

---

### 3. setup.py Configuration

**Current State**: Deprecated wrapper (all config in pyproject.toml)

**Action Required**: Update pyproject.toml URLs from:
```toml
repository = "https://github.com/belizechain/belizechain/tree/main/nawal"
```

**To**:
```toml
repository = "https://github.com/BelizeChain/nawal-ai"
documentation = "https://github.com/BelizeChain/nawal-ai#readme"
```

---

## 🟡 Missing Configuration Files

### Need to Create (6 files):

1. **`.env.example`** - Environment variables template
   - Blockchain integration settings
   - Kinich API endpoints
   - Pakit storage endpoints
   - Database credentials
   - Redis configuration
   - Security keys

2. **`.editorconfig`** - Editor settings
   - Python indentation (4 spaces)
   - Line length (100)
   - Charset UTF-8

3. **`.pre-commit-config.yaml`** - Code quality hooks
   - Black (formatting)
   - Ruff (linting)
   - MyPy (type checking)
   - Security scans

4. **`.dockerignore`** - Optimize Docker builds
   - Exclude tests, docs, .venv
   - Exclude model checkpoints

---

## 🟡 Missing CI/CD Workflows

### Need to Create (4 workflows):

1. **`.github/workflows/ci.yml`**
   - Test matrix: Python 3.11, 3.12, 3.13
   - Services: PostgreSQL, Redis
   - Run pytest with coverage
   - Upload to Codecov

2. **`.github/workflows/docker.yml`**
   - Multi-arch builds (amd64, arm64)
   - Push to GHCR (ghcr.io/belizechain/nawal-ai)
   - Tag with version + latest

3. **`.github/workflows/publish.yml`**
   - Trigger on release tag (v*)
   - Publish to PyPI
   - Create GitHub Release

4. **`.github/workflows/security.yml`**
   - Bandit security scan
   - Safety dependency check
   - Trivy container scan
   - CodeQL analysis

---

## 🔵 Documentation Updates Needed

### README.md Enhancements

Add **"Integration Architecture"** section:

```markdown
## 🔗 Integration Architecture

Nawal AI is part of the BelizeChain ecosystem:

| Component | Protocol | Purpose |
|-----------|----------|---------|
| **BelizeChain** | Substrate RPC (ws:9944) | Submit PoUW rewards, query staking |
| **Kinich** | HTTP REST (8888) | Quantum-enhanced ML processing |
| **Pakit** | HTTP REST (8080) | Store trained models, FL aggregates |

### Deployment Modes

**Development (Standalone)**:
```bash
# Mock external integrations
BLOCKCHAIN_ENABLED=false \
KINICH_ENABLED=false \
PAKIT_ENABLED=false \
python -m nawal.orchestrator server
```

**Integration (Full Stack)**:
```bash
docker-compose -f ../docker-compose.integrated.yml up
```

**Production (Kubernetes)**:
```bash
helm install nawal belizechain/nawal --namespace belizechain
```
```

---

## ✅ Already Configured

### Existing Files (Good State):

1. ✅ **pyproject.toml** - Modern Python packaging (PEP 621)
   - Dependencies: torch, transformers, flwr
   - Dev tools: pytest, ruff, mypy
   - Entry points: nawal-server, nawal-client

2. ✅ **requirements.txt** - Minimal dependencies list
   - Core ML libraries only
   - Could be expanded with full dependency tree

3. ✅ **pytest.ini** - Test configuration
   - Test discovery patterns
   - Coverage settings

4. ✅ **.coveragerc** - Coverage configuration
   - Source paths
   - Exclusion patterns

5. ✅ **.gitignore** - Python-specific ignores
   - __pycache__, .venv, checkpoints

6. ✅ **Dockerfile** - Container build
   - Likely needs minor updates for standalone repo

---

## 🚀 Extraction Preparation Tasks

### Phase 1: Code Fixes (Required Before Extraction)

- [ ] **Fix kinich_connector.py** - Convert to HTTP client
  - Remove sys.path hack (lines 110-111)
  - Replace `NawalQuantumBridge` import with HTTP client
  - Add `KINICH_API_URL` environment variable
  - Add health check endpoint validation
  - Add fallback to classical processing

- [ ] **Fix tests/test_blockchain.py** - Remove sys.path hack
  - Remove line 20: `sys.path.insert(0, ...)`
  - Verify imports work with pip-installed package

- [ ] **Fix tests/conftest.py** - Remove sys.path hack
  - Remove line 27: `sys.path.insert(0, ...)`
  - pytest automatically handles project root

- [ ] **Update pyproject.toml** - Standalone repo URLs
  - Change repository URL to `BelizeChain/nawal-ai`
  - Update documentation URL
  - Update bug tracker URL

### Phase 2: Configuration Files (Before Extraction)

- [ ] **Create .env.example** - All environment variables
- [ ] **Create .editorconfig** - Editor consistency
- [ ] **Create .pre-commit-config.yaml** - Code quality
- [ ] **Create .dockerignore** - Optimize Docker builds

### Phase 3: CI/CD Workflows (Before Extraction)

- [ ] **Create .github/workflows/ci.yml** - Test automation
- [ ] **Create .github/workflows/docker.yml** - Container builds
- [ ] **Create .github/workflows/publish.yml** - Release automation
- [ ] **Create .github/workflows/security.yml** - Security scans

### Phase 4: Documentation (Before Extraction)

- [ ] **Update README.md** - Add integration architecture section
- [ ] **Verify CONTRIBUTING.md** - Check for monorepo references
- [ ] **Update .github-instructions.md** - Standalone context

### Phase 5: Verification (Final Pre-Flight Check)

- [ ] **Verify**: 0 sys.path hacks remaining
- [ ] **Verify**: 0 direct Kinich/Pakit imports
- [ ] **Verify**: All URLs point to nawal-ai repo
- [ ] **Verify**: All config files created
- [ ] **Verify**: All workflows created
- [ ] **Run**: `grep -r "sys.path" nawal --include="*.py"` → 0 results
- [ ] **Run**: `grep -r "from kinich" nawal --include="*.py"` → 0 results
- [ ] **Run**: `grep -r "belizechain/belizechain" nawal` → 0 results

---

## 📝 Unique Nawal Considerations

### ML-Specific Requirements

1. **Model Checkpoints** - Large binary files
   - Add to .gitignore
   - Document model storage strategy (Pakit integration)
   - Add checkpoint cleanup scripts

2. **Federated Learning** - Multi-node orchestration
   - Document client-server architecture
   - Add docker-compose for FL simulation
   - Document production deployment (Kubernetes + Flower)

3. **Privacy Compliance** - Differential privacy
   - Document privacy budget management
   - Add DP-SGD configuration examples
   - Security audit for privacy guarantees

4. **Blockchain Integration** - PoUW rewards
   - Document reward submission workflow
   - Add example scripts for staking integration
   - Test suite for blockchain connector

### Integration Test Requirements

Nawal needs to test with:
- ✅ **Blockchain**: Staking pallet, Community pallet
- ✅ **Kinich**: Quantum ML processing (HTTP API)
- ✅ **Pakit**: Model storage, FL aggregation storage
- ✅ **Redis**: Caching, job queue
- ✅ **PostgreSQL**: Metadata, FL rounds tracking

**Integration test environment**:
```yaml
services:
  blockchain: ghcr.io/belizechain/blockchain:latest
  redis: redis:7-alpine
  postgres: postgres:15
  kinich: ghcr.io/belizechain/kinich-quantum:latest
  pakit: ghcr.io/belizechain/pakit-storage:latest
  nawal: (build from source)
```

---

## 🎯 Estimated Effort

- **Code Fixes**: 2-3 hours
  - kinich_connector.py refactor: 1.5 hours
  - Test fixes: 30 minutes
  - pyproject.toml updates: 15 minutes

- **Configuration Files**: 1 hour
  - .env.example: 30 minutes
  - Other configs: 30 minutes

- **CI/CD Workflows**: 1.5 hours
  - Adapt from Kinich/Pakit: 1 hour
  - Nawal-specific customizations: 30 minutes

- **Documentation**: 1 hour
  - README.md updates: 45 minutes
  - Verification: 15 minutes

**Total**: ~5-6 hours for complete preparation

---

## 🔄 Comparison with Previous Extractions

| Component | Files | Python | sys.path | External Imports | Config Files | Workflows |
|-----------|-------|--------|----------|------------------|--------------|-----------|
| **Kinich** | 127 | 85 | 5 | 0 (isolated) | 6 created | 4 created |
| **Pakit** | 123 | 85 | 4 | 1 (Kinich→HTTP) | 6 created | 4 created |
| **Nawal** | 147 | 90 | 4 | 3 (Kinich→HTTP) | 4 needed | 4 needed |

**Nawal Similarities**:
- Similar sys.path hack count (4 vs 4)
- Similar Kinich dependency issue (3 imports vs 1)
- Same configuration file needs
- Same CI/CD workflow needs

**Nawal Differences**:
- **More files** (147 vs 123/127) - larger codebase
- **More complex integration** - FL orchestration + ML models
- **Additional services** - PostgreSQL, Redis (not just blockchain)
- **pyproject.toml already exists** - Don't need to create from scratch

---

## ✅ Readiness Checklist

Before running extraction script:

### Code Quality
- [ ] 0 sys.path hacks (currently 4)
- [ ] 0 direct Kinich imports (currently 3)
- [ ] 0 monorepo paths in pyproject.toml
- [ ] All tests pass with pip-installed package

### Configuration
- [ ] .env.example created
- [ ] .editorconfig created
- [ ] .pre-commit-config.yaml created
- [ ] .dockerignore created

### CI/CD
- [ ] ci.yml workflow created
- [ ] docker.yml workflow created
- [ ] publish.yml workflow created
- [ ] security.yml workflow created

### Documentation
- [ ] README.md has integration architecture
- [ ] CONTRIBUTING.md updated
- [ ] .github-instructions.md updated
- [ ] No references to "belizechain/belizechain" monorepo

---

## 🎉 Post-Extraction Verification

After extraction, verify:

```bash
# File counts match
find /tmp/nawal-ai-extract -type f | wc -l  # Should be ~155

# No sys.path hacks
grep -r "sys.path" /tmp/nawal-ai-extract --include="*.py"  # Should be 0

# No Kinich direct imports
grep -r "from kinich" /tmp/nawal-ai-extract --include="*.py"  # Should be 0

# All critical files present
ls -la /tmp/nawal-ai-extract/{setup.py,pyproject.toml,requirements.txt,.env.example,Dockerfile}

# Git initialized
cd /tmp/nawal-ai-extract && git log --oneline  # Should show initial commit

# Ready for GitHub push
cd /tmp/nawal-ai-extract && git status  # Should be clean
```

---

## 📚 References

- **Kinich Extraction**: `/home/wicked/belizechain-belizechain/EXTRACT_KINICH.sh`
- **Pakit Extraction**: `/home/wicked/belizechain-belizechain/EXTRACT_PAKIT.sh`
- **Integration Architecture**: `/home/wicked/belizechain-belizechain/INTEGRATION_ARCHITECTURE.md`
- **Copilot Instructions**: `/home/wicked/belizechain-belizechain/.github/copilot-instructions.md`

---

**Status**: 🟡 READY TO START PREPARATION  
**Next Step**: Fix code issues (4 sys.path hacks, 3 Kinich imports)  
**Target**: github.com/BelizeChain/nawal-ai
