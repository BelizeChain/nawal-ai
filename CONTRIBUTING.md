# Contributing to Nawal AI

Thank you for your interest in contributing to Nawal, BelizeChain's sovereign AI platform! We welcome contributions from developers worldwide, with special emphasis on Belizean developers building national infrastructure.

## 🌟 Ways to Contribute

### 1. **Code Contributions**
- Pure Nawal transformer improvements (architecture, attention, embeddings)
- Hybrid engine enhancements (confidence scoring, routing)
- Genome evolution algorithms (mutation strategies, fitness functions)
- Federated learning optimizations (aggregation, differential privacy)
- Blockchain integration (PoUW rewards, consensus)
- Storage layer (Pakit IPFS/Arweave improvements)

### 2. **Documentation**
- Tutorials and examples
- API reference improvements
- Architecture diagrams
- Multilingual translations (Spanish, Kriol, Garifuna, Maya)

### 3. **Testing**
- Unit tests (pytest)
- Integration tests (blockchain, IPFS, Redis)
- Performance benchmarks
- Security audits

### 4. **Data & Training**
- Belizean corpora (legal, cultural, economic texts)
- Multilingual datasets (Kriol, Garifuna, Maya)
- Federated learning datasets (privacy-preserving)
- Tokenizer vocabulary extensions

### 5. **Bug Reports & Feature Requests**
- GitHub Issues with detailed reproduction steps
- Feature proposals aligned with sovereignty goals

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (required for async, Pydantic v2)
- **Git** for version control
- **Poetry** or **pip** for dependency management
- **Docker** (optional, for integration tests)

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/belizechain.git
cd belizechain/nawal

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies in editable mode
pip install -e ".[dev,server,monitoring]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
pytest tests/ -v -m unit
```

### Project Structure

```
nawal/
├── architecture/       # Pure Nawal transformer (NO external deps)
├── hybrid/            # Teacher-student system (DeepSeek integration)
├── genome/            # Evolutionary architecture optimization
├── client/            # Federated learning client
├── server/            # Federated learning server (Flower)
├── security/          # Differential privacy, Byzantine tolerance
├── blockchain/        # Substrate integration (PoUW, consensus)
├── storage/           # Pakit IPFS/Arweave client
└── training/          # Knowledge distillation (COMING SOON)
```

## 📝 Contribution Process

### 1. **Find or Create an Issue**

Before writing code, check [GitHub Issues](https://github.com/belizechain/belizechain/issues) for:
- Open bugs or feature requests
- Issues labeled `good-first-issue` (beginner-friendly)
- Issues labeled `help-wanted` (community priorities)

If no issue exists, create one with:
- **Title**: Clear, concise summary
- **Description**: Problem statement, expected behavior, current behavior
- **Environment**: Python version, OS, GPU (if applicable)
- **Reproduction**: Minimal code to reproduce bug

### 2. **Branch Naming Convention**

```bash
git checkout -b <type>/<short-description>
```

**Types:**
- `feature/` - New functionality (e.g., `feature/genome-crossover`)
- `fix/` - Bug fixes (e.g., `fix/attention-nan-values`)
- `docs/` - Documentation (e.g., `docs/hybrid-engine-tutorial`)
- `test/` - Test additions/improvements (e.g., `test/federated-integration`)
- `refactor/` - Code restructuring (e.g., `refactor/config-validation`)

**Examples:**
```bash
git checkout -b feature/krum-aggregation-variance
git checkout -b fix/tokenizer-kriol-encoding
git checkout -b docs/genome-evolution-guide
```

### 3. **Code Standards**

#### Python Style (PEP 8 + Google Docstrings)

```python
"""Module docstring with overview.

Detailed description of module purpose, key classes/functions, and usage examples.

Examples:
    Basic usage::
    
        from nawal.architecture import NawalConfig
        config = NawalConfig.nawal_small()
"""

from typing import Optional, List, Dict
import torch


def compute_confidence(
    logits: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.75,
) -> float:
    """Compute confidence score from logits using entropy and perplexity.
    
    Args:
        logits: Model output logits of shape [vocab_size] or [batch_size, vocab_size].
        temperature: Softmax temperature for probability scaling. Higher = more uniform.
        threshold: Minimum confidence score for routing to Nawal (vs. teacher).
    
    Returns:
        float: Confidence score in range [0.0, 1.0]. Higher = more confident.
    
    Raises:
        ValueError: If temperature <= 0 or threshold not in [0, 1].
    
    Examples:
        Single prediction::
        
            logits = model(input_ids)["logits"][-1]  # Last token
            conf = compute_confidence(logits, temperature=0.8)
            if conf > 0.75:
                print("High confidence, use Nawal")
        
        Batch processing::
        
            logits = model(input_ids)["logits"]  # [batch, seq_len, vocab]
            for i in range(len(logits)):
                conf = compute_confidence(logits[i, -1])
                print(f"Sample {i}: {conf:.2f}")
    
    Note:
        This function combines entropy (uncertainty) and perplexity (model surprise)
        to provide a robust confidence estimate. Used by HybridNawalEngine for routing.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
    
    # Implementation here...
    return confidence_score
```

**Key Requirements:**
- **Type hints** for all function parameters and return values
- **Google-style docstrings** (Args, Returns, Raises, Examples, Note)
- **Examples section** showing realistic usage
- **Input validation** with clear error messages
- **Line length ≤ 100 characters** (enforced by ruff)

#### Code Formatting

We use **Ruff** for linting and formatting:

```bash
# Auto-format code
ruff format nawal/

# Check for linting errors
ruff check nawal/

# Auto-fix common issues
ruff check --fix nawal/
```

#### Type Checking

We use **MyPy** for static type analysis:

```bash
# Run type checker
mypy nawal/

# Specific module
mypy nawal/architecture/transformer.py
```

### 4. **Writing Tests**

Every contribution **MUST** include tests:

#### Unit Tests (Fast, No External Dependencies)

```python
# tests/test_confidence.py
import pytest
import torch
from nawal.hybrid.confidence import compute_confidence


def test_compute_confidence_high():
    """Test confidence for peaked distribution (should be high)."""
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])  # Peaked at first token
    conf = compute_confidence(logits, temperature=1.0)
    assert conf > 0.9, f"Expected high confidence, got {conf:.2f}"


def test_compute_confidence_low():
    """Test confidence for uniform distribution (should be low)."""
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Uniform distribution
    conf = compute_confidence(logits, temperature=1.0)
    assert conf < 0.3, f"Expected low confidence, got {conf:.2f}"


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5, 2.0])
def test_compute_confidence_temperature_invariance(temperature):
    """Test that relative confidence ordering is preserved across temperatures."""
    peaked = torch.tensor([10.0, 0.0, 0.0, 0.0])
    uniform = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    conf_peaked = compute_confidence(peaked, temperature=temperature)
    conf_uniform = compute_confidence(uniform, temperature=temperature)
    
    assert conf_peaked > conf_uniform, \
        f"Peaked distribution should have higher confidence at T={temperature}"


def test_compute_confidence_invalid_temperature():
    """Test that invalid temperature raises ValueError."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Temperature must be > 0"):
        compute_confidence(logits, temperature=0.0)
```

**Test Markers:**
```python
@pytest.mark.unit         # Fast tests, no external services
@pytest.mark.integration  # Requires blockchain, IPFS, Redis
@pytest.mark.slow         # Long-running tests (>10 seconds)
@pytest.mark.gpu          # Requires CUDA GPU
```

**Running Tests:**
```bash
# All unit tests (fast)
pytest tests/ -v -m unit

# Specific test file
pytest tests/test_confidence.py -v

# Integration tests (requires services)
pytest tests/ -v -m integration

# With coverage report
pytest --cov=nawal --cov-report=html tests/
```

### 5. **Commit Messages**

Use **conventional commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/improvements
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvements
- `chore`: Build, dependencies, tooling

**Examples:**
```
feat(genome): Add variance-based crossover strategy

Implements adaptive crossover that selects genes based on fitness
variance across population. Improves convergence speed by 23% on
benchmark tasks.

Closes #142

---

fix(attention): Handle NaN values in attention scores

Attention computation could produce NaN when input contains inf values.
Added pre-softmax clamping to prevent numerical instability.

Fixes #158

---

docs(hybrid): Add tutorial for confidence threshold tuning

Provides step-by-step guide with code examples for optimizing
confidence_threshold parameter based on accuracy/sovereignty trade-off.

Related to #165
```

### 6. **Pull Request Process**

#### Before Submitting

```bash
# 1. Ensure all tests pass
pytest tests/ -v

# 2. Run linting
ruff check nawal/

# 3. Run type checking
mypy nawal/

# 4. Update CHANGELOG.md (if applicable)
echo "## [Unreleased]\n- feat: Your feature description" >> CHANGELOG.md

# 5. Rebase on latest main
git fetch upstream
git rebase upstream/main
```

#### PR Template

When creating a PR, include:

**Title:** `<type>(<scope>): <short description>`

**Description:**

```markdown
## Summary
Brief overview of changes and motivation.

## Changes
- Bullet list of specific changes
- New files created
- Modified files

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines (ruff, mypy)
- [ ] Docstrings updated (Google style)
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)
- [ ] Screenshots/GIFs (for UI changes)

## Related Issues
Closes #123
Related to #456
```

#### Review Process

1. **Automated Checks**: GitHub Actions will run tests, linting, type checking
2. **Code Review**: At least **1 maintainer** approval required
3. **CI/CD**: All checks must pass (green checkmarks)
4. **Merge**: Maintainer will merge using **squash and merge**

## 🧬 Special: Genome Evolution Contributions

Nawal's genome system is a unique feature. When contributing genome-related code:

### 1. **DNA Encoding Format**

```python
# genome/encoding.py
class GenomeDNA:
    """Compact 5KB representation of Nawal architecture.
    
    Attributes:
        version (int): Genome format version (current: 3).
        hidden_size (int): Transformer hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_attention_heads (int): Multi-head attention heads.
        intermediate_size (int): FFN intermediate dimension.
        activation (str): Activation function (gelu, relu, silu).
        ...
    """
```

### 2. **Fitness Functions**

```python
# genome/evolution.py
def evaluate_genome_fitness(
    genome: GenomeDNA,
    dataset: str = "belize-validation",
    metric: str = "perplexity",
) -> float:
    """Evaluate genome fitness on validation dataset.
    
    Args:
        genome: Genome DNA to evaluate.
        dataset: Validation dataset name (belize-validation, wikitext-103).
        metric: Fitness metric (perplexity, accuracy, f1).
    
    Returns:
        float: Fitness score (higher = better). Normalized to [0, 1].
    
    Note:
        Lower perplexity = higher fitness (inverted for maximization).
    """
```

### 3. **Mutation Strategies**

When adding mutation strategies, preserve **valid architecture constraints**:

```python
def mutate_hidden_size(genome: GenomeDNA, rate: float = 0.15) -> GenomeDNA:
    """Mutate hidden_size while maintaining attention_heads divisibility.
    
    Args:
        genome: Original genome.
        rate: Mutation strength (0.0-1.0).
    
    Returns:
        GenomeDNA: Mutated genome with valid hidden_size.
    
    Note:
        Ensures hidden_size % num_attention_heads == 0 (required by attention).
    """
    if random.random() < rate:
        # Mutate but preserve divisibility
        multiplier = random.choice([0.75, 1.25, 1.5])
        new_hidden = int(genome.hidden_size * multiplier)
        # Round to nearest multiple of attention_heads
        new_hidden = (new_hidden // genome.num_attention_heads) * genome.num_attention_heads
        genome.hidden_size = max(256, min(4096, new_hidden))  # Clamp to valid range
    return genome
```

## 🇧🇿 Belizean Cultural Guidelines

When contributing content involving Belizean culture:

### 1. **Language Accuracy**
- **Kriol**: Verify spelling with Kriol Council resources
- **Garifuna**: Consult National Garifuna Council
- **Maya**: Distinguish Q'eqchi', Mopan, Yucatec dialects

### 2. **Cultural Sensitivity**
- Avoid stereotypes about ethnic groups
- Respect sacred sites (ATM Cave, Caracol, Lamanai)
- Use gender-neutral language where appropriate

### 3. **Legal Compliance**
- Financial Services Commission (FSC) regulations for KYC/AML
- Data Protection Act for privacy
- Copyright Act for training data sources

## 🏆 Recognition

Contributors will be recognized in:
- **CHANGELOG.md** for each release
- **README.md** contributors section
- **On-chain credits** (optional, DALLA microtransactions)

Top contributors may be invited to:
- BelizeChain Developer Council
- Governance proposals for Nawal roadmap
- Annual BelizeChain Summit (Belize City)

## 📞 Getting Help

- **Discord**: [discord.gg/belizechain](https://discord.gg/belizechain)
- **GitHub Discussions**: [github.com/belizechain/belizechain/discussions](https://github.com/belizechain/belizechain/discussions)
- **Email**: dev@belizechain.org
- **Office Hours**: Every Tuesday 3-5 PM BZT (Zoom link in Discord)

## 🎯 Roadmap Priorities

Current focus areas (Q1-Q2 2026):

1. **Knowledge Distillation** (nawal/training/distillation.py)
   - Transfer learning from DeepSeek → Nawal
   - KL divergence loss optimization
   - Target: 95%+ sovereignty rate

2. **Multilingual Expansion**
   - Kriol tokenizer (BPE trained on Belizean corpus)
   - Garifuna language model (collaboration with NGC)
   - Maya Q'eqchi' pilot (Toledo District schools)

3. **Production Deployment**
   - Kubernetes Helm charts for federated learning
   - Azure Quantum integration (via Kinich)
   - IPFS/Arweave mirroring (via Pakit)

4. **Performance Optimization**
   - FlashAttention-3 integration
   - Quantization (INT8, FP16) for mobile deployment
   - Speculative decoding (draft + verify)

---

**Thank you for contributing to Belize's sovereign AI future! 🇧🇿**
