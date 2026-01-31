# Nawal AI - Testing Infrastructure

Comprehensive test suite for the Nawal AI federated learning and genome evolution system.

## 📊 Test Coverage

| Component | Test File | Lines | Coverage |
|-----------|-----------|-------|----------|
| Genome System | `test_genome.py` | 480 | DNA, operators, population, history |
| Model Builder | `test_model_builder.py` | 410 | 30+ layer types, model generation |
| Training | `test_training.py` | 380 | Client training, fitness scoring |
| Federation | `test_federation.py` | 430 | FedAvg, aggregation, metrics |
| Evolution | `test_evolution.py` | 400 | Multi-generation orchestration |
| **Total** | **5 files** | **~2,100** | **>90% expected** |

## 🚀 Quick Start

### Install Dependencies

```bash
# From belizechain root
pip install -r requirements.txt

# Or install test dependencies directly
pip install pytest pytest-cov pytest-asyncio pytest-mock pytest-xdist
```

### Run All Tests

```bash
# From belizechain/nawal directory
pytest

# With coverage report
pytest --cov=nawal --cov-report=html

# Parallel execution (faster)
pytest -n auto
```

## 📖 Test Organization

### Unit Tests

Test individual components in isolation:

```bash
# Test genome system only
pytest tests/test_genome.py

# Test model builder only
pytest tests/test_model_builder.py

# Test specific test class
pytest tests/test_training.py::TestGenomeTrainer

# Test specific test method
pytest tests/test_genome.py::TestDNA::test_dna_initialization
```

### Integration Tests

Test component interactions:

```bash
# Run integration tests only
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```

### Performance Benchmarks

```bash
# Run benchmark tests
pytest -m benchmark

# Skip benchmarks (faster)
pytest -m "not benchmark"
```

### Slow Tests

```bash
# Run all tests including slow ones
pytest --runslow

# Skip slow tests (default)
pytest -m "not slow"
```

## 🧪 Test Categories

### 1. Genome Tests (`test_genome.py`)

- **DNA Encoding/Decoding**: Serialize/deserialize genomes
- **Layer Genes**: Create and manipulate layer genes
- **Connection Genes**: Test network topology
- **Mutation Operators**: Add/remove layers, mutate parameters
- **Crossover Operators**: Single-point, uniform crossover
- **Population Management**: Initialize, evolve, select
- **Innovation History**: Track structural innovations

**Example**:
```python
def test_dna_clone(sample_dna):
    cloned = sample_dna.clone()
    assert len(cloned.layer_genes) == len(sample_dna.layer_genes)
```

### 2. Model Builder Tests (`test_model_builder.py`)

- **30+ Layer Types**: Linear, Conv2D, LSTM, Transformer, etc.
- **Model Construction**: Build PyTorch models from genomes
- **Forward Pass**: Verify model execution
- **Complex Architectures**: Deep networks, CNNs, RNNs
- **Error Handling**: Invalid configurations

**Example**:
```python
def test_build_simple_model(sample_dna):
    builder = ModelBuilder()
    model = builder.build(sample_dna)
    x = torch.randn(4, sample_dna.input_size)
    output = model(x)
    assert output.shape[1] == sample_dna.output_size
```

### 3. Training Tests (`test_training.py`)

- **Local Training**: Train models on validators
- **Fitness Calculation**: PoUW scoring (quality, timeliness, honesty)
- **Optimization**: Adam, SGD, AdamW, RMSprop
- **Gradient Clipping**: Prevent gradient explosion
- **Checkpointing**: Save/load training state
- **Loss Functions**: CrossEntropy, MSE, etc.

**Example**:
```python
def test_full_training_loop(sample_model, train_val_dataloaders):
    trainer = GenomeTrainer(config=training_config)
    trainer.set_model(sample_model)
    train_loader, val_loader = train_val_dataloaders
    history = trainer.train(train_loader, val_loader, epochs=2)
    assert "train_loss" in history
```

### 4. Federation Tests (`test_federation.py`)

- **FedAvg Aggregation**: Average client model updates
- **Weighted Aggregation**: Different client contributions
- **Client Selection**: Random, fraction-based
- **Federated Rounds**: Multi-round training
- **Metrics Tracking**: Per-client and global metrics
- **Byzantine Resilience**: Detect malicious clients (placeholder)

**Example**:
```python
def test_fedavg_aggregation(client_models):
    aggregator = FederatedAggregator()
    client_params = [m.state_dict() for m in client_models]
    global_params = aggregator.fedavg_aggregate(client_params)
    assert global_params is not None
```

### 5. Evolution Tests (`test_evolution.py`)

- **Multi-Generation Evolution**: Evolve over many generations
- **Selection Strategies**: Tournament, roulette wheel, rank
- **Fitness Improvement**: Verify fitness increases
- **Elitism**: Preserve best genomes
- **Population Diversity**: Maintain architectural variety
- **Checkpointing**: Save/resume evolution
- **Federated Evolution**: Distribute fitness evaluation

**Example**:
```python
def test_fitness_improvement(orchestrator, sample_dataloader):
    history = orchestrator.evolve(dataloader=sample_dataloader, generations=5)
    assert history["best_fitness"][-1] >= history["best_fitness"][0]
```

## 🔧 Test Fixtures

Reusable test fixtures defined in `conftest.py`:

### Configuration Fixtures
- `genome_config`: Genome architecture settings
- `evolution_config`: Evolution hyperparameters
- `training_config`: Training settings
- `federated_config`: Federation settings
- `nawal_config`: Complete system configuration

### Data Fixtures
- `sample_dataset`: Random tensor dataset
- `sample_dataloader`: DataLoader with batching
- `train_val_dataloaders`: Split train/validation
- `client_dataloaders`: Multiple client datasets

### Model Fixtures
- `sample_model`: Simple PyTorch model
- `client_models`: Multiple models for federation
- `sample_dna`: Pre-configured genome

### Directory Fixtures
- `temp_dir`: Temporary directory for tests
- `checkpoint_dir`: Checkpoint storage
- `data_dir`: Data storage

### Mock Fixtures
- `mock_fitness_scores`: Random fitness values
- `byzantine_client_indices`: Malicious client IDs
- `poisoned_gradients`: Adversarial gradients

## 📈 Coverage Reports

### Generate HTML Coverage Report

```bash
pytest --cov=nawal --cov-report=html
# Open htmlcov/index.html in browser
```

### Coverage Requirements

- **Minimum Coverage**: 70% (enforced by pytest.ini)
- **Target Coverage**: 90%
- **Critical Paths**: 100% coverage recommended
  - Aggregation algorithms
  - Fitness calculation
  - Checkpoint save/load

### View Coverage

```bash
# Terminal output
pytest --cov=nawal --cov-report=term-missing

# Show uncovered lines
coverage report --show-missing
```

## 🐛 Debugging Tests

### Verbose Output

```bash
# Very verbose
pytest -vv

# Show print statements
pytest -s

# Both
pytest -vvs
```

### Run Single Test

```bash
pytest tests/test_genome.py::TestDNA::test_dna_initialization -vv
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Stop on First Failure

```bash
pytest -x
```

## ⚡ Performance Testing

### Benchmark Tests

```bash
# Run benchmarks
pytest -m benchmark

# With timing
pytest -m benchmark --durations=10
```

### Memory Profiling

```bash
# Requires pytest-memray
pytest --memray tests/test_training.py
```

### Parallel Execution

```bash
# Auto-detect CPUs
pytest -n auto

# Specific number of workers
pytest -n 4
```

## 🔐 Security Tests (Placeholder)

Placeholder tests for security module (to be expanded):

```bash
pytest -m security
```

Tests include:
- Byzantine client detection
- Model poisoning detection
- Gradient manipulation
- Privacy violation detection

## 🧩 Custom Markers

```python
@pytest.mark.slow
def test_long_evolution():
    # Long-running test
    pass

@pytest.mark.integration
def test_end_to_end():
    # Integration test
    pass

@pytest.mark.benchmark
def test_performance():
    # Performance benchmark
    pass

@pytest.mark.gpu
def test_gpu_training():
    # Requires GPU
    pass
```

## 📝 Writing New Tests

### Test Structure

```python
import pytest

class TestMyComponent:
    """Test MyComponent functionality."""
    
    def test_initialization(self, my_fixture):
        """Test component initialization."""
        component = MyComponent(my_fixture)
        assert component is not None
    
    def test_main_functionality(self):
        """Test main feature."""
        result = my_function()
        assert result == expected_value
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test expensive operation (marked as slow)."""
        # Long-running test
        pass
```

### Fixture Example

```python
@pytest.fixture
def my_fixture():
    """Create test data."""
    data = create_test_data()
    yield data
    # Cleanup (optional)
    cleanup(data)
```

## 🚨 Common Issues

### Import Errors

```bash
# Add nawal to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/belizechain"

# Or use pytest with --import-mode
pytest --import-mode=importlib
```

### Fixture Not Found

Ensure `conftest.py` is in the correct location:
- `nawal/tests/conftest.py` for test-specific fixtures
- `nawal/conftest.py` for package-wide fixtures

### Slow Tests

```bash
# Skip slow tests by default
pytest -m "not slow"

# Run only fast tests
pytest -m "not slow and not integration"
```

## 📊 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=nawal --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## 🎯 Next Steps

After Phase 2 (Security Module):
1. Expand security tests in `test_security.py`
2. Add differential privacy tests
3. Add secure aggregation tests
4. Add Byzantine detection tests

After Phase 3 (Data Management):
5. Add data loading tests
6. Add tokenization tests
7. Add preprocessing tests

After Phase 4 (Blockchain Integration):
8. Add blockchain integration tests
9. Add on-chain fitness submission tests
10. Add genome registry tests

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## ✅ Testing Checklist

- [ ] All tests pass locally
- [ ] Coverage > 70%
- [ ] No warnings or errors
- [ ] Integration tests pass
- [ ] Performance benchmarks acceptable
- [ ] Security tests pass (after security module)
- [ ] Documentation updated
- [ ] CI/CD passing

---

**Testing Infrastructure Status**: ✅ Complete (2,100+ lines)  
**Next**: Security Module → Data Management → Blockchain Integration
