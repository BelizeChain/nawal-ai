"""
Pytest configuration and shared fixtures for Nawal AI tests.

This module provides reusable test fixtures for:
- Temporary directories and file management
- Mock configurations
- Sample genomes and populations
- PyTorch models and dataloaders
- Federated learning components

Author: BelizeChain Team
License: MIT
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nawal.genome.dna import DNA, LayerGene, ConnectionGene
from nawal.genome.population import Population
from nawal.genome.history import InnovationHistory
from nawal.config import (
    GenomeConfig,
    EvolutionConfig,
    TrainingConfig,
    FederatedConfig,
    NawalConfig,
)


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a checkpoint directory."""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


@pytest.fixture
def data_dir(temp_dir):
    """Create a data directory."""
    data_path = temp_dir / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def genome_config():
    """Create a minimal genome configuration for testing."""
    return GenomeConfig(
        input_size=10,
        output_size=2,
        hidden_layers=[16, 8],
        activation="relu",
        dropout=0.1,
        max_layers=5,
        max_neurons=32,
    )


@pytest.fixture
def evolution_config():
    """Create evolution configuration for testing."""
    return EvolutionConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=2,
        tournament_size=3,
        fitness_threshold=0.8,
    )


@pytest.fixture
def training_config():
    """Create training configuration for testing."""
    return TrainingConfig(
        batch_size=8,
        learning_rate=0.001,
        epochs=2,
        optimizer="adam",
        loss_function="cross_entropy",
        device="cpu",
        gradient_clip=1.0,
    )


@pytest.fixture
def federated_config():
    """Create federated learning configuration for testing."""
    return FederatedConfig(
        num_clients=3,
        aggregation_method="fedavg",
        min_clients=2,
        client_fraction=1.0,
        local_epochs=1,
        differential_privacy=False,
        secure_aggregation=False,
    )


@pytest.fixture
def nawal_config(genome_config, evolution_config, training_config, federated_config):
    """Create complete Nawal configuration for testing."""
    return NawalConfig(
        genome=genome_config,
        evolution=evolution_config,
        training=training_config,
        federated=federated_config,
        checkpoint_dir="checkpoints",
        log_level="INFO",
    )


# ============================================================================
# Genome Fixtures
# ============================================================================

@pytest.fixture
def sample_layer_genes():
    """Create sample layer genes for testing."""
    return [
        LayerGene(
            innovation_id=1,
            layer_type="linear",
            params={"in_features": 10, "out_features": 16},
            enabled=True,
        ),
        LayerGene(
            innovation_id=2,
            layer_type="relu",
            params={},
            enabled=True,
        ),
        LayerGene(
            innovation_id=3,
            layer_type="linear",
            params={"in_features": 16, "out_features": 2},
            enabled=True,
        ),
    ]


@pytest.fixture
def sample_connection_genes():
    """Create sample connection genes for testing."""
    return [
        ConnectionGene(
            innovation_id=101,
            source_layer=0,
            target_layer=1,
            enabled=True,
        ),
        ConnectionGene(
            innovation_id=102,
            source_layer=1,
            target_layer=2,
            enabled=True,
        ),
    ]


@pytest.fixture
def sample_dna(sample_layer_genes, sample_connection_genes):
    """Create a sample DNA instance for testing."""
    dna = DNA(input_size=10, output_size=2)
    dna.layer_genes = sample_layer_genes
    dna.connection_genes = sample_connection_genes
    return dna


@pytest.fixture
def innovation_history():
    """Create an innovation history instance."""
    return InnovationHistory()


@pytest.fixture
def sample_population(genome_config, evolution_config, innovation_history):
    """Create a sample population for testing."""
    from nawal.genome.population import PopulationConfig, PopulationManager
    from nawal.genome.encoding import Genome, ArchitectureLayer, LayerType
    
    # Create population config from evolution config
    pop_config = PopulationConfig(
        target_size=evolution_config.population_size,
        max_size=evolution_config.population_size + 10,
        min_size=max(2, evolution_config.population_size // 2),
    )
    
    population = PopulationManager(config=pop_config)
    
    # Get hidden size from genome config or use default
    hidden_size = getattr(genome_config, 'hidden_size', 64)
    if hasattr(genome_config, 'hidden_layers') and genome_config.hidden_layers:
        hidden_size = genome_config.hidden_layers[0]
    
    # Initialize with some genomes that have encoder layers
    for i in range(pop_config.target_size):
        genome = Genome(
            genome_id=f"test_genome_{i}",
            hidden_size=hidden_size,
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=hidden_size,
                    num_heads=8,
                    dropout_rate=0.1,
                ),
            ],
        )
        population.add_genome(genome)
    
    # Add legacy attributes for backward compatibility
    population.genome_config = genome_config
    population.evolution_config = evolution_config
    population.innovation_history = innovation_history
    population.generation = 0
    population.genomes = list(population.genomes.values())
    
    # Add legacy methods
    def initialize():
        pass
    
    def tournament_selection(tournament_size=3):
        if len(population.genomes) == 0:
            return None
        import random
        return random.choice(population.genomes)
    
    def evolve():
        population.generation += 1
    
    def get_best_genome():
        """Get genome with highest fitness_score."""
        valid_genomes = [g for g in population.genomes if g.fitness_score is not None]
        if not valid_genomes:
            return None
        return max(valid_genomes, key=lambda g: g.fitness_score)
    
    population.initialize = initialize
    population.tournament_selection = tournament_selection
    population.evolve = evolve
    population.get_best_genome = get_best_genome
    
    return population


# ============================================================================
# PyTorch Fixtures
# ============================================================================

@pytest.fixture
def sample_model():
    """Create a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def sample_dataloader(sample_dataset):
    """Create a sample dataloader for testing."""
    return DataLoader(sample_dataset, batch_size=8, shuffle=True)


@pytest.fixture
def train_val_dataloaders(sample_dataset):
    """Create train and validation dataloaders."""
    train_size = int(0.8 * len(sample_dataset))
    val_size = len(sample_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        sample_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader


# ============================================================================
# Federated Learning Fixtures
# ============================================================================

@pytest.fixture
def client_models():
    """Create multiple client models for federated learning."""
    return [
        nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        for _ in range(3)
    ]


@pytest.fixture
def client_dataloaders():
    """Create dataloaders for multiple clients."""
    dataloaders = []
    for _ in range(3):
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        dataloaders.append(loader)
    return dataloaders


@pytest.fixture
def model_state_dicts(client_models):
    """Get state dicts from client models."""
    return [model.state_dict() for model in client_models]


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def mock_fitness_scores():
    """Generate mock fitness scores for testing."""
    return [0.85, 0.72, 0.91, 0.68, 0.79, 0.88, 0.75, 0.82, 0.77, 0.89]


@pytest.fixture
def device():
    """Get the device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    return 42


# ============================================================================
# Mock Response Fixtures
# ============================================================================

@pytest.fixture
def mock_blockchain_response():
    """Mock blockchain response for testing."""
    return {
        "validator_id": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "stake": 100000,
        "fitness_submitted": True,
        "transaction_hash": "0x1234567890abcdef",
        "block_number": 12345,
    }


@pytest.fixture
def mock_ipfs_response():
    """Mock IPFS response for testing."""
    return {
        "Hash": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
        "Size": 1024,
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_params():
    """Parameters for benchmark testing."""
    return {
        "population_sizes": [10, 50, 100],
        "genome_sizes": [10, 50, 100],
        "training_epochs": [1, 5, 10],
        "batch_sizes": [8, 32, 128],
    }


# ============================================================================
# Error Injection Fixtures
# ============================================================================

@pytest.fixture
def byzantine_client_indices():
    """Indices of Byzantine clients for adversarial testing."""
    return [1, 3]  # Clients 1 and 3 are malicious


@pytest.fixture
def poisoned_gradients():
    """Create poisoned gradients for adversarial testing."""
    # Normal gradients
    normal = {
        "weight": torch.randn(16, 10) * 0.01,
        "bias": torch.randn(16) * 0.01,
    }
    
    # Poisoned gradients (10x larger)
    poisoned = {
        "weight": torch.randn(16, 10) * 0.1,
        "bias": torch.randn(16) * 0.1,
    }
    
    return {"normal": normal, "poisoned": poisoned}


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    skip_slow = pytest.mark.skip(reason="slow test (use --runslow to run)")
    skip_gpu = pytest.mark.skip(reason="requires GPU")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow", default=False):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
