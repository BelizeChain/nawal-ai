"""
Nawal - Federated AI with Genomic Evolution

A privacy-preserving federated learning system with evolutionary AI.
"""

# Core genome system
from nawal.genome import (
    Genome,
    GenomeEncoder,
    ModelBuilder,
    GenomeModel,
    LayerFactory,
    # Nawal-specific genome adapter
    NawalGenomeBuilder,
    genome_to_nawal,
)

# Federated server
from nawal.server import (
    FederatedAggregator,
    ModelUpdate,
    TrainingMetrics,
)

# Training client
from nawal.client import (
    GenomeTrainer,
    TrainingConfig,
)

# Configuration
from nawal.config import (
    NawalConfig,
    EvolutionConfig,
    FederatedConfig,
    TrainingConfig as ConfigTrainingConfig,
    ModelConfig,
    ComplianceConfig,
    StorageConfig,
    load_config,
)

# Orchestrator
from nawal.orchestrator import (
    EvolutionOrchestrator,
    GenerationState,
)

__all__ = [
    # Genome system
    "Genome",
    "GenomeEncoder",
    "ModelBuilder",
    "GenomeModel",
    "LayerFactory",
    # Server
    "FederatedAggregator",
    "ModelUpdate",
    "TrainingMetrics",
    # Client
    "GenomeTrainer",
    "TrainingConfig",
    # Configuration
    "NawalConfig",
    "EvolutionConfig",
    "FederatedConfig",
    "ConfigTrainingConfig",
    "ModelConfig",
    "ComplianceConfig",
    "StorageConfig",
    "load_config",
    # Orchestrator
    "EvolutionOrchestrator",
    "GenerationState",
]
