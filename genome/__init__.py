"""
Nawal Genome Evolution System

This module implements the genetic encoding and evolution of AI model architectures.
The genome system allows Nawal to evolve and improve over time through natural selection.

Latest Technologies:
- Python 3.13+ (latest type hints, pattern matching)
- Pydantic v2 for validation
- Modern async/await patterns
- Protocol-based interfaces for extensibility

Author: BelizeChain Development Team
Date: October 2025
Version: 1.0.0
"""

# Core encoding
from .encoding import (
    ArchitectureLayer,
    Genome,
    GenomeEncoder,
    Hyperparameters,
    LayerType,
)

# Fitness evaluation
from .fitness import (
    FitnessEvaluator,
    FitnessMetric,
    FitnessScore,
    PoUWAlignment,
)

# Evolution operators
from .operators import (
    CrossoverConfig,
    CrossoverOperator,
    CrossoverType,
    EvolutionConfig,
    EvolutionStrategy,
    MutationConfig,
    MutationOperator,
    MutationType,
    SelectionStrategy,
)

# Population management
from .population import (
    PopulationConfig,
    PopulationManager,
    PopulationStatistics,
)

# Alias for backward compatibility
Population = PopulationManager


# Stub functions for backward compatibility with orchestrator
def select_parents(population, count: int):
    """Select parents from population (stub - use PopulationManager.select_parents)."""
    return population.select_parents(count)


def crossover(parent1, parent2, generation: int = 0):
    """Crossover two genomes (stub - use CrossoverOperator.crossover)."""
    from .operators import CrossoverConfig, CrossoverOperator

    op = CrossoverOperator(CrossoverConfig())
    return op.crossover(parent1, parent2, generation)


def mutate(genome, generation: int | None = None):
    """Mutate a genome (stub - use MutationOperator.mutate)."""
    from .operators import MutationConfig, MutationOperator

    op = MutationOperator(MutationConfig())
    gen = generation if generation is not None else genome.generation + 1
    return op.mutate(genome, gen)


# Evolution history
from .history import (
    EvolutionHistory,
    GenerationRecord,
    GenomeLineage,
)

# Model builder - now using full implementation
from .model_builder import (
    GenomeModel,
    LayerFactory,
    ModelBuilder,
)

# Nawal-specific genome adapter (NEW - connects to pure Nawal architecture)
from .nawal_adapter import (
    GenomeToNawalAdapter,
    NawalGenomeBuilder,
    create_baseline_nawal_genome,
    genome_to_nawal,
)

__version__ = "1.0.0"
__all__ = [
    "ArchitectureLayer",
    "CrossoverConfig",
    "CrossoverConfig",
    "CrossoverOperator",
    "CrossoverOperator",
    "CrossoverType",
    "CrossoverType",
    "EvolutionConfig",
    "EvolutionConfig",
    # History
    "EvolutionHistory",
    # History
    "EvolutionHistory",
    "EvolutionStrategy",
    "EvolutionStrategy",
    # Fitness
    "FitnessEvaluator",
    "FitnessMetric",
    "FitnessMetric",
    "FitnessScore",
    "GenerationRecord",
    "GenerationRecord",
    # Encoding
    "Genome",
    "GenomeEncoder",
    "GenomeLineage",
    "GenomeLineage",
    "GenomeModel",
    "GenomeModel",
    # Nawal Integration (NEW)
    "GenomeToNawalAdapter",
    "Hyperparameters",
    "LayerFactory",
    "LayerFactory",
    "LayerType",
    # Model Building
    "ModelBuilder",
    # Model Builder
    "ModelBuilder",
    "MutationConfig",
    "MutationConfig",
    # Evolution
    "MutationOperator",
    # Operators
    "MutationOperator",
    "MutationType",
    "MutationType",
    "NawalGenomeBuilder",
    "PoUWAlignment",
    "PoUWAlignment",
    "Population",
    "PopulationConfig",
    "PopulationConfig",
    # Population
    "PopulationManager",
    # Population
    "PopulationManager",
    "PopulationStatistics",
    "PopulationStatistics",
    "SelectionStrategy",
    "SelectionStrategy",
    "create_baseline_nawal_genome",
    "crossover",
    "genome_to_nawal",
    "mutate",
    # Stub functions
    "select_parents",
]
