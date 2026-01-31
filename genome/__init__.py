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
    Genome,
    GenomeEncoder,
    ArchitectureLayer,
    LayerType,
    Hyperparameters,
)

# Fitness evaluation
from .fitness import (
    FitnessEvaluator,
    FitnessScore,
    FitnessMetric,
    PoUWAlignment,
)

# Evolution operators
from .operators import (
    MutationOperator,
    MutationConfig,
    CrossoverOperator,
    CrossoverConfig,
    EvolutionStrategy,
    EvolutionConfig,
    MutationType,
    CrossoverType,
    SelectionStrategy,
)

# Population management
from .population import (
    PopulationManager,
    PopulationConfig,
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
    from .operators import CrossoverOperator, CrossoverConfig
    op = CrossoverOperator(CrossoverConfig())
    return op.crossover(parent1, parent2, generation)

def mutate(genome, generation: int | None = None):
    """Mutate a genome (stub - use MutationOperator.mutate)."""
    from .operators import MutationOperator, MutationConfig
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
    ModelBuilder,
    GenomeModel,
    LayerFactory,
)

# Nawal-specific genome adapter (NEW - connects to pure Nawal architecture)
from .nawal_adapter import (
    GenomeToNawalAdapter,
    NawalGenomeBuilder,
    genome_to_nawal,
    create_baseline_nawal_genome,
)

__version__ = "1.0.0"
__all__ = [
    # Encoding
    "Genome",
    "GenomeEncoder",
    "ArchitectureLayer",
    "LayerType",
    "Hyperparameters",
    # Fitness
    "FitnessEvaluator",
    "FitnessScore",
    "FitnessMetric",
    "PoUWAlignment",
    # Evolution
    "MutationOperator",
    "MutationConfig",
    "CrossoverOperator",
    "CrossoverConfig",
    "EvolutionStrategy",
    "EvolutionConfig",
    "MutationType",
    "CrossoverType",
    "SelectionStrategy",
    # Population
    "PopulationManager",
    "PopulationConfig",
    "PopulationStatistics",
    "Population",
    # History
    "EvolutionHistory",
    "GenerationRecord",
    "GenomeLineage",
    # Model Building
    "ModelBuilder",
    "GenomeModel",
    "LayerFactory",
    # Nawal Integration (NEW)
    "GenomeToNawalAdapter",
    "NawalGenomeBuilder",
    "genome_to_nawal",
    "create_baseline_nawal_genome",
    # Stub functions
    "select_parents",
    "crossover",
    "mutate",
    "FitnessMetric",
    "PoUWAlignment",
    # Operators
    "MutationOperator",
    "MutationConfig",
    "CrossoverOperator",
    "CrossoverConfig",
    "EvolutionStrategy",
    "EvolutionConfig",
    "MutationType",
    "CrossoverType",
    "SelectionStrategy",
    # Population
    "PopulationManager",
    "PopulationConfig",
    "PopulationStatistics",
    # History
    "EvolutionHistory",
    "GenerationRecord",
    "GenomeLineage",
    # Model Builder
    "ModelBuilder",
    "GenomeModel",
    "LayerFactory",
]
