"""
Genome Evolution Operators - Mutation & Crossover

This module implements genetic operators for evolving neural architectures:
- Mutation: Modify genome parameters to explore new architectures
- Crossover: Combine parent genomes to create offspring
- Evolution strategies: Control evolution process

Aligned with BelizeChain's PoUW consensus mechanism.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, Any
from loguru import logger

from .encoding import (
    Genome,
    GenomeEncoder,
    ArchitectureLayer,
    LayerType,
    Hyperparameters,
)


# =============================================================================
# Enums & Configuration
# =============================================================================


class MutationType(Enum):
    """Types of mutations that can occur to a genome."""
    
    # Layer-level mutations
    ADD_LAYER = auto()           # Add new layer to architecture
    REMOVE_LAYER = auto()        # Remove existing layer
    REPLACE_LAYER = auto()       # Replace layer with different type
    MODIFY_LAYER = auto()        # Modify layer parameters
    
    # Hyperparameter mutations
    LEARNING_RATE = auto()       # Mutate learning rate
    BATCH_SIZE = auto()          # Mutate batch size
    OPTIMIZER = auto()           # Change optimizer
    PRECISION = auto()           # Change training precision
    
    # Architecture-level mutations
    ADD_SKIP_CONNECTION = auto()  # Add residual connection
    ADD_ATTENTION = auto()       # Add attention mechanism
    ADD_MOE = auto()             # Add Mixture of Experts
    ADD_SSM = auto()             # Add State Space Model


class CrossoverType(Enum):
    """Types of crossover operations between parent genomes."""
    
    UNIFORM = auto()             # Random selection from each parent
    SINGLE_POINT = auto()        # Split at single point
    TWO_POINT = auto()           # Split at two points
    LAYER_WISE = auto()          # Exchange complete layers
    HYPERPARAMETER = auto()      # Exchange hyperparameters only


class SelectionStrategy(Enum):
    """Strategies for selecting parents from population."""
    
    TOURNAMENT = auto()          # Tournament selection (k individuals)
    ROULETTE = auto()            # Fitness-proportional selection
    RANK = auto()                # Rank-based selection
    ELITE = auto()               # Select top performers only


# =============================================================================
# Mutation Operators
# =============================================================================


@dataclass
class MutationConfig:
    """Configuration for mutation operations."""
    
    # Mutation rates (probability of each mutation type)
    add_layer_rate: float = 0.1
    remove_layer_rate: float = 0.05
    replace_layer_rate: float = 0.15
    modify_layer_rate: float = 0.35
    
    hyperparameter_rate: float = 0.25
    architecture_rate: float = 0.1
    
    # Constraints
    min_layers: int = 2
    max_layers: int = 24
    max_mutations_per_genome: int = 3
    
    # Layer type preferences (for adding new layers)
    prefer_modern_layers: bool = True  # Prefer MoE, SSM, Flash Attention
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration."""
        errors = []
        
        if self.min_layers < 1:
            errors.append("min_layers must be >= 1")
        
        if self.max_layers < self.min_layers:
            errors.append("max_layers must be >= min_layers")
        
        total_rate = (
            self.add_layer_rate + self.remove_layer_rate +
            self.replace_layer_rate + self.modify_layer_rate +
            self.hyperparameter_rate + self.architecture_rate
        )
        
        if not (0.99 <= total_rate <= 1.01):
            errors.append(f"Mutation rates should sum to ~1.0 (got {total_rate})")
        
        return (len(errors) == 0, errors)


class MutationOperator:
    """Applies mutations to genomes to create genetic variation."""
    
    def __init__(self, config: MutationConfig | None = None):
        """
        Initialize mutation operator.
        
        Args:
            config: Mutation configuration (uses defaults if None)
        """
        self.config = config or MutationConfig()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid mutation config: {', '.join(errors)}")
        
        logger.info(
            "Initialized MutationOperator",
            add_rate=self.config.add_layer_rate,
            remove_rate=self.config.remove_layer_rate,
            modify_rate=self.config.modify_layer_rate,
        )
    
    def mutate(self, genome: Genome, generation: int) -> Genome:
        """
        Apply mutations to a genome.
        
        Args:
            genome: Parent genome to mutate
            generation: Current generation number
        
        Returns:
            Mutated genome (child)
        """
        # Clone genome to create child
        child = genome.clone()
        child.generation = generation
        child.parent_genomes = [genome.genome_id]
        
        # Determine number of mutations
        num_mutations = random.randint(1, self.config.max_mutations_per_genome)
        
        mutations_applied = []
        
        for _ in range(num_mutations):
            mutation_type = self._select_mutation_type()
            
            try:
                match mutation_type:
                    case MutationType.ADD_LAYER:
                        self._mutate_add_layer(child)
                        mutations_applied.append("add_layer")
                    
                    case MutationType.REMOVE_LAYER:
                        self._mutate_remove_layer(child)
                        mutations_applied.append("remove_layer")
                    
                    case MutationType.REPLACE_LAYER:
                        self._mutate_replace_layer(child)
                        mutations_applied.append("replace_layer")
                    
                    case MutationType.MODIFY_LAYER:
                        self._mutate_modify_layer(child)
                        mutations_applied.append("modify_layer")
                    
                    case MutationType.LEARNING_RATE:
                        self._mutate_learning_rate(child)
                        mutations_applied.append("learning_rate")
                    
                    case MutationType.BATCH_SIZE:
                        self._mutate_batch_size(child)
                        mutations_applied.append("batch_size")
                    
                    case MutationType.OPTIMIZER:
                        self._mutate_optimizer(child)
                        mutations_applied.append("optimizer")
                    
                    case MutationType.PRECISION:
                        self._mutate_precision(child)
                        mutations_applied.append("precision")
                    
                    case MutationType.ADD_ATTENTION:
                        self._mutate_add_attention(child)
                        mutations_applied.append("add_attention")
                    
                    case MutationType.ADD_MOE:
                        self._mutate_add_moe(child)
                        mutations_applied.append("add_moe")
                    
                    case MutationType.ADD_SSM:
                        self._mutate_add_ssm(child)
                        mutations_applied.append("add_ssm")
                    
                    case _:
                        logger.warning(f"Unknown mutation type: {mutation_type}")
            
            except Exception as e:
                logger.error(
                    f"Mutation failed: {mutation_type}",
                    error=str(e),
                    genome_id=child.genome_id,
                )
        
        # Reset fitness scores (child needs evaluation)
        child.fitness_score = None
        child.quality_score = None
        child.timeliness_score = None
        child.honesty_score = None
        
        logger.info(
            "Genome mutated",
            parent_id=genome.genome_id,
            child_id=child.genome_id,
            generation=generation,
            mutations=mutations_applied,
        )
        
        return child
    
    def _select_mutation_type(self) -> MutationType:
        """Select mutation type based on configured probabilities."""
        roll = random.random()
        cumulative = 0.0
        
        # Layer mutations
        cumulative += self.config.add_layer_rate
        if roll < cumulative:
            return MutationType.ADD_LAYER
        
        cumulative += self.config.remove_layer_rate
        if roll < cumulative:
            return MutationType.REMOVE_LAYER
        
        cumulative += self.config.replace_layer_rate
        if roll < cumulative:
            return MutationType.REPLACE_LAYER
        
        cumulative += self.config.modify_layer_rate
        if roll < cumulative:
            return MutationType.MODIFY_LAYER
        
        # Hyperparameter mutations (distributed)
        cumulative += self.config.hyperparameter_rate
        if roll < cumulative:
            return random.choice([
                MutationType.LEARNING_RATE,
                MutationType.BATCH_SIZE,
                MutationType.OPTIMIZER,
                MutationType.PRECISION,
            ])
        
        # Architecture mutations (distributed)
        return random.choice([
            MutationType.ADD_ATTENTION,
            MutationType.ADD_MOE,
            MutationType.ADD_SSM,
        ])
    
    def _mutate_add_layer(self, genome: Genome) -> None:
        """Add a new layer to the genome."""
        total_layers = len(genome.encoder_layers) + len(genome.decoder_layers)
        
        if total_layers >= self.config.max_layers:
            logger.debug("Cannot add layer: max layers reached")
            return
        
        # Choose whether to add to encoder or decoder
        add_to_encoder = random.choice([True, False])
        layers = genome.encoder_layers if add_to_encoder else genome.decoder_layers
        
        # Select layer type
        if self.config.prefer_modern_layers and random.random() < 0.6:
            # Prefer modern architectures
            layer_type = random.choice([
                LayerType.MIXTURE_OF_EXPERTS,
                LayerType.STATE_SPACE_MODEL,
                LayerType.FLASH_ATTENTION,
                LayerType.MULTIHEAD_ATTENTION,
            ])
        else:
            # Any layer type
            layer_type = random.choice(list(LayerType))
        
        # Create new layer
        hidden_size = genome.hyperparameters.hidden_size
        new_layer = ArchitectureLayer(
            layer_id=f"layer_{len(layers)}_{random.randint(1000, 9999)}",
            layer_type=layer_type,
            input_size=hidden_size,
            output_size=hidden_size,
            hidden_size=hidden_size,
            parameters={},
        )
        
        # Add layer at random position
        insert_pos = random.randint(0, len(layers))
        layers.insert(insert_pos, new_layer)
    
    def _mutate_remove_layer(self, genome: Genome) -> None:
        """Remove a layer from the genome."""
        total_layers = len(genome.encoder_layers) + len(genome.decoder_layers)
        
        if total_layers <= self.config.min_layers:
            logger.debug("Cannot remove layer: min layers reached")
            return
        
        # Choose whether to remove from encoder or decoder
        remove_from_encoder = random.choice([True, False])
        layers = genome.encoder_layers if remove_from_encoder else genome.decoder_layers
        
        if len(layers) == 0:
            return
        
        # Remove random layer
        remove_idx = random.randint(0, len(layers) - 1)
        del layers[remove_idx]
    
    def _mutate_replace_layer(self, genome: Genome) -> None:
        """Replace a layer with a different type."""
        # Choose encoder or decoder
        use_encoder = random.choice([True, False])
        layers = genome.encoder_layers if use_encoder else genome.decoder_layers
        
        if len(layers) == 0:
            return
        
        # Select layer to replace
        replace_idx = random.randint(0, len(layers) - 1)
        old_layer = layers[replace_idx]
        
        # Select new layer type (different from current)
        available_types = [lt for lt in LayerType if lt != old_layer.layer_type]
        new_type = random.choice(available_types)
        
        # Create replacement layer
        new_layer = ArchitectureLayer(
            layer_id=f"layer_{replace_idx}_{random.randint(1000, 9999)}",
            layer_type=new_type,
            input_size=old_layer.input_size,
            output_size=old_layer.output_size,
            hidden_size=old_layer.hidden_size,
            parameters={},
        )
        
        layers[replace_idx] = new_layer
    
    def _mutate_modify_layer(self, genome: Genome) -> None:
        """Modify parameters of an existing layer."""
        # Choose encoder or decoder
        use_encoder = random.choice([True, False])
        layers = genome.encoder_layers if use_encoder else genome.decoder_layers
        
        if len(layers) == 0:
            return
        
        # Select layer to modify
        modify_idx = random.randint(0, len(layers) - 1)
        layer = layers[modify_idx]
        
        # Modify size parameters (±20%)
        if random.random() < 0.5 and layer.hidden_size is not None:
            multiplier = random.uniform(0.8, 1.2)
            layer.hidden_size = max(64, int(layer.hidden_size * multiplier))
        
        # Modify type-specific parameters
        if hasattr(layer, 'num_heads') and layer.num_heads:
            layer.num_heads = random.choice([4, 8, 12, 16])
        
        if hasattr(layer, 'num_experts') and layer.num_experts:
            layer.num_experts = random.choice([4, 8, 16, 32])
    
    def _mutate_learning_rate(self, genome: Genome) -> None:
        """Mutate learning rate (±50%)."""
        current_lr = genome.hyperparameters.learning_rate
        multiplier = random.uniform(0.5, 1.5)
        new_lr = max(1e-6, min(1e-2, current_lr * multiplier))
        genome.hyperparameters.learning_rate = new_lr
    
    def _mutate_batch_size(self, genome: Genome) -> None:
        """Mutate batch size."""
        current_batch = genome.hyperparameters.batch_size
        
        # Batch sizes are typically powers of 2
        possible_batches = [8, 16, 32, 64, 128, 256]
        possible_batches = [b for b in possible_batches if b != current_batch]
        
        genome.hyperparameters.batch_size = random.choice(possible_batches)
    
    def _mutate_optimizer(self, genome: Genome) -> None:
        """Change optimizer."""
        optimizers = ["adam", "adamw", "lion", "sophia", "sgd"]
        current_opt = genome.hyperparameters.optimizer
        
        available = [opt for opt in optimizers if opt != current_opt]
        genome.hyperparameters.optimizer = random.choice(available)
    
    def _mutate_precision(self, genome: Genome) -> None:
        """Change training precision."""
        precisions = ["fp32", "fp16", "bf16", "fp8"]
        current_precision = genome.hyperparameters.precision
        
        available = [p for p in precisions if p != current_precision]
        genome.hyperparameters.precision = random.choice(available)
    
    def _mutate_add_attention(self, genome: Genome) -> None:
        """Add attention mechanism to a layer."""
        from nawal.genome import LayerType
        
        # Try to add attention to encoder layers
        if genome.encoder_layers:
            # Check if transformer encoder already exists
            has_transformer = any(
                layer.layer_type == LayerType.TRANSFORMER_ENCODER 
                for layer in genome.encoder_layers
            )
            if not has_transformer:
                # Add transformer encoder layer
                self._mutate_add_layer(genome)
        else:
            # Fallback to standard add layer
            self._mutate_add_layer(genome)
    
    def _mutate_add_moe(self, genome: Genome) -> None:
        """Add Mixture of Experts layer."""
        from nawal.genome import LayerType, ArchitectureLayer
        
        total_layers = len(genome.encoder_layers) + len(genome.decoder_layers)
        if total_layers >= self.config.max_layers:
            logger.debug("Cannot add MoE: max layers reached")
            return
        
        # Add MoE to encoder
        moe_layer = ArchitectureLayer(
            layer_type=LayerType.MIXTURE_OF_EXPERTS,
            hidden_size=genome.hidden_size,
            num_experts=8,
            expert_capacity=64,
        )
        genome.encoder_layers.append(moe_layer)
    
    def _mutate_add_ssm(self, genome: Genome) -> None:
        """Add State Space Model layer."""
        from nawal.genome import LayerType, ArchitectureLayer
        
        total_layers = len(genome.encoder_layers) + len(genome.decoder_layers)
        if total_layers >= self.config.max_layers:
            logger.debug("Cannot add SSM: max layers reached")
            return
        
        # Add SSM layer (e.g., Mamba/S4)
        ssm_layer = ArchitectureLayer(
            layer_type=LayerType.LINEAR,  # Simplified as LINEAR for now
            hidden_size=genome.hidden_size,
            input_size=genome.hidden_size,
            output_size=genome.hidden_size,
        )
        genome.encoder_layers.append(ssm_layer)


# =============================================================================
# Crossover Operators
# =============================================================================


@dataclass
class CrossoverConfig:
    """Configuration for crossover operations."""
    
    # Crossover type probabilities
    uniform_rate: float = 0.3
    single_point_rate: float = 0.2
    two_point_rate: float = 0.2
    layer_wise_rate: float = 0.2
    hyperparameter_rate: float = 0.1
    
    # Configuration
    preserve_architecture_size: bool = False  # Keep similar layer counts
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration."""
        errors = []
        
        total_rate = (
            self.uniform_rate + self.single_point_rate +
            self.two_point_rate + self.layer_wise_rate +
            self.hyperparameter_rate
        )
        
        if not (0.99 <= total_rate <= 1.01):
            errors.append(f"Crossover rates should sum to ~1.0 (got {total_rate})")
        
        return (len(errors) == 0, errors)


class CrossoverOperator:
    """Combines parent genomes to create offspring."""
    
    def __init__(self, config: CrossoverConfig | None = None):
        """
        Initialize crossover operator.
        
        Args:
            config: Crossover configuration (uses defaults if None)
        """
        self.config = config or CrossoverConfig()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid crossover config: {', '.join(errors)}")
        
        logger.info(
            "Initialized CrossoverOperator",
            uniform_rate=self.config.uniform_rate,
            single_point_rate=self.config.single_point_rate,
        )
    
    def crossover(
        self,
        parent1: Genome,
        parent2: Genome,
        generation: int,
    ) -> Genome:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            generation: Current generation number
        
        Returns:
            Child genome combining traits from both parents
        """
        # Select crossover type
        crossover_type = self._select_crossover_type()
        
        # Perform crossover
        match crossover_type:
            case CrossoverType.UNIFORM:
                child = self._uniform_crossover(parent1, parent2)
            
            case CrossoverType.SINGLE_POINT:
                child = self._single_point_crossover(parent1, parent2)
            
            case CrossoverType.TWO_POINT:
                child = self._two_point_crossover(parent1, parent2)
            
            case CrossoverType.LAYER_WISE:
                child = self._layer_wise_crossover(parent1, parent2)
            
            case CrossoverType.HYPERPARAMETER:
                child = self._hyperparameter_crossover(parent1, parent2)
            
            case _:
                # Default to uniform
                child = self._uniform_crossover(parent1, parent2)
        
        # Set child metadata
        child.generation = generation
        child.parent_genomes = [parent1.genome_id, parent2.genome_id]
        
        # Reset fitness scores
        child.fitness_score = None
        child.quality_score = None
        child.timeliness_score = None
        child.honesty_score = None
        
        logger.info(
            "Crossover performed",
            parent1_id=parent1.genome_id,
            parent2_id=parent2.genome_id,
            child_id=child.genome_id,
            crossover_type=crossover_type.name,
            generation=generation,
        )
        
        return child
    
    def _select_crossover_type(self) -> CrossoverType:
        """Select crossover type based on configured probabilities."""
        roll = random.random()
        cumulative = 0.0
        
        cumulative += self.config.uniform_rate
        if roll < cumulative:
            return CrossoverType.UNIFORM
        
        cumulative += self.config.single_point_rate
        if roll < cumulative:
            return CrossoverType.SINGLE_POINT
        
        cumulative += self.config.two_point_rate
        if roll < cumulative:
            return CrossoverType.TWO_POINT
        
        cumulative += self.config.layer_wise_rate
        if roll < cumulative:
            return CrossoverType.LAYER_WISE
        
        return CrossoverType.HYPERPARAMETER
    
    def _uniform_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Uniform crossover: randomly select each component from either parent."""
        child = parent1.clone()
        
        # Crossover encoder layers
        min_encoder_len = min(len(parent1.encoder_layers), len(parent2.encoder_layers))
        for i in range(min_encoder_len):
            if random.random() < 0.5:
                child.encoder_layers[i] = parent2.encoder_layers[i]
        
        # Crossover decoder layers
        min_decoder_len = min(len(parent1.decoder_layers), len(parent2.decoder_layers))
        for i in range(min_decoder_len):
            if random.random() < 0.5:
                child.decoder_layers[i] = parent2.decoder_layers[i]
        
        # Crossover hyperparameters
        if random.random() < 0.5:
            child.hyperparameters.learning_rate = parent2.hyperparameters.learning_rate
        if random.random() < 0.5:
            child.hyperparameters.batch_size = parent2.hyperparameters.batch_size
        if random.random() < 0.5:
            child.hyperparameters.optimizer = parent2.hyperparameters.optimizer
        
        return child
    
    def _single_point_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Single-point crossover: split at one point."""
        child = parent1.clone()
        
        # Split encoder layers
        if len(parent1.encoder_layers) > 0 and len(parent2.encoder_layers) > 0:
            split_point = random.randint(0, min(
                len(parent1.encoder_layers),
                len(parent2.encoder_layers)
            ))
            child.encoder_layers = (
                parent1.encoder_layers[:split_point] +
                parent2.encoder_layers[split_point:]
            )
        
        return child
    
    def _two_point_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Two-point crossover: exchange middle segment."""
        child = parent1.clone()
        
        if len(parent1.encoder_layers) > 1 and len(parent2.encoder_layers) > 1:
            min_len = min(len(parent1.encoder_layers), len(parent2.encoder_layers))
            point1 = random.randint(0, min_len - 1)
            point2 = random.randint(point1, min_len)
            
            child.encoder_layers = (
                parent1.encoder_layers[:point1] +
                parent2.encoder_layers[point1:point2] +
                parent1.encoder_layers[point2:]
            )
        
        return child
    
    def _layer_wise_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Layer-wise crossover: exchange complete layer types."""
        child = parent1.clone()
        
        # Exchange layers of specific types
        for i, layer in enumerate(child.encoder_layers):
            if i < len(parent2.encoder_layers):
                if layer.layer_type == parent2.encoder_layers[i].layer_type:
                    if random.random() < 0.5:
                        child.encoder_layers[i] = parent2.encoder_layers[i]
        
        return child
    
    def _hyperparameter_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Crossover only hyperparameters, keep architecture from one parent."""
        # Choose which parent's architecture to keep
        if random.random() < 0.5:
            child = parent1.clone()
            other = parent2
        else:
            child = parent2.clone()
            other = parent1
        
        # Mix hyperparameters
        child.hyperparameters.learning_rate = (
            parent1.hyperparameters.learning_rate + parent2.hyperparameters.learning_rate
        ) / 2
        
        child.hyperparameters.batch_size = random.choice([
            parent1.hyperparameters.batch_size,
            parent2.hyperparameters.batch_size,
        ])
        
        child.hyperparameters.optimizer = random.choice([
            parent1.hyperparameters.optimizer,
            parent2.hyperparameters.optimizer,
        ])
        
        return child


# =============================================================================
# Evolution Strategy
# =============================================================================


@dataclass
class EvolutionConfig:
    """Complete evolution strategy configuration."""
    
    mutation_config: MutationConfig = field(default_factory=MutationConfig)
    crossover_config: CrossoverConfig = field(default_factory=CrossoverConfig)
    
    # Evolution strategy
    mutation_rate: float = 0.3       # Probability of mutation
    crossover_rate: float = 0.7      # Probability of crossover
    
    elitism_count: int = 2           # Number of elite genomes to preserve
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate complete configuration."""
        errors = []
        
        if not (0.0 <= self.mutation_rate <= 1.0):
            errors.append("mutation_rate must be in [0, 1]")
        
        if not (0.0 <= self.crossover_rate <= 1.0):
            errors.append("crossover_rate must be in [0, 1]")
        
        if self.elitism_count < 0:
            errors.append("elitism_count must be >= 0")
        
        # Validate sub-configs
        valid, sub_errors = self.mutation_config.validate()
        if not valid:
            errors.extend([f"Mutation config: {e}" for e in sub_errors])
        
        valid, sub_errors = self.crossover_config.validate()
        if not valid:
            errors.extend([f"Crossover config: {e}" for e in sub_errors])
        
        return (len(errors) == 0, errors)


class EvolutionStrategy:
    """Orchestrates genome evolution using mutation and crossover."""
    
    def __init__(self, config: EvolutionConfig | None = None):
        """
        Initialize evolution strategy.
        
        Args:
            config: Evolution configuration (uses defaults if None)
        """
        self.config = config or EvolutionConfig()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid evolution config: {', '.join(errors)}")
        
        # Initialize operators
        self.mutation_operator = MutationOperator(self.config.mutation_config)
        self.crossover_operator = CrossoverOperator(self.config.crossover_config)
        
        logger.info(
            "Initialized EvolutionStrategy",
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elitism=self.config.elitism_count,
        )
    
    def evolve(
        self,
        parent1: Genome,
        parent2: Genome | None,
        generation: int,
    ) -> Genome:
        """
        Create offspring from parent genome(s).
        
        Args:
            parent1: Primary parent genome
            parent2: Optional second parent for crossover
            generation: Current generation number
        
        Returns:
            Child genome
        """
        # Decide whether to use crossover or mutation
        if parent2 is not None and random.random() < self.config.crossover_rate:
            # Crossover
            child = self.crossover_operator.crossover(parent1, parent2, generation)
            
            # Apply mutation with some probability
            if random.random() < self.config.mutation_rate:
                child = self.mutation_operator.mutate(child, generation)
        else:
            # Mutation only
            child = self.mutation_operator.mutate(parent1, generation)
        
        return child


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MutationType",
    "CrossoverType",
    "SelectionStrategy",
    "MutationConfig",
    "MutationOperator",
    "CrossoverConfig",
    "CrossoverOperator",
    "EvolutionConfig",
    "EvolutionStrategy",
]
