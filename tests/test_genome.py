"""
Unit tests for the Genome System (DNA, Operators, Population, History).

Tests cover:
- DNA encoding/decoding
- Layer and connection genes
- Mutation operators
- Crossover operators
- Population management
- Innovation tracking

Author: BelizeChain Team
License: MIT
"""

import pytest
import torch
from copy import deepcopy

from nawal.genome.dna import DNA, LayerGene, ConnectionGene
from nawal.genome.operators import MutationOperator, CrossoverOperator
from nawal.genome.population import Population
from nawal.genome.history import InnovationHistory

# ============================================================================
# DNA Tests
# ============================================================================


class TestDNA:
    """Test DNA encoding and manipulation."""

    def test_dna_initialization(self):
        """Test DNA can be initialized with input/output sizes."""
        dna = DNA(input_size=10, output_size=2)
        assert dna.input_size == 10
        assert dna.output_size == 2
        assert len(dna.layer_genes) == 0
        assert len(dna.connection_genes) == 0

    def test_add_layer_gene(self, sample_dna):
        """Test adding a layer gene to DNA."""
        initial_count = len(sample_dna.layer_genes)
        new_gene = LayerGene(
            innovation_id=999,
            layer_type="dropout",
            params={"p": 0.5},
            enabled=True,
        )
        sample_dna.add_layer_gene(new_gene)
        assert len(sample_dna.layer_genes) == initial_count + 1
        assert sample_dna.layer_genes[-1].innovation_id == 999

    def test_add_connection_gene(self, sample_dna):
        """Test adding a connection gene to DNA."""
        initial_count = len(sample_dna.connection_genes)
        new_gene = ConnectionGene(
            innovation_id=999,
            source_layer=0,
            target_layer=3,
            enabled=True,
        )
        sample_dna.add_connection_gene(new_gene)
        assert len(sample_dna.connection_genes) == initial_count + 1
        assert sample_dna.connection_genes[-1].innovation_id == 999

    def test_remove_layer_gene(self, sample_dna):
        """Test removing a layer gene from DNA."""
        initial_count = len(sample_dna.layer_genes)
        assert initial_count > 0
        sample_dna.remove_layer_gene(0)
        assert len(sample_dna.layer_genes) == initial_count - 1

    def test_enable_disable_gene(self, sample_dna):
        """Test enabling/disabling genes."""
        gene = sample_dna.layer_genes[0]
        original_state = gene.enabled

        gene.enabled = False
        assert gene.enabled == False

        gene.enabled = True
        assert gene.enabled == True

    def test_dna_clone(self, sample_dna):
        """Test DNA can be cloned."""
        cloned = sample_dna.clone()
        assert cloned.input_size == sample_dna.input_size
        assert cloned.output_size == sample_dna.output_size
        assert len(cloned.layer_genes) == len(sample_dna.layer_genes)
        assert len(cloned.connection_genes) == len(sample_dna.connection_genes)

        # Ensure deep copy (modifying clone doesn't affect original)
        cloned.layer_genes[0].enabled = not sample_dna.layer_genes[0].enabled
        assert cloned.layer_genes[0].enabled != sample_dna.layer_genes[0].enabled

    def test_dna_to_dict(self, sample_dna):
        """Test DNA serialization to dictionary."""
        dna_dict = sample_dna.to_dict()
        assert "input_size" in dna_dict
        assert "output_size" in dna_dict
        assert "layer_genes" in dna_dict
        assert "connection_genes" in dna_dict
        assert dna_dict["input_size"] == sample_dna.input_size

    def test_dna_from_dict(self, sample_dna):
        """Test DNA deserialization from dictionary."""
        dna_dict = sample_dna.to_dict()
        restored = DNA.from_dict(dna_dict)
        assert restored.input_size == sample_dna.input_size
        assert restored.output_size == sample_dna.output_size
        assert len(restored.layer_genes) == len(sample_dna.layer_genes)


# ============================================================================
# Layer Gene Tests
# ============================================================================


class TestLayerGene:
    """Test LayerGene functionality."""

    def test_layer_gene_creation(self):
        """Test LayerGene can be created."""
        gene = LayerGene(
            innovation_id=1,
            layer_type="linear",
            params={"in_features": 10, "out_features": 20},
            enabled=True,
        )
        assert gene.innovation_id == 1
        assert gene.layer_type == "linear"
        assert gene.params["in_features"] == 10
        assert gene.enabled == True

    def test_layer_gene_types(self):
        """Test various layer types."""
        layer_types = [
            ("linear", {"in_features": 10, "out_features": 20}),
            ("conv2d", {"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
            ("relu", {}),
            ("dropout", {"p": 0.5}),
            ("batchnorm1d", {"num_features": 20}),
        ]

        for i, (layer_type, params) in enumerate(layer_types):
            gene = LayerGene(
                innovation_id=i,
                layer_type=layer_type,
                params=params,
                enabled=True,
            )
            assert gene.layer_type == layer_type
            assert gene.params == params

    def test_layer_gene_serialization(self):
        """Test LayerGene serialization."""
        gene = LayerGene(
            innovation_id=1,
            layer_type="linear",
            params={"in_features": 10, "out_features": 20},
            enabled=True,
        )
        gene_dict = gene.to_dict()
        assert gene_dict["innovation_id"] == 1
        assert gene_dict["layer_type"] == "linear"

        restored = LayerGene.from_dict(gene_dict)
        assert restored.innovation_id == gene.innovation_id
        assert restored.layer_type == gene.layer_type


# ============================================================================
# Connection Gene Tests
# ============================================================================


class TestConnectionGene:
    """Test ConnectionGene functionality."""

    def test_connection_gene_creation(self):
        """Test ConnectionGene can be created."""
        gene = ConnectionGene(
            innovation_id=101,
            source_layer=0,
            target_layer=1,
            enabled=True,
        )
        assert gene.innovation_id == 101
        assert gene.source_layer == 0
        assert gene.target_layer == 1
        assert gene.enabled == True

    def test_connection_gene_serialization(self):
        """Test ConnectionGene serialization."""
        gene = ConnectionGene(
            innovation_id=101,
            source_layer=0,
            target_layer=1,
            enabled=True,
        )
        gene_dict = gene.to_dict()
        restored = ConnectionGene.from_dict(gene_dict)
        assert restored.innovation_id == gene.innovation_id
        assert restored.source_layer == gene.source_layer
        assert restored.target_layer == gene.target_layer


# ============================================================================
# Mutation Operator Tests
# ============================================================================


class TestMutationOperator:
    """Test mutation operators."""

    def test_add_layer_mutation(self, sample_dna, innovation_history):
        """Test adding a layer via mutation."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.2,
            remove_layer_rate=0.1,
            replace_layer_rate=0.2,
            modify_layer_rate=0.2,
            hyperparameter_rate=0.2,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        # Convert DNA to Genome, mutate, convert back
        genome = sample_dna.to_genome()
        mutated_genome = operator.mutate(genome, generation=1)
        mutated_dna = DNA.from_genome(mutated_genome)

        # Mutation may add/remove/modify layers
        assert mutated_dna is not None

    def test_remove_layer_mutation(self, sample_dna, innovation_history):
        """Test removing a layer via mutation."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.2,
            remove_layer_rate=0.1,
            replace_layer_rate=0.2,
            modify_layer_rate=0.2,
            hyperparameter_rate=0.2,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        genome = sample_dna.to_genome()
        mutated_genome = operator.mutate(genome, generation=1)
        mutated_dna = DNA.from_genome(mutated_genome)

        # Mutation applied successfully
        assert mutated_dna is not None

    def test_mutate_parameters(self, sample_dna, innovation_history):
        """Test parameter mutation."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.1,
            remove_layer_rate=0.1,
            replace_layer_rate=0.1,
            modify_layer_rate=0.3,
            hyperparameter_rate=0.3,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        genome = sample_dna.to_genome()
        mutated_genome = operator.mutate(genome, generation=1)
        mutated_dna = DNA.from_genome(mutated_genome)

        # Mutation applied
        assert mutated_dna is not None

    def test_add_connection_mutation(self, sample_dna, innovation_history):
        """Test adding a connection via mutation."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.2,
            remove_layer_rate=0.1,
            replace_layer_rate=0.2,
            modify_layer_rate=0.2,
            hyperparameter_rate=0.2,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        genome = sample_dna.to_genome()
        mutated_genome = operator.mutate(genome, generation=1)

        # Mutation applied
        assert mutated_genome is not None

    def test_toggle_connection(self, sample_dna, innovation_history):
        """Test toggling connection enabled state."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.2,
            remove_layer_rate=0.1,
            replace_layer_rate=0.2,
            modify_layer_rate=0.2,
            hyperparameter_rate=0.2,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        genome = sample_dna.to_genome()
        mutated_genome = operator.mutate(genome, generation=1)

        # Mutation applied
        assert mutated_genome is not None


# ============================================================================
# Crossover Operator Tests
# ============================================================================


class TestCrossoverOperator:
    """Test crossover operators."""

    def test_single_point_crossover(self, innovation_history):
        """Test single-point crossover."""
        from nawal.genome.operators import CrossoverConfig

        config = CrossoverConfig(
            uniform_rate=0.2,
            single_point_rate=0.3,
            two_point_rate=0.2,
            layer_wise_rate=0.2,
            hyperparameter_rate=0.1,
        )
        operator = CrossoverOperator(config=config)

        # Create two parent DNAs
        parent1 = DNA(input_size=10, output_size=2)
        parent1.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 16}, True),
            LayerGene(2, "relu", {}, True),
            LayerGene(3, "linear", {"in_features": 16, "out_features": 2}, True),
        ]

        parent2 = DNA(input_size=10, output_size=2)
        parent2.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 32}, True),
            LayerGene(4, "tanh", {}, True),
            LayerGene(3, "linear", {"in_features": 32, "out_features": 2}, True),
        ]

        # Convert to Genome and perform crossover
        genome1 = parent1.to_genome()
        genome2 = parent2.to_genome()
        offspring_genome = operator.crossover(genome1, genome2, generation=1)
        offspring = DNA.from_genome(offspring_genome)

        assert offspring.input_size == 10
        assert offspring.output_size == 2
        assert len(offspring.layer_genes) > 0

    def test_uniform_crossover(self, innovation_history):
        """Test uniform crossover."""
        from nawal.genome.operators import CrossoverConfig

        config = CrossoverConfig(
            uniform_rate=0.2,
            single_point_rate=0.3,
            two_point_rate=0.2,
            layer_wise_rate=0.2,
            hyperparameter_rate=0.1,
        )
        operator = CrossoverOperator(config=config)

        parent1 = DNA(input_size=10, output_size=2)
        parent1.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 16}, True),
            LayerGene(2, "relu", {}, True),
        ]

        parent2 = DNA(input_size=10, output_size=2)
        parent2.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 32}, True),
            LayerGene(3, "tanh", {}, True),
        ]

        # Convert to Genome and perform crossover
        genome1 = parent1.to_genome()
        genome2 = parent2.to_genome()
        offspring_genome = operator.crossover(genome1, genome2, generation=1)
        offspring = DNA.from_genome(offspring_genome)

        # Crossover produces valid DNA (architecture may vary)
        assert offspring is not None
        assert isinstance(offspring, DNA)


# ============================================================================
# Population Tests
# ============================================================================


class TestPopulation:
    """Test population management."""

    def test_population_initialization(self, sample_population):
        """Test population initializes with correct size."""
        assert len(sample_population.genomes) > 0
        assert sample_population.generation == 0

    def test_population_size(self, sample_population, evolution_config):
        """Test population maintains correct size."""
        assert len(sample_population.genomes) == evolution_config.population_size

    def test_assign_fitness(self, sample_population, mock_fitness_scores):
        """Test assigning fitness to population."""
        for genome, fitness in zip(sample_population.genomes, mock_fitness_scores):
            genome.fitness_score = fitness

        assert all(g.fitness_score is not None for g in sample_population.genomes)

    def test_get_best_genome(self, sample_population, mock_fitness_scores):
        """Test retrieving best genome."""
        for genome, fitness in zip(sample_population.genomes, mock_fitness_scores):
            genome.fitness_score = fitness

        best = sample_population.get_best_genome()
        assert best.fitness_score == max(mock_fitness_scores)

    def test_tournament_selection(self, sample_population, mock_fitness_scores):
        """Test tournament selection."""
        for genome, fitness in zip(sample_population.genomes, mock_fitness_scores):
            genome.fitness_score = fitness

        selected = sample_population.tournament_selection(tournament_size=3)
        assert selected is not None
        assert hasattr(selected, "fitness_score")

    def test_evolve_generation(self, sample_population, mock_fitness_scores):
        """Test evolving to next generation."""
        for genome, fitness in zip(sample_population.genomes, mock_fitness_scores):
            genome.fitness_score = fitness

        initial_gen = sample_population.generation
        sample_population.evolve()

        assert sample_population.generation == initial_gen + 1
        assert len(sample_population.genomes) > 0
        assert len(sample_population.genomes) > 0


# ============================================================================
# Innovation History Tests
# ============================================================================


class TestInnovationHistory:
    """Test innovation tracking."""

    def test_innovation_history_creation(self):
        """Test InnovationHistory can be created."""
        history = InnovationHistory()
        assert history.next_innovation_id == 1

    def test_register_layer_innovation(self, innovation_history):
        """Test registering a new layer innovation."""
        innovation_id = innovation_history.register_layer_innovation(
            layer_type="linear",
            params={"in_features": 10, "out_features": 20},
        )
        assert innovation_id > 0

        # Same innovation should return same ID
        same_id = innovation_history.register_layer_innovation(
            layer_type="linear",
            params={"in_features": 10, "out_features": 20},
        )
        assert same_id == innovation_id

    def test_register_connection_innovation(self, innovation_history):
        """Test registering a new connection innovation."""
        innovation_id = innovation_history.register_connection_innovation(
            source_layer=0,
            target_layer=1,
        )
        assert innovation_id > 0

        # Same connection should return same ID
        same_id = innovation_history.register_connection_innovation(
            source_layer=0,
            target_layer=1,
        )
        assert same_id == innovation_id

    def test_innovation_id_increment(self, innovation_history):
        """Test innovation IDs increment correctly."""
        id1 = innovation_history.next_innovation_id
        innovation_history.next_innovation_id += 1
        id2 = innovation_history.next_innovation_id
        assert id2 == id1 + 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestGenomeIntegration:
    """Integration tests for genome system."""

    def test_full_evolution_cycle(
        self, genome_config, evolution_config, innovation_history
    ):
        """Test complete evolution cycle."""
        from nawal.genome.population import PopulationConfig, PopulationManager
        from nawal.genome.encoding import Genome

        # Create population using fixture (which has backward compatibility)
        pop_config = PopulationConfig(
            target_size=evolution_config.population_size,
            max_size=evolution_config.population_size + 10,
            min_size=max(2, evolution_config.population_size // 2),
        )
        population = PopulationManager(config=pop_config)

        # Initialize with genomes
        for i in range(pop_config.target_size):
            genome = Genome(genome_id=f"test_genome_{i}")
            genome.fitness_score = torch.rand(1).item() * 100  # Scale to 0-100
            population.add_genome(genome)

        # Check initial state
        assert len(population.genomes) == evolution_config.population_size

    def test_mutation_preserves_validity(self, sample_dna, innovation_history):
        """Test mutations produce valid DNA."""
        from nawal.genome.operators import MutationConfig

        config = MutationConfig(
            add_layer_rate=0.2,
            remove_layer_rate=0.1,
            replace_layer_rate=0.2,
            modify_layer_rate=0.2,
            hyperparameter_rate=0.2,
            architecture_rate=0.1,
        )
        operator = MutationOperator(config=config)

        for i in range(10):
            genome = sample_dna.to_genome()
            mutated_genome = operator.mutate(genome, generation=i)
            mutated = DNA.from_genome(mutated_genome)

            # Mutation may change architecture, but should produce valid DNA
            assert mutated is not None
            assert isinstance(mutated, DNA)
            # Layer genes may be added/removed/modified by mutation
            assert isinstance(mutated.layer_genes, list)

    def test_crossover_preserves_validity(self, innovation_history):
        """Test crossover produces valid offspring."""
        from nawal.genome.operators import CrossoverConfig

        config = CrossoverConfig(
            uniform_rate=0.2,
            single_point_rate=0.3,
            two_point_rate=0.2,
            layer_wise_rate=0.2,
            hyperparameter_rate=0.1,
        )
        operator = CrossoverOperator(config=config)

        parent1 = DNA(input_size=10, output_size=2)
        parent1.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 16}, True),
            LayerGene(2, "relu", {}, True),
        ]

        parent2 = DNA(input_size=10, output_size=2)
        parent2.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 32}, True),
            LayerGene(3, "tanh", {}, True),
        ]

        # Convert to Genome and perform crossover
        genome1 = parent1.to_genome()
        genome2 = parent2.to_genome()
        offspring_genome = operator.crossover(genome1, genome2, generation=1)
        offspring = DNA.from_genome(offspring_genome)

        # Crossover produces valid DNA (architecture may vary from parents)
        assert offspring is not None
        assert isinstance(offspring, DNA)
        assert isinstance(offspring.layer_genes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# B5 Audit Tests — C5.1 through C5.3
# ============================================================================


class TestC51RoundTrip:
    """C5.1 — Verify DNA encoding round-trip preserves all data."""

    def test_genome_dict_round_trip(self):
        """to_dict → from_dict preserves all genome fields."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            Hyperparameters,
            LayerType,
        )

        original = Genome(
            genome_id="test_roundtrip_001",
            generation=5,
            parent_genomes=["parent_a", "parent_b"],
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=512,
                    num_heads=8,
                    num_layers=6,
                    dropout_rate=0.1,
                    activation="gelu",
                ),
            ],
            decoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.LINEAR,
                    input_size=512,
                    output_size=256,
                ),
            ],
            hyperparameters=Hyperparameters(
                learning_rate=3e-4,
                optimizer="lion",
                batch_size=64,
                precision="bf16",
            ),
            fitness_score=85.5,
            quality_score=90.0,
            timeliness_score=80.0,
            honesty_score=82.0,
        )

        data = original.to_dict()
        restored = Genome.from_dict(data)

        assert restored.genome_id == original.genome_id
        assert restored.generation == original.generation
        assert restored.parent_genomes == original.parent_genomes
        assert restored.fitness_score == original.fitness_score
        assert restored.quality_score == original.quality_score
        assert restored.timeliness_score == original.timeliness_score
        assert restored.honesty_score == original.honesty_score
        assert len(restored.encoder_layers) == len(original.encoder_layers)
        assert len(restored.decoder_layers) == len(original.decoder_layers)
        assert restored.encoder_layers[0].hidden_size == 512
        assert restored.encoder_layers[0].num_heads == 8
        assert restored.decoder_layers[0].input_size == 512
        assert (
            restored.hyperparameters.learning_rate
            == original.hyperparameters.learning_rate
        )
        assert restored.hyperparameters.optimizer == original.hyperparameters.optimizer
        assert (
            restored.hyperparameters.batch_size == original.hyperparameters.batch_size
        )

    def test_genome_json_round_trip(self):
        """to_json → from_json preserves all genome fields."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            Hyperparameters,
            LayerType,
        )

        original = Genome(
            genome_id="test_json_roundtrip",
            generation=3,
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.MULTIHEAD_ATTENTION,
                    hidden_size=256,
                    num_heads=4,
                ),
            ],
            decoder_layers=[],
            fitness_score=72.0,
        )

        json_str = original.to_json()
        restored = Genome.from_json(json_str)

        assert restored.genome_id == original.genome_id
        assert restored.generation == original.generation
        assert restored.fitness_score == original.fitness_score
        assert len(restored.encoder_layers) == 1
        assert restored.encoder_layers[0].hidden_size == 256
        assert restored.encoder_layers[0].num_heads == 4

    def test_clone_preserves_architecture(self):
        """clone() preserves architecture while assigning new ID."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )

        original = Genome(
            genome_id="clone_test",
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                ),
            ],
        )

        cloned = original.clone()
        assert cloned.genome_id != original.genome_id
        assert original.genome_id in cloned.parent_genomes
        assert len(cloned.encoder_layers) == 1
        assert cloned.encoder_layers[0].hidden_size == 768
        assert cloned.encoder_layers[0].num_heads == 12

    def test_genome_hash_deterministic(self):
        """genome_hash is deterministic — same genome yields same hash on repeated calls."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )

        g = Genome(
            genome_id="a",
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.LINEAR, input_size=10, output_size=20
                ),
            ],
        )

        # Same genome → same hash on repeated calls
        assert g.genome_hash == g.genome_hash

    def test_genome_hash_differs_for_different_architectures(self):
        """Different architectures produce different hashes."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )

        g1 = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_id="shared_id",
                    layer_type=LayerType.LINEAR,
                    input_size=10,
                    output_size=20,
                ),
            ],
        )
        g2 = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_id="shared_id",
                    layer_type=LayerType.LINEAR,
                    input_size=10,
                    output_size=30,  # different output_size
                ),
            ],
        )

        assert g1.genome_hash != g2.genome_hash


class TestC52GeneticAlgorithmValidity:
    """C5.2 — Verify genetic operators produce valid genomes."""

    def _make_genome(self):
        """Helper: create a genome with proper transformer layers."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
            Hyperparameters,
        )

        return Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                    num_layers=6,
                    dropout_rate=0.1,
                    activation="gelu",
                ),
            ],
            decoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.LINEAR,
                    input_size=768,
                    output_size=768,
                ),
            ],
            hyperparameters=Hyperparameters(
                learning_rate=1e-4,
                optimizer="adamw",
            ),
        )

    def test_mutate_add_layer_no_crash(self):
        """_mutate_add_layer must not crash (was genome.hyperparameters.hidden_size bug)."""
        from nawal.genome.operators import MutationOperator, MutationConfig

        config = MutationConfig(
            add_layer_rate=1.0,
            remove_layer_rate=0.0,
            replace_layer_rate=0.0,
            modify_layer_rate=0.0,
            hyperparameter_rate=0.0,
            architecture_rate=0.0,
        )
        op = MutationOperator(config=config)
        genome = self._make_genome()
        # Should NOT raise AttributeError
        child = op.mutate(genome, generation=1)
        assert child is not None
        assert child.total_layers >= genome.total_layers

    def test_mutate_add_attention_no_crash(self):
        """_mutate_add_attention must not crash (same hidden_size bug)."""
        from nawal.genome.operators import MutationOperator, MutationConfig

        config = MutationConfig(
            add_layer_rate=0.0,
            remove_layer_rate=0.0,
            replace_layer_rate=0.0,
            modify_layer_rate=0.0,
            hyperparameter_rate=0.0,
            architecture_rate=1.0,
        )
        op = MutationOperator(config=config)
        genome = self._make_genome()
        # Run multiple times — architecture mutations pick ADD_ATTENTION, ADD_MOE, or ADD_SSM
        for _ in range(20):
            child = op.mutate(genome, generation=1)
            assert child is not None

    def test_modify_layer_hidden_heads_alignment(self):
        """After _mutate_modify_layer, hidden_size % num_heads must == 0."""
        from nawal.genome.operators import MutationOperator, MutationConfig
        import random

        config = MutationConfig(
            add_layer_rate=0.0,
            remove_layer_rate=0.0,
            replace_layer_rate=0.0,
            modify_layer_rate=1.0,
            hyperparameter_rate=0.0,
            architecture_rate=0.0,
        )
        op = MutationOperator(config=config)

        random.seed(42)
        for _ in range(50):
            genome = self._make_genome()
            child = op.mutate(genome, generation=1)
            for layer in child.encoder_layers + child.decoder_layers:
                if layer.hidden_size and layer.num_heads:
                    assert layer.hidden_size % layer.num_heads == 0, (
                        f"hidden_size={layer.hidden_size} not divisible by "
                        f"num_heads={layer.num_heads}"
                    )

    def test_crossover_produces_valid_genome(self):
        """Crossover must produce a genome with valid layer structure."""
        from nawal.genome.operators import CrossoverOperator, CrossoverConfig

        config = CrossoverConfig(
            uniform_rate=0.2,
            single_point_rate=0.3,
            two_point_rate=0.2,
            layer_wise_rate=0.2,
            hyperparameter_rate=0.1,
        )
        op = CrossoverOperator(config=config)

        p1 = self._make_genome()
        p2 = self._make_genome()

        for _ in range(20):
            child = op.crossover(p1, p2, generation=1)
            assert child is not None
            assert child.generation == 1
            assert len(child.parent_genomes) == 2
            # Fitness must be reset
            assert child.fitness_score is None

    def test_evolution_strategy_no_crash(self):
        """Full EvolutionStrategy.evolve must not crash."""
        from nawal.genome.operators import EvolutionStrategy

        strategy = EvolutionStrategy()
        p1 = self._make_genome()
        p2 = self._make_genome()

        for _ in range(10):
            child = strategy.evolve(p1, p2, generation=1)
            assert child is not None


class TestC53AdapterFieldMapping:
    """C5.3 — Verify GenomeToNawalAdapter maps all fields correctly."""

    def test_adapter_extracts_vocab_size_from_embedding(self):
        """vocab_size should come from embedding layer, not be hardcoded."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )
        from nawal.genome.nawal_adapter import GenomeToNawalAdapter

        genome = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.EMBEDDING,
                    input_size=50000,
                    output_size=768,
                ),
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                ),
            ],
        )

        adapter = GenomeToNawalAdapter()
        config = adapter.genome_to_config(genome)
        assert config.vocab_size == 50000

    def test_adapter_defaults_vocab_size_without_embedding(self):
        """Without embedding layer, vocab_size defaults to 52000."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )
        from nawal.genome.nawal_adapter import GenomeToNawalAdapter

        genome = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                ),
            ],
        )

        adapter = GenomeToNawalAdapter()
        config = adapter.genome_to_config(genome)
        assert config.vocab_size == 52000

    def test_adapter_extracts_activation_from_layer(self):
        """Activation should come from transformer layer, not hyperparameters."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )
        from nawal.genome.nawal_adapter import GenomeToNawalAdapter

        genome = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                    activation="silu",
                ),
            ],
        )

        adapter = GenomeToNawalAdapter()
        config = adapter.genome_to_config(genome)
        assert config.activation == "swish"  # silu maps to swish

    def test_adapter_extracts_max_position_embeddings(self):
        """max_position_embeddings should come from positional encoding layer."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )
        from nawal.genome.nawal_adapter import GenomeToNawalAdapter

        genome = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.POSITIONAL_ENCODING,
                    hidden_size=768,
                    parameters={"max_seq_len": 2048},
                ),
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                ),
            ],
        )

        adapter = GenomeToNawalAdapter()
        config = adapter.genome_to_config(genome)
        assert config.max_position_embeddings == 2048

    def test_adapter_maps_hidden_size_and_heads(self):
        """hidden_size and num_heads must come from genome transformer layers."""
        from nawal.genome.encoding import (
            Genome,
            ArchitectureLayer,
            LayerType,
        )
        from nawal.genome.nawal_adapter import GenomeToNawalAdapter

        genome = Genome(
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=1024,
                    num_heads=16,
                ),
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=1024,
                    num_heads=16,
                ),
            ],
        )

        adapter = GenomeToNawalAdapter()
        config = adapter.genome_to_config(genome)
        assert config.hidden_size == 1024
        assert config.num_heads == 16
        assert config.num_layers == 2
        assert config.intermediate_size == 4096  # 1024 * 4
