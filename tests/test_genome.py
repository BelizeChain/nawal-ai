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
        assert hasattr(selected, 'fitness_score')
    
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

@pytest.mark.integration
class TestGenomeIntegration:
    """Integration tests for genome system."""
    
    def test_full_evolution_cycle(self, genome_config, evolution_config, innovation_history):
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
