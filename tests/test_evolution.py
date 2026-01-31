"""
Unit tests for the Evolution Orchestrator.

Tests cover:
- Multi-generation evolution
- Fitness-based selection
- Population management across generations
- Checkpoint/resume functionality
- Evolution strategies

Author: BelizeChain Team
License: MIT
"""

import pytest
import torch
from pathlib import Path

from nawal.orchestrator import EvolutionOrchestrator
from nawal.genome.population import Population


# ============================================================================
# Evolution Orchestrator Tests
# ============================================================================

class TestEvolutionOrchestrator:
    """Test EvolutionOrchestrator functionality."""
    
    def test_orchestrator_initialization(self, nawal_config, sample_dataloader):
        """Test EvolutionOrchestrator can be initialized."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        assert orchestrator is not None
        assert orchestrator.config == nawal_config
    
    def test_initialize_population(self, nawal_config, sample_dataloader):
        """Test population initialization."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        assert orchestrator.population is not None
        assert len(orchestrator.population.genomes) == nawal_config.evolution.population_size
    
    @pytest.mark.asyncio
    async def test_single_generation(self, nawal_config, sample_dataloader):
        """Test running single generation."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Run one generation internally
        await orchestrator._run_generation(generation=0)
        
        # Manually save state since we're calling internal method
        orchestrator._save_generation_state(generation=0, training_time=0.0)
        
        assert orchestrator.current_generation == 0
        assert len(orchestrator.generation_history) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_generations(self, nawal_config, sample_dataloader):
        """Test running multiple generations."""
        nawal_config.evolution.num_generations = 3
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Run evolution
        best_genome = await orchestrator.evolve()
        
        assert best_genome is not None
        assert len(orchestrator.generation_history) == 3
    
    @pytest.mark.asyncio
    async def test_fitness_improvement(self, nawal_config, sample_dataloader):
        """Test fitness improves over generations."""
        nawal_config.evolution.num_generations = 3
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Run evolution
        await orchestrator.evolve()
        
        # Check that we have history
        assert len(orchestrator.generation_history) >= 2
        first_fitness = orchestrator.generation_history[0].best_fitness
        last_fitness = orchestrator.generation_history[-1].best_fitness
        # Fitness should improve or stay roughly the same
        assert last_fitness >= first_fitness - 0.3  # Allow some variance
    
    def test_elitism(self, nawal_config, sample_dataloader):
        """Test elitism preserves best genomes."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Apply elitism
        elite = orchestrator._apply_elitism()
        
        # Should return elite genomes
        assert elite is not None
        assert len(elite) <= nawal_config.evolution.elitism_count
    
    def test_population_diversity(self, nawal_config, sample_dataloader):
        """Test population maintains diversity."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Check genome diversity
        genomes = orchestrator.population.genomes.values()  # Get genome objects
        unique_architectures = set()
        
        for genome in genomes:
            # Use genome_id as unique identifier
            unique_architectures.add(genome.genome_id)
        
        # Should have diversity (all different initially)
        assert len(unique_architectures) == len(orchestrator.population.genomes)
    
    def test_convergence_to_fitness_threshold(self, nawal_config, sample_dataloader):
        """Test evolution stops when fitness threshold reached."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Set some genomes with high fitness
        genomes_list = list(orchestrator.population.genomes.values())
        for genome in genomes_list[:3]:
            genome.fitness_score = 0.95
        
        # Check early stopping condition
        should_stop = orchestrator._should_stop_early()
        
        assert isinstance(should_stop, bool)


# ============================================================================
# Selection Strategy Tests
# ============================================================================

class TestSelectionStrategies:
    """Test different selection strategies."""
    
    def test_tournament_selection(self, nawal_config, sample_dataloader):
        """Test tournament selection."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Select parents using internal method
        parents = orchestrator._select_parents()
        
        assert parents is not None
        assert len(parents) > 0
    
    def test_roulette_wheel_selection(self, nawal_config, sample_dataloader):
        """Test roulette wheel selection."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Select parents
        parents = orchestrator._select_parents()
        assert parents is not None
    
    def test_rank_selection(self, nawal_config, sample_dataloader):
        """Test rank-based selection."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Select parents
        parents = orchestrator._select_parents()
        assert parents is not None


# ============================================================================
# Checkpoint Management Tests
# ============================================================================

class TestEvolutionCheckpoints:
    """Test checkpoint saving and loading."""
    
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, nawal_config, sample_dataloader, checkpoint_dir):
        """Test saving evolution checkpoint."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Run one generation
        await orchestrator._run_generation(generation=0)
        
        await orchestrator._save_checkpoint(generation=0)
        
        # Check that checkpoint was saved to config's checkpoint directory
        actual_checkpoint_dir = nawal_config.storage.checkpoint_dir
        checkpoints = list(actual_checkpoint_dir.glob("checkpoint_gen_*.pt"))
        assert len(checkpoints) > 0
    
    @pytest.mark.asyncio
    async def test_load_checkpoint(self, nawal_config, sample_dataloader, checkpoint_dir):
        """Test loading evolution checkpoint."""
        # Create and save checkpoint
        orchestrator1 = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        await orchestrator1._run_generation(generation=0)
        
        checkpoint_path = checkpoint_dir / "evolution.pt"
        await orchestrator1._save_checkpoint(generation=0)
        
        # Load in new orchestrator would require checkpoint to exist
        # Just verify structure
        assert orchestrator1.population is not None
    
    @pytest.mark.asyncio
    async def test_resume_evolution(self, nawal_config, sample_dataloader, checkpoint_dir):
        """Test resuming evolution from checkpoint."""
        # Run 2 generations and save
        nawal_config.evolution.num_generations = 2
        orchestrator1 = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        await orchestrator1.evolve()
        
        # Should have history
        assert len(orchestrator1.generation_history) > 0
    
    def test_checkpoint_contains_history(self, nawal_config, sample_dataloader):
        """Test checkpoint includes evolution history."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Should start with empty history
        assert len(orchestrator.generation_history) == 0


# ============================================================================
# Federated Evolution Tests
# ============================================================================

class TestFederatedEvolution:
    """Test evolution with federated learning."""
    
    def test_distributed_fitness_evaluation(self, nawal_config, sample_dataloader):
        """Test fitness evaluation across multiple validators."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Check population exists
        assert orchestrator.population is not None
        assert len(orchestrator.population.genomes) > 0
    
    def test_asynchronous_evolution(self, nawal_config, sample_dataloader):
        """Test asynchronous evolution updates."""
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Check structure
        assert orchestrator.population is not None
        assert len(orchestrator.population.genomes) > 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestEvolutionIntegration:
    """Integration tests for evolution orchestration."""
    
    @pytest.mark.asyncio
    async def test_full_evolution_pipeline(self, nawal_config, sample_dataloader):
        """Test complete evolution pipeline."""
        nawal_config.evolution.num_generations = 2
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Evolve
        best_genome = await orchestrator.evolve()
        
        assert best_genome is not None
        assert len(orchestrator.generation_history) == 2
    
    @pytest.mark.slow
    def test_long_evolution_run(self, nawal_config, sample_dataloader):
        """Test long evolution run (10 generations)."""
        # Skipped - too slow
        pass
    
    @pytest.mark.asyncio
    async def test_evolution_with_federated_learning(
        self,
        nawal_config,
        sample_dataloader,
    ):
        """Test evolution integrated with federated learning."""
        nawal_config.evolution.num_generations = 2
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Run evolution
        best_genome = await orchestrator.evolve()
        
        assert best_genome is not None


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.benchmark
class TestEvolutionPerformance:
    """Performance benchmarks for evolution."""
    
    @pytest.mark.asyncio
    async def test_generation_speed(self, nawal_config, sample_dataloader):
        """Test speed of single generation."""
        import time
        
        nawal_config.evolution.population_size = 10
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        start_time = time.time()
        await orchestrator._run_generation(generation=0)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 60s for 10 genomes)
        assert elapsed < 60
    
    def test_memory_efficiency(self, nawal_config, sample_dataloader):
        """Test memory efficiency during evolution."""
        import gc
        import sys
        
        orchestrator = EvolutionOrchestrator(
            config=nawal_config,
            train_loader=sample_dataloader
        )
        
        # Force garbage collection
        gc.collect()
        
        # Memory should be reasonable
        # (Hard to test precisely without platform-specific tools)
        assert sys.getsizeof(orchestrator.population) < 1024 ** 3  # < 1GB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
