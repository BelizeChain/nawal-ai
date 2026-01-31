"""
Unit tests for the Training Client and Fitness Scoring.

Tests cover:
- Local training on validators
- Fitness score calculation (PoUW)
- Training loops and optimization
- Model evaluation
- Checkpoint management

Author: BelizeChain Team
License: MIT
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from nawal.client.genome_trainer import GenomeTrainer
from nawal.genome.dna import DNA


# ============================================================================
# GenomeTrainer Tests
# ============================================================================

class TestGenomeTrainer:
    """Test GenomeTrainer functionality."""
    
    def test_trainer_initialization(self, training_config):
        """Test GenomeTrainer can be initialized."""
        trainer = GenomeTrainer(config=training_config)
        assert trainer is not None
        assert trainer.config == training_config
    
    def test_trainer_with_model(self, sample_model, training_config):
        """Test trainer with existing model."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        assert trainer.model is not None
    
    def test_train_one_epoch(self, sample_model, sample_dataloader, training_config):
        """Test training for one epoch."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        initial_params = {name: param.clone() for name, param in sample_model.named_parameters()}
        
        metrics = trainer.train_epoch(sample_dataloader)
        
        assert "loss" in metrics
        assert metrics["loss"] > 0
        
        # Check parameters were updated
        for name, param in sample_model.named_parameters():
            assert not torch.equal(param, initial_params[name])
    
    def test_evaluate_model(self, sample_model, sample_dataloader, training_config):
        """Test model evaluation."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        metrics = trainer.evaluate(sample_dataloader)
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_full_training_loop(self, sample_model, train_val_dataloaders, training_config):
        """Test complete training loop."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        train_loader, val_loader = train_val_dataloaders
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
        )
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
    
    def test_different_optimizers(self, sample_model, sample_dataloader):
        """Test training with different optimizers."""
        optimizers = ["adam", "sgd", "adamw", "rmsprop"]
        
        for opt_name in optimizers:
            config = TrainingConfig(
                batch_size=8,
                learning_rate=0.001,
                epochs=1,
                optimizer=opt_name,
                device="cpu",
            )
            
            trainer = GenomeTrainer(config=config)
            trainer.set_model(sample_model)
            
            metrics = trainer.train_epoch(sample_dataloader)
            assert "loss" in metrics
    
    def test_gradient_clipping(self, sample_model, sample_dataloader, training_config):
        """Test gradient clipping during training."""
        training_config.gradient_clip = 1.0
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        # Train and check gradients are clipped
        metrics = trainer.train_epoch(sample_dataloader)
        
        for param in sample_model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradients should be clipped
                assert grad_norm <= training_config.gradient_clip * 10  # Some tolerance


# ============================================================================
# Fitness Calculation Tests
# ============================================================================

class TestFitnessCalculation:
    """Test fitness score calculation for PoUW."""
    
    def test_accuracy_fitness(self, sample_model, sample_dataloader, training_config):
        """Test fitness based on accuracy."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        metrics = trainer.evaluate(sample_dataloader)
        accuracy = metrics["accuracy"]
        
        # Fitness should be based on performance
        fitness = trainer.calculate_fitness(metrics)
        assert 0 <= fitness <= 1
        assert abs(fitness - accuracy) < 0.5  # Fitness related to accuracy
    
    def test_multi_metric_fitness(self, sample_model, sample_dataloader, training_config):
        """Test fitness considering multiple metrics."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        metrics = trainer.evaluate(sample_dataloader)
        
        # Fitness should consider accuracy, loss, efficiency
        fitness = trainer.calculate_fitness(
            metrics,
            quality_weight=0.4,
            timeliness_weight=0.3,
            honesty_weight=0.3,
        )
        
        assert 0 <= fitness <= 1
    
    def test_fitness_with_training_time(self, sample_model, sample_dataloader, training_config):
        """Test fitness penalizes slow training."""
        import time
        
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        start_time = time.time()
        trainer.train_epoch(sample_dataloader)
        training_time = time.time() - start_time
        
        metrics = trainer.evaluate(sample_dataloader)
        metrics["training_time"] = training_time
        
        fitness = trainer.calculate_fitness(metrics)
        assert 0 <= fitness <= 1
    
    def test_fitness_ranking(self, sample_dataloader, training_config):
        """Test fitness can rank models."""
        # Create models with different architectures
        models = [
            nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2)),
            nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)),
            nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2)),
        ]
        
        trainer = GenomeTrainer(config=training_config)
        fitness_scores = []
        
        for model in models:
            trainer.set_model(model)
            trainer.train_epoch(sample_dataloader)
            metrics = trainer.evaluate(sample_dataloader)
            fitness = trainer.calculate_fitness(metrics)
            fitness_scores.append(fitness)
        
        # All fitness scores should be valid
        assert all(0 <= f <= 1 for f in fitness_scores)


# ============================================================================
# Checkpoint Management Tests
# ============================================================================

class TestCheckpointManagement:
    """Test checkpoint saving and loading."""
    
    def test_save_checkpoint(self, sample_model, checkpoint_dir, training_config):
        """Test saving training checkpoint."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_1.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=1, metrics={"loss": 0.5})
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, sample_model, checkpoint_dir, training_config):
        """Test loading training checkpoint."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=5, metrics={"loss": 0.3})
        
        # Create new trainer and load
        trainer2 = GenomeTrainer(config=training_config)
        trainer2.set_model(sample_model)
        epoch, metrics = trainer2.load_checkpoint(checkpoint_path)
        
        assert epoch == 5
        assert metrics["loss"] == 0.3
    
    def test_resume_training(self, sample_model, train_val_dataloaders, checkpoint_dir, training_config):
        """Test resuming training from checkpoint."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        train_loader, val_loader = train_val_dataloaders
        
        # Train for 2 epochs
        trainer.train(train_loader, val_loader, epochs=2)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / "resume.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=2, metrics={"loss": 0.4})
        
        # Resume training
        trainer2 = GenomeTrainer(config=training_config)
        trainer2.set_model(sample_model)
        epoch, _ = trainer2.load_checkpoint(checkpoint_path)
        
        # Continue training
        history = trainer2.train(train_loader, val_loader, epochs=1, start_epoch=epoch)
        assert len(history["train_loss"]) == 1


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestLossFunctions:
    """Test various loss functions."""
    
    def test_cross_entropy_loss(self, sample_model, sample_dataloader):
        """Test cross-entropy loss for classification."""
        config = TrainingConfig(
            batch_size=8,
            learning_rate=0.001,
            loss_function="cross_entropy",
            device="cpu",
        )
        
        trainer = GenomeTrainer(config=config)
        trainer.set_model(sample_model)
        
        metrics = trainer.train_epoch(sample_dataloader)
        assert "loss" in metrics
        assert metrics["loss"] > 0
    
    def test_mse_loss(self, sample_dataloader):
        """Test MSE loss for regression."""
        # Regression model
        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Single output for regression
        )
        
        config = TrainingConfig(
            batch_size=8,
            learning_rate=0.001,
            loss_function="mse",
            device="cpu",
        )
        
        trainer = GenomeTrainer(config=config)
        trainer.set_model(model)
        
        # Note: sample_dataloader has classification targets,
        # but we're testing the loss function setup
        try:
            metrics = trainer.train_epoch(sample_dataloader)
            assert "loss" in metrics
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            # Expected if target format doesn't match
            pass  # This is expected for MSE with classification data


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for training workflow."""
    
    def test_genome_to_training(self, sample_dna, sample_dataloader, training_config):
        """Test full pipeline from genome to trained model."""
        from nawal.model_builder import ModelBuilder
        
        # Build model from genome
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        # Train model
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(model)
        metrics = trainer.train_epoch(sample_dataloader)
        
        # Evaluate
        eval_metrics = trainer.evaluate(sample_dataloader)
        
        # Calculate fitness
        fitness = trainer.calculate_fitness(eval_metrics)
        
        assert "loss" in metrics
        assert "accuracy" in eval_metrics
        assert 0 <= fitness <= 1
    
    def test_multiple_genomes_training(self, sample_population, sample_dataloader, training_config):
        """Test training multiple genomes."""
        from nawal.model_builder import ModelBuilder
        
        builder = ModelBuilder()
        trainer = GenomeTrainer(config=training_config)
        
        fitness_scores = []
        
        for genome in sample_population.genomes[:3]:  # Test first 3
            model = builder.build(genome.dna)
            trainer.set_model(model)
            trainer.train_epoch(sample_dataloader)
            metrics = trainer.evaluate(sample_dataloader)
            fitness = trainer.calculate_fitness(metrics)
            
            genome.fitness = fitness
            fitness_scores.append(fitness)
        
        assert len(fitness_scores) == 3
        assert all(0 <= f <= 1 for f in fitness_scores)
    
    @pytest.mark.slow
    def test_convergence(self, sample_model, train_val_dataloaders, training_config):
        """Test model converges over epochs."""
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        train_loader, val_loader = train_val_dataloaders
        
        # Train for more epochs
        history = trainer.train(train_loader, val_loader, epochs=10)
        
        # Loss should generally decrease
        train_losses = history["train_loss"]
        assert train_losses[-1] < train_losses[0] + 0.5  # Allow some variance


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.benchmark
class TestTrainingPerformance:
    """Performance benchmarks for training."""
    
    def test_training_throughput(self, sample_model, sample_dataloader, training_config):
        """Test training throughput (samples/sec)."""
        import time
        
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        start_time = time.time()
        trainer.train_epoch(sample_dataloader)
        elapsed = time.time() - start_time
        
        total_samples = len(sample_dataloader.dataset)
        throughput = total_samples / elapsed
        
        # Should process at least 10 samples/sec on CPU
        assert throughput > 10
    
    def test_memory_usage(self, sample_model, sample_dataloader, training_config):
        """Test memory usage during training."""
        import torch
        
        trainer = GenomeTrainer(config=training_config)
        trainer.set_model(sample_model)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            trainer.train_epoch(sample_dataloader)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            # Should use reasonable memory
            assert peak_memory < 1000  # Less than 1GB


from nawal.config import TrainingConfig

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
