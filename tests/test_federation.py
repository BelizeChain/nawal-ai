"""
Unit tests for Federated Learning components.

Tests cover:
- FedAvg aggregation
- Multi-client federated training
- Metrics tracking
- Client selection
- Model aggregation correctness

Author: BelizeChain Team
License: MIT
"""

import pytest
import torch
import torch.nn as nn
from copy import deepcopy

from nawal.server.aggregator import FederatedAggregator
from nawal.server.metrics_tracker import MetricsTracker


# ============================================================================
# Federated Aggregator Tests
# ============================================================================

class TestFederatedAggregator:
    """Test federated aggregation functionality."""
    
    def test_aggregator_initialization(self, federated_config):
        """Test FederatedAggregator can be initialized."""
        aggregator = FederatedAggregator(config=federated_config)
        assert aggregator is not None
        assert aggregator.config == federated_config
    
    def test_fedavg_aggregation(self, client_models, federated_config):
        """Test FedAvg aggregation algorithm."""
        aggregator = FederatedAggregator(config=federated_config)
        
        # Get client model parameters
        client_params = [model.state_dict() for model in client_models]
        
        # Aggregate
        global_params = aggregator.fedavg_aggregate(client_params)
        
        assert global_params is not None
        assert len(global_params) == len(client_params[0])
    
    def test_weighted_aggregation(self, client_models, federated_config):
        """Test weighted aggregation with different client contributions."""
        aggregator = FederatedAggregator(config=federated_config)
        
        client_params = [model.state_dict() for model in client_models]
        weights = [0.5, 0.3, 0.2]  # Different weights for clients
        
        global_params = aggregator.weighted_aggregate(client_params, weights)
        
        assert global_params is not None
    
    def test_aggregation_preserves_structure(self, client_models):
        """Test aggregation preserves model structure."""
        aggregator = FederatedAggregator()
        
        client_params = [model.state_dict() for model in client_models]
        global_params = aggregator.fedavg_aggregate(client_params)
        
        # Check all keys are present
        for key in client_params[0].keys():
            assert key in global_params
            assert global_params[key].shape == client_params[0][key].shape
    
    def test_aggregation_averaging(self):
        """Test aggregation correctly averages parameters."""
        # Create simple models with known parameters
        models = []
        for val in [1.0, 2.0, 3.0]:
            model = nn.Linear(10, 2)
            with torch.no_grad():
                model.weight.fill_(val)
                model.bias.fill_(val)
            models.append(model)
        
        aggregator = FederatedAggregator()
        client_params = [m.state_dict() for m in models]
        global_params = aggregator.fedavg_aggregate(client_params)
        
        # Average should be 2.0
        expected_value = 2.0
        assert torch.allclose(global_params["weight"].mean(), torch.tensor(expected_value), atol=0.1)
        assert torch.allclose(global_params["bias"].mean(), torch.tensor(expected_value), atol=0.1)
    
    def test_single_client_aggregation(self, client_models):
        """Test aggregation with single client."""
        aggregator = FederatedAggregator()
        
        single_client = [client_models[0].state_dict()]
        global_params = aggregator.fedavg_aggregate(single_client)
        
        # Should equal single client parameters
        for key in single_client[0].keys():
            assert torch.allclose(global_params[key], single_client[0][key])
    
    def test_empty_client_list(self):
        """Test handling of empty client list."""
        aggregator = FederatedAggregator()
        
        with pytest.raises((ValueError, IndexError)):
            aggregator.fedavg_aggregate([])


# ============================================================================
# Client Selection Tests
# ============================================================================

class TestClientSelection:
    """Test client selection strategies."""
    
    def test_random_client_selection(self, federated_config):
        """Test random client selection."""
        aggregator = FederatedAggregator(config=federated_config)
        
        total_clients = 10
        num_to_select = 5
        
        selected = aggregator.select_clients(total_clients, num_to_select)
        
        assert len(selected) == num_to_select
        assert all(0 <= c < total_clients for c in selected)
        assert len(set(selected)) == num_to_select  # No duplicates
    
    def test_client_fraction_selection(self, federated_config):
        """Test selecting fraction of clients."""
        federated_config.client_fraction = 0.5
        aggregator = FederatedAggregator(config=federated_config)
        
        total_clients = 10
        selected = aggregator.select_clients_by_fraction(total_clients)
        
        assert len(selected) == 5  # 50% of 10
    
    def test_minimum_clients(self, federated_config):
        """Test minimum client requirement."""
        federated_config.min_clients = 3
        aggregator = FederatedAggregator(config=federated_config)
        
        # Should enforce minimum
        selected = aggregator.select_clients(total_clients=10, num_to_select=2)
        assert len(selected) >= federated_config.min_clients


# ============================================================================
# Federated Training Round Tests
# ============================================================================

class TestFederatedRound:
    """Test complete federated training round."""
    
    def test_single_round(self, client_models, client_dataloaders, federated_config):
        """Test single federated learning round."""
        from nawal.client.train import GenomeTrainer
        from nawal.config import TrainingConfig
        
        aggregator = FederatedAggregator(config=federated_config)
        
        # Initialize global model
        global_model = client_models[0]
        
        # Distribute global model to clients
        client_updates = []
        
        for client_model, dataloader in zip(client_models, client_dataloaders):
            # Load global parameters
            client_model.load_state_dict(global_model.state_dict())
            
            # Local training
            config = TrainingConfig(
                batch_size=8,
                learning_rate=0.001,
                epochs=1,
                device="cpu",
            )
            trainer = GenomeTrainer(config=config)
            trainer.set_model(client_model)
            trainer.train_epoch(dataloader)
            
            # Collect update
            client_updates.append(client_model.state_dict())
        
        # Aggregate
        global_params = aggregator.fedavg_aggregate(client_updates)
        global_model.load_state_dict(global_params)
        
        assert global_model is not None
    
    def test_multiple_rounds(self, client_models, client_dataloaders, federated_config):
        """Test multiple federated learning rounds."""
        from nawal.client.train import GenomeTrainer
        from nawal.config import TrainingConfig
        
        aggregator = FederatedAggregator(config=federated_config)
        global_model = client_models[0]
        
        num_rounds = 3
        round_losses = []
        
        for round_idx in range(num_rounds):
            client_updates = []
            round_loss = 0
            
            for client_model, dataloader in zip(client_models, client_dataloaders):
                client_model.load_state_dict(global_model.state_dict())
                
                config = TrainingConfig(batch_size=8, learning_rate=0.001, device="cpu")
                trainer = GenomeTrainer(config=config)
                trainer.set_model(client_model)
                metrics = trainer.train_epoch(dataloader)
                
                client_updates.append(client_model.state_dict())
                round_loss += metrics["loss"]
            
            global_params = aggregator.fedavg_aggregate(client_updates)
            global_model.load_state_dict(global_params)
            
            round_losses.append(round_loss / len(client_models))
        
        assert len(round_losses) == num_rounds
        # Loss should generally improve
        assert round_losses[-1] <= round_losses[0] + 0.5


# ============================================================================
# Metrics Tracking Tests
# ============================================================================

class TestMetricsTracker:
    """Test metrics tracking during federated learning."""
    
    def test_tracker_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()
        assert tracker is not None
    
    def test_record_metric(self):
        """Test recording a single metric."""
        tracker = MetricsTracker()
        
        tracker.record("loss", 0.5, round_num=1)
        tracker.record("accuracy", 0.85, round_num=1)
        
        assert tracker.get("loss", round_num=1) == 0.5
        assert tracker.get("accuracy", round_num=1) == 0.85
    
    def test_record_multiple_rounds(self):
        """Test recording metrics across multiple rounds."""
        tracker = MetricsTracker()
        
        for round_num in range(5):
            tracker.record("loss", 0.5 - round_num * 0.1, round_num=round_num)
        
        assert len(tracker.get_history("loss")) == 5
        assert tracker.get_history("loss")[0] > tracker.get_history("loss")[-1]
    
    def test_client_metrics(self):
        """Test tracking per-client metrics."""
        tracker = MetricsTracker()
        
        for client_id in range(3):
            tracker.record_client_metric(
                client_id=client_id,
                metric_name="loss",
                value=0.5 + client_id * 0.1,
                round_num=1,
            )
        
        client_losses = tracker.get_client_metrics("loss", round_num=1)
        assert len(client_losses) == 3
    
    def test_aggregated_metrics(self):
        """Test computing aggregated metrics."""
        tracker = MetricsTracker()
        
        # Record metrics for multiple clients
        for client_id in range(3):
            tracker.record_client_metric(client_id, "accuracy", 0.8 + client_id * 0.05, round_num=1)
        
        avg_accuracy = tracker.aggregate_client_metrics("accuracy", round_num=1, method="mean")
        assert 0.8 <= avg_accuracy <= 0.95
    
    def test_metrics_export(self, temp_dir):
        """Test exporting metrics to file."""
        tracker = MetricsTracker()
        
        for round_num in range(3):
            tracker.record("loss", 0.5 - round_num * 0.1, round_num=round_num)
            tracker.record("accuracy", 0.7 + round_num * 0.1, round_num=round_num)
        
        export_path = temp_dir / "metrics.json"
        tracker.export(export_path)
        
        assert export_path.exists()
    
    def test_metrics_import(self, temp_dir):
        """Test importing metrics from file."""
        tracker1 = MetricsTracker()
        tracker1.record("loss", 0.5, round_num=1)
        
        export_path = temp_dir / "metrics.json"
        tracker1.export(export_path)
        
        tracker2 = MetricsTracker()
        tracker2.load(export_path)
        
        assert tracker2.get("loss", round_num=1) == 0.5


# ============================================================================
# Byzantine Resilience Tests (Placeholder for Security Module)
# ============================================================================

class TestByzantineResilience:
    """Test resilience against Byzantine clients (placeholder)."""
    
    def test_detect_outlier_updates(self, client_models, byzantine_client_indices):
        """Test detecting Byzantine/malicious client updates."""
        aggregator = FederatedAggregator()
        
        client_params = []
        for idx, model in enumerate(client_models):
            params = model.state_dict()
            
            # Poison Byzantine clients
            if idx in byzantine_client_indices:
                for key in params:
                    params[key] = params[key] * 10  # 10x larger updates
            
            client_params.append(params)
        
        # Should detect outliers (will implement in security module)
        # outliers = aggregator.detect_outliers(client_params)
        # assert set(outliers) == set(byzantine_client_indices)
        pass  # Placeholder for security module
    
    def test_robust_aggregation(self, poisoned_gradients):
        """Test robust aggregation methods (Krum, trimmed mean)."""
        # Will implement in security module
        pass


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestFederatedIntegration:
    """Integration tests for federated learning workflow."""
    
    def test_end_to_end_federated_training(
        self,
        sample_population,
        client_dataloaders,
        federated_config,
        training_config,
    ):
        """Test complete federated training workflow."""
        from nawal.model_builder import ModelBuilder
        from nawal.client.train import GenomeTrainer
        
        builder = ModelBuilder()
        aggregator = FederatedAggregator(config=federated_config)
        tracker = MetricsTracker()
        
        # Select best genome
        genome = sample_population.genomes[0]
        global_model = builder.build(genome.dna)
        
        # Federated training rounds
        for round_num in range(2):
            client_updates = []
            round_metrics = []
            
            for client_id, dataloader in enumerate(client_dataloaders):
                # Build client model
                client_model = builder.build(genome.dna)
                client_model.load_state_dict(global_model.state_dict())
                
                # Local training
                trainer = GenomeTrainer(config=training_config)
                trainer.set_model(client_model)
                metrics = trainer.train_epoch(dataloader)
                
                # Record metrics
                tracker.record_client_metric(client_id, "loss", metrics["loss"], round_num)
                
                # Collect update
                client_updates.append(client_model.state_dict())
                round_metrics.append(metrics)
            
            # Aggregate
            global_params = aggregator.fedavg_aggregate(client_updates)
            global_model.load_state_dict(global_params)
            
            # Record round metrics
            avg_loss = sum(m["loss"] for m in round_metrics) / len(round_metrics)
            tracker.record("loss", avg_loss, round_num=round_num)
        
        # Check metrics were recorded
        assert len(tracker.get_history("loss")) == 2


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.benchmark
class TestFederationPerformance:
    """Performance benchmarks for federated learning."""
    
    def test_aggregation_speed(self, client_models):
        """Test aggregation speed with many clients."""
        import time
        
        aggregator = FederatedAggregator()
        client_params = [model.state_dict() for model in client_models]
        
        start_time = time.time()
        global_params = aggregator.fedavg_aggregate(client_params)
        elapsed = time.time() - start_time
        
        # Should be fast (< 1 second for 3 clients)
        assert elapsed < 1.0
    
    @pytest.mark.slow
    def test_scalability(self):
        """Test scalability with many clients."""
        import time
        
        aggregator = FederatedAggregator()
        
        # Create many clients
        num_clients = 100
        models = [nn.Linear(10, 2) for _ in range(num_clients)]
        client_params = [m.state_dict() for m in models]
        
        start_time = time.time()
        global_params = aggregator.fedavg_aggregate(client_params)
        elapsed = time.time() - start_time
        
        # Should handle 100 clients reasonably fast
        assert elapsed < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
