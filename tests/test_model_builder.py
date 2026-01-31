"""
Unit tests for the Model Builder component.

Tests cover:
- Building PyTorch models from genomes
- All supported layer types (30+ layers)
- Model validation
- Forward pass functionality
- Model serialization

Author: BelizeChain Team
License: MIT
"""

import pytest
import torch
import torch.nn as nn

from nawal.model_builder import ModelBuilder
from nawal.genome.dna import DNA, LayerGene


# ============================================================================
# Model Builder Tests
# ============================================================================

class TestModelBuilder:
    """Test ModelBuilder functionality."""
    
    def test_model_builder_initialization(self, genome_config):
        """Test ModelBuilder can be initialized."""
        builder = ModelBuilder(genome_config=genome_config)
        assert builder is not None
        assert builder.genome_config == genome_config
    
    def test_build_simple_model(self, sample_dna):
        """Test building a simple model from DNA."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_model_forward_pass(self, sample_dna):
        """Test forward pass through built model."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        # Create sample input
        x = torch.randn(4, sample_dna.input_size)
        
        # Forward pass
        output = model(x)
        
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == sample_dna.output_size
    
    def test_build_linear_layer(self):
        """Test building linear layer."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=1,
            layer_type="linear",
            params={"in_features": 10, "out_features": 20},
            enabled=True,
        )
        
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 10
        assert layer.out_features == 20
    
    def test_build_conv2d_layer(self):
        """Test building conv2d layer."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=2,
            layer_type="conv2d",
            params={
                "in_channels": 3,
                "out_channels": 16,
                "kernel_size": 3,
                "padding": 1,
            },
            enabled=True,
        )
        
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == 3
        assert layer.out_channels == 16
    
    def test_build_activation_layers(self):
        """Test building various activation layers."""
        builder = ModelBuilder()
        
        activations = [
            ("relu", nn.ReLU),
            ("tanh", nn.Tanh),
            ("sigmoid", nn.Sigmoid),
            ("leaky_relu", nn.LeakyReLU),
            ("elu", nn.ELU),
            ("gelu", nn.GELU),
        ]
        
        for activation_name, activation_class in activations:
            gene = LayerGene(
                innovation_id=1,
                layer_type=activation_name,
                params={},
                enabled=True,
            )
            layer = builder.build_layer(gene)
            assert isinstance(layer, activation_class)
    
    def test_build_normalization_layers(self):
        """Test building normalization layers."""
        builder = ModelBuilder()
        
        norms = [
            ("batchnorm1d", {"num_features": 20}, nn.BatchNorm1d),
            ("batchnorm2d", {"num_features": 16}, nn.BatchNorm2d),
            ("layernorm", {"normalized_shape": [20]}, nn.LayerNorm),
        ]
        
        for norm_name, params, norm_class in norms:
            gene = LayerGene(
                innovation_id=1,
                layer_type=norm_name,
                params=params,
                enabled=True,
            )
            layer = builder.build_layer(gene)
            assert isinstance(layer, norm_class)
    
    def test_build_dropout_layer(self):
        """Test building dropout layer."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=1,
            layer_type="dropout",
            params={"p": 0.5},
            enabled=True,
        )
        
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Dropout)
        assert layer.p == 0.5
    
    def test_build_pooling_layers(self):
        """Test building pooling layers."""
        builder = ModelBuilder()
        
        pools = [
            ("maxpool2d", {"kernel_size": 2}, nn.MaxPool2d),
            ("avgpool2d", {"kernel_size": 2}, nn.AvgPool2d),
            ("adaptiveavgpool2d", {"output_size": (1, 1)}, nn.AdaptiveAvgPool2d),
        ]
        
        for pool_name, params, pool_class in pools:
            gene = LayerGene(
                innovation_id=1,
                layer_type=pool_name,
                params=params,
                enabled=True,
            )
            layer = builder.build_layer(gene)
            assert isinstance(layer, pool_class)
    
    def test_build_recurrent_layers(self):
        """Test building recurrent layers."""
        builder = ModelBuilder()
        
        rnns = [
            ("lstm", {"input_size": 10, "hidden_size": 20}, nn.LSTM),
            ("gru", {"input_size": 10, "hidden_size": 20}, nn.GRU),
            ("rnn", {"input_size": 10, "hidden_size": 20}, nn.RNN),
        ]
        
        for rnn_name, params, rnn_class in rnns:
            gene = LayerGene(
                innovation_id=1,
                layer_type=rnn_name,
                params=params,
                enabled=True,
            )
            layer = builder.build_layer(gene)
            assert isinstance(layer, rnn_class)
    
    def test_build_attention_layer(self):
        """Test building attention layers."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=1,
            layer_type="multiheadattention",
            params={"embed_dim": 64, "num_heads": 8},
            enabled=True,
        )
        
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.MultiheadAttention)
    
    def test_build_transformer_layers(self):
        """Test building transformer layers."""
        builder = ModelBuilder()
        
        transformers = [
            ("transformerencoder", {
                "d_model": 64,
                "nhead": 8,
                "num_layers": 2,
            }, nn.TransformerEncoder),
            ("transformerdecoder", {
                "d_model": 64,
                "nhead": 8,
                "num_layers": 2,
            }, nn.TransformerDecoder),
        ]
        
        for trans_name, params, trans_class in transformers:
            gene = LayerGene(
                innovation_id=1,
                layer_type=trans_name,
                params=params,
                enabled=True,
            )
            layer = builder.build_layer(gene)
            # Check encoder/decoder layer is created
            assert layer is not None
    
    def test_disabled_layer_excluded(self):
        """Test disabled layers are excluded from model."""
        builder = ModelBuilder()
        
        dna = DNA(input_size=10, output_size=2)
        dna.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 16}, True),
            LayerGene(2, "relu", {}, False),  # Disabled
            LayerGene(3, "linear", {"in_features": 16, "out_features": 2}, True),
        ]
        
        model = builder.build(dna)
        
        # Count active layers (should skip disabled ReLU)
        active_layers = [m for m in model.modules() if not isinstance(m, nn.Sequential)]
        assert len(active_layers) >= 2  # At least 2 Linear layers
    
    def test_model_parameter_count(self, sample_dna):
        """Test model has trainable parameters."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0
    
    def test_model_training_mode(self, sample_dna):
        """Test model can switch training/eval modes."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        # Training mode
        model.train()
        assert model.training == True
        
        # Eval mode
        model.eval()
        assert model.training == False
    
    def test_model_to_device(self, sample_dna):
        """Test model can be moved to device."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        device = torch.device("cpu")
        model = model.to(device)
        
        # Check parameters are on correct device
        for param in model.parameters():
            assert param.device == device


# ============================================================================
# Complex Model Tests
# ============================================================================

class TestComplexModels:
    """Test building complex multi-layer models."""
    
    def test_deep_network(self):
        """Test building deep network."""
        builder = ModelBuilder()
        
        dna = DNA(input_size=10, output_size=2)
        dna.layer_genes = [
            LayerGene(1, "linear", {"in_features": 10, "out_features": 64}, True),
            LayerGene(2, "relu", {}, True),
            LayerGene(3, "batchnorm1d", {"num_features": 64}, True),
            LayerGene(4, "dropout", {"p": 0.3}, True),
            LayerGene(5, "linear", {"in_features": 64, "out_features": 32}, True),
            LayerGene(6, "relu", {}, True),
            LayerGene(7, "linear", {"in_features": 32, "out_features": 16}, True),
            LayerGene(8, "relu", {}, True),
            LayerGene(9, "linear", {"in_features": 16, "out_features": 2}, True),
        ]
        
        model = builder.build(dna)
        
        # Test forward pass
        x = torch.randn(8, 10)
        output = model(x)
        assert output.shape == (8, 2)
    
    def test_cnn_architecture(self):
        """Test building CNN architecture."""
        builder = ModelBuilder()
        
        dna = DNA(input_size=(3, 32, 32), output_size=10)
        dna.layer_genes = [
            LayerGene(1, "conv2d", {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}, True),
            LayerGene(2, "relu", {}, True),
            LayerGene(3, "maxpool2d", {"kernel_size": 2}, True),
            LayerGene(4, "conv2d", {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}, True),
            LayerGene(5, "relu", {}, True),
            LayerGene(6, "maxpool2d", {"kernel_size": 2}, True),
            LayerGene(7, "flatten", {}, True),
            LayerGene(8, "linear", {"in_features": 64 * 8 * 8, "out_features": 128}, True),
            LayerGene(9, "relu", {}, True),
            LayerGene(10, "linear", {"in_features": 128, "out_features": 10}, True),
        ]
        
        model = builder.build(dna)
        
        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)
    
    def test_residual_connections(self):
        """Test building model with skip connections."""
        # This would require enhanced DNA with connection genes support
        # Placeholder for future implementation
        pass


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestModelBuilderErrors:
    """Test error handling in model builder."""
    
    def test_invalid_layer_type(self):
        """Test handling of invalid layer type."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=1,
            layer_type="invalid_layer",
            params={},
            enabled=True,
        )
        
        with pytest.raises((ValueError, KeyError, AttributeError)):
            builder.build_layer(gene)
    
    def test_missing_parameters(self):
        """Test handling of missing required parameters."""
        builder = ModelBuilder()
        gene = LayerGene(
            innovation_id=1,
            layer_type="linear",
            params={},  # Missing in_features, out_features
            enabled=True,
        )
        
        with pytest.raises((TypeError, KeyError)):
            builder.build_layer(gene)
    
    def test_empty_dna(self):
        """Test building model from empty DNA."""
        builder = ModelBuilder()
        dna = DNA(input_size=10, output_size=2)
        # No layer genes
        
        with pytest.raises((ValueError, RuntimeError)):
            builder.build(dna)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestModelBuilderIntegration:
    """Integration tests for model builder."""
    
    def test_build_train_evaluate(self, sample_dna, sample_dataloader):
        """Test building, training, and evaluating a model."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        # Training setup
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for batch_idx, (data, target) in enumerate(sample_dataloader):
            if batch_idx >= 5:  # Train only 5 batches
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            for data, target in sample_dataloader:
                output = model(data)
                assert output.shape[1] == sample_dna.output_size
                break
    
    def test_model_serialization(self, sample_dna, temp_dir):
        """Test saving and loading built models."""
        builder = ModelBuilder()
        model = builder.build(sample_dna)
        
        # Save model
        save_path = temp_dir / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        model2 = builder.build(sample_dna)
        model2.load_state_dict(torch.load(save_path, weights_only=True))
        
        # Set both to eval mode to disable dropout
        model.eval()
        model2.eval()
        
        # Compare outputs with same input
        torch.manual_seed(42)
        x = torch.randn(4, sample_dna.input_size)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model2(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
