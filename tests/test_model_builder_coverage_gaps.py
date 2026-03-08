"""
Coverage gap tests for genome/model_builder.py (116 miss lines).

Covers:
- MultiQueryAttention forward
- GroupedQueryAttention forward
- AttentionFactory create (multi_query, grouped_query, flash_attention)
- MoELayer forward
- TransformerBlock with MoE
- LayerFactory (moe_transformer, attention, normalization, dropout, default)
- GenomeModel forward (4D CNN path, float tensor path, label shift, return_dict)
- GenomeModel generate (top-k, top-p)
- ModelBuilder.build (layer_genes, to_genome path)
- build_layer (adaptiveavgpool2d, flatten, unknown type, complex types)
- validate_genome
- estimate_flops (decoder layers path)
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch

from genome.model_builder import (
    MultiQueryAttention,
    GroupedQueryAttention,
    AttentionFactory,
    MoELayer,
    TransformerBlock,
    LayerFactory,
    GenomeModel,
    ModelBuilder,
    FeedForward,
    NormalizationFactory,
    ActivationFactory,
    RMSNorm,
)
from genome.encoding import Genome, ArchitectureLayer, LayerType
from nawal.genome.dna import DNA, LayerGene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genome(
    hidden_size=64,
    num_encoder_layers=1,
    num_heads=4,
    layer_type=LayerType.TRANSFORMER_ENCODER,
    decoder_layers=None,
):
    """Create a minimal Genome for testing.

    Note: Genome.hidden_size is a @property that reads from
    encoder_layers[0].hidden_size, so the ArchitectureLayer's
    hidden_size must match the desired model hidden_size.
    """
    encoder_layers = []
    for _ in range(num_encoder_layers):
        encoder_layers.append(
            ArchitectureLayer(
                layer_type=layer_type,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=0.0,
                activation=LayerType.GELU,
            )
        )
    genome = Genome(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers or [],
    )
    return genome


# ===========================================================================
# Attention
# ===========================================================================


class TestMultiQueryAttention:
    def test_forward(self):
        mqa = MultiQueryAttention(hidden_size=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out = mqa(x)
        assert out.shape == (2, 8, 64)

    def test_forward_with_mask(self):
        mqa = MultiQueryAttention(hidden_size=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 8, 64)
        mask = torch.zeros(2, 1, 1, 8)
        out = mqa(x, attention_mask=mask)
        assert out.shape == (2, 8, 64)


class TestGroupedQueryAttention:
    def test_forward(self):
        gqa = GroupedQueryAttention(hidden_size=64, num_heads=4, num_kv_heads=2, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out = gqa(x)
        assert out.shape == (2, 8, 64)

    def test_forward_with_mask(self):
        gqa = GroupedQueryAttention(hidden_size=64, num_heads=4, num_kv_heads=2, dropout=0.0)
        x = torch.randn(2, 8, 64)
        mask = torch.zeros(2, 1, 1, 8)
        out = gqa(x, attention_mask=mask)
        assert out.shape == (2, 8, 64)


class TestAttentionFactory:
    def test_multi_query(self):
        attn = AttentionFactory.create("multi_query", 64, 4, 0.0)
        assert isinstance(attn, MultiQueryAttention)

    def test_grouped_query(self):
        attn = AttentionFactory.create("grouped_query", 64, 4, 0.0)
        assert isinstance(attn, GroupedQueryAttention)

    def test_flash_attention(self):
        attn = AttentionFactory.create("flash_attention", 64, 4, 0.0)
        # Falls back to MultiHeadAttention
        from genome.model_builder import MultiHeadAttention
        assert isinstance(attn, MultiHeadAttention)

    def test_unknown_type(self):
        attn = AttentionFactory.create("xxunknownxx", 64, 4, 0.0)
        from genome.model_builder import MultiHeadAttention
        assert isinstance(attn, MultiHeadAttention)


# ===========================================================================
# MoE
# ===========================================================================


class TestMoELayer:
    def test_forward(self):
        moe = MoELayer(hidden_size=64, intermediate_size=128, num_experts=4, num_experts_per_token=2, dropout=0.0)
        x = torch.randn(2, 4, 64)
        out = moe(x)
        assert out.shape == (2, 4, 64)


class TestTransformerBlockMoE:
    def test_moe_transformer(self):
        block = TransformerBlock(
            hidden_size=64, num_heads=4, intermediate_size=128,
            use_moe=True, num_experts=4, dropout=0.0,
        )
        x = torch.randn(1, 4, 64)
        out = block(x)
        assert out.shape == (1, 4, 64)


# ===========================================================================
# LayerFactory
# ===========================================================================


class TestLayerFactory:
    def test_moe_transformer_type(self):
        layer_config = ArchitectureLayer(
            layer_type=LayerType.MIXTURE_OF_EXPERTS,
            hidden_size=128,
            num_heads=4,
            num_experts=4,
            dropout_rate=0.0,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert isinstance(layer, TransformerBlock)

    def test_attention_type(self):
        layer_config = ArchitectureLayer(
            layer_type=LayerType.MULTIHEAD_ATTENTION,
            num_heads=4,
            dropout_rate=0.0,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert layer is not None

    def test_normalization_type(self):
        layer_config = ArchitectureLayer(
            layer_type=LayerType.LAYER_NORM,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert isinstance(layer, nn.LayerNorm)

    def test_dropout_type(self):
        layer_config = ArchitectureLayer(
            layer_type=LayerType.DROPOUT,
            dropout_rate=0.3,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert isinstance(layer, nn.Dropout)

    def test_feedforward_type(self):
        layer_config = ArchitectureLayer(
            layer_type=LayerType.LINEAR,
            hidden_size=128,
            dropout_rate=0.0,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert isinstance(layer, FeedForward)

    def test_unsupported_defaults(self):
        """Unknown layer type falls back to TransformerBlock (line 486)."""
        layer_config = ArchitectureLayer(
            layer_type=LayerType.EMBEDDING,
        )
        layer = LayerFactory.create_layer(layer_config, 64)
        assert isinstance(layer, TransformerBlock)


# ===========================================================================
# GenomeModel - forward paths
# ===========================================================================


class TestGenomeModelForward:
    """Cover all forward-path branches."""

    def test_4d_cnn_path_no_labels(self):
        """4D input (batch, channels, height, width) → CNN path, no labels → return logits."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        # Replace encoder with Conv2d that keeps last dim = hidden_size
        # Conv2d(3, 64, 1) on (B, 3, H, W) -> (B, 64, H, W)
        # lm_head is Linear(64, 10), applied to last dim.
        # Need last dim = 64, so use (B, 3, H, 64) input shape
        model.encoder_layers = nn.ModuleList([nn.Conv2d(3, 64, kernel_size=1)])
        x = torch.randn(2, 3, 4, 64)
        # Conv2d with kernel_size=1 on last two dims: (B,3,4,64) -> (B,64,4,64)
        # lm_head(Linear(64,10)) applied to last dim 64 -> works
        out = model.forward(x)
        assert isinstance(out, torch.Tensor)

    def test_4d_cnn_path_with_labels(self):
        """4D input with labels → return dict with loss."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        # Use Identity to pass through 4D tensor, then lm_head(Linear(64,10)) on last dim
        model.encoder_layers = nn.ModuleList([nn.Identity()])
        x = torch.randn(2, 10, 4, 64)  # (B, C, H, hidden_size=64)
        # lm_head(Linear(64, 10)) applied to last dim → logits: (2, 10, 4, 10)
        # CrossEntropyLoss: input (N, C, d1, d2) → target (N, d1, d2) with C=10
        labels = torch.randint(0, 10, (2, 4, 10))
        out = model.forward(x, labels=labels.long())
        assert isinstance(out, dict)
        assert "loss" in out

    def test_float_tensor_3d_path(self):
        """3D float input → skip token embedding."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        x = torch.randn(2, 4, 64)  # (batch, seq, hidden)
        out = model.forward(x)
        assert isinstance(out, dict)
        assert "logits" in out

    def test_float_tensor_2d_hidden_path(self):
        """2D float tensor with shape (batch, hidden_size) → add seq dim."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        x = torch.rand(2, 64)  # Uniform [0, 1) guarantees max <= 1.0
        out = model.forward(x)
        assert isinstance(out, dict)

    def test_standard_token_ids_with_labels(self):
        """Standard int input with labels → shifted loss calculation."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (2, 8))
        labels = torch.randint(0, 10, (2, 8))
        out = model.forward(input_ids, labels=labels)
        assert "loss" in out
        assert out["loss"] is not None

    def test_return_dict_false_inference(self):
        """return_dict=False → return just logits tensor."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (2, 8))
        out = model.forward(input_ids, return_dict=False)
        assert isinstance(out, torch.Tensor)

    def test_single_sample_unsqueeze(self):
        """1D input → unsqueeze to add batch dimension."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (8,))
        out = model.forward(input_ids)
        assert isinstance(out, dict)

    def test_attention_mask(self):
        """Provide attention mask → covers mask processing."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (2, 8))
        mask = torch.ones(2, 8)
        mask[:, -3:] = 0
        out = model.forward(input_ids, attention_mask=mask)
        assert "logits" in out

    def test_with_decoder_layers(self):
        """Model with decoder layers."""
        decoder_layers = [
            ArchitectureLayer(
                layer_type=LayerType.TRANSFORMER_ENCODER,
                hidden_size=256,
                num_heads=4,
                dropout_rate=0.0,
            )
        ]
        genome = _make_genome(hidden_size=64, decoder_layers=decoder_layers)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (2, 8))
        out = model.forward(input_ids)
        assert "logits" in out

    def test_float_labels_converted(self):
        """Labels as float → should be converted to long."""
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=16)
        input_ids = torch.randint(0, 10, (2, 8))
        labels = torch.randint(0, 10, (2, 8)).float()
        out = model.forward(input_ids, labels=labels)
        assert "loss" in out


# ===========================================================================
# GenomeModel - generate
# ===========================================================================


class TestGenomeModelGenerate:
    def test_generate_basic(self):
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=64)
        input_ids = torch.randint(0, 10, (1, 4))
        out = model.generate(input_ids, max_length=3)
        assert out.shape[1] > 4

    def test_generate_top_k(self):
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=64)
        input_ids = torch.randint(0, 10, (1, 4))
        out = model.generate(input_ids, max_length=2, top_k=5)
        assert out.shape[1] > 4

    def test_generate_top_p(self):
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=64)
        input_ids = torch.randint(0, 10, (1, 4))
        out = model.generate(input_ids, max_length=2, top_p=0.9)
        assert out.shape[1] > 4

    def test_generate_top_k_and_top_p(self):
        genome = _make_genome(hidden_size=64)
        model = GenomeModel(genome, vocab_size=10, max_seq_length=64)
        input_ids = torch.randint(0, 10, (1, 4))
        out = model.generate(input_ids, max_length=2, top_k=5, top_p=0.9)
        assert out.shape[1] > 4


# ===========================================================================
# ModelBuilder.build
# ===========================================================================


class TestModelBuilderBuild:
    def test_build_with_genome(self):
        builder = ModelBuilder(vocab_size=10, max_seq_length=16)
        genome = _make_genome(hidden_size=64)
        model = builder.build(genome)
        assert isinstance(model, GenomeModel)

    def test_build_with_dna_layer_genes(self):
        """DNA with layer_genes → build Sequential model."""
        builder = ModelBuilder(vocab_size=10, max_seq_length=16)
        dna = DNA(input_size=10, output_size=5)
        dna.layer_genes = [
            LayerGene(innovation_id=1, layer_type="linear", params={"in_features": 10, "out_features": 5}, enabled=True),
        ]
        model = builder.build(dna)
        assert isinstance(model, nn.Sequential)

    def test_build_with_dna_disabled_gene(self):
        """DNA with disabled gene → skipped."""
        builder = ModelBuilder(vocab_size=10, max_seq_length=16)
        dna = DNA(input_size=10, output_size=5)
        dna.layer_genes = [
            LayerGene(innovation_id=1, layer_type="linear", params={"in_features": 10, "out_features": 5}, enabled=True),
            LayerGene(innovation_id=2, layer_type="relu", params={}, enabled=False),
        ]
        model = builder.build(dna)
        assert isinstance(model, nn.Sequential)
        assert len(model) == 1  # Only enabled layer

    def test_build_with_dna_to_genome(self):
        """DNA without layer_genes → to_genome fallback."""
        builder = ModelBuilder(vocab_size=10, max_seq_length=16)
        dna = DNA(input_size=10, output_size=0)  # output_size=0 → else branch
        dna.layer_genes = []
        # Pre-seed cached genome so to_genome() returns a valid Genome
        dna._genome = _make_genome(hidden_size=64)
        model = builder.build(dna)
        assert isinstance(model, GenomeModel)

    def test_build_with_dna_output_size(self):
        """DNA with output_size → uses it as vocab_size."""
        builder = ModelBuilder(vocab_size=10, max_seq_length=16)
        dna = DNA(input_size=10, output_size=20)
        dna.layer_genes = []
        # Pre-seed cached genome so to_genome() returns a valid Genome
        dna._genome = _make_genome(hidden_size=64)
        model = builder.build(dna)
        assert isinstance(model, GenomeModel)
        # vocab_size should be restored
        assert builder.vocab_size == 10


# ===========================================================================
# build_layer
# ===========================================================================


class TestBuildLayer:
    def test_linear(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "linear", {"in_features": 10, "out_features": 5}, enabled=True)
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Linear)

    def test_conv2d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "conv2d", {"in_channels": 3, "out_channels": 16, "kernel_size": 3}, enabled=True)
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Conv2d)

    def test_conv_alias(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "conv", {"in_channels": 3, "out_channels": 16}, enabled=True)
        layer = builder.build_layer(gene)
        assert isinstance(layer, nn.Conv2d)

    def test_relu(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "relu", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.ReLU)

    def test_leaky_relu(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "leaky_relu", {"negative_slope": 0.1}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.LeakyReLU)

    def test_elu(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "elu", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.ELU)

    def test_gelu(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "gelu", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.GELU)

    def test_tanh(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "tanh", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.Tanh)

    def test_sigmoid(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "sigmoid", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.Sigmoid)

    def test_dropout(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "dropout", {"p": 0.3}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.Dropout)

    def test_batchnorm1d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "batchnorm", {"num_features": 64}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.BatchNorm1d)

    def test_batchnorm2d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "batchnorm2d", {"num_features": 64}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.BatchNorm2d)

    def test_layernorm(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "layernorm", {"normalized_shape": 64}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.LayerNorm)

    def test_maxpool2d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "maxpool2d", {"kernel_size": 2}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.MaxPool2d)

    def test_avgpool2d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "avgpool2d", {"kernel_size": 2}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.AvgPool2d)

    def test_adaptiveavgpool2d(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "adaptiveavgpool2d", {"output_size": (1, 1)}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.AdaptiveAvgPool2d)

    def test_lstm(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "lstm", {"input_size": 64, "hidden_size": 32}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.LSTM)

    def test_gru(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "gru", {"input_size": 64, "hidden_size": 32}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.GRU)

    def test_rnn(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "rnn", {"input_size": 64, "hidden_size": 32}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.RNN)

    def test_multiheadattention(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "multihead_attention", {"embed_dim": 64, "num_heads": 4}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.MultiheadAttention)

    def test_flatten(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "flatten", {}, enabled=True)
        assert isinstance(builder.build_layer(gene), nn.Flatten)

    def test_unknown_type_raises(self):
        builder = ModelBuilder()
        gene = LayerGene(1, "xyzunknown", {}, enabled=True)
        with pytest.raises(ValueError, match="Unknown or unsupported"):
            builder.build_layer(gene)

    def test_not_layer_gene_raises(self):
        builder = ModelBuilder()
        with pytest.raises(ValueError, match="Expected LayerGene"):
            builder.build_layer("not_a_gene")


# ===========================================================================
# validate_genome / estimate_flops
# ===========================================================================


class TestValidateGenome:
    def test_no_encoder_layers(self):
        builder = ModelBuilder()
        genome = _make_genome(num_encoder_layers=0)
        errors = builder.validate_genome(genome)
        assert any("no encoder" in e.lower() for e in errors)

    def test_too_many_layers(self):
        builder = ModelBuilder()
        genome = _make_genome(num_encoder_layers=101)
        errors = builder.validate_genome(genome)
        assert any("Too many" in e for e in errors)

    def test_hidden_size_small(self):
        builder = ModelBuilder()
        genome = _make_genome(hidden_size=32, num_heads=4)
        errors = builder.validate_genome(genome)
        assert any("too small" in e.lower() for e in errors)

    def test_hidden_size_large(self):
        builder = ModelBuilder()
        genome = _make_genome(hidden_size=8192 + 64, num_heads=4)
        errors = builder.validate_genome(genome)
        assert any("too large" in e.lower() for e in errors)

    def test_heads_not_divisible(self):
        builder = ModelBuilder()
        genome = _make_genome(hidden_size=64, num_heads=5)
        errors = builder.validate_genome(genome)
        assert any("not divisible" in e for e in errors)

    def test_build_model_raises_on_critical(self):
        builder = ModelBuilder()
        genome = _make_genome(num_encoder_layers=0)
        with pytest.raises(ValueError, match="Cannot build model"):
            builder.build_model(genome)


class TestEstimateFlops:
    def test_basic(self):
        builder = ModelBuilder(vocab_size=100)
        genome = _make_genome(hidden_size=64)
        flops = builder.estimate_flops(genome, seq_length=16)
        assert flops > 0

    def test_with_decoder_layers(self):
        builder = ModelBuilder(vocab_size=100)
        decoder_layers = [
            ArchitectureLayer(
                layer_type=LayerType.TRANSFORMER_ENCODER,
                hidden_size=256,
                num_heads=4,
            )
        ]
        genome = _make_genome(hidden_size=64, decoder_layers=decoder_layers)
        flops = builder.estimate_flops(genome, seq_length=16)
        assert flops > 0

    def test_estimate_memory(self):
        builder = ModelBuilder(vocab_size=100, max_seq_length=16)
        genome = _make_genome(hidden_size=64)
        mem = builder.estimate_memory(genome, batch_size=2)
        assert "parameters_mb" in mem
        assert "total_mb_fp32" in mem


# ===========================================================================
# Extra coverage: NormalizationFactory, ActivationFactory, RMSNorm
# ===========================================================================


class TestNormalizationFactory:
    def test_rms_norm(self):
        norm = NormalizationFactory.create("rms_norm", 64)
        assert isinstance(norm, RMSNorm)

    def test_group_norm(self):
        norm = NormalizationFactory.create("group_norm", 64)
        assert isinstance(norm, nn.GroupNorm)

    def test_batch_norm(self):
        norm = NormalizationFactory.create("batch_norm", 64)
        assert isinstance(norm, nn.BatchNorm1d)

    def test_unknown_norm(self):
        norm = NormalizationFactory.create("xxunknown", 64)
        assert isinstance(norm, nn.LayerNorm)


class TestActivationFactory:
    def test_unknown_activation(self):
        act = ActivationFactory.create("xxunknown")
        assert isinstance(act, nn.GELU)

    def test_mish(self):
        act = ActivationFactory.create("mish")
        assert isinstance(act, nn.Mish)

    def test_sigmoid(self):
        act = ActivationFactory.create("sigmoid")
        assert isinstance(act, nn.Sigmoid)
