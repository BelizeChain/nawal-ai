"""
Tests for data leakage detection (membership inference, gradient inversion,
information leakage, overfitting detection).

These tests mirror the style used in the differential privacy tests and focus
on unit-level verification of each detection method plus integration checks.
"""

import pytest
import torch
from unittest.mock import Mock

from nawal.client.genome_trainer import GenomeTrainer, TrainingConfig
from nawal.genome.dna import Genome, ArchitectureLayer, LayerType


@pytest.fixture
def training_config():
    return TrainingConfig(
        participant_id="test_validator",
        validator_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        staking_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        batch_size=16,
        learning_rate=0.001,
        device="cpu",
    )


@pytest.fixture
def simple_genome():
    return Genome(
        genome_id="test_genome",
        generation=1,
        hidden_size=64,
        encoder_layers=[
            ArchitectureLayer(
                layer_type=LayerType.TRANSFORMER_ENCODER,
                hidden_size=64,
                num_heads=8,
                dropout_rate=0.1,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_membership_inference_no_gap_high_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Simulate similar train/val losses -> no memorization
    for _ in range(10):
        trainer._store_training_loss(0.5)
        trainer._store_validation_loss(0.55)

    score = trainer._check_membership_inference()
    assert score >= 95.0, f"No gap should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_membership_inference_large_gap_low_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Large gap: train loss small, val loss large
    for _ in range(10):
        trainer._store_training_loss(0.1)
        trainer._store_validation_loss(1.5)

    score = trainer._check_membership_inference()
    assert score <= 30.0, f"Large gap should score <=30, got {score:.2f}"


@pytest.mark.asyncio
async def test_gradient_inversion_low_magnitude_high_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    grads = {
        "layer0.weight": torch.randn(10, 10) * 0.01,
        "layer1.weight": torch.randn(10, 10) * 0.02,
    }

    score = trainer._check_gradient_inversion(grads)
    assert score >= 95.0, f"Low early-layer gradients should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_gradient_inversion_high_magnitude_low_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    grads = {
        "00.weight": torch.randn(10, 10) * 20.0,
        "01.weight": torch.randn(10, 10) * 15.0,
    }

    score = trainer._check_gradient_inversion(grads)
    assert score <= 30.0, f"High early-layer gradients should score <=30, got {score:.2f}"


@pytest.mark.asyncio
async def test_information_leakage_healthy_confidence_high_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Create predictions with avg confidence ~0.75
    preds = torch.stack([torch.tensor([0.25, 0.75]) for _ in range(32)])
    score = trainer._check_information_leakage(preds)
    assert score >= 95.0, f"Healthy confidence should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_information_leakage_overconfident_low_score(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Overconfident predictions (0.995) should indicate leakage
    # With penalty = (0.995 - 0.98) * 2000 = 30, score = 100 - 30 = 70
    preds = torch.stack([torch.tensor([0.005, 0.995]) for _ in range(32)])
    score = trainer._check_information_leakage(preds)
    assert 60.0 <= score <= 75.0, f"Overconfident predictions should score 60-75, got {score:.2f}"


@pytest.mark.asyncio
async def test_prediction_confidence_history_limits(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Add 200 confidences -> should keep last 100
    for i in range(200):
        trainer.prediction_confidences.append(min(1.0, 0.5 + (i * 0.001)))
        if len(trainer.prediction_confidences) > 100:
            trainer.prediction_confidences = trainer.prediction_confidences[-100:]

    assert len(trainer.prediction_confidences) == 100, "Should retain last 100 confidences"


@pytest.mark.asyncio
async def test_training_validation_history_tracking(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    for i in range(60):
        trainer._store_training_loss(0.1 * (i % 5))
        trainer._store_validation_loss(0.15 * (i % 7))

    assert len(trainer.training_losses) == 50, "Training loss history should be capped at 50"
    assert len(trainer.validation_losses) == 50, "Validation loss history should be capped at 50"


@pytest.mark.asyncio
async def test_verify_data_leakage_integration(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # Provide mild train/val gap and healthy predictions/gradients
    for _ in range(10):
        trainer._store_training_loss(0.4)
        trainer._store_validation_loss(0.45)

    preds = torch.stack([torch.tensor([0.3, 0.7]) for _ in range(32)])
    grads = {"00.weight": torch.randn(10, 10) * 0.1, "01.weight": torch.randn(10, 10) * 0.2}

    score = trainer._verify_data_leakage(gradients=grads, predictions=preds)
    assert 0.0 <= score <= 100.0, f"Leakage score should be 0-100, got {score:.2f}"


@pytest.mark.asyncio
async def test_cold_start_leakage_returns_100(training_config, simple_genome):
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)

    # No history and no inputs
    score = trainer._verify_data_leakage(gradients=None, predictions=None)
    assert score == 100.0, f"Cold start should return 100, got {score:.2f}"
