"""
Tests for differential privacy verification in federated learning.

Tests cover:
- Gradient clipping compliance
- Noise injection verification
- Privacy budget tracking
- Integration with honesty scoring
- Storage limits

Author: BelizeChain Team
"""

import pytest
import torch
from unittest.mock import Mock

from nawal.client.genome_trainer import GenomeTrainer, TrainingConfig
from nawal.genome.dna import Genome, ArchitectureLayer, LayerType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def training_config():
    """Create training configuration."""
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
    """Create simple genome for testing."""
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


@pytest.fixture
def mock_dp_config():
    """Create mock differential privacy configuration."""
    dp_config = Mock()
    dp_config.clip_norm = 1.0
    dp_config.noise_multiplier = 1.0
    dp_config.budget = Mock()
    dp_config.budget.epsilon = 1.0
    dp_config.budget.delta = 1e-5
    dp_config.budget.spent_epsilon = 0.0
    dp_config.budget.steps = 0
    return dp_config


# =============================================================================
# Gradient Clipping Tests
# =============================================================================


@pytest.mark.asyncio
async def test_proper_gradient_clipping_high_score(training_config, simple_genome, mock_dp_config):
    """Properly clipped gradients should score high."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Simulate properly clipped gradients (all below clip_norm=1.0)
    clipped_gradients = {
        "layer1.weight": torch.randn(32, 10) * 0.5,  # Norm ~0.5
        "layer1.bias": torch.randn(32) * 0.3,         # Norm ~0.3
        "layer2.weight": torch.randn(10, 32) * 0.7,  # Norm ~0.7
        "layer2.bias": torch.randn(10) * 0.2,         # Norm ~0.2
    }
    
    score = trainer._check_gradient_clipping(clipped_gradients, mock_dp_config)
    
    assert score >= 95.0, f"Properly clipped gradients should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_unclipped_gradients_low_score(training_config, simple_genome, mock_dp_config):
    """Unclipped gradients exceeding clip_norm should score low."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Simulate unclipped gradients (exceeding clip_norm=1.0)
    unclipped_gradients = {
        "layer1.weight": torch.randn(32, 10) * 5.0,  # Norm ~5.0 >> 1.0
        "layer1.bias": torch.randn(32) * 3.0,         # Norm ~3.0 >> 1.0
        "layer2.weight": torch.randn(10, 32) * 4.0,  # Norm ~4.0 >> 1.0
        "layer2.bias": torch.randn(10) * 2.0,         # Norm ~2.0 >> 1.0
    }
    
    score = trainer._check_gradient_clipping(unclipped_gradients, mock_dp_config)
    
    assert score <= 20.0, f"Unclipped gradients should score <=20, got {score:.2f}"


@pytest.mark.asyncio
async def test_inconsistent_clipping_detected(training_config, simple_genome, mock_dp_config):
    """Inconsistent clipping (some OK, some not) should be detected."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Simulate mixed clipping (50% clipped, 50% not)
    mixed_gradients = {
        "layer1.weight": torch.randn(32, 10) * 0.5,  # OK: ~0.5
        "layer1.bias": torch.randn(32) * 0.3,         # OK: ~0.3
        "layer2.weight": torch.randn(10, 32) * 3.0,  # BAD: ~3.0
        "layer2.bias": torch.randn(10) * 2.5,         # BAD: ~2.5
    }
    
    score = trainer._check_gradient_clipping(mixed_gradients, mock_dp_config)
    
    # Should detect 50% violation
    assert 30.0 <= score <= 70.0, f"Mixed clipping should score 30-70, got {score:.2f}"


@pytest.mark.asyncio
async def test_clipping_history_tracking(training_config, simple_genome, mock_dp_config):
    """Gradient clip history should be tracked properly."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add multiple gradient checks
    for i in range(5):
        gradients = {"layer1.weight": torch.randn(32, 10) * 0.8}
        trainer._check_gradient_clipping(gradients, mock_dp_config)
    
    # Check history is populated
    assert len(trainer.gradient_clip_history) == 5, "Should track 5 gradient norms"
    assert all(0.0 <= norm <= 20.0 for norm in trainer.gradient_clip_history), "Norms should be reasonable"


# =============================================================================
# Privacy Budget Tests
# =============================================================================


@pytest.mark.asyncio
async def test_privacy_budget_healthy_high_score(training_config, simple_genome, mock_dp_config):
    """Healthy privacy budget (plenty remaining) should score high."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Set healthy budget (spent 10% of epsilon)
    mock_dp_config.budget.epsilon = 1.0
    mock_dp_config.budget.spent_epsilon = 0.1
    mock_dp_config.budget.steps = 100
    
    score = trainer._check_privacy_budget(mock_dp_config)
    
    assert score >= 95.0, f"Healthy budget (90% remaining) should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_privacy_budget_exhausted_low_score(training_config, simple_genome, mock_dp_config):
    """Exhausted privacy budget should score 0."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Set exhausted budget (spent > epsilon)
    mock_dp_config.budget.epsilon = 1.0
    mock_dp_config.budget.spent_epsilon = 1.2  # Over budget!
    mock_dp_config.budget.steps = 1000
    
    score = trainer._check_privacy_budget(mock_dp_config)
    
    assert score == 0.0, f"Exhausted budget should score 0, got {score:.2f}"


@pytest.mark.asyncio
async def test_privacy_budget_near_limit_warning(training_config, simple_genome, mock_dp_config):
    """Budget near exhaustion (80% spent) should score moderately."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Set budget at 80% spent (20% remaining)
    mock_dp_config.budget.epsilon = 1.0
    mock_dp_config.budget.spent_epsilon = 0.8
    mock_dp_config.budget.steps = 800
    
    score = trainer._check_privacy_budget(mock_dp_config)
    
    # Should be in warning range (50-75 points)
    assert 40.0 <= score <= 80.0, f"Near-exhausted budget should score 40-80, got {score:.2f}"


@pytest.mark.asyncio
async def test_budget_tracking_accuracy(training_config, simple_genome, mock_dp_config):
    """Privacy budget tracking should be accurate."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Check multiple budget states
    for i in range(5):
        mock_dp_config.budget.spent_epsilon = 0.2 * i  # 0.0, 0.2, 0.4, 0.6, 0.8
        mock_dp_config.budget.steps = 100 * i
        trainer._check_privacy_budget(mock_dp_config)
    
    # Check history is populated
    assert len(trainer.privacy_spent_history) == 5, "Should track 5 budget states"
    
    # Check values are sensible
    for spent, steps in trainer.privacy_spent_history:
        assert 0.0 <= spent <= 1.0, f"Spent epsilon should be 0-1.0, got {spent}"
        assert 0 <= steps <= 500, f"Steps should be 0-500, got {steps}"


# =============================================================================
# Noise Consistency Tests
# =============================================================================


@pytest.mark.asyncio
async def test_proper_noise_consistency_high_score(training_config, simple_genome, mock_dp_config):
    """Consistent noise injection should score high."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add consistent noise history (expected: noise_multiplier * clip_norm = 1.0)
    expected_noise = 1.0 * 1.0  # noise_multiplier * clip_norm
    for _ in range(10):
        trainer._store_noise_scale(expected_noise + torch.randn(1).item() * 0.05)  # Small variance
    
    score = trainer._check_noise_consistency(mock_dp_config)
    
    assert score >= 95.0, f"Consistent noise should score >=95, got {score:.2f}"


@pytest.mark.asyncio
async def test_insufficient_noise_detected(training_config, simple_genome, mock_dp_config):
    """Insufficient noise (too low) should be detected."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add insufficient noise history (expected: 1.0, actual: 0.3)
    for _ in range(10):
        trainer._store_noise_scale(0.3)  # 70% too low!
    
    score = trainer._check_noise_consistency(mock_dp_config)
    
    assert score <= 70.0, f"Insufficient noise should score <=70, got {score:.2f}"


@pytest.mark.asyncio
async def test_excessive_noise_detected(training_config, simple_genome, mock_dp_config):
    """Excessive noise (too high) should be detected."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add excessive noise history (expected: 1.0, actual: 3.0)
    for _ in range(10):
        trainer._store_noise_scale(3.0)  # 200% too high!
    
    score = trainer._check_noise_consistency(mock_dp_config)
    
    assert score <= 50.0, f"Excessive noise should score <=50, got {score:.2f}"


@pytest.mark.asyncio
async def test_noise_history_tracking(training_config, simple_genome):
    """Noise scale history should be tracked with limits."""
    trainer = GenomeTrainer(training_config)
    
    # Add 50 noise scales (should keep last 30)
    for i in range(50):
        trainer._store_noise_scale(1.0 + i * 0.01)
    
    assert len(trainer.noise_scale_history) == 30, "Should keep last 30 noise scales"
    assert trainer.noise_scale_history[0] == pytest.approx(1.2, abs=0.01), "Should keep recent values"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_dp_compliance_clean(training_config, simple_genome, mock_dp_config):
    """Full DP compliance with clean data should score high."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    trainer.dp_config = mock_dp_config
    
    # Simulate properly clipped gradients
    clean_gradients = {
        "layer1.weight": torch.randn(32, 10) * 0.8,
        "layer2.weight": torch.randn(10, 32) * 0.7,
    }
    
    # Add some noise history
    for _ in range(10):
        trainer._store_noise_scale(1.0)
    
    # Set healthy budget
    mock_dp_config.budget.spent_epsilon = 0.2
    
    score = trainer._verify_differential_privacy(clean_gradients, mock_dp_config)
    
    assert score >= 85.0, f"Full DP compliance should score >=85, got {score:.2f}"


@pytest.mark.asyncio
async def test_dp_violations_detected(training_config, simple_genome, mock_dp_config):
    """DP violations should be detected and scored low."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    trainer.dp_config = mock_dp_config
    
    # Simulate unclipped gradients
    bad_gradients = {
        "layer1.weight": torch.randn(32, 10) * 5.0,  # Unclipped!
        "layer2.weight": torch.randn(10, 32) * 4.0,  # Unclipped!
    }
    
    # Set exhausted budget
    mock_dp_config.budget.spent_epsilon = 1.5  # Over budget!
    
    score = trainer._verify_differential_privacy(bad_gradients, mock_dp_config)
    
    assert score <= 30.0, f"DP violations should score <=30, got {score:.2f}"


@pytest.mark.asyncio
async def test_dp_disabled_graceful_skip(training_config, simple_genome):
    """When DP is disabled, verification should gracefully return 100."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # No DP config set
    assert trainer.dp_config is None
    
    # Should return perfect score (not applicable)
    gradients = {"layer1.weight": torch.randn(32, 10)}
    score = trainer._verify_differential_privacy(gradients, None)
    
    assert score == 100.0, f"Disabled DP should score 100, got {score:.2f}"


@pytest.mark.asyncio
async def test_honesty_score_integrates_dp(training_config, simple_genome, mock_dp_config):
    """Honesty score should integrate DP checks when enabled."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    trainer.dp_config = mock_dp_config
    
    # Simulate clean gradients
    clean_gradients = {
        "layer1.weight": torch.randn(32, 10) * 0.5,
        "layer2.weight": torch.randn(10, 32) * 0.5,
    }
    
    # Set healthy budget
    mock_dp_config.budget.spent_epsilon = 0.1
    
    # Calculate honesty with DP
    honesty_score = trainer._calculate_honesty_score(gradients=clean_gradients)
    
    # Should produce valid score
    assert 0.0 <= honesty_score <= 100.0, f"Honesty score should be 0-100, got {honesty_score:.2f}"


@pytest.mark.asyncio
async def test_cold_start_with_dp(training_config, simple_genome, mock_dp_config):
    """DP verification should handle cold start (no history)."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    trainer.dp_config = mock_dp_config
    
    # No noise history yet
    assert len(trainer.noise_scale_history) == 0
    
    gradients = {"layer1.weight": torch.randn(32, 10) * 0.8}
    score = trainer._verify_differential_privacy(gradients, mock_dp_config)
    
    # Should handle gracefully (only clipping + budget checks)
    assert score >= 50.0, f"Cold start DP should score >=50, got {score:.2f}"


# =============================================================================
# Storage Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gradient_clip_history_limits(training_config, simple_genome, mock_dp_config):
    """Gradient clip history should respect 30-item limit."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add 50 gradient checks (should keep last 30)
    for i in range(50):
        gradients = {"layer1.weight": torch.randn(32, 10) * 0.5}
        trainer._check_gradient_clipping(gradients, mock_dp_config)
    
    assert len(trainer.gradient_clip_history) == 30, "Should maintain 30-item limit"


@pytest.mark.asyncio
async def test_noise_scale_history_limits(training_config, simple_genome):
    """Noise scale history should respect 30-item limit."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add 50 noise scales
    for i in range(50):
        trainer._store_noise_scale(1.0 + i * 0.01)
    
    assert len(trainer.noise_scale_history) == 30, "Should maintain 30-item limit"


@pytest.mark.asyncio
async def test_privacy_spent_history_limits(training_config, simple_genome, mock_dp_config):
    """Privacy spent history should respect 50-item limit."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Add 100 budget checks (should keep last 50)
    for i in range(100):
        mock_dp_config.budget.spent_epsilon = 0.01 * i
        mock_dp_config.budget.steps = i
        trainer._check_privacy_budget(mock_dp_config)
    
    assert len(trainer.privacy_spent_history) == 50, "Should maintain 50-item limit"
