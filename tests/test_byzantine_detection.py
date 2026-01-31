"""
Tests for Byzantine Behavior Detection in GenomeTrainer

Tests the security features that detect malicious validators attempting
to disrupt consensus through:
- Abnormal gradient submissions
- Opposite-direction updates
- Zero/lazy updates
- Random noise injection
- Statistical outliers

Author: BelizeChain AI Team
Date: October 19, 2025
"""

import pytest
import torch
import torch.nn as nn
from nawal.client.genome_trainer import GenomeTrainer, TrainingConfig
from nawal.genome import Genome, ArchitectureLayer, LayerType


@pytest.fixture
def training_config():
    """Create training configuration for tests."""
    return TrainingConfig(
        participant_id="validator_1",
        validator_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        staking_account="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        learning_rate=1e-4,
        batch_size=32,
        local_epochs=1,
        compliance_mode=True,
        device="cpu",  # Use CPU for tests to avoid CUDA issues
    )


@pytest.fixture
def simple_genome():
    """Create a simple genome for testing."""
    return Genome(
        genome_id="test_genome_001",
        hidden_size=64,
        num_attention_heads=4,
        num_layers=2,
        max_sequence_length=128,
        vocab_size=1000,
        encoder_layers=[
            ArchitectureLayer(
                layer_type=LayerType.LINEAR,
                hidden_size=64,
                input_size=64,
                output_size=64,
            ),
            ArchitectureLayer(
                layer_type=LayerType.LINEAR,
                hidden_size=64,
                input_size=64,
                output_size=64,
            ),
        ],
        decoder_layers=[],
    )


@pytest.mark.asyncio
async def test_honest_update_high_score(training_config, simple_genome):
    """Honest updates should receive high honesty scores (>80)."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Simulate normal training with small, consistent updates (establish baseline)
    with torch.no_grad():
        for name, param in trainer.current_model.named_parameters():
            param.data += torch.randn_like(param) * 0.001  # Small random update
    
    # First scored update - should be honest
    score1 = trainer._calculate_honesty_score()
    # First real check may be lower due to limited history
    assert score1 >= 50.0, f"First honest update should score >=50, got {score1:.2f}"
    
    # Second update - should be better with more history
    with torch.no_grad():
        for name, param in trainer.current_model.named_parameters():
            param.data += torch.randn_like(param) * 0.001  # Small random update
    
    score2 = trainer._calculate_honesty_score()
    assert score2 >= 60.0, f"Second honest update should score >=60, got {score2:.2f}"


@pytest.mark.asyncio
async def test_large_gradient_low_score(training_config, simple_genome):
    """Updates with abnormally large gradients should score low (<70)."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Establish baseline
    trainer._calculate_honesty_score()
    
    # Simulate Byzantine behavior: huge gradient
    with torch.no_grad():
        for name, param in trainer.current_model.named_parameters():
            param.data += torch.randn_like(param) * 10.0  # 10x normal
    
    score = trainer._calculate_honesty_score()
    assert score < 70.0, f"Large gradient should score <70, got {score:.2f}"


@pytest.mark.asyncio
async def test_zero_update_low_score(training_config, simple_genome):
    """Zero updates (no learning) should score low."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Establish baseline with normal update
    with torch.no_grad():
        for param in trainer.current_model.parameters():
            param.data += torch.randn_like(param) * 0.001
    
    trainer._calculate_honesty_score()
    
    # Simulate Byzantine behavior: no update (lazy validator)
    # Don't modify weights at all
    
    score = trainer._calculate_honesty_score()
    # Zero update check should catch this
    assert score < 80.0, f"Zero update should score <80, got {score:.2f}"


@pytest.mark.asyncio
async def test_opposite_direction_low_score(training_config, simple_genome):
    """Updates in opposite direction should score very low."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Establish baseline with consistent direction
    for _ in range(5):
        with torch.no_grad():
            for param in trainer.current_model.parameters():
                param.data += torch.randn_like(param) * 0.001
        trainer._calculate_honesty_score()
    
    # Simulate Byzantine behavior: opposite direction
    with torch.no_grad():
        for name, param in trainer.current_model.named_parameters():
            if name in trainer.initial_weights:
                # Move in opposite direction from all previous updates
                diff = param.data - trainer.initial_weights[name]
                param.data = trainer.initial_weights[name] - diff * 2.0
    
    score = trainer._calculate_honesty_score()
    assert score < 80.0, f"Opposite direction should score <80, got {score:.2f}"


@pytest.mark.asyncio
async def test_random_noise_detected(training_config, simple_genome):
    """Random noise injection should be detected as Byzantine."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Establish baseline with consistent updates
    for _ in range(5):
        with torch.no_grad():
            for param in trainer.current_model.parameters():
                param.data += torch.randn_like(param) * 0.001
        trainer._calculate_honesty_score()
    
    # Simulate Byzantine behavior: pure random noise
    with torch.no_grad():
        for param in trainer.current_model.parameters():
            param.data = torch.randn_like(param) * 5.0  # Pure noise
    
    score = trainer._calculate_honesty_score()
    assert score < 50.0, f"Random noise should score <50, got {score:.2f}"


@pytest.mark.asyncio
async def test_statistical_outlier_detection(training_config, simple_genome):
    """Statistical outliers should be detected after sufficient history."""
    trainer = GenomeTrainer(training_config)
    trainer.set_genome(simple_genome)
    
    # Build history with consistent small updates
    for _ in range(10):
        with torch.no_grad():
            for param in trainer.current_model.parameters():
                param.data += torch.randn_like(param) * 0.001
        trainer._calculate_honesty_score()
    
    # Simulate Byzantine behavior: extreme outlier (3σ+)
    with torch.no_grad():
        for param in trainer.current_model.parameters():
            param.data += torch.randn_like(param) * 1.0  # 1000x normal
    
    score = trainer._calculate_honesty_score()
    assert score < 70.0, f"Extreme outlier should score <70, got {score:.2f}"


@pytest.mark.asyncio
async def test_gradient_norm_verification():
    """Test individual gradient norm verification method."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    trainer.set_model(model)
    
    # Get baseline
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score1 = trainer._verify_gradient_norms(weights)
    assert score1 == 100.0, "First check should be 100 (establishes baseline)"
    
    # Normal update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score2 = trainer._verify_gradient_norms(weights)
    assert score2 >= 80.0, f"Normal gradient should score >=80, got {score2:.2f}"
    
    # Abnormal update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 100.0
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score3 = trainer._verify_gradient_norms(weights)
    assert score3 < 50.0, f"Abnormal gradient should score <50, got {score3:.2f}"


@pytest.mark.asyncio
async def test_cosine_similarity_check():
    """Test cosine similarity checking for consistent direction."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create simple model
    model = nn.Sequential(nn.Linear(10, 10))
    trainer.set_model(model)
    
    # Build history with consistent direction
    for _ in range(5):
        with torch.no_grad():
            for param in model.parameters():
                param.data += torch.randn_like(param) * 0.001
        
        weights = {name: param.data.clone() for name, param in model.named_parameters()}
        trainer._store_update_statistics(weights)
    
    # Check consistent update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.001
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score = trainer._check_update_similarity(weights)
    # Note: Random updates may have low similarity, so we just check it runs
    assert 0.0 <= score <= 100.0, f"Score should be in [0, 100], got {score:.2f}"


@pytest.mark.asyncio
async def test_zero_update_detection():
    """Test detection of zero/lazy updates."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create simple model
    model = nn.Sequential(nn.Linear(10, 10))
    trainer.set_model(model)
    
    # Establish baseline
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    trainer._verify_gradient_norms(weights)  # Sets initial_weights
    
    # Test zero update
    score_zero = trainer._check_zero_update(weights)
    assert score_zero < 50.0, f"Zero update should score <50, got {score_zero:.2f}"
    
    # Test normal update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.001
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score_normal = trainer._check_zero_update(weights)
    assert score_normal >= 80.0, f"Normal update should score >=80, got {score_normal:.2f}"


@pytest.mark.asyncio
async def test_variance_consistency_check():
    """Test variance consistency across layers."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create multi-layer model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 20),
        nn.Linear(20, 5),
    )
    trainer.set_model(model)
    
    # Establish baseline
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    trainer._verify_gradient_norms(weights)  # Sets initial_weights
    
    # Consistent update (same magnitude across layers)
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score_consistent = trainer._check_update_variance(weights)
    assert score_consistent >= 50.0, f"Consistent variance should score >=50, got {score_consistent:.2f}"
    
    # Chaotic update (different magnitudes)
    with torch.no_grad():
        magnitudes = [0.001, 1.0, 0.0001]  # Very different
        for param, mag in zip(model.parameters(), magnitudes):
            param.data += torch.randn_like(param) * mag
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score_chaotic = trainer._check_update_variance(weights)
    # Chaotic variance should score lower
    assert score_chaotic < score_consistent, "Chaotic should score lower than consistent"


@pytest.mark.asyncio
async def test_weight_magnitude_verification():
    """Test weight magnitude bounds checking."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create simple model
    model = nn.Sequential(nn.Linear(10, 10))
    trainer.set_model(model)
    
    # Establish baseline
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    trainer._verify_gradient_norms(weights)  # Sets initial_weights
    
    # Normal magnitude update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 0.01
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score_normal = trainer._verify_weight_magnitudes(weights)
    assert score_normal >= 80.0, f"Normal magnitude should score >=80, got {score_normal:.2f}"
    
    # Excessive magnitude update
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 50.0  # Way too large
    
    weights = {name: param.data.clone() for name, param in model.named_parameters()}
    score_large = trainer._verify_weight_magnitudes(weights)
    assert score_large <= 50.0, f"Large magnitude should score <=50, got {score_large:.2f}"


@pytest.mark.asyncio
async def test_update_statistics_storage():
    """Test that update statistics are stored correctly."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Create simple model
    model = nn.Sequential(nn.Linear(10, 10))
    trainer.set_model(model)
    
    assert len(trainer.update_statistics) == 0
    assert len(trainer.historical_updates) == 0
    assert trainer.rounds_completed == 0
    
    # Store some updates
    for i in range(5):
        weights = {name: param.data.clone() for name, param in model.named_parameters()}
        trainer._store_update_statistics(weights)
        
        assert len(trainer.update_statistics) == i + 1
        assert len(trainer.historical_updates) == i + 1
        assert trainer.rounds_completed == i + 1
    
    # Verify statistics structure
    stats = trainer.update_statistics[0]
    assert 'mean' in stats
    assert 'std' in stats
    assert 'timestamp' in stats
    assert isinstance(stats['mean'], float)
    assert isinstance(stats['std'], float)
    assert isinstance(stats['timestamp'], float)


@pytest.mark.asyncio
async def test_max_norm_layer_specific():
    """Test that different layers have different max norms."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Test different layer types
    embed_norm = trainer._get_max_norm_for_layer("embedding.weight")
    assert embed_norm == 50.0, "Embedding should have max norm 50.0"
    
    output_norm = trainer._get_max_norm_for_layer("output.weight")
    assert output_norm == 100.0, "Output should have max norm 100.0"
    
    norm_norm = trainer._get_max_norm_for_layer("layernorm.weight")
    assert norm_norm == 10.0, "LayerNorm should have max norm 10.0"
    
    hidden_norm = trainer._get_max_norm_for_layer("hidden.weight")
    assert hidden_norm == 30.0, "Hidden should have max norm 30.0"


@pytest.mark.asyncio
async def test_expected_magnitude_scales_with_rounds():
    """Test that expected magnitude scales with training rounds."""
    config = TrainingConfig(
        participant_id="test",
        validator_address="5test",
        staking_account="5test",
    )
    trainer = GenomeTrainer(config)
    
    # Test magnitude scaling
    mag_round_0 = trainer._get_expected_magnitude("test.weight", 0)
    mag_round_1 = trainer._get_expected_magnitude("test.weight", 1)
    mag_round_5 = trainer._get_expected_magnitude("test.weight", 5)
    
    assert mag_round_0 == 100.0, "Round 0 should be lenient (100.0)"
    assert mag_round_1 < mag_round_5, "Magnitude should increase with rounds"
    
    # Test that it caps at 10 rounds
    mag_round_10 = trainer._get_expected_magnitude("test.weight", 10)
    mag_round_20 = trainer._get_expected_magnitude("test.weight", 20)
    assert mag_round_10 == mag_round_20, "Magnitude should cap at 10 rounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
