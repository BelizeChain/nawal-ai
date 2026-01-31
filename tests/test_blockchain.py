"""
Blockchain Integration Tests

Tests for staking connector, reward calculation, and event listening.
All tests use mock mode to avoid requiring actual blockchain connection.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

# Import blockchain components
from pathlib import Path

from blockchain.staking_connector import (
    StakingConnector,
    ParticipantInfo,
    TrainingSubmission,
)
from blockchain.rewards import (
    RewardCalculator,
    RewardDistributor,
    FitnessScores,
    RewardCalculation,
    dalla_to_planck,
    planck_to_dalla,
)
from blockchain.events import (
    BlockchainEventListener,
    EventType,
    TrainingEvent,
    create_training_round_handler,
)


# =============================================================================
# Utility Tests
# =============================================================================

class TestCurrencyConversion:
    """Test DALLA/planck conversion utilities."""
    
    def test_dalla_to_planck(self):
        """Test DALLA to planck conversion."""
        assert dalla_to_planck(1.0) == 1_000_000_000_000
        assert dalla_to_planck(5.5) == 5_500_000_000_000
        assert dalla_to_planck(1000) == 1_000_000_000_000_000
        assert dalla_to_planck(0.000001) == 1_000_000
    
    def test_planck_to_dalla(self):
        """Test planck to DALLA conversion."""
        assert planck_to_dalla(1_000_000_000_000) == 1.0
        assert planck_to_dalla(5_500_000_000_000) == 5.5
        assert planck_to_dalla(1_000_000_000_000_000) == 1000.0
        assert planck_to_dalla(1_000_000) == 0.000001
    
    def test_roundtrip_conversion(self):
        """Test that conversions are reversible."""
        original_dalla = 12345.6789
        planck = dalla_to_planck(original_dalla)
        converted_back = planck_to_dalla(planck)
        assert abs(converted_back - original_dalla) < 1e-6


# =============================================================================
# Staking Connector Tests
# =============================================================================

class TestStakingConnector:
    """Test blockchain staking pallet integration."""
    
    @pytest.fixture
    def connector(self):
        """Create mock staking connector."""
        return StakingConnector(
            node_url="ws://localhost:9944",
            mock_mode=True
        )
    
    @pytest.fixture
    def mock_keypair(self):
        """Create mock keypair for signing."""
        keypair = Mock()
        keypair.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        return keypair
    
    @pytest.mark.asyncio
    async def test_enroll_participant_success(self, connector, mock_keypair):
        """Test successful participant enrollment."""
        account_id = "alice"
        stake_amount = dalla_to_planck(5000)  # 5K DALLA
        
        success = await connector.enroll_participant(
            account_id=account_id,
            stake_amount=stake_amount,
            keypair=mock_keypair
        )
        
        assert success is True
        
        # Verify participant info was stored
        info = await connector.get_participant_info(account_id)
        assert info is not None
        assert info.account_id == account_id
        assert info.stake_amount == stake_amount
        assert info.is_enrolled is True
    
    @pytest.mark.asyncio
    async def test_enroll_duplicate_participant(self, connector, mock_keypair):
        """Test enrolling same participant twice fails."""
        account_id = "bob"
        stake_amount = dalla_to_planck(3000)
        
        # First enrollment should succeed
        success1 = await connector.enroll_participant(
            account_id, stake_amount, mock_keypair
        )
        assert success1 is True
        
        # Second enrollment should fail
        success2 = await connector.enroll_participant(
            account_id, stake_amount, mock_keypair
        )
        assert success2 is False
    
    @pytest.mark.asyncio
    async def test_submit_training_proof_success(self, connector, mock_keypair):
        """Test successful training proof submission."""
        # Enroll participant first
        account_id = "charlie"
        await connector.enroll_participant(
            account_id,
            dalla_to_planck(2000),
            mock_keypair
        )
        
        # Create training submission
        submission = TrainingSubmission(
            participant_id=account_id,
            round_number=1,
            genome_id="genome_001",
            samples_trained=1000,
            training_time=120.5,
            quality_score=85.0,
            timeliness_score=90.0,
            honesty_score=95.0,
            fitness_score=89.5,  # Weighted average
            model_hash="QmXyZ123"
        )
        
        success = await connector.submit_training_proof(
            submission, mock_keypair
        )
        
        assert success is True
        
        # Verify participant stats updated
        info = await connector.get_participant_info(account_id)
        assert info.training_rounds_completed == 1
        assert info.avg_fitness_score > 0
    
    @pytest.mark.asyncio
    async def test_submit_proof_unenrolled_participant(self, connector, mock_keypair):
        """Test submitting proof without enrollment fails."""
        submission = TrainingSubmission(
            participant_id="unknown",
            round_number=1,
            genome_id="genome_002",
            samples_trained=500,
            training_time=100.0,
            quality_score=80.0,
            timeliness_score=80.0,
            honesty_score=80.0,
            fitness_score=80.0,
            model_hash="QmTest"
        )
        
        success = await connector.submit_training_proof(
            submission, mock_keypair
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_claim_rewards_success(self, connector, mock_keypair):
        """Test successful reward claiming."""
        # Enroll and train
        account_id = "dave"
        await connector.enroll_participant(
            account_id,
            dalla_to_planck(4000),
            mock_keypair
        )
        
        submission = TrainingSubmission(
            participant_id=account_id,
            round_number=1,
            genome_id="genome_003",
            samples_trained=1200,
            training_time=110.0,
            quality_score=90.0,
            timeliness_score=85.0,
            honesty_score=92.0,
            fitness_score=88.9,  # Weighted average
            model_hash="QmReward"
        )
        
        await connector.submit_training_proof(submission, mock_keypair)
        
        # Claim rewards (returns tuple: success, amount)
        success, rewards = await connector.claim_rewards(account_id, mock_keypair)
        
        # Should have earned some rewards
        assert success is True
        assert rewards > 0
        assert rewards > dalla_to_planck(5)  # At least 5 DALLA
    
    @pytest.mark.asyncio
    async def test_unenroll_participant(self, connector, mock_keypair):
        """Test participant unenrollment."""
        account_id = "eve"
        await connector.enroll_participant(
            account_id,
            dalla_to_planck(1000),
            mock_keypair
        )
        
        # Verify enrolled
        info1 = await connector.get_participant_info(account_id)
        assert info1.is_enrolled is True
        
        # Unenroll
        success = await connector.unenroll_participant(account_id, mock_keypair)
        assert success is True
        
        # Verify unenrolled
        info2 = await connector.get_participant_info(account_id)
        assert info2.is_enrolled is False


# =============================================================================
# Reward Calculator Tests
# =============================================================================

class TestRewardCalculator:
    """Test reward calculation logic."""
    
    @pytest.fixture
    def calculator(self):
        """Create reward calculator."""
        return RewardCalculator()
    
    def test_calculate_overall_fitness(self, calculator):
        """Test fitness score calculation."""
        fitness = FitnessScores(
            quality=80.0,
            timeliness=90.0,
            honesty=85.0
        )
        
        overall = fitness.calculate_overall()
        
        # Formula: 0.40×80 + 0.30×90 + 0.30×85 = 32 + 27 + 25.5 = 84.5
        assert abs(overall - 84.5) < 0.01
    
    def test_calculate_stake_multiplier_ranges(self, calculator):
        """Test stake multiplier for different amounts."""
        # Minimum stake (1K DALLA) = 1.0x
        mult1 = calculator.calculate_stake_multiplier(1000)
        assert mult1 == 1.0
        
        # Mid-range (5K DALLA) ≈ 1.4x
        mult2 = calculator.calculate_stake_multiplier(5000)
        assert 1.3 < mult2 < 1.5
        
        # Maximum stake (10K+ DALLA) = 2.0x
        mult3 = calculator.calculate_stake_multiplier(10000)
        assert mult3 == 2.0
        
        mult4 = calculator.calculate_stake_multiplier(20000)
        assert mult4 == 2.0
    
    def test_calculate_reward_perfect_performance(self, calculator):
        """Test reward with perfect fitness scores."""
        fitness = FitnessScores(
            quality=100.0,
            timeliness=100.0,
            honesty=100.0
        )
        
        stake_amount = dalla_to_planck(10000)  # Max stake
        
        reward = calculator.calculate_reward(
            participant_id="alice",
            round_number=1,
            fitness_scores=fitness,
            stake_amount_planck=stake_amount
        )
        
        # Perfect score + max stake = 10 × 1.0 × 2.0 = 20 DALLA
        assert abs(reward.total_reward_dalla - 20.0) < 0.01
        assert reward.stake_multiplier == 2.0
        assert reward.fitness_multiplier == 1.0
    
    def test_calculate_reward_average_performance(self, calculator):
        """Test reward with average fitness scores."""
        fitness = FitnessScores(
            quality=75.0,
            timeliness=80.0,
            honesty=70.0
        )
        
        stake_amount = dalla_to_planck(3000)
        
        reward = calculator.calculate_reward(
            participant_id="bob",
            round_number=2,
            fitness_scores=fitness,
            stake_amount_planck=stake_amount
        )
        
        # Overall fitness ≈ 75.5, stake mult ≈ 1.22
        # Reward ≈ 10 × 0.755 × 1.22 ≈ 9.2 DALLA
        assert 8.5 < reward.total_reward_dalla < 10.0
        assert 0.7 < reward.fitness_multiplier < 0.8
    
    def test_calculate_reward_poor_performance(self, calculator):
        """Test reward with poor fitness scores."""
        fitness = FitnessScores(
            quality=30.0,
            timeliness=40.0,
            honesty=50.0
        )
        
        stake_amount = dalla_to_planck(2000)
        
        reward = calculator.calculate_reward(
            participant_id="charlie",
            round_number=3,
            fitness_scores=fitness,
            stake_amount_planck=stake_amount
        )
        
        # Overall fitness ≈ 38, should get low reward
        assert reward.total_reward_dalla < 5.0
        assert reward.fitness_multiplier < 0.4
    
    def test_estimate_monthly_rewards(self, calculator):
        """Test monthly reward estimation."""
        # 4 rounds/day, 80% avg fitness, 5K DALLA staked
        monthly = calculator.estimate_monthly_rewards(
            rounds_per_day=4,
            avg_fitness=80.0,
            stake_amount_dalla=5000.0
        )
        
        # Approx: 4 × 30 × 10 × 0.8 × 1.4 ≈ 1,344 DALLA
        assert 1200 < monthly < 1500


class TestRewardDistributor:
    """Test reward distribution and tracking."""
    
    @pytest.fixture
    def distributor(self):
        """Create reward distributor."""
        return RewardDistributor()
    
    def test_add_pending_reward(self, distributor):
        """Test adding pending rewards."""
        fitness_scores = FitnessScores(85.0, 90.0, 88.0)
        calculation = RewardCalculation(
            participant_id="alice",
            round_number=1,
            fitness_scores=fitness_scores,
            overall_fitness=fitness_scores.calculate_overall(),
            stake_amount_dalla=5000.0,
            stake_multiplier=1.5,
            base_reward_dalla=10.0,
            fitness_multiplier=0.875,
            stake_bonus_dalla=5.0,
            total_reward_dalla=13.125,
            total_reward_planck=dalla_to_planck(13.125)
        )
        
        distributor.add_pending_reward(calculation)
        
        pending = distributor.get_total_pending("alice")
        assert pending == calculation.total_reward_planck
    
    def test_multiple_pending_rewards(self, distributor):
        """Test accumulating multiple pending rewards."""
        for i in range(5):
            fitness_scores = FitnessScores(80.0, 80.0, 80.0)
            calculation = RewardCalculation(
                participant_id="bob",
                round_number=i + 1,
                fitness_scores=fitness_scores,
                overall_fitness=fitness_scores.calculate_overall(),
                stake_amount_dalla=3000.0,
                stake_multiplier=1.2,
                base_reward_dalla=10.0,
                fitness_multiplier=0.8,
                stake_bonus_dalla=2.0,
                total_reward_dalla=9.6,
                total_reward_planck=dalla_to_planck(9.6)
            )
            distributor.add_pending_reward(calculation)
        
        total_pending = distributor.get_total_pending("bob")
        expected = dalla_to_planck(9.6) * 5
        assert total_pending == expected
    
    def test_mark_distributed(self, distributor):
        """Test marking rewards as distributed."""
        # Add pending rewards
        fitness_scores = FitnessScores(90.0, 90.0, 90.0)
        calculation = RewardCalculation(
            participant_id="charlie",
            round_number=1,
            fitness_scores=fitness_scores,
            overall_fitness=fitness_scores.calculate_overall(),
            stake_amount_dalla=7000.0,
            stake_multiplier=1.8,
            base_reward_dalla=10.0,
            fitness_multiplier=0.9,
            stake_bonus_dalla=8.0,
            total_reward_dalla=16.2,
            total_reward_planck=dalla_to_planck(16.2)
        )
        distributor.add_pending_reward(calculation)
        
        # Distribute
        amount = distributor.get_total_pending("charlie")
        distributor.mark_distributed("charlie", amount)
        
        # Should have no pending rewards
        assert distributor.get_total_pending("charlie") == 0
        
        # Should track distributed amount
        distributed = distributor.get_total_distributed("charlie")
        assert distributed == amount
    
    def test_get_statistics(self, distributor):
        """Test overall distribution statistics."""
        # Add rewards for multiple participants
        for participant_id in ["alice", "bob", "charlie"]:
            for round_id in range(3):
                fitness_scores = FitnessScores(85.0, 85.0, 85.0)
                calculation = RewardCalculation(
                    participant_id=participant_id,
                    round_number=round_id + 1,
                    fitness_scores=fitness_scores,
                    overall_fitness=fitness_scores.calculate_overall(),
                    stake_amount_dalla=5000.0,
                    stake_multiplier=1.5,
                    base_reward_dalla=10.0,
                    fitness_multiplier=0.85,
                    stake_bonus_dalla=5.0,
                    total_reward_dalla=12.75,
                    total_reward_planck=dalla_to_planck(12.75)
                )
                distributor.add_pending_reward(calculation)
        
        stats = distributor.get_statistics()
        
        assert stats["participants_with_pending"] == 3
        assert stats["total_pending_planck"] == dalla_to_planck(12.75) * 9
        assert stats["total_pending_dalla"] == 12.75 * 9


# =============================================================================
# Event Listener Tests
# =============================================================================

class TestBlockchainEventListener:
    """Test blockchain event listening and handling."""
    
    @pytest.fixture
    def listener(self):
        """Create mock event listener."""
        return BlockchainEventListener(
            node_url="ws://localhost:9944",
            mock_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_register_handler(self, listener):
        """Test event handler registration."""
        events_received = []
        
        def handler(event: TrainingEvent):
            events_received.append(event)
        
        listener.register_handler(
            EventType.TRAINING_ROUND_STARTED,
            handler
        )
        
        # Emit mock event
        await listener.emit_mock_event(
            EventType.TRAINING_ROUND_STARTED,
            {"round_id": 42, "global_model_hash": "QmTest123"}
        )
        
        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.TRAINING_ROUND_STARTED
        assert events_received[0].data["round_id"] == 42
    
    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, listener):
        """Test multiple handlers for same event type."""
        handler1_calls = []
        handler2_calls = []
        
        def handler1(event):
            handler1_calls.append(event)
        
        def handler2(event):
            handler2_calls.append(event)
        
        listener.register_handler(EventType.TRAINER_ENROLLED, handler1)
        listener.register_handler(EventType.TRAINER_ENROLLED, handler2)
        
        await listener.emit_mock_event(
            EventType.TRAINER_ENROLLED,
            {"account_id": "alice", "stake_amount": dalla_to_planck(5000)}
        )
        
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
    
    @pytest.mark.asyncio
    async def test_different_event_types(self, listener):
        """Test handling different event types."""
        training_events = []
        reward_events = []
        
        listener.register_handler(
            EventType.TRAINING_PROOF_SUBMITTED,
            lambda e: training_events.append(e)
        )
        listener.register_handler(
            EventType.REWARDS_CLAIMED,
            lambda e: reward_events.append(e)
        )
        
        # Emit different events
        await listener.emit_mock_event(
            EventType.TRAINING_PROOF_SUBMITTED,
            {"participant_id": "bob", "round_id": 1}
        )
        await listener.emit_mock_event(
            EventType.REWARDS_CLAIMED,
            {"participant_id": "bob", "amount": dalla_to_planck(15)}
        )
        await listener.emit_mock_event(
            EventType.TRAINING_PROOF_SUBMITTED,
            {"participant_id": "charlie", "round_id": 1}
        )
        
        assert len(training_events) == 2
        assert len(reward_events) == 1
    
    @pytest.mark.asyncio
    async def test_event_history_tracking(self, listener):
        """Test event history is maintained."""
        # Emit several events
        for i in range(5):
            await listener.emit_mock_event(
                EventType.TRAINING_ROUND_STARTED,
                {"round_id": i + 1}
            )
        
        history = listener.get_event_history()
        assert len(history) == 5
        
        # Events should be in chronological order
        for i, event in enumerate(history):
            assert event.data["round_id"] == i + 1
    
    @pytest.mark.asyncio
    async def test_event_history_limit(self, listener):
        """Test event history has maximum size."""
        # Emit more than max history size (1000)
        for i in range(1100):
            await listener.emit_mock_event(
                EventType.TRAINING_ROUND_STARTED,
                {"round_id": i + 1}
            )
        
        history = listener.get_event_history(limit=10000)  # Get all events
        
        # Should only keep last 1000 events (max_history_size)
        assert len(history) == 1000
        
        # Should be the most recent ones (101-1100)
        assert history[0].data["round_id"] == 101
        assert history[-1].data["round_id"] == 1100
    
    @pytest.mark.asyncio
    async def test_create_training_round_handler(self):
        """Test convenience handler creator."""
        calls = []
        
        async def on_round_start(round_num, genome_id):
            calls.append((round_num, genome_id))
        
        handlers = await create_training_round_handler(on_round_started=on_round_start)
        
        # Create mock event
        event = TrainingEvent(
            event_type=EventType.TRAINING_ROUND_STARTED,
            block_number=100,
            block_hash="0x1234567890abcdef",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "round_number": 42,
                "genome_id": "genome_001"
            }
        )
        
        # Call handler
        handler_fn = handlers[EventType.TRAINING_ROUND_STARTED]
        await handler_fn(event)
        
        assert len(calls) == 1
        assert calls[0] == (42, "genome_001")


# =============================================================================
# Integration Tests
# =============================================================================

class TestBlockchainIntegration:
    """Test integration between blockchain components."""
    
    @pytest.fixture
    def setup(self):
        """Create integrated test environment."""
        connector = StakingConnector(mock_mode=True)
        calculator = RewardCalculator()
        distributor = RewardDistributor()
        listener = BlockchainEventListener(mock_mode=True)
        
        return {
            "connector": connector,
            "calculator": calculator,
            "distributor": distributor,
            "listener": listener,
            "keypair": Mock(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        }
    
    @pytest.mark.asyncio
    async def test_full_training_cycle(self, setup):
        """Test complete PoUW training cycle."""
        connector = setup["connector"]
        calculator = setup["calculator"]
        distributor = setup["distributor"]
        listener = setup["listener"]
        keypair = setup["keypair"]
        
        # Step 1: Enroll participant
        participant_id = "alice"
        stake_amount = dalla_to_planck(5000)
        
        enrolled = await connector.enroll_participant(
            participant_id, stake_amount, keypair
        )
        assert enrolled is True
        
        # Step 2: Listen for training round event
        training_started = asyncio.Event()
        received_round_id = None
        
        def on_training_start(event):
            nonlocal received_round_id
            received_round_id = event.data["round_id"]
            training_started.set()
        
        listener.register_handler(
            EventType.TRAINING_ROUND_STARTED,
            on_training_start
        )
        
        # Emit training round event
        await listener.emit_mock_event(
            EventType.TRAINING_ROUND_STARTED,
            {"round_id": 1, "global_model_hash": "QmGlobal123"}
        )
        
        await asyncio.wait_for(training_started.wait(), timeout=1.0)
        assert received_round_id == 1
        
        # Step 3: Submit training proof
        submission = TrainingSubmission(
            participant_id=participant_id,
            round_number=1,
            genome_id="genome_test_001",
            samples_trained=1500,
            training_time=115.3,
            quality_score=88.0,
            timeliness_score=92.0,
            honesty_score=90.0,
            fitness_score=89.6,  # Weighted average
            model_hash="QmLocal123"
        )
        
        proof_submitted = await connector.submit_training_proof(
            submission, keypair
        )
        assert proof_submitted is True
        
        # Step 4: Calculate rewards
        fitness_scores = FitnessScores(88.0, 92.0, 90.0)
        reward_calc = calculator.calculate_reward(
            participant_id=participant_id,
            round_number=1,
            fitness_scores=fitness_scores,
            stake_amount_planck=stake_amount
        )
        
        # Should earn ~12-13 DALLA (base 10 × fitness 0.896 × stake 1.4)
        assert 12.0 < reward_calc.total_reward_dalla < 14.0
        
        # Step 5: Queue reward for distribution
        distributor.add_pending_reward(reward_calc)
        
        pending = distributor.get_total_pending(participant_id)
        assert pending == reward_calc.total_reward_planck
        
        # Step 6: Claim rewards
        success, claimed_amount = await connector.claim_rewards(participant_id, keypair)
        assert success is True
        assert claimed_amount > 0
        
        # Step 7: Mark as distributed
        distributor.mark_distributed(participant_id, claimed_amount)
        
        # Verify participant stats updated
        info = await connector.get_participant_info(participant_id)
        assert info.training_rounds_completed == 1
        assert info.avg_fitness_score > 85.0
        
        # Verify no pending rewards
        assert distributor.get_total_pending(participant_id) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_validators_competition(self, setup):
        """Test multiple validators competing for rewards."""
        connector = setup["connector"]
        calculator = setup["calculator"]
        keypair = setup["keypair"]
        
        # Enroll 3 validators with different stakes
        validators = [
            ("alice", dalla_to_planck(10000)),
            ("bob", dalla_to_planck(5000)),
            ("charlie", dalla_to_planck(2000))
        ]
        
        for participant_id, stake in validators:
            await connector.enroll_participant(
                participant_id, stake, keypair
            )
        
        # Simulate training round with different performance
        fitness_scores = [
            FitnessScores(95, 92, 98),  # Alice: high quality
            FitnessScores(80, 85, 82),  # Bob: average
            FitnessScores(70, 75, 68),  # Charlie: lower quality
        ]
        
        rewards = []
        
        for (participant_id, stake), fitness in zip(validators, fitness_scores):
            # Submit proof
            submission = TrainingSubmission(
                participant_id=participant_id,
                round_number=1,
                genome_id=f"genome_{participant_id}",
                samples_trained=1000,
                training_time=100.0,
                quality_score=fitness.quality,
                timeliness_score=fitness.timeliness,
                honesty_score=fitness.honesty,
                fitness_score=fitness.calculate_overall(),
                model_hash=f"Qm{participant_id}"
            )
            
            await connector.submit_training_proof(submission, keypair)
            
            # Calculate reward
            reward = calculator.calculate_reward(
                participant_id, 1, fitness, stake
            )
            rewards.append(reward.total_reward_dalla)
        
        # Alice should earn most (high fitness + max stake)
        # Bob should earn medium (avg fitness + mid stake)
        # Charlie should earn least (low fitness + low stake)
        assert rewards[0] > rewards[1] > rewards[2]
        
        # Alice should earn close to 20 DALLA (near perfect × 2.0x)
        assert rewards[0] > 18.0
        
        # Charlie should earn significantly less
        assert rewards[2] < 10.0


# =============================================================================
# Pytest Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
