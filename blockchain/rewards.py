"""
Reward Calculator and Distributor

Calculates and distributes DALLA rewards for AI training contributions
based on Proof of Useful Work (PoUW) fitness scores.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

# Reward calculation constants (aligned with Staking pallet)
BASE_REWARD_DALLA = 10.0  # Base reward per training round
DALLA_DECIMALS = 12  # DALLA has 12 decimals
PLANCK_PER_DALLA = 10 ** DALLA_DECIMALS

# Fitness weight components (PoUW)
QUALITY_WEIGHT = 0.40  # 40% - Model accuracy
TIMELINESS_WEIGHT = 0.30  # 30% - Training speed
HONESTY_WEIGHT = 0.30  # 30% - Privacy compliance

# Stake multipliers
MIN_STAKE_MULTIPLIER = 1.0  # No bonus for minimum stake
MAX_STAKE_MULTIPLIER = 2.0  # 2x bonus for maximum stake
MIN_STAKE_DALLA = 1000  # Minimum stake requirement
MAX_BONUS_STAKE_DALLA = 10000  # Stake amount for maximum bonus


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FitnessScores:
    """Component fitness scores for PoUW."""
    
    quality: float  # 0-100: Model accuracy/improvement
    timeliness: float  # 0-100: Training speed/efficiency
    honesty: float  # 0-100: Privacy compliance/Byzantine resistance
    
    def calculate_overall(self) -> float:
        """Calculate weighted overall fitness score."""
        return (
            QUALITY_WEIGHT * self.quality +
            TIMELINESS_WEIGHT * self.timeliness +
            HONESTY_WEIGHT * self.honesty
        )
    
    def validate(self) -> list[str]:
        """Validate fitness scores."""
        errors = []
        for name, value in [
            ("quality", self.quality),
            ("timeliness", self.timeliness),
            ("honesty", self.honesty),
        ]:
            if not (0 <= value <= 100):
                errors.append(f"{name} must be between 0 and 100")
        return errors


@dataclass
class RewardCalculation:
    """Detailed reward calculation breakdown."""
    
    participant_id: str
    round_number: int
    
    # Fitness components
    fitness_scores: FitnessScores
    overall_fitness: float
    
    # Stake info
    stake_amount_dalla: float
    stake_multiplier: float
    
    # Reward calculation
    base_reward_dalla: float
    fitness_multiplier: float
    stake_bonus_dalla: float
    total_reward_dalla: float
    total_reward_planck: int
    
    def __str__(self) -> str:
        return (
            f"Reward for {self.participant_id} (Round {self.round_number}):\n"
            f"  Fitness: {self.overall_fitness:.2f} "
            f"(Q:{self.fitness_scores.quality:.1f} "
            f"T:{self.fitness_scores.timeliness:.1f} "
            f"H:{self.fitness_scores.honesty:.1f})\n"
            f"  Stake: {self.stake_amount_dalla:.0f} DALLA "
            f"(multiplier: {self.stake_multiplier:.2f}x)\n"
            f"  Base Reward: {self.base_reward_dalla:.2f} DALLA\n"
            f"  Fitness Bonus: {self.fitness_multiplier:.2f}x\n"
            f"  Stake Bonus: +{self.stake_bonus_dalla:.2f} DALLA\n"
            f"  Total: {self.total_reward_dalla:.2f} DALLA"
        )


# =============================================================================
# Reward Calculator
# =============================================================================


class RewardCalculator:
    """
    Calculate training rewards based on PoUW fitness scores.
    
    Formula:
        reward = base_reward * (fitness / 100) * stake_multiplier
    
    Where:
        - base_reward = 10 DALLA per round
        - fitness = weighted score (Quality 40%, Timeliness 30%, Honesty 30%)
        - stake_multiplier = 1.0-2.0x based on staked amount
    """
    
    def __init__(
        self,
        base_reward_dalla: float = BASE_REWARD_DALLA,
        min_stake_dalla: float = MIN_STAKE_DALLA,
        max_bonus_stake_dalla: float = MAX_BONUS_STAKE_DALLA,
    ):
        """
        Initialize reward calculator.
        
        Args:
            base_reward_dalla: Base reward per training round
            min_stake_dalla: Minimum stake requirement
            max_bonus_stake_dalla: Stake amount for maximum bonus
        """
        self.base_reward_dalla = base_reward_dalla
        self.min_stake_dalla = min_stake_dalla
        self.max_bonus_stake_dalla = max_bonus_stake_dalla
        
        logger.info(
            "Initialized RewardCalculator",
            base_reward=base_reward_dalla,
            min_stake=min_stake_dalla,
            max_bonus_stake=max_bonus_stake_dalla,
        )
    
    def calculate_stake_multiplier(self, stake_amount_dalla: float) -> float:
        """
        Calculate stake multiplier (1.0-2.0x).
        
        Args:
            stake_amount_dalla: Staked amount in DALLA
        
        Returns:
            Stake multiplier between 1.0 and 2.0
        """
        if stake_amount_dalla < self.min_stake_dalla:
            return 0.0  # Below minimum stake
        
        if stake_amount_dalla >= self.max_bonus_stake_dalla:
            return MAX_STAKE_MULTIPLIER  # Maximum bonus
        
        # Linear interpolation between min and max
        stake_ratio = (
            (stake_amount_dalla - self.min_stake_dalla) /
            (self.max_bonus_stake_dalla - self.min_stake_dalla)
        )
        
        return MIN_STAKE_MULTIPLIER + stake_ratio * (
            MAX_STAKE_MULTIPLIER - MIN_STAKE_MULTIPLIER
        )
    
    def calculate_reward(
        self,
        participant_id: str,
        round_number: int,
        fitness_scores: FitnessScores,
        stake_amount_planck: int,
    ) -> RewardCalculation:
        """
        Calculate reward for training contribution.
        
        Args:
            participant_id: Validator account ID
            round_number: Training round number
            fitness_scores: PoUW fitness scores
            stake_amount_planck: Staked amount in Planck
        
        Returns:
            RewardCalculation with detailed breakdown
        """
        # Validate fitness scores
        errors = fitness_scores.validate()
        if errors:
            raise ValueError(f"Invalid fitness scores: {errors}")
        
        # Convert stake to DALLA
        stake_amount_dalla = stake_amount_planck / PLANCK_PER_DALLA
        
        # Calculate components
        overall_fitness = fitness_scores.calculate_overall()
        fitness_multiplier = overall_fitness / 100.0
        stake_multiplier = self.calculate_stake_multiplier(stake_amount_dalla)
        
        # Calculate rewards
        base_reward_dalla = self.base_reward_dalla
        stake_bonus_dalla = base_reward_dalla * (stake_multiplier - 1.0)
        total_reward_dalla = base_reward_dalla * fitness_multiplier * stake_multiplier
        total_reward_planck = int(total_reward_dalla * PLANCK_PER_DALLA)
        
        calculation = RewardCalculation(
            participant_id=participant_id,
            round_number=round_number,
            fitness_scores=fitness_scores,
            overall_fitness=overall_fitness,
            stake_amount_dalla=stake_amount_dalla,
            stake_multiplier=stake_multiplier,
            base_reward_dalla=base_reward_dalla,
            fitness_multiplier=fitness_multiplier,
            stake_bonus_dalla=stake_bonus_dalla,
            total_reward_dalla=total_reward_dalla,
            total_reward_planck=total_reward_planck,
        )
        
        logger.debug(
            "Calculated reward",
            participant=participant_id,
            round=round_number,
            fitness=f"{overall_fitness:.2f}",
            reward_dalla=f"{total_reward_dalla:.2f}",
        )
        
        return calculation
    
    def estimate_monthly_rewards(
        self,
        rounds_per_day: int,
        avg_fitness: float,
        stake_amount_dalla: float,
    ) -> float:
        """
        Estimate monthly rewards for validator.
        
        Args:
            rounds_per_day: Expected training rounds per day
            avg_fitness: Expected average fitness score (0-100)
            stake_amount_dalla: Staked amount in DALLA
        
        Returns:
            Estimated monthly rewards in DALLA
        """
        # Calculate per-round reward
        fitness_scores = FitnessScores(
            quality=avg_fitness,
            timeliness=avg_fitness,
            honesty=avg_fitness,
        )
        
        calculation = self.calculate_reward(
            participant_id="estimator",
            round_number=0,
            fitness_scores=fitness_scores,
            stake_amount_planck=int(stake_amount_dalla * PLANCK_PER_DALLA),
        )
        
        # Monthly estimate (30 days)
        monthly_rewards = calculation.total_reward_dalla * rounds_per_day * 30
        
        logger.info(
            "Monthly reward estimate",
            rounds_per_day=rounds_per_day,
            avg_fitness=avg_fitness,
            stake_dalla=stake_amount_dalla,
            monthly_dalla=f"{monthly_rewards:.2f}",
        )
        
        return monthly_rewards


# =============================================================================
# Reward Distributor
# =============================================================================


class RewardDistributor:
    """
    Distribute rewards to training participants.
    
    Coordinates with StakingConnector to process reward claims.
    """
    
    def __init__(
        self,
        calculator: RewardCalculator | None = None,
    ):
        """
        Initialize reward distributor.
        
        Args:
            calculator: RewardCalculator instance (creates default if None)
        """
        self.calculator = calculator or RewardCalculator()
        self.pending_rewards: dict[str, list[RewardCalculation]] = {}
        self.distributed_rewards: dict[str, int] = {}  # participant_id -> total_planck
        
        logger.info("Initialized RewardDistributor")
    
    def add_pending_reward(self, calculation: RewardCalculation) -> None:
        """
        Add pending reward for participant.
        
        Args:
            calculation: Reward calculation to queue
        """
        participant_id = calculation.participant_id
        
        if participant_id not in self.pending_rewards:
            self.pending_rewards[participant_id] = []
        
        self.pending_rewards[participant_id].append(calculation)
        
        logger.debug(
            "Added pending reward",
            participant=participant_id,
            round=calculation.round_number,
            amount_dalla=f"{calculation.total_reward_dalla:.2f}",
        )
    
    def get_pending_rewards(self, participant_id: str) -> list[RewardCalculation]:
        """Get all pending rewards for participant."""
        return self.pending_rewards.get(participant_id, [])
    
    def get_total_pending(self, participant_id: str) -> int:
        """
        Get total pending rewards in Planck.
        
        Args:
            participant_id: Validator account ID
        
        Returns:
            Total pending rewards in Planck
        """
        rewards = self.get_pending_rewards(participant_id)
        return sum(r.total_reward_planck for r in rewards)
    
    def mark_distributed(self, participant_id: str, amount_planck: int) -> None:
        """
        Mark rewards as distributed.
        
        Args:
            participant_id: Validator account ID
            amount_planck: Amount distributed in Planck
        """
        # Clear pending rewards
        if participant_id in self.pending_rewards:
            del self.pending_rewards[participant_id]
        
        # Update distributed total
        if participant_id not in self.distributed_rewards:
            self.distributed_rewards[participant_id] = 0
        
        self.distributed_rewards[participant_id] += amount_planck
        
        logger.info(
            "Marked rewards as distributed",
            participant=participant_id,
            amount_dalla=f"{amount_planck / PLANCK_PER_DALLA:.2f}",
            total_dalla=f"{self.distributed_rewards[participant_id] / PLANCK_PER_DALLA:.2f}",
        )
    
    def get_total_distributed(self, participant_id: str) -> int:
        """Get total distributed rewards in Planck."""
        return self.distributed_rewards.get(participant_id, 0)
    
    def get_statistics(self) -> dict[str, any]:
        """
        Get reward distribution statistics.
        
        Returns:
            Statistics dictionary
        """
        total_pending = sum(
            self.get_total_pending(pid)
            for pid in self.pending_rewards.keys()
        )
        
        total_distributed = sum(self.distributed_rewards.values())
        
        return {
            "participants_with_pending": len(self.pending_rewards),
            "total_pending_planck": total_pending,
            "total_pending_dalla": total_pending / PLANCK_PER_DALLA,
            "participants_rewarded": len(self.distributed_rewards),
            "total_distributed_planck": total_distributed,
            "total_distributed_dalla": total_distributed / PLANCK_PER_DALLA,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def dalla_to_planck(dalla: float) -> int:
    """Convert DALLA to Planck (smallest unit)."""
    return int(dalla * PLANCK_PER_DALLA)


def planck_to_dalla(planck: int) -> float:
    """Convert Planck to DALLA."""
    return planck / PLANCK_PER_DALLA


def format_dalla(planck: int, decimals: int = 2) -> str:
    """Format Planck as DALLA string."""
    dalla = planck_to_dalla(planck)
    return f"{dalla:.{decimals}f} DALLA"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FitnessScores",
    "RewardCalculation",
    "RewardCalculator",
    "RewardDistributor",
    "dalla_to_planck",
    "planck_to_dalla",
    "format_dalla",
    "BASE_REWARD_DALLA",
    "PLANCK_PER_DALLA",
    "QUALITY_WEIGHT",
    "TIMELINESS_WEIGHT",
    "HONESTY_WEIGHT",
]
