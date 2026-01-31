"""
Staking Interface for BelizeChain Proof of Useful Work.

Integrates Nawal AI with BelizeChain's staking pallet to submit
fitness scores and participate in PoUW consensus.

Key Features:
- Submit fitness scores (quality, timeliness, honesty)
- Query validator information
- Manage stake
- Track rewards

PoUW Components:
- Quality (40%): Model improvement on validation set
- Timeliness (30%): Submission within deadline
- Honesty (30%): Privacy compliance verification

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from loguru import logger

from .substrate_client import SubstrateClient, ExtrinsicReceipt


class ValidatorStatus(Enum):
    """Validator status on BelizeChain."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    JAILED = "jailed"
    KICKED = "kicked"


@dataclass
class FitnessScore:
    """
    Proof of Useful Work fitness score.
    
    Components:
    - Quality: Model improvement (0-100)
    - Timeliness: Submission timing (0-100)
    - Honesty: Privacy compliance (0-100)
    
    Final score = 0.4*Q + 0.3*T + 0.3*H
    
    Attributes:
        quality: Model quality score (0-100)
        timeliness: Timeliness score (0-100)
        honesty: Privacy compliance score (0-100)
        total: Weighted total score (0-100)
        round: Training round number
        timestamp: Submission timestamp
    """
    quality: float
    timeliness: float
    honesty: float
    round: int
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validate scores
        for score_name in ['quality', 'timeliness', 'honesty']:
            score = getattr(self, score_name)
            if not 0 <= score <= 100:
                raise ValueError(f"{score_name} must be 0-100, got {score}")
    
    @property
    def total(self) -> float:
        """Calculate weighted total score."""
        return (
            0.4 * self.quality +
            0.3 * self.timeliness +
            0.3 * self.honesty
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for chain submission."""
        return {
            'quality': int(self.quality * 100),  # Convert to basis points
            'timeliness': int(self.timeliness * 100),
            'honesty': int(self.honesty * 100),
            'round': self.round,
        }


@dataclass
class ValidatorInfo:
    """
    BelizeChain validator information.
    
    Attributes:
        address: Validator SS58 address
        stake: Total staked amount (DALLA)
        status: Validator status
        commission: Commission rate (0-100%)
        total_score: Cumulative fitness score
        rounds_participated: Number of rounds participated
        last_fitness: Last submitted fitness score
        reputation: Reputation score (0-100)
    """
    address: str
    stake: int
    status: ValidatorStatus
    commission: float = 0.0
    total_score: float = 0.0
    rounds_participated: int = 0
    last_fitness: Optional[FitnessScore] = None
    reputation: float = 100.0


@dataclass
class StakeInfo:
    """
    Validator stake information.
    
    Attributes:
        total: Total staked amount
        own: Validator's own stake
        delegated: Delegated stake from others
        min_required: Minimum required stake
        is_sufficient: Whether stake meets requirement
    """
    total: int
    own: int
    delegated: int
    min_required: int
    
    @property
    def is_sufficient(self) -> bool:
        """Check if stake is sufficient."""
        return self.total >= self.min_required


class StakingInterface:
    """
    Interface to BelizeChain staking pallet.
    
    Provides high-level operations for:
    - Submitting PoUW fitness scores
    - Querying validator information
    - Managing stake
    - Tracking rewards
    
    Usage:
        # Initialize
        client = SubstrateClient(ChainConfig.local())
        staking = StakingInterface(client)
        
        # Submit fitness score
        score = FitnessScore(
            quality=95.0,
            timeliness=90.0,
            honesty=100.0,
            round=42,
        )
        
        keypair = client.create_keypair(uri="//Alice")
        receipt = staking.submit_fitness(keypair, score)
        
        # Query validator info
        info = staking.get_validator_info(keypair.ss58_address)
    """
    
    def __init__(self, client: SubstrateClient):
        """
        Initialize StakingInterface.
        
        Args:
            client: Substrate client instance
        """
        self.client = client
        logger.info("StakingInterface initialized")
    
    def submit_fitness(
        self,
        keypair,
        score: FitnessScore,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Submit fitness score to staking pallet.
        
        Args:
            keypair: Validator keypair
            score: Fitness score to submit
            wait_for_finalization: Wait for block finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(
            f"Submitting fitness score: round={score.round}, "
            f"total={score.total:.2f} "
            f"(Q={score.quality:.1f}, T={score.timeliness:.1f}, H={score.honesty:.1f})"
        )
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="submit_fitness",
            call_params=score.to_dict(),
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success(
                f"Fitness score submitted successfully "
                f"(block #{receipt.block_number})"
            )
        else:
            logger.error(f"Fitness submission failed: {receipt.error}")
        
        return receipt
    
    def get_validator_info(self, address: str) -> Optional[ValidatorInfo]:
        """
        Get validator information.
        
        Args:
            address: Validator SS58 address
        
        Returns:
            Validator info or None if not found
        """
        try:
            # Query validator storage
            validator_data = self.client.query_storage(
                module="Staking",
                storage_function="Validators",
                params=[address],
            )
            
            if validator_data is None:
                logger.warning(f"Validator not found: {address}")
                return None
            
            # Parse validator data
            info = ValidatorInfo(
                address=address,
                stake=validator_data.get('total_stake', 0),
                status=ValidatorStatus(validator_data.get('status', 'inactive')),
                commission=validator_data.get('commission', 0.0) / 100.0,
                total_score=validator_data.get('total_score', 0.0),
                rounds_participated=validator_data.get('rounds_participated', 0),
                reputation=validator_data.get('reputation', 100.0),
            )
            
            logger.debug(f"Retrieved validator info: {address}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get validator info: {e}")
            return None
    
    def get_stake_info(self, address: str) -> Optional[StakeInfo]:
        """
        Get validator stake information.
        
        Args:
            address: Validator SS58 address
        
        Returns:
            Stake info or None if not found
        """
        try:
            # Query stake data
            stake_data = self.client.query_storage(
                module="Staking",
                storage_function="Ledger",
                params=[address],
            )
            
            if stake_data is None:
                return None
            
            # Get minimum required stake
            min_stake = self.get_minimum_stake()
            
            info = StakeInfo(
                total=stake_data.get('total', 0),
                own=stake_data.get('own', 0),
                delegated=stake_data.get('delegated', 0),
                min_required=min_stake,
            )
            
            logger.debug(
                f"Stake info: total={info.total}, "
                f"sufficient={info.is_sufficient}"
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get stake info: {e}")
            return None
    
    def bond(
        self,
        keypair,
        amount: int,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Bond tokens for staking.
        
        Args:
            keypair: Controller keypair
            amount: Amount to bond (in plancks)
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(f"Bonding {amount} DALLA for staking")
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="bond",
            call_params={
                'value': amount,
            },
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success(f"Bonded {amount} DALLA successfully")
        else:
            logger.error(f"Bond failed: {receipt.error}")
        
        return receipt
    
    def unbond(
        self,
        keypair,
        amount: int,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Unbond tokens from staking.
        
        Args:
            keypair: Controller keypair
            amount: Amount to unbond
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(f"Unbonding {amount} DALLA")
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="unbond",
            call_params={
                'value': amount,
            },
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success(f"Unbonded {amount} DALLA successfully")
        else:
            logger.error(f"Unbond failed: {receipt.error}")
        
        return receipt
    
    def validate(
        self,
        keypair,
        commission: float = 0.0,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Declare intention to validate.
        
        Args:
            keypair: Validator keypair
            commission: Commission rate (0-100%)
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(f"Declaring validator with commission={commission}%")
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="validate",
            call_params={
                'commission': int(commission * 100),  # Basis points
            },
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success("Validator declaration successful")
        else:
            logger.error(f"Validation declaration failed: {receipt.error}")
        
        return receipt
    
    def get_minimum_stake(self) -> int:
        """
        Get minimum required stake for validators.
        
        Returns:
            Minimum stake in plancks
        """
        try:
            min_stake = self.client.get_runtime_constant(
                module="Staking",
                constant_name="MinValidatorStake",
            )
            return min_stake
        except Exception as e:
            logger.warning(f"Failed to get minimum stake: {e}, using default")
            return 1_000_000_000_000  # Default: 1000 DALLA
    
    def get_current_era(self) -> int:
        """
        Get current era number.
        
        Returns:
            Era number
        """
        try:
            era = self.client.query_storage(
                module="Staking",
                storage_function="CurrentEra",
            )
            return era if era is not None else 0
        except Exception as e:
            logger.error(f"Failed to get current era: {e}")
            return 0
    
    def get_validator_rewards(
        self,
        address: str,
        era: Optional[int] = None,
    ) -> int:
        """
        Get validator rewards for an era.
        
        Args:
            address: Validator address
            era: Era number (None = current)
        
        Returns:
            Reward amount
        """
        if era is None:
            era = self.get_current_era()
        
        try:
            rewards = self.client.query_storage(
                module="Staking",
                storage_function="ErasValidatorReward",
                params=[era, address],
            )
            return rewards if rewards is not None else 0
        except Exception as e:
            logger.error(f"Failed to get rewards: {e}")
            return 0
    
    def get_active_validators(self) -> List[str]:
        """
        Get list of active validator addresses.
        
        Returns:
            List of validator SS58 addresses
        """
        try:
            validators = self.client.query_storage(
                module="Staking",
                storage_function="ActiveValidators",
            )
            return validators if validators else []
        except Exception as e:
            logger.error(f"Failed to get active validators: {e}")
            return []
    
    def calculate_fitness_score(
        self,
        initial_loss: float,
        final_loss: float,
        submission_time: datetime,
        deadline: datetime,
        privacy_compliant: bool,
    ) -> FitnessScore:
        """
        Calculate PoUW fitness score from training results.
        
        Args:
            initial_loss: Loss before training
            final_loss: Loss after training
            submission_time: When score was submitted
            deadline: Submission deadline
            privacy_compliant: Whether privacy requirements met
        
        Returns:
            Fitness score
        """
        # Quality: Improvement ratio (0-100)
        if initial_loss > 0:
            improvement = (initial_loss - final_loss) / initial_loss
            quality = min(100.0, max(0.0, improvement * 100))
        else:
            quality = 0.0
        
        # Timeliness: Time before deadline (0-100)
        time_diff = (deadline - submission_time).total_seconds()
        deadline_window = 3600  # 1 hour window
        
        if time_diff >= 0:
            timeliness = min(100.0, (time_diff / deadline_window) * 100)
        else:
            timeliness = 0.0  # Late submission
        
        # Honesty: Privacy compliance (0 or 100)
        honesty = 100.0 if privacy_compliant else 0.0
        
        return FitnessScore(
            quality=quality,
            timeliness=timeliness,
            honesty=honesty,
            round=self.get_current_era(),
        )
