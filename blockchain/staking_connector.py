"""
Staking Connector - Connect to BelizeChain Staking Pallet

Manages validator enrollment, stake verification, and PoUW submissions
for federated AI training participants.

Integrates with Community pallet to automatically record federated learning
contributions for Social Responsibility Score (SRS) tracking.

Author: BelizeChain AI Team
Date: January 2026
Python: 3.13+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from loguru import logger

try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logger.warning("substrateinterface not installed, using mock mode")

# Import Community connector for SRS integration
try:
    from .community_connector import CommunityConnector
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False
    logger.warning("CommunityConnector not available, SRS tracking disabled")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParticipantInfo:
    """Information about a training participant (validator)."""
    
    account_id: str
    stake_amount: int  # in Planck (smallest unit)
    is_enrolled: bool
    training_rounds_completed: int
    total_samples_trained: int
    avg_fitness_score: float
    last_submission: str | None = None
    slashed_amount: int = 0
    reputation_score: float = 100.0
    
    def __post_init__(self):
        if self.avg_fitness_score < 0 or self.avg_fitness_score > 100:
            raise ValueError("Fitness score must be between 0 and 100")


@dataclass
class TrainingSubmission:
    """Training submission for PoUW verification."""
    
    participant_id: str
    round_number: int
    genome_id: str
    samples_trained: int
    training_time: float  # seconds
    quality_score: float  # 0-100
    timeliness_score: float  # 0-100
    honesty_score: float  # 0-100
    fitness_score: float  # Weighted average
    model_hash: str  # Hash of model weights
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def validate(self) -> list[str]:
        """Validate submission data."""
        errors = []
        
        if self.samples_trained <= 0:
            errors.append("samples_trained must be positive")
        
        if self.training_time <= 0:
            errors.append("training_time must be positive")
        
        for score_name, score in [
            ("quality_score", self.quality_score),
            ("timeliness_score", self.timeliness_score),
            ("honesty_score", self.honesty_score),
            ("fitness_score", self.fitness_score),
        ]:
            if not (0 <= score <= 100):
                errors.append(f"{score_name} must be between 0 and 100")
        
        if not self.model_hash:
            errors.append("model_hash is required")
        
        return errors


# =============================================================================
# Staking Connector
# =============================================================================


class StakingConnector:
    """
    Connector to BelizeChain Staking pallet.
    
    Manages:
    - Validator enrollment for AI training
    - Stake verification
    - Training submission (PoUW)
    - Reward claims
    - Slashing conditions
    - Community pallet SRS integration (automatic participation tracking)
    """
    
    def __init__(
        self,
        node_url: str = "ws://127.0.0.1:9944",
        mock_mode: bool = False,
        enable_community_tracking: bool = True,
    ):
        """
        Initialize staking connector.
        
        Args:
            node_url: WebSocket URL of BelizeChain node
            mock_mode: Use mock mode for testing (no actual blockchain)
            enable_community_tracking: Enable automatic SRS tracking (default: True)
        """
        self.node_url = node_url
        self.mock_mode = mock_mode or not SUBSTRATE_AVAILABLE
        self.substrate: SubstrateInterface | None = None
        self.is_connected = False
        
        # Community pallet integration
        self.enable_community_tracking = enable_community_tracking and COMMUNITY_AVAILABLE
        self.community_connector: CommunityConnector | None = None
        
        if self.enable_community_tracking:
            try:
                self.community_connector = CommunityConnector(
                    websocket_url=node_url,
                    mock_mode=mock_mode
                )
                logger.info("Community SRS tracking ENABLED")
            except Exception as e:
                logger.warning(f"Failed to initialize CommunityConnector: {e}")
                self.enable_community_tracking = False
        else:
            logger.info("Community SRS tracking DISABLED")
        
        # Mock data for testing
        self._mock_participants: dict[str, ParticipantInfo] = {}
        self._mock_submissions: list[TrainingSubmission] = []
        
        logger.info(
            "Initialized StakingConnector",
            node_url=node_url,
            mock_mode=self.mock_mode,
        )
    
    async def connect(self) -> bool:
        """
        Connect to BelizeChain node.
        
        Returns:
            True if connected successfully
        """
        if self.mock_mode:
            logger.info("Running in mock mode, skipping blockchain connection")
            self.is_connected = True
            
            # Also connect community connector if enabled
            if self.enable_community_tracking and self.community_connector:
                await self.community_connector.connect()
            
            return True
        
        try:
            self.substrate = SubstrateInterface(url=self.node_url)
            self.is_connected = True
            
            # Get chain info
            chain = self.substrate.chain
            properties = self.substrate.properties
            
            logger.info(
                "Connected to BelizeChain",
                chain=chain,
                properties=properties,
            )
            
            # Also connect community connector if enabled
            if self.enable_community_tracking and self.community_connector:
                await self.community_connector.connect()
                logger.info("Community connector initialized")
            
            return True
        
        except Exception as e:
            logger.error("Failed to connect to BelizeChain", error=str(e))
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from BelizeChain node."""
        if self.substrate:
            self.substrate.close()
            self.substrate = None
        self.is_connected = False
        logger.info("Disconnected from BelizeChain")
    
    async def enroll_participant(
        self,
        account_id: str,
        stake_amount: int,
        keypair: Keypair | None = None,
    ) -> bool:
        """
        Enroll validator as AI training participant.
        
        Args:
            account_id: Validator account ID (SS58 address)
            stake_amount: Amount to stake (in Planck)
            keypair: Keypair for signing transaction
        
        Returns:
            True if enrollment successful
        """
        if self.mock_mode:
            # Check if already enrolled
            if account_id in self._mock_participants:
                logger.warning("Participant already enrolled", account_id=account_id)
                return False
            
            # Mock enrollment
            self._mock_participants[account_id] = ParticipantInfo(
                account_id=account_id,
                stake_amount=stake_amount,
                is_enrolled=True,
                training_rounds_completed=0,
                total_samples_trained=0,
                avg_fitness_score=0.0,
            )
            logger.info(
                "Mock: Enrolled participant",
                account_id=account_id,
                stake_amount=stake_amount,
            )
            return True
        
        if not self.is_connected:
            logger.error("Not connected to blockchain")
            return False
        
        try:
            # Create enrollment call
            call = self.substrate.compose_call(
                call_module='Staking',
                call_function='enroll_ai_trainer',
                call_params={
                    'stake_amount': stake_amount,
                }
            )
            
            # Create and submit extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                logger.info(
                    "Enrolled participant",
                    account_id=account_id,
                    stake_amount=stake_amount,
                    block_hash=receipt.block_hash,
                )
                return True
            else:
                logger.error(
                    "Enrollment failed",
                    account_id=account_id,
                    error=receipt.error_message,
                )
                return False
        
        except Exception as e:
            logger.error("Enrollment exception", account_id=account_id, error=str(e))
            return False
    
    async def unenroll_participant(
        self,
        account_id: str,
        keypair: Keypair | None = None,
    ) -> bool:
        """
        Unenroll validator from AI training participation.
        
        Args:
            account_id: Validator account ID
            keypair: Keypair for signing transaction
        
        Returns:
            True if unenrollment successful
        """
        if self.mock_mode:
            # Check if enrolled
            if account_id not in self._mock_participants:
                logger.warning("Participant not found", account_id=account_id)
                return False
            
            # Mark as unenrolled
            self._mock_participants[account_id].is_enrolled = False
            logger.info("Mock: Unenrolled participant", account_id=account_id)
            return True
        
        if not self.is_connected:
            logger.error("Not connected to blockchain")
            return False
        
        try:
            # Create unenrollment call
            call = self.substrate.compose_call(
                call_module='Staking',
                call_function='unenroll_ai_trainer',
                call_params={}
            )
            
            # Create and submit extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                logger.info(
                    "Unenrolled participant",
                    account_id=account_id,
                    block_hash=receipt.block_hash,
                )
                return True
            else:
                logger.error(
                    "Unenrollment failed",
                    account_id=account_id,
                    error=receipt.error_message,
                )
                return False
        
        except Exception as e:
            logger.error("Unenrollment exception", account_id=account_id, error=str(e))
            return False
    
    async def get_participant_info(self, account_id: str) -> ParticipantInfo | None:
        """
        Get participant information.
        
        Args:
            account_id: Validator account ID
        
        Returns:
            ParticipantInfo or None if not found
        """
        if self.mock_mode:
            return self._mock_participants.get(account_id)
        
        if not self.is_connected:
            logger.error("Not connected to blockchain")
            return None
        
        try:
            # Query staking storage
            result = self.substrate.query(
                module='Staking',
                storage_function='AITrainers',
                params=[account_id],
            )
            
            if result.value:
                return ParticipantInfo(
                    account_id=account_id,
                    stake_amount=result.value['stake_amount'],
                    is_enrolled=result.value['is_enrolled'],
                    training_rounds_completed=result.value['rounds_completed'],
                    total_samples_trained=result.value['total_samples'],
                    avg_fitness_score=result.value['avg_fitness'],
                    last_submission=result.value.get('last_submission'),
                    slashed_amount=result.value.get('slashed_amount', 0),
                    reputation_score=result.value.get('reputation', 100.0),
                )
            
            return None
        
        except Exception as e:
            logger.error("Failed to get participant info", account_id=account_id, error=str(e))
            return None
    
    async def submit_training_proof(
        self,
        submission: TrainingSubmission,
        keypair: Keypair | None = None,
    ) -> bool:
        """
        Submit training proof for PoUW verification.
        
        Args:
            submission: Training submission data
            keypair: Keypair for signing transaction
        
        Returns:
            True if submission successful
        """
        # Validate submission
        errors = submission.validate()
        if errors:
            logger.error("Invalid submission", errors=errors)
            return False
        
        if self.mock_mode:
            # Check if participant is enrolled
            if submission.participant_id not in self._mock_participants:
                logger.error("Participant not enrolled", participant_id=submission.participant_id)
                return False
            
            # Mock submission
            self._mock_submissions.append(submission)
            
            # Update mock participant stats
            participant = self._mock_participants[submission.participant_id]
            participant.training_rounds_completed += 1
            participant.total_samples_trained += submission.samples_trained
            
            # Update average fitness (running average)
            n = participant.training_rounds_completed
            participant.avg_fitness_score = (
                (participant.avg_fitness_score * (n - 1) + submission.fitness_score) / n
            )
            participant.last_submission = submission.timestamp
            
            logger.info(
                "Mock: Submitted training proof",
                participant=submission.participant_id,
                round=submission.round_number,
                fitness=submission.fitness_score,
            )
            return True
        
        if not self.is_connected:
            logger.error("Not connected to blockchain")
            return False
        
        try:
            # Create submission call
            call = self.substrate.compose_call(
                call_module='Staking',
                call_function='submit_training_proof',
                call_params={
                    'round_number': submission.round_number,
                    'genome_id': submission.genome_id,
                    'samples_trained': submission.samples_trained,
                    'training_time': int(submission.training_time * 1000),  # Convert to ms
                    'quality_score': int(submission.quality_score * 100),  # Fixed point
                    'timeliness_score': int(submission.timeliness_score * 100),
                    'honesty_score': int(submission.honesty_score * 100),
                    'model_hash': submission.model_hash,
                }
            )
            
            # Create and submit extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                logger.info(
                    "Submitted training proof",
                    participant=submission.participant_id,
                    round=submission.round_number,
                    fitness=submission.fitness_score,
                    block_hash=receipt.block_hash,
                )
                
                # Record participation in Community pallet for SRS tracking
                if self.enable_community_tracking and self.community_connector:
                    try:
                        success, tx_hash = await self.community_connector.record_federated_learning_contribution(
                            account_id=submission.participant_id,
                            round_number=submission.round_number,
                            quality_score=submission.quality_score,
                            samples_trained=submission.samples_trained,
                            training_duration_seconds=int(submission.training_time)
                        )
                        
                        if success:
                            logger.info(f"Community participation recorded (SRS updated): {tx_hash}")
                        else:
                            logger.warning("Failed to record community participation (SRS not updated)")
                    
                    except Exception as e:
                        logger.error(f"Community tracking error (continuing anyway): {e}")
                
                return True
            else:
                logger.error(
                    "Submission failed",
                    participant=submission.participant_id,
                    error=receipt.error_message,
                )
                return False
        
        except Exception as e:
            logger.error(
                "Submission exception",
                participant=submission.participant_id,
                error=str(e),
            )
            return False
    
    async def claim_rewards(
        self,
        account_id: str,
        keypair: Keypair | None = None,
    ) -> tuple[bool, int]:
        """
        Claim training rewards.
        
        Args:
            account_id: Validator account ID
            keypair: Keypair for signing transaction
        
        Returns:
            (success, reward_amount) tuple
        """
        if self.mock_mode:
            # Mock reward claim
            if account_id in self._mock_participants:
                participant = self._mock_participants[account_id]
                # Simple reward calculation: 10 DALLA per round * avg fitness
                reward = int(
                    participant.training_rounds_completed * 10 * 1e12  # 10 DALLA
                    * (participant.avg_fitness_score / 100)
                )
                logger.info(
                    "Mock: Claimed rewards",
                    account_id=account_id,
                    reward_dalla=reward / 1e12,
                )
                return True, reward
            return False, 0
        
        if not self.is_connected:
            logger.error("Not connected to blockchain")
            return False, 0
        
        try:
            # Create claim call
            call = self.substrate.compose_call(
                call_module='Staking',
                call_function='claim_training_rewards',
                call_params={}
            )
            
            # Create and submit extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True,
            )
            
            if receipt.is_success:
                # Extract reward amount from events
                reward_amount = 0
                for event in receipt.triggered_events:
                    if event.value['module_id'] == 'Staking' and \
                       event.value['event_id'] == 'RewardsClaimed':
                        reward_amount = event.value['attributes']['amount']
                        break
                
                logger.info(
                    "Claimed rewards",
                    account_id=account_id,
                    reward_dalla=reward_amount / 1e12,
                    block_hash=receipt.block_hash,
                )
                return True, reward_amount
            else:
                logger.error(
                    "Claim failed",
                    account_id=account_id,
                    error=receipt.error_message,
                )
                return False, 0
        
        except Exception as e:
            logger.error("Claim exception", account_id=account_id, error=str(e))
            return False, 0
    
    async def get_total_staked(self) -> int:
        """Get total amount staked by all AI trainers."""
        if self.mock_mode:
            return sum(p.stake_amount for p in self._mock_participants.values())
        
        if not self.is_connected:
            return 0
        
        try:
            result = self.substrate.query(
                module='Staking',
                storage_function='TotalAITrainerStake',
            )
            return result.value if result.value else 0
        except Exception as e:
            logger.error("Failed to get total staked", error=str(e))
            return 0
    
    async def get_all_participants(self) -> list[ParticipantInfo]:
        """Get all enrolled AI training participants."""
        if self.mock_mode:
            return list(self._mock_participants.values())
        
        if not self.is_connected:
            return []
        
        try:
            # Query all AI trainers
            result = self.substrate.query_map(
                module='Staking',
                storage_function='AITrainers',
            )
            
            participants = []
            for account_id, data in result:
                participants.append(ParticipantInfo(
                    account_id=str(account_id),
                    stake_amount=data.value['stake_amount'],
                    is_enrolled=data.value['is_enrolled'],
                    training_rounds_completed=data.value['rounds_completed'],
                    total_samples_trained=data.value['total_samples'],
                    avg_fitness_score=data.value['avg_fitness'],
                    last_submission=data.value.get('last_submission'),
                    slashed_amount=data.value.get('slashed_amount', 0),
                    reputation_score=data.value.get('reputation', 100.0),
                ))
            
            return participants
        
        except Exception as e:
            logger.error("Failed to get all participants", error=str(e))
            return []


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ParticipantInfo",
    "TrainingSubmission",
    "StakingConnector",
]
