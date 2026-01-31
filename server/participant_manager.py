"""
Participant Management for Federated Learning

Manages validators participating in federated training:
- Participant enrollment and verification
- Status tracking (active, idle, offline, Byzantine)
- Contribution tracking
- Reward calculations
- Blockchain integration

Integrates with BelizeChain's Staking pallet.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any
from loguru import logger


# =============================================================================
# Participant Status
# =============================================================================


class ParticipantStatus(Enum):
    """Status of a participant in federated learning."""
    
    PENDING = auto()        # Waiting for enrollment approval
    ACTIVE = auto()         # Actively participating
    IDLE = auto()           # Enrolled but not submitting updates
    OFFLINE = auto()        # Not responding
    SUSPENDED = auto()      # Temporarily suspended
    BYZANTINE = auto()      # Detected as malicious
    SLASHED = auto()        # Slashed for poor performance


# =============================================================================
# Participant Record
# =============================================================================


@dataclass
class Participant:
    """
    Record of a federated learning participant (validator).
    
    Tracks:
    - Identity and status
    - Contribution metrics
    - Fitness scores
    - Rewards earned
    """
    
    # Identity
    participant_id: str
    validator_address: str  # Blockchain address
    staking_account: str    # Staking account ID
    
    # Status
    status: ParticipantStatus = ParticipantStatus.PENDING
    enrolled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Contributions
    rounds_participated: int = 0
    total_samples_trained: int = 0
    total_training_time: float = 0.0  # seconds
    
    # Performance
    avg_quality: float = 0.0
    avg_timeliness: float = 0.0
    avg_honesty: float = 0.0
    avg_fitness: float = 0.0
    
    # Rewards
    total_rewards: float = 0.0  # DALLA tokens
    pending_rewards: float = 0.0
    
    # Reputation
    reputation_score: float = 100.0  # 0-100
    byzantine_detections: int = 0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, new_status: ParticipantStatus, reason: str = "") -> None:
        """
        Update participant status.
        
        Args:
            new_status: New status
            reason: Reason for status change
        """
        old_status = self.status
        self.status = new_status
        self.last_seen = datetime.now(timezone.utc).isoformat()
        
        logger.info(
            "Participant status updated",
            participant_id=self.participant_id,
            old_status=old_status.name,
            new_status=new_status.name,
            reason=reason,
        )
    
    def record_contribution(
        self,
        samples: int,
        training_time: float,
        quality: float,
        timeliness: float,
        honesty: float,
    ) -> None:
        """
        Record training contribution.
        
        Args:
            samples: Number of samples trained
            training_time: Training time in seconds
            quality: Quality score (0-100)
            timeliness: Timeliness score (0-100)
            honesty: Honesty score (0-100)
        """
        self.rounds_participated += 1
        self.total_samples_trained += samples
        self.total_training_time += training_time
        
        # Update rolling averages
        n = self.rounds_participated
        self.avg_quality = ((n - 1) * self.avg_quality + quality) / n
        self.avg_timeliness = ((n - 1) * self.avg_timeliness + timeliness) / n
        self.avg_honesty = ((n - 1) * self.avg_honesty + honesty) / n
        self.avg_fitness = 0.4 * self.avg_quality + 0.3 * self.avg_timeliness + 0.3 * self.avg_honesty
        
        self.last_seen = datetime.now(timezone.utc).isoformat()
        
        logger.debug(
            "Contribution recorded",
            participant_id=self.participant_id,
            samples=samples,
            fitness=f"{self.avg_fitness:.2f}",
        )
    
    def calculate_reward(self, base_reward: float) -> float:
        """
        Calculate reward for this participant.
        
        Args:
            base_reward: Base reward amount
        
        Returns:
            Calculated reward (with multipliers)
        """
        # Fitness multiplier (0.0 - 2.0x)
        if self.avg_fitness >= 90:
            fitness_multiplier = 2.0
        elif self.avg_fitness >= 80:
            fitness_multiplier = 1.5
        elif self.avg_fitness >= 70:
            fitness_multiplier = 1.0
        elif self.avg_fitness >= 60:
            fitness_multiplier = 0.5
        else:
            fitness_multiplier = 0.0
        
        # Reputation multiplier (0.5 - 1.5x)
        reputation_multiplier = 0.5 + (self.reputation_score / 100.0)
        
        # Calculate total reward
        reward = base_reward * fitness_multiplier * reputation_multiplier
        
        return max(0.0, reward)
    
    def add_reward(self, amount: float) -> None:
        """
        Add reward to participant.
        
        Args:
            amount: Reward amount in DALLA
        """
        self.pending_rewards += amount
        logger.info(
            "Reward added",
            participant_id=self.participant_id,
            amount=f"{amount:.2f} DALLA",
            pending=f"{self.pending_rewards:.2f} DALLA",
        )
    
    def claim_rewards(self) -> float:
        """
        Claim pending rewards.
        
        Returns:
            Amount claimed
        """
        claimed = self.pending_rewards
        self.total_rewards += claimed
        self.pending_rewards = 0.0
        
        logger.info(
            "Rewards claimed",
            participant_id=self.participant_id,
            claimed=f"{claimed:.2f} DALLA",
            total=f"{self.total_rewards:.2f} DALLA",
        )
        
        return claimed
    
    def adjust_reputation(self, delta: float, reason: str = "") -> None:
        """
        Adjust reputation score.
        
        Args:
            delta: Change in reputation (-100 to +100)
            reason: Reason for adjustment
        """
        old_reputation = self.reputation_score
        self.reputation_score = max(0.0, min(100.0, self.reputation_score + delta))
        
        logger.info(
            "Reputation adjusted",
            participant_id=self.participant_id,
            old=f"{old_reputation:.1f}",
            new=f"{self.reputation_score:.1f}",
            delta=f"{delta:+.1f}",
            reason=reason,
        )
    
    def is_active(self, timeout: int = 600) -> bool:
        """
        Check if participant is active.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            True if active, False otherwise
        """
        if self.status not in [ParticipantStatus.ACTIVE, ParticipantStatus.IDLE]:
            return False
        
        last_seen = datetime.fromisoformat(self.last_seen)
        now = datetime.now(timezone.utc)
        
        return (now - last_seen).total_seconds() < timeout


# =============================================================================
# Participant Manager
# =============================================================================


class ParticipantManager:
    """
    Manages all participants in federated learning.
    
    Responsibilities:
    - Enroll and verify participants
    - Track status and contributions
    - Calculate and distribute rewards
    - Detect and handle Byzantine participants
    - Integrate with blockchain
    """
    
    def __init__(
        self,
        min_reputation: float = 50.0,
        byzantine_threshold: int = 3,
        activity_timeout: int = 600,  # 10 minutes
    ):
        """
        Initialize participant manager.
        
        Args:
            min_reputation: Minimum reputation to participate
            byzantine_threshold: Number of detections before suspending
            activity_timeout: Inactivity timeout in seconds
        """
        self.min_reputation = min_reputation
        self.byzantine_threshold = byzantine_threshold
        self.activity_timeout = activity_timeout
        
        # Participants storage
        self.participants: dict[str, Participant] = {}
        
        logger.info(
            "Initialized ParticipantManager",
            min_reputation=min_reputation,
            byzantine_threshold=byzantine_threshold,
            activity_timeout=activity_timeout,
        )
    
    def enroll_participant(
        self,
        participant_id: str,
        validator_address: str,
        staking_account: str,
    ) -> Participant:
        """
        Enroll new participant.
        
        Args:
            participant_id: Unique participant ID
            validator_address: Blockchain validator address
            staking_account: Staking account ID
        
        Returns:
            Participant record
        """
        if participant_id in self.participants:
            logger.warning(f"Participant {participant_id} already enrolled")
            return self.participants[participant_id]
        
        participant = Participant(
            participant_id=participant_id,
            validator_address=validator_address,
            staking_account=staking_account,
            status=ParticipantStatus.ACTIVE,  # Auto-activate for now
        )
        
        self.participants[participant_id] = participant
        
        logger.info(
            "Participant enrolled",
            participant_id=participant_id,
            validator=validator_address,
        )
        
        return participant
    
    def get_participant(self, participant_id: str) -> Participant | None:
        """
        Get participant by ID.
        
        Args:
            participant_id: Participant ID
        
        Returns:
            Participant if found, None otherwise
        """
        return self.participants.get(participant_id)
    
    def get_active_participants(self) -> list[Participant]:
        """Get all active participants."""
        return [
            p for p in self.participants.values()
            if p.is_active(self.activity_timeout)
        ]
    
    def update_participant_status(
        self,
        participant_id: str,
        status: ParticipantStatus,
        reason: str = "",
    ) -> bool:
        """
        Update participant status.
        
        Args:
            participant_id: Participant ID
            status: New status
            reason: Reason for update
        
        Returns:
            True if successful, False otherwise
        """
        participant = self.get_participant(participant_id)
        if not participant:
            logger.warning(f"Participant {participant_id} not found")
            return False
        
        participant.update_status(status, reason)
        return True
    
    def record_contribution(
        self,
        participant_id: str,
        samples: int,
        training_time: float,
        quality: float,
        timeliness: float,
        honesty: float,
    ) -> bool:
        """
        Record participant contribution.
        
        Args:
            participant_id: Participant ID
            samples: Samples trained
            training_time: Training time
            quality: Quality score
            timeliness: Timeliness score
            honesty: Honesty score
        
        Returns:
            True if successful, False otherwise
        """
        participant = self.get_participant(participant_id)
        if not participant:
            logger.warning(f"Participant {participant_id} not found")
            return False
        
        participant.record_contribution(
            samples=samples,
            training_time=training_time,
            quality=quality,
            timeliness=timeliness,
            honesty=honesty,
        )
        
        # Check for Byzantine behavior
        if honesty < 50.0:
            self._handle_potential_byzantine(participant_id)
        
        return True
    
    def _handle_potential_byzantine(self, participant_id: str) -> None:
        """
        Handle potential Byzantine participant.
        
        Args:
            participant_id: Participant ID
        """
        participant = self.get_participant(participant_id)
        if not participant:
            return
        
        participant.byzantine_detections += 1
        participant.adjust_reputation(-10.0, "Byzantine behavior detected")
        
        if participant.byzantine_detections >= self.byzantine_threshold:
            participant.update_status(
                ParticipantStatus.BYZANTINE,
                f"Exceeded Byzantine threshold ({self.byzantine_threshold})"
            )
            
            logger.warning(
                "Participant marked as Byzantine",
                participant_id=participant_id,
                detections=participant.byzantine_detections,
            )
    
    def distribute_rewards(
        self,
        base_reward: float,
        round_number: int,
    ) -> dict[str, float]:
        """
        Distribute rewards to active participants.
        
        Args:
            base_reward: Base reward per participant
            round_number: Current round number
        
        Returns:
            Dictionary of participant_id -> reward amount
        """
        rewards = {}
        active_participants = self.get_active_participants()
        
        for participant in active_participants:
            if participant.avg_fitness < 50.0:
                # Slashing threshold
                logger.warning(
                    f"Participant {participant.participant_id} below slashing threshold",
                    fitness=participant.avg_fitness,
                )
                participant.update_status(ParticipantStatus.SLASHED, "Fitness below 50%")
                continue
            
            # Calculate reward
            reward = participant.calculate_reward(base_reward)
            participant.add_reward(reward)
            rewards[participant.participant_id] = reward
        
        logger.info(
            "Rewards distributed",
            round=round_number,
            participants=len(rewards),
            total_rewards=f"{sum(rewards.values()):.2f} DALLA",
        )
        
        return rewards
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get participant statistics.
        
        Returns:
            Statistics dictionary
        """
        active = self.get_active_participants()
        
        if not self.participants:
            return {
                "total_participants": 0,
                "active_participants": 0,
                "avg_fitness": 0.0,
                "total_contributions": 0,
            }
        
        return {
            "total_participants": len(self.participants),
            "active_participants": len(active),
            "pending": len([p for p in self.participants.values() if p.status == ParticipantStatus.PENDING]),
            "idle": len([p for p in self.participants.values() if p.status == ParticipantStatus.IDLE]),
            "offline": len([p for p in self.participants.values() if p.status == ParticipantStatus.OFFLINE]),
            "byzantine": len([p for p in self.participants.values() if p.status == ParticipantStatus.BYZANTINE]),
            "slashed": len([p for p in self.participants.values() if p.status == ParticipantStatus.SLASHED]),
            "avg_fitness": sum(p.avg_fitness for p in active) / len(active) if active else 0.0,
            "avg_reputation": sum(p.reputation_score for p in self.participants.values()) / len(self.participants),
            "total_contributions": sum(p.rounds_participated for p in self.participants.values()),
            "total_samples": sum(p.total_samples_trained for p in self.participants.values()),
            "total_rewards_distributed": sum(p.total_rewards for p in self.participants.values()),
            "pending_rewards": sum(p.pending_rewards for p in self.participants.values()),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ParticipantStatus",
    "Participant",
    "ParticipantManager",
]
