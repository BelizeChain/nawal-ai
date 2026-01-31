"""
Community Connector - Connect to BelizeChain Community Pallet

Manages SRS (Social Responsibility Score) integration for Nawal AI participants.
Records federated learning contributions, tracks education module completions,
and monitors green project contributions.

Author: BelizeChain AI Team
Date: January 2026
Python: 3.13+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SRSInfo:
    """Social Responsibility Score information for an account."""
    
    account_id: str
    score: int  # 0-10,000
    tier: int  # 1-5 (Bronze, Silver, Gold, Platinum, Diamond)
    participation_count: int
    volunteer_hours: int
    education_modules_completed: int
    green_project_contributions: int  # in Planck
    monthly_fee_exemption: int  # in Planck
    last_updated: int  # timestamp


@dataclass
class ParticipationRecord:
    """Record of community participation activity."""
    
    account_id: str
    activity_type: str  # 'FederatedLearning', 'EducationModule', 'GreenProject', 'Volunteer'
    points_earned: int
    metadata: dict[str, Any]
    timestamp: int


# =============================================================================
# Community Connector
# =============================================================================


class CommunityConnector:
    """
    Connector to BelizeChain Community pallet for SRS tracking.
    
    Integrates with Nawal AI federated learning to automatically
    record participation and update SRS scores for contributors.
    """
    
    def __init__(
        self,
        websocket_url: str = "ws://127.0.0.1:9944",
        keypair: Keypair | None = None,
        mock_mode: bool = False
    ):
        """
        Initialize Community pallet connector.
        
        Args:
            websocket_url: BelizeChain node WebSocket endpoint
            keypair: Account keypair for signing transactions (optional)
            mock_mode: Use mock responses instead of real blockchain (for testing)
        """
        self.websocket_url = websocket_url
        self.keypair = keypair
        self.mock_mode = mock_mode or not SUBSTRATE_AVAILABLE
        self.substrate: SubstrateInterface | None = None
        self._connected = False
        
        if self.mock_mode:
            logger.warning("Community connector running in MOCK MODE")
    
    async def connect(self) -> bool:
        """
        Connect to BelizeChain node.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.mock_mode:
            logger.info("Mock mode: simulating connection")
            self._connected = True
            return True
        
        try:
            self.substrate = SubstrateInterface(
                url=self.websocket_url,
                ss58_format=42,  # BelizeChain uses Substrate default
                type_registry_preset='polkadot'
            )
            self._connected = True
            logger.info(f"Connected to BelizeChain at {self.websocket_url}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to BelizeChain: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from BelizeChain node."""
        if self.substrate:
            self.substrate.close()
            self._connected = False
            logger.info("Disconnected from BelizeChain")
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    async def get_srs_info(self, account_id: str) -> SRSInfo | None:
        """
        Get Social Responsibility Score information for an account.
        
        Args:
            account_id: Account address (SS58 format)
        
        Returns:
            SRSInfo object or None if account not found
        """
        if self.mock_mode:
            logger.info(f"Mock: Getting SRS info for {account_id[:8]}...")
            return SRSInfo(
                account_id=account_id,
                score=3500,
                tier=2,  # Silver
                participation_count=12,
                volunteer_hours=25,
                education_modules_completed=2,
                green_project_contributions=5000 * 10**12,  # 5000 DALLA
                monthly_fee_exemption=100 * 10**12,  # 100 DALLA
                last_updated=int(datetime.now(timezone.utc).timestamp())
            )
        
        if not self._connected or not self.substrate:
            logger.error("Not connected to blockchain")
            return None
        
        try:
            result = self.substrate.query(
                module='Community',
                storage_function='SocialResponsibilityScores',
                params=[account_id]
            )
            
            if not result or result.value is None:
                logger.warning(f"No SRS data found for {account_id}")
                return None
            
            data = result.value
            return SRSInfo(
                account_id=account_id,
                score=int(data['score']),
                tier=int(data['tier']),
                participation_count=int(data['participation_count']),
                volunteer_hours=int(data['volunteer_hours']),
                education_modules_completed=int(data['education_modules_completed']),
                green_project_contributions=int(data['green_project_contributions']),
                monthly_fee_exemption=int(data['monthly_fee_exemption']),
                last_updated=int(data['last_updated'])
            )
        
        except Exception as e:
            logger.error(f"Failed to query SRS info: {e}")
            return None
    
    async def get_tier_name(self, tier: int) -> str:
        """Get human-readable tier name."""
        tiers = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond"}
        return tiers.get(tier, "None")
    
    # =========================================================================
    # Transaction Methods
    # =========================================================================
    
    async def record_participation(
        self,
        account_id: str,
        activity_type: str,
        quality_score: float | None = None,
        contribution_amount: int | None = None,
        metadata: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """
        Record community participation activity (updates SRS).
        
        Args:
            account_id: Participant account address
            activity_type: Type of activity ('FederatedLearning', 'EducationModule', etc.)
            quality_score: Quality score for the activity (0-100, optional)
            contribution_amount: Financial contribution in Planck (optional)
            metadata: Additional activity metadata (optional)
        
        Returns:
            (success: bool, tx_hash: str)
        """
        if self.mock_mode:
            logger.info(f"Mock: Recording {activity_type} participation for {account_id[:8]}...")
            return (True, "0xMOCK_TX_HASH")
        
        if not self._connected or not self.substrate:
            logger.error("Not connected to blockchain")
            return (False, "")
        
        if not self.keypair:
            logger.error("No keypair configured for signing transactions")
            return (False, "")
        
        try:
            # Build extrinsic call
            call = self.substrate.compose_call(
                call_module='Community',
                call_function='record_participation',
                call_params={
                    'account': account_id,
                    'activity_type': activity_type,
                    'quality_score': quality_score if quality_score is not None else 0,
                    'contribution_amount': contribution_amount if contribution_amount is not None else 0,
                    'metadata': str(metadata) if metadata else ""
                }
            )
            
            # Create signed extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            # Submit extrinsic
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info(f"Participation recorded: {receipt.extrinsic_hash}")
                return (True, receipt.extrinsic_hash)
            else:
                logger.error(f"Participation record failed: {receipt.error_message}")
                return (False, "")
        
        except Exception as e:
            logger.error(f"Failed to record participation: {e}")
            return (False, "")
    
    async def record_federated_learning_contribution(
        self,
        account_id: str,
        round_number: int,
        quality_score: float,
        samples_trained: int,
        training_duration_seconds: int
    ) -> tuple[bool, str]:
        """
        Record federated learning contribution (Nawal AI specific).
        
        Args:
            account_id: Validator account
            round_number: FL round number
            quality_score: Model improvement quality (0-100)
            samples_trained: Number of samples used in training
            training_duration_seconds: Time spent training
        
        Returns:
            (success: bool, tx_hash: str)
        """
        metadata = {
            'round_number': round_number,
            'samples_trained': samples_trained,
            'training_duration': training_duration_seconds,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return await self.record_participation(
            account_id=account_id,
            activity_type='FederatedLearning',
            quality_score=quality_score,
            metadata=metadata
        )
    
    async def record_education_completion(
        self,
        account_id: str,
        module_id: int,
        completion_score: float
    ) -> tuple[bool, str]:
        """
        Record education module completion.
        
        Args:
            account_id: Student account
            module_id: Education module ID
            completion_score: Final score (0-100)
        
        Returns:
            (success: bool, tx_hash: str)
        """
        metadata = {
            'module_id': module_id,
            'completion_date': datetime.now(timezone.utc).isoformat()
        }
        
        return await self.record_participation(
            account_id=account_id,
            activity_type='EducationModule',
            quality_score=completion_score,
            metadata=metadata
        )
    
    async def record_green_project_contribution(
        self,
        account_id: str,
        project_id: int,
        amount_dalla: float
    ) -> tuple[bool, str]:
        """
        Record green project contribution.
        
        Args:
            account_id: Contributor account
            project_id: Green project ID
            amount_dalla: Contribution amount in DALLA
        
        Returns:
            (success: bool, tx_hash: str)
        """
        amount_planck = int(amount_dalla * 10**12)
        metadata = {
            'project_id': project_id,
            'contribution_date': datetime.now(timezone.utc).isoformat()
        }
        
        return await self.record_participation(
            account_id=account_id,
            activity_type='GreenProject',
            contribution_amount=amount_planck,
            metadata=metadata
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def format_balance(self, planck: int) -> str:
        """Convert Planck (smallest unit) to DALLA."""
        return f"{planck / 10**12:.4f} DALLA"
    
    def parse_balance(self, dalla: float) -> int:
        """Convert DALLA to Planck (smallest unit)."""
        return int(dalla * 10**12)


# =============================================================================
# Usage Example
# =============================================================================


async def example_usage():
    """Example usage of CommunityConnector."""
    
    # Initialize connector (mock mode for demonstration)
    connector = CommunityConnector(
        websocket_url="ws://127.0.0.1:9944",
        mock_mode=True
    )
    
    # Connect to blockchain
    await connector.connect()
    
    # Get SRS info
    account = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    srs_info = await connector.get_srs_info(account)
    
    if srs_info:
        print(f"SRS Score: {srs_info.score}")
        print(f"Tier: {await connector.get_tier_name(srs_info.tier)}")
        print(f"Participation: {srs_info.participation_count} activities")
        print(f"Education: {srs_info.education_modules_completed} modules")
    
    # Record federated learning contribution
    success, tx_hash = await connector.record_federated_learning_contribution(
        account_id=account,
        round_number=42,
        quality_score=87.5,
        samples_trained=1000,
        training_duration_seconds=3600
    )
    
    print(f"Participation recorded: {success} (tx: {tx_hash})")
    
    # Disconnect
    await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
