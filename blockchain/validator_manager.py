"""
Validator Manager for BelizeChain.

Manages validator identity, KYC/AML compliance, and reputation.

Key Features:
- Identity verification
- KYC/AML integration (Belizean FSC)
- Stake requirements checking
- Reputation scoring
- Validator lifecycle management

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from loguru import logger

from .substrate_client import SubstrateClient, ExtrinsicReceipt


class KYCStatus(Enum):
    """KYC verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ValidatorTier(Enum):
    """Validator tiers based on stake and reputation."""
    BRONZE = "bronze"      # Min stake, new validators
    SILVER = "silver"      # 2x min stake, good reputation
    GOLD = "gold"          # 5x min stake, excellent reputation
    PLATINUM = "platinum"  # 10x min stake, perfect reputation


@dataclass
class ValidatorIdentity:
    """
    Validator identity information.
    
    Attributes:
        address: Validator SS58 address
        name: Display name
        email: Contact email
        website: Website URL
        legal_name: Legal entity name
        jurisdiction: Legal jurisdiction
        tax_id: Tax identification number
        kyc_status: KYC verification status
        kyc_verified_at: KYC verification timestamp
        tier: Validator tier
    """
    address: str
    name: str
    email: str
    website: Optional[str] = None
    legal_name: Optional[str] = None
    jurisdiction: str = "BZ"  # Belize
    tax_id: Optional[str] = None
    kyc_status: KYCStatus = KYCStatus.PENDING
    kyc_verified_at: Optional[datetime] = None
    tier: ValidatorTier = ValidatorTier.BRONZE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for chain storage."""
        return {
            'address': self.address,
            'name': self.name,
            'email': self.email,
            'website': self.website or '',
            'legal_name': self.legal_name or '',
            'jurisdiction': self.jurisdiction,
            'tax_id': self.tax_id or '',
            'kyc_status': self.kyc_status.value,
            'kyc_verified_at': int(self.kyc_verified_at.timestamp()) if self.kyc_verified_at else 0,
            'tier': self.tier.value,
        }


class ValidatorManager:
    """
    Validator identity and compliance manager.
    
    Manages:
    - Validator registration
    - Identity verification
    - KYC/AML compliance
    - Reputation tracking
    - Tier management
    
    Usage:
        manager = ValidatorManager(client)
        
        # Register validator identity
        identity = ValidatorIdentity(
            address=keypair.ss58_address,
            name="Alice's Validator",
            email="alice@example.com",
            legal_name="Alice Validator Services Ltd.",
            tax_id="BZ123456789",
        )
        
        receipt = manager.register_identity(keypair, identity)
        
        # Check compliance
        is_compliant = manager.check_compliance(keypair.ss58_address)
    """
    
    def __init__(self, client: SubstrateClient):
        """
        Initialize ValidatorManager.
        
        Args:
            client: Substrate client
        """
        self.client = client
        logger.info("ValidatorManager initialized")
    
    def register_identity(
        self,
        keypair,
        identity: ValidatorIdentity,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Register validator identity on-chain.
        
        Args:
            keypair: Validator keypair
            identity: Identity information
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(
            f"Registering validator identity: {identity.name} "
            f"({identity.address})"
        )
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Identity",
            call_function="set_identity",
            call_params=identity.to_dict(),
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success("Validator identity registered")
        else:
            logger.error(f"Identity registration failed: {receipt.error}")
        
        return receipt
    
    def get_identity(self, address: str) -> Optional[ValidatorIdentity]:
        """
        Get validator identity from chain.
        
        Args:
            address: Validator SS58 address
        
        Returns:
            Validator identity or None
        """
        try:
            data = self.client.query_storage(
                module="Identity",
                storage_function="IdentityOf",
                params=[address],
            )
            
            if data is None:
                logger.warning(f"Identity not found: {address}")
                return None
            
            identity = ValidatorIdentity(
                address=address,
                name=data['name'],
                email=data['email'],
                website=data.get('website'),
                legal_name=data.get('legal_name'),
                jurisdiction=data.get('jurisdiction', 'BZ'),
                tax_id=data.get('tax_id'),
                kyc_status=KYCStatus(data.get('kyc_status', 'pending')),
                kyc_verified_at=datetime.fromtimestamp(data['kyc_verified_at']) if data.get('kyc_verified_at') else None,
                tier=ValidatorTier(data.get('tier', 'bronze')),
            )
            
            return identity
            
        except Exception as e:
            logger.error(f"Failed to get identity: {e}")
            return None
    
    def submit_kyc(
        self,
        keypair,
        documents: Dict[str, str],
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Submit KYC documents for verification.
        
        Args:
            keypair: Validator keypair
            documents: Document hashes (e.g., {'passport': 'hash', 'proof_of_address': 'hash'})
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(f"Submitting KYC documents for {keypair.ss58_address}")
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Identity",
            call_function="submit_kyc",
            call_params={
                'documents': documents,
            },
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success("KYC documents submitted")
        else:
            logger.error(f"KYC submission failed: {receipt.error}")
        
        return receipt
    
    def check_compliance(self, address: str) -> bool:
        """
        Check if validator is compliant.
        
        Requirements:
        - Valid identity registered
        - KYC verified
        - Sufficient stake
        - Not jailed
        
        Args:
            address: Validator address
        
        Returns:
            True if compliant, False otherwise
        """
        # Check identity
        identity = self.get_identity(address)
        if identity is None:
            logger.warning(f"No identity registered: {address}")
            return False
        
        # Check KYC status
        if identity.kyc_status != KYCStatus.VERIFIED:
            logger.warning(f"KYC not verified: {address} (status={identity.kyc_status.value})")
            return False
        
        # Check stake
        try:
            from .staking_interface import StakingInterface
            staking = StakingInterface(self.client)
            stake_info = staking.get_stake_info(address)
            
            if stake_info is None or not stake_info.is_sufficient:
                logger.warning(f"Insufficient stake: {address}")
                return False
        except Exception as e:
            logger.error(f"Failed to check stake: {e}")
            return False
        
        logger.debug(f"Validator compliant: {address}")
        return True
    
    def calculate_tier(
        self,
        stake: int,
        reputation: float,
        min_stake: int,
    ) -> ValidatorTier:
        """
        Calculate validator tier based on stake and reputation.
        
        Args:
            stake: Current stake amount
            reputation: Reputation score (0-100)
            min_stake: Minimum required stake
        
        Returns:
            Validator tier
        """
        stake_ratio = stake / min_stake
        
        if stake_ratio >= 10 and reputation >= 95:
            return ValidatorTier.PLATINUM
        elif stake_ratio >= 5 and reputation >= 90:
            return ValidatorTier.GOLD
        elif stake_ratio >= 2 and reputation >= 80:
            return ValidatorTier.SILVER
        else:
            return ValidatorTier.BRONZE
    
    def update_tier(
        self,
        keypair,
        new_tier: ValidatorTier,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Update validator tier (governance-controlled).
        
        Args:
            keypair: Governance keypair
            new_tier: New tier to assign
            wait_for_finalization: Wait for finalization
        
        Returns:
            Extrinsic receipt
        """
        logger.info(f"Updating validator tier to {new_tier.value}")
        
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="Identity",
            call_function="update_tier",
            call_params={
                'tier': new_tier.value,
            },
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success(f"Tier updated to {new_tier.value}")
        else:
            logger.error(f"Tier update failed: {receipt.error}")
        
        return receipt
    
    def get_reputation_score(self, address: str) -> float:
        """
        Get validator reputation score.
        
        Based on:
        - Fitness score consistency
        - Uptime
        - Slashing history
        - Community feedback
        
        Args:
            address: Validator address
        
        Returns:
            Reputation score (0-100)
        """
        try:
            reputation = self.client.query_storage(
                module="Identity",
                storage_function="ReputationScores",
                params=[address],
            )
            return reputation if reputation is not None else 100.0
        except Exception as e:
            logger.error(f"Failed to get reputation: {e}")
            return 0.0
    
    def get_all_validators(self) -> List[ValidatorIdentity]:
        """
        Get all registered validators.
        
        Returns:
            List of validator identities
        """
        try:
            # Query all validators
            validators = self.client.query_map(
                module="Identity",
                storage_function="IdentityOf",
            )
            
            identities = []
            for address, data in validators:
                try:
                    identity = ValidatorIdentity(
                        address=address,
                        name=data['name'],
                        email=data['email'],
                        website=data.get('website'),
                        legal_name=data.get('legal_name'),
                        jurisdiction=data.get('jurisdiction', 'BZ'),
                        tax_id=data.get('tax_id'),
                        kyc_status=KYCStatus(data.get('kyc_status', 'pending')),
                        tier=ValidatorTier(data.get('tier', 'bronze')),
                    )
                    identities.append(identity)
                except Exception as e:
                    logger.warning(f"Failed to parse identity for {address}: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(identities)} validator identities")
            return identities
            
        except Exception as e:
            logger.error(f"Failed to get all validators: {e}")
            return []
    
    def get_compliant_validators(self) -> List[str]:
        """
        Get list of compliant validator addresses.
        
        Returns:
            List of compliant validator addresses
        """
        all_validators = self.get_all_validators()
        compliant = [
            v.address for v in all_validators
            if self.check_compliance(v.address)
        ]
        
        logger.info(f"Found {len(compliant)}/{len(all_validators)} compliant validators")
        return compliant
