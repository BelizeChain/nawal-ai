"""
Blockchain Integration for Nawal AI.

Connects Nawal's federated learning with BelizeChain's
Proof of Useful Work (PoUW) consensus mechanism.

Key Features:
- Substrate RPC client for BelizeChain
- Staking pallet integration (submit fitness scores)
- Genome registry (on-chain storage with IPFS)
- Validator identity management

Components:
- SubstrateClient: Core blockchain connection
- StakingInterface: PoUW fitness score submission
- GenomeRegistry: On-chain genome storage
- ValidatorManager: Identity and KYC verification

Author: BelizeChain Team
License: MIT
"""

from .substrate_client import (
    SubstrateClient,
    ChainConfig,
    ExtrinsicReceipt,
)
from .staking_interface import (
    StakingInterface,
    FitnessScore,
    ValidatorInfo,
    StakeInfo,
)
from .genome_registry import (
    GenomeRegistry,
    GenomeMetadata,
    StorageBackend,
)
from .validator_manager import (
    ValidatorManager,
    ValidatorIdentity,
    KYCStatus,
)

__all__ = [
    # Substrate Client
    "SubstrateClient",
    "ChainConfig",
    "ExtrinsicReceipt",
    # Staking Interface
    "StakingInterface",
    "FitnessScore",
    "ValidatorInfo",
    "StakeInfo",
    # Genome Registry
    "GenomeRegistry",
    "GenomeMetadata",
    "StorageBackend",
    # Validator Manager
    "ValidatorManager",
    "ValidatorIdentity",
    "KYCStatus",
]

__version__ = "0.1.0"
