"""
Blockchain Integration for Nawal AI.

Connects Nawal's federated learning with BelizeChain's
Proof of Useful Work (PoUW) consensus mechanism.

Key Features:
- Substrate RPC client for BelizeChain
- Staking pallet integration (submit fitness scores)
- Genome registry (on-chain storage with IPFS)
- Validator identity management
- Mesh networking for P2P communication
- ZK-proof payroll integration

Components:
- SubstrateClient: Core blockchain connection
- StakingInterface: PoUW fitness score submission
- GenomeRegistry: On-chain genome storage
- ValidatorManager: Identity and KYC verification
- MeshNetworkClient: P2P mesh networking for validators
- PayrollConnector: Zero-knowledge payroll system

Author: BelizeChain Team
License: MIT
"""

from .genome_registry import (
    GenomeMetadata,
    GenomeRegistry,
    StorageBackend,
)
from .mesh_network import (
    FLRoundAnnouncement,
    MeshMessage,
    MeshNetworkClient,
    MessageType,
    PeerInfo,
)
from .payroll_connector import (
    EmployeePaystub,
    EmployeeType,
    PayrollConnector,
    PayrollEntry,
    PayrollStatus,
    PayrollSubmission,
)
from .staking_interface import (
    FitnessScore,
    StakeInfo,
    StakingInterface,
    ValidatorInfo,
)
from .substrate_client import (
    ChainConfig,
    ExtrinsicReceipt,
    SubstrateClient,
)
from .validator_manager import (
    KYCStatus,
    ValidatorIdentity,
    ValidatorManager,
)

__all__ = [
    "ChainConfig",
    "EmployeePaystub",
    "EmployeeType",
    "ExtrinsicReceipt",
    "FLRoundAnnouncement",
    "FitnessScore",
    "GenomeMetadata",
    # Genome Registry
    "GenomeRegistry",
    "KYCStatus",
    "MeshMessage",
    # Mesh Network
    "MeshNetworkClient",
    "MessageType",
    # Payroll
    "PayrollConnector",
    "PayrollEntry",
    "PayrollStatus",
    "PayrollSubmission",
    "PeerInfo",
    "StakeInfo",
    # Staking Interface
    "StakingInterface",
    "StorageBackend",
    # Substrate Client
    "SubstrateClient",
    "ValidatorIdentity",
    "ValidatorInfo",
    # Validator Manager
    "ValidatorManager",
]

__version__ = "0.2.0"  # Updated for mesh network and payroll integration
