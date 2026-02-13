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
from .mesh_network import (
    MeshNetworkClient,
    MeshMessage,
    MessageType,
    PeerInfo,
    FLRoundAnnouncement,
)
from .payroll_connector import (
    PayrollConnector,
    PayrollEntry,
    PayrollSubmission,
    PayrollStatus,
    EmployeeType,
    EmployeePaystub,
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
    # Mesh Network
    "MeshNetworkClient",
    "MeshMessage",
    "MessageType",
    "PeerInfo",
    "FLRoundAnnouncement",
    # Payroll
    "PayrollConnector",
    "PayrollEntry",
    "PayrollSubmission",
    "PayrollStatus",
    "EmployeeType",
    "EmployeePaystub",
]

__version__ = "0.2.0"  # Updated for mesh network and payroll integration
