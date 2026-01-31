"""
Command Line Interface for Nawal AI.

Provides user-friendly commands for training, evolution,
federated learning, and blockchain operations.

Commands:
- nawal train: Local model training
- nawal evolve: Evolutionary optimization
- nawal federate: Federated learning server
- nawal validator: Validator operations
- nawal genome: Genome management
- nawal config: Configuration management

Author: BelizeChain Team
License: MIT
"""

from .commands import cli

__all__ = ["cli"]
__version__ = "0.1.0"
