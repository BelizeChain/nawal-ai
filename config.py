"""
Nawal Configuration System

Pydantic v2-based configuration with YAML/JSON support and environment overrides.

Features:
- Type-safe configuration models
- YAML/JSON file loading
- Environment variable overrides
- Validation with defaults
- Easy validator setup

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from loguru import logger


# =============================================================================
# Evolution Configuration
# =============================================================================


class EvolutionConfig(BaseModel):
    """Configuration for genome evolution."""
    
    # Population
    population_size: int = Field(
        default=50,
        ge=2,
        le=1000,
        description="Number of genomes in population",
    )
    
    # Generations
    num_generations: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of evolution generations",
    )
    
    # Selection
    selection_pressure: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Fraction of population selected for breeding",
    )
    
    # Mutation
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of mutation per genome",
    )
    
    mutation_strength: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Strength of mutations (0=small, 1=large)",
    )
    
    # Crossover
    crossover_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability of crossover",
    )
    
    # Elitism
    elitism_count: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of top genomes to preserve unchanged",
    )
    
    # Diversity
    diversity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum diversity to maintain",
    )
    
    # Checkpointing
    checkpoint_frequency: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Save checkpoint every N generations",
    )
    
    @field_validator("elitism_count")
    @classmethod
    def validate_elitism(cls, v: int, info) -> int:
        """Ensure elitism_count is less than population_size."""
        if "population_size" in info.data and v >= info.data["population_size"]:
            raise ValueError(f"elitism_count ({v}) must be < population_size ({info.data['population_size']})")
        return v


# =============================================================================
# Federated Learning Configuration
# =============================================================================


class FederatedConfig(BaseModel):
    """Configuration for federated learning."""
    
    # Strategy
    aggregation_strategy: Literal["fedavg", "fedprox", "fedadam"] = Field(
        default="fedavg",
        description="Federated aggregation strategy",
    )
    
    # Participants
    min_participants: int = Field(
        default=3,
        ge=1,
        le=10000,
        description="Minimum participants required for aggregation",
    )
    
    max_participants: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum participants per round",
    )
    
    # Client selection (backward compatibility)
    client_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of clients to select per round",
    )
    
    min_clients: int = Field(
        default=3,
        ge=1,
        le=10000,
        description="Minimum number of clients to select",
    )
    
    # Rounds
    num_rounds: int = Field(
        default=100,
        ge=1,
        le=100000,
        description="Number of federated learning rounds",
    )
    
    # Thresholds
    aggregation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of participants required",
    )
    
    # Timeouts
    round_timeout: float = Field(
        default=3600.0,
        ge=60.0,
        le=86400.0,
        description="Round timeout in seconds",
    )
    
    participant_timeout: float = Field(
        default=1800.0,
        ge=30.0,
        le=7200.0,
        description="Participant submission timeout in seconds",
    )
    
    # Quality control
    min_samples: int = Field(
        default=100,
        ge=1,
        le=1000000,
        description="Minimum samples per participant",
    )
    
    min_quality_score: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Minimum quality score to accept update",
    )


# =============================================================================
# Training Configuration
# =============================================================================


class TrainingConfig(BaseModel):
    """Configuration for local training."""
    
    # Optimization
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-8,
        le=1.0,
        description="Learning rate",
    )
    
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size",
    )
    
    local_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Local training epochs",
    )
    
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Maximum gradient norm for clipping",
    )
    
    warmup_steps: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Learning rate warmup steps",
    )
    
    # Optimizer
    optimizer: Literal["adamw", "adam", "sgd", "rmsprop"] = Field(
        default="adamw",
        description="Optimizer type",
    )
    
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay (L2 regularization)",
    )
    
    # Scheduler
    scheduler: Literal["cosine", "linear", "none"] | None = Field(
        default="cosine",
        description="Learning rate scheduler",
    )
    
    # Performance
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Device to use for training",
    )
    
    mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision (FP16) training",
    )
    
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Gradient accumulation steps",
    )
    
    # Privacy
    privacy_epsilon: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Differential privacy epsilon (None = disabled)",
    )
    
    gradient_clipping: bool = Field(
        default=True,
        description="Enable gradient clipping",
    )
    
    # Backward compatibility fields
    participant_id: str | None = Field(
        default=None,
        description="Participant ID (backward compatibility)",
    )
    
    epochs: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Training epochs (backward compatibility - same as local_epochs)",
    )
    
    loss_function: str | None = Field(
        default=None,
        description="Loss function name (backward compatibility)",
    )
    
    gradient_clip: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Gradient clipping value (backward compatibility - same as max_grad_norm)",
    )


# =============================================================================
# Model Configuration
# =============================================================================


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    
    # Vocabulary
    vocab_size: int = Field(
        default=50257,
        ge=1000,
        le=1000000,
        description="Vocabulary size",
    )
    
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=32768,
        description="Maximum sequence length",
    )
    
    # Architecture constraints
    min_hidden_size: int = Field(
        default=256,
        ge=64,
        le=8192,
        description="Minimum hidden size",
    )
    
    max_hidden_size: int = Field(
        default=4096,
        ge=64,
        le=16384,
        description="Maximum hidden size",
    )
    
    min_layers: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Minimum number of layers",
    )
    
    max_layers: int = Field(
        default=48,
        ge=1,
        le=200,
        description="Maximum number of layers",
    )


# =============================================================================
# Compliance Configuration
# =============================================================================


class ComplianceConfig(BaseModel):
    """Configuration for Belizean compliance and data sovereignty."""
    
    # Data sovereignty
    data_sovereignty_check: bool = Field(
        default=True,
        description="Enforce data sovereignty checks",
    )
    
    compliance_mode: bool = Field(
        default=True,
        description="Enable Belizean regulatory compliance",
    )
    
    # KYC/AML
    require_kyc: bool = Field(
        default=True,
        description="Require KYC verification for validators",
    )
    
    kyc_provider: str = Field(
        default="belizean_fsc",
        description="KYC provider",
    )
    
    # Auditing
    audit_logging: bool = Field(
        default=True,
        description="Enable audit logging",
    )
    
    audit_retention_days: int = Field(
        default=730,
        ge=30,
        le=3650,
        description="Audit log retention in days (2 years default)",
    )
    
    # Privacy
    data_encryption: bool = Field(
        default=True,
        description="Encrypt data at rest and in transit",
    )
    
    anonymization: bool = Field(
        default=True,
        description="Anonymize participant identities",
    )


# =============================================================================
# Storage Configuration
# =============================================================================


class StorageConfig(BaseModel):
    """Configuration for storage and checkpointing."""
    
    # Paths
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory for checkpoints",
    )
    
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory for logs",
    )
    
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for data",
    )
    
    # Checkpoint settings
    save_best_only: bool = Field(
        default=False,
        description="Save only the best checkpoint",
    )
    
    max_checkpoints: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of checkpoints to keep",
    )
    
    # Compression
    compress_checkpoints: bool = Field(
        default=True,
        description="Compress checkpoints",
    )
    
    @field_validator("checkpoint_dir", "log_dir", "data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand environment variables and resolve paths."""
        path = Path(v).expanduser()
        return path.resolve() if path.is_absolute() else path


# =============================================================================
# Main Configuration
# =============================================================================


class NawalConfig(BaseModel):
    """Main Nawal configuration."""
    
    # System
    project_name: str = Field(
        default="nawal",
        description="Project name",
    )
    
    version: str = Field(
        default="0.1.0",
        description="Configuration version",
    )
    
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environment",
    )
    
    # Components
    evolution: EvolutionConfig = Field(
        default_factory=EvolutionConfig,
        description="Evolution configuration",
    )
    
    federated: FederatedConfig = Field(
        default_factory=FederatedConfig,
        description="Federated learning configuration",
    )
    
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )
    
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration",
    )
    
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Compliance configuration",
    )
    
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration",
    )
    
    # Validator identity
    validator_id: str | None = Field(
        default=None,
        description="Validator identifier",
    )
    
    validator_address: str | None = Field(
        default=None,
        description="Validator blockchain address",
    )
    
    staking_account: str | None = Field(
        default=None,
        description="Staking account identifier",
    )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> NawalConfig:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
        
        Returns:
            NawalConfig instance
        """
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {path}")
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str | Path) -> NawalConfig:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            NawalConfig instance
        """
        import json
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        logger.info(f"Loaded configuration from {path}")
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "NAWAL_") -> NawalConfig:
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format:
        NAWAL_EVOLUTION__POPULATION_SIZE=100
        NAWAL_TRAINING__LEARNING_RATE=0.001
        
        Args:
            prefix: Environment variable prefix
        
        Returns:
            NawalConfig instance
        """
        config_dict: dict[str, Any] = {}
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Remove prefix and convert to nested dict
            key = key[len(prefix):].lower()
            parts = key.split("__")
            
            # Navigate/create nested structure
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value (try to parse as number/bool)
            final_key = parts[-1]
            try:
                # Try bool
                if value.lower() in ("true", "false"):
                    current[final_key] = value.lower() == "true"
                # Try int
                elif value.isdigit():
                    current[final_key] = int(value)
                # Try float
                elif "." in value and value.replace(".", "").replace("-", "").isdigit():
                    current[final_key] = float(value)
                # String
                else:
                    current[final_key] = value
            except (ValueError, AttributeError):
                current[final_key] = value
        
        logger.info(f"Loaded configuration from environment variables (prefix={prefix})")
        return cls(**config_dict)
    
    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to YAML file
        """
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        
        logger.info(f"Saved configuration to {path}")
    
    def to_json(self, path: str | Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to JSON file
        """
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(
                self.model_dump(),
                f,
                indent=2,
                default=str,
            )
        
        logger.info(f"Saved configuration to {path}")
    
    def create_directories(self) -> None:
        """Create all configured directories."""
        self.storage.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.storage.log_dir.mkdir(parents=True, exist_ok=True)
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created storage directories")


# =============================================================================
# Configuration Factory
# =============================================================================


def load_config(
    config_path: str | Path | None = None,
    env_prefix: str = "NAWAL_",
) -> NawalConfig:
    """
    Load configuration with automatic format detection.
    
    Priority:
    1. Explicit config file (YAML or JSON)
    2. Environment variables
    3. Defaults
    
    Args:
        config_path: Path to config file (YAML or JSON)
        env_prefix: Environment variable prefix
    
    Returns:
        NawalConfig instance
    """
    # Try explicit config file
    if config_path:
        path = Path(config_path)
        if path.suffix in (".yaml", ".yml"):
            return NawalConfig.from_yaml(path)
        elif path.suffix == ".json":
            return NawalConfig.from_json(path)
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
    
    # Try environment variables
    env_config = NawalConfig.from_env(env_prefix)
    if env_config.validator_id or env_config.validator_address:
        logger.info("Using configuration from environment variables")
        return env_config
    
    # Use defaults
    logger.info("Using default configuration")
    return NawalConfig()


# =============================================================================
# Backward Compatibility
# =============================================================================

# Old tests expect "GenomeConfig" class name
GenomeConfig = EvolutionConfig


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "NawalConfig",
    "EvolutionConfig",
    "FederatedConfig",
    "TrainingConfig",
    "ModelConfig",
    "ComplianceConfig",
    "StorageConfig",
    "GenomeConfig",  # Alias for backward compatibility
    "load_config",
]
