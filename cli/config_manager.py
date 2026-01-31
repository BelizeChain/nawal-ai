"""
Configuration Management for Nawal AI.

Handles loading, validation, and merging of configuration files
from multiple sources (YAML/JSON files, environment variables, CLI args).

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Any, Optional
from pathlib import Path
import os
import json

# Optional YAML library
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigManager:
    """
    Configuration manager for Nawal AI.
    
    Manages configuration files, profiles, and environment variables.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.profiles = ["dev", "test", "prod"]
        self.active_profile = os.getenv("NAWAL_PROFILE", "dev")
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Determine format
        suffix = config_path.suffix.lower()
        
        if suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML configs. Install: pip install pyyaml")
            
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        
        elif suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        # Merge with environment variables
        config = self._merge_env_vars(config)
        
        # Validate
        self._validate_config(config)
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        # Validate before saving
        self._validate_config(config)
        
        # Determine format
        suffix = config_path.suffix.lower()
        
        # Create parent directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required. Install: pip install pyyaml")
            
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        
        elif suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    
    def create_default_config(self, config_path: Path) -> None:
        """
        Create default configuration file.
        
        Args:
            config_path: Path to create configuration
        """
        default_config = self._get_default_config()
        self.save_config(default_config, config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "profile": "dev",
            
            # Training configuration
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "checkpoint_dir": "./checkpoints",
                "save_interval": 1,
            },
            
            # Evolution configuration
            "evolution": {
                "num_generations": 20,
                "population_size": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "selection_method": "tournament",
                "tournament_size": 5,
                "elitism_count": 2,
                "checkpoint_dir": "./evolution",
            },
            
            # Federated learning configuration
            "federated": {
                "num_clients": 10,
                "num_rounds": 100,
                "min_clients_per_round": 5,
                "client_fraction": 0.5,
                "local_epochs": 5,
                "aggregation_strategy": "fedavg",
                "host": "127.0.0.1",  # Localhost by default for security
                "port": 8080,
            },
            
            # Blockchain configuration
            "blockchain": {
                "chain": "local",
                "rpc_url": "ws://localhost:9944",
                "keypair_uri": "//Alice",
                "timeout": 30,
                "max_retries": 3,
            },
            
            # Data configuration
            "data": {
                "dataset": "wikitext-2",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "seed": 42,
                "cache_dir": "./data_cache",
            },
            
            # Model configuration
            "model": {
                "architecture": "transformer",
                "hidden_size": 256,
                "num_layers": 4,
                "num_heads": 8,
                "dropout": 0.1,
            },
            
            # Security configuration
            "security": {
                "differential_privacy": {
                    "enabled": False,
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "max_grad_norm": 1.0,
                },
                "secure_aggregation": {
                    "enabled": False,
                    "threshold": 3,
                },
                "byzantine_detection": {
                    "enabled": True,
                    "strategy": "krum",
                    "byzantine_threshold": 0.3,
                },
            },
            
            # Monitoring configuration
            "monitoring": {
                "prometheus_enabled": False,
                "prometheus_port": 9090,
                "log_level": "INFO",
                "log_file": "./logs/nawal.log",
            },
        }
    
    def _merge_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge environment variables into configuration.
        
        Environment variables override file-based config.
        Naming convention: NAWAL_<SECTION>_<KEY> (e.g., NAWAL_BLOCKCHAIN_RPC_URL)
        
        Args:
            config: Base configuration dictionary
        
        Returns:
            Merged configuration dictionary
        """
        # Common environment variables
        env_mappings = {
            "NAWAL_PROFILE": ("profile",),
            "NAWAL_RPC_URL": ("blockchain", "rpc_url"),
            "NAWAL_KEYPAIR_URI": ("blockchain", "keypair_uri"),
            "NAWAL_CHAIN": ("blockchain", "chain"),
            "NAWAL_LOG_LEVEL": ("monitoring", "log_level"),
            "NAWAL_DATASET": ("data", "dataset"),
            "NAWAL_EPOCHS": ("training", "epochs"),
            "NAWAL_BATCH_SIZE": ("training", "batch_size"),
            "NAWAL_LEARNING_RATE": ("training", "learning_rate"),
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to nested dict
                current = config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set value (with type conversion)
                final_key = path[-1]
                if isinstance(current.get(final_key), int):
                    current[final_key] = int(value)
                elif isinstance(current.get(final_key), float):
                    current[final_key] = float(value)
                elif isinstance(current.get(final_key), bool):
                    current[final_key] = value.lower() in ["true", "1", "yes"]
                else:
                    current[final_key] = value
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ["training", "evolution", "federated", "blockchain"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate training config
        training = config["training"]
        if training["epochs"] <= 0:
            raise ValueError("Training epochs must be positive")
        if training["batch_size"] <= 0:
            raise ValueError("Batch size must be positive")
        if training["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate evolution config
        evolution = config["evolution"]
        if evolution["num_generations"] <= 0:
            raise ValueError("Number of generations must be positive")
        if evolution["population_size"] <= 0:
            raise ValueError("Population size must be positive")
        if not (0 <= evolution["mutation_rate"] <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if not (0 <= evolution["crossover_rate"] <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        
        # Validate federated config
        federated = config["federated"]
        if federated["num_clients"] <= 0:
            raise ValueError("Number of clients must be positive")
        if federated["num_rounds"] <= 0:
            raise ValueError("Number of rounds must be positive")
        if federated["min_clients_per_round"] > federated["num_clients"]:
            raise ValueError("Min clients per round cannot exceed total clients")
        
        # Validate blockchain config
        blockchain = config["blockchain"]
        valid_chains = ["local", "testnet", "mainnet"]
        if blockchain["chain"] not in valid_chains:
            raise ValueError(f"Chain must be one of: {valid_chains}")
    
    def get_profile_config(self, profile: str) -> Dict[str, Any]:
        """
        Get configuration for specific profile.
        
        Args:
            profile: Profile name (dev, test, prod)
        
        Returns:
            Profile-specific configuration
        
        Raises:
            ValueError: If profile is invalid
        """
        if profile not in self.profiles:
            raise ValueError(f"Invalid profile: {profile}. Must be one of: {self.profiles}")
        
        config = self._get_default_config()
        config["profile"] = profile
        
        # Adjust for profile
        if profile == "dev":
            config["training"]["epochs"] = 5
            config["evolution"]["num_generations"] = 10
            config["federated"]["num_rounds"] = 20
            config["blockchain"]["chain"] = "local"
            config["monitoring"]["log_level"] = "DEBUG"
        
        elif profile == "test":
            config["training"]["epochs"] = 10
            config["evolution"]["num_generations"] = 20
            config["federated"]["num_rounds"] = 50
            config["blockchain"]["chain"] = "testnet"
            config["monitoring"]["log_level"] = "INFO"
        
        elif profile == "prod":
            config["training"]["epochs"] = 50
            config["evolution"]["num_generations"] = 100
            config["federated"]["num_rounds"] = 500
            config["blockchain"]["chain"] = "mainnet"
            config["monitoring"]["log_level"] = "WARNING"
            config["security"]["differential_privacy"]["enabled"] = True
            config["security"]["secure_aggregation"]["enabled"] = True
        
        return config
    
    def set_profile(self, profile: str) -> None:
        """
        Set active profile.
        
        Args:
            profile: Profile name to activate
        
        Raises:
            ValueError: If profile is invalid
        """
        if profile not in self.profiles:
            raise ValueError(f"Invalid profile: {profile}. Must be one of: {self.profiles}")
        
        self.active_profile = profile
        os.environ["NAWAL_PROFILE"] = profile
