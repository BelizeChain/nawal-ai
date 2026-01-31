"""
Data Manager for Nawal AI.

Handles dataset loading, caching, and distribution for federated learning.

Key Features:
- Multi-format support (HuggingFace, local files, custom)
- Federated data partitioning (IID & non-IID)
- Automatic caching and preprocessing
- Train/val/test splitting
- Batch generation

Supported Dataset Types:
- Text: WikiText-2, WikiText-103, OpenWebText, The Stack
- Vision: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST
- Custom: JSON, CSV, Parquet

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
from loguru import logger

# Optional HuggingFace datasets (not required)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. Install: pip install datasets")


class DatasetType(Enum):
    """Supported dataset types."""
    # Text datasets
    WIKITEXT2 = "wikitext-2-raw-v1"
    WIKITEXT103 = "wikitext-103-raw-v1"
    OPENWEBTEXT = "openwebtext"
    THE_STACK = "the_stack"
    
    # Vision datasets
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    
    # Custom
    CUSTOM_JSON = "custom_json"
    CUSTOM_CSV = "custom_csv"
    CUSTOM_PARQUET = "custom_parquet"


@dataclass
class SplitConfig:
    """
    Configuration for train/val/test splits.
    
    Attributes:
        train_ratio: Fraction for training (0-1)
        val_ratio: Fraction for validation (0-1)
        test_ratio: Fraction for testing (0-1)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle: bool = True
    seed: int = 42
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading.
    
    Attributes:
        dataset_type: Type of dataset to load
        cache_dir: Directory for caching datasets
        split_config: Train/val/test split configuration
        max_samples: Maximum samples to load (None = all)
        num_workers: DataLoader worker processes
        batch_size: Batch size for training
        preprocessing: Custom preprocessing function
    """
    dataset_type: DatasetType
    cache_dir: Path = Path("./data_cache")
    split_config: SplitConfig = field(default_factory=SplitConfig)
    max_samples: Optional[int] = None
    num_workers: int = 4
    batch_size: int = 32
    preprocessing: Optional[Any] = None
    
    # Dataset-specific parameters
    subset: Optional[str] = None  # For multi-subset datasets
    language: Optional[str] = None  # For multilingual datasets
    custom_path: Optional[Path] = None  # For custom datasets


class DataManager:
    """
    Central data management system for Nawal AI.
    
    Handles:
    - Dataset loading from multiple sources
    - Caching for faster reloads
    - Train/val/test splitting
    - Federated data partitioning
    - Batch generation
    
    Usage:
        config = DatasetConfig(
            dataset_type=DatasetType.WIKITEXT2,
            batch_size=32,
        )
        
        manager = DataManager(config)
        train_loader, val_loader, test_loader = manager.get_dataloaders()
        
        # Federated partitioning
        client_loaders = manager.partition_federated(
            num_clients=10,
            iid=False,
        )
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize DataManager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataManager initialized: {config.dataset_type.value}")
    
    def load_dataset(self, force_reload: bool = False) -> None:
        """
        Load dataset from cache or download.
        
        Args:
            force_reload: Force reload even if cached
        """
        cache_path = self._get_cache_path()
        
        # Try loading from cache
        if not force_reload and cache_path.exists():
            logger.info(f"Loading dataset from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    self.dataset = pickle.load(f)
                logger.success("Dataset loaded from cache")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, reloading...")
        
        # Load dataset
        logger.info(f"Loading dataset: {self.config.dataset_type.value}")
        
        if self.config.dataset_type in self._get_hf_datasets():
            self.dataset = self._load_huggingface_dataset()
        elif self.config.dataset_type in self._get_custom_datasets():
            self.dataset = self._load_custom_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_type}")
        
        # Apply max samples limit
        if self.config.max_samples is not None:
            self.dataset = Subset(self.dataset, range(min(self.config.max_samples, len(self.dataset))))
            logger.info(f"Limited to {len(self.dataset)} samples")
        
        # Cache dataset
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.dataset, f)
            logger.success(f"Dataset cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/val/test.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.dataset is None:
            self.load_dataset()
        
        split_config = self.config.split_config
        dataset_size = len(self.dataset)
        
        # Calculate split sizes
        train_size = int(dataset_size * split_config.train_ratio)
        val_size = int(dataset_size * split_config.val_ratio)
        test_size = dataset_size - train_size - val_size
        
        # Random split
        if split_config.shuffle:
            generator = torch.Generator().manual_seed(split_config.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )
        else:
            indices = list(range(dataset_size))
            self.train_dataset = Subset(self.dataset, indices[:train_size])
            self.val_dataset = Subset(self.dataset, indices[train_size:train_size+val_size])
            self.test_dataset = Subset(self.dataset, indices[train_size+val_size:])
        
        logger.info(
            f"Dataset split: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_dataloaders(
        self,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get DataLoaders for train/val/test.
        
        Args:
            shuffle_train: Whether to shuffle training data
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_dataset is None:
            self.split_dataset()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        logger.info(
            f"DataLoaders created: {len(train_loader)} train batches, "
            f"{len(val_loader)} val batches, {len(test_loader)} test batches"
        )
        
        return train_loader, val_loader, test_loader
    
    def partition_federated(
        self,
        num_clients: int,
        iid: bool = True,
        alpha: float = 0.5,
    ) -> List[DataLoader]:
        """
        Partition data for federated learning.
        
        Args:
            num_clients: Number of federated clients
            iid: Whether to use IID (independent & identically distributed) partitioning
            alpha: Dirichlet distribution parameter for non-IID (lower = more skewed)
        
        Returns:
            List of DataLoaders, one per client
        """
        if self.train_dataset is None:
            self.split_dataset()
        
        dataset_size = len(self.train_dataset)
        
        if iid:
            # IID partitioning: random uniform split
            logger.info(f"IID partitioning for {num_clients} clients")
            indices = torch.randperm(dataset_size).tolist()
            client_indices = np.array_split(indices, num_clients)
        else:
            # Non-IID partitioning: Dirichlet distribution
            logger.info(f"Non-IID partitioning (alpha={alpha}) for {num_clients} clients")
            
            # Get labels (assume dataset has 'label' or 'target' attribute)
            try:
                if hasattr(self.train_dataset.dataset, 'targets'):
                    labels = np.array(self.train_dataset.dataset.targets)[self.train_dataset.indices]
                elif hasattr(self.train_dataset.dataset, 'labels'):
                    labels = np.array(self.train_dataset.dataset.labels)[self.train_dataset.indices]
                else:
                    logger.warning("No labels found, falling back to IID")
                    return self.partition_federated(num_clients, iid=True)
                
                num_classes = len(np.unique(labels))
                client_indices = self._dirichlet_partition(labels, num_clients, num_classes, alpha)
            except Exception as e:
                logger.warning(f"Non-IID partitioning failed: {e}, falling back to IID")
                return self.partition_federated(num_clients, iid=True)
        
        # Create DataLoaders for each client
        client_loaders = []
        for i, indices in enumerate(client_indices):
            client_dataset = Subset(self.train_dataset, indices)
            client_loader = DataLoader(
                client_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            client_loaders.append(client_loader)
            logger.debug(f"Client {i}: {len(client_dataset)} samples, {len(client_loader)} batches")
        
        logger.success(f"Federated partitioning complete: {num_clients} clients")
        return client_loaders
    
    def _dirichlet_partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        num_classes: int,
        alpha: float,
    ) -> List[np.ndarray]:
        """
        Partition data using Dirichlet distribution (non-IID).
        
        Args:
            labels: Dataset labels
            num_clients: Number of clients
            num_classes: Number of classes
            alpha: Dirichlet parameter
        
        Returns:
            List of index arrays, one per client
        """
        # Sample proportions from Dirichlet
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        # Organize samples by class
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        # Allocate samples to clients
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            # Shuffle class samples
            np.random.shuffle(class_indices[c])
            
            # Split according to Dirichlet proportions
            proportions = label_distribution[c]
            proportions = proportions / proportions.sum()  # Normalize
            
            splits = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
            class_splits = np.split(class_indices[c], splits)
            
            for i, split in enumerate(class_splits):
                client_indices[i].extend(split)
        
        # Shuffle each client's data
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
            client_indices[i] = np.array(client_indices[i])
        
        return client_indices
    
    def _load_huggingface_dataset(self) -> Dataset:
        """Load dataset from HuggingFace."""
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets not installed")
        
        dataset_name = self.config.dataset_type.value
        
        # Load dataset
        if self.config.subset:
            hf_dataset = load_dataset(dataset_name, self.config.subset)
        else:
            hf_dataset = load_dataset(dataset_name)
        
        # Use train split by default
        if "train" in hf_dataset:
            dataset = hf_dataset["train"]
        else:
            dataset = list(hf_dataset.values())[0]
        
        logger.success(f"Loaded {len(dataset)} samples from HuggingFace")
        return dataset
    
    def _load_custom_dataset(self) -> Dataset:
        """Load custom dataset from local files."""
        if self.config.custom_path is None:
            raise ValueError("custom_path required for custom datasets")
        
        path = self.config.custom_path
        
        if self.config.dataset_type == DatasetType.CUSTOM_JSON:
            return self._load_json_dataset(path)
        elif self.config.dataset_type == DatasetType.CUSTOM_CSV:
            return self._load_csv_dataset(path)
        elif self.config.dataset_type == DatasetType.CUSTOM_PARQUET:
            return self._load_parquet_dataset(path)
        else:
            raise ValueError(f"Unknown custom dataset type: {self.config.dataset_type}")
    
    def _load_json_dataset(self, path: Path) -> Dataset:
        """Load dataset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        logger.success(f"Loaded {len(data)} samples from JSON")
        return ListDataset(data)
    
    def _load_csv_dataset(self, path: Path) -> Dataset:
        """Load dataset from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            data = df.to_dict('records')
            logger.success(f"Loaded {len(data)} samples from CSV")
            return ListDataset(data)
        except ImportError:
            raise RuntimeError("pandas required for CSV loading: pip install pandas")
    
    def _load_parquet_dataset(self, path: Path) -> Dataset:
        """Load dataset from Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            data = df.to_dict('records')
            logger.success(f"Loaded {len(data)} samples from Parquet")
            return ListDataset(data)
        except ImportError:
            raise RuntimeError("pandas required for Parquet loading: pip install pandas")
    
    def _get_cache_path(self) -> Path:
        """Get cache file path for current configuration."""
        # Create hash of configuration
        config_str = f"{self.config.dataset_type.value}_{self.config.subset}_{self.config.max_samples}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_file = f"{self.config.dataset_type.value}_{config_hash}.pkl"
        return self.config.cache_dir / cache_file
    
    @staticmethod
    def _get_hf_datasets() -> List[DatasetType]:
        """Get list of HuggingFace datasets."""
        return [
            DatasetType.WIKITEXT2,
            DatasetType.WIKITEXT103,
            DatasetType.OPENWEBTEXT,
            DatasetType.THE_STACK,
            DatasetType.CIFAR10,
            DatasetType.CIFAR100,
            DatasetType.MNIST,
            DatasetType.FASHION_MNIST,
        ]
    
    @staticmethod
    def _get_custom_datasets() -> List[DatasetType]:
        """Get list of custom dataset types."""
        return [
            DatasetType.CUSTOM_JSON,
            DatasetType.CUSTOM_CSV,
            DatasetType.CUSTOM_PARQUET,
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.dataset is None:
            self.load_dataset()
        
        stats = {
            "dataset_type": self.config.dataset_type.value,
            "total_samples": len(self.dataset),
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.config.batch_size,
        }
        
        return stats


class ListDataset(Dataset):
    """
    Simple dataset wrapper for list of samples.
    
    Used for custom datasets loaded from JSON/CSV/Parquet.
    """
    
    def __init__(self, data: List[Any]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
