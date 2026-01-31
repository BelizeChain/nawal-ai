"""
Data Management Module for Nawal AI.

Provides data loading, preprocessing, and tokenization
for federated learning on diverse datasets.

Components:
- DataManager: Main interface for dataset loading
- Tokenizers: Text tokenization (GPT-2, BERT, SentencePiece)
- Preprocessing: Data cleaning, validation, augmentation

Supported Datasets:
- Text: WikiText, OpenWebText, The Stack (code)
- Vision: CIFAR-10, CIFAR-100, ImageNet subset
- Custom: JSON, CSV, Parquet formats

Author: BelizeChain Team
License: MIT
"""

from .data_manager import (
    DataManager,
    DatasetConfig,
    DatasetType,
    SplitConfig,
)
from .tokenizers import (
    TokenizerType,
    TokenizerConfig,
    TextTokenizer,
    create_tokenizer,
)
from .preprocessing import (
    DataPreprocessor,
    TextCleaner,
    DataValidator,
    DataAugmenter,
)

__all__ = [
    # Data Manager
    "DataManager",
    "DatasetConfig",
    "DatasetType",
    "SplitConfig",
    # Tokenizers
    "TokenizerType",
    "TokenizerConfig",
    "TextTokenizer",
    "create_tokenizer",
    # Preprocessing
    "DataPreprocessor",
    "TextCleaner",
    "DataValidator",
    "DataAugmenter",
]

__version__ = "0.1.0"
