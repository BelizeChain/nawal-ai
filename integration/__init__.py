"""
Integration layer between Oracle pallet IoT data and Nawal domain models.

This module provides data pipelines for:
- Fetching IoT data from Oracle pallet storage
- Preprocessing data for domain-specific models
- Running inference with appropriate domain models
- Submitting predictions back to Oracle pallet
- Tracking PoUW rewards for operators
"""

from .oracle_pipeline import (
    OracleDataFetcher,
    DataPreprocessor,
    ModelInferenceRunner,
    ResultSubmitter,
    OraclePipeline,
)

__all__ = [
    'OracleDataFetcher',
    'DataPreprocessor',
    'ModelInferenceRunner',
    'ResultSubmitter',
    'OraclePipeline',
]
