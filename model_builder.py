"""
BelizeChain Model Builder - Main Module

This module provides the primary interface for building PyTorch models from genome specifications.
It re-exports the ModelBuilder class from the genome package for backward compatibility with
legacy code that imports from nawal.model_builder.

New code should import directly from nawal.genome.model_builder for better organization.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from nawal.genome.model_builder import (
    ModelBuilder,
    ActivationFactory,
    NormalizationFactory,
    AttentionFactory,
    LayerFactory,
    GenomeModel,
)

__all__ = [
    "ModelBuilder",
    "ActivationFactory",
    "NormalizationFactory",
    "AttentionFactory",
    "LayerFactory",
    "GenomeModel",
]
