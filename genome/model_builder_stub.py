"""
Model Builder Stub - Temporary compatibility layer

This stub allows imports to work while the full model_builder.py is being fixed.
Once model_builder.py is fully compatible with encoding.py, this file can be removed.

Author: BelizeChain AI Team
Date: October 16, 2025
"""

import torch.nn as nn
from typing import Any


class ModelBuilder:
    """Stub ModelBuilder class for import compatibility."""
    
    def __init__(self):
        pass
    
    def build(self, genome: Any) -> nn.Module:
        """Build a model from genome (stub)."""
        raise NotImplementedError("ModelBuilder.build() is not yet implemented")


class GenomeModel(nn.Module):
    """Stub GenomeModel class for import compatibility."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass (stub)."""
        raise NotImplementedError("GenomeModel.forward() is not yet implemented")


class LayerFactory:
    """Stub LayerFactory class for import compatibility."""
    
    @staticmethod
    def create_layer(layer_type: str, **kwargs) -> nn.Module:
        """Create a layer (stub)."""
        raise NotImplementedError("LayerFactory.create_layer() is not yet implemented")


__all__ = ["ModelBuilder", "GenomeModel", "LayerFactory"]
