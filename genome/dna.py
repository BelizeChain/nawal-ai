"""
Backward compatibility layer for old DNA/LayerGene/ConnectionGene API.

This module provides compatibility with test code written for the old API
while mapping to the new Genome/ArchitectureLayer implementation.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Any, Optional
from .encoding import Genome, ArchitectureLayer, LayerType


class LayerGene:
    """
    Compatibility wrapper for architecture layers.
    
    Old API used LayerGene with:
    - innovation_id: int
    - layer_type: str (e.g., "linear", "conv2d", "relu")
    - params: dict (layer-specific parameters)
    - enabled: bool
    """
    
    def __init__(
        self,
        innovation_id: int,
        layer_type: str,
        params: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.innovation_id = innovation_id
        self.layer_type = layer_type
        self.params = params or {}
        self.enabled = enabled
    
    def to_architecture_layer(self) -> ArchitectureLayer:
        """Convert to new ArchitectureLayer format."""
        # Map old layer_type strings to new LayerType enum
        type_mapping = {
            "linear": LayerType.LINEAR,
            "conv2d": LayerType.CONV2D,
            "conv1d": LayerType.CONV1D,
            "relu": LayerType.RELU,
            "sigmoid": LayerType.SIGMOID,
            "softmax": LayerType.SOFTMAX,
            "gelu": LayerType.GELU,
            "tanh": LayerType.TANH,
            "dropout": LayerType.DROPOUT,
            "batchnorm": LayerType.BATCH_NORM,
            "batchnorm1d": LayerType.BATCH_NORM,
            "layernorm": LayerType.LAYER_NORM,
            "attention": LayerType.MULTIHEAD_ATTENTION,
            "embedding": LayerType.EMBEDDING,
        }
        
        layer_type_enum = type_mapping.get(self.layer_type, LayerType.LINEAR)
        
        return ArchitectureLayer(
            layer_id=str(self.innovation_id),
            layer_type=layer_type_enum,
            parameters=self.params.copy(),
        )
    
    def to_dict(self) -> dict:
        """Serialize LayerGene to dictionary.
        
        Returns:
            Dictionary representation of LayerGene
        """
        return {
            "innovation_id": self.innovation_id,
            "layer_type": self.layer_type,
            "params": self.params.copy(),
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LayerGene":
        """Deserialize LayerGene from dictionary.
        
        Args:
            data: Dictionary representation of LayerGene
            
        Returns:
            LayerGene instance restored from dictionary
        """
        return cls(
            innovation_id=data["innovation_id"],
            layer_type=data["layer_type"],
            params=data.get("params", {}),
            enabled=data.get("enabled", True),
        )
        self.innovation_id = innovation_id
        self.layer_type = layer_type
        self.params = params or {}
        self.enabled = enabled
        
    def to_architecture_layer(self) -> ArchitectureLayer:
        """Convert to new ArchitectureLayer format."""
        # Map old layer_type strings to new LayerType enum
        type_mapping = {
            "linear": LayerType.LINEAR,
            "relu": LayerType.RELU,
            "gelu": LayerType.GELU,
            "sigmoid": LayerType.TANH,  # Close approximation
            "tanh": LayerType.TANH,
            "dropout": LayerType.DROPOUT,
            "batchnorm": LayerType.BATCH_NORM,
            "layernorm": LayerType.LAYER_NORM,
            "attention": LayerType.MULTIHEAD_ATTENTION,
            "embedding": LayerType.EMBEDDING,
        }
        
        layer_enum = type_mapping.get(self.layer_type.lower(), LayerType.LINEAR)
        
        return ArchitectureLayer(
            layer_id=str(self.innovation_id),
            layer_type=layer_enum,
            parameters=self.params,
        )


class ConnectionGene:
    """
    Compatibility wrapper for genome connections.
    
    Old API used ConnectionGene with:
    - innovation_id: int
    - source_layer: int
    - target_layer: int
    - enabled: bool
    """
    
    def __init__(
        self,
        innovation_id: int,
        source_layer: int,
        target_layer: int,
        enabled: bool = True,
    ):
        self.innovation_id = innovation_id
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.enabled = enabled
    
    def to_dict(self) -> dict:
        """Serialize ConnectionGene to dictionary.
        
        Returns:
            Dictionary representation of ConnectionGene
        """
        return {
            "innovation_id": self.innovation_id,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConnectionGene":
        """Deserialize ConnectionGene from dictionary.
        
        Args:
            data: Dictionary representation of ConnectionGene
            
        Returns:
            ConnectionGene instance restored from dictionary
        """
        return cls(
            innovation_id=data["innovation_id"],
            source_layer=data["source_layer"],
            target_layer=data["target_layer"],
            enabled=data.get("enabled", True),
        )


class DNA:
    """
    Compatibility wrapper for Genome class.
    
    Old API used DNA with:
    - input_size: int
    - output_size: int
    - layer_genes: List[LayerGene]
    - connection_genes: List[ConnectionGene]
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_genes: Optional[List[LayerGene]] = None,
        connection_genes: Optional[List[ConnectionGene]] = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_genes = layer_genes or []
        self.connection_genes = connection_genes or []
        self._genome: Optional[Genome] = None
        
    def to_genome(self) -> Genome:
        """Convert to new Genome format."""
        if self._genome is None:
            # Convert LayerGenes to ArchitectureLayers
            layers = [gene.to_architecture_layer() for gene in self.layer_genes if gene.enabled]
            
            self._genome = Genome(
                genome_id=f"dna_{id(self)}",
                generation=0,
                parent_genomes=[],
                encoder_layers=layers,  # Use encoder_layers instead of architecture
                decoder_layers=[],  # Empty decoder for simple DNA
            )
            
        return self._genome
    
    @classmethod
    def from_genome(cls, genome: Genome) -> "DNA":
        """Create DNA instance from Genome."""
        # Get input/output sizes from first/last layer parameters if available
        input_size = 0
        output_size = 0
        
        # Combine encoder and decoder layers
        all_layers = genome.encoder_layers + genome.decoder_layers
        
        if all_layers:
            first_params = all_layers[0].parameters
            last_params = all_layers[-1].parameters
            input_size = first_params.get("in_features", 0)
            output_size = last_params.get("out_features", 0)
        
        dna = cls(
            input_size=input_size,
            output_size=output_size,
        )
        
        # Convert ArchitectureLayers back to LayerGenes
        for layer in all_layers:
            # Map LayerType enum back to string
            type_mapping = {
                LayerType.LINEAR: "linear",
                LayerType.RELU: "relu",
                LayerType.GELU: "gelu",
                LayerType.TANH: "tanh",
                LayerType.DROPOUT: "dropout",
                LayerType.BATCH_NORM: "batchnorm",
                LayerType.LAYER_NORM: "layernorm",
                LayerType.MULTIHEAD_ATTENTION: "attention",
                LayerType.EMBEDDING: "embedding",
            }
            
            layer_type_str = type_mapping.get(layer.layer_type, "linear")
            
            # Try to extract layer_id as int, or use hash
            try:
                innovation_id = int(layer.layer_id)
            except (ValueError, TypeError):
                innovation_id = hash(layer.layer_id) % 1000000
            
            gene = LayerGene(
                innovation_id=innovation_id,
                layer_type=layer_type_str,
                params=layer.parameters.copy(),
                enabled=True,
            )
            dna.layer_genes.append(gene)
        
        dna._genome = genome
        return dna
    
    def add_layer_gene(self, gene: LayerGene) -> None:
        """Add a layer gene."""
        self.layer_genes.append(gene)
        self._genome = None  # Invalidate cached genome
        
    def add_connection_gene(self, gene: ConnectionGene) -> None:
        """Add a connection gene."""
        self.connection_genes.append(gene)
        self._genome = None  # Invalidate cached genome
    
    def remove_layer_gene(self, gene_id: int) -> bool:
        """Remove a layer gene by index.
        
        Args:
            gene_id: Index of the layer gene to remove
            
        Returns:
            True if gene was removed, False otherwise
        """
        if 0 <= gene_id < len(self.layer_genes):
            self.layer_genes.pop(gene_id)
            self._genome = None  # Invalidate cached genome
            return True
        return False
    
    def clone(self) -> "DNA":
        """Create a deep copy of the DNA.
        
        Returns:
            A new DNA instance with copied genes
        """
        import copy
        cloned = DNA(
            input_size=self.input_size,
            output_size=self.output_size,
            layer_genes=copy.deepcopy(self.layer_genes),
            connection_genes=copy.deepcopy(self.connection_genes),
        )
        return cloned
    
    def to_dict(self) -> dict:
        """Serialize DNA to dictionary.
        
        Returns:
            Dictionary representation of DNA
        """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layer_genes": [gene.to_dict() for gene in self.layer_genes],
            "connection_genes": [gene.to_dict() for gene in self.connection_genes],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DNA":
        """Deserialize DNA from dictionary.
        
        Args:
            data: Dictionary representation of DNA
            
        Returns:
            DNA instance restored from dictionary
        """
        layer_genes = [LayerGene.from_dict(g) for g in data.get("layer_genes", [])]
        connection_genes = [ConnectionGene.from_dict(g) for g in data.get("connection_genes", [])]
        
        return cls(
            input_size=data["input_size"],
            output_size=data["output_size"],
            layer_genes=layer_genes,
            connection_genes=connection_genes,
        )


__all__ = ["DNA", "LayerGene", "ConnectionGene"]
