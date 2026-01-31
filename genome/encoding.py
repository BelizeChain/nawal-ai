"""
Genome Encoding Module

Defines the genetic representation of AI model architectures using modern Python 3.13
features including type hints, Pydantic v2 models, and pattern matching.

This module encodes neural network architectures as "DNA" that can be evolved through
genetic algorithms.
"""

from enum import Enum, auto
from typing import Any, Protocol, TypeAlias
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import hashlib

try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator
except ImportError:
    # Fallback for environments without Pydantic
    from typing import Any as BaseModel
    def Field(*args, **kwargs): return None
    def ConfigDict(*args, **kwargs): return {}
    def field_validator(*args, **kwargs): return lambda f: f

import torch.nn as nn
from typing import Literal


class LayerType(str, Enum):
    """
    Supported neural network layer types for genome encoding.
    
    Includes modern architectures as of October 2025:
    - Traditional: Embedding, Linear, Conv, LSTM, GRU
    - Transformer: MultiheadAttention, TransformerEncoder
    - Modern: MoE (Mixture of Experts), StateSpaceModel (Mamba/S4)
    - Activation: ReLU, GELU, SiLU, Swish
    - Regularization: Dropout, LayerNorm, BatchNorm
    """
    # Embedding & Input
    EMBEDDING = "embedding"
    POSITIONAL_ENCODING = "positional_encoding"
    
    # Core Layers
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    
    # Recurrent
    LSTM = "lstm"
    GRU = "gru"
    
    # Attention & Transformers
    MULTIHEAD_ATTENTION = "multihead_attention"
    TRANSFORMER_ENCODER = "transformer_encoder"
    TRANSFORMER_DECODER = "transformer_decoder"
    CROSS_ATTENTION = "cross_attention"
    
    # Modern Architectures (2024-2025)
    MIXTURE_OF_EXPERTS = "moe"  # MoE for scalability
    STATE_SPACE_MODEL = "ssm"   # Mamba/S4 for long sequences
    ROTARY_EMBEDDING = "rope"   # RoPE for better position encoding
    FLASH_ATTENTION = "flash_attention"  # Efficient attention
    
    # Activation Functions
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWISH = "swish"
    TANH = "tanh"
    
    # Normalization & Regularization
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RMS_NORM = "rms_norm"  # Modern alternative to LayerNorm
    DROPOUT = "dropout"
    
    # Pooling
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    ADAPTIVE_AVG_POOL = "adaptive_avg_pool"


class ArchitectureLayer(BaseModel):
    """
    Represents a single layer in the neural network architecture.
    
    Uses Pydantic v2 for validation and serialization.
    """
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    layer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    layer_type: LayerType
    parameters: dict[str, Any] = Field(default_factory=dict)
    
    # Common parameters
    input_size: int | None = None
    output_size: int | None = None
    
    # Layer-specific parameters
    hidden_size: int | None = None
    num_heads: int | None = None
    num_layers: int | None = None
    dropout_rate: float | None = None
    activation: str | None = None
    
    # MoE-specific
    num_experts: int | None = None
    expert_capacity: int | None = None
    
    # SSM-specific (Mamba/S4)
    state_size: int | None = None
    conv_kernel_size: int | None = None
    
    def __hash__(self) -> int:
        """Make layer hashable for set operations"""
        return hash(self.layer_id)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ArchitectureLayer':
        """Create layer from dictionary"""
        return cls(**data)


class Hyperparameters(BaseModel):
    """
    Training hyperparameters for the genome.
    
    Includes modern optimization techniques and strategies.
    """
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    # Optimization
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    optimizer: Literal["adam", "adamw", "sgd", "lion", "sophia"] = "adamw"  # Lion, Sophia are 2024-2025 optimizers
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    
    # Learning rate schedule
    lr_scheduler: Literal["cosine", "linear", "polynomial", "constant", "warmup_stable_decay"] = "cosine"
    warmup_steps: int = Field(default=1000, ge=0)
    
    # Training configuration
    batch_size: int = Field(default=32, ge=1, le=1024)
    micro_batch_size: int | None = None  # For gradient accumulation
    epochs: int = Field(default=10, ge=1)
    
    # Regularization
    dropout_rate: float = Field(default=0.1, ge=0, le=0.5)
    attention_dropout: float = Field(default=0.1, ge=0, le=0.5)
    
    # Mixed precision training
    use_mixed_precision: bool = True
    precision: Literal["fp32", "fp16", "bf16", "fp8"] = "bf16"  # BF16, FP8 are modern
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = False
    
    # Flash Attention for efficiency (2024 technique)
    use_flash_attention: bool = True
    
    # Compile optimization (PyTorch 2.5+)
    compile_model: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()


class Genome(BaseModel):
    """
    Complete genome representing an AI model architecture.
    
    This is the "DNA" of the AI that evolves over time through genetic algorithms.
    """
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    # Identity
    genome_id: str = Field(default_factory=lambda: f"nawal_{uuid.uuid4().hex[:12]}")
    generation: int = Field(default=0, ge=0)
    parent_genomes: list[str] = Field(default_factory=list)
    
    # Architecture
    encoder_layers: list[ArchitectureLayer] = Field(default_factory=list)
    decoder_layers: list[ArchitectureLayer] = Field(default_factory=list)
    
    # Hyperparameters
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    
    # Fitness tracking
    fitness_score: float | None = Field(default=None, ge=0, le=100)
    quality_score: float | None = Field(default=None, ge=0, le=100)
    timeliness_score: float | None = Field(default=None, ge=0, le=100)
    honesty_score: float | None = Field(default=None, ge=0, le=100)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    trained_by: list[str] = Field(default_factory=list)  # Validator IDs
    training_samples: int = Field(default=0, ge=0)
    
    # Storage references
    ipfs_hash: str | None = None
    arweave_tx: str | None = None
    blockchain_anchor: str | None = None
    
    # Model size estimation (MB)
    estimated_size_mb: float | None = None
    
    def __hash__(self) -> int:
        """Make genome hashable"""
        return hash(self.genome_id)
    
    @property
    def genome_hash(self) -> str:
        """Generate deterministic hash of genome architecture"""
        content = json.dumps({
            "encoder": [layer.to_dict() for layer in self.encoder_layers],
            "decoder": [layer.to_dict() for layer in self.decoder_layers],
            "hyperparameters": self.hyperparameters.to_dict()
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    def total_layers(self) -> int:
        """Total number of layers in architecture"""
        return len(self.encoder_layers) + len(self.decoder_layers)
    
    @property
    def architecture_summary(self) -> str:
        """Human-readable architecture summary"""
        encoder_types = [layer.layer_type.value for layer in self.encoder_layers]
        decoder_types = [layer.layer_type.value for layer in self.decoder_layers]
        return f"Encoder: {encoder_types} | Decoder: {decoder_types}"
    
    # Backward compatibility properties for model_builder
    @property
    def hidden_size(self) -> int:
        """Get hidden size from first encoder layer or default"""
        if self.encoder_layers and self.encoder_layers[0].hidden_size:
            return self.encoder_layers[0].hidden_size
        return 768  # Default GPT-2 size
    
    @property
    def dropout_rate(self) -> float:
        """Get dropout rate from hyperparameters"""
        return self.hyperparameters.dropout_rate
    
    @property
    def output_normalization(self) -> str:
        """Get output normalization type"""
        return LayerType.LAYER_NORM  # Default to LayerNorm
    
    @property
    def tie_word_embeddings(self) -> bool:
        """Whether to tie word embeddings (common in LLMs)"""
        return True  # Default to tying weights
    
    @property
    def dna(self) -> "Genome":
        """
        Backward compatibility property - return self since Genome is the new format.
        In the old API, genomes had a .dna attribute pointing to the DNA object.
        Now Genome IS the data structure, so we return self.
        """
        return self
    
    @property
    def fitness(self) -> float | None:
        """Backward compatibility alias for fitness_score"""
        return self.fitness_score
    
    @fitness.setter
    def fitness(self, value: float) -> None:
        """Backward compatibility setter for fitness_score"""
        self.fitness_score = value
    
    def calculate_fitness(
        self,
        quality: float,
        timeliness: float,
        honesty: float
    ) -> float:
        """
        Calculate overall fitness score aligned with PoUW consensus.
        
        Fitness = 0.40 × Quality + 0.30 × Timeliness + 0.30 × Honesty
        
        Args:
            quality: Model accuracy/performance (0-100)
            timeliness: Training efficiency (0-100)
            honesty: Privacy compliance (0-100)
            
        Returns:
            Overall fitness score (0-100)
        """
        self.quality_score = quality
        self.timeliness_score = timeliness
        self.honesty_score = honesty
        
        self.fitness_score = (
            0.40 * quality +
            0.30 * timeliness +
            0.30 * honesty
        )
        
        return self.fitness_score
    
    def to_dict(self) -> dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        return self.model_dump(exclude_none=True)
    
    def to_json(self, pretty: bool = True) -> str:
        """Convert genome to JSON string"""
        data = self.to_dict()
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Genome':
        """Create genome from dictionary"""
        # Convert layer dictionaries to ArchitectureLayer objects
        if 'encoder_layers' in data:
            data['encoder_layers'] = [
                ArchitectureLayer.from_dict(layer) if isinstance(layer, dict) else layer
                for layer in data['encoder_layers']
            ]
        
        if 'decoder_layers' in data:
            data['decoder_layers'] = [
                ArchitectureLayer.from_dict(layer) if isinstance(layer, dict) else layer
                for layer in data['decoder_layers']
            ]
        
        # Convert hyperparameters dictionary
        if 'hyperparameters' in data and isinstance(data['hyperparameters'], dict):
            data['hyperparameters'] = Hyperparameters(**data['hyperparameters'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Genome':
        """Create genome from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def clone(self) -> 'Genome':
        """Create a deep copy of the genome"""
        data = self.to_dict()
        data['genome_id'] = f"nawal_{uuid.uuid4().hex[:12]}"  # New ID
        data['parent_genomes'] = [self.genome_id]  # Track lineage
        return self.from_dict(data)


class GenomeEncoder:
    """
    Utility class for encoding/decoding genomes and architectures.
    
    Provides methods for:
    - Creating genomes from model definitions
    - Converting genomes to PyTorch nn.Module
    - Validating genome structures
    - Estimating model sizes
    """
    
    @staticmethod
    def create_baseline_genome() -> Genome:
        """
        Create a baseline genome with proven architecture.
        
        Uses a modern transformer architecture with MoE and optimizations.
        """
        genome = Genome(
            genome_id="nawal_baseline_v1",
            generation=0,
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.EMBEDDING,
                    input_size=50000,  # Vocab size
                    output_size=768,
                    parameters={"padding_idx": 0}
                ),
                ArchitectureLayer(
                    layer_type=LayerType.POSITIONAL_ENCODING,
                    hidden_size=768,
                    parameters={"max_seq_len": 2048}
                ),
                ArchitectureLayer(
                    layer_type=LayerType.TRANSFORMER_ENCODER,
                    hidden_size=768,
                    num_heads=12,
                    num_layers=6,
                    dropout_rate=0.1,
                    parameters={
                        "use_flash_attention": True,
                        "use_rope": True  # Rotary Position Embedding
                    }
                ),
                ArchitectureLayer(
                    layer_type=LayerType.MIXTURE_OF_EXPERTS,
                    hidden_size=768,
                    num_experts=8,
                    expert_capacity=64,
                    parameters={
                        "top_k": 2,  # Top-2 routing
                        "load_balancing": True
                    }
                ),
            ],
            decoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.LINEAR,
                    input_size=768,
                    output_size=512,
                ),
                ArchitectureLayer(
                    layer_type=LayerType.GELU,
                    parameters={"approximate": "tanh"}
                ),
                ArchitectureLayer(
                    layer_type=LayerType.DROPOUT,
                    dropout_rate=0.1
                ),
                ArchitectureLayer(
                    layer_type=LayerType.LINEAR,
                    input_size=512,
                    output_size=50000,  # Vocab size
                ),
            ],
            hyperparameters=Hyperparameters(
                learning_rate=5e-5,
                optimizer="adamw",
                batch_size=32,
                use_mixed_precision=True,
                precision="bf16",
                use_flash_attention=True,
                compile_model=True
            )
        )
        
        return genome
    
    @staticmethod
    def estimate_model_size(genome: Genome) -> float:
        """
        Estimate model size in MB based on architecture.
        
        This is a rough estimation for planning purposes.
        """
        total_params = 0
        
        for layer in genome.encoder_layers + genome.decoder_layers:
            match layer.layer_type:
                case LayerType.EMBEDDING:
                    params = (layer.input_size or 0) * (layer.output_size or 0)
                case LayerType.LINEAR:
                    params = (layer.input_size or 0) * (layer.output_size or 0)
                case LayerType.TRANSFORMER_ENCODER | LayerType.TRANSFORMER_DECODER:
                    # Rough estimate: 4 * hidden_size^2 per layer
                    hidden = layer.hidden_size or 768
                    num_layers = layer.num_layers or 1
                    params = 4 * (hidden ** 2) * num_layers
                case LayerType.MIXTURE_OF_EXPERTS:
                    # MoE: num_experts * expert_size
                    hidden = layer.hidden_size or 768
                    num_experts = layer.num_experts or 8
                    params = num_experts * (hidden ** 2)
                case _:
                    params = 0
            
            total_params += params
        
        # Convert to MB (assuming fp32 = 4 bytes, bf16 = 2 bytes)
        bytes_per_param = 2 if genome.hyperparameters.precision == "bf16" else 4
        size_mb = (total_params * bytes_per_param) / (1024 ** 2)
        
        genome.estimated_size_mb = size_mb
        return size_mb
    
    @staticmethod
    def validate_genome(genome: Genome) -> tuple[bool, list[str]]:
        """
        Validate genome structure and parameters.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check encoder layers
        if not genome.encoder_layers:
            errors.append("Genome must have at least one encoder layer")
        
        # Check decoder layers
        if not genome.decoder_layers:
            errors.append("Genome must have at least one decoder layer")
        
        # Check layer compatibility
        for i, layer in enumerate(genome.encoder_layers[:-1]):
            next_layer = genome.encoder_layers[i + 1]
            if layer.output_size and next_layer.input_size:
                if layer.output_size != next_layer.input_size:
                    errors.append(
                        f"Layer {i} output size ({layer.output_size}) doesn't match "
                        f"next layer input size ({next_layer.input_size})"
                    )
        
        # Check hyperparameters
        if genome.hyperparameters.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if genome.hyperparameters.batch_size < 1:
            errors.append("Batch size must be at least 1")
        
        return len(errors) == 0, errors


# Protocol for genome compatibility
class GenomeCompatible(Protocol):
    """Protocol for objects that can work with genomes"""
    
    def encode_genome(self, model: nn.Module) -> Genome:
        """Encode a PyTorch model as a genome"""
        ...
    
    def decode_genome(self, genome: Genome) -> nn.Module:
        """Decode a genome into a PyTorch model"""
        ...


if __name__ == "__main__":
    # Example usage
    print("Creating baseline genome...")
    genome = GenomeEncoder.create_baseline_genome()
    
    print(f"\nGenome ID: {genome.genome_id}")
    print(f"Generation: {genome.generation}")
    print(f"Total layers: {genome.total_layers}")
    print(f"Architecture: {genome.architecture_summary}")
    
    # Estimate size
    size_mb = GenomeEncoder.estimate_model_size(genome)
    print(f"\nEstimated model size: {size_mb:.2f} MB")
    
    # Validate
    is_valid, errors = GenomeEncoder.validate_genome(genome)
    print(f"\nGenome valid: {is_valid}")
    if errors:
        print("Errors:", errors)
    
    # Test serialization
    json_str = genome.to_json()
    print(f"\nJSON serialization: {len(json_str)} characters")
    
    # Test deserialization
    genome2 = Genome.from_json(json_str)
    print(f"Deserialized genome: {genome2.genome_id}")
    
    print("\n✅ Genome encoding module test successful!")
