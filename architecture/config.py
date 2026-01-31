"""
NawalConfig - Configuration for Pure Nawal Transformer

This is a sovereign implementation with NO inheritance from HuggingFace or
Microsoft models. All configuration designed specifically for Belizean needs.
"""

from typing import List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class NawalConfig:
    """
    Configuration for Nawal Transformer
    
    Pure sovereign design with NO external dependencies on GPT-2, DialoGPT, or
    any pretrained models. Built from scratch for Belize.
    
    Architecture Sizes:
    - nawal-small:  117M params (768 hidden, 12 layers, 12 heads)
    - nawal-medium: 350M params (1024 hidden, 24 layers, 16 heads)
    - nawal-large:  1.3B params (1536 hidden, 36 layers, 24 heads)
    
    Attributes:
        vocab_size: Vocabulary size including Belizean tokens (52000)
        hidden_size: Hidden dimension size (768/1024/1536)
        num_layers: Number of transformer blocks (12/24/36)
        num_heads: Number of attention heads (12/16/24)
        intermediate_size: FFN intermediate dimension (3072/4096/6144)
        max_position_embeddings: Maximum sequence length (1024/2048)
        dropout: Dropout probability (0.1)
        layer_norm_eps: Layer normalization epsilon (1e-5)
        activation: Activation function ("gelu")
        initializer_range: Weight initialization range (0.02)
        use_cache: Enable KV caching for generation (True)
        pad_token_id: Padding token ID (0)
        bos_token_id: Beginning of sequence token ID (1)
        eos_token_id: End of sequence token ID (2)
        belizean_vocab_extension: Enable Belizean token extension (True)
        multilingual_mode: Enable multi-language support (True)
        supported_languages: List of supported language codes
    """
    
    # Core Architecture
    vocab_size: int = 52000  # Extended for Belizean terms
    hidden_size: int = 768  # nawal-small default
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072  # 4 * hidden_size
    max_position_embeddings: int = 1024
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Activation and Initialization
    activation: str = "gelu"  # "gelu", "relu", "swish"
    initializer_range: float = 0.02
    
    # Generation
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Belizean-Specific
    belizean_vocab_extension: bool = True
    multilingual_mode: bool = True
    supported_languages: List[str] = field(default_factory=lambda: [
        "en",  # English
        "es",  # Spanish
        "bzj", # Belizean Kriol
        "cab", # Garifuna
        "mop"  # Mopan Maya
    ])
    
    # Model Metadata
    model_type: str = "nawal"
    model_size: str = "small"  # "small", "medium", "large"
    version: str = "1.0.0"
    
    @classmethod
    def nawal_small(cls) -> "NawalConfig":
        """117M parameter configuration"""
        return cls(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            model_size="small"
        )
    
    @classmethod
    def nawal_medium(cls) -> "NawalConfig":
        """350M parameter configuration"""
        return cls(
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            intermediate_size=4096,
            max_position_embeddings=2048,
            model_size="medium"
        )
    
    @classmethod
    def nawal_large(cls) -> "NawalConfig":
        """1.3B parameter configuration"""
        return cls(
            hidden_size=1536,
            num_layers=36,
            num_heads=24,
            intermediate_size=6144,
            max_position_embeddings=2048,
            model_size="large"
        )
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load_from_json(cls, path: str) -> "NawalConfig":
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @property
    def num_parameters(self) -> int:
        """Estimate total number of parameters"""
        # Token embeddings
        token_emb = self.vocab_size * self.hidden_size
        # Position embeddings
        pos_emb = self.max_position_embeddings * self.hidden_size
        
        # Transformer blocks
        # Each block: QKV (3 * hidden^2), output proj (hidden^2), 
        # FFN (2 * hidden * intermediate), LayerNorms (4 * hidden)
        per_block = (
            3 * self.hidden_size * self.hidden_size +  # QKV
            self.hidden_size * self.hidden_size +      # Attention output
            2 * self.hidden_size * self.intermediate_size +  # FFN
            4 * self.hidden_size  # LayerNorms
        )
        transformer_params = self.num_layers * per_block
        
        # LM head
        lm_head = self.vocab_size * self.hidden_size
        
        total = token_emb + pos_emb + transformer_params + lm_head
        return total
    
    def __repr__(self) -> str:
        return (
            f"NawalConfig(\n"
            f"  model_size={self.model_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  parameters={self.num_parameters:,}\n"
            f")"
        )
