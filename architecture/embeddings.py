"""
Nawal Embeddings - Token and Position Embeddings

Pure PyTorch implementation with NO dependencies on HuggingFace or pretrained models.
All embeddings initialized randomly for sovereign training.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .config import NawalConfig


class NawalEmbeddings(nn.Module):
    """
    Combined token and position embeddings for Nawal Transformer
    
    Pure sovereign implementation:
    - Random initialization (NO pretrained weights)
    - Belizean vocabulary (52K tokens)
    - Learned positional embeddings (up to 2048 positions)
    - Dropout for regularization
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (vocab_size -> hidden_size)
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings (max_positions -> hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform"""
        nn.init.normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=self.config.initializer_range
        )
        # Keep padding token embedding at zero
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.config.pad_token_id].fill_(0)
        
        nn.init.normal_(
            self.position_embeddings.weight,
            mean=0.0,
            std=self.config.initializer_range
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to create input embeddings
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)
        
        Returns:
            embeddings: Combined embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Get position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_token_embeddings(self) -> nn.Embedding:
        """Get token embedding layer (useful for weight tying with LM head)"""
        return self.token_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Alternative: Sinusoidal (fixed) positional embeddings
    
    Based on "Attention Is All You Need" (Vaswani et al., 2017)
    Can be used instead of learned embeddings for better length extrapolation
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "positional_encoding",
            self._create_sinusoidal_embeddings(
                config.max_position_embeddings,
                config.hidden_size
            )
        )
    
    def _create_sinusoidal_embeddings(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional embeddings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get sinusoidal positional embeddings
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs (optional, not used)
        
        Returns:
            embeddings: Positional embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.size()
        return self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
