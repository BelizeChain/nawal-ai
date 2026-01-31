"""
Feed-Forward Network - Position-wise FFN for Nawal Transformer

Pure PyTorch implementation with NO dependencies on external models.
Two-layer MLP with GELU activation applied to each position independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NawalConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Pure sovereign implementation:
    - Two-layer MLP: hidden_size -> intermediate_size -> hidden_size
    - GELU activation (Gaussian Error Linear Unit)
    - Dropout for regularization
    - Applied independently to each position in sequence
    
    Architecture:
        x -> Linear(hidden -> intermediate) -> GELU -> Dropout 
          -> Linear(intermediate -> hidden) -> Dropout -> output
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        # First linear layer (expansion)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        
        # Second linear layer (compression)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        
        # Activation function
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "swish" or config.activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution"""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=self.config.initializer_range)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network
        
        Args:
            hidden_states: Input [batch_size, seq_len, hidden_size]
        
        Returns:
            output: Transformed features [batch_size, seq_len, hidden_size]
        """
        # Expansion
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Compression
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network (GLU variant)
    
    Alternative FFN architecture using gating mechanism for better expressiveness.
    Used in models like LLaMA and PaLM.
    
    Architecture:
        x -> [Linear(hidden -> intermediate), Linear(hidden -> intermediate)]
          -> gate * activation(value) -> Linear(intermediate -> hidden)
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        # Gate and value projections
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        # Down projection
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # Activation
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "silu" or config.activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.gate_proj, self.value_proj, self.down_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gating mechanism
        
        Args:
            hidden_states: Input [batch_size, seq_len, hidden_size]
        
        Returns:
            output: Gated features [batch_size, seq_len, hidden_size]
        """
        gate = self.gate_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # Apply gating: gate * activation(value)
        hidden_states = self.activation(gate) * value
        hidden_states = self.dropout(hidden_states)
        
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
