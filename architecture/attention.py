"""
Multi-Head Attention - Core attention mechanism for Nawal Transformer

Pure PyTorch implementation with NO dependencies on HuggingFace or external models.
Implements scaled dot-product attention with multi-head parallel processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .config import NawalConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention
    
    Pure sovereign implementation based on "Attention Is All You Need" (Vaswani et al., 2017)
    NO dependencies on GPT-2, DialoGPT, or any pretrained models.
    
    Features:
    - Parallel attention heads for diverse representation learning
    - Scaled dot-product for numerical stability
    - Causal masking for autoregressive generation
    - KV caching for efficient inference
    - Flash attention compatible structure
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Scaling factor for dot product
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into multiple attention heads
        
        Args:
            x: [batch_size, seq_len, hidden_size]
        
        Returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge attention heads back into hidden dimension
        
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        return x.view(batch_size, seq_len, self.hidden_size)
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create causal attention mask for autoregressive generation
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            dtype: Data type for mask
        
        Returns:
            Causal mask [seq_len, seq_len] with -inf for future positions
        """
        # Create lower triangular matrix (1 for allowed positions, 0 for masked)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        # Convert 0s to -inf for masking in softmax
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention
        
        Args:
            hidden_states: Input [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len]
            past_key_value: Cached (key, value) for efficient generation
            use_cache: Whether to return cached key/value
            is_causal: Whether to apply causal masking
        
        Returns:
            output: Attention output [batch_size, seq_len, hidden_size]
            new_past_key_value: Updated (key, value) cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Split into multiple heads
        query = self._split_heads(query)  # [batch, num_heads, seq_len, head_dim]
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Use cached key/value if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)  # Concat along seq_len
            value = torch.cat([past_value, value], dim=2)
        
        # Cache key/value if requested
        new_past_key_value = (key, value) if use_cache else None
        
        # Compute scaled dot-product attention
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, kv_len]
        # -> [batch, num_heads, seq_len, kv_len]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        kv_len = key.size(2)
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = self._create_causal_mask(
                seq_len, hidden_states.device, attn_weights.dtype
            )
            # If using cache, only mask the new positions
            if past_key_value is not None:
                # Extend causal mask for cached positions
                causal_mask = F.pad(causal_mask, (kv_len - seq_len, 0), value=0.0)
            attn_weights = attn_weights + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to [batch, 1, seq_len, kv_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + attention_mask
        
        # Softmax to get attention probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        # [batch, num_heads, seq_len, kv_len] @ [batch, num_heads, kv_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_probs, value)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)  # [batch, seq_len, hidden_size]
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, new_past_key_value
