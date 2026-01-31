"""
Hybrid Quantum-Classical LLM

Combines Nawal's classical transformer with Kinich's quantum processing.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - HybridLLM will not work")


class HybridQuantumClassicalLLM(nn.Module if TORCH_AVAILABLE else object):
    """
    Hybrid Language Model combining classical and quantum processing.
    
    Architecture:
    1. Classical transformer encoder (Nawal BelizeChainLLM)
    2. Quantum feature enhancement (Kinich QNN)
    3. Classical decoder for generation
    
    The quantum layer enhances intermediate representations,
    potentially capturing non-classical correlations.
    
    Example:
        >>> model = HybridQuantumClassicalLLM(
        ...     vocab_size=10000,
        ...     hidden_dim=768,
        ...     quantum_dim=8,
        ...     num_layers=6
        ... )
        >>> 
        >>> input_ids = torch.randint(0, 10000, (4, 128))  # [batch, seq_len]
        >>> outputs = model(input_ids)
        >>> logits = outputs['logits']  # [batch, seq_len, vocab_size]
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        quantum_dim: int = 8,
        num_layers: int = 6,
        num_heads: int = 12,
        ff_dim: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        enable_quantum: bool = True,
        quantum_position: str = "middle"
    ):
        """
        Initialize Hybrid Quantum-Classical LLM.
        
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension (must match quantum_dim for compatibility)
            quantum_dim: Quantum feature dimension (number of qubits)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            enable_quantum: Enable quantum processing
            quantum_position: Where to insert quantum layer ('early', 'middle', 'late')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HybridQuantumClassicalLLM")
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.quantum_dim = quantum_dim
        self.num_layers = num_layers
        self.enable_quantum = enable_quantum
        self.quantum_position = quantum_position
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)
        
        # Classical transformer layers
        self.layers_before_quantum = nn.ModuleList()
        self.layers_after_quantum = nn.ModuleList()
        
        # Determine quantum insertion point
        quantum_layer_idx = self._get_quantum_layer_idx()
        
        for i in range(num_layers):
            layer = TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            
            if i < quantum_layer_idx:
                self.layers_before_quantum.append(layer)
            else:
                self.layers_after_quantum.append(layer)
        
        # Quantum enhancement layer
        if enable_quantum:
            from nawal.integration.kinich_connector import QuantumEnhancedLayer
            self.quantum_layer = QuantumEnhancedLayer(
                classical_dim=hidden_dim,
                quantum_dim=quantum_dim,
                model_type="qnn"
            )
        else:
            self.quantum_layer = None
        
        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized HybridQuantumClassicalLLM: "
            f"vocab={vocab_size}, hidden={hidden_dim}, "
            f"quantum={quantum_dim}, layers={num_layers}, "
            f"quantum_enabled={enable_quantum}"
        )
    
    def _get_quantum_layer_idx(self) -> int:
        """Determine where to insert quantum layer."""
        if self.quantum_position == "early":
            return self.num_layers // 4
        elif self.quantum_position == "middle":
            return self.num_layers // 2
        elif self.quantum_position == "late":
            return (3 * self.num_layers) // 4
        else:
            return self.num_layers // 2
    
    def forward(
        self,
        input_ids: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None,
        return_intermediate: bool = False
    ) -> Dict[str, 'torch.Tensor']:
        """
        Forward pass through hybrid model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            return_intermediate: Return intermediate representations
            
        Returns:
            Dictionary with:
                - logits: Output logits [batch_size, seq_length, vocab_size]
                - hidden_states: Final hidden states (if return_intermediate)
                - quantum_enhanced: Quantum-enhanced features (if return_intermediate)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(
            seq_length, 
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Classical transformer layers (before quantum)
        for layer in self.layers_before_quantum:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Quantum enhancement
        quantum_enhanced = None
        if self.enable_quantum and self.quantum_layer is not None:
            # Process each token position through quantum layer
            # Reshape: [batch, seq, hidden] → [batch*seq, hidden]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)
            
            # Quantum processing
            quantum_flat = self.quantum_layer(hidden_flat)
            
            # Reshape back: [batch*seq, hidden] → [batch, seq, hidden]
            quantum_enhanced = quantum_flat.reshape(batch_size, seq_len, hidden_dim)
            
            # Residual connection
            hidden_states = hidden_states + quantum_enhanced
        
        # Classical transformer layers (after quantum)
        for layer in self.layers_after_quantum:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Output projection
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        # Prepare outputs
        outputs = {'logits': logits}
        
        if return_intermediate:
            outputs['hidden_states'] = hidden_states
            if quantum_enhanced is not None:
                outputs['quantum_enhanced'] = quantum_enhanced
        
        return outputs
    
    def generate(
        self,
        input_ids: 'torch.Tensor',
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> 'torch.Tensor':
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs [batch_size, prompt_length]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs [batch_size, total_length]
        """
        self.eval()
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_quantum_statistics(self) -> Dict[str, Any]:
        """Get statistics from quantum layer."""
        if self.quantum_layer is not None:
            return self.quantum_layer.connector.get_statistics()
        return {}


class TransformerLayer(nn.Module):
    """Standard transformer layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: 'torch.Tensor',
        attention_mask: Optional['torch.Tensor'] = None
    ) -> 'torch.Tensor':
        # Self-attention with residual
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x
