"""
Nawal Transformer - Pure Sovereign Language Model

100% built from scratch with NO pretrained weights, NO GPT-2, NO DialoGPT.
Complete transformer implementation for Belizean AI sovereignty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

from .config import NawalConfig
from .embeddings import NawalEmbeddings
from .attention import MultiHeadAttention
from .feedforward import FeedForward

logger = logging.getLogger(__name__)


class NawalTransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> residual -> LayerNorm -> FFN -> residual
    
    Uses pre-normalization (LayerNorm before attention/FFN) for training stability.
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        # Layer normalization (pre-norm)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(config)
        
        # Position-wise feed-forward
        self.ffn = FeedForward(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block
        
        Args:
            hidden_states: Input [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Cached key/value from previous step
            use_cache: Whether to return key/value cache
            is_causal: Whether to use causal masking
        
        Returns:
            output: Block output [batch_size, seq_len, hidden_size]
            new_past_key_value: Updated cache if use_cache=True
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, new_past_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            is_causal=is_causal,
        )
        hidden_states = residual + attn_output
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states, new_past_key_value


class NawalTransformer(nn.Module):
    """
    Pure Nawal Transformer - Sovereign Language Model for Belize
    
    100% sovereign implementation:
    - NO Microsoft DialoGPT or GPT-2 dependencies
    - NO pretrained weights from external sources
    - Random initialization for sovereign training
    - Built from scratch using pure PyTorch
    
    Features:
    - Multi-layer transformer architecture (12/24/36 layers)
    - Multi-head self-attention (12/16/24 heads)
    - Position-wise feed-forward networks
    - Causal language modeling objective
    - KV caching for efficient generation
    - Multilingual support (5 Belizean languages)
    
    Model Sizes:
    - nawal-small:  117M params (768 hidden, 12 layers)
    - nawal-medium: 350M params (1024 hidden, 24 layers)  
    - nawal-large:  1.3B params (1536 hidden, 36 layers)
    """
    
    def __init__(self, config: NawalConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.embeddings = NawalEmbeddings(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NawalTransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings and LM head weights (weight sharing)
        self.lm_head.weight = self.embeddings.get_token_embeddings().weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialized Nawal Transformer:\n"
            f"  Model size: {config.model_size}\n"
            f"  Layers: {config.num_layers}\n"
            f"  Hidden size: {config.hidden_size}\n"
            f"  Attention heads: {config.num_heads}\n"
            f"  Total parameters: {num_params:,}\n"
            f"  Sovereignty: 100% (NO pretrained weights)"
        )
    
    def _init_weights(self, module):
        """Initialize weights for all modules"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> dict:
        """
        Forward pass through Nawal Transformer
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)
            past_key_values: List of cached (key, value) tuples for each layer
            labels: Target labels [batch_size, seq_len] for language modeling
            use_cache: Whether to return cached key/values
            return_dict: Whether to return dict or tuple
        
        Returns:
            Dictionary containing:
                - loss: Language modeling loss (if labels provided)
                - logits: Predicted token logits [batch_size, seq_len, vocab_size]
                - past_key_values: Cached key/values (if use_cache=True)
                - hidden_states: Final hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        
        # Initialize cache if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        
        # Pass through transformer blocks
        new_past_key_values = []
        for i, (block, past_kv) in enumerate(zip(self.blocks, past_key_values)):
            hidden_states, new_past_kv = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                is_causal=True,
            )
            if use_cache:
                new_past_key_values.append(new_past_kv)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
        
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (new_past_key_values,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_past_key_values if use_cache else None,
            "hidden_states": hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (True) or use greedy (False)
        
        Returns:
            generated: Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        with torch.no_grad():
            past_key_values = None
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                # Get next token logits
                next_token_logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                if do_sample:
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability > top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token generated
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return input_ids
    
    @classmethod
    def from_config(cls, config: NawalConfig) -> "NawalTransformer":
        """Create model from configuration"""
        return cls(config)
    
    @classmethod
    def nawal_small(cls) -> "NawalTransformer":
        """Create small model (117M parameters)"""
        config = NawalConfig.nawal_small()
        return cls(config)
    
    @classmethod
    def nawal_medium(cls) -> "NawalTransformer":
        """Create medium model (350M parameters)"""
        config = NawalConfig.nawal_medium()
        return cls(config)
    
    @classmethod
    def nawal_large(cls) -> "NawalTransformer":
        """Create large model (1.3B parameters)"""
        config = NawalConfig.nawal_large()
        return cls(config)
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and configuration"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        self.config.save_to_json(config_path)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def load_pretrained(cls, load_directory: str) -> "NawalTransformer":
        """Load model weights and configuration"""
        import os
        
        # Load config
        config_path = os.path.join(load_directory, "config.json")
        config = NawalConfig.load_from_json(config_path)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {load_directory}")
        return model
