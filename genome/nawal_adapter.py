"""
Genome-to-Nawal Bridge - Connect genome evolution to pure Nawal architecture

This module bridges the genome evolution system with the new pure Nawal transformer,
allowing evolved architectures to be instantiated as NawalTransformer models.

Integration:
    Genome DNA → NawalConfig → NawalTransformer (pure sovereign)
    
Instead of generic ModelBuilder, we now use the Nawal-specific architecture.
"""

import torch
from typing import Optional
import logging

from nawal.architecture import NawalConfig, NawalTransformer
from nawal.genome.encoding import Genome, LayerType

logger = logging.getLogger(__name__)


class GenomeToNawalAdapter:
    """
    Adapt genome DNA to NawalConfig specifications
    
    Converts genome architecture layers into NawalConfig parameters:
    - encoder_layers → num_layers
    - hidden_size from genome → hidden_size
    - attention heads from genome → num_heads
    - activation type from genome → activation
    """
    
    def __init__(self):
        logger.info("Initialized GenomeToNawalAdapter")
    
    def genome_to_config(self, genome: Genome) -> NawalConfig:
        """
        Convert Genome to NawalConfig
        
        Args:
            genome: Genome with architecture DNA
        
        Returns:
            NawalConfig configured from genome specifications
        """
        # Extract transformer layers from genome
        transformer_layers = [
            layer for layer in genome.encoder_layers
            if layer.layer_type in [
                LayerType.TRANSFORMER_ENCODER,
                LayerType.MULTIHEAD_ATTENTION,
            ]
        ]
        
        # Determine architecture size from genome
        if transformer_layers:
            first_layer = transformer_layers[0]
            hidden_size = first_layer.hidden_size or 768
            num_heads = first_layer.num_heads or 12
            num_layers = len(transformer_layers)
        else:
            # Default to small if no transformer layers specified
            logger.warning("No transformer layers in genome, using small defaults")
            hidden_size = 768
            num_heads = 12
            num_layers = 12
        
        # Extract activation from genome hyperparameters
        activation_map = {
            "gelu": "gelu",
            "relu": "relu",
            "silu": "swish",
            "swish": "swish",
        }
        activation = activation_map.get(
            genome.hyperparameters.get("activation", "gelu").lower(),
            "gelu"
        )
        
        # Extract dropout
        dropout = genome.hyperparameters.get("dropout_rate", 0.1)
        
        # Create NawalConfig from genome specifications
        config = NawalConfig(
            vocab_size=52000,  # Belizean extended vocab
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=hidden_size * 4,  # Standard 4x expansion
            max_position_embeddings=1024,
            dropout=dropout,
            attention_dropout=dropout,
            activation=activation,
            # Preserve genome metadata
            model_type="nawal",
            version=f"genome-{genome.genome_id[:8]}",
        )
        
        logger.info(
            f"Converted genome {genome.genome_id[:8]} to NawalConfig:\n"
            f"  Layers: {num_layers}\n"
            f"  Hidden: {hidden_size}\n"
            f"  Heads: {num_heads}\n"
            f"  Activation: {activation}\n"
            f"  Parameters: {config.num_parameters:,}"
        )
        
        return config
    
    def build_model(self, genome: Genome) -> NawalTransformer:
        """
        Build NawalTransformer from genome DNA
        
        Args:
            genome: Genome with architecture specifications
        
        Returns:
            Initialized NawalTransformer with random weights
        """
        config = self.genome_to_config(genome)
        model = NawalTransformer(config)
        
        logger.info(f"Built NawalTransformer from genome {genome.genome_id[:8]}")
        
        return model
    
    def estimate_flops(self, genome: Genome, seq_len: int = 512) -> int:
        """
        Estimate FLOPs for genome architecture
        
        Args:
            genome: Genome to estimate
            seq_len: Sequence length for estimation
        
        Returns:
            Estimated FLOPs per forward pass
        """
        config = self.genome_to_config(genome)
        
        # Simplified FLOPs estimation for transformer
        # Attention: 4 * hidden^2 * seq_len * num_layers
        # FFN: 8 * hidden^2 * seq_len * num_layers
        attention_flops = (
            4 * config.hidden_size * config.hidden_size * 
            seq_len * config.num_layers
        )
        ffn_flops = (
            8 * config.hidden_size * config.intermediate_size * 
            seq_len * config.num_layers
        )
        
        total_flops = attention_flops + ffn_flops
        
        return total_flops
    
    def estimate_memory(self, genome: Genome) -> dict:
        """
        Estimate memory requirements
        
        Args:
            genome: Genome to estimate
        
        Returns:
            Dictionary with memory estimates in bytes
        """
        config = self.genome_to_config(genome)
        
        # Parameters memory (FP32)
        params_memory = config.num_parameters * 4  # 4 bytes per FP32
        
        # Gradients memory (same as params in FP32)
        gradients_memory = params_memory
        
        # Optimizer states (AdamW uses 2x params)
        optimizer_memory = params_memory * 2
        
        # Activations (approximate, depends on batch size)
        # For batch_size=8, seq_len=512
        activations_memory = (
            8 * 512 * config.hidden_size * config.num_layers * 4
        )
        
        return {
            "parameters_bytes": params_memory,
            "parameters_mb": params_memory / (1024 ** 2),
            "gradients_bytes": gradients_memory,
            "optimizer_bytes": optimizer_memory,
            "activations_bytes": activations_memory,
            "total_training_bytes": (
                params_memory + gradients_memory + 
                optimizer_memory + activations_memory
            ),
            "total_training_gb": (
                params_memory + gradients_memory + 
                optimizer_memory + activations_memory
            ) / (1024 ** 3),
        }


class NawalGenomeBuilder:
    """
    High-level builder for creating Nawal models from genomes
    
    This replaces the generic ModelBuilder for Nawal-specific use cases.
    """
    
    def __init__(self):
        self.adapter = GenomeToNawalAdapter()
        logger.info("Initialized NawalGenomeBuilder")
    
    def build_from_genome(
        self,
        genome: Genome,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> NawalTransformer:
        """
        Build and initialize Nawal model from genome
        
        Args:
            genome: Genome with architecture DNA
            device: Device to place model on ("cuda", "cpu")
            dtype: Data type (torch.float32, torch.float16, torch.bfloat16)
        
        Returns:
            NawalTransformer ready for training
        """
        # Build model
        model = self.adapter.build_model(genome)
        
        # Move to device if specified
        if device:
            model = model.to(device)
        
        # Convert dtype if specified
        if dtype:
            model = model.to(dtype)
        
        # Log memory requirements
        memory_stats = self.adapter.estimate_memory(genome)
        logger.info(
            f"Model memory requirements:\n"
            f"  Parameters: {memory_stats['parameters_mb']:.1f} MB\n"
            f"  Training total: {memory_stats['total_training_gb']:.2f} GB"
        )
        
        return model
    
    def get_genome_fitness_score(
        self,
        genome: Genome,
        validation_loss: float,
        training_time: float,
        privacy_epsilon: float,
    ) -> float:
        """
        Calculate PoUW fitness score for genome
        
        Aligned with BelizeChain's Proof of Useful Work:
        - Quality (40%): Lower validation loss = higher score
        - Timeliness (30%): Faster training = higher score  
        - Honesty (30%): Better privacy (lower epsilon) = higher score
        
        Args:
            genome: Genome being evaluated
            validation_loss: Validation loss after training
            training_time: Training time in seconds
            privacy_epsilon: Privacy budget used (lower = better)
        
        Returns:
            Fitness score (0-1, higher is better)
        """
        # Quality score (inverse of loss, normalized)
        quality_score = 1.0 / (1.0 + validation_loss)
        
        # Timeliness score (inverse of time, normalized to 0-1)
        # Assume target time is 3600s (1 hour)
        target_time = 3600
        timeliness_score = min(1.0, target_time / max(training_time, 1.0))
        
        # Honesty/Privacy score (inverse of epsilon, normalized)
        # Lower epsilon = better privacy = higher score
        # Target epsilon is 1.0
        privacy_score = 1.0 / (1.0 + privacy_epsilon)
        
        # PoUW weighted combination: 40% quality, 30% timeliness, 30% honesty
        fitness = (
            0.4 * quality_score +
            0.3 * timeliness_score +
            0.3 * privacy_score
        )
        
        logger.info(
            f"Genome {genome.genome_id[:8]} fitness:\n"
            f"  Quality: {quality_score:.3f} (40%)\n"
            f"  Timeliness: {timeliness_score:.3f} (30%)\n"
            f"  Privacy: {privacy_score:.3f} (30%)\n"
            f"  Overall: {fitness:.3f}"
        )
        
        return fitness


# Convenience functions
def genome_to_nawal(genome: Genome) -> NawalTransformer:
    """
    Quick conversion from genome to Nawal model
    
    Args:
        genome: Genome DNA
    
    Returns:
        NawalTransformer instance
    """
    builder = NawalGenomeBuilder()
    return builder.build_from_genome(genome)


def create_baseline_nawal_genome() -> Genome:
    """
    Create a baseline genome for Nawal-small (117M params)
    
    Returns:
        Genome configured for nawal-small architecture
    """
    from nawal.genome.encoding import GenomeEncoder, ArchitectureLayer
    
    encoder = GenomeEncoder()
    genome = encoder.create_baseline_genome()
    
    # Override with Nawal-small specifications
    genome.encoder_layers = [
        ArchitectureLayer(
            layer_type=LayerType.TRANSFORMER_ENCODER,
            hidden_size=768,
            num_heads=12,
            num_layers=12,
            dropout_rate=0.1,
            activation="gelu",
        )
    ]
    
    genome.hyperparameters["hidden_size"] = 768
    genome.hyperparameters["num_layers"] = 12
    genome.hyperparameters["num_heads"] = 12
    genome.hyperparameters["activation"] = "gelu"
    
    return genome
