"""
Complete Genome Training Example

Demonstrates end-to-end workflow:
1. Create genome
2. Build PyTorch model
3. Train locally
4. Submit to federated server

Author: BelizeChain AI Team
Date: October 2025
"""

import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset

# Nawal imports
from nawal.genome import GenomeEncoder, ModelBuilder
from nawal.client import GenomeTrainer, TrainingConfig
from nawal.server import FederatedAggregator, AggregationStrategy


async def main():
    print("=" * 80)
    print("NAWAL COMPLETE TRAINING EXAMPLE")
    print("=" * 80)
    print()
    
    # ==========================================================================
    # STEP 1: Create Genome
    # ==========================================================================
    print("📦 Step 1: Creating genome...")
    encoder = GenomeEncoder()
    genome = encoder.create_baseline_genome()
    
    print(f"   Genome ID: {genome.genome_id}")
    print(f"   Hidden Size: {genome.hidden_size}")
    print(f"   Num Layers: {len(genome.encoder_layers)}")
    print(f"   Architecture: Transformer")
    print()
    
    # ==========================================================================
    # STEP 2: Build PyTorch Model
    # ==========================================================================
    print("🔨 Step 2: Building PyTorch model...")
    builder = ModelBuilder(vocab_size=50257, max_seq_length=512)
    
    # Validate genome
    errors = builder.validate_genome(genome)
    if errors:
        print(f"   ⚠️  Validation warnings: {errors}")
    else:
        print("   ✅ Genome validated successfully!")
    
    # Build model
    model = builder.build_model(genome)
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Memory: {model.get_memory_footprint() / 1024**2:.1f} MB")
    
    # Estimate resources
    flops = builder.estimate_flops(genome, seq_length=512)
    memory = builder.estimate_memory(genome, batch_size=8)
    
    print(f"   FLOPs: {flops:,}")
    print(f"   Total Memory (FP32): {memory['total_mb_fp32']:.1f} MB")
    print(f"   Total Memory (FP16): {memory['total_mb_fp16']:.1f} MB")
    print()
    
    # ==========================================================================
    # STEP 3: Setup Federated Server
    # ==========================================================================
    print("🌐 Step 3: Setting up federated server...")
    aggregator = FederatedAggregator(
        genome=genome,
        strategy=AggregationStrategy.FEDAVG,
        min_participants=1,  # Allow 1 for demo
        aggregation_threshold=0.8,
    )
    
    await aggregator.start_round(1)
    print(f"   Server started: Round {aggregator.current_round}")
    print(f"   Strategy: {aggregator.strategy.value}")
    print()
    
    # ==========================================================================
    # STEP 4: Setup Training Client
    # ==========================================================================
    print("🎓 Step 4: Setting up training client...")
    config = TrainingConfig(
        participant_id="validator-001",
        validator_address="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        staking_account="belizechain:validator001",
        learning_rate=1e-4,
        batch_size=16,
        local_epochs=2,
        gradient_clipping=True,
        mixed_precision=False,  # Set to False for CPU demo
        device="cpu",  # Use CPU for demo
    )
    
    trainer = GenomeTrainer(config=config)
    
    print(f"   Participant: {config.participant_id}")
    print(f"   Device: {trainer.device}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Local Epochs: {config.local_epochs}")
    print()
    
    # ==========================================================================
    # STEP 5: Create Dummy Training Data
    # ==========================================================================
    print("📊 Step 5: Creating training data...")
    
    # Create dummy data (random token IDs)
    num_train_samples = 100
    num_val_samples = 20
    seq_length = 128
    vocab_size = 50257
    
    # Training data
    train_input_ids = torch.randint(0, vocab_size, (num_train_samples, seq_length))
    train_dataset = TensorDataset(train_input_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # Validation data
    val_input_ids = torch.randint(0, vocab_size, (num_val_samples, seq_length))
    val_dataset = TensorDataset(val_input_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    # Convert to proper format
    def collate_fn(batch):
        input_ids = batch[0]
        return {
            "input_ids": input_ids,
            "labels": input_ids,  # Language modeling: predict next token
        }
    
    train_loader.collate_fn = collate_fn
    val_loader.collate_fn = collate_fn
    
    print(f"   Training samples: {num_train_samples}")
    print(f"   Validation samples: {num_val_samples}")
    print(f"   Sequence length: {seq_length}")
    print()
    
    # ==========================================================================
    # STEP 6: Train Genome
    # ==========================================================================
    print("🚀 Step 6: Training genome locally...")
    print()
    
    # Get global weights from server (first round, so it's just the initial model)
    global_weights = aggregator.global_weights
    
    # Train
    updated_weights, metrics = await trainer.train_genome(
        genome=genome,
        train_loader=train_loader,
        val_loader=val_loader,
        global_weights=global_weights,
        round_number=aggregator.current_round,
    )
    
    print()
    print("   Training complete!")
    print(f"   Train Loss: {metrics.train_loss:.4f}")
    print(f"   Val Loss: {metrics.val_loss:.4f}" if metrics.val_loss else "   Val Loss: N/A")
    print(f"   Samples: {metrics.samples_trained}")
    print(f"   Time: {metrics.training_time:.1f}s")
    print(f"   Throughput: {metrics.throughput:.1f} samples/s")
    print()
    print("   Fitness Scores:")
    print(f"   - Quality: {metrics.quality_score:.2f}/100")
    print(f"   - Timeliness: {metrics.timeliness_score:.2f}/100")
    print(f"   - Honesty: {metrics.honesty_score:.2f}/100")
    print(f"   - Overall Fitness: {metrics.fitness_score:.2f}/100")
    print()
    
    # ==========================================================================
    # STEP 7: Submit Update to Server
    # ==========================================================================
    print("📤 Step 7: Submitting update to server...")
    
    # Create update
    update = await trainer.submit_update(updated_weights, metrics)
    
    # Submit to server
    await aggregator.receive_update(update)
    
    print(f"   Update submitted!")
    print(f"   Participant: {update.participant_id}")
    print(f"   Round: {update.round_number}")
    print(f"   Fitness: {update.fitness_score:.2f}")
    print()
    
    # ==========================================================================
    # STEP 8: Aggregate Updates
    # ==========================================================================
    print("🔄 Step 8: Aggregating updates...")
    
    # Aggregate (we have 1 participant, so aggregation is trivial)
    success = await aggregator.aggregate_updates()
    
    if success:
        print("   ✅ Aggregation successful!")
        print(f"   Participants: {len(aggregator.current_updates)}")
        
        # Get aggregated weights
        aggregated_weights = aggregator.global_weights
        print(f"   Global model updated")
    else:
        print("   ⚠️  Aggregation failed (not enough participants)")
    
    print()
    
    # ==========================================================================
    # STEP 9: Statistics
    # ==========================================================================
    print("📈 Step 9: Training statistics...")
    
    stats = trainer.get_statistics()
    
    print(f"   Total Rounds: {stats['total_rounds']}")
    print(f"   Total Samples: {stats['total_samples']}")
    print(f"   Total Time: {stats['total_training_time']:.1f}s")
    print(f"   Avg Loss: {stats['avg_loss']:.4f}")
    print(f"   Avg Fitness: {stats['avg_fitness']:.2f}/100")
    print()
    
    # ==========================================================================
    # COMPLETE!
    # ==========================================================================
    print("=" * 80)
    print("✅ COMPLETE! End-to-end genome training workflow successful!")
    print("=" * 80)
    print()
    print("What we demonstrated:")
    print("  ✅ Create genome with GenomeEncoder")
    print("  ✅ Build PyTorch model with ModelBuilder")
    print("  ✅ Validate architecture and estimate resources")
    print("  ✅ Setup federated server (FederatedAggregator)")
    print("  ✅ Setup training client (GenomeTrainer)")
    print("  ✅ Train genome locally with privacy preservation")
    print("  ✅ Calculate fitness scores (Quality, Timeliness, Honesty)")
    print("  ✅ Submit model update to server")
    print("  ✅ Aggregate updates (FedAvg)")
    print("  ✅ Track training statistics")
    print()
    print("Next Steps:")
    print("  - Add Evolution Orchestrator for multi-generation evolution")
    print("  - Implement mutation and crossover operators")
    print("  - Add configuration system for easy setup")
    print("  - Create comprehensive test suite")
    print("  - Integrate with BelizeChain staking pallet")
    print()


if __name__ == "__main__":
    asyncio.run(main())
