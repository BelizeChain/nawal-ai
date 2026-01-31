"""
Complete Evolution Example

Demonstrates end-to-end multi-generation genome evolution with federated learning.

Author: BelizeChain AI Team
Date: October 2025
"""

import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset

from nawal import (
    NawalConfig,
    EvolutionOrchestrator,
    load_config,
)


async def main():
    print("=" * 80)
    print("NAWAL COMPLETE EVOLUTION EXAMPLE")
    print("=" * 80)
    print()
    
    # ==========================================================================
    # STEP 1: Load Configuration
    # ==========================================================================
    print("📋 Step 1: Loading configuration...")
    
    # Load from YAML file (or use defaults)
    try:
        config = load_config("nawal/config.dev.yaml")
        print(f"   ✅ Loaded config from file: {config.environment}")
    except FileNotFoundError:
        config = NawalConfig()
        print("   ℹ️  Using default configuration")
    
    print(f"   Population: {config.evolution.population_size}")
    print(f"   Generations: {config.evolution.num_generations}")
    print(f"   Environment: {config.environment}")
    print()
    
    # ==========================================================================
    # STEP 2: Create Training Data
    # ==========================================================================
    print("📊 Step 2: Creating training data...")
    
    # Create dummy data (in production, use real datasets)
    num_train_samples = 500
    num_val_samples = 100
    seq_length = 128
    vocab_size = config.model.vocab_size
    
    # Training data
    train_input_ids = torch.randint(0, vocab_size, (num_train_samples, seq_length))
    train_dataset = TensorDataset(train_input_ids)
    
    def collate_fn(batch):
        input_ids = batch[0]
        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Validation data
    val_input_ids = torch.randint(0, vocab_size, (num_val_samples, seq_length))
    val_dataset = TensorDataset(val_input_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"   Training samples: {num_train_samples}")
    print(f"   Validation samples: {num_val_samples}")
    print(f"   Sequence length: {seq_length}")
    print()
    
    # ==========================================================================
    # STEP 3: Initialize Orchestrator
    # ==========================================================================
    print("🎭 Step 3: Initializing Evolution Orchestrator...")
    
    orchestrator = EvolutionOrchestrator(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    print(f"   Population initialized: {len(orchestrator.population.genomes)} genomes")
    print(f"   Target generations: {config.evolution.num_generations}")
    print(f"   Checkpoint dir: {config.storage.checkpoint_dir}")
    print()
    
    # ==========================================================================
    # STEP 4: Run Evolution
    # ==========================================================================
    print("🧬 Step 4: Starting evolution...")
    print()
    print("-" * 80)
    
    # Run evolution (this will take time!)
    best_genome = await orchestrator.evolve()
    
    print("-" * 80)
    print()
    
    # ==========================================================================
    # STEP 5: Results
    # ==========================================================================
    print("📈 Step 5: Evolution Results")
    print()
    
    stats = orchestrator.get_statistics()
    
    print(f"   Generations Completed: {stats['generations']}")
    print(f"   Best Fitness: {stats['best_fitness']:.2f}/100")
    print(f"   Final Avg Fitness: {stats['avg_fitness']:.2f}/100")
    print(f"   Final Diversity: {stats['diversity']:.2f}")
    print(f"   Total Training Time: {stats['total_training_time']:.1f}s")
    print(f"   Avg Generation Time: {stats['avg_generation_time']:.1f}s")
    print()
    
    print("   Best Genome:")
    print(f"   - Genome ID: {best_genome.genome_id}")
    print(f"   - Hidden Size: {best_genome.hidden_size}")
    print(f"   - Num Layers: {len(best_genome.encoder_layers)}")
    print(f"   - Fitness: {best_genome.fitness:.2f}/100")
    print()
    
    # ==========================================================================
    # STEP 6: Fitness History
    # ==========================================================================
    print("📊 Step 6: Fitness History")
    print()
    
    fitness_history = stats['fitness_history']
    
    print("   Generation | Best Fitness | Avg Fitness | Diversity")
    print("   " + "-" * 60)
    
    for i, state in enumerate(orchestrator.generation_history[:10]):  # First 10
        print(
            f"   {state.generation + 1:>10} | "
            f"{state.best_fitness:>12.2f} | "
            f"{state.avg_fitness:>11.2f} | "
            f"{state.diversity:>9.2f}"
        )
    
    if len(orchestrator.generation_history) > 10:
        print("   " + " " * 10 + "...")
        
        # Last 5
        for state in orchestrator.generation_history[-5:]:
            print(
                f"   {state.generation + 1:>10} | "
                f"{state.best_fitness:>12.2f} | "
                f"{state.avg_fitness:>11.2f} | "
                f"{state.diversity:>9.2f}"
            )
    
    print()
    
    # ==========================================================================
    # STEP 7: Improvement Analysis
    # ==========================================================================
    print("📈 Step 7: Improvement Analysis")
    print()
    
    if len(fitness_history) > 0:
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        improvement = final_fitness - initial_fitness
        improvement_pct = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
        
        print(f"   Initial Fitness: {initial_fitness:.2f}/100")
        print(f"   Final Fitness: {final_fitness:.2f}/100")
        print(f"   Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
        print()
        
        # Find best generation
        best_gen = max(
            enumerate(fitness_history),
            key=lambda x: x[1],
        )
        print(f"   Best Generation: {best_gen[0] + 1}")
        print(f"   Best Fitness: {best_gen[1]:.2f}/100")
        print()
    
    # ==========================================================================
    # STEP 8: Checkpoints
    # ==========================================================================
    print("💾 Step 8: Saved Checkpoints")
    print()
    
    checkpoint_dir = config.storage.checkpoint_dir
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        print(f"   Checkpoint directory: {checkpoint_dir}")
        print(f"   Total checkpoints: {len(checkpoints)}")
        
        if checkpoints:
            print("   Files:")
            for cp in checkpoints[-5:]:  # Last 5
                size_mb = cp.stat().st_size / 1024**2
                print(f"   - {cp.name} ({size_mb:.1f} MB)")
    else:
        print("   No checkpoints saved")
    
    print()
    
    # ==========================================================================
    # COMPLETE!
    # ==========================================================================
    print("=" * 80)
    print("✅ EVOLUTION COMPLETE!")
    print("=" * 80)
    print()
    print("What we demonstrated:")
    print("  ✅ Load configuration from YAML")
    print("  ✅ Initialize population with random genomes")
    print("  ✅ Run multi-generation evolution")
    print("  ✅ Evaluate genomes with federated learning")
    print("  ✅ Apply selection, crossover, mutation")
    print("  ✅ Preserve elite genomes")
    print("  ✅ Checkpoint evolution state")
    print("  ✅ Track fitness improvement over time")
    print("  ✅ Find best genome architecture")
    print()
    print("The best genome can now be:")
    print("  • Deployed to production")
    print("  • Integrated with BelizeChain staking")
    print("  • Fine-tuned for specific tasks")
    print("  • Used as parent for next evolution cycle")
    print()
    print("Next Steps:")
    print("  - Integrate with BelizeChain blockchain")
    print("  - Connect real validators via network")
    print("  - Add quantum computing integration")
    print("  - Implement advanced mutation operators")
    print("  - Deploy to production infrastructure")
    print()


if __name__ == "__main__":
    # Run evolution
    asyncio.run(main())
