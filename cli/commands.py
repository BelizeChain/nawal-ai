"""
CLI Commands for Nawal AI.

Provides command-line interface using Click framework.

Author: BelizeChain Team
License: MIT
"""

from typing import Optional
from pathlib import Path
import sys

from loguru import logger

# Optional Click library
try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    logger.warning("Click not available. Install: pip install click")


# Main CLI group
@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """
    Nawal AI - Federated Learning for BelizeChain.
    
    Evolutionary AI with Proof of Useful Work consensus.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)


# Training command
@cli.command()
@click.option("--dataset", "-d", default="wikitext-2", help="Dataset to use")
@click.option("--epochs", "-e", type=int, default=10, help="Number of epochs")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
@click.option("--checkpoint-dir", type=click.Path(), default="./checkpoints", help="Checkpoint directory")
@click.pass_context
def train(ctx, dataset: str, epochs: int, batch_size: int, learning_rate: float, checkpoint_dir: str):
    """Train AI model locally."""
    logger.info(f"Starting local training: dataset={dataset}, epochs={epochs}")
    
    try:
        from nawal.training import ValidatorTrainer, TrainingConfig
        from nawal.data import DataManager, DatasetConfig, DatasetType
        
        # Load dataset
        data_config = DatasetConfig(
            dataset_type=DatasetType(dataset),
            batch_size=batch_size,
        )
        data_manager = DataManager(data_config)
        train_loader, val_loader, _ = data_manager.get_dataloaders()
        
        # Create trainer
        training_config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_dir=Path(checkpoint_dir),
        )
        
        # Create simple model (placeholder)
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        
        trainer = ValidatorTrainer(training_config)
        
        # Train
        results = trainer.train(model, train_loader, val_loader)
        
        logger.success(f"Training complete! Final loss: {results['final_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


# Evolution command
@cli.command()
@click.option("--generations", "-g", type=int, default=20, help="Number of generations")
@click.option("--population", "-p", type=int, default=50, help="Population size")
@click.option("--mutation-rate", "-m", type=float, default=0.1, help="Mutation rate")
@click.option("--crossover-rate", "-x", type=float, default=0.7, help="Crossover rate")
@click.option("--checkpoint-dir", type=click.Path(), default="./evolution", help="Checkpoint directory")
@click.pass_context
def evolve(ctx, generations: int, population: int, mutation_rate: float, crossover_rate: float, checkpoint_dir: str):
    """Run evolutionary optimization."""
    logger.info(f"Starting evolution: generations={generations}, population={population}")
    
    try:
        from nawal.orchestrator import EvolutionOrchestrator
        from nawal.genome import GenomeConfig, GeneticAlgorithmConfig
        from nawal.config import EvolutionConfig
        
        # Configure evolution
        genome_config = GenomeConfig()
        ga_config = GeneticAlgorithmConfig(
            population_size=population,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )
        evolution_config = EvolutionConfig(
            num_generations=generations,
            checkpoint_dir=Path(checkpoint_dir),
        )
        
        # Create orchestrator
        orchestrator = EvolutionOrchestrator(
            genome_config=genome_config,
            ga_config=ga_config,
            evolution_config=evolution_config,
        )
        
        # Run evolution
        best_genome = orchestrator.run()
        
        logger.success(f"Evolution complete! Best fitness: {best_genome.fitness:.4f}")
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        sys.exit(1)


# Federated learning command
@cli.command()
@click.option("--num-clients", "-n", type=int, default=10, help="Number of clients")
@click.option("--rounds", "-r", type=int, default=100, help="Federated rounds")
@click.option("--min-clients", "-m", type=int, default=5, help="Minimum clients per round")
@click.option("--port", "-p", type=int, default=8080, help="Server port")
@click.pass_context
def federate(ctx, num_clients: int, rounds: int, min_clients: int, port: int):
    """Start federated learning server."""
    logger.info(f"Starting federated server: clients={num_clients}, rounds={rounds}")
    
    try:
        from nawal.server import FederatedServer, ServerConfig
        
        # Configure server
        server_config = ServerConfig(
            num_clients=num_clients,
            num_rounds=rounds,
            min_clients_per_round=min_clients,
            port=port,
        )
        
        # Create and start server
        server = FederatedServer(server_config)
        server.start()
        
        logger.success("Federated server started successfully")
        
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


# Validator command group
@cli.group()
def validator():
    """Validator operations (identity, staking, fitness)."""
    pass


@validator.command(name="register")
@click.option("--name", required=True, help="Validator name")
@click.option("--email", required=True, help="Contact email")
@click.option("--legal-name", help="Legal entity name")
@click.option("--tax-id", help="Tax ID")
@click.option("--keypair-uri", default="//Alice", help="Keypair URI")
@click.option("--chain", default="local", type=click.Choice(["local", "testnet", "mainnet"]))
def validator_register(name: str, email: str, legal_name: Optional[str], tax_id: Optional[str], keypair_uri: str, chain: str):
    """Register validator identity on-chain."""
    logger.info(f"Registering validator: {name}")
    
    try:
        from nawal.blockchain import SubstrateClient, ChainConfig, ValidatorManager, ValidatorIdentity
        
        # Connect to chain
        if chain == "local":
            config = ChainConfig.local()
        elif chain == "testnet":
            config = ChainConfig.testnet()
        else:
            config = ChainConfig.mainnet()
        
        client = SubstrateClient(config)
        client.connect()
        
        # Create keypair
        keypair = client.create_keypair(uri=keypair_uri)
        
        # Create identity
        identity = ValidatorIdentity(
            address=keypair.ss58_address,
            name=name,
            email=email,
            legal_name=legal_name,
            tax_id=tax_id,
        )
        
        # Register
        manager = ValidatorManager(client)
        receipt = manager.register_identity(keypair, identity)
        
        if receipt.success:
            logger.success(f"Validator registered: {keypair.ss58_address}")
        else:
            logger.error(f"Registration failed: {receipt.error}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Validator registration failed: {e}")
        sys.exit(1)


@validator.command(name="submit-fitness")
@click.option("--quality", type=float, required=True, help="Quality score (0-100)")
@click.option("--timeliness", type=float, required=True, help="Timeliness score (0-100)")
@click.option("--honesty", type=float, required=True, help="Honesty score (0-100)")
@click.option("--round", type=int, required=True, help="Training round")
@click.option("--keypair-uri", default="//Alice", help="Keypair URI")
@click.option("--chain", default="local", type=click.Choice(["local", "testnet", "mainnet"]))
def validator_submit_fitness(quality: float, timeliness: float, honesty: float, round: int, keypair_uri: str, chain: str):
    """Submit PoUW fitness score."""
    logger.info(f"Submitting fitness: Q={quality}, T={timeliness}, H={honesty}")
    
    try:
        from nawal.blockchain import SubstrateClient, ChainConfig, StakingInterface, FitnessScore
        
        # Connect to chain
        if chain == "local":
            config = ChainConfig.local()
        elif chain == "testnet":
            config = ChainConfig.testnet()
        else:
            config = ChainConfig.mainnet()
        
        client = SubstrateClient(config)
        client.connect()
        
        # Create keypair
        keypair = client.create_keypair(uri=keypair_uri)
        
        # Create fitness score
        score = FitnessScore(
            quality=quality,
            timeliness=timeliness,
            honesty=honesty,
            round=round,
        )
        
        # Submit
        staking = StakingInterface(client)
        receipt = staking.submit_fitness(keypair, score)
        
        if receipt.success:
            logger.success(f"Fitness submitted: total={score.total:.2f}")
        else:
            logger.error(f"Submission failed: {receipt.error}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fitness submission failed: {e}")
        sys.exit(1)


# Genome command group
@cli.group()
def genome():
    """Genome management (store, retrieve, lineage)."""
    pass


@genome.command(name="store")
@click.option("--genome-file", type=click.Path(exists=True), required=True, help="Genome JSON file")
@click.option("--fitness", type=float, required=True, help="Fitness score")
@click.option("--generation", type=int, required=True, help="Generation number")
@click.option("--keypair-uri", default="//Alice", help="Keypair URI")
@click.option("--chain", default="local", type=click.Choice(["local", "testnet", "mainnet"]))
@click.option("--storage", default="local", type=click.Choice(["local", "pakit"]))
def genome_store(genome_file: str, fitness: float, generation: int, keypair_uri: str, chain: str, storage: str):
    """Store genome on-chain."""
    logger.info(f"Storing genome: generation={generation}, fitness={fitness}")
    
    try:
        import json
        from nawal.blockchain import SubstrateClient, ChainConfig, GenomeRegistry, StorageBackend
        
        # Load genome
        with open(genome_file, "r") as f:
            genome_data = json.load(f)
        
        # Connect to chain
        if chain == "local":
            config = ChainConfig.local()
        elif chain == "testnet":
            config = ChainConfig.testnet()
        else:
            config = ChainConfig.mainnet()
        
        client = SubstrateClient(config)
        client.connect()
        
        # Create keypair
        keypair = client.create_keypair(uri=keypair_uri)
        
        # Store genome
        backend = StorageBackend(storage)
        registry = GenomeRegistry(client, storage_backend=backend)
        metadata = registry.store_genome(
            keypair=keypair,
            genome=genome_data,
            fitness=fitness,
            generation=generation,
        )
        
        logger.success(f"Genome stored: ID={metadata.genome_id[:16]}...")
        click.echo(f"Genome ID: {metadata.genome_id}")
        click.echo(f"Content Hash: {metadata.content_hash}")
        
    except Exception as e:
        logger.error(f"Genome storage failed: {e}")
        sys.exit(1)


@genome.command(name="get")
@click.argument("genome_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--chain", default="local", type=click.Choice(["local", "testnet", "mainnet"]))
def genome_get(genome_id: str, output: Optional[str], chain: str):
    """Retrieve genome from chain."""
    logger.info(f"Retrieving genome: {genome_id[:16]}...")
    
    try:
        import json
        from nawal.blockchain import SubstrateClient, ChainConfig, GenomeRegistry, StorageBackend
        
        # Connect to chain
        if chain == "local":
            config = ChainConfig.local()
        elif chain == "testnet":
            config = ChainConfig.testnet()
        else:
            config = ChainConfig.mainnet()
        
        client = SubstrateClient(config)
        client.connect()
        
        # Retrieve genome
        registry = GenomeRegistry(client, storage_backend=StorageBackend.LOCAL)
        genome_data = registry.get_genome(genome_id)
        
        if genome_data is None:
            logger.error(f"Genome not found: {genome_id}")
            sys.exit(1)
        
        # Output
        if output:
            with open(output, "w") as f:
                json.dump(genome_data, f, indent=2)
            logger.success(f"Genome saved to {output}")
        else:
            click.echo(json.dumps(genome_data, indent=2))
        
    except Exception as e:
        logger.error(f"Genome retrieval failed: {e}")
        sys.exit(1)


# Config command
@cli.command()
@click.option("--init", is_flag=True, help="Initialize default config")
@click.option("--validate", is_flag=True, help="Validate config file")
@click.option("--show", is_flag=True, help="Show current config")
@click.pass_context
def config(ctx, init: bool, validate: bool, show: bool):
    """Manage configuration files."""
    config_file = ctx.obj.get("config", "config.yaml")
    
    if init:
        logger.info("Initializing default configuration")
        try:
            from nawal.cli.config_manager import ConfigManager
            manager = ConfigManager()
            manager.create_default_config(Path(config_file))
            logger.success(f"Config created: {config_file}")
        except Exception as e:
            logger.error(f"Config initialization failed: {e}")
            sys.exit(1)
    
    elif validate:
        logger.info(f"Validating config: {config_file}")
        try:
            from nawal.cli.config_manager import ConfigManager
            manager = ConfigManager()
            manager.load_config(Path(config_file))
            logger.success("Config is valid")
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            sys.exit(1)
    
    elif show:
        try:
            from nawal.cli.config_manager import ConfigManager
            import yaml
            manager = ConfigManager()
            config_data = manager.load_config(Path(config_file))
            click.echo(yaml.dump(config_data, default_flow_style=False))
        except Exception as e:
            logger.error(f"Failed to show config: {e}")
            sys.exit(1)
    
    else:
        click.echo("Use --init, --validate, or --show")


if __name__ == "__main__":
    if not CLICK_AVAILABLE:
        print("Click library required. Install: pip install click")
        sys.exit(1)
    
    cli()
