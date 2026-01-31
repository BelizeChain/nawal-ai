"""
Genome Trainer - Train Genome-Based Models

Enhanced client for training evolved genomes in federated learning.
Integrates with the genome system, model builder, and federated server.

Features:
- Build models from genomes using ModelBuilder
- Local training with privacy preservation
- Fitness score calculation (Quality, Timeliness, Honesty)
- Integration with FederatedAggregator
- Belizean compliance checks
- Async training workflow

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from nawal.genome import Genome, ModelBuilder, GenomeModel
from nawal.server import ModelUpdate, TrainingMetrics


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Configuration for genome-based training.
    
    Pydantic v2-ready dataclass for training parameters.
    """
    
    # Identity
    participant_id: str
    validator_address: str
    staking_account: str
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    local_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str | None = "cosine"
    
    # Privacy
    privacy_epsilon: float | None = None  # Differential privacy
    gradient_clipping: bool = True
    
    # Compliance
    data_sovereignty_check: bool = True
    compliance_mode: bool = True
    
    # Performance
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = True  # FP16 training
    gradient_accumulation_steps: int = 1
    
    # Timeouts
    training_timeout: float = 3600.0  # 1 hour max
    submission_deadline: float = 300.0  # 5 minutes after round start
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Genome Trainer
# =============================================================================


class GenomeTrainer:
    """
    Train genome-based models locally for federated learning.
    
    This is the client-side component that validators run to participate
    in federated training of evolved AI genomes.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model_builder: ModelBuilder | None = None,
    ):
        """
        Initialize genome trainer.
        
        Args:
            config: Training configuration
            model_builder: Model builder (creates new one if not provided)
        """
        self.config = config
        self.model_builder = model_builder or ModelBuilder()
        
        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # State
        self.current_genome: Genome | None = None
        self.current_model: GenomeModel | None = None
        self.current_round: int = 0
        
        # Metrics
        self.training_start_time: float | None = None
        self.training_metrics: list[TrainingMetrics] = []
        
        # Byzantine detection state
        self.historical_updates: list[dict[str, torch.Tensor]] = []
        self.initial_weights: dict[str, torch.Tensor] = {}
        self.update_statistics: list[dict[str, float]] = []
        self.rounds_completed: int = 0
        
        # Data poisoning detection state
        self.prediction_history: list[dict[str, torch.Tensor]] = []  # Last 30 prediction distributions
        self.loss_history: list[float] = []  # Last 50 loss values
        self.activation_patterns: list[dict[str, torch.Tensor]] = []  # Last 20 activation snapshots
        self.clean_data_baseline: dict[str, float] | None = None  # Baseline statistics on clean data
        
        # Differential privacy state
        self.dp_config: Any | None = None  # DifferentialPrivacy config if enabled
        self.gradient_clip_history: list[float] = []  # Last 30 gradient norms before clipping
        self.noise_scale_history: list[float] = []  # Last 30 noise scales applied
        self.privacy_spent_history: list[tuple[float, int]] = []  # (epsilon_spent, steps) tracking
        
        # Data leakage detection state
        self.training_losses: list[float] = []  # Last 50 training losses
        self.validation_losses: list[float] = []  # Last 50 validation losses
        self.prediction_confidences: list[float] = []  # Last 100 prediction confidences
        self.weight_update_magnitudes: list[float] = []  # Last 30 weight update magnitudes
        
        logger.info(
            "Initialized GenomeTrainer",
            participant_id=config.participant_id,
            device=str(self.device),
            mixed_precision=config.mixed_precision,
        )
    
    def set_genome(self, genome: Genome, initial_weights: dict[str, torch.Tensor] | None = None) -> None:
        """
        Set genome to train.
        
        Args:
            genome: Genome specification
            initial_weights: Optional initial model weights (from server)
        """
        logger.info(f"Setting genome {genome.genome_id} for training")
        
        # Validate genome
        errors = self.model_builder.validate_genome(genome)
        if errors:
            logger.warning(f"Genome validation warnings: {errors}")
        
        # Build model
        self.current_genome = genome
        self.current_model = self.model_builder.build_model(genome)
        self.current_model = self.current_model.to(self.device)
        
        # Load initial weights if provided
        if initial_weights:
            self.current_model.load_state_dict(initial_weights)
            logger.info("Loaded initial weights from server")
        
        # Log model info
        logger.info(
            "Model ready for training",
            genome_id=genome.genome_id,
            parameters=f"{self.current_model.count_parameters():,}",
            memory=f"{self.current_model.get_memory_footprint() / 1024**2:.1f} MB",
        )
    
    def set_model(self, model: nn.Module) -> None:
        """
        Set model directly (backward compatibility).
        
        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)
        logger.info(f"Model set for training (backward compatibility mode)")
    
    def train_epoch(self, dataloader: Any) -> dict[str, float]:
        """
        Train for one epoch (backward compatibility).
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Dictionary of metrics
        """
        if not hasattr(self, 'model'):
            raise ValueError("No model set. Call set_model() first.")
        
        model = self.model
        model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Setup loss function based on config
        loss_function = getattr(self.config, 'loss_function', 'cross_entropy')
        if loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()  # Default
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs  # Self-supervised
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle dict outputs from GenomeModel (returns {"logits": ..., "loss": ...})
            if isinstance(outputs, dict):
                if "loss" in outputs and outputs["loss"] is not None:
                    # Model calculated loss internally
                    loss = outputs["loss"]
                else:
                    # Extract logits and calculate loss
                    logits = outputs.get("logits", outputs.get("output", None))
                    if logits is None:
                        raise ValueError("Model output dict must contain 'logits' or 'output' key")
                    
                    # Handle different output shapes
                    if logits.dim() == 3:  # (batch, seq, vocab)
                        # For sequence outputs with single targets (classification):
                        # Use the last position's output
                        if targets.dim() == 1 and targets.size(0) == logits.size(0):
                            # Classification: targets are (batch,), use last sequence position
                            logits_pooled = logits[:, -1, :]  # (batch, vocab)
                            loss = criterion(logits_pooled, targets)
                        else:
                            # Sequence-to-sequence: flatten everything
                            outputs_flat = logits.view(-1, logits.size(-1))
                            targets_flat = targets.view(-1)
                            loss = criterion(outputs_flat, targets_flat)
                    else:
                        loss = criterion(logits, targets)
            else:
                # Handle tensor outputs (standard case)
                # Handle different output shapes
                if outputs.dim() == 3:  # (batch, seq, vocab)
                    # Check if this is classification or sequence-to-sequence
                    if targets.dim() == 1 and targets.size(0) == outputs.size(0):
                        # Classification: use last position
                        outputs_pooled = outputs[:, -1, :]
                        loss = criterion(outputs_pooled, targets)
                    else:
                        # Sequence-to-sequence: flatten
                        outputs = outputs.view(-1, outputs.size(-1))
                        targets = targets.view(-1)
                        loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
        }
    
    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """
        Evaluate model (backward compatibility).
        
        Args:
            dataloader: Evaluation data loader
        
        Returns:
            Dictionary of metrics
        """
        if not hasattr(self, 'model'):
            raise ValueError("No model set. Call set_model() first.")
        
        model = self.model
        model.eval()
        
        # Setup loss function based on config
        loss_function = getattr(self.config, 'loss_function', 'cross_entropy')
        if loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()  # Default
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs
                
                outputs = model(inputs)
                
                # Track if we already calculated loss
                loss_calculated = False
                
                # Handle dict outputs from GenomeModel
                if isinstance(outputs, dict):
                    if "loss" in outputs and outputs["loss"] is not None:
                        loss = outputs["loss"]
                        loss_calculated = True
                        logits = outputs.get("logits")
                    else:
                        logits = outputs.get("logits", outputs.get("output", None))
                        if logits is None:
                            raise ValueError("Model output dict must contain 'logits' or 'output' key")
                    
                    # Use logits for subsequent processing
                    outputs = logits if logits is not None else outputs["output"]
                
                # Calculate loss if not already done
                if not loss_calculated:
                    if outputs.dim() == 3:
                        if targets.dim() == 1 and targets.size(0) == outputs.size(0):
                            # Classification: use last position
                            outputs_pooled = outputs[:, -1, :]
                            loss = criterion(outputs_pooled, targets)
                            # Use pooled outputs for predictions
                            outputs = outputs_pooled
                        else:
                            # Sequence-to-sequence: flatten
                            outputs_flat = outputs.view(-1, outputs.size(-1))
                            targets_flat = targets.view(-1)
                            loss = criterion(outputs_flat, targets_flat)
                            # Use flattened for predictions
                            outputs = outputs_flat
                            targets = targets_flat
                    else:
                        loss = criterion(outputs, targets)
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=-1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                total_loss += loss.item()
        
        accuracy = correct / max(total, 1)  # Return as fraction 0-1, not percentage
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
        }
    
    def calculate_fitness(
        self,
        metrics: dict[str, float],
        quality_weight: float = 0.4,
        timeliness_weight: float = 0.3,
        honesty_weight: float = 0.3,
    ) -> float:
        """
        Calculate fitness score based on metrics (backward compatibility).
        
        Args:
            metrics: Dictionary of metrics including accuracy, loss, etc.
            quality_weight: Weight for quality component
            timeliness_weight: Weight for timeliness component
            honesty_weight: Weight for honesty component
        
        Returns:
            Fitness score between 0 and 1
        """
        # Quality: Based on accuracy
        accuracy = metrics.get("accuracy", 0.0)
        quality_score = accuracy
        
        # Timeliness: Based on training time (if available)
        training_time = metrics.get("training_time", 1.0)
        # Normalize training time (assume 10 seconds is baseline)
        timeliness_score = max(0.0, 1.0 - (training_time / 10.0))
        
        # Honesty: Based on loss consistency (lower is better)
        loss = metrics.get("loss", 1.0)
        # Normalize loss (assume loss < 2.0 is good)
        honesty_score = max(0.0, 1.0 - (loss / 2.0))
        
        # Combined fitness
        fitness = (
            quality_weight * quality_score +
            timeliness_weight * timeliness_score +
            honesty_weight * honesty_score
        )
        
        return max(0.0, min(1.0, fitness))  # Clamp to [0, 1]
    
    def save_checkpoint(
        self,
        filepath: str | Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Save training checkpoint (backward compatibility).
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics
        """
        import torch
        from pathlib import Path
        
        if not hasattr(self, 'model'):
            raise ValueError("No model to save")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filepath: str | Path) -> tuple[int, dict[str, float]]:
        """
        Load training checkpoint (backward compatibility).
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            Tuple of (epoch, metrics)
        """
        import torch
        from pathlib import Path
        
        if not hasattr(self, 'model'):
            raise ValueError("No model to load checkpoint into")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        
        return epoch, metrics
    
    def train(
        self,
        *args,
        dataloader: Any = None,
        epochs: int | None = None,
        train_loader: Any = None,
        val_loader: Any = None,
        start_epoch: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train model for multiple epochs (backward compatibility).
        
        Args:
            *args: Positional arguments (train_loader, val_loader, epochs in old API)
            dataloader: Training data loader (legacy keyword)
            epochs: Number of epochs (uses config.epochs if not specified)
            train_loader: Training data loader (new API keyword)
            val_loader: Validation data loader (optional keyword)
            start_epoch: Starting epoch number (for resuming)
            **kwargs: Additional keyword arguments
        
        Returns:
            Dictionary of final metrics with history
        """
        if not hasattr(self, 'model'):
            raise ValueError("No model set. Call set_model() first.")
        
        # Handle positional arguments for backward compatibility
        # Old API: train(train_loader, val_loader, epochs=N)
        if len(args) >= 1:
            train_data = args[0]
            if len(args) >= 2:
                val_data = args[1]
            else:
                val_data = val_loader
        else:
            # Use keyword arguments
            train_data = train_loader or dataloader
            val_data = val_loader
        
        if train_data is None:
            raise ValueError("Must provide training data loader")
        
        num_epochs = epochs or getattr(self.config, 'epochs', None) or self.config.local_epochs
        
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_data)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validation if provided
            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                history["val_loss"].append(val_metrics["loss"])
            
            logger.info(f"Epoch {epoch + 1}/{start_epoch + num_epochs} - Loss: {train_metrics['loss']:.4f}")
        
        # Final evaluation on validation set if provided, otherwise on training set
        eval_data = val_data if val_data is not None else train_data
        final_metrics = self.evaluate(eval_data)
        
        # Add final loss to history if validation wasn't tracked
        if val_data is None and len(history["val_loss"]) == 0:
            history["val_loss"] = [final_metrics["loss"]]
        
        return {
            **history,
            "final_loss": final_metrics["loss"],
            "final_accuracy": final_metrics.get("accuracy", 0.0),
        }
    
    async def train_genome(
        self,
        genome: Genome,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        global_weights: dict[str, torch.Tensor] | None = None,
        round_number: int = 0,
    ) -> tuple[dict[str, torch.Tensor], TrainingMetrics]:
        """
        Train genome locally.
        
        Args:
            genome: Genome to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            global_weights: Global model weights from server
            round_number: Current federated learning round
        
        Returns:
            Tuple of (updated weights, training metrics)
        """
        self.current_round = round_number
        self.training_start_time = time.time()
        
        # Set genome and weights
        self.set_genome(genome, global_weights)
        
        # Setup optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # Training metrics
        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        
        # Training loop
        self.current_model.train()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Check timeout
                if time.time() - self.training_start_time > self.config.training_timeout:
                    logger.warning("Training timeout reached")
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                # Forward pass with mixed precision
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.current_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs["loss"]
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    outputs = self.current_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clipping:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.current_model.parameters(),
                            self.config.max_grad_norm,
                        )
                    
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    if scheduler:
                        scheduler.step()
                
                # Track metrics
                batch_loss = loss.item() * self.config.gradient_accumulation_steps
                batch_samples = input_ids.size(0)
                
                epoch_loss += batch_loss * batch_samples
                epoch_samples += batch_samples
                total_loss += batch_loss * batch_samples
                total_samples += batch_samples
                num_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.config.local_epochs}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {batch_loss:.4f}"
                    )
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            logger.info(
                f"Epoch {epoch + 1} complete",
                avg_loss=f"{avg_epoch_loss:.4f}",
                samples=epoch_samples,
            )
        
        # Training complete
        training_time = time.time() - self.training_start_time
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Validation (if provided)
        val_loss = None
        val_accuracy = None
        if val_loader:
            val_loss, val_accuracy = await self._validate(val_loader)
        
        # Calculate fitness scores
        quality_score = self._calculate_quality_score(avg_loss, val_loss)
        timeliness_score = self._calculate_timeliness_score(training_time)
        honesty_score = self._calculate_honesty_score()
        fitness_score = 0.4 * quality_score + 0.3 * timeliness_score + 0.3 * honesty_score
        
        # Create training metrics
        metrics = TrainingMetrics(
            participant_id=self.config.participant_id,
            genome_id=genome.genome_id,
            round_number=round_number,
            train_loss=avg_loss,
            train_accuracy=None,  # Not calculated for language modeling
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            samples_trained=total_samples,
            batches_processed=num_batches,
            epochs=self.config.local_epochs,
            training_time=training_time,
            throughput=total_samples / training_time if training_time > 0 else 0.0,
            quality_score=quality_score,
            timeliness_score=timeliness_score,
            honesty_score=honesty_score,
            fitness_score=fitness_score,
        )
        
        # Calculate throughput
        metrics.calculate_throughput()
        
        # Store metrics
        self.training_metrics.append(metrics)
        
        # Get updated weights
        updated_weights = self.current_model.state_dict()
        
        logger.info(
            "Training complete",
            round=round_number,
            loss=f"{avg_loss:.4f}",
            samples=total_samples,
            time=f"{training_time:.1f}s",
            fitness=f"{fitness_score:.2f}",
        )
        
        return updated_weights, metrics
    
    async def _validate(self, val_loader: DataLoader) -> tuple[float, float | None]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        self.current_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                outputs = self.current_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs["loss"]
                batch_samples = input_ids.size(0)
                
                total_loss += loss.item() * batch_samples
                total_samples += batch_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Validation complete: loss={avg_loss:.4f}")
        
        return avg_loss, None  # Accuracy not calculated for language modeling
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                self.current_model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.current_model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.current_model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            logger.warning(f"Unknown optimizer {self.config.optimizer}, using AdamW")
            return torch.optim.AdamW(
                self.current_model.parameters(),
                lr=self.config.learning_rate,
            )
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        if not self.config.scheduler:
            return None
        
        total_steps = num_training_steps * self.config.local_epochs
        
        if self.config.scheduler.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )
        elif self.config.scheduler.lower() == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps,
            )
        else:
            logger.warning(f"Unknown scheduler {self.config.scheduler}")
            return None
    
    def _calculate_quality_score(
        self,
        train_loss: float,
        val_loss: float | None = None,
    ) -> float:
        """
        Calculate quality score (0-100).
        
        Based on model improvement - lower loss = higher quality.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss (if available)
        
        Returns:
            Quality score (0-100)
        """
        # Use validation loss if available, otherwise training loss
        loss = val_loss if val_loss is not None else train_loss
        
        # Convert loss to score (exponential decay)
        # Loss of 0.0 = 100, loss of 2.0 = ~10, loss of 4.0 = ~1
        quality = 100.0 * (0.9 ** (loss * 10))
        
        return min(100.0, max(0.0, quality))
    
    def _calculate_timeliness_score(self, training_time: float) -> float:
        """
        Calculate timeliness score (0-100).
        
        Based on submission deadline adherence.
        
        Args:
            training_time: Training time in seconds
        
        Returns:
            Timeliness score (0-100)
        """
        deadline = self.config.submission_deadline
        
        if training_time <= deadline * 0.5:
            # Very fast: 100 points
            return 100.0
        elif training_time <= deadline:
            # On time: 70-100 points (linear decay)
            ratio = (training_time - deadline * 0.5) / (deadline * 0.5)
            return 100.0 - (30.0 * ratio)
        elif training_time <= deadline * 2:
            # Late: 30-70 points
            ratio = (training_time - deadline) / deadline
            return 70.0 - (40.0 * ratio)
        else:
            # Very late: 0-30 points
            return max(0.0, 30.0 - (training_time - deadline * 2) / 10.0)
    
    def _calculate_honesty_score(
        self,
        predictions: torch.Tensor | None = None,
        losses: list[float] | None = None,
        activations: dict[str, torch.Tensor] | None = None,
        gradients: dict[str, torch.Tensor] | None = None,
    ) -> float:
        """
        Calculate honesty score (0-100).
        
        Checks for:
        - Gradient integrity
        - Byzantine behavior
        - Data poisoning attempts
        - Differential privacy compliance (if enabled)
        
        Args:
            predictions: Optional model predictions for poisoning detection
            losses: Optional per-sample losses for distribution analysis
            activations: Optional layer activations for pattern analysis
            gradients: Optional gradients for DP verification
        
        Returns:
            Honesty score (0-100)
        """
        if self.current_model is None:
            return 100.0  # No model to verify yet
        
        # Get current weights
        current_weights = self.current_model.state_dict()
        
        # Run all verification checks
        scores = []
        
        # Check 1: Gradient norm verification
        try:
            norm_score = self._verify_gradient_norms(current_weights)
            scores.append(("gradient_norms", norm_score))
        except Exception as e:
            logger.warning(f"Gradient norm verification failed: {e}")
            scores.append(("gradient_norms", 100.0))
        
        # Check 2: Byzantine behavior detection
        try:
            byzantine_score = self._detect_byzantine_behavior(current_weights)
            scores.append(("byzantine", byzantine_score))
        except Exception as e:
            logger.warning(f"Byzantine detection failed: {e}")
            scores.append(("byzantine", 100.0))
        
        # Check 3: Weight magnitude verification
        try:
            magnitude_score = self._verify_weight_magnitudes(current_weights)
            scores.append(("magnitude", magnitude_score))
        except Exception as e:
            logger.warning(f"Magnitude verification failed: {e}")
            scores.append(("magnitude", 100.0))
        
        # Check 4: Data poisoning detection (if data available)
        if predictions is not None or losses or activations:
            try:
                poisoning_score = self._detect_data_poisoning(predictions, losses, activations)
                scores.append(("data_poisoning", poisoning_score))
            except Exception as e:
                logger.warning(f"Data poisoning detection failed: {e}")
                scores.append(("data_poisoning", 100.0))
        
        # Check 5: Differential privacy compliance (if DP enabled)
        if self.dp_config is not None and gradients is not None:
            try:
                dp_score = self._verify_differential_privacy(gradients, self.dp_config)
                scores.append(("differential_privacy", dp_score))
            except Exception as e:
                logger.warning(f"Differential privacy verification failed: {e}")
                scores.append(("differential_privacy", 100.0))

        # Check 6: Data leakage detection (train/val losses, predictions, gradients)
        if (self.training_losses or self.validation_losses) or predictions is not None or gradients is not None:
            try:
                leakage_score = self._verify_data_leakage(gradients=gradients, predictions=predictions)
                scores.append(("data_leakage", leakage_score))
            except Exception as e:
                logger.warning(f"Data leakage verification failed: {e}")
                scores.append(("data_leakage", 100.0))
        
        # Calculate weighted average
        if not scores:
            return 100.0
        
        # Dynamic weighting based on available checks
        if len(scores) == 6:  # All checks available (including DP and leakage)
            final_score = (
                scores[0][1] * 0.12 +  # Gradient norms
                scores[1][1] * 0.28 +  # Byzantine behavior
                scores[2][1] * 0.10 +  # Weight magnitudes
                scores[3][1] * 0.22 +  # Data poisoning
                scores[4][1] * 0.18 +  # Differential privacy
                scores[5][1] * 0.10    # Data leakage
            )
        elif len(scores) == 5:  # Without DP (data poisoning + leakage may be included)
            # If five checks, assume leakage or DP is missing; keep previous weighting but slightly adjust
            final_score = (
                scores[0][1] * 0.18 +  # Gradient norms
                scores[1][1] * 0.33 +  # Byzantine behavior
                scores[2][1] * 0.12 +  # Weight magnitudes
                scores[3][1] * 0.27 +  # Data poisoning
                scores[4][1] * 0.10    # Remaining check (DP or leakage)
            )
        elif len(scores) == 4:  # Without DP and leakage (data poisoning included)
            final_score = (
                scores[0][1] * 0.20 +  # Gradient norms
                scores[1][1] * 0.35 +  # Byzantine behavior
                scores[2][1] * 0.15 +  # Weight magnitudes
                scores[3][1] * 0.30    # Data poisoning
            )
        else:  # Only weight-based checks (fallback)
            final_score = (
                scores[0][1] * 0.30 +  # Gradient norms
                scores[1][1] * 0.50 +  # Byzantine behavior
                scores[2][1] * 0.20    # Weight magnitudes
            )
        
        if final_score < 70.0:
            logger.warning(
                "Low honesty score detected",
                final_score=f"{final_score:.2f}",
                checks={name: f"{score:.2f}" for name, score in scores},
            )
        
        return final_score
    
    async def submit_update(
        self,
        updated_weights: dict[str, torch.Tensor],
        metrics: TrainingMetrics,
    ) -> ModelUpdate:
        """
        Create model update for submission to server.
        
        Args:
            updated_weights: Updated model weights
            metrics: Training metrics
        
        Returns:
            Model update ready for submission
        """
        update = ModelUpdate(
            participant_id=self.config.participant_id,
            genome_id=self.current_genome.genome_id,
            round_number=self.current_round,
            weights=updated_weights,
            samples_trained=metrics.samples_trained,
            training_time=metrics.training_time,
            quality_score=metrics.quality_score,
            timeliness_score=metrics.timeliness_score,
            honesty_score=metrics.honesty_score,
            fitness_score=metrics.fitness_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        logger.info(
            "Model update ready for submission",
            round=self.current_round,
            fitness=f"{metrics.fitness_score:.2f}",
        )
        
        return update
    
    def validate_privacy_compliance(self, gradients: dict[str, torch.Tensor]) -> bool:
        """
        Validate privacy compliance.
        
        Checks:
        - Gradient norms within bounds
        - No data leakage patterns
        - Differential privacy if enabled
        
        Args:
            gradients: Model gradients
        
        Returns:
            True if compliant, False otherwise
        """
        if not self.config.compliance_mode:
            return True
        
        # Check gradient norms
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            grad_norm = grad.norm().item()
            if grad_norm > 100.0:  # Suspicious large gradient
                logger.warning(f"Large gradient detected: {name} = {grad_norm}")
                return False
        
        # =====================================================================
        # Differential Privacy Compliance Checks
        # =====================================================================
        
        if self.dp_config is not None:
            # Check 1: Verify gradient clipping is applied
            clip_norm = getattr(self.dp_config, "clip_norm", 1.0)
            for name, grad in gradients.items():
                if grad is None:
                    continue
                
                per_sample_grad_norm = grad.flatten().norm(2).item() / (grad.numel() ** 0.5)
                
                if per_sample_grad_norm > clip_norm * 1.1:  # 10% tolerance
                    logger.error(
                        f"DP violation: Gradient {name} norm {per_sample_grad_norm:.4f} "
                        f"exceeds clip_norm {clip_norm} - gradient clipping not applied!"
                    )
                    return False
            
            # Check 2: Verify privacy budget not exhausted
            if self.config.privacy_epsilon is not None:
                total_epsilon_spent = sum(eps for eps, _ in self.privacy_spent_history)
                
                if total_epsilon_spent > self.config.privacy_epsilon:
                    logger.error(
                        f"DP violation: Privacy budget exhausted! "
                        f"Spent: {total_epsilon_spent:.4f}, "
                        f"Budget: {self.config.privacy_epsilon:.4f}"
                    )
                    return False
                
                # Warn if close to budget (>90%)
                if total_epsilon_spent > self.config.privacy_epsilon * 0.9:
                    logger.warning(
                        f"Privacy budget nearly exhausted: "
                        f"{total_epsilon_spent/self.config.privacy_epsilon*100:.1f}% used"
                    )
            
            # Check 3: Verify noise is being added (check noise scale history)
            if hasattr(self.dp_config, "noise_multiplier"):
                noise_multiplier = self.dp_config.noise_multiplier
                
                if noise_multiplier < 0.5:
                    logger.warning(
                        f"DP warning: Noise multiplier {noise_multiplier} is very low. "
                        "Privacy guarantees may be weak."
                    )
                
                # Check that noise scale is reasonable
                expected_noise_scale = clip_norm * noise_multiplier
                if self.noise_scale_history and len(self.noise_scale_history) > 5:
                    recent_noise = self.noise_scale_history[-5:]
                    avg_recent_noise = sum(recent_noise) / len(recent_noise)
                    
                    # Verify noise scale is consistent with DP config
                    if abs(avg_recent_noise - expected_noise_scale) > expected_noise_scale * 0.2:
                        logger.error(
                            f"DP violation: Noise scale mismatch! "
                            f"Expected: {expected_noise_scale:.4f}, "
                            f"Actual: {avg_recent_noise:.4f}"
                        )
                        return False
        
        # =====================================================================
        # Data Leakage Detection
        # =====================================================================
        
        # Check 1: No PII patterns in gradients (basic heuristic)
        # PII patterns: SSN, credit card, phone numbers, email addresses
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Check for suspiciously structured gradients (could encode data)
            # Legitimate gradients should have certain statistical properties
            grad_flat = grad.flatten()
            
            # Check for zeros (could indicate sparse encoding of data)
            zero_ratio = (grad_flat == 0).sum().item() / grad_flat.numel()
            if zero_ratio > 0.95:  # >95% zeros is suspicious
                logger.warning(
                    f"Data leakage risk: Gradient {name} is {zero_ratio*100:.1f}% zeros. "
                    "May indicate sparse data encoding."
                )
            
            # Check for suspiciously uniform values (could encode specific data)
            if grad_flat.numel() > 100:  # Only check larger tensors
                unique_values = torch.unique(grad_flat)
                if len(unique_values) < grad_flat.numel() * 0.01:  # <1% unique values
                    logger.warning(
                        f"Data leakage risk: Gradient {name} has only "
                        f"{len(unique_values)} unique values out of {grad_flat.numel()}. "
                        "May indicate data encoding."
                    )
        
        # Check 2: Membership inference defense - gradients shouldn't be too specific
        # Large gradients on small models can leak training data presence
        if self.model_size_mb is not None and self.model_size_mb < 10.0:  # Small models
            for name, grad in gradients.items():
                if grad is None:
                    continue
                
                grad_norm = grad.norm().item()
                # Small models with large gradients are membership inference risks
                if grad_norm > 10.0:
                    logger.warning(
                        f"Membership inference risk: Small model ({self.model_size_mb:.1f}MB) "
                        f"with large gradient {name} = {grad_norm:.4f}. "
                        "May leak training sample presence."
                    )
        
        # Check 3: Model inversion defense - check for gradient consistency
        # Rapidly changing gradients can indicate overfitting to specific samples
        if len(self.gradient_clip_history) >= 10:
            recent_norms = self.gradient_clip_history[-10:]
            # Calculate variance in recent gradient norms
            import statistics
            
            if len(recent_norms) > 1:
                variance = statistics.variance(recent_norms)
                mean = statistics.mean(recent_norms)
                
                # High variance suggests overfitting to individual samples
                if mean > 0 and variance / (mean ** 2) > 2.0:  # Coefficient of variation > sqrt(2)
                    logger.warning(
                        f"Model inversion risk: High gradient norm variance "
                        f"(mean={mean:.4f}, var={variance:.4f}). "
                        "May indicate overfitting to specific samples."
                    )
        
        return True
    
    # =========================================================================
    # Byzantine Behavior Detection Methods
    # =========================================================================
    
    def _verify_gradient_norms(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Verify gradient norms are within acceptable bounds.
        
        Byzantine validators may submit updates with abnormally large or small
        gradients to disrupt consensus or poison the model.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): 100 = all gradients normal, 0 = highly suspicious
        """
        if not self.initial_weights:
            # First update, store as baseline (move to CPU for storage)
            self.initial_weights = {k: v.clone().detach().cpu() for k, v in weights.items()}
            return 100.0
        
        suspicious_count = 0
        total_count = 0
        
        for name, weight in weights.items():
            if name not in self.initial_weights:
                continue
            
            total_count += 1
            
            # Calculate gradient (weight change) - ensure same device
            weight_cpu = weight.detach().cpu()
            gradient = weight_cpu - self.initial_weights[name]
            grad_norm = gradient.norm().item()
            
            # Define acceptable range based on layer type
            max_norm = self._get_max_norm_for_layer(name)
            
            if grad_norm > max_norm:
                suspicious_count += 1
                logger.debug(
                    "Suspicious gradient norm",
                    layer=name,
                    norm=f"{grad_norm:.4f}",
                    max_expected=f"{max_norm:.4f}",
                )
        
        if total_count == 0:
            return 100.0
        
        # Score: 100 - (percentage_suspicious * 100)
        suspicious_ratio = suspicious_count / total_count
        score = max(0.0, 100.0 - (suspicious_ratio * 100.0))
        
        return score
    
    def _get_max_norm_for_layer(self, layer_name: str) -> float:
        """
        Get maximum acceptable gradient norm for a layer type.
        
        Different layer types have different expected gradient magnitudes.
        
        Args:
            layer_name: Name of the layer
        
        Returns:
            Maximum acceptable L2 norm
        """
        layer_lower = layer_name.lower()
        
        if "embed" in layer_lower or "embedding" in layer_lower:
            return 50.0  # Embeddings can have larger gradients
        elif "output" in layer_lower or "classifier" in layer_lower or "head" in layer_lower:
            return 100.0  # Output layers vary more
        elif "norm" in layer_lower or "layernorm" in layer_lower:
            return 10.0  # Normalization layers should be small
        else:
            return 30.0  # Hidden layers default
    
    def _detect_byzantine_behavior(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Detect Byzantine behavior through multiple statistical checks.
        
        Byzantine validators may:
        1. Submit random updates (high variance)
        2. Submit opposite-direction updates (negative similarity)
        3. Submit zero updates (no learning)
        4. Submit duplicate updates (copying others)
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): 100 = honest, 0 = Byzantine
        """
        checks = []
        
        # Check 1: Cosine similarity with historical updates
        if len(self.historical_updates) >= 3:
            similarity_score = self._check_update_similarity(weights)
            checks.append(("similarity", similarity_score))
        
        # Check 2: Statistical outlier detection
        if len(self.update_statistics) >= 5:
            outlier_score = self._check_statistical_outliers(weights)
            checks.append(("outlier", outlier_score))
        
        # Check 3: Zero update detection
        zero_score = self._check_zero_update(weights)
        checks.append(("zero_update", zero_score))
        
        # Check 4: Variance consistency
        variance_score = self._check_update_variance(weights)
        checks.append(("variance", variance_score))
        
        if not checks:
            return 100.0  # Not enough data yet
        
        # Average all checks
        avg_score = sum(score for _, score in checks) / len(checks)
        
        # Store update statistics for future checks
        self._store_update_statistics(weights)
        
        return avg_score
    
    def _check_update_similarity(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Check if update direction is similar to historical honest updates.
        
        Byzantine validators often submit updates in opposite directions
        or random directions unrelated to the learning objective.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): High similarity = honest, low/negative = Byzantine
        """
        if not self.historical_updates or not self.initial_weights:
            return 100.0
        
        # Flatten current update (difference from initial)
        current_diffs = []
        for name, weight in weights.items():
            if name in self.initial_weights:
                weight_cpu = weight.detach().cpu()
                diff = weight_cpu - self.initial_weights[name]
                current_diffs.append(diff.flatten())
        
        if not current_diffs:
            return 100.0
        
        current_flat = torch.cat(current_diffs)
        
        # Compare with recent historical updates
        similarities = []
        for hist_weights in self.historical_updates[-10:]:  # Last 10 updates
            hist_diffs = []
            for name, weight in hist_weights.items():
                if name in self.initial_weights:
                    # hist_weights already on CPU from storage
                    diff = weight - self.initial_weights[name]
                    hist_diffs.append(diff.flatten())
            
            if not hist_diffs:
                continue
            
            hist_flat = torch.cat(hist_diffs)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                current_flat.unsqueeze(0),
                hist_flat.unsqueeze(0),
            ).item()
            
            similarities.append(similarity)
        
        if not similarities:
            return 100.0
        
        avg_similarity = sum(similarities) / len(similarities)
        
        # Score based on similarity
        if avg_similarity < -0.5:
            return 0.0  # Opposite direction = Byzantine
        elif avg_similarity < 0.0:
            return 25.0  # Negative correlation = suspicious
        elif avg_similarity < 0.3:
            return 50.0  # Low similarity = questionable
        else:
            # High similarity = honest (0.3 to 1.0 maps to 50 to 100)
            return 50.0 + (avg_similarity - 0.3) * (50.0 / 0.7)
    
    def _check_statistical_outliers(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Detect if update is a statistical outlier using Z-score analysis.
        
        Byzantine validators may submit updates with abnormal statistics
        that differ significantly from the historical distribution.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): Within normal range = 100, outlier = 0
        """
        if len(self.update_statistics) < 5:
            return 100.0  # Need at least 5 updates for statistics
        
        # Calculate current update statistics
        weight_values = [w.flatten().cpu() for w in weights.values()]
        if not weight_values:
            return 100.0
        
        flat_weights = torch.cat(weight_values)
        current_mean = flat_weights.mean().item()
        current_std = flat_weights.std().item()
        
        # Compare with historical distribution
        historical_means = [stats['mean'] for stats in self.update_statistics]
        historical_stds = [stats['std'] for stats in self.update_statistics]
        
        mean_of_means = sum(historical_means) / len(historical_means)
        std_of_means = (
            sum((m - mean_of_means) ** 2 for m in historical_means) / len(historical_means)
        ) ** 0.5
        
        mean_of_stds = sum(historical_stds) / len(historical_stds)
        std_of_stds = (
            sum((s - mean_of_stds) ** 2 for s in historical_stds) / len(historical_stds)
        ) ** 0.5
        
        # Calculate Z-scores
        z_score_mean = 0.0
        if std_of_means > 0:
            z_score_mean = abs(current_mean - mean_of_means) / std_of_means
        
        z_score_std = 0.0
        if std_of_stds > 0:
            z_score_std = abs(current_std - mean_of_stds) / std_of_stds
        
        # Average Z-score
        avg_z_score = (z_score_mean + z_score_std) / 2.0
        
        # Score: >3σ is outlier (0 points), <1σ is normal (100 points)
        if avg_z_score > 3.0:
            return 0.0  # Extreme outlier
        elif avg_z_score > 2.0:
            return 33.0  # Significant outlier
        elif avg_z_score > 1.0:
            return 66.0  # Mild outlier
        else:
            # Linear interpolation from 66 to 100 for z < 1.0
            return 66.0 + (1.0 - avg_z_score) * 34.0
    
    def _check_zero_update(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Check if update is suspiciously close to zero (no learning).
        
        Byzantine validators may submit unchanged weights to avoid
        computational work while still claiming rewards.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): Normal update = 100, zero update = 0
        """
        if not self.initial_weights:
            return 100.0
        
        total_change = 0.0
        total_params = 0
        
        for name, weight in weights.items():
            if name in self.initial_weights:
                weight_cpu = weight.detach().cpu()
                diff = weight_cpu - self.initial_weights[name]
                total_change += diff.abs().sum().item()
                total_params += weight.numel()
        
        if total_params == 0:
            return 100.0
        
        # Average absolute change per parameter
        avg_change = total_change / total_params
        
        # Score based on change magnitude
        # Expect at least 1e-4 average change for meaningful learning
        if avg_change < 1e-6:
            return 0.0  # Essentially no change = lazy/Byzantine
        elif avg_change < 1e-5:
            return 25.0  # Very small change = suspicious
        elif avg_change < 1e-4:
            return 50.0  # Small change = questionable
        else:
            return 100.0  # Normal change = honest
    
    def _check_update_variance(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Check if update has consistent variance across layers.
        
        Byzantine validators submitting random noise will have
        inconsistent or abnormally high variance.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): Consistent variance = 100, chaotic = 0
        """
        if not self.initial_weights:
            return 100.0
        
        layer_variances = []
        
        for name, weight in weights.items():
            if name in self.initial_weights:
                weight_cpu = weight.detach().cpu()
                diff = weight_cpu - self.initial_weights[name]
                variance = diff.var().item()
                layer_variances.append(variance)
        
        if len(layer_variances) < 2:
            return 100.0
        
        # Check coefficient of variation (std / mean)
        mean_var = sum(layer_variances) / len(layer_variances)
        std_var = (
            sum((v - mean_var) ** 2 for v in layer_variances) / len(layer_variances)
        ) ** 0.5
        
        if mean_var < 1e-10:
            # Very low variance overall = zero update
            return 50.0
        
        coef_of_variation = std_var / mean_var
        
        # Score based on consistency
        # CoV < 1.0 = consistent, CoV > 5.0 = chaotic
        if coef_of_variation < 1.0:
            return 100.0  # Very consistent
        elif coef_of_variation < 2.0:
            return 80.0  # Reasonably consistent
        elif coef_of_variation < 5.0:
            return 50.0  # Somewhat inconsistent
        else:
            return 0.0  # Chaotic = likely Byzantine
    
    def _store_update_statistics(self, weights: dict[str, torch.Tensor]) -> None:
        """
        Store statistics of current update for future outlier detection.
        
        Args:
            weights: Current model weights
        """
        weight_values = [w.flatten().cpu() for w in weights.values()]
        if not weight_values:
            return
        
        flat_weights = torch.cat(weight_values)
        
        self.update_statistics.append({
            'mean': flat_weights.mean().item(),
            'std': flat_weights.std().item(),
            'timestamp': time.time(),
        })
        
        # Keep only last 50 updates
        if len(self.update_statistics) > 50:
            self.update_statistics = self.update_statistics[-50:]
        
        # Store in historical_updates (clone to prevent mutation)
        self.historical_updates.append({k: v.clone().cpu() for k, v in weights.items()})
        if len(self.historical_updates) > 20:
            self.historical_updates = self.historical_updates[-20:]
        
        self.rounds_completed += 1
    
    def _verify_weight_magnitudes(self, weights: dict[str, torch.Tensor]) -> float:
        """
        Verify each layer's weight magnitude is within expected bounds.
        
        Byzantine validators may submit weights with abnormally large
        magnitudes to disrupt the global model.
        
        Args:
            weights: Current model weights
        
        Returns:
            Score (0-100): All magnitudes normal = 100, abnormal = 0
        """
        if not self.initial_weights:
            return 100.0
        
        suspicious_layers = 0
        total_layers = 0
        
        for name, weight in weights.items():
            if name not in self.initial_weights:
                continue
            
            total_layers += 1
            
            # Calculate change magnitude (L2 norm of difference)
            weight_cpu = weight.detach().cpu()
            diff = weight_cpu - self.initial_weights[name]
            magnitude = diff.norm().item()
            
            # Expected magnitude based on training rounds
            expected_max = self._get_expected_magnitude(name, self.rounds_completed)
            
            if magnitude > expected_max * 2.0:  # More than 2x expected
                suspicious_layers += 1
                logger.debug(
                    "Large weight magnitude",
                    layer=name,
                    magnitude=f"{magnitude:.4f}",
                    expected_max=f"{expected_max:.4f}",
                )
        
        if total_layers == 0:
            return 100.0
        
        suspicious_ratio = suspicious_layers / total_layers
        score = max(0.0, 100.0 - (suspicious_ratio * 100.0))
        
        return score
    
    def _get_expected_magnitude(self, layer_name: str, rounds: int) -> float:
        """
        Get expected weight change magnitude for a layer after N rounds.
        
        Args:
            layer_name: Name of the layer
            rounds: Number of training rounds completed
        
        Returns:
            Expected maximum L2 norm of weight change
        """
        if rounds == 0:
            return 100.0  # First round, be lenient
        
        base_magnitude = 5.0  # Base per round
        layer_multiplier = 1.0
        
        layer_lower = layer_name.lower()
        if "embed" in layer_lower:
            layer_multiplier = 1.5
        elif "output" in layer_lower or "classifier" in layer_lower:
            layer_multiplier = 2.0
        elif "norm" in layer_lower:
            layer_multiplier = 0.5
        
        return base_magnitude * min(rounds, 10) * layer_multiplier
    
    # =========================================================================
    # Data Poisoning Detection Methods
    # =========================================================================
    
    def _detect_data_poisoning(
        self,
        predictions: torch.Tensor | None = None,
        losses: list[float] | None = None,
        activations: dict[str, torch.Tensor] | None = None,
    ) -> float:
        """
        Detect data poisoning attempts through multi-metric analysis.
        
        Data poisoning attacks involve training on corrupted data that causes
        the model to behave incorrectly on specific inputs (backdoors) or
        degrade general performance.
        
        Detection strategies:
        1. Loss distribution analysis (bimodal = poisoned)
        2. Prediction consistency (divergence from consensus)
        3. Feature distribution analysis (unusual patterns)
        4. Activation pattern monitoring (backdoor triggers)
        
        Args:
            predictions: Model predictions on validation set (optional)
            losses: Per-sample loss values (optional)
            activations: Layer activations for analysis (optional)
        
        Returns:
            Score (0-100): 100 = clean data, 0 = highly suspicious poisoning
        """
        checks = []
        
        # Check 1: Loss distribution analysis
        if losses and len(losses) >= 10:
            try:
                loss_score = self._check_loss_distribution(losses)
                checks.append(("loss_distribution", loss_score))
            except Exception as e:
                logger.warning(f"Loss distribution check failed: {e}")
        
        # Check 2: Prediction consistency
        if predictions is not None and len(self.prediction_history) >= 3:
            try:
                pred_score = self._check_prediction_consistency(predictions)
                checks.append(("prediction_consistency", pred_score))
            except Exception as e:
                logger.warning(f"Prediction consistency check failed: {e}")
        
        # Check 3: Feature distribution analysis
        if activations and len(self.activation_patterns) >= 5:
            try:
                feature_score = self._check_feature_distribution(activations)
                checks.append(("feature_distribution", feature_score))
            except Exception as e:
                logger.warning(f"Feature distribution check failed: {e}")
        
        # Check 4: Activation pattern monitoring
        if activations:
            try:
                activation_score = self._check_activation_patterns(activations)
                checks.append(("activation_patterns", activation_score))
            except Exception as e:
                logger.warning(f"Activation pattern check failed: {e}")
        
        # Store current data for future checks
        if predictions is not None:
            self._store_predictions(predictions)
        if losses:
            self.loss_history.extend(losses)
            self.loss_history = self.loss_history[-50:]  # Keep last 50
        if activations:
            self._store_activations(activations)
        
        # Return average score or 100 if no checks ran
        if not checks:
            return 100.0
        
        avg_score = sum(score for _, score in checks) / len(checks)
        
        if avg_score < 70.0:
            logger.warning(
                "Potential data poisoning detected",
                score=f"{avg_score:.2f}",
                checks={name: f"{score:.2f}" for name, score in checks},
            )
        
        return avg_score
    
    def _check_loss_distribution(self, losses: list[float]) -> float:
        """
        Check if loss distribution is bimodal (indicator of poisoning).
        
        Poisoned models often have bimodal loss distribution:
        - Low loss on clean data
        - High loss on poisoned samples (or vice versa for backdoors)
        
        Args:
            losses: Per-sample loss values
        
        Returns:
            Score (0-100): 100 = unimodal (clean), 0 = bimodal (poisoned)
        """
        if len(losses) < 10:
            return 100.0  # Not enough data
        
        # Convert to tensor for analysis
        loss_tensor = torch.tensor(losses)
        
        # Calculate distribution statistics
        mean_loss = loss_tensor.mean().item()
        std_loss = loss_tensor.std().item()
        median_loss = loss_tensor.median().item()
        
        # Check for bimodality using coefficient of variation
        cv = std_loss / mean_loss if mean_loss > 1e-6 else 0.0
        
        # Check mean-median divergence (bimodal distributions have divergent mean/median)
        mean_median_ratio = abs(mean_loss - median_loss) / (mean_loss + 1e-6)
        
        # Check for outliers (potential poisoned samples)
        q75 = loss_tensor.quantile(0.75).item()
        q25 = loss_tensor.quantile(0.25).item()
        iqr = q75 - q25
        outlier_count = ((loss_tensor > q75 + 1.5 * iqr) | (loss_tensor < q25 - 1.5 * iqr)).sum().item()
        outlier_ratio = outlier_count / len(losses)
        
        # Scoring logic
        score = 100.0
        
        # High CV suggests high variance (possible bimodal)
        if cv > 0.9:
            score -= 30.0
        elif cv > 0.7:
            score -= 15.0
        
        # Mean-median divergence suggests skewed/bimodal
        if mean_median_ratio > 0.3:
            score -= 30.0
        elif mean_median_ratio > 0.2:
            score -= 15.0
        
        # Too many outliers suggests poisoning
        if outlier_ratio > 0.15:  # >15% outliers
            score -= 40.0
        elif outlier_ratio > 0.10:  # >10% outliers
            score -= 20.0
        
        return max(0.0, score)
    
    def _check_prediction_consistency(self, predictions: torch.Tensor) -> float:
        """
        Check if predictions are consistent with historical consensus.
        
        Poisoned models may produce predictions that diverge significantly
        from the consensus of honest validators.
        
        Args:
            predictions: Current model predictions (logits or probabilities)
        
        Returns:
            Score (0-100): 100 = consistent, 0 = highly divergent
        """
        if len(self.prediction_history) < 3:
            return 100.0  # Not enough history
        
        # Flatten predictions for comparison
        current_flat = predictions.detach().cpu().flatten()
        
        # Compare with recent historical predictions
        similarities = []
        for hist_entry in self.prediction_history[-10:]:
            # Extract tensor from storage dict
            hist_preds = hist_entry["predictions"]
            hist_flat = hist_preds.flatten()
            
            # Ensure same size (pad or truncate if needed)
            min_len = min(len(current_flat), len(hist_flat))
            current_sample = current_flat[:min_len]
            hist_sample = hist_flat[:min_len]
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                current_sample.unsqueeze(0),
                hist_sample.unsqueeze(0),
            ).item()
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)

        # If history itself is noisy (no consensus), be more lenient
        if len(similarities) > 1:
            sim_std = torch.tensor(similarities).std(unbiased=False).item()
            if sim_std > 0.4 and avg_similarity > -0.2:
                return 85.0
        
        # Scoring based on similarity
        if avg_similarity < 0.0:
            return 0.0  # Opposite predictions = poisoned
        elif avg_similarity < 0.3:
            return 25.0  # Low similarity = suspicious
        elif avg_similarity < 0.5:
            return 50.0  # Moderate similarity = questionable
        elif avg_similarity < 0.7:
            return 75.0  # Good similarity
        else:
            # High similarity (0.7-1.0) = 75-100 points
            return 75.0 + (avg_similarity - 0.7) * (25.0 / 0.3)
    
    def _check_feature_distribution(self, activations: dict[str, torch.Tensor]) -> float:
        """
        Check if feature distributions match historical patterns.
        
        Poisoned data may cause unusual activation distributions in
        intermediate layers (e.g., dead neurons, extreme activations).
        
        Args:
            activations: Layer activations
        
        Returns:
            Score (0-100): 100 = normal distribution, 0 = anomalous
        """
        if len(self.activation_patterns) < 5:
            return 100.0  # Not enough history
        
        suspicious_layers = 0
        total_layers = 0
        
        for layer_name, current_act in activations.items():
            # Find historical activations for this layer
            hist_acts = []
            for hist_pattern in self.activation_patterns[-10:]:
                if layer_name in hist_pattern:
                    hist_acts.append(hist_pattern[layer_name])
            
            if not hist_acts:
                continue
            
            total_layers += 1
            
            # Calculate current statistics
            current_act_cpu = current_act.detach().cpu().flatten()
            current_mean = current_act_cpu.mean().item()
            current_std = current_act_cpu.std().item()
            
            # Calculate historical statistics
            hist_means = [act.flatten().mean().item() for act in hist_acts]
            hist_mean_avg = sum(hist_means) / len(hist_means)
            hist_mean_std = (sum((m - hist_mean_avg) ** 2 for m in hist_means) / len(hist_means)) ** 0.5

            # Guard against tiny variance to avoid over-triggering
            variance_floor = max(abs(hist_mean_avg) * 0.05, 1e-3)
            hist_mean_std = max(hist_mean_std, variance_floor)
            
            # Z-score for current mean
            z_score = abs(current_mean - hist_mean_avg) / (hist_mean_std + 1e-6)

            # Track standard deviation drift vs historical patterns
            hist_stds = [act.flatten().std().item() for act in hist_acts]
            hist_std_avg = sum(hist_stds) / len(hist_stds)
            hist_std_avg = max(hist_std_avg, 1e-3)
            std_drift = abs(current_std - hist_std_avg) / hist_std_avg
            
            # Check for dead neurons (all zeros or near-zero)
            zero_ratio = (current_act_cpu.abs() < 1e-4).float().mean().item()
            
            # Flag as suspicious if outlier or too many dead neurons
            mean_drift = z_score > 4.0
            variance_spike = std_drift > 2.0  # >200% std drift is suspicious
            dead_neurons = zero_ratio > 0.3  # moderate dead-neuron ratio

            if dead_neurons or (mean_drift and variance_spike):
                suspicious_layers += 1
        
        if total_layers == 0:
            return 100.0
        
        # Score based on suspicious layer ratio
        suspicious_ratio = suspicious_layers / total_layers
        score = max(0.0, 100.0 - (suspicious_ratio * 80.0))
        
        return score
    
    def _check_activation_patterns(self, activations: dict[str, torch.Tensor]) -> float:
        """
        Check for unusual activation patterns (potential backdoor triggers).
        
        Backdoor attacks often cause specific neurons to have abnormally high
        activations when the trigger is present.
        
        Args:
            activations: Layer activations
        
        Returns:
            Score (0-100): 100 = normal patterns, 0 = backdoor suspected
        """
        suspicious_count = 0
        total_checks = 0
        
        for layer_name, act in activations.items():
            total_checks += 1
            
            act_cpu = act.detach().cpu().flatten()
            
            # Check for abnormally high activations (potential trigger)
            max_act = act_cpu.max().item()
            mean_act = act_cpu.mean().item()
            std_act = act_cpu.std().item()
            
            # Z-score of maximum activation
            z_max = (max_act - mean_act) / (std_act + 1e-6)
            
            # Check for sparse high activations (backdoor signature)
            high_act_count = (act_cpu > mean_act + 3 * std_act).sum().item()
            high_act_ratio = high_act_count / len(act_cpu)
            
            # Flag suspicious if:
            # 1. Max activation is extreme outlier (>5σ)
            # 2. Very sparse high activations (<1% but >5σ)
            if z_max > 5.0:
                suspicious_count += 1
            elif high_act_ratio < 0.01 and z_max > 3.0:
                suspicious_count += 1
        
        if total_checks == 0:
            return 100.0
        
        # Score based on suspicious pattern ratio
        suspicious_ratio = suspicious_count / total_checks
        score = max(0.0, 100.0 - (suspicious_ratio * 100.0))
        
        return score
    
    def _store_predictions(self, predictions: torch.Tensor) -> None:
        """Store predictions for historical comparison."""
        # Store on CPU to avoid device issues
        pred_cpu = predictions.detach().cpu().clone()
        self.prediction_history.append({"predictions": pred_cpu})
        
        # Keep last 30 predictions
        if len(self.prediction_history) > 30:
            self.prediction_history = self.prediction_history[-30:]
    
    def _store_activations(self, activations: dict[str, torch.Tensor]) -> None:
        """Store activation patterns for historical comparison."""
        # Store on CPU to avoid device issues
        act_cpu = {name: act.detach().cpu().clone() for name, act in activations.items()}
        self.activation_patterns.append(act_cpu)
        
        # Keep last 20 activation patterns
        if len(self.activation_patterns) > 20:
            self.activation_patterns = self.activation_patterns[-20:]
    
    # =========================================================================
    # Differential Privacy Verification Methods
    # =========================================================================
    
    def _verify_differential_privacy(
        self,
        gradients: dict[str, torch.Tensor],
        dp_config: Any | None = None,
    ) -> float:
        """
        Verify differential privacy compliance.
        
        Checks:
        1. Gradient clipping (bounded sensitivity)
        2. Noise injection (privacy guarantee)
        3. Privacy budget tracking (epsilon not exhausted)
        
        Args:
            gradients: Current gradients to check
            dp_config: DifferentialPrivacy configuration (uses self.dp_config if None)
        
        Returns:
            Score from 0-100 (100 = fully compliant)
        """
        if dp_config is None:
            dp_config = self.dp_config
        
        if dp_config is None:
            # No DP configured - return perfect score (not required)
            return 100.0
        
        checks = []
        
        # Check 1: Gradient clipping compliance
        if gradients:
            clip_score = self._check_gradient_clipping(gradients, dp_config)
            checks.append(("gradient_clipping", clip_score))
        
        # Check 2: Privacy budget health
        budget_score = self._check_privacy_budget(dp_config)
        checks.append(("privacy_budget", budget_score))
        
        # Check 3: Noise scale tracking (if history available)
        if self.noise_scale_history:
            noise_score = self._check_noise_consistency(dp_config)
            checks.append(("noise_consistency", noise_score))
        
        # Average all available checks
        if not checks:
            return 100.0
        
        total_score = sum(score for _, score in checks) / len(checks)
        
        return total_score
    
    def _check_gradient_clipping(
        self,
        gradients: dict[str, torch.Tensor],
        dp_config: Any,
    ) -> float:
        """
        Check gradient clipping compliance.
        
        Verifies:
        - No gradient exceeds clip_norm
        - Clipping is consistently applied
        - Gradient norms are within expected range
        
        Args:
            gradients: Gradients to check
            dp_config: DP configuration with clip_norm
        
        Returns:
            Score from 0-100
        """
        clip_norm = getattr(dp_config, "clip_norm", 1.0)
        
        violations = 0
        total_grads = 0
        max_norm = 0.0
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            total_grads += 1
            # Per-element norm (not full tensor norm) for realistic clipping check
            grad_norm = grad.flatten().norm(2).item() / (grad.numel() ** 0.5)
            max_norm = max(max_norm, grad_norm)
            
            # Check if exceeds clip norm (with small tolerance for numerical precision)
            if grad_norm > clip_norm * 1.05:  # 5% tolerance for numerical variance
                violations += 1
        
        if total_grads == 0:
            return 100.0
        
        # Store gradient norm for history
        self.gradient_clip_history.append(max_norm)
        if len(self.gradient_clip_history) > 30:
            self.gradient_clip_history = self.gradient_clip_history[-30:]
        
        # Scoring: penalize violations
        violation_ratio = violations / total_grads
        score = 100.0 - (violation_ratio * 100.0)
        
        # Additional penalty if max norm is way over clip norm (>2x)
        if max_norm > clip_norm * 2.0:
            score -= 20.0
        
        return max(0.0, score)
    
    def _check_privacy_budget(self, dp_config: Any) -> float:
        """
        Check privacy budget health.
        
        Verifies:
        - Epsilon not exhausted
        - Spending rate is sustainable
        - Delta is reasonable
        
        Args:
            dp_config: DP configuration with privacy budget
        
        Returns:
            Score from 0-100
        """
        # Get budget information
        budget = getattr(dp_config, "budget", None)
        if budget is None:
            return 100.0  # No budget tracking
        
        target_epsilon = getattr(budget, "epsilon", 1.0)
        spent_epsilon = getattr(budget, "spent_epsilon", 0.0)
        delta = getattr(budget, "delta", 1e-5)
        
        # Store privacy spending
        steps = getattr(budget, "steps", 0)
        self.privacy_spent_history.append((spent_epsilon, steps))
        if len(self.privacy_spent_history) > 50:
            self.privacy_spent_history = self.privacy_spent_history[-50:]
        
        # Calculate remaining budget ratio
        remaining_ratio = (target_epsilon - spent_epsilon) / target_epsilon if target_epsilon > 0 else 0.0
        
        # Scoring based on remaining budget
        if spent_epsilon > target_epsilon:
            # Budget exhausted - critical failure
            return 0.0
        elif remaining_ratio < 0.1:
            # Less than 10% remaining - very concerning
            return 20.0
        elif remaining_ratio < 0.25:
            # Less than 25% remaining - concerning
            return 50.0
        elif remaining_ratio < 0.5:
            # Less than 50% remaining - warning
            return 75.0
        else:
            # Plenty of budget remaining - healthy
            return 100.0
    
    def _check_noise_consistency(self, dp_config: Any) -> float:
        """
        Check noise injection consistency.
        
        Verifies:
        - Noise scale is consistent with configuration
        - No significant deviations from expected noise
        
        Args:
            dp_config: DP configuration with noise_multiplier
        
        Returns:
            Score from 0-100
        """
        if not self.noise_scale_history:
            return 100.0
        
        clip_norm = getattr(dp_config, "clip_norm", 1.0)
        noise_multiplier = getattr(dp_config, "noise_multiplier", 1.0)
        expected_noise_scale = noise_multiplier * clip_norm
        
        # Check recent noise scales
        recent_noise = self.noise_scale_history[-10:]
        
        # Calculate deviation from expected
        deviations = []
        for noise_scale in recent_noise:
            relative_deviation = abs(noise_scale - expected_noise_scale) / (expected_noise_scale + 1e-6)
            deviations.append(relative_deviation)
        
        avg_deviation = sum(deviations) / len(deviations)
        
        # Scoring: penalize large deviations
        if avg_deviation < 0.1:
            # Within 10% - excellent
            return 100.0
        elif avg_deviation < 0.25:
            # Within 25% - good
            return 85.0
        elif avg_deviation < 0.5:
            # Within 50% - acceptable
            return 70.0
        else:
            # Large deviation - concerning
            return max(0.0, 100.0 - (avg_deviation * 100.0))
    
    def _store_noise_scale(self, noise_scale: float) -> None:
        """Store noise scale for history tracking."""
        self.noise_scale_history.append(noise_scale)
        if len(self.noise_scale_history) > 30:
            self.noise_scale_history = self.noise_scale_history[-30:]
    
    # =====================================================================
    # Data Leakage Detection Methods
    # =====================================================================
    
    def _verify_data_leakage(
        self,
        gradients: dict[str, torch.Tensor] | None = None,
        predictions: torch.Tensor | None = None,
    ) -> float:
        """
        Verify data leakage protection.
        
        Detects:
        1. Membership inference (model memorization)
        2. Gradient inversion (gradient attacks)
        3. Information leakage (overfitting, high confidence)
        
        Args:
            gradients: Model gradients (for inversion check)
            predictions: Model predictions (for information leakage)
        
        Returns:
            Score from 0-100 (100 = no leakage detected)
        """
        checks = []
        
        # Check 1: Membership inference (train/val loss gap)
        if self.training_losses and self.validation_losses:
            membership_score = self._check_membership_inference()
            checks.append(("membership_inference", membership_score))
        
        # Check 2: Gradient inversion (early layer gradient magnitude)
        if gradients:
            inversion_score = self._check_gradient_inversion(gradients)
            checks.append(("gradient_inversion", inversion_score))
        
        # Check 3: Information leakage (prediction confidence/entropy)
        if predictions is not None:
            info_score = self._check_information_leakage(predictions)
            checks.append(("information_leakage", info_score))
        
        if not checks:
            return 100.0  # No leakage checks available - assume safe
        
        # Average all available checks
        return sum(score for _, score in checks) / len(checks)
    
    def _check_membership_inference(self) -> float:
        """
        Check for membership inference vulnerability.
        
        Detects model memorization by analyzing train/val loss gap.
        Large gap indicates overfitting and potential memorization.
        
        Returns:
            Score from 0-100 (100 = no memorization)
        """
        if not self.training_losses or not self.validation_losses:
            return 100.0
        
        # Use recent losses (last 10 of each)
        recent_train = self.training_losses[-10:]
        recent_val = self.validation_losses[-10:]
        
        avg_train_loss = sum(recent_train) / len(recent_train)
        avg_val_loss = sum(recent_val) / len(recent_val)
        
        # Calculate loss gap (relative to validation loss)
        if avg_val_loss < 1e-6:
            return 100.0  # Avoid division by zero
        
        loss_gap = (avg_val_loss - avg_train_loss) / avg_val_loss
        
        # Scoring based on loss gap
        # Make thresholds slightly more permissive to avoid false positives on small gaps
        if loss_gap < 0.10:
            # <10% gap - excellent generalization
            return 100.0
        elif loss_gap < 0.20:
            # <20% gap - good generalization
            return 85.0
        elif loss_gap < 0.35:
            # <35% gap - acceptable generalization
            return 70.0
        elif loss_gap < 0.50:
            # <50% gap - concerning overfitting
            return 50.0
        else:
            # ≥50% gap - severe memorization
            return max(0.0, 100.0 - (loss_gap * 100.0))
    
    def _check_gradient_inversion(
        self,
        gradients: dict[str, torch.Tensor],
    ) -> float:
        """
        Check for gradient inversion vulnerability.
        
        Detects gradient attacks by analyzing early layer gradient magnitudes.
        High magnitudes in early layers can leak input information.
        
        Args:
            gradients: Model gradients
        
        Returns:
            Score from 0-100 (100 = safe gradients)
        """
        if not gradients:
            return 100.0
        
        # Identify early layer gradients (first 2 layers)
        early_layer_gradients = []
        layer_names = sorted(gradients.keys())  # Sort for consistent ordering
        
        for name in layer_names[:2]:  # First 2 layers
            grad = gradients.get(name)
            if grad is not None:
                early_layer_gradients.append(grad)
        
        if not early_layer_gradients:
            return 100.0
        
        # Calculate max gradient magnitude across early layers
        max_grad_magnitude = 0.0
        for grad in early_layer_gradients:
            grad_magnitude = grad.abs().max().item()
            max_grad_magnitude = max(max_grad_magnitude, grad_magnitude)
        
        # Scoring based on magnitude (threshold: 1.0 is typical)
        if max_grad_magnitude < 1.0:
            # Low magnitude - safe
            return 100.0
        elif max_grad_magnitude < 5.0:
            # Moderate magnitude - acceptable
            return 80.0
        elif max_grad_magnitude < 10.0:
            # High magnitude - concerning
            return 60.0
        elif max_grad_magnitude < 50.0:
            # Very high magnitude - dangerous
            return 30.0
        else:
            # Extreme magnitude - critical
            return 0.0
    
    def _check_information_leakage(
        self,
        predictions: torch.Tensor,
    ) -> float:
        """
        Check for information leakage through predictions.
        
        Detects overfitting via prediction confidence and entropy.
        Overconfident predictions can leak training data information.
        
        Args:
            predictions: Model predictions (probabilities)
        
        Returns:
            Score from 0-100 (100 = healthy confidence)
        """
        # Store prediction confidences
        if predictions.dim() > 1:
            # Multi-class: use max probability as confidence
            confidences = predictions.max(dim=-1).values
        else:
            # Binary/regression: use absolute value
            confidences = predictions.abs()
        
        # Store confidences for history
        for conf in confidences.tolist():
            self.prediction_confidences.append(conf)
        
        if len(self.prediction_confidences) > 100:
            self.prediction_confidences = self.prediction_confidences[-100:]
        
        if not self.prediction_confidences:
            return 100.0
        
        # Calculate average confidence
        avg_confidence = sum(self.prediction_confidences) / len(self.prediction_confidences)
        
        # Scoring based on confidence (sweet spot: 0.6-0.9)
        if 0.6 <= avg_confidence <= 0.9:
            # Healthy confidence - good calibration
            return 100.0
        elif 0.5 <= avg_confidence < 0.6 or 0.9 < avg_confidence <= 0.95:
            # Slightly off - acceptable
            return 85.0
        elif 0.4 <= avg_confidence < 0.5:
            # More concerning (underconfident)
            return 70.0
        elif 0.95 < avg_confidence <= 0.98:
            # Overconfident - likely overfitting (moderate)
            return 50.0
        elif avg_confidence > 0.98:
            # Extreme overconfidence - strong signal of leakage
            # Penalize sharply
            penalty = (avg_confidence - 0.98) * 2000.0
            return max(0.0, 100.0 - penalty)
        else:
            # Underconfident - model not learning
            return max(0.0, 100.0 - ((0.4 - avg_confidence) * 200.0))
    
    def _store_training_loss(self, loss: float) -> None:
        """Store training loss for history tracking."""
        self.training_losses.append(loss)
        if len(self.training_losses) > 50:
            self.training_losses = self.training_losses[-50:]
    
    def _store_validation_loss(self, loss: float) -> None:
        """Store validation loss for history tracking."""
        self.validation_losses.append(loss)
        if len(self.validation_losses) > 50:
            self.validation_losses = self.validation_losses[-50:]
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.training_metrics:
            return {
                "total_rounds": 0,
                "total_samples": 0,
                "avg_fitness": 0.0,
            }
        
        return {
            "total_rounds": len(self.training_metrics),
            "total_samples": sum(m.samples_trained for m in self.training_metrics),
            "total_training_time": sum(m.training_time for m in self.training_metrics),
            "avg_loss": sum(m.train_loss for m in self.training_metrics) / len(self.training_metrics),
            "avg_quality": sum(m.quality_score for m in self.training_metrics) / len(self.training_metrics),
            "avg_timeliness": sum(m.timeliness_score for m in self.training_metrics) / len(self.training_metrics),
            "avg_honesty": sum(m.honesty_score for m in self.training_metrics) / len(self.training_metrics),
            "avg_fitness": sum(m.fitness_score for m in self.training_metrics) / len(self.training_metrics),
            "current_genome_id": self.current_genome.genome_id if self.current_genome else None,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GenomeTrainer",
    "TrainingConfig",
]
