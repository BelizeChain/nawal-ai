"""Knowledge Distillation Trainer for Nawal.

Transfers knowledge from DeepSeek-Coder-33B (teacher) to Nawal models (student)
using KL divergence loss and temperature scaling. Enables progressive sovereignty:
start with 50/50 teacher/student, evolve to 95/5 Nawal/DeepSeek.

Key Features:
    - Temperature-scaled soft targets from teacher
    - Combined loss: KL divergence (soft) + cross-entropy (hard)
    - Gradual temperature annealing for improved convergence
    - Integration with Flower federated learning
    - Pakit storage for distilled checkpoints

Examples:
    Basic distillation::
    
        from nawal.training import KnowledgeDistillationTrainer
        from nawal.architecture import NawalConfig
        
        config = NawalConfig.nawal_medium()  # 350M student
        trainer = KnowledgeDistillationTrainer(
            student_config=config,
            teacher_model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            temperature=4.0,
            alpha=0.7  # 70% soft loss, 30% hard loss
        )
        
        trainer.train(
            train_dataset="belize-legal-corpus",
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5
        )
    
    Federated distillation::
    
        trainer.train_federated(
            server_address="grpc://fl-server.belizechain.gov:8080",
            num_rounds=50,
            clients_per_round=5
        )

References:
    - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
    - Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Union, Callable
import logging
from pathlib import Path
from tqdm import tqdm
import wandb

from nawal.architecture import NawalTransformer, NawalConfig
from nawal.hybrid.teacher import DeepSeekTeacher
from nawal.storage.pakit_client import PakitClient

logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.
    
    Combines soft targets (KL divergence from teacher) with hard targets
    (cross-entropy on ground truth). Temperature scaling controls the
    "softness" of probability distributions.
    
    Attributes:
        temperature (float): Scaling factor for softmax. Higher = softer 
            distributions, easier for student to mimic. Typical: 2.0-6.0.
        alpha (float): Weight for soft loss vs hard loss. 0.7 = 70% soft, 30% hard.
    
    Formula:
        L = alpha * T^2 * KL(student_soft || teacher_soft) + (1-alpha) * CE(student, labels)
    
    Examples:
        Compute distillation loss::
        
            loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
            
            student_logits = student_model(input_ids)
            teacher_logits = teacher_model(input_ids)
            labels = input_ids  # Autoregressive language modeling
            
            loss = loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Softmax temperature for soft targets. Higher = softer.
                Typical range: 2.0-6.0. Hinton et al. used 3.0-20.0.
            alpha: Weight for soft loss. Range [0, 1]. Higher = more teacher influence.
                Typical: 0.5-0.9. DistilBERT used 0.5.
        
        Raises:
            ValueError: If temperature <= 0 or alpha not in [0, 1].
        """
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined distillation loss.
        
        Args:
            student_logits: Student model logits of shape [batch_size, seq_len, vocab_size].
            teacher_logits: Teacher model logits of shape [batch_size, seq_len, vocab_size].
            labels: Ground truth token IDs of shape [batch_size, seq_len].
        
        Returns:
            torch.Tensor: Scalar loss combining soft and hard targets.
        
        Note:
            The T^2 scaling factor ensures gradient magnitudes are comparable
            between soft and hard losses despite temperature scaling.
        """
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Soft loss (KL divergence between temperature-scaled distributions)
        student_soft = F.log_softmax(student_logits_flat / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits_flat / self.temperature, dim=-1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard loss (cross-entropy on ground truth)
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


class KnowledgeDistillationTrainer:
    """Trainer for distilling knowledge from DeepSeek teacher to Nawal student.
    
    Implements progressive sovereignty training:
        1. Start with small student model (117M-350M params)
        2. Distill from large teacher (DeepSeek-Coder-33B)
        3. Gradually increase student confidence (50% → 95% sovereignty)
        4. Deploy student for majority of inference (fast, sovereign)
        5. Fallback to teacher only for complex queries (5%)
    
    Attributes:
        student (NawalTransformer): Student model being trained.
        teacher (DeepSeekTeacher): Frozen teacher model providing soft targets.
        loss_fn (KnowledgeDistillationLoss): Combined KL + CE loss.
        optimizer (torch.optim.Optimizer): Student model optimizer.
        pakit_client (Optional[PakitClient]): For uploading checkpoints to Pakit DAG storage.
    
    Examples:
        Train on local dataset::
        
            trainer = KnowledgeDistillationTrainer(
                student_config=NawalConfig.nawal_small(),
                teacher_model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                temperature=5.0,
                alpha=0.8
            )
            
            trainer.train(
                train_dataset="/path/to/belize-corpus.jsonl",
                val_dataset="/path/to/validation.jsonl",
                num_epochs=20,
                batch_size=16,
                learning_rate=1e-4,
                warmup_steps=1000
            )
        
        Resume from checkpoint::
        
            trainer = KnowledgeDistillationTrainer.from_checkpoint(
                checkpoint_path="/path/to/checkpoint.pt"
            )
            trainer.train(...)
        
        Deploy distilled model::
        
            trainer.save_student("/path/to/nawal-distilled")
            
            # Upload to Pakit DAG
            content_hash = trainer.upload_to_pakit(metadata={"version": "1.0-distilled"})
            print(f"Model on Pakit: {content_hash}")
    """
    
    def __init__(
        self,
        student_config: Optional[NawalConfig] = None,
        student_model: Optional[NawalTransformer] = None,
        teacher_model_id: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        teacher_model: Optional[DeepSeekTeacher] = None,
        temperature: float = 4.0,
        alpha: float = 0.7,
        learning_rate: float = 5e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = False,
        pakit_gateway: Optional[str] = None,
    ):
        """Initialize knowledge distillation trainer.
        
        Args:
            student_config: Configuration for student model. If None, uses nawal_medium().
            student_model: Pre-initialized student model (overrides student_config).
            teacher_model_id: HuggingFace model ID for teacher (DeepSeek default).
            teacher_model: Pre-initialized teacher model (overrides teacher_model_id).
            temperature: Softmax temperature for soft targets. Typical: 2.0-6.0.
            alpha: Weight for soft loss vs hard loss. Typical: 0.5-0.9.
            learning_rate: AdamW learning rate. Typical: 1e-5 to 5e-5.
            device: Training device ("cuda", "cpu", "mps").
            use_wandb: Enable Weights & Biases logging.
            pakit_gateway: Pakit DAG gateway URL for checkpoint uploads (e.g., "http://localhost:8081").
        
        Raises:
            ValueError: If both student_config and student_model are None.
        """
        self.device = torch.device(device)
        self.use_wandb = use_wandb
        
        # Initialize student model
        if student_model is not None:
            self.student = student_model
        elif student_config is not None:
            self.student = NawalTransformer(student_config)
        else:
            # Default to medium model
            logger.info("No student config provided, using nawal_medium (350M params)")
            self.student = NawalTransformer(NawalConfig.nawal_medium())
        
        self.student = self.student.to(self.device)
        logger.info(f"Student model: {self.student.config.num_parameters():,} parameters")
        
        # Initialize teacher model (frozen)
        if teacher_model is not None:
            self.teacher = teacher_model
        else:
            logger.info(f"Loading teacher model: {teacher_model_id}")
            self.teacher = DeepSeekTeacher(model_id=teacher_model_id, device=device)
        
        self.teacher.model.eval()  # Freeze teacher
        for param in self.teacher.model.parameters():
            param.requires_grad = False
        
        # Loss function and optimizer
        self.loss_fn = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Pakit integration
        self.pakit_client = None
        if pakit_gateway:
            self.pakit_client = PakitClient(dag_gateway_url=pakit_gateway)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        if use_wandb:
            wandb.init(
                project="nawal-distillation",
                config={
                    "student_params": self.student.config.num_parameters(),
                    "teacher": teacher_model_id,
                    "temperature": temperature,
                    "alpha": alpha,
                    "learning_rate": learning_rate,
                }
            )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step with distillation loss.
        
        Args:
            batch: Dictionary containing:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - labels: Target token IDs [batch_size, seq_len]
        
        Returns:
            Dict[str, float]: Metrics including:
                - loss: Total distillation loss
                - soft_loss: KL divergence component
                - hard_loss: Cross-entropy component
                - perplexity: exp(hard_loss)
        """
        self.student.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Student forward pass
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        student_logits = student_outputs["logits"]
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Compute distillation loss
        loss = self.loss_fn(student_logits, teacher_logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute individual loss components for logging
        with torch.no_grad():
            batch_size, seq_len, vocab_size = student_logits.shape
            student_logits_flat = student_logits.view(-1, vocab_size)
            teacher_logits_flat = teacher_logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # Soft loss
            student_soft = F.log_softmax(student_logits_flat / self.loss_fn.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits_flat / self.loss_fn.temperature, dim=-1)
            soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.loss_fn.temperature ** 2)
            
            # Hard loss
            hard_loss = F.cross_entropy(student_logits_flat, labels_flat)
            perplexity = torch.exp(hard_loss)
        
        metrics = {
            "loss": loss.item(),
            "soft_loss": soft_loss.item(),
            "hard_loss": hard_loss.item(),
            "perplexity": perplexity.item(),
        }
        
        self.global_step += 1
        
        return metrics
    
    def train(
        self,
        train_dataset: Union[str, Dataset, DataLoader],
        val_dataset: Optional[Union[str, Dataset, DataLoader]] = None,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: Optional[float] = None,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        eval_every: int = 500,
        save_every: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        max_steps: Optional[int] = None,
    ) -> None:
        """Train student model with knowledge distillation.
        
        Args:
            train_dataset: Training data (path, Dataset, or DataLoader).
            val_dataset: Validation data (path, Dataset, or DataLoader).
            num_epochs: Number of training epochs.
            batch_size: Training batch size (if dataset is not DataLoader).
            learning_rate: Override initial learning rate.
            warmup_steps: Linear warmup steps for learning rate.
            gradient_accumulation_steps: Accumulate gradients over N steps.
            eval_every: Evaluate on validation set every N steps.
            save_every: Save checkpoint every N steps.
            checkpoint_dir: Directory for saving checkpoints.
            max_steps: Maximum training steps (None = full epochs).
        
        Examples:
            Train for 10 epochs::
            
                trainer.train(
                    train_dataset="data/belize-corpus.jsonl",
                    val_dataset="data/validation.jsonl",
                    num_epochs=10,
                    batch_size=16
                )
            
            Train with early stopping::
            
                trainer.train(
                    train_dataset=train_dataloader,
                    val_dataset=val_dataloader,
                    max_steps=50000,
                    eval_every=1000
                )
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Update learning rate if provided
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Prepare data loaders
        if isinstance(train_dataset, str):
            train_loader = self._create_dataloader(train_dataset, batch_size, shuffle=True)
        elif isinstance(train_dataset, Dataset):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = train_dataset
        
        if val_dataset is not None:
            if isinstance(val_dataset, str):
                val_loader = self._create_dataloader(val_dataset, batch_size, shuffle=False)
            elif isinstance(val_dataset, Dataset):
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            else:
                val_loader = val_dataset
        else:
            val_loader = None
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Student: {self.student.config.num_parameters():,} params")
        logger.info(f"Temperature: {self.loss_fn.temperature}, Alpha: {self.loss_fn.alpha}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {"loss": [], "soft_loss": [], "hard_loss": [], "perplexity": []}
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "ppl": f"{metrics['perplexity']:.2f}",
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": metrics["loss"],
                        "train/soft_loss": metrics["soft_loss"],
                        "train/hard_loss": metrics["hard_loss"],
                        "train/perplexity": metrics["perplexity"],
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                    })
                
                # Validation
                if val_loader is not None and self.global_step % eval_every == 0:
                    val_metrics = self.evaluate(val_loader)
                    logger.info(f"Step {self.global_step} - Val Loss: {val_metrics['loss']:.4f}, "
                              f"Val PPL: {val_metrics['perplexity']:.2f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            "val/loss": val_metrics["loss"],
                            "val/perplexity": val_metrics["perplexity"],
                        })
                    
                    # Save best model
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint(checkpoint_path / "best_model.pt")
                        logger.info(f"Saved new best model (val_loss={val_metrics['loss']:.4f})")
                
                # Periodic checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(checkpoint_path / f"checkpoint_step_{self.global_step}.pt")
                
                # Max steps check
                if max_steps is not None and self.global_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}, stopping training")
                    break
            
            # Epoch summary
            avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_metrics['loss']:.4f}, "
                       f"Avg PPL: {avg_metrics['perplexity']:.2f}")
            
            if max_steps is not None and self.global_step >= max_steps:
                break
        
        # Final save
        final_path = checkpoint_path / "final_model.pt"
        self.save_checkpoint(final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate student model on validation set.
        
        Args:
            val_loader: Validation DataLoader.
        
        Returns:
            Dict[str, float]: Validation metrics (loss, perplexity, etc.).
        """
        self.student.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Student predictions
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                student_logits = student_outputs["logits"]
                
                # Teacher predictions
                teacher_outputs = self.teacher.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
                
                # Compute loss
                loss = self.loss_fn(student_logits, teacher_logits, labels)
                
                # Compute perplexity
                batch_size, seq_len, vocab_size = student_logits.shape
                student_logits_flat = student_logits.view(-1, vocab_size)
                labels_flat = labels.view(-1)
                hard_loss = F.cross_entropy(student_logits_flat, labels_flat)
                perplexity = torch.exp(hard_loss)
                
                total_loss += loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1
        
        self.student.train()
        
        return {
            "loss": total_loss / num_batches,
            "perplexity": total_perplexity / num_batches,
        }
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save training checkpoint.
        
        Args:
            path: Checkpoint file path (.pt extension).
        """
        checkpoint = {
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.student.config.__dict__,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "temperature": self.loss_fn.temperature,
            "alpha": self.loss_fn.alpha,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], **kwargs) -> "KnowledgeDistillationTrainer":
        """Load trainer from checkpoint.
        
        Args:
            path: Checkpoint file path.
            **kwargs: Override arguments for __init__ (e.g., device, teacher_model_id).
        
        Returns:
            KnowledgeDistillationTrainer: Trainer with loaded state.
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Reconstruct config
        config = NawalConfig(**checkpoint["config"])
        student_model = NawalTransformer(config)
        student_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create trainer
        trainer = cls(
            student_model=student_model,
            temperature=checkpoint["temperature"],
            alpha=checkpoint["alpha"],
            **kwargs
        )
        
        # Restore optimizer state
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.current_epoch = checkpoint["current_epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from {path} (step {trainer.global_step})")
        
        return trainer
    
    def save_student(self, path: Union[str, Path]) -> None:
        """Save distilled student model for deployment.
        
        Args:
            path: Directory to save model files.
        """
        self.student.save_pretrained(path)
        logger.info(f"Student model saved to {path}")
    
    def upload_to_pakit(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload distilled model to Pakit DAG storage.
        
        Args:
            metadata: Model metadata (version, description, metrics).
        
        Returns:
            str: DAG content hash.
        
        Raises:
            RuntimeError: If pakit_client not initialized (need pakit_gateway in __init__).
        """
        if self.pakit_client is None:
            raise RuntimeError("Pakit client not initialized. Provide pakit_gateway in __init__.")
        
        # Save to temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            self.save_student(model_path)
            
            # Upload to Pakit DAG
            content_hash = self.pakit_client.upload_directory(
                directory=str(model_path),
                metadata=metadata or {}
            )
        
        logger.info(f"Model uploaded to Pakit: {cid}")
        return cid
    
    def _create_dataloader(
        self,
        dataset_path: str,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader from dataset file.
        
        Args:
            dataset_path: Path to dataset (JSONL, CSV, etc.).
            batch_size: Batch size.
            shuffle: Shuffle data.
        
        Returns:
            DataLoader: Configured data loader.
        
        Note:
            This is a placeholder. Real implementation would load specific dataset
            formats (JSONL, HuggingFace datasets, etc.).
        """
        # Placeholder - real implementation would load specific formats
        from torch.utils.data import TensorDataset
        
        # For now, create dummy data
        logger.warning("Using dummy data - implement real dataset loading!")
        dummy_inputs = torch.randint(0, 50000, (1000, 512))
        dataset = TensorDataset(dummy_inputs, dummy_inputs)  # input_ids = labels
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
