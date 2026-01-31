"""
Logging Configuration for Nawal AI.

Provides structured logging with loguru integration.

Author: BelizeChain Team
License: MIT
"""

import sys
from pathlib import Path
from typing import Optional

# Optional loguru library
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger("nawal")


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None,
    serialize: bool = False,
) -> None:
    """
    Configure logging for Nawal AI.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        rotation: Log rotation size/time
        retention: Log retention period
        format_string: Custom format string
        serialize: Whether to serialize logs as JSON
    """
    if not LOGURU_AVAILABLE:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return
    
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        if serialize:
            format_string = "{message}"
        else:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level.upper(),
        colorize=not serialize,
        serialize=serialize,
    )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            format=format_string,
            level=log_level.upper(),
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=serialize,
        )
    
    # Add context
    logger.configure(extra={"component": "nawal"})


def get_logger(name: str):
    """
    Get logger instance for component.
    
    Args:
        name: Component name
    
    Returns:
        Logger instance
    """
    if LOGURU_AVAILABLE:
        return logger.bind(component=name)
    else:
        return logging.getLogger(f"nawal.{name}")


class LogContext:
    """Context manager for structured logging."""
    
    def __init__(self, **kwargs):
        """
        Initialize log context.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context = kwargs
        self.token = None
    
    def __enter__(self):
        if LOGURU_AVAILABLE:
            self.token = logger.contextualize(**self.context)
            self.token.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if LOGURU_AVAILABLE and self.token:
            self.token.__exit__(exc_type, exc_val, exc_tb)


def log_training_start(epochs: int, batch_size: int, learning_rate: float):
    """Log training start."""
    with LogContext(phase="training"):
        logger.info(
            f"Starting training: epochs={epochs}, "
            f"batch_size={batch_size}, lr={learning_rate}"
        )


def log_training_epoch(epoch: int, train_loss: float, train_acc: float,
                      val_loss: float, val_acc: float, epoch_time: float):
    """Log training epoch results."""
    with LogContext(phase="training", epoch=epoch):
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, "
            f"time={epoch_time:.2f}s"
        )


def log_training_complete(best_loss: float, best_acc: float, total_time: float):
    """Log training completion."""
    with LogContext(phase="training"):
        logger.success(
            f"Training complete: "
            f"best_loss={best_loss:.4f}, best_acc={best_acc:.2%}, "
            f"total_time={total_time:.2f}s"
        )


def log_evolution_start(generations: int, population_size: int):
    """Log evolution start."""
    with LogContext(phase="evolution"):
        logger.info(
            f"Starting evolution: generations={generations}, "
            f"population_size={population_size}"
        )


def log_evolution_generation(generation: int, best_fitness: float,
                            avg_fitness: float, generation_time: float):
    """Log evolution generation results."""
    with LogContext(phase="evolution", generation=generation):
        logger.info(
            f"Generation {generation}: "
            f"best={best_fitness:.4f}, avg={avg_fitness:.4f}, "
            f"time={generation_time:.2f}s"
        )


def log_evolution_complete(best_fitness: float, total_generations: int,
                          total_time: float):
    """Log evolution completion."""
    with LogContext(phase="evolution"):
        logger.success(
            f"Evolution complete: "
            f"best_fitness={best_fitness:.4f}, "
            f"generations={total_generations}, "
            f"total_time={total_time:.2f}s"
        )


def log_federated_round_start(round_num: int, num_clients: int):
    """Log federated round start."""
    with LogContext(phase="federated", round=round_num):
        logger.info(
            f"Starting federated round {round_num} with {num_clients} clients"
        )


def log_federated_round_complete(round_num: int, accuracy: float,
                                 round_time: float):
    """Log federated round completion."""
    with LogContext(phase="federated", round=round_num):
        logger.info(
            f"Round {round_num} complete: "
            f"accuracy={accuracy:.2%}, time={round_time:.2f}s"
        )


def log_blockchain_transaction(tx_type: str, success: bool,
                               block_number: Optional[int] = None,
                               tx_time: Optional[float] = None):
    """Log blockchain transaction."""
    with LogContext(phase="blockchain", tx_type=tx_type):
        status = "success" if success else "failed"
        
        message = f"Transaction {status}: type={tx_type}"
        if block_number is not None:
            message += f", block={block_number}"
        if tx_time is not None:
            message += f", time={tx_time:.2f}s"
        
        if success:
            logger.info(message)
        else:
            logger.error(message)


def log_genome_stored(genome_id: str, fitness: float, generation: int):
    """Log genome storage."""
    with LogContext(phase="blockchain", operation="genome_store"):
        logger.info(
            f"Genome stored: id={genome_id[:16]}..., "
            f"fitness={fitness:.4f}, generation={generation}"
        )


def log_validator_registered(address: str, name: str):
    """Log validator registration."""
    with LogContext(phase="blockchain", operation="validator_register"):
        logger.info(
            f"Validator registered: address={address[:16]}..., name={name}"
        )


def log_fitness_submitted(quality: float, timeliness: float, honesty: float,
                         total: float):
    """Log fitness score submission."""
    with LogContext(phase="blockchain", operation="fitness_submit"):
        logger.info(
            f"Fitness submitted: Q={quality:.1f}, T={timeliness:.1f}, "
            f"H={honesty:.1f}, Total={total:.2f}"
        )


def log_error(message: str, exception: Optional[Exception] = None):
    """Log error with optional exception."""
    if exception:
        logger.exception(f"{message}: {exception}")
    else:
        logger.error(message)


def log_warning(message: str):
    """Log warning."""
    logger.warning(message)


def log_debug(message: str):
    """Log debug message."""
    logger.debug(message)
