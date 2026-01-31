"""Training utilities for Nawal knowledge distillation and federated learning."""

from .distillation import (
    KnowledgeDistillationLoss,
    KnowledgeDistillationTrainer,
)

__all__ = [
    "KnowledgeDistillationLoss",
    "KnowledgeDistillationTrainer",
]
