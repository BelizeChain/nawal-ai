"""
Nawal Hybrid Architecture - Intelligent routing between sovereign Nawal and DeepSeek

This module implements a confidence-based hybrid system where:
1. Nawal (sovereign model) handles most requests (target: 95%)
2. DeepSeek-Coder-33B serves as teacher/fallback (target: 5%)
3. Continuous knowledge distillation improves Nawal over time
"""

from .engine import HybridNawalEngine
from .confidence import ConfidenceScorer
from .router import IntelligentRouter
from .teacher import DeepSeekTeacher
from .distillation import KnowledgeDistillationTrainer

__all__ = [
    "HybridNawalEngine",
    "ConfidenceScorer",
    "IntelligentRouter",
    "DeepSeekTeacher",
    "KnowledgeDistillationTrainer",
]
