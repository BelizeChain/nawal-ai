"""
Hybrid Nawal Engine - Main orchestration layer

Combines:
1. Nawal (sovereign transformer) - Primary model
2. DeepSeek (teacher model) - Fallback + knowledge source
3. Confidence scorer - Decision making
4. Intelligent router - Query routing
5. Distillation trainer - Continuous improvement

Usage:
    >>> from nawal.hybrid import HybridNawalEngine
    >>> engine = HybridNawalEngine()
    >>> response = engine.generate("What is DALLA?")
"""

import torch
from typing import Optional, Dict, List
import logging
from datetime import datetime

from nawal.client.nawal import Nawal, create_nawal
from .confidence import ConfidenceScorer
from .router import IntelligentRouter
from .teacher import DeepSeekTeacher, create_deepseek_teacher

logger = logging.getLogger(__name__)


class HybridNawalEngine:
    """
    Hybrid Nawal Engine - Intelligent routing between sovereign and teacher models
    
    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                   User Query                          │
    └───────────────────┬──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────────────┐
    │           Nawal (Sovereign Model)                     │
    │         - 117M/350M/1.3B parameters                   │
    │         - 100% Belizean ownership                     │
    │         - Trained on national data                    │
    └───────────────────┬──────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────────────┐
    │        Confidence Scorer (Multi-factor)               │
    │  - Entropy (uncertainty)                              │
    │  - Perplexity (familiarity)                           │
    │  - Length (complexity)                                │
    │  - Language (native vs foreign)                       │
    └───────────────────┬──────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────────────┐
    │   Confidence >= 0.75?                           │
    └─────┬───────────────────────────────────────────┴──┐
          │ YES (95% target)          NO (5% target)     │
          ▼                                   ▼
    ┌─────────────┐              ┌───────────────────────┐
    │ Return      │              │ DeepSeek Teacher      │
    │ Nawal       │              │ - 33B parameters      │
    │ Response    │              │ - MIT license         │
    │             │              │ - Coding expert       │
    │ 🇧🇿 Sovereign│              │ - Returns response    │
    └─────────────┘              │ - Logs for distill    │
                                 └──────────┬────────────┘
                                            │
                                            ▼
                                 ┌────────────────────────┐
                                 │ Knowledge Distillation │
                                 │ (Nightly batch job)    │
                                 │ - Improve Nawal        │
                                 │ - Increase threshold   │
                                 └────────────────────────┘
    
    Economic Model:
    - Nawal query: 0.01 DALLA (cheap, sovereign)
    - DeepSeek query: 0.10 DALLA (10x cost, but teaches Nawal)
    - Over time: Nawal handles more → Lower costs → Higher sovereignty
    
    Performance Targets (Month 6):
    - 95% Nawal handling rate (sovereignty)
    - 5% DeepSeek fallback rate (learning)
    - <2s average response time
    - >90% user satisfaction
    """
    
    def __init__(
        self,
        nawal_model: Optional[Nawal] = None,
        teacher_model: Optional[DeepSeekTeacher] = None,
        confidence_threshold: float = 0.75,
        auto_load_teacher: bool = False,
        enable_distillation_logging: bool = True,
    ):
        """
        Initialize Hybrid Nawal Engine
        
        Args:
            nawal_model: Nawal model instance (creates small if None)
            teacher_model: DeepSeek teacher (loads on first fallback if None)
            confidence_threshold: Routing threshold (0-1)
            auto_load_teacher: Load DeepSeek at startup (slower init, faster first fallback)
            enable_distillation_logging: Log fallbacks for training
        """
        # Initialize Nawal (sovereign model)
        self.nawal = nawal_model or create_nawal(size="small")
        logger.info("Nawal sovereign model loaded")
        
        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer(
            confidence_threshold=confidence_threshold
        )
        
        # Initialize router
        self.router = IntelligentRouter(
            confidence_threshold=confidence_threshold,
            log_fallbacks=enable_distillation_logging,
        )
        
        # Initialize teacher (lazy load by default)
        self.teacher = teacher_model
        if auto_load_teacher and self.teacher is None:
            logger.info("Pre-loading DeepSeek teacher model...")
            self.teacher = create_deepseek_teacher()
        
        logger.info(
            f"Hybrid Nawal Engine initialized:\n"
            f"  Nawal size: {self.nawal.config.model_size}\n"
            f"  Confidence threshold: {confidence_threshold}\n"
            f"  Teacher pre-loaded: {self.teacher is not None}\n"
            f"  Distillation logging: {enable_distillation_logging}"
        )
    
    def _ensure_teacher_loaded(self) -> None:
        """Lazy load DeepSeek teacher on first fallback"""
        if self.teacher is None:
            logger.info("Loading DeepSeek teacher for first fallback...")
            self.teacher = create_deepseek_teacher()
            logger.info("DeepSeek teacher loaded")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        detect_language: bool = True,
    ) -> Dict:
        """
        Generate response using hybrid routing
        
        Args:
            prompt: Input query
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            detect_language: Auto-detect language
        
        Returns:
            Dictionary containing:
                - text: Generated response
                - model_used: "nawal" or "deepseek"
                - confidence: Confidence score
                - latency_ms: Response time in milliseconds
                - metadata: Additional routing info
        """
        start_time = datetime.utcnow()
        
        # Detect language
        detected_lang = "en"
        if detect_language:
            detected_lang = self.nawal.language_detector.detect(prompt)
        
        # Get Nawal's prediction
        input_ids = self.nawal.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            nawal_outputs = self.nawal.forward(
                input_ids=input_ids,
                use_cache=False,
            )
        
        nawal_logits = nawal_outputs["logits"]
        
        # Compute confidence
        confidence_scores = self.confidence_scorer.compute_confidence(
            logits=nawal_logits,
            input_ids=input_ids,
            detected_language=detected_lang,
        )
        
        # Route based on confidence
        decision, routing_metadata = self.router.route(
            query=prompt,
            nawal_logits=nawal_logits,
            confidence_scores=confidence_scores,
        )
        
        # Generate response
        if decision == "nawal":
            # Use Nawal (sovereign path)
            responses = self.nawal.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                detect_language=False,  # Already detected
            )
            response_text = responses[0]
            model_used = "nawal"
            
        else:
            # Use DeepSeek (teacher path)
            self._ensure_teacher_loaded()
            teacher_response = self.teacher.generate(
                prompt=prompt,
                max_tokens=max_length,
            )
            response_text = teacher_response["text"]
            model_used = "deepseek"
        
        # Calculate latency
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            "text": response_text,
            "model_used": model_used,
            "confidence": confidence_scores["overall"],
            "confidence_breakdown": confidence_scores,
            "detected_language": detected_lang,
            "latency_ms": latency_ms,
            "metadata": routing_metadata,
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "routing": self.router.get_statistics(),
            "nawal_config": {
                "model_size": self.nawal.config.model_size,
                "num_parameters": self.nawal.config.num_parameters,
                "num_layers": self.nawal.config.num_layers,
            },
            "teacher_loaded": self.teacher is not None,
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold globally
        
        Affects both confidence scorer and router
        """
        self.confidence_scorer.update_threshold(new_threshold)
        self.router.update_threshold(new_threshold)
    
    def export_distillation_data(self, output_path: str) -> int:
        """
        Export logged fallback queries for distillation training
        
        Args:
            output_path: Path to export training data
        
        Returns:
            Number of queries exported
        """
        return self.router.export_fallback_logs(output_path)


# Convenience function
def create_hybrid_engine(
    nawal_size: str = "small",
    confidence_threshold: float = 0.75,
    auto_load_teacher: bool = False,
) -> HybridNawalEngine:
    """
    Create hybrid engine with common configuration
    
    Args:
        nawal_size: Nawal model size ("small", "medium", "large")
        confidence_threshold: Routing threshold (0-1)
        auto_load_teacher: Pre-load DeepSeek at startup
    
    Returns:
        Initialized hybrid engine
    """
    nawal = create_nawal(size=nawal_size)
    
    return HybridNawalEngine(
        nawal_model=nawal,
        confidence_threshold=confidence_threshold,
        auto_load_teacher=auto_load_teacher,
    )


if __name__ == "__main__":
    # Demo usage
    print("🇧🇿 Hybrid Nawal Engine - Sovereign AI with World-Class Fallback")
    print("=" * 70)
    
    # Create engine
    engine = create_hybrid_engine(nawal_size="small", confidence_threshold=0.75)
    
    # Example queries
    queries = [
        "What is the capital of Belize?",
        "Explain DALLA token staking rewards.",
        "Weh di gat fi du fi register wan land?",  # Kriol
        "Implement a quantum circuit for Grover's algorithm.",  # Complex coding
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        result = engine.generate(query, max_length=50)
        print(f"[{result['model_used'].upper()}] {result['text'][:200]}")
        print(f"Confidence: {result['confidence']:.3f} | Latency: {result['latency_ms']:.1f}ms")
    
    # Show statistics
    print("\n" + "=" * 70)
    stats = engine.get_statistics()
    print(f"Nawal Sovereignty Rate: {stats['routing']['sovereignty_rate']*100:.1f}%")
    print(f"Total Queries: {stats['routing']['total_queries']}")
    print(f"Nawal Handled: {stats['routing']['nawal_handled']}")
    print(f"DeepSeek Fallback: {stats['routing']['deepseek_fallback']}")
