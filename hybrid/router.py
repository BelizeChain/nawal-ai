"""
Intelligent Router - Routes queries between Nawal and DeepSeek based on confidence

Decision logic:
1. Nawal processes query → compute confidence score
2. If confidence >= threshold (0.75): Return Nawal response
3. If confidence < threshold: Route to DeepSeek teacher
4. Log fallback for knowledge distillation training
"""

import torch
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """
    Route queries between Nawal (sovereign) and DeepSeek (teacher)
    
    Routing Strategy:
    - HIGH confidence (>=0.75): Use Nawal → Sovereignty maintained
    - LOW confidence (<0.75): Use DeepSeek → Learning opportunity
    
    Tracks:
    - Routing decisions (for analytics)
    - Fallback queries (for distillation training)
    - Performance metrics (latency, accuracy)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        log_fallbacks: bool = True,
        fallback_log_path: str = "./logs/nawal_fallbacks.jsonl",
    ):
        self.threshold = confidence_threshold
        self.log_fallbacks = log_fallbacks
        self.fallback_log_path = fallback_log_path
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "nawal_handled": 0,
            "deepseek_fallback": 0,
            "sovereignty_rate": 0.0,  # % handled by Nawal
        }
        
        logger.info(f"Initialized IntelligentRouter with threshold={confidence_threshold}")
    
    def route(
        self,
        query: str,
        nawal_logits: torch.Tensor,
        confidence_scores: Dict[str, float],
    ) -> Tuple[str, Dict]:
        """
        Decide whether to use Nawal or DeepSeek
        
        Args:
            query: User input query
            nawal_logits: Nawal's output logits
            confidence_scores: Confidence metrics from ConfidenceScorer
        
        Returns:
            Tuple of (decision, metadata):
                - decision: "nawal" or "deepseek"
                - metadata: Routing information for logging
        """
        self.stats["total_queries"] += 1
        
        # Check confidence threshold
        overall_confidence = confidence_scores["overall"]
        use_nawal = overall_confidence >= self.threshold
        
        decision = "nawal" if use_nawal else "deepseek"
        
        # Update statistics
        if use_nawal:
            self.stats["nawal_handled"] += 1
        else:
            self.stats["deepseek_fallback"] += 1
        
        # Update sovereignty rate
        self.stats["sovereignty_rate"] = (
            self.stats["nawal_handled"] / self.stats["total_queries"]
        )
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "confidence": overall_confidence,
            "threshold": self.threshold,
            "query_preview": query[:100],  # First 100 chars
            "confidence_breakdown": confidence_scores,
        }
        
        # Log fallback for distillation
        if not use_nawal and self.log_fallbacks:
            self._log_fallback(query, metadata)
        
        logger.info(
            f"Routed to {decision.upper()} "
            f"(confidence: {overall_confidence:.3f}, threshold: {self.threshold:.3f})"
        )
        
        return decision, metadata
    
    def _log_fallback(self, query: str, metadata: Dict) -> None:
        """
        Log fallback query for knowledge distillation
        
        These logs are used to:
        1. Train Nawal on DeepSeek's responses (distillation)
        2. Analyze where Nawal needs improvement
        3. Track progress over time
        """
        import json
        import os
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.fallback_log_path), exist_ok=True)
        
        # Prepare log entry
        log_entry = {
            "timestamp": metadata["timestamp"],
            "query": query,
            "confidence": metadata["confidence"],
            "confidence_breakdown": metadata["confidence_breakdown"],
        }
        
        # Append to JSONL file
        with open(self.fallback_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get routing statistics"""
        return {
            **self.stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def reset_statistics(self) -> None:
        """Reset routing statistics"""
        self.stats = {
            "total_queries": 0,
            "nawal_handled": 0,
            "deepseek_fallback": 0,
            "sovereignty_rate": 0.0,
        }
        logger.info("Reset routing statistics")
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold
        
        Over time, as Nawal improves through distillation, we can
        increase the threshold to maintain quality while reducing fallbacks
        
        Args:
            new_threshold: New confidence threshold (0-1)
        """
        old_threshold = self.threshold
        self.threshold = max(0.0, min(1.0, new_threshold))
        logger.info(
            f"Updated routing threshold: {old_threshold:.3f} -> {self.threshold:.3f}\n"
            f"Expected impact: {'More' if new_threshold > old_threshold else 'Fewer'} "
            f"DeepSeek fallbacks"
        )
    
    def export_fallback_logs(self, output_path: str) -> int:
        """
        Export fallback logs for analysis or distillation training
        
        Args:
            output_path: Path to export logs
        
        Returns:
            Number of fallback queries exported
        """
        import shutil
        import os
        
        if os.path.exists(self.fallback_log_path):
            shutil.copy(self.fallback_log_path, output_path)
            count = sum(1 for _ in open(self.fallback_log_path))
            logger.info(f"Exported {count} fallback queries to {output_path}")
            return count
        else:
            logger.warning("No fallback logs found")
            return 0
