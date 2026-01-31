"""
Confidence Scorer - Evaluates Nawal's confidence in its predictions

Uses multiple signals to determine if Nawal can handle a query:
- Softmax entropy (uncertainty in predictions)
- Perplexity (how familiar is the text)
- Length (can Nawal handle this sequence length)
- Language (which Belizean language is this)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Evaluate Nawal's confidence in handling a query
    
    Confidence is computed using multiple factors:
    1. Entropy of output distribution (low entropy = high confidence)
    2. Perplexity of the sequence (low perplexity = familiar text)
    3. Sequence length (shorter = easier to handle)
    4. Language detection (native languages = higher confidence)
    
    Threshold (default 0.75):
    - >= 0.75: Nawal handles the request (sovereign path)
    - < 0.75: Route to DeepSeek teacher (learning opportunity)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        entropy_weight: float = 0.4,
        perplexity_weight: float = 0.3,
        length_weight: float = 0.2,
        language_weight: float = 0.1,
    ):
        self.threshold = confidence_threshold
        self.entropy_weight = entropy_weight
        self.perplexity_weight = perplexity_weight
        self.length_weight = length_weight
        self.language_weight = language_weight
        
        # Supported Belizean languages (higher confidence)
        self.native_languages = ["en", "es", "bzj", "cab", "mop"]
        
        logger.info(f"Initialized ConfidenceScorer with threshold={confidence_threshold}")
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute softmax entropy of predictions
        
        Low entropy = model is confident in its prediction
        High entropy = model is uncertain
        
        Args:
            logits: Model output logits [vocab_size] or [batch, seq_len, vocab_size]
        
        Returns:
            Normalized entropy score (0 = high confidence, 1 = low confidence)
        """
        # Take last position if batch format
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # Compute probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Normalize by max possible entropy (uniform distribution)
        max_entropy = math.log(logits.size(-1))
        normalized_entropy = entropy / max_entropy
        
        # Convert to confidence (low entropy = high confidence)
        confidence = 1.0 - normalized_entropy.mean().item()
        
        return confidence
    
    def compute_perplexity(self, logits: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Compute perplexity of the sequence
        
        Low perplexity = model finds the text familiar
        High perplexity = model struggles with the text
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            target_ids: Target token IDs [batch, seq_len]
        
        Returns:
            Perplexity-based confidence score (0-1, higher is better)
        """
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='mean'
        )
        
        # Compute perplexity: exp(loss)
        perplexity = torch.exp(loss).item()
        
        # Convert to confidence (low perplexity = high confidence)
        # Perplexity typically ranges from 1 to infinity
        # We normalize assuming perplexity > 100 is very uncertain
        confidence = 1.0 / (1.0 + perplexity / 100.0)
        
        return confidence
    
    def compute_length_confidence(self, sequence_length: int, max_length: int = 1024) -> float:
        """
        Compute confidence based on sequence length
        
        Shorter sequences are easier to handle confidently
        
        Args:
            sequence_length: Current sequence length
            max_length: Maximum supported length
        
        Returns:
            Length-based confidence (0-1, higher is better)
        """
        if sequence_length > max_length:
            return 0.0
        
        # Linear decay: 1.0 at length 0, 0.5 at max_length
        confidence = 1.0 - (sequence_length / (2 * max_length))
        return max(0.0, confidence)
    
    def compute_language_confidence(self, detected_language: str) -> float:
        """
        Compute confidence based on detected language
        
        Nawal is trained on Belizean languages, so native languages
        should have higher confidence
        
        Args:
            detected_language: Language code (e.g., "en", "es", "bzj")
        
        Returns:
            Language confidence (0-1)
        """
        if detected_language in self.native_languages:
            return 1.0
        else:
            return 0.5  # Reduced confidence for unknown languages
    
    def compute_confidence(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        detected_language: str = "en",
    ) -> Dict[str, float]:
        """
        Compute overall confidence score using all factors
        
        Args:
            logits: Model output logits
            input_ids: Input token IDs (optional, for perplexity)
            detected_language: Detected language code
        
        Returns:
            Dictionary containing:
                - overall: Overall confidence score (0-1)
                - entropy: Entropy-based confidence
                - perplexity: Perplexity-based confidence (if input_ids provided)
                - length: Length-based confidence
                - language: Language-based confidence
                - should_use_nawal: Boolean decision (>= threshold)
        """
        # Compute individual scores
        entropy_score = self.compute_entropy(logits)
        
        perplexity_score = 0.5  # Default if not provided
        if input_ids is not None:
            perplexity_score = self.compute_perplexity(logits, input_ids)
        
        length = input_ids.size(1) if input_ids is not None else 0
        length_score = self.compute_length_confidence(length)
        
        language_score = self.compute_language_confidence(detected_language)
        
        # Weighted combination
        overall = (
            self.entropy_weight * entropy_score +
            self.perplexity_weight * perplexity_score +
            self.length_weight * length_score +
            self.language_weight * language_score
        )
        
        # Decision: use Nawal if confidence >= threshold
        should_use_nawal = overall >= self.threshold
        
        return {
            "overall": overall,
            "entropy": entropy_score,
            "perplexity": perplexity_score,
            "length": length_score,
            "language": language_score,
            "should_use_nawal": should_use_nawal,
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold dynamically
        
        Over time, as Nawal improves through distillation, we can
        increase the threshold to reduce DeepSeek fallback rate
        
        Args:
            new_threshold: New confidence threshold (0-1)
        """
        old_threshold = self.threshold
        self.threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"Updated confidence threshold: {old_threshold:.3f} -> {self.threshold:.3f}")
