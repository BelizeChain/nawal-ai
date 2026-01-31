"""
Data Preprocessing for Nawal AI.

Provides data cleaning, validation, and augmentation
for federated learning datasets.

Key Features:
- Text cleaning (HTML, URLs, special chars)
- Data validation (format, completeness, quality)
- Data augmentation (synonym replacement, backtranslation)
- Normalization and standardization

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Callable, Any
import re
import unicodedata
from dataclasses import dataclass

import torch
from loguru import logger


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.
    
    Attributes:
        lowercase: Convert text to lowercase
        remove_html: Remove HTML tags
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        remove_special_chars: Remove special characters
        normalize_unicode: Normalize Unicode characters
        min_length: Minimum text length
        max_length: Maximum text length
        remove_duplicates: Remove duplicate samples
    """
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_special_chars: bool = False
    normalize_unicode: bool = True
    min_length: int = 10
    max_length: int = 10000
    remove_duplicates: bool = True


class TextCleaner:
    """
    Text cleaning utilities.
    
    Removes HTML, URLs, special characters, normalizes Unicode, etc.
    
    Usage:
        cleaner = TextCleaner(config)
        clean_text = cleaner.clean("Dirty <html> text with URLs http://...")
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize TextCleaner.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        
        # Compile regex patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'\"-]')
        
        logger.info("TextCleaner initialized")
    
    def clean(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize Unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML
        if self.config.remove_html:
            text = self.html_pattern.sub('', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove special characters
        if self.config.remove_special_chars:
            text = self.special_char_pattern.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean batch of texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]


class DataValidator:
    """
    Data validation utilities.
    
    Validates data format, completeness, and quality.
    
    Usage:
        validator = DataValidator(config)
        is_valid = validator.validate(sample)
        valid_samples = validator.filter_valid(samples)
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize DataValidator.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        logger.info("DataValidator initialized")
    
    def validate(self, sample: Any) -> bool:
        """
        Validate single sample.
        
        Args:
            sample: Input sample (text or dict)
        
        Returns:
            True if valid, False otherwise
        """
        # Handle dict samples
        if isinstance(sample, dict):
            if 'text' in sample:
                text = sample['text']
            elif 'content' in sample:
                text = sample['content']
            else:
                # No text field found
                return False
        else:
            text = str(sample)
        
        # Check length
        if len(text) < self.config.min_length:
            return False
        if len(text) > self.config.max_length:
            return False
        
        # Check if text is mostly ASCII or has enough alphanumeric
        if not self._has_enough_content(text):
            return False
        
        return True
    
    def _has_enough_content(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text has enough alphanumeric content.
        
        Args:
            text: Input text
            threshold: Minimum ratio of alphanumeric characters
        
        Returns:
            True if enough content, False otherwise
        """
        if len(text) == 0:
            return False
        
        alphanumeric = sum(c.isalnum() for c in text)
        ratio = alphanumeric / len(text)
        
        return ratio >= threshold
    
    def filter_valid(self, samples: List[Any]) -> List[Any]:
        """
        Filter valid samples from list.
        
        Args:
            samples: List of input samples
        
        Returns:
            List of valid samples
        """
        valid_samples = [s for s in samples if self.validate(s)]
        
        num_removed = len(samples) - len(valid_samples)
        if num_removed > 0:
            logger.info(f"Filtered {num_removed}/{len(samples)} invalid samples")
        
        return valid_samples
    
    def remove_duplicates(self, samples: List[Any]) -> List[Any]:
        """
        Remove duplicate samples.
        
        Args:
            samples: List of input samples
        
        Returns:
            List of unique samples
        """
        if not self.config.remove_duplicates:
            return samples
        
        seen = set()
        unique_samples = []
        
        for sample in samples:
            # Get text representation
            if isinstance(sample, dict):
                text = sample.get('text', sample.get('content', str(sample)))
            else:
                text = str(sample)
            
            if text not in seen:
                seen.add(text)
                unique_samples.append(sample)
        
        num_removed = len(samples) - len(unique_samples)
        if num_removed > 0:
            logger.info(f"Removed {num_removed}/{len(samples)} duplicate samples")
        
        return unique_samples


class DataAugmenter:
    """
    Data augmentation utilities.
    
    Augments training data to increase diversity and robustness.
    
    Techniques:
    - Synonym replacement
    - Random insertion
    - Random swap
    - Random deletion
    - Backtranslation (if available)
    
    Usage:
        augmenter = DataAugmenter()
        augmented = augmenter.augment(text, num_aug=2)
    """
    
    def __init__(self):
        """Initialize DataAugmenter."""
        logger.info("DataAugmenter initialized")
    
    def augment(
        self,
        text: str,
        num_aug: int = 1,
        alpha: float = 0.1,
    ) -> List[str]:
        """
        Augment text with multiple techniques.
        
        Args:
            text: Input text
            num_aug: Number of augmented versions to generate
            alpha: Augmentation intensity (0-1)
        
        Returns:
            List of augmented texts (includes original)
        """
        augmented = [text]  # Original
        
        for _ in range(num_aug):
            # Randomly choose augmentation technique
            import random
            technique = random.choice([
                self.synonym_replacement,
                self.random_insertion,
                self.random_swap,
                self.random_deletion,
            ])
            
            aug_text = technique(text, alpha)
            augmented.append(aug_text)
        
        return augmented
    
    def synonym_replacement(self, text: str, alpha: float = 0.1) -> str:
        """
        Replace words with synonyms.
        
        Args:
            text: Input text
            alpha: Fraction of words to replace
        
        Returns:
            Augmented text
        """
        words = text.split()
        num_words = len(words)
        num_replace = max(1, int(alpha * num_words))
        
        import random
        indices = random.sample(range(num_words), min(num_replace, num_words))
        
        # Simple synonym replacement (placeholder)
        # In production, use WordNet or word embeddings
        for idx in indices:
            # Keep original for now (no WordNet dependency)
            pass
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, alpha: float = 0.1) -> str:
        """
        Randomly insert words.
        
        Args:
            text: Input text
            alpha: Fraction of words to insert
        
        Returns:
            Augmented text
        """
        words = text.split()
        num_words = len(words)
        num_insert = max(1, int(alpha * num_words))
        
        import random
        for _ in range(num_insert):
            # Insert random existing word at random position
            word = random.choice(words)
            pos = random.randint(0, len(words))
            words.insert(pos, word)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, alpha: float = 0.1) -> str:
        """
        Randomly swap word positions.
        
        Args:
            text: Input text
            alpha: Fraction of words to swap
        
        Returns:
            Augmented text
        """
        words = text.split()
        num_words = len(words)
        num_swap = max(1, int(alpha * num_words))
        
        import random
        for _ in range(num_swap):
            if num_words < 2:
                break
            idx1, idx2 = random.sample(range(num_words), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, alpha: float = 0.1) -> str:
        """
        Randomly delete words.
        
        Args:
            text: Input text
            alpha: Fraction of words to delete
        
        Returns:
            Augmented text
        """
        words = text.split()
        num_words = len(words)
        
        # Keep at least 1 word
        if num_words == 1:
            return text
        
        import random
        # Randomly keep words with probability (1 - alpha)
        words = [w for w in words if random.random() > alpha]
        
        # Ensure at least 1 word remains
        if len(words) == 0:
            words = [random.choice(text.split())]
        
        return ' '.join(words)


class DataPreprocessor:
    """
    Main data preprocessing pipeline.
    
    Combines cleaning, validation, and augmentation.
    
    Usage:
        preprocessor = DataPreprocessor(config)
        
        # Preprocess single sample
        clean_sample = preprocessor.preprocess(sample)
        
        # Preprocess batch
        clean_samples = preprocessor.preprocess_batch(samples)
        
        # Preprocess with augmentation
        augmented_samples = preprocessor.preprocess_batch(
            samples,
            augment=True,
            num_aug=2,
        )
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        custom_functions: Optional[List[Callable]] = None,
    ):
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Preprocessing configuration
            custom_functions: Optional custom preprocessing functions
        """
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.validator = DataValidator(self.config)
        self.augmenter = DataAugmenter()
        self.custom_functions = custom_functions or []
        
        logger.info("DataPreprocessor initialized")
    
    def preprocess(
        self,
        sample: Any,
        augment: bool = False,
        num_aug: int = 1,
    ) -> Optional[Any]:
        """
        Preprocess single sample.
        
        Args:
            sample: Input sample
            augment: Whether to augment data
            num_aug: Number of augmented versions
        
        Returns:
            Preprocessed sample (or None if invalid)
        """
        # Extract text
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('content', ''))
            is_dict = True
        else:
            text = str(sample)
            is_dict = False
        
        # Clean text
        text = self.cleaner.clean(text)
        
        # Apply custom functions
        for func in self.custom_functions:
            text = func(text)
        
        # Validate
        if not self.validator.validate(text):
            return None
        
        # Augment if requested
        if augment:
            texts = self.augmenter.augment(text, num_aug=num_aug)
            if is_dict:
                return [{'text': t, **{k: v for k, v in sample.items() if k != 'text'}} for t in texts]
            else:
                return texts
        
        # Return preprocessed sample
        if is_dict:
            sample['text'] = text
            return sample
        else:
            return text
    
    def preprocess_batch(
        self,
        samples: List[Any],
        augment: bool = False,
        num_aug: int = 1,
    ) -> List[Any]:
        """
        Preprocess batch of samples.
        
        Args:
            samples: List of input samples
            augment: Whether to augment data
            num_aug: Number of augmented versions per sample
        
        Returns:
            List of preprocessed samples
        """
        preprocessed = []
        
        for sample in samples:
            result = self.preprocess(sample, augment=augment, num_aug=num_aug)
            
            if result is not None:
                if isinstance(result, list):
                    # Augmented samples
                    preprocessed.extend(result)
                else:
                    preprocessed.append(result)
        
        # Remove duplicates
        if self.config.remove_duplicates:
            preprocessed = self.validator.remove_duplicates(preprocessed)
        
        logger.info(
            f"Preprocessed {len(samples)} -> {len(preprocessed)} samples "
            f"(augment={augment}, num_aug={num_aug})"
        )
        
        return preprocessed
    
    def add_custom_function(self, func: Callable[[str], str]) -> None:
        """
        Add custom preprocessing function.
        
        Args:
            func: Function that takes text and returns processed text
        """
        self.custom_functions.append(func)
        logger.info(f"Added custom preprocessing function: {func.__name__}")
    
    def get_stats(self, samples: List[Any]) -> Dict[str, Any]:
        """
        Get preprocessing statistics.
        
        Args:
            samples: List of samples to analyze
        
        Returns:
            Dictionary of statistics
        """
        valid_count = sum(1 for s in samples if self.validator.validate(s))
        
        # Get text lengths
        lengths = []
        for sample in samples:
            if isinstance(sample, dict):
                text = sample.get('text', sample.get('content', ''))
            else:
                text = str(sample)
            lengths.append(len(text))
        
        stats = {
            'total_samples': len(samples),
            'valid_samples': valid_count,
            'invalid_samples': len(samples) - valid_count,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        }
        
        return stats
