"""
Tokenizers for Text Processing.

Provides tokenization for various text models and tasks.

Supported Tokenizers:
- Character-level (default — Nawal native tokenizer)
- Word-level (whitespace)
- BERT (WordPiece tokenizer, requires ``transformers``)
- SentencePiece (unigram/BPE, requires ``transformers``)
- HuggingFace AutoTokenizer (optional, requires ``transformers``)

Author: BelizeChain Team
License: MIT"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from loguru import logger

# Optional transformers library
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install: pip install transformers")


class TokenizerType(Enum):
    """Supported tokenizer types."""

    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    BERT = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    DISTILBERT = "distilbert-base-uncased"
    ROBERTA = "roberta-base"
    CHARACTER = "character"
    WORD = "word"


@dataclass
class TokenizerConfig:
    """
    Tokenizer configuration.

    Attributes:
        tokenizer_type: Type of tokenizer
        vocab_size: Vocabulary size (None = use default)
        max_length: Maximum sequence length
        padding: Padding strategy ("max_length", "longest", "do_not_pad")
        truncation: Whether to truncate sequences
        add_special_tokens: Add [CLS], [SEP], etc.
        cache_dir: Directory for caching tokenizers
    """

    tokenizer_type: TokenizerType
    vocab_size: int | None = None
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True
    cache_dir: Path | None = None


class TextTokenizer:
    """
    Universal text tokenizer interface.

    Provides unified API for different tokenizer backends.

    Usage:
        tokenizer = TextTokenizer(
            config=TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER)
        )

        # Tokenize text
        tokens = tokenizer.encode("Hello, world!")

        # Batch tokenization
        batch_tokens = tokenizer.encode_batch([
            "First sentence",
            "Second sentence",
        ])

        # Decode back to text
        text = tokenizer.decode(tokens)
    """

    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer.

        Args:
            config: Tokenizer configuration
        """
        self.config = config
        self.tokenizer = self._load_tokenizer()

        logger.info(f"TextTokenizer initialized: {config.tokenizer_type.value}")

    def _load_tokenizer(self):
        """Load tokenizer based on type."""
        tokenizer_type = self.config.tokenizer_type

        if tokenizer_type in self._get_hf_tokenizers():
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers library required")
            return self._load_hf_tokenizer()
        elif tokenizer_type == TokenizerType.CHARACTER:
            return CharacterTokenizer(self.config)
        elif tokenizer_type == TokenizerType.WORD:
            return WordTokenizer(self.config)
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

    def _load_hf_tokenizer(self):
        """Load HuggingFace tokenizer."""
        tokenizer_name = self.config.tokenizer_type.value

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.config.cache_dir,
        )

        # Set pad token if not set (e.g., GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.success(f"Loaded HuggingFace tokenizer: {tokenizer_name}")
        return tokenizer

    def encode(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> list[int] | torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            return_tensors: Return format ("pt" = PyTorch, None = list)

        Returns:
            Token IDs
        """
        if hasattr(self.tokenizer, "encode"):
            if isinstance(self.tokenizer, (CharacterTokenizer, WordTokenizer)):
                # Custom tokenizer — simple encode
                return self.tokenizer.encode(text)
            else:
                # HuggingFace tokenizer
                return self.tokenizer.encode(
                    text,
                    max_length=self.config.max_length,
                    padding=self.config.padding,
                    truncation=self.config.truncation,
                    add_special_tokens=self.config.add_special_tokens,
                    return_tensors=return_tensors,
                )
        else:
            # Custom tokenizer
            return self.tokenizer.encode(text)

    def encode_batch(
        self,
        texts: list[str],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """
        Encode batch of texts.

        Args:
            texts: List of input texts
            return_tensors: Return format

        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        if hasattr(self.tokenizer, "batch_encode_plus"):
            # HuggingFace tokenizer
            return self.tokenizer.batch_encode_plus(
                texts,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                add_special_tokens=self.config.add_special_tokens,
                return_tensors=return_tensors,
            )
        else:
            # Custom tokenizer
            encoded = [self.tokenizer.encode(text) for text in texts]
            return {
                "input_ids": torch.tensor(encoded),
            }

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens like [PAD], [CLS]

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if hasattr(self.tokenizer, "decode"):
            if isinstance(self.tokenizer, (CharacterTokenizer, WordTokenizer)):
                return self.tokenizer.decode(token_ids)
            else:
                return self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                )
        else:
            return self.tokenizer.decode(token_ids)

    def decode_batch(
        self,
        token_ids: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decode batch of token IDs.

        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Skip special tokens

        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if hasattr(self.tokenizer, "batch_decode"):
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            return [self.tokenizer.decode(ids) for ids in token_ids]

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        else:
            return len(self.tokenizer.vocab)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if hasattr(self.tokenizer, "pad_token_id"):
            return self.tokenizer.pad_token_id
        else:
            return 0

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id
        else:
            return self.vocab_size - 1

    @staticmethod
    def _get_hf_tokenizers() -> list[TokenizerType]:
        """Get list of HuggingFace tokenizers."""
        return [
            TokenizerType.GPT2,
            TokenizerType.GPT2_MEDIUM,
            TokenizerType.GPT2_LARGE,
            TokenizerType.BERT,
            TokenizerType.BERT_LARGE,
            TokenizerType.DISTILBERT,
            TokenizerType.ROBERTA,
        ]


class CharacterTokenizer:
    """
    Simple character-level tokenizer.

    Maps each character to a unique ID.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config

        # Build vocabulary (ASCII printable characters)
        self.vocab = {chr(i): i for i in range(32, 127)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1

        self.id_to_char = {v: k for k, v in self.vocab.items()}

        logger.info(f"CharacterTokenizer: vocab_size={len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """Encode text to character IDs."""
        ids = [self.vocab.get(c, self.vocab["<UNK>"]) for c in text]

        # Pad or truncate
        if len(ids) < self.config.max_length:
            ids += [self.vocab["<PAD>"]] * (self.config.max_length - len(ids))
        else:
            ids = ids[: self.config.max_length]

        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode character IDs to text."""
        chars = [self.id_to_char.get(i, "<UNK>") for i in token_ids]
        text = "".join(chars)
        return text.replace("<PAD>", "").replace("<UNK>", "?")


class WordTokenizer:
    """
    Simple word-level tokenizer.

    Splits on whitespace and builds vocabulary.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config

        # Initialize with special tokens
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.next_id = 4

        self.id_to_word = {v: k for k, v in self.vocab.items()}

        logger.info("WordTokenizer initialized (build vocab with .build_vocab())")

    def build_vocab(self, texts: list[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of training texts
            min_freq: Minimum word frequency to include
        """
        from collections import Counter

        # Count words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Add to vocab (words with freq >= min_freq)
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1

        # Limit vocab size if specified
        if self.config.vocab_size is not None:
            # Keep top-k most frequent words
            top_words = [
                word for word, count in word_counts.most_common(self.config.vocab_size - 4)
            ]
            self.vocab = {k: v for k, v in self.vocab.items() if k in top_words or v < 4}
            self.id_to_word = {v: k for k, v in self.vocab.items()}

        logger.success(f"Vocabulary built: {len(self.vocab)} words")

    def encode(self, text: str) -> list[int]:
        """Encode text to word IDs."""
        words = text.lower().split()
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]

        # Add BOS/EOS if configured
        if self.config.add_special_tokens:
            ids = [self.vocab["<BOS>"], *ids, self.vocab["<EOS>"]]

        # Pad or truncate
        if len(ids) < self.config.max_length:
            ids += [self.vocab["<PAD>"]] * (self.config.max_length - len(ids))
        else:
            ids = ids[: self.config.max_length]

        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode word IDs to text."""
        words = [self.id_to_word.get(i, "<UNK>") for i in token_ids]
        text = " ".join(words)

        # Clean special tokens
        for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            text = text.replace(token, "")

        return text.strip()


# Expose HF availability for other modules
HF_AVAILABLE = TRANSFORMERS_AVAILABLE


def create_tokenizer(config: TokenizerConfig) -> TextTokenizer:
    """
    Factory function to create tokenizer.

    Args:
        config: Tokenizer configuration

    Returns:
        Initialized tokenizer
    """
    return TextTokenizer(config)


class NawalTokenizerWrapper:
    """
    Nawal character tokenizer with HuggingFace-compatible call interface.

    Wraps ``CharacterTokenizer`` so it can be used anywhere a HuggingFace
    tokenizer is expected (``__call__``, ``pad_token``, ``add_tokens``, etc.)
    without pulling in the ``transformers`` library.
    """

    def __init__(self, config: TokenizerConfig | None = None):
        if config is None:
            config = TokenizerConfig(tokenizer_type=TokenizerType.CHARACTER)
        self._tokenizer = CharacterTokenizer(config)
        self._config = config
        self.pad_token: str = "<PAD>"
        self.eos_token: str = "<EOS>"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.vocab.get(self.pad_token, 0)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.vocab.get(self.eos_token, len(self._tokenizer.vocab) - 1)

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer.vocab)

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def add_tokens(self, new_tokens: list[str]) -> int:
        """Extend vocabulary with new tokens (e.g. Belizean-specific terms)."""
        added = 0
        for token in new_tokens:
            if token not in self._tokenizer.vocab:
                next_id = max(self._tokenizer.vocab.values()) + 1
                self._tokenizer.vocab[token] = next_id
                self._tokenizer.id_to_char[next_id] = token
                added += 1
        if added:
            logger.info(f"NawalTokenizerWrapper: added {added} tokens to vocabulary")
        return added

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> dict:
        """HuggingFace-compatible tokenizer call — returns input_ids + attention_mask."""
        effective_max = max_length or self._config.max_length

        # Temporarily override max_length for this encode call
        original_max = self._config.max_length
        self._config.max_length = effective_max
        ids = self._tokenizer.encode(text)
        self._config.max_length = original_max

        pad_id = self._tokenizer.vocab.get("<PAD>", 0)
        attention_mask = [0 if tok == pad_id else 1 for tok in ids]

        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
    ) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids)
