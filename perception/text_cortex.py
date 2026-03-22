"""
TextCortex — Encode raw text into a dense embedding vector.

This is the primary modality for Nawal's perception layer.  It converts
a raw text string into a fixed-size float vector that lives in Nawal's
hidden embedding space.

Implementation strategy
-----------------------
- Uses a lightweight **hash-based bag-of-n-grams** approach that requires
  NO external model download, ensuring tests always pass.
- When *model_name* is set to a HuggingFace BERT-style model, the cortex
  lazily loads the model on first use and produces proper mean-pool
  sentence embeddings.
- Both paths expose the same ``encode()`` / ``perceive()`` interface.

PhaseHook
---------
Phase 5a fine-tunes a sentence-transformer on Belize domain data.
Replace ``_bert_embed()`` internals — the public API is unchanged.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from perception.interfaces import AbstractCortex, WorldState


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall must can i you he she it we "
    "they me him her us them my your his its our their of in on at to for "
    "and or but not with by from".split()
)


def _hash_token(token: str, dim: int) -> List[float]:
    """
    Map a token to a pseudo-random unit vector using SHA-256.
    Deterministic and consistent across runs.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Use every 4 bytes as one float component, wrapping around digest
    components: List[float] = []
    for i in range(dim):
        byte_idx = (i * 4) % len(digest)
        val = int.from_bytes(digest[byte_idx : byte_idx + 4], "big", signed=True)
        components.append(float(val) / 2**31)
    return components


def _l2_normalize(vec: List[float]) -> List[float]:
    mag = math.sqrt(sum(x * x for x in vec))
    if mag < 1e-9:
        return vec
    return [x / mag for x in vec]


# --------------------------------------------------------------------------- #
# TextCortex                                                                   #
# --------------------------------------------------------------------------- #

class TextCortex(AbstractCortex):
    """
    Text-modality sensory cortex.

    Two modes:
    - ``model_name=None`` (default) — fast hash-bag-of-n-grams, zero deps.
    - ``model_name="bert-base-uncased"`` etc. — HuggingFace mean-pool
      sentence embeddings (lazy-loaded on first call).

    Args:
        embed_dim   : Output embedding dimension (default 256).
        model_name  : HuggingFace model name, or None for hash mode.
        ngram_range : (min_n, max_n) for character n-grams (hash mode only).
        max_length  : Max token count (BERT mode tokenization limit).
        device      : "cpu", "cuda", or "auto".
    """

    # Maximum character length for hash-ngram mode (prevents unbounded n-gram lists)
    MAX_TEXT_LENGTH = 100_000

    def __init__(
        self,
        embed_dim: int = 256,
        model_name: Optional[str] = None,
        ngram_range: tuple[int, int] = (1, 2),
        max_length: int = 512,
        device: str = "auto",
    ) -> None:
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.ngram_range = ngram_range
        self.max_length = max_length
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

        # Lazy-loaded BERT assets
        self._tokenizer: Any = None
        self._model: Any = None
        self._model_loaded = False

        logger.debug(
            f"TextCortex init: dim={embed_dim} mode="
            f"{'bert:' + model_name if model_name else 'hash-ngram'}"
        )

    # ------------------------------------------------------------------ #
    # AbstractCortex interface                                             #
    # ------------------------------------------------------------------ #

    def preprocess(self, raw_input: Any) -> str:
        """Coerce input to a clean text string."""
        if not isinstance(raw_input, str):
            raw_input = str(raw_input)
        # Strip null bytes and control characters (keep newlines/tabs)
        raw_input = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw_input)
        # Collapse whitespace
        text = " ".join(raw_input.split())
        # Enforce length limit for hash-ngram mode (BERT has its own max_length)
        if not self.model_name and len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
        return text

    def encode(self, raw_input: Any) -> List[float]:
        """
        Encode text to a dense float embedding.

        Args:
            raw_input : str (or coercible to str).

        Returns:
            List[float] of length ``embed_dim``, L2-normalised.
        """
        text = self.preprocess(raw_input)
        if not text:
            return [0.0] * self.embed_dim

        if self.model_name:
            return self._bert_embed(text)
        return self._hash_ngram_embed(text)

    def _to_world_state(self, embedding: List[float], raw_input: Any) -> WorldState:
        return WorldState(
            text_embedding=embedding,
            raw_text=raw_input if isinstance(raw_input, str) else str(raw_input),
            metadata={"cortex": "TextCortex", "mode": "bert" if self.model_name else "hash"},
        )

    # ------------------------------------------------------------------ #
    # Hash n-gram mode                                                     #
    # ------------------------------------------------------------------ #

    def _hash_ngram_embed(self, text: str) -> List[float]:
        """
        Bag-of-n-grams embedding using SHA-256 token hashing.

        1. Tokenise (lower-case word tokens).
        2. Drop stopwords.
        3. Build word unigrams + bigrams (per ngram_range).
        4. Sum token vectors → mean-pool → L2-normalise.
        """
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        tokens = [t for t in tokens if t not in _STOP] or tokens  # fallback if all stop

        ngrams: List[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("_".join(tokens[i : i + n]))

        if not ngrams:
            ngrams = tokens  # last resort

        # Accumulate
        acc = [0.0] * self.embed_dim
        for ng in ngrams:
            vec = _hash_token(ng, self.embed_dim)
            for j, v in enumerate(vec):
                acc[j] += v

        # Mean-pool
        acc = [x / len(ngrams) for x in acc]
        return _l2_normalize(acc)

    # ------------------------------------------------------------------ #
    # BERT mean-pool mode (lazy-loaded)                                    #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            logger.info(f"TextCortex: loading model '{self.model_name}'")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._model_loaded = True
        except Exception as exc:
            logger.warning(
                f"TextCortex: failed to load '{self.model_name}': {exc}. "
                "Falling back to hash-ngram mode."
            )
            self.model_name = None
            self._model_loaded = True  # don't retry

    def _bert_embed(self, text: str) -> List[float]:
        """Mean-pool BERT hidden states over non-padding positions."""
        self._load_model()
        if not self.model_name:  # fallback engaged
            return self._hash_ngram_embed(text)

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            out = self._model(**enc)

        # Mean-pool over sequence length (mask out padding)
        hidden = out.last_hidden_state          # [1, seq, hidden]
        mask   = enc["attention_mask"].unsqueeze(-1).float()  # [1, seq, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1) # [1, hidden]
        vec = pooled[0].cpu().tolist()

        # Project to embed_dim if mismatch
        if len(vec) != self.embed_dim:
            vec = _project(vec, self.embed_dim)

        return _l2_normalize(vec)


# --------------------------------------------------------------------------- #
# Utility: linear projection (dimension mismatch)                             #
# --------------------------------------------------------------------------- #

def _project(vec: List[float], out_dim: int) -> List[float]:
    """Truncate or repeat-tile a vector to *out_dim*."""
    if len(vec) >= out_dim:
        return vec[:out_dim]
    ratio = out_dim / len(vec)
    result: List[float] = []
    for i in range(out_dim):
        result.append(vec[int(i / ratio)])
    return result
