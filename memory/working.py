"""
Working Memory — fast, volatile, bounded short-term store.

Analogous to the prefrontal-cortex working memory buffer:
- Holds the most recent N MemoryRecords in insertion order.
- Items automatically evict from the front when the buffer is full.
- Optional TTL expiry checked on every read.
- Zero external dependencies (pure Python + stdlib).

Typical use:
    - Current conversation turns
    - Scratchpad reasoning steps
    - Per-request tool-call results

Phase 4 note: a QuantumHippocampus overlay will not extend WorkingMemory —
quantum search is reserved for the much-larger EpisodicMemory store.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from memory.interfaces import AbstractMemory, MemoryRecord


class WorkingMemory(AbstractMemory):
    """
    Bounded FIFO working-memory buffer.

    Stores up to *max_size* MemoryRecords in insertion order.
    When the buffer is full the oldest record is evicted to make room.
    Expired records (TTL) are lazily pruned on ``retrieve`` and ``get``.

    Args:
        max_size : Maximum number of records to hold simultaneously.

    Example::

        wm = WorkingMemory(max_size=64)
        wm.store(MemoryRecord(key="t1", content="Hello", embedding=[0.1, 0.2]))
        results = wm.retrieve(query_embedding=[0.1, 0.2], top_k=3)
    """

    def __init__(self, max_size: int = 256) -> None:
        if max_size < 1:
            raise ValueError("max_size must be ≥ 1")
        self.max_size = max_size
        self._store: OrderedDict[str, MemoryRecord] = OrderedDict()

    # ------------------------------------------------------------------ #
    # AbstractMemory implementation                                        #
    # ------------------------------------------------------------------ #

    def store(self, record: MemoryRecord) -> None:
        """
        Add *record* to the buffer, evicting the oldest entry if full.

        If a record with the same key already exists it is updated in-place
        (moved to the end of the insertion order).
        """
        if record.key in self._store:
            del self._store[record.key]
        elif len(self._store) >= self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(f"WorkingMemory evicted key={evicted_key!r} (buffer full)")

        self._store[record.key] = record
        logger.debug(
            f"WorkingMemory stored key={record.key!r}  size={len(self._store)}"
        )

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """
        Return the top-k records ranked by cosine similarity to *query_embedding*.

        Expired records are silently skipped.
        If no record carries an embedding, falls back to returning the
        most-recent *top_k* non-expired records.
        """
        self._prune_expired()

        candidates = [r for r in self._store.values() if self._matches(r, filters)]
        if not candidates:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 0 or np.linalg.norm(q) == 0:
            # No query signal — return most-recent
            return list(reversed(candidates))[:top_k]

        scored: list[tuple[float, MemoryRecord]] = []
        for record in candidates:
            if record.embedding is not None:
                scored.append(
                    (_cosine(q, np.asarray(record.embedding, dtype=np.float32)), record)
                )
            else:
                scored.append((0.0, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def get(self, key: str) -> Optional[MemoryRecord]:
        record = self._store.get(key)
        if record is not None and record.is_expired():
            del self._store[key]
            return None
        return record

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        logger.debug("WorkingMemory cleared")

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    # Convenience API                                                      #
    # ------------------------------------------------------------------ #

    def recent(self, n: int = 10) -> List[MemoryRecord]:
        """Return the *n* most-recently stored records (newest first)."""
        self._prune_expired()
        items = list(self._store.values())
        return list(reversed(items))[:n]

    def snapshot(self) -> List[MemoryRecord]:
        """Return all non-expired records in insertion order."""
        self._prune_expired()
        return list(self._store.values())

    def store_text(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> MemoryRecord:
        """
        Convenience wrapper — store a raw text string as a MemoryRecord.

        Returns the created record (key is auto-generated UUID).
        """
        record = MemoryRecord(
            key=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            ttl=ttl,
        )
        self.store(record)
        return record

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _prune_expired(self) -> None:
        expired = [k for k, r in self._store.items() if r.is_expired()]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug(f"WorkingMemory pruned {len(expired)} expired record(s)")

    @staticmethod
    def _matches(record: MemoryRecord, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        return all(record.metadata.get(k) == v for k, v in filters.items())

    def __repr__(self) -> str:
        return f"WorkingMemory(size={len(self._store)}/{self.max_size})"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity in [−1, 1]. Returns 0.0 if either vector is zero."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
