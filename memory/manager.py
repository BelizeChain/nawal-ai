"""
Memory Manager — unified access point for all three memory stores.

The MemoryManager is the single entry-point used by the Orchestrator,
Core Cortex, and Control layers to read and write memory.

Architecture::

    ┌────────────────────────────────────────────────────────┐
    │                    MemoryManager                       │
    │                                                        │
    │  ┌───────────────┐  ┌───────────────┐  ┌───────────┐  │
    │  │ WorkingMemory │  │EpisodicMemory │  │SemanticMem│  │
    │  │ (fast FIFO)   │  │ (vector DB)   │  │(knowledge │  │
    │  │               │  │               │  │  graph)   │  │
    │  └───────────────┘  └───────────────┘  └───────────┘  │
    └────────────────────────────────────────────────────────┘

Routing rules (via MemoryRecord.metadata["store"]):
  "working"  → WorkingMemory  (default if unspecified)
  "episodic" → EpisodicMemory
  "semantic" → SemanticMemory
  "all"      → store in all three

Memory consolidation:
  ``consolidate()`` moves working-memory items that have been held for
  longer than *consolidation_age* seconds into episodic memory — mimicking
  hippocampal replay that converts short-term to long-term memory.

Usage::

    from memory.manager import MemoryManager, MemoryRecord

    mm = MemoryManager()

    # Store a conversation turn in working memory
    mm.store_text("User said: Hello", store="working",
                  metadata={"role": "user", "session": "s1"})

    # Store a knowledge fact in semantic memory
    mm.store_text("Python is a programming language", store="semantic",
                  metadata={"type": "fact"})

    # Retrieve across all stores
    results = mm.retrieve(query_embedding=emb, top_k=10)

    # Consolidate (call periodically or at session end)
    mm.consolidate()
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory.interfaces import AbstractMemory, MemoryRecord
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory


# --------------------------------------------------------------------------- #
# MemoryManager                                                                #
# --------------------------------------------------------------------------- #

class MemoryManager:
    """
    Unified facade over Working, Episodic, and Semantic memory stores.

    Args:
        working_max_size     : Max records in working memory buffer.
        episodic_persist_path: Directory for episodic DB persistence.
        semantic_persist_path: File path for semantic graph persistence.
        consolidation_age    : Seconds a working-memory item must be held
                               before being eligible for consolidation to
                               episodic storage (default: 300 s = 5 min).
        embedding_dim        : Expected embedding dimension.
        qdrant_url           : Optional Qdrant server URL for episodic store.
    """

    # Store routing key
    STORE_WORKING  = "working"
    STORE_EPISODIC = "episodic"
    STORE_SEMANTIC = "semantic"
    STORE_ALL      = "all"

    def __init__(
        self,
        working_max_size: int = 256,
        episodic_persist_path: Optional[str] = "./data/episodic_db",
        semantic_persist_path: Optional[str] = None,
        consolidation_age: float = 300.0,
        embedding_dim: int = 768,
        qdrant_url: Optional[str] = None,
    ) -> None:
        self.consolidation_age = consolidation_age
        self.embedding_dim = embedding_dim

        self.working = WorkingMemory(max_size=working_max_size)
        self.episodic = EpisodicMemory(
            persist_path=episodic_persist_path,
            embedding_dim=embedding_dim,
            qdrant_url=qdrant_url,
        )
        self.semantic = SemanticMemory(
            persist_path=semantic_persist_path,
        )

        logger.info(
            f"MemoryManager initialised: "
            f"working(max={working_max_size}), "
            f"episodic(backend={self.episodic._backend!r}), "
            f"semantic(nx={self.semantic._graph is not None})"
        )

    # ------------------------------------------------------------------ #
    # Write API                                                            #
    # ------------------------------------------------------------------ #

    def store(self, record: MemoryRecord, store: Optional[str] = None) -> None:
        """
        Store *record* in the appropriate memory subsystem(s).

        Routing priority:
          1. *store* argument (explicit override)
          2. ``record.metadata["store"]`` (caller-specified)
          3. Default: working memory

        Args:
            record : The MemoryRecord to persist.
            store  : One of "working", "episodic", "semantic", "all".
        """
        target = store or record.metadata.get("store", self.STORE_WORKING)

        if target == self.STORE_ALL:
            self.working.store(record)
            self.episodic.store(record)
            self.semantic.store(record)
        elif target == self.STORE_EPISODIC:
            self.episodic.store(record)
        elif target == self.STORE_SEMANTIC:
            self.semantic.store(record)
        else:  # default → working
            self.working.store(record)

    def store_text(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store: str = STORE_WORKING,
        ttl: Optional[float] = None,
        key: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Convenience wrapper — wrap a text string in a MemoryRecord and store it.

        Returns the created MemoryRecord (key is auto-generated UUID if not given).
        """
        record = MemoryRecord(
            key=key or str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata={**(metadata or {}), "store": store},
            ttl=ttl,
        )
        self.store(record, store=store)
        return record

    # ------------------------------------------------------------------ #
    # Read API                                                             #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        stores: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        dedup: bool = True,
    ) -> List[MemoryRecord]:
        """
        Query one or more memory stores and merge results.

        Args:
            query_embedding : Dense query vector.
            top_k           : Total results to return.
            stores          : Which stores to query. Defaults to all three.
            filters         : Metadata filter applied to each store.
            dedup           : If True, deduplicate by key across stores.

        Returns:
            Merged list sorted by descending relevance, up to *top_k* items.
        """
        if stores is None:
            stores = [self.STORE_WORKING, self.STORE_EPISODIC, self.STORE_SEMANTIC]

        q = np.asarray(query_embedding, dtype=np.float32)
        gathered: List[Tuple[float, MemoryRecord]] = []
        seen_keys: set[str] = set()

        for store_name in stores:
            store_obj: AbstractMemory = self._store_by_name(store_name)
            try:
                results = store_obj.retrieve(query_embedding, top_k=top_k, filters=filters)
            except Exception as exc:
                logger.warning(f"MemoryManager retrieve error from {store_name!r}: {exc}")
                continue
            for rec in results:
                if dedup and rec.key in seen_keys:
                    continue
                seen_keys.add(rec.key)
                sim = _cosine(q, np.asarray(rec.embedding, dtype=np.float32)) \
                      if rec.embedding else 0.0
                gathered.append((sim, rec))

        gathered.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in gathered[:top_k]]

    def get(self, key: str) -> Optional[MemoryRecord]:
        """Exact-key lookup — checks working, then episodic, then semantic."""
        for store in (self.working, self.episodic, self.semantic):
            rec = store.get(key)
            if rec is not None:
                return rec
        return None

    def context_window(
        self,
        query_embedding: Optional[List[float]] = None,
        n_recent: int = 8,
        n_episodic: int = 4,
        n_semantic: int = 3,
    ) -> List[MemoryRecord]:
        """
        Build a combined context window for the Core Cortex.

        Returns a unified ordered list:
          [recent working-memory turns] + [relevant episodic memories]
          + [relevant semantic facts]

        This is the primary read path for the Core Cortex at inference time.

        Args:
            query_embedding : If provided, ranks episodic/semantic by similarity.
            n_recent        : Working-memory items to include (most recent).
            n_episodic      : Episodic memories to retrieve.
            n_semantic      : Semantic facts to retrieve.
        """
        context: List[MemoryRecord] = []

        # 1. Most-recent working memory
        context.extend(self.working.recent(n_recent))

        # 2. Relevant episodic memories
        if query_embedding and n_episodic > 0:
            epi = self.episodic.retrieve(query_embedding, top_k=n_episodic)
            context.extend(epi)

        # 3. Relevant semantic facts
        if query_embedding and n_semantic > 0:
            sem = self.semantic.retrieve(query_embedding, top_k=n_semantic)
            context.extend(sem)

        return context

    # ------------------------------------------------------------------ #
    # Consolidation                                                        #
    # ------------------------------------------------------------------ #

    def consolidate(self, max_items: int = 64) -> int:
        """
        Move mature working-memory items into episodic storage.

        Items that have been in working memory for longer than
        ``self.consolidation_age`` seconds are transferred to the
        episodic store and removed from working memory.

        Args:
            max_items : Maximum number of items to consolidate per call.

        Returns:
            Number of items consolidated.
        """
        import time
        now = time.time()
        to_consolidate: List[MemoryRecord] = []

        for rec in self.working.snapshot():
            age = now - rec.timestamp
            if age >= self.consolidation_age:
                to_consolidate.append(rec)
            if len(to_consolidate) >= max_items:
                break

        for rec in to_consolidate:
            self.episodic.store(rec)
            self.working.delete(rec.key)
            logger.debug(f"Consolidated key={rec.key!r} from working → episodic")

        if to_consolidate:
            logger.info(f"MemoryManager consolidated {len(to_consolidate)} item(s)")
        return len(to_consolidate)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """Return a summary of all memory store sizes."""
        return {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "episodic_backend": self.episodic._backend,
        }

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _store_by_name(self, name: str) -> AbstractMemory:
        mapping = {
            self.STORE_WORKING:  self.working,
            self.STORE_EPISODIC: self.episodic,
            self.STORE_SEMANTIC: self.semantic,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown store {name!r}. "
                f"Valid options: {list(mapping.keys())}"
            )
        return mapping[name]

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryManager("
            f"working={s['working']}, "
            f"episodic={s['episodic']}[{s['episodic_backend']}], "
            f"semantic={s['semantic']})"
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
