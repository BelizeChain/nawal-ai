"""
Memory interfaces — Abstract Base Classes for all memory subsystems.

All concrete memory implementations (WorkingMemory, EpisodicMemory,
SemanticMemory, QuantumHippocampus) must implement AbstractMemory.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    """
    A single item stored in any memory subsystem.

    Attributes:
        key       : Unique identifier (UUID or hash of content).
        content   : Raw payload — text, embedding, structured dict, etc.
        embedding : Vector representation for similarity search (optional).
        metadata  : Arbitrary tags (source, timestamp, importance, …).
        timestamp : Unix epoch creation time.
        ttl       : Time-to-live in seconds; None = permanent.
    """

    key: str
    content: Any
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds; None = permanent

    def is_expired(self) -> bool:
        """Return True if this record has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class AbstractMemory(ABC):
    """
    Abstract interface for all memory subsystems in the Nawal brain.

    Subclasses implement one of three memory stores:
      - WorkingMemory    (fast, volatile, context-window-bound)
      - EpisodicMemory   (vector DB persisted, event-driven)
      - SemanticMemory   (knowledge graph / long-term facts)

    Phase 4 addition: QuantumHippocampus wraps EpisodicMemory and
    overrides ``retrieve`` with a Grover-style quantum search.
    """

    # ------------------------------------------------------------------ #
    # Write API                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def store(self, record: MemoryRecord) -> None:
        """Persist a MemoryRecord."""

    # ------------------------------------------------------------------ #
    # Read API                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """
        Return the top-k most relevant MemoryRecords for *query_embedding*.

        Args:
            query_embedding : Dense vector representation of the query.
            top_k           : Maximum number of results to return.
            filters         : Optional metadata filters (e.g. {"source": "chat"}).

        Returns:
            List of MemoryRecord, sorted by descending relevance.
        """

    @abstractmethod
    def get(self, key: str) -> Optional[MemoryRecord]:
        """Exact-key lookup. Returns None if not found."""

    # ------------------------------------------------------------------ #
    # Management API                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a record by key. Returns True if it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe all records from this store."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of records currently held."""
