"""
Memory Module — Nawal Brain Architecture (Hippocampus + Neocortex)

Three-store memory system modelled after the mammalian memory hierarchy:

  WorkingMemory   — fast volatile FIFO buffer (prefrontal context window)
  EpisodicMemory  — vector-DB-backed event store (hippocampal recall)
  SemanticMemory  — knowledge-graph concept store (neocortical facts)
  MemoryManager   — unified facade aggregating all three stores

Phase 1: all concrete implementations complete.
Phase 4: QuantumHippocampus overlay (Grover-style retrieval on EpisodicMemory).

Canonical imports::

    from nawal.memory import MemoryManager, MemoryRecord
    from nawal.memory import WorkingMemory, EpisodicMemory, SemanticMemory
    from nawal.memory.interfaces import AbstractMemory
"""

from memory.interfaces import AbstractMemory, MemoryRecord
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.manager import MemoryManager

__all__ = [
    # Interfaces
    "AbstractMemory",
    "MemoryRecord",
    # Concrete stores
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    # Unified facade
    "MemoryManager",
]
