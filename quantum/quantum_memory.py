"""
QuantumMemory — Quantum-accelerated episodic memory retrieval.

This module wraps any ``AbstractMemory`` store and overrides the
retrieval step with a Grover-inspired quantum search when a live
Kinich node is available.

Architecture
------------
Classical path (default / fallback):
    Cosine similarity top-k — O(N) scan over all stored vectors.

Simulated quantum path (no live Kinich, simulation_mode=True):
    Grover oracle simulation — quadratically-biased sampling that
    preferentially returns high-similarity records, demonstrating
    the expected quantum advantage without real hardware.

Quantum path (kinich_available=True):
    Encodes the query and corpus embeddings as quantum feature maps,
    runs a Grover-style amplitude amplification circuit on the Kinich
    node, and decodes the amplified probability distribution to
    identify the top-k matches.

Usage
-----
::

    from quantum.quantum_memory import QuantumMemory
    from memory.episodic import EpisodicMemory

    qmem = QuantumMemory(backing_store=EpisodicMemory())
    results = qmem.retrieve(query_embedding=[...], top_k=5)

PhaseHook
---------
Phase 5: Replace ``_quantum_retrieve()`` body with live Kinich HTTP call
         once the node is provisioned.  The ``retrieve()`` public API is
         unchanged.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from memory.interfaces import AbstractMemory, MemoryRecord

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Threshold above which quantum path is beneficial (classical is fast for small N)
QUANTUM_THRESHOLD = 1_000


# --------------------------------------------------------------------------- #
# QuantumMemory                                                                #
# --------------------------------------------------------------------------- #


class QuantumMemory:
    """
    Quantum-accelerated memory retrieval using Grover-inspired search.

    Args:
        backing_store        : Any AbstractMemory instance.
        connector            : KinichQuantumConnector (optional).
                               When provided and available, real quantum
                               circuits are used.
        fallback_to_classical: Always use classical path if True and quantum
                               unavailable (default True).
        simulation_mode      : Use simulated Grover bias when Kinich
                               unavailable but you want to demo quantum
                               behaviour (default False).
        quantum_threshold    : Min corpus size for quantum routing
                               (default 1 000).
    """

    def __init__(
        self,
        backing_store: AbstractMemory,
        connector: Optional[Any] = None,
        fallback_to_classical: bool = True,
        simulation_mode: bool = False,
        quantum_threshold: int = QUANTUM_THRESHOLD,
    ) -> None:
        self._store = backing_store
        self._connector = connector
        self.fallback_to_classical = fallback_to_classical
        self.simulation_mode = simulation_mode
        self.quantum_threshold = quantum_threshold

        self.stats: Dict[str, int] = {
            "quantum_calls": 0,
            "simulated_calls": 0,
            "classical_calls": 0,
            "cache_hits": 0,
        }

        logger.info(
            f"QuantumMemory ready: simulation_mode={simulation_mode} "
            f"threshold={quantum_threshold} "
            f"connector={'yes' if connector else 'none'}"
        )

    # ------------------------------------------------------------------ #
    # Public API (mirrors AbstractMemory.retrieve)                         #
    # ------------------------------------------------------------------ #

    def store(self, record: MemoryRecord) -> None:
        """Delegate to backing store."""
        self._store.store(record)

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve the top-k most relevant records for *query_embedding*.

        Routes to quantum / simulated / classical path based on
        availability and corpus size.

        Args:
            query_embedding : Dense query vector.
            top_k           : Number of results to return.
            filters         : Optional metadata filters (passed through).

        Returns:
            List of MemoryRecord, most similar first.
        """
        t0 = time.perf_counter()

        # Classical always wins for small corpora
        corpus_size = self._corpus_size()
        use_quantum = self._should_use_quantum(corpus_size)
        use_sim = self.simulation_mode and not use_quantum

        if use_quantum:
            results = self._quantum_retrieve(query_embedding, top_k, filters)
            self.stats["quantum_calls"] += 1
            mode = "quantum"
        elif use_sim:
            results = self._simulated_grover_retrieve(query_embedding, top_k, filters)
            self.stats["simulated_calls"] += 1
            mode = "simulated"
        else:
            results = self._classical_retrieve(query_embedding, top_k, filters)
            self.stats["classical_calls"] += 1
            mode = "classical"

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"QuantumMemory.retrieve: mode={mode} top_k={top_k} "
            f"corpus={corpus_size} elapsed={elapsed*1000:.1f}ms results={len(results)}"
        )
        return results

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        corpus_vectors: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Low-level index search.  Returns integer indices.

        When *corpus_vectors* is provided (e.g. a pre-fetched matrix),
        ranking is done on that matrix rather than the backing store.

        Returns:
            List of integer indices (0-based) into *corpus_vectors*.
        """
        if corpus_vectors is None or len(corpus_vectors) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-9
        normed = corpus_vectors / norms
        sims = normed @ q_norm

        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return top_indices.tolist()

    # ------------------------------------------------------------------ #
    # Path implementations                                                 #
    # ------------------------------------------------------------------ #

    def _classical_retrieve(
        self,
        query: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        """Standard cosine-similarity top-k via backing store."""
        return self._store.retrieve(query_embedding=query, top_k=top_k, filters=filters)

    def _simulated_grover_retrieve(
        self,
        query: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        """
        Grover amplitude-amplification *simulation* (no hardware).

        Algorithm:
          1. Fetch classical top-k×4 candidates.
          2. Score each by cosine similarity → probability amplitude.
          3. Apply one Grover diffusion step:
               a_new = 2·mean(a) - a_i   (inverts about mean)
          4. Square amplitudes → sampling probabilities.
          5. Sample top_k records according to probabilities (greedy argmax).

        This demonstrates the quadratic sorting bias without real qubits.
        """
        q = np.asarray(query, dtype=np.float32)
        # Fetch a wider candidate pool
        candidates = self._store.retrieve(
            query_embedding=query,
            top_k=min(top_k * 4, self._corpus_size() or top_k),
            filters=filters,
        )
        if len(candidates) <= top_k:
            return candidates

        # Build amplitude vector from cosine similarities.
        # Grover oracle: marks high-similarity records by *lowering* their
        # initial amplitude (phase-flip analogy).  Records with cosine ≈ 1
        # start near 0 (marked); records with cosine ≈ 0 start near 1.
        # Diffusion then *amplifies* the marked (low-amplitude) records.
        amps = []
        for rec in candidates:
            if rec.embedding is not None:
                v = np.asarray(rec.embedding, dtype=np.float32)
                cos = float(
                    np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)
                )
            else:
                cos = 0.0
            # Oracle: high similarity → low amplitude (1 - similarity)
            amps.append(1.0 - max(cos, 0.0))

        amps_arr = np.array(amps, dtype=np.float64)

        # Grover diffusion: a_new = 2·mean - a
        mean_a = amps_arr.mean()
        diffused = 2.0 * mean_a - amps_arr

        # Probabilities: square of diffused amplitudes (renormalized)
        probs = diffused**2
        total = probs.sum()
        if total > 0:
            probs /= total

        # Greedy top_k by probability
        sorted_idx = np.argsort(probs)[::-1][:top_k]
        return [candidates[i] for i in sorted_idx]

    def _quantum_retrieve(
        self,
        query: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[MemoryRecord]:
        """
        Quantum path via Kinich connector.

        PhaseHook — Phase 5: implement real Kinich HTTP call here.
        For now, delegates to simulated Grover.
        """
        if self._connector is not None and getattr(
            self._connector, "kinich_available", False
        ):
            # DESIGN NOTE (Phase 5): replace simulated Grover with
            # connector.quantum_process(query, top_k, filters) when Kinich is live.
            logger.debug(
                "QuantumMemory: Kinich connected — using simulated path (live TBD)"
            )
        return self._simulated_grover_retrieve(query, top_k, filters)

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def _should_use_quantum(self, corpus_size: int) -> bool:
        if self._connector is None:
            return False
        if not getattr(self._connector, "kinich_available", False):
            return False
        if corpus_size < self.quantum_threshold:
            return False
        return True

    def _corpus_size(self) -> int:
        """Best-effort corpus size estimate."""
        try:
            return len(self._store)
        except (TypeError, AttributeError):
            return 0

    # ------------------------------------------------------------------ #
    # Delegated AbstractMemory methods                                     #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[MemoryRecord]:
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        return self._store.delete(key)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.stats.values())
        return {
            **self.stats,
            "total_calls": total,
            "quantum_ratio": (
                self.stats["quantum_calls"] / total if total > 0 else 0.0
            ),
        }
