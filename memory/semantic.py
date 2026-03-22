"""
Semantic Memory — knowledge-graph-backed long-term conceptual store.

Analogous to the neocortical semantic memory system: stores *facts,
concepts, and relationships* rather than specific past events.

Structure:
  - Nodes represent concepts / entities (MemoryRecords).
  - Directed edges represent typed relationships between nodes
    (e.g. "is-a", "part-of", "related-to", "causes").
  - Retrieval combines vector similarity (cosine on node embeddings)
    with graph proximity (BFS spreading-activation).

Backend:
  - NetworkX directed multigraph (pure-Python, in-memory).
  - Optional JSON / pickle serialisation for persistence.
  - No heavy external dependencies required.

Phase 1: complete.
Phase 5: Quantum associative memory overlay can replace the BFS hop
         with quantum walk-based activation, reaching exponentially more
         relevant nodes in the same compute budget.

Usage::

    sm = SemanticMemory()
    sm.store(MemoryRecord(key="dog", content="A domesticated canine",
                          embedding=[0.1]*768, metadata={"type": "animal"}))
    sm.store(MemoryRecord(key="cat", content="A domesticated feline",
                          embedding=[0.09]*768, metadata={"type": "animal"}))
    sm.add_relation("dog", "cat", relation="related-to", weight=0.6)
    results = sm.retrieve(query_embedding=[0.1]*768, top_k=5)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory.interfaces import AbstractMemory, MemoryRecord

try:
    import networkx as nx

    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logger.warning(
        "networkx not installed — SemanticMemory will use a simple dict backend. "
        "Install: pip install networkx"
    )


# --------------------------------------------------------------------------- #
# SemanticMemory                                                               #
# --------------------------------------------------------------------------- #


class SemanticMemory(AbstractMemory):
    """
    Knowledge-graph semantic memory.

    Nodes in the graph are MemoryRecords (concepts / facts).
    Edges encode typed relationships, each with an optional weight.

    Retrieval uses spreading-activation:
      1. Compute cosine similarity to all nodes that carry an embedding.
      2. Perform BFS from the top-k similarity seeds up to *hop_depth* hops.
      3. Re-rank all reached nodes by a combined similarity+proximity score.

    Args:
        hop_depth       : BFS depth for spreading-activation (default 2).
        proximity_decay : Weight applied per BFS hop (0 < decay ≤ 1).
        persist_path    : If set, save/load graph to this file on
                          ``save()`` / ``load()``.
    """

    def __init__(
        self,
        hop_depth: int = 2,
        proximity_decay: float = 0.5,
        persist_path: Optional[str] = None,
    ) -> None:
        if not (0.0 < proximity_decay <= 1.0):
            raise ValueError("proximity_decay must be in (0, 1]")

        self.hop_depth = hop_depth
        self.proximity_decay = proximity_decay
        self.persist_path = persist_path

        if NX_AVAILABLE:
            self._graph: "nx.MultiDiGraph" = nx.MultiDiGraph()
        else:
            self._graph = None  # type: ignore[assignment]

        # Plain dict fallback for when networkx is absent
        self._records: Dict[str, MemoryRecord] = {}

        if persist_path and Path(persist_path).exists():
            self.load(persist_path)
            logger.info(f"SemanticMemory loaded from {persist_path!r}")

    # ------------------------------------------------------------------ #
    # AbstractMemory implementation                                        #
    # ------------------------------------------------------------------ #

    def store(self, record: MemoryRecord) -> None:
        """Add or update a concept node."""
        self._records[record.key] = record
        if NX_AVAILABLE:
            self._graph.add_node(record.key, record=record)
        logger.debug(f"SemanticMemory stored concept key={record.key!r}")

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve top-k concepts by spreading-activation search.

        Step 1: score all nodes by cosine similarity.
        Step 2: BFS from top-4 seed nodes up to *hop_depth* hops.
        Step 3: merge similarity scores with proximity bonus, re-rank.
        """
        if not self._records:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        candidates = {
            k: rec
            for k, rec in self._records.items()
            if not rec.is_expired() and _meta_matches(rec, filters)
        }

        if not candidates:
            return []

        # Step 1: cosine similarity scoring
        sim_scores: Dict[str, float] = {}
        for key, rec in candidates.items():
            if rec.embedding is not None:
                sim_scores[key] = _cosine(
                    q, np.asarray(rec.embedding, dtype=np.float32)
                )
            else:
                sim_scores[key] = 0.0

        sorted_by_sim = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 2: BFS spreading-activation from top seeds
        activation: Dict[str, float] = {}
        if NX_AVAILABLE:
            seed_keys = [k for k, _ in sorted_by_sim[:4]]
            for seed in seed_keys:
                self._bfs_activate(seed, sim_scores.get(seed, 0.0), activation)
        else:
            # Fallback — just use similarity
            activation = dict(sorted_by_sim)

        # Step 3: merge
        final_scores: Dict[str, float] = {}
        for key in candidates:
            sim = sim_scores.get(key, 0.0)
            act = activation.get(key, 0.0)
            final_scores[key] = 0.7 * sim + 0.3 * act

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [candidates[k] for k, _ in ranked[:top_k]]

    def get(self, key: str) -> Optional[MemoryRecord]:
        rec = self._records.get(key)
        if rec is not None and rec.is_expired():
            self.delete(key)
            return None
        return rec

    def delete(self, key: str) -> bool:
        existed = key in self._records
        self._records.pop(key, None)
        if NX_AVAILABLE and self._graph.has_node(key):
            self._graph.remove_node(key)
        return existed

    def clear(self) -> None:
        self._records.clear()
        if NX_AVAILABLE:
            self._graph.clear()
        logger.debug("SemanticMemory cleared")

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------ #
    # Graph-specific API                                                   #
    # ------------------------------------------------------------------ #

    def add_relation(
        self,
        source_key: str,
        target_key: str,
        relation: str = "related-to",
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a typed, weighted directed edge between two concept nodes.

        Both nodes must already exist via ``store()``.

        Args:
            source_key : Key of the source concept.
            target_key : Key of the target concept.
            relation   : Edge label (e.g. "is-a", "part-of", "causes").
            weight     : Edge strength in (0, 1].
            metadata   : Optional dict stored on the edge.
        """
        if not NX_AVAILABLE:
            logger.warning("add_relation requires networkx")
            return
        if source_key not in self._records:
            raise KeyError(f"Source concept {source_key!r} not found in SemanticMemory")
        if target_key not in self._records:
            raise KeyError(f"Target concept {target_key!r} not found in SemanticMemory")

        self._graph.add_edge(
            source_key,
            target_key,
            relation=relation,
            weight=weight,
            **(metadata or {}),
        )
        logger.debug(
            f"SemanticMemory relation: {source_key!r} --[{relation}]--> {target_key!r} "
            f"weight={weight:.2f}"
        )

    def neighbours(
        self,
        key: str,
        relation: Optional[str] = None,
        hops: int = 1,
    ) -> List[Tuple[str, str, float]]:
        """
        Return neighbouring concept keys within *hops* edges.

        Args:
            key      : Source concept key.
            relation : If set, only traverse edges with this relation label.
            hops     : Maximum BFS depth.

        Returns:
            List of (neighbour_key, relation_label, weight) tuples.
        """
        if not NX_AVAILABLE or not self._graph.has_node(key):
            return []

        visited: Dict[str, Tuple[str, float]] = {}
        frontier = [key]
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for _, nbr, edge_data in self._graph.out_edges(node, data=True):
                    rel = edge_data.get("relation", "related-to")
                    w = float(edge_data.get("weight", 1.0))
                    if relation and rel != relation:
                        continue
                    if nbr not in visited and nbr != key:
                        visited[nbr] = (rel, w)
                        next_frontier.append(nbr)
            frontier = next_frontier

        return [(nbr, rel, w) for nbr, (rel, w) in visited.items()]

    def path(self, source_key: str, target_key: str) -> Optional[List[str]]:
        """
        Shortest path between two concept nodes (key sequence).

        Returns ``None`` if no path exists or networkx is not available.
        """
        if not NX_AVAILABLE:
            return None
        try:
            return nx.shortest_path(self._graph, source_key, target_key)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def concept_summary(self, key: str) -> Dict[str, Any]:
        """
        Return a dict summarising a concept and its direct relations.
        """
        rec = self.get(key)
        if rec is None:
            return {}
        neighbours = self.neighbours(key, hops=1)
        return {
            "key": key,
            "content": rec.content,
            "metadata": rec.metadata,
            "relations": [
                {"target": nbr, "relation": rel, "weight": w}
                for nbr, rel, w in neighbours
            ],
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        """
        Persist the graph to disk (pickle format).

        Saves both the _records dict and (if networkx) the graph structure.
        """
        target = path or self.persist_path
        if not target:
            raise ValueError("No persist_path configured")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": self._records,
            "graph": nx.node_link_data(self._graph) if NX_AVAILABLE else None,
        }
        with open(target, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"SemanticMemory saved to {target!r} ({len(self._records)} nodes)")

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved graph from disk."""
        target = path or self.persist_path
        if not target:
            raise ValueError("No persist_path configured")
        with open(target, "rb") as fh:
            data = pickle.load(fh)  # nosec B301
        self._records = data.get("records", {})
        if NX_AVAILABLE and data.get("graph"):
            self._graph = nx.node_link_graph(
                data["graph"], directed=True, multigraph=True
            )
        logger.info(
            f"SemanticMemory loaded from {target!r} ({len(self._records)} nodes)"
        )

    # ------------------------------------------------------------------ #
    # BFS spreading-activation (internal)                                  #
    # ------------------------------------------------------------------ #

    def _bfs_activate(
        self,
        seed: str,
        seed_score: float,
        activation: Dict[str, float],
    ) -> None:
        """Propagate activation from *seed* via BFS up to *hop_depth* hops."""
        frontier = [(seed, seed_score)]
        for hop in range(self.hop_depth):
            next_frontier = []
            decay = self.proximity_decay ** (hop + 1)
            for node, score in frontier:
                for _, nbr, edge_data in self._graph.out_edges(node, data=True):
                    edge_w = float(edge_data.get("weight", 1.0))
                    bonus = score * decay * edge_w
                    if bonus > activation.get(nbr, 0.0):
                        activation[nbr] = bonus
                        next_frontier.append((nbr, bonus))
            frontier = next_frontier

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        edge_count = self._graph.number_of_edges() if NX_AVAILABLE else 0
        return (
            f"SemanticMemory(concepts={len(self._records)}, "
            f"relations={edge_count}, hop_depth={self.hop_depth})"
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


def _meta_matches(record: MemoryRecord, filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    return all(record.metadata.get(k) == v for k, v in filters.items())
