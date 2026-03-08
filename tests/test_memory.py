"""
Tests for the Phase 1 Memory subsystem.

Covers WorkingMemory, EpisodicMemory (numpy fallback), SemanticMemory,
and MemoryManager — all without requiring external databases.
"""
from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np
import pytest

from memory.interfaces import MemoryRecord
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.manager import MemoryManager


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _rec(
    key: str = None,
    content: str = "test",
    dim: int = 8,
    metadata: dict = None,
    ttl: float = None,
) -> MemoryRecord:
    """Create a MemoryRecord with a random embedding."""
    emb = list(np.random.randn(dim).astype(float))
    return MemoryRecord(
        key=key or str(uuid.uuid4()),
        content=content,
        embedding=emb,
        metadata=metadata or {},
        ttl=ttl,
    )


def _unit(dim: int = 8, scale: float = 1.0) -> List[float]:
    """Return a random unit-normalised vector."""
    v = np.random.randn(dim).astype(float)
    v /= np.linalg.norm(v) + 1e-9
    return (v * scale).tolist()


# --------------------------------------------------------------------------- #
# MemoryRecord                                                                 #
# --------------------------------------------------------------------------- #

class TestMemoryRecord:
    def test_not_expired_no_ttl(self):
        r = _rec(ttl=None)
        assert not r.is_expired()

    def test_not_expired_future(self):
        r = _rec(ttl=3600.0)
        assert not r.is_expired()

    def test_expired_past(self):
        r = _rec()
        r.timestamp = time.time() - 10.0
        r.ttl = 5.0
        assert r.is_expired()


# --------------------------------------------------------------------------- #
# WorkingMemory                                                                #
# --------------------------------------------------------------------------- #

class TestWorkingMemory:
    def test_store_and_len(self):
        wm = WorkingMemory(max_size=10)
        wm.store(_rec("a"))
        wm.store(_rec("b"))
        assert len(wm) == 2

    def test_eviction_on_full(self):
        wm = WorkingMemory(max_size=3)
        for i in range(4):
            wm.store(_rec(f"k{i}"))
        assert len(wm) == 3
        assert wm.get("k0") is None   # evicted
        assert wm.get("k3") is not None

    def test_update_in_place(self):
        wm = WorkingMemory(max_size=5)
        r = _rec("dup")
        wm.store(r)
        r2 = MemoryRecord(key="dup", content="updated")
        wm.store(r2)
        assert len(wm) == 1
        assert wm.get("dup").content == "updated"

    def test_retrieve_cosine_ranking(self):
        wm = WorkingMemory(max_size=20)
        target_emb = np.ones(8, dtype=float)
        target_emb /= np.linalg.norm(target_emb)

        # r_close has embedding parallel to query
        r_close = MemoryRecord(key="close", content="close",
                               embedding=target_emb.tolist())
        # r_far has orthogonal embedding
        far_emb = np.zeros(8, dtype=float)
        far_emb[0] = 1.0
        far_emb[1] = -1.0
        far_emb /= np.linalg.norm(far_emb)
        r_far = MemoryRecord(key="far", content="far",
                             embedding=far_emb.tolist())
        wm.store(r_far)
        wm.store(r_close)

        results = wm.retrieve(query_embedding=target_emb.tolist(), top_k=2)
        assert results[0].key == "close"

    def test_retrieve_respects_filters(self):
        wm = WorkingMemory(max_size=10)
        wm.store(_rec("a", metadata={"role": "user"}))
        wm.store(_rec("b", metadata={"role": "assistant"}))
        results = wm.retrieve(query_embedding=[1.0] * 8,
                              filters={"role": "user"})
        assert all(r.metadata.get("role") == "user" for r in results)

    def test_ttl_expiry_pruned_on_retrieve(self):
        wm = WorkingMemory(max_size=10)
        r = _rec("expiring", ttl=0.01)
        wm.store(r)
        time.sleep(0.05)
        results = wm.retrieve(query_embedding=[0.0] * 8, top_k=5)
        assert not any(res.key == "expiring" for res in results)

    def test_delete(self):
        wm = WorkingMemory(max_size=5)
        wm.store(_rec("del_me"))
        assert wm.delete("del_me")
        assert not wm.delete("del_me")
        assert len(wm) == 0

    def test_clear(self):
        wm = WorkingMemory(max_size=5)
        for i in range(5):
            wm.store(_rec(f"k{i}"))
        wm.clear()
        assert len(wm) == 0

    def test_recent(self):
        wm = WorkingMemory(max_size=10)
        for i in range(5):
            wm.store(_rec(f"r{i}"))
        recent = wm.recent(3)
        assert len(recent) == 3
        assert recent[0].key == "r4"  # newest first

    def test_store_text_helper(self):
        wm = WorkingMemory()
        rec = wm.store_text("hello world", ttl=60.0)
        assert wm.get(rec.key) is not None
        assert wm.get(rec.key).content == "hello world"

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            WorkingMemory(max_size=0)


# --------------------------------------------------------------------------- #
# EpisodicMemory (numpy backend — no external DB required)                     #
# --------------------------------------------------------------------------- #

class TestEpisodicMemoryNumpy:
    """Force numpy fallback by passing persist_path=None."""

    def _em(self) -> EpisodicMemory:
        return EpisodicMemory(persist_path=None)

    def test_backend_is_numpy(self):
        em = self._em()
        assert em._backend == "numpy"

    def test_store_and_len(self):
        em = self._em()
        em.store(_rec("e1"))
        em.store(_rec("e2"))
        assert len(em) == 2

    def test_retrieve_returns_similar(self):
        em = self._em()
        query = np.ones(8, dtype=float) / np.sqrt(8)
        close = MemoryRecord(key="close", content="x",
                             embedding=(query * 0.99).tolist())
        far_emb = np.zeros(8, dtype=float)
        far_emb[0] = -1.0
        far = MemoryRecord(key="far", content="y",
                           embedding=far_emb.tolist())
        em.store(far)
        em.store(close)
        results = em.retrieve(query_embedding=query.tolist(), top_k=2)
        assert results[0].key == "close"

    def test_get_existing(self):
        em = self._em()
        r = _rec("g1")
        em.store(r)
        assert em.get("g1").key == "g1"

    def test_get_missing_returns_none(self):
        em = self._em()
        assert em.get("nope") is None

    def test_delete(self):
        em = self._em()
        em.store(_rec("del"))
        assert em.delete("del")
        assert len(em) == 0
        assert not em.delete("del")

    def test_clear(self):
        em = self._em()
        for i in range(3):
            em.store(_rec(f"c{i}"))
        em.clear()
        assert len(em) == 0

    def test_ttl_expiry_not_returned(self):
        em = self._em()
        r = _rec("ttl_test", ttl=0.01)
        em.store(r)
        time.sleep(0.05)
        results = em.retrieve(query_embedding=[0.0] * 8, top_k=5)
        assert not any(res.key == "ttl_test" for res in results)

    def test_metadata_filter(self):
        em = self._em()
        em.store(_rec("u1", metadata={"role": "user"}))
        em.store(_rec("u2", metadata={"role": "bot"}))
        results = em.retrieve([1.0] * 8, filters={"role": "user"})
        assert all(r.metadata.get("role") == "user" for r in results)

    def test_repr(self):
        em = self._em()
        assert "numpy" in repr(em)


# --------------------------------------------------------------------------- #
# SemanticMemory                                                               #
# --------------------------------------------------------------------------- #

class TestSemanticMemory:
    def test_store_and_len(self):
        sm = SemanticMemory()
        sm.store(_rec("dog", "A dog"))
        sm.store(_rec("cat", "A cat"))
        assert len(sm) == 2

    def test_retrieve_by_similarity(self):
        sm = SemanticMemory()
        q = np.ones(8, dtype=float) / np.sqrt(8)
        close = MemoryRecord(key="close", content="close",
                             embedding=(q * 1.0).tolist())
        far_emb = np.zeros(8)
        far_emb[0] = -1.0
        far = MemoryRecord(key="far", content="far",
                           embedding=far_emb.tolist())
        sm.store(far)
        sm.store(close)
        results = sm.retrieve(q.tolist(), top_k=2)
        assert results[0].key == "close"

    def test_add_and_query_relation(self):
        sm = SemanticMemory()
        sm.store(_rec("dog"))
        sm.store(_rec("animal"))
        sm.add_relation("dog", "animal", relation="is-a", weight=1.0)
        neighbours = sm.neighbours("dog", hops=1)
        keys = [k for k, _, _ in neighbours]
        assert "animal" in keys

    def test_add_relation_missing_key_raises(self):
        sm = SemanticMemory()
        sm.store(_rec("exists"))
        with pytest.raises(KeyError):
            sm.add_relation("exists", "missing", relation="related-to")

    def test_path_between_nodes(self):
        sm = SemanticMemory()
        sm.store(_rec("a"))
        sm.store(_rec("b"))
        sm.store(_rec("c"))
        sm.add_relation("a", "b")
        sm.add_relation("b", "c")
        p = sm.path("a", "c")
        assert p is not None
        assert p[0] == "a" and p[-1] == "c"

    def test_delete_node(self):
        sm = SemanticMemory()
        sm.store(_rec("d"))
        assert sm.delete("d")
        assert len(sm) == 0

    def test_clear(self):
        sm = SemanticMemory()
        for i in range(3):
            sm.store(_rec(f"n{i}"))
        sm.clear()
        assert len(sm) == 0

    def test_concept_summary(self):
        sm = SemanticMemory()
        sm.store(_rec("x"))
        sm.store(_rec("y"))
        sm.add_relation("x", "y", relation="causes")
        summary = sm.concept_summary("x")
        assert summary["key"] == "x"
        assert any(r["relation"] == "causes" for r in summary["relations"])

    def test_spreading_activation_boosts_neighbours(self):
        """Connected neighbours should score higher than unconnected strangers."""
        sm = SemanticMemory(hop_depth=2, proximity_decay=0.8)
        q = np.ones(8, dtype=float) / np.sqrt(8)

        # "seed" is highly similar to query
        seed_emb = (q * 0.99).tolist()
        sm.store(MemoryRecord(key="seed", content="seed", embedding=seed_emb))

        # "linked" has low sim but is directly connected to "seed"
        linked_emb = list(-q * 0.1)
        sm.store(MemoryRecord(key="linked", content="linked", embedding=linked_emb))
        sm.add_relation("seed", "linked", weight=1.0)

        # "isolated" has same low sim but no relation
        sm.store(MemoryRecord(key="isolated", content="iso", embedding=linked_emb))

        results = sm.retrieve(q.tolist(), top_k=3)
        keys = [r.key for r in results]
        # "linked" should rank above "isolated" due to activation
        assert keys.index("linked") < keys.index("isolated")

    def test_save_and_load(self, tmp_path):
        sm = SemanticMemory(persist_path=str(tmp_path / "kg.pkl"))
        sm.store(_rec("n1", "concept1"))
        sm.store(_rec("n2", "concept2"))
        sm.add_relation("n1", "n2", relation="leads-to")
        sm.save()

        sm2 = SemanticMemory(persist_path=str(tmp_path / "kg.pkl"))
        assert len(sm2) == 2
        assert sm2.get("n1") is not None


# --------------------------------------------------------------------------- #
# MemoryManager                                                                #
# --------------------------------------------------------------------------- #

class TestMemoryManager:
    def _mm(self) -> MemoryManager:
        """Return a MemoryManager with all in-memory backends."""
        return MemoryManager(episodic_persist_path=None)

    def test_stats_initially_zero(self):
        mm = self._mm()
        s = mm.stats()
        assert s["working"] == 0
        assert s["episodic"] == 0
        assert s["semantic"] == 0

    def test_store_default_is_working(self):
        mm = self._mm()
        mm.store(_rec("w1"))
        assert len(mm.working) == 1
        assert len(mm.episodic) == 0

    def test_store_explicit_episodic(self):
        mm = self._mm()
        mm.store(_rec("e1"), store="episodic")
        assert len(mm.episodic) == 1
        assert len(mm.working) == 0

    def test_store_explicit_semantic(self):
        mm = self._mm()
        mm.store(_rec("s1"), store="semantic")
        assert len(mm.semantic) == 1

    def test_store_all(self):
        mm = self._mm()
        mm.store(_rec("all1"), store="all")
        assert len(mm.working) == 1
        assert len(mm.episodic) == 1
        assert len(mm.semantic) == 1

    def test_store_text_helper(self):
        mm = self._mm()
        rec = mm.store_text("hello", store="working")
        assert mm.get(rec.key) is not None

    def test_get_searches_all_stores(self):
        mm = self._mm()
        r1 = _rec("in_working")
        r2 = _rec("in_episodic")
        r3 = _rec("in_semantic")
        mm.store(r1, store="working")
        mm.store(r2, store="episodic")
        mm.store(r3, store="semantic")
        assert mm.get("in_working") is not None
        assert mm.get("in_episodic") is not None
        assert mm.get("in_semantic") is not None
        assert mm.get("nonexistent") is None

    def test_retrieve_merges_stores(self):
        mm = self._mm()
        q = np.ones(8, dtype=float) / np.sqrt(8)
        mm.store_text("working item",  embedding=q.tolist(),  store="working")
        mm.store_text("episodic item", embedding=q.tolist(),  store="episodic")
        mm.store_text("semantic item", embedding=q.tolist(),  store="semantic")
        results = mm.retrieve(q.tolist(), top_k=10)
        assert len(results) == 3

    def test_retrieve_dedup(self):
        mm = self._mm()
        q = np.ones(8, dtype=float) / np.sqrt(8)
        rec = MemoryRecord(key="shared", content="shared",
                           embedding=q.tolist(), metadata={"store": "all"})
        mm.store(rec, store="all")   # same key in all three
        results = mm.retrieve(q.tolist(), top_k=10, dedup=True)
        keys = [r.key for r in results]
        assert keys.count("shared") == 1

    def test_context_window(self):
        mm = self._mm()
        q = np.ones(8, dtype=float) / np.sqrt(8)
        for i in range(5):
            mm.store_text(f"turn {i}", embedding=q.tolist(), store="working")
        mm.store_text("past event", embedding=q.tolist(), store="episodic")
        mm.store_text("dog fact",   embedding=q.tolist(), store="semantic")
        ctx = mm.context_window(
            query_embedding=q.tolist(),
            n_recent=3, n_episodic=1, n_semantic=1,
        )
        assert len(ctx) >= 3

    def test_consolidate_moves_old_items(self):
        mm = self._mm()
        mm.consolidation_age = 0.01  # 10 ms for test speed
        q = np.ones(8, dtype=float) / np.sqrt(8)
        mm.store_text("old item", embedding=q.tolist(), store="working")
        time.sleep(0.05)
        consolidated = mm.consolidate()
        assert consolidated >= 1
        assert len(mm.working) == 0
        assert len(mm.episodic) >= 1

    def test_consolidate_skips_recent_items(self):
        mm = self._mm()
        mm.consolidation_age = 3600.0   # 1 hour — nothing should consolidate
        q = np.ones(8, dtype=float) / np.sqrt(8)
        mm.store_text("fresh item", embedding=q.tolist(), store="working")
        consolidated = mm.consolidate()
        assert consolidated == 0
        assert len(mm.working) == 1

    def test_repr(self):
        mm = self._mm()
        r = repr(mm)
        assert "MemoryManager" in r
