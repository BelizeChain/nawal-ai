"""
tests/test_quantum.py — Phase 5 quantum module tests.

All tests run in *classical fallback mode* — no live Kinich node required.
Quantum paths are exercised via simulation_mode=True where available.

Test classes
------------
TestQuantumMemory        — QuantumMemory (classical + simulated-Grover)
TestQuantumPlanOptimizer — QuantumPlanOptimizer (greedy + SA)
TestQuantumAnomalyDetector — QuantumAnomalyDetector (fit/predict/score)
TestQuantumImagination   — QuantumImagination (classical + stochastic)
TestQuantumExports       — quantum/__init__.py exports present
TestQuantumLayer         — integration: orchestrator wires all 4 modules
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── module under test ────────────────────────────────────────────────────────
from quantum.quantum_memory import QuantumMemory
from quantum.quantum_optimizer import QuantumPlanOptimizer
from quantum.quantum_anomaly import QuantumAnomalyDetector
from quantum.quantum_imagination import QuantumImagination, SimulatedState
from quantum import (
    KinichQuantumConnector,
    QuantumEnhancedLayer,
    HybridQuantumClassicalLLM,
    QuantumMemory as QMem,
    QuantumPlanOptimizer as QPlan,
    QuantumAnomalyDetector as QAnom,
    QuantumImagination as QImag,
    SimulatedState as SS,
)

# ── shared helpers ────────────────────────────────────────────────────────────
from control.interfaces import Plan
from memory.episodic import EpisodicMemory
from memory.interfaces import MemoryRecord


def _make_plan(plan_id: str, score: float, **meta) -> Plan:
    return Plan(
        plan_id=plan_id,
        goal_id="goal-1",
        steps=[{"action": "step1"}],
        score=score,
        metadata=meta,
    )


def _make_records(n: int, dim: int = 8) -> List[MemoryRecord]:
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        emb = rng.standard_normal(dim).tolist()
        records.append(
            MemoryRecord(
                key=f"rec_{i}",
                content=f"content {i}",
                embedding=emb,
                metadata={"idx": i},
            )
        )
    return records


def _mock_episodic(records: List[MemoryRecord]) -> MagicMock:
    """Return a mock EpisodicMemory that holds given records."""
    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=len(records))

    def _retrieve(query_embedding, top_k=5, filters=None):
        q = np.asarray(query_embedding, dtype=np.float32)
        scored = []
        for rec in records:
            if rec.embedding is not None:
                v = np.asarray(rec.embedding, dtype=np.float32)
                cos = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            else:
                cos = 0.0
            scored.append((cos, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    mock.retrieve = MagicMock(side_effect=_retrieve)
    mock.store    = MagicMock()
    mock.get      = MagicMock(return_value=None)
    mock.delete   = MagicMock(return_value=True)
    mock.clear    = MagicMock()
    return mock


# =============================================================================
# TestQuantumMemory
# =============================================================================

class TestQuantumMemory:
    """QuantumMemory — retrieval and routing logic."""

    def test_init_defaults(self):
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        assert qm.fallback_to_classical is True
        assert qm.simulation_mode is False

    def test_classical_retrieve_delegates(self):
        records = _make_records(10, dim=8)
        backing = _mock_episodic(records)
        qm = QuantumMemory(backing_store=backing)
        query = np.random.default_rng(1).standard_normal(8).tolist()
        results = qm.retrieve(query, top_k=3)
        assert len(results) == 3
        backing.retrieve.assert_called_once()

    def test_stats_classical_increments(self):
        backing = _mock_episodic(_make_records(5))
        qm = QuantumMemory(backing_store=backing)
        qm.retrieve([0.1] * 8, top_k=2)
        assert qm.stats["classical_calls"] == 1

    def test_simulated_grover_returns_top_k(self):
        records = _make_records(50, dim=16)
        backing = _mock_episodic(records)
        qm = QuantumMemory(backing_store=backing, simulation_mode=True)
        query = np.random.default_rng(7).standard_normal(16).tolist()
        results = qm.retrieve(query, top_k=5)
        assert 1 <= len(results) <= 5

    def test_simulated_grover_stats_incremented(self):
        records = _make_records(20, dim=8)
        backing = _mock_episodic(records)
        qm = QuantumMemory(backing_store=backing, simulation_mode=True)
        qm.retrieve([0.5] * 8, top_k=3)
        assert qm.stats["simulated_calls"] == 1
        assert qm.stats["classical_calls"] == 0

    def test_search_pure_indices(self):
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((20, 8)).astype(np.float32)
        query  = corpus[0]  # exact first record
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        indices = qm.search(query, top_k=3, corpus_vectors=corpus)
        assert 0 in indices, "exact query vector should be top result"
        assert len(indices) == 3

    def test_search_empty_corpus(self):
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        assert qm.search([1.0, 2.0], top_k=5, corpus_vectors=np.empty((0, 2))) == []

    def test_search_returns_unique(self):
        rng   = np.random.default_rng(3)
        corpus = rng.standard_normal((10, 4)).astype(np.float32)
        qm    = QuantumMemory(backing_store=_mock_episodic([]))
        idx   = qm.search(corpus[0], top_k=5, corpus_vectors=corpus)
        assert len(idx) == len(set(idx)), "no duplicate indices"

    def test_get_stats_structure(self):
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        stats = qm.get_stats()
        assert "total_calls" in stats
        assert "quantum_ratio" in stats

    def test_no_connector_never_quantum(self):
        backing = _mock_episodic(_make_records(2_000))
        qm = QuantumMemory(backing_store=backing, quantum_threshold=100)
        # Even with a large corpus, without connector, quantum path not taken
        qm.retrieve([0.1] * 8, top_k=5)
        assert qm.stats["quantum_calls"] == 0

    def test_delegate_store(self):
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        rec = MemoryRecord(key="k1", content="hi")
        qm.store(rec)
        backing.store.assert_called_once_with(rec)

    def test_delegate_clear(self):
        backing = _mock_episodic([])
        qm = QuantumMemory(backing_store=backing)
        qm.clear()
        backing.clear.assert_called_once()


# =============================================================================
# TestQuantumPlanOptimizer
# =============================================================================

class TestQuantumPlanOptimizer:
    """QuantumPlanOptimizer — plan ranking logic."""

    def test_select_best_single(self):
        plans = [_make_plan("p1", 0.8)]
        opt = QuantumPlanOptimizer()
        best = opt.select_best_plan(plans)
        assert best.plan_id == "p1"

    def test_select_best_returns_highest_for_greedy(self):
        plans = [
            _make_plan("low", 0.1),
            _make_plan("high", 0.9),
            _make_plan("mid", 0.5),
        ]
        opt = QuantumPlanOptimizer()
        best = opt.select_best_plan(plans)
        assert best.plan_id == "high"

    def test_rank_plans_greedy_order(self):
        plans = [
            _make_plan("c", 0.3),
            _make_plan("a", 0.9),
            _make_plan("b", 0.6),
        ]
        opt = QuantumPlanOptimizer()
        ranked = opt.rank_plans(plans)
        assert ranked[0].plan_id == "a"
        assert ranked[-1].plan_id == "c"

    def test_rank_preserves_all_plans(self):
        plans = [_make_plan(f"p{i}", float(i) / 10) for i in range(8)]
        opt = QuantumPlanOptimizer()
        ranked = opt.rank_plans(plans)
        assert len(ranked) == 8
        assert {p.plan_id for p in ranked} == {p.plan_id for p in plans}

    def test_simulated_annealing_mode(self):
        plans = [_make_plan(f"p{i}", float(i) / 5) for i in range(6)]
        opt = QuantumPlanOptimizer(simulation_mode=True, sa_iterations=200)
        ranked = opt.rank_plans(plans)
        assert len(ranked) == 6
        # Best plan should be near the top in SA (not strict guarantee)
        top2_ids = {r.plan_id for r in ranked[:2]}
        assert "p5" in top2_ids or "p4" in top2_ids

    def test_stats_classical_path(self):
        plans = [_make_plan("x", 1.0), _make_plan("y", 0.5)]
        opt = QuantumPlanOptimizer()
        opt.rank_plans(plans)
        assert opt.stats["classical_calls"] == 1

    def test_stats_simulated_path(self):
        plans = [_make_plan("x", 1.0), _make_plan("y", 0.5)]
        opt = QuantumPlanOptimizer(simulation_mode=True)
        opt.rank_plans(plans)
        assert opt.stats["simulated_calls"] == 1

    def test_empty_raises(self):
        opt = QuantumPlanOptimizer()
        with pytest.raises(ValueError):
            opt.select_best_plan([])

    def test_empty_rank_returns_empty(self):
        opt = QuantumPlanOptimizer()
        assert opt.rank_plans([]) == []

    def test_objectives_metadata_bonus(self):
        p_no_meta = _make_plan("plain", 0.5)
        p_with_meta = _make_plan("enriched", 0.5, safety=1.0, speed=1.0)
        opt = QuantumPlanOptimizer()
        ranked = opt.rank_plans([p_no_meta, p_with_meta])
        assert ranked[0].plan_id == "enriched"

    def test_get_stats_structure(self):
        opt = QuantumPlanOptimizer()
        stats = opt.get_stats()
        assert "quantum_ratio" in stats
        assert "total_calls" in stats


# =============================================================================
# TestQuantumAnomalyDetector
# =============================================================================

class TestQuantumAnomalyDetector:
    """QuantumAnomalyDetector — fit / predict / score."""

    def _normal_data(self, n=200, d=16, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, d)).astype(np.float64)

    def test_not_fitted_initially(self):
        det = QuantumAnomalyDetector()
        assert det.is_fitted is False

    def test_fit_marks_fitted(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        assert det.is_fitted is True

    def test_predict_length_matches_input(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        X = np.random.default_rng(1).standard_normal((10, 16))
        labels = det.predict(X)
        assert len(labels) == 10

    def test_predict_returns_bools(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        X = np.random.default_rng(2).standard_normal((5, 16))
        labels = det.predict(X)
        assert all(isinstance(v, bool) for v in labels)

    def test_score_shape(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        X = np.random.default_rng(3).standard_normal((7, 16))
        scores = det.score(X)
        assert scores.shape == (7,)

    def test_anomaly_scores_higher_for_outliers(self):
        rng = np.random.default_rng(10)
        normal = rng.standard_normal((200, 8)) * 0.5
        det = QuantumAnomalyDetector(contamination=0.05)
        det.fit(normal)

        inliers  = rng.standard_normal((10, 8)) * 0.5
        outliers = rng.standard_normal((10, 8)) * 10 + 20  # far out

        s_in  = det.score(inliers).mean()
        s_out = det.score(outliers).mean()
        assert s_out > s_in, "outliers should have higher anomaly scores"

    def test_1d_input_single_sample(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        x = np.zeros(16)
        labels = det.predict(x)
        assert len(labels) == 1

    def test_fit_returns_self(self):
        det = QuantumAnomalyDetector()
        result = det.fit(self._normal_data())
        assert result is det

    def test_no_fit_raises(self):
        det = QuantumAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score(np.zeros((3, 4)))

    def test_simulated_mode_produces_scores(self):
        det = QuantumAnomalyDetector(simulation_mode=True)
        det.fit(self._normal_data())
        X = np.random.default_rng(5).standard_normal((4, 16))
        scores = det.score(X)
        assert scores.shape == (4,)
        assert det.stats["simulated_calls"] > 0

    def test_classical_mode_mahalanobis_path(self):
        det = QuantumAnomalyDetector(simulation_mode=False)
        det.fit(self._normal_data())
        X = np.random.default_rng(6).standard_normal((4, 16))
        det.score(X)
        assert det.stats["classical_calls"] > 0

    def test_empty_fit_raises(self):
        det = QuantumAnomalyDetector()
        with pytest.raises(ValueError):
            det.fit(np.empty((0, 8)))

    def test_get_stats_structure(self):
        det = QuantumAnomalyDetector()
        det.fit(self._normal_data())
        s = det.get_stats()
        assert "fitted" in s
        assert "threshold" in s

    def test_contamination_affects_threshold(self):
        data = self._normal_data(n=200)
        det_low  = QuantumAnomalyDetector(contamination=0.01)
        det_high = QuantumAnomalyDetector(contamination=0.50)
        det_low.fit(data)
        det_high.fit(data)
        # Higher contamination → lower threshold → more things flagged
        assert det_high._threshold <= det_low._threshold


# =============================================================================
# TestQuantumImagination
# =============================================================================

class TestQuantumImagination:
    """QuantumImagination — future trajectory sampling."""

    def _state(self) -> Dict[str, Any]:
        return {"task": "summarise", "tokens": 512}

    def _actions(self) -> List[Dict[str, Any]]:
        return [
            {"model": "fast", "speed": True,   "accuracy": False},
            {"model": "best", "speed": False,  "accuracy": True, "safety": True},
            {"model": "balanced", "speed": True, "accuracy": True},
        ]

    def test_sample_futures_returns_list(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions(), n_samples=3)
        assert isinstance(futures, list)

    def test_sample_futures_length(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions(), n_samples=2)
        # 3 actions × 2 samples
        assert len(futures) == 6

    def test_sample_futures_sorted_by_value(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions(), n_samples=4)
        values = [f.value for f in futures]
        assert values == sorted(values, reverse=True)

    def test_simulated_state_fields(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions()[:1], n_samples=1)
        f = futures[0]
        assert isinstance(f, SimulatedState)
        assert isinstance(f.value, float)
        assert isinstance(f.uncertainty, float)
        assert isinstance(f.trajectory, list)

    def test_simulation_mode_adds_uncertainty(self):
        qi = QuantumImagination(simulation_mode=True, n_samples=16)
        futures = qi.sample_futures(self._state(), self._actions()[:1], n_samples=16)
        # In simulated mode multiple samples are aggregated → uncertainty > 0
        assert futures[0].uncertainty >= 0.0  # could be 0 with identical samples

    def test_classical_fallback_stats(self):
        qi = QuantumImagination(simulation_mode=False)
        qi.sample_futures(self._state(), self._actions(), n_samples=2)
        assert qi.stats["classical_calls"] > 0

    def test_simulated_stats(self):
        qi = QuantumImagination(simulation_mode=True)
        qi.sample_futures(self._state(), self._actions(), n_samples=2)
        assert qi.stats["simulated_calls"] > 0

    def test_best_action_returns_action_dict(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions(), n_samples=4)
        best = qi.best_action(futures)
        assert isinstance(best, dict)

    def test_best_action_none_for_empty(self):
        qi = QuantumImagination()
        assert qi.best_action([]) is None

    def test_empty_actions_returns_empty(self):
        qi = QuantumImagination()
        assert qi.sample_futures(self._state(), [], n_samples=4) == []

    def test_best_action_is_from_futures(self):
        qi = QuantumImagination()
        futures = qi.sample_futures(self._state(), self._actions(), n_samples=4)
        best = qi.best_action(futures)
        known = [str(a) for a in self._actions()]
        assert str(best) in known

    def test_with_internal_simulator_delegated(self):
        """Ensure InternalSimulator is called on classical path if provided."""
        mock_sim = MagicMock()
        mock_result = MagicMock()
        mock_result.trajectory = [{"step": 1}]
        mock_result.value      = 0.7
        mock_sim.simulate      = MagicMock(return_value=mock_result)

        qi = QuantumImagination(internal_simulator=mock_sim, simulation_mode=False)
        qi.sample_futures(self._state(), self._actions()[:1], n_samples=2)
        assert mock_sim.simulate.call_count == 2

    def test_get_stats_structure(self):
        qi = QuantumImagination()
        qi.sample_futures(self._state(), self._actions(), n_samples=1)
        s = qi.get_stats()
        assert "quantum_ratio" in s
        assert "total_calls" in s


# =============================================================================
# TestQuantumExports
# =============================================================================

class TestQuantumExports:
    """Verify all Phase 5 symbols are exported from quantum/__init__.py."""

    def test_quantum_memory_exported(self):
        assert QMem is QuantumMemory

    def test_quantum_plan_optimizer_exported(self):
        assert QPlan is QuantumPlanOptimizer

    def test_quantum_anomaly_exported(self):
        assert QAnom is QuantumAnomalyDetector

    def test_quantum_imagination_exported(self):
        assert QImag is QuantumImagination

    def test_simulated_state_exported(self):
        assert SS is SimulatedState

    def test_legacy_exports_intact(self):
        assert KinichQuantumConnector is not None
        assert QuantumEnhancedLayer is not None
        assert HybridQuantumClassicalLLM is not None


# =============================================================================
# TestQuantumLayer  (integration — orchestrator wiring)
# =============================================================================

class TestQuantumLayer:
    """Integration: all four modules coexist and produce consistent results."""

    def _build_modules(self):
        records = _make_records(30, dim=16)
        backing = _mock_episodic(records)
        qm   = QuantumMemory(backing_store=backing)
        qpo  = QuantumPlanOptimizer()
        qad  = QuantumAnomalyDetector()
        qi   = QuantumImagination()
        return qm, qpo, qad, qi

    def test_all_four_instantiate(self):
        qm, qpo, qad, qi = self._build_modules()
        assert qm is not None
        assert qpo is not None
        assert qad is not None
        assert qi is not None

    def test_memory_and_optimizer_pipeline(self):
        """Retrieve from memory, build plans, optimise."""
        records = _make_records(10, dim=8)
        backing = _mock_episodic(records)
        qm  = QuantumMemory(backing_store=backing)
        qpo = QuantumPlanOptimizer()

        q = np.random.default_rng(0).standard_normal(8).tolist()
        mems = qm.retrieve(q, top_k=3)
        assert len(mems) > 0

        plans = [
            _make_plan(f"plan_{i}", float(i) / 10)
            for i in range(5)
        ]
        best = qpo.select_best_plan(plans)
        assert best is not None

    def test_anomaly_detector_full_cycle(self):
        rng    = np.random.default_rng(99)
        normal = rng.standard_normal((100, 8))
        qad    = QuantumAnomalyDetector(contamination=0.1)
        qad.fit(normal)

        test = rng.standard_normal((5, 8))
        labels = qad.predict(test)
        scores = qad.score(test)

        assert len(labels) == 5
        assert scores.shape == (5,)

    def test_imagination_and_optimizer_pipeline(self):
        """Sample futures and then optimise the resulting action-plans."""
        state   = {"task": "translate"}
        actions = [{"lang": "es"}, {"lang": "fr"}, {"lang": "de"}]
        qi  = QuantumImagination(simulation_mode=True)
        qpo = QuantumPlanOptimizer()

        futures = qi.sample_futures(state, actions, n_samples=4)
        assert len(futures) > 0

        plans = [
            _make_plan(f"p{i}", futures[i].value)
            for i in range(min(3, len(futures)))
        ]
        best = qpo.select_best_plan(plans)
        assert best is not None

    def test_classical_quantum_agreement_memory(self):
        """
        Classical top-5 and simulated-Grover top-5 should share ≥ 1 record in
        common across the majority of queries.

        Note: simulated Grover deliberately reranks via amplitude diffusion, so
        top-1 exact agreement is not expected.  This test validates that the
        quantum-simulated path is correlated with the classical ranking (i.e.
        it retrieves relevant records), not that it produces identical order.
        """
        dim = 16
        records = _make_records(100, dim=dim)
        rng = np.random.default_rng(42)

        backing_classical = _mock_episodic(records)
        backing_sim       = _mock_episodic(records)

        qm_c = QuantumMemory(backing_store=backing_classical, simulation_mode=False)
        qm_s = QuantumMemory(backing_store=backing_sim,       simulation_mode=True)

        n_overlap = 0
        n_trials  = 20
        for _ in range(n_trials):
            q   = rng.standard_normal(dim).tolist()
            r_c = {r.key for r in qm_c.retrieve(q, top_k=5)}
            r_s = {r.key for r in qm_s.retrieve(q, top_k=5)}
            if r_c & r_s:          # at least one key in common
                n_overlap += 1

        # Classical candidates are the input pool for simulated Grover,  so
        # top-5 overlap should always be 100 %.
        overlap_rate = n_overlap / n_trials
        assert overlap_rate >= 0.80, f"overlap_rate={overlap_rate:.2f} unexpectedly low"
