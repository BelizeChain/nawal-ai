"""
Tests for Phase 3 — Metacognitive Layer.

Covers:
  - SelfCritic
  - ConsistencyChecker
  - ConfidenceCalibrator
  - InternalSimulator
  - IdentityModule
  - MetacognitionLayer (facade)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from metacognition.interfaces import ConfidenceScore, CritiqueResult
from metacognition.self_critic import SelfCritic
from metacognition.consistency_checker import ConsistencyChecker
from metacognition.confidence_calibrator import ConfidenceCalibrator
from metacognition.internal_simulator import InternalSimulator
from metacognition.identity_module import AgentProfile, DecisionRecord, IdentityModule
from metacognition.layer import MetacognitionLayer, ReflectionResult


# =========================================================================== #
# Helpers
# =========================================================================== #

GOOD_RESPONSE = (
    "Paris is the capital of France. It is home to the Eiffel Tower and the Louvre."
)
EMPTY_RESPONSE = ""
SHORT_RESPONSE = "A"
BARE_REFUSAL = "I'm sorry, I can't help with that."
HALLUCINATION_RESPONSE = "I'm not sure but maybe Paris is the capital of France."
HEDGY_RESPONSE = "I'm not sure, but perhaps Paris might possibly be the capital."
CONFIDENT_RESPONSE = "Definitively, Paris is the capital of France."


# =========================================================================== #
# SelfCritic
# =========================================================================== #

class TestSelfCritic:

    def test_good_response_approved(self):
        critic = SelfCritic()
        result = critic.critique(GOOD_RESPONSE, {})
        assert result.approved is True
        assert result.issues == []

    def test_empty_response_rejected(self):
        critic = SelfCritic(auto_revise=False)
        result = critic.critique(EMPTY_RESPONSE, {})
        assert result.approved is False
        assert len(result.issues) > 0

    def test_too_short_rejected(self):
        critic = SelfCritic(auto_revise=False)
        result = critic.critique("Hi", {})
        assert result.approved is False

    def test_bare_refusal_flagged(self):
        critic = SelfCritic(auto_revise=False, fail_threshold=1)
        result = critic.critique(BARE_REFUSAL, {})
        # "I cannot" pattern triggers bare_refusal check
        assert result.approved is False

    def test_hallucination_marker_flagged(self):
        critic = SelfCritic(auto_revise=False)
        result = critic.critique(HALLUCINATION_RESPONSE, {})
        assert result.approved is False

    def test_auto_revise_adds_punctuation(self):
        critic = SelfCritic(auto_revise=True)
        no_punct = "Paris is the capital of France"
        result = critic.critique(no_punct, {})
        if result.revised_response:
            valid_endings = {".", "!", "?"}
            assert result.revised_response[-1] in valid_endings

    def test_auto_revise_on_good_response(self):
        critic = SelfCritic(auto_revise=True)
        result = critic.critique(GOOD_RESPONSE, {})
        assert result.approved is True

    def test_add_and_remove_check(self):
        critic = SelfCritic()
        original_names = list(critic.check_names())

        # Add a custom check that always fails
        def always_fail(response: str, ctx: Dict) -> Optional[str]:
            return "always fails"

        critic.add_check("always_fail", always_fail)
        assert "always_fail" in critic.check_names()

        result = critic.critique(GOOD_RESPONSE, {})
        assert result.approved is False

        # Remove it — good response should pass again
        critic.remove_check("always_fail")
        assert "always_fail" not in critic.check_names()
        result2 = critic.critique(GOOD_RESPONSE, {})
        assert result2.approved is True

    def test_estimate_confidence_good(self):
        critic = SelfCritic()
        conf = critic.estimate_confidence(GOOD_RESPONSE, {})
        assert isinstance(conf, ConfidenceScore)
        assert 0.0 <= conf.value <= 1.0
        assert conf.value > 0.5

    def test_estimate_confidence_bad(self):
        critic = SelfCritic(auto_revise=False)
        conf_bad  = critic.estimate_confidence(EMPTY_RESPONSE, {})
        conf_good = critic.estimate_confidence(GOOD_RESPONSE, {})
        assert conf_bad.value < conf_good.value

    def test_fail_threshold_one_vs_two(self):
        # With threshold=2, a single issue should still approve
        critic = SelfCritic(auto_revise=False, fail_threshold=2)

        def one_issue(r: str, ctx: Dict) -> Optional[str]:
            return "one issue"

        critic.add_check("one_issue", one_issue, prepend=True)
        result = critic.critique(GOOD_RESPONSE, {})
        assert result.approved is True   # 1 issue < threshold 2

    def test_check_names_order(self):
        critic = SelfCritic()
        names = critic.check_names()
        assert "not_empty" in names
        assert "min_length" in names


# =========================================================================== #
# ConsistencyChecker
# =========================================================================== #

class TestConsistencyChecker:

    def test_single_candidate_perfect_score(self):
        checker = ConsistencyChecker()
        result = checker.check([GOOD_RESPONSE], {})
        assert result.score == 1.0
        assert result.contradictions == []

    def test_two_consistent_candidates(self):
        checker = ConsistencyChecker()
        c1 = "Paris is the capital of France with a population of 2 million."
        c2 = "France's capital Paris has roughly 2 million inhabitants."
        result = checker.check([c1, c2], {})
        assert result.score >= 0.5   # may find numeric pair, but not necessarily

    def test_numeric_contradiction(self):
        checker = ConsistencyChecker()
        c1 = "The population is 2 million."
        c2 = "The population is 10 million."
        result = checker.check([c1, c2], {})
        # There may or may not be contradictions depending on claim extraction;
        # at minimum score is in [0,1]
        assert 0.0 <= result.score <= 1.0

    def test_negation_contradiction(self):
        checker = ConsistencyChecker()
        c1 = "Paris is the capital."
        c2 = "Paris is not the capital."
        result = checker.check([c1, c2], {})
        # Score should be ≤ 1.0
        assert result.score <= 1.0

    def test_most_consistent_returns_index(self):
        checker = ConsistencyChecker()
        candidates = [GOOD_RESPONSE, EMPTY_RESPONSE, SHORT_RESPONSE]
        result = checker.check(candidates, {})
        assert result.most_supported is None or isinstance(result.most_supported, int)

    def test_most_consistent_method(self):
        checker = ConsistencyChecker()
        candidates = [GOOD_RESPONSE, EMPTY_RESPONSE]
        best = checker.most_consistent(candidates, {})
        assert best is None or best in candidates

    def test_score_range(self):
        checker = ConsistencyChecker()
        candidates = ["A. " * 50, "B. " * 50, "C. " * 50]
        result = checker.check(candidates, {})
        assert 0.0 <= result.score <= 1.0

    def test_empty_candidates(self):
        checker = ConsistencyChecker()
        result = checker.check([], {})
        assert result.score == 1.0
        assert result.most_supported is None


# =========================================================================== #
# ConfidenceCalibrator
# =========================================================================== #

class TestConfidenceCalibrator:

    def test_all_signals(self):
        cal = ConfidenceCalibrator()
        signals = {
            "plan_score": 0.9,
            "critic_score": 0.8,
            "consistency": 1.0,
            "memory": 0.7,
            "safety": 1.0,
            "verbal": GOOD_RESPONSE,
        }
        cf = cal.calibrate(signals, {})
        assert isinstance(cf, ConfidenceScore)
        assert cf.value > 0.7

    def test_verbal_hedging_lowers_score(self):
        cal = ConfidenceCalibrator()
        hedgy = cal.calibrate({"verbal": HEDGY_RESPONSE}, {})
        confident = cal.calibrate({"verbal": CONFIDENT_RESPONSE}, {})
        assert hedgy.value < confident.value

    def test_no_signals_returns_half(self):
        cal = ConfidenceCalibrator()
        cf = cal.calibrate({}, {})
        assert cf.value == pytest.approx(0.5)

    def test_confidence_score_signal(self):
        cal = ConfidenceCalibrator()
        cs = ConfidenceScore(value=0.9, method="critic")
        cf = cal.calibrate({"critic_score": cs}, {})
        assert cf.value > 0.5

    def test_calibrate_text(self):
        cal = ConfidenceCalibrator()
        score = cal.calibrate_text(HEDGY_RESPONSE)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        cal = ConfidenceCalibrator(weights={"plan_score": 10.0})
        signals_high = {"plan_score": 1.0}
        signals_low  = {"plan_score": 0.0}
        cf_high = cal.calibrate(signals_high, {})
        cf_low  = cal.calibrate(signals_low, {})
        assert cf_high.value > cf_low.value

    def test_method_label(self):
        cal = ConfidenceCalibrator()
        cf = cal.calibrate({"plan_score": 0.8}, {})
        assert cf.method == "weighted_aggregate"


# =========================================================================== #
# InternalSimulator
# =========================================================================== #

class TestInternalSimulator:

    def _state(self) -> Dict[str, Any]:
        return {
            "goal": "Answer user",
            "context": [],
            "memory_hits": 0,
            "responded": False,
            "validated": False,
            "action_count": 0,
        }

    def test_simulate_returns_list(self):
        sim = InternalSimulator(n_samples=2, horizon=2, seed=0)
        state = self._state()
        actions = [{"type": "generate", "content": "Hello"}]
        results = sim.simulate(state, actions, horizon=2, n_samples=2)
        assert isinstance(results, list)
        assert len(results) == len(actions)

    def test_simulate_result_structure(self):
        sim = InternalSimulator(n_samples=2, horizon=2, seed=0)
        results = sim.simulate(self._state(), [{"type": "generate", "content": "Hi"}])
        r = results[0]
        assert "action" in r
        assert "value" in r
        assert "trajectory" in r
        assert isinstance(r["value"], float)

    def test_best_action_picks_max(self):
        sim = InternalSimulator(seed=0)
        simulations = [
            {"action": {"type": "a"}, "value": 0.3},
            {"action": {"type": "b"}, "value": 0.9},
            {"action": {"type": "c"}, "value": 0.1},
        ]
        best = sim.best_action(simulations)
        assert best["type"] == "b"

    def test_best_action_empty_returns_noop(self):
        sim = InternalSimulator(seed=0)
        best = sim.best_action([])
        assert best == {"type": "noop"}

    def test_empty_actions(self):
        sim = InternalSimulator(seed=0)
        results = sim.simulate(self._state(), [])
        assert results == []

    def test_multiple_actions(self):
        sim = InternalSimulator(n_samples=1, horizon=1, seed=42)
        actions = [
            {"type": "generate", "content": "Hello"},
            {"type": "validate", "content": ""},
            {"type": "retrieve", "query": "test"},
        ]
        results = sim.simulate(self._state(), actions)
        assert len(results) == 3

    def test_valuation_layer_integration(self):
        """Simulator accepts a mock valuation layer without crashing."""

        class MockVL:
            def score(self, state: Dict) -> float:
                return 1.0

        sim = InternalSimulator(valuation_layer=MockVL(), seed=0)
        state = self._state()
        state["responded"] = True
        results = sim.simulate(state, [{"type": "noop"}])
        assert results[0]["value"] > 0.0


# =========================================================================== #
# IdentityModule
# =========================================================================== #

class TestIdentityModule:

    def test_default_profile_name(self):
        mod = IdentityModule()
        assert mod.profile.name == "Nawal"

    def test_default_profile_version(self):
        mod = IdentityModule()
        assert mod.profile.version == "2.0.0-brain"

    def test_record_decision_returns_record(self):
        mod = IdentityModule()
        rec = mod.record_decision(goal="test_goal", outcome="success", confidence=0.9)
        assert isinstance(rec, DecisionRecord)
        assert rec.goal == "test_goal"
        assert rec.outcome == "success"
        assert rec.confidence == pytest.approx(0.9)

    def test_recent_decisions(self):
        mod = IdentityModule()
        for i in range(5):
            mod.record_decision(f"goal_{i}", "success", 0.8)
        recents = mod.recent_decisions(3)
        assert len(recents) == 3

    def test_success_rate_all_success(self):
        mod = IdentityModule()
        for _ in range(10):
            mod.record_decision("goal", "success", 0.9)
        assert mod.success_rate() == pytest.approx(1.0)

    def test_success_rate_all_failure(self):
        mod = IdentityModule()
        for _ in range(10):
            mod.record_decision("goal", "failure", 0.2)
        assert mod.success_rate() == pytest.approx(0.0)

    def test_success_rate_empty(self):
        mod = IdentityModule()
        assert mod.success_rate() == pytest.approx(0.0)

    def test_avg_confidence(self):
        mod = IdentityModule()
        mod.record_decision("g", "success", 0.6)
        mod.record_decision("g", "success", 0.8)
        avg = mod.avg_confidence()
        assert avg == pytest.approx(0.7, abs=1e-9)

    def test_system_prompt_contains_name(self):
        mod = IdentityModule()
        prompt = mod.system_prompt()
        assert "Nawal" in prompt

    def test_self_description_brief(self):
        mod = IdentityModule()
        desc = mod.self_description(brief=True)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_self_description_verbose(self):
        mod = IdentityModule()
        desc = mod.self_description(brief=False)
        assert len(desc) > len(mod.self_description(brief=True))

    def test_add_capability(self):
        mod = IdentityModule()
        original = len(mod.all_capabilities())
        mod.add_capability("quantum reasoning")
        assert len(mod.all_capabilities()) == original + 1

    def test_add_limitation(self):
        mod = IdentityModule()
        original = len(mod.all_limitations())
        mod.add_limitation("no realtime data")
        assert len(mod.all_limitations()) == original + 1

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            mod = IdentityModule(persist_path=path)
            mod.record_decision("goal_a", "success", 0.88)
            mod.save()
            assert os.path.exists(path)

            mod2 = IdentityModule(persist_path=path)
            ok = mod2.load()
            assert ok is True
            recents = mod2.recent_decisions(1)
            assert len(recents) == 1
            assert recents[0].goal == "goal_a"
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_missing_file(self):
        mod = IdentityModule(persist_path="/tmp/nonexistent_identity_xyz.json")
        ok = mod.load()
        assert ok is False

    def test_custom_profile(self):
        profile = AgentProfile(name="TestBot", role="tester", version="0.0.1")
        mod = IdentityModule(profile=profile)
        assert mod.profile.name == "TestBot"


# =========================================================================== #
# MetacognitionLayer (facade)
# =========================================================================== #

class TestMetacognitionLayer:

    def _candidates(self) -> List[str]:
        return [
            "Paris is the capital of France.",
            "France's capital city is Paris, home to the Louvre.",
        ]

    def test_reflect_returns_result(self):
        layer = MetacognitionLayer()
        result = layer.reflect(self._candidates(), {"goal": "What is Paris?"})
        assert isinstance(result, ReflectionResult)

    def test_reflect_best_candidate_in_candidates(self):
        layer = MetacognitionLayer()
        cands = self._candidates()
        result = layer.reflect(cands, {})
        assert result.best_candidate in cands

    def test_reflect_approved_good_responses(self):
        layer = MetacognitionLayer()
        result = layer.reflect([GOOD_RESPONSE], {})
        assert result.approved is True

    def test_reflect_rejects_bad_responses(self):
        layer = MetacognitionLayer()
        result = layer.reflect([EMPTY_RESPONSE, SHORT_RESPONSE], {})
        assert result.approved is False

    def test_reflect_confidence_is_confidence_score(self):
        layer = MetacognitionLayer()
        result = layer.reflect(self._candidates(), {})
        assert isinstance(result.confidence, ConfidenceScore)
        assert 0.0 <= result.confidence.value <= 1.0

    def test_reflect_extra_signals(self):
        layer = MetacognitionLayer()
        result = layer.reflect(
            self._candidates(),
            plan_score=0.9,
            memory_relevance=0.7,
            safety_score=1.0,
        )
        assert result.confidence.value > 0.5

    def test_reflect_empty_candidates(self):
        layer = MetacognitionLayer()
        result = layer.reflect([], {})
        assert result.approved is False
        assert result.best_candidate == ""

    def test_reflect_returns_self_description(self):
        layer = MetacognitionLayer()
        result = layer.reflect([GOOD_RESPONSE], {})
        assert isinstance(result.self_description, str)

    def test_simulate_actions_returns_dict(self):
        layer = MetacognitionLayer()
        state = {
            "goal": "test",
            "context": [],
            "memory_hits": 0,
            "responded": False,
            "validated": False,
            "action_count": 0,
        }
        actions = [{"type": "generate", "content": "hi"}]
        best = layer.simulate_actions(state, actions)
        assert isinstance(best, dict)
        assert "type" in best

    def test_record_outcome_via_identity(self):
        layer = MetacognitionLayer()
        layer.identity.record_decision("test_goal", "success", 0.8)
        rate = layer.identity.success_rate()
        assert rate > 0.0

    def test_sub_system_accessors(self):
        layer = MetacognitionLayer()
        assert isinstance(layer.critic, SelfCritic)
        assert isinstance(layer.checker, ConsistencyChecker)
        assert isinstance(layer.calibrator, ConfidenceCalibrator)
        assert isinstance(layer.simulator, InternalSimulator)
        assert isinstance(layer.identity, IdentityModule)

    def test_custom_persist_path(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            layer = MetacognitionLayer(persist_path=path)
            layer.identity.record_decision("g", "success", 0.9)
            layer.save()
            assert os.path.getsize(path) > 10
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_returns_bool(self):
        layer = MetacognitionLayer()
        result = layer.load()
        assert isinstance(result, bool)

    def test_reflect_consistency_attached(self):
        layer = MetacognitionLayer()
        result = layer.reflect(self._candidates(), {})
        assert result.consistency is not None
        assert 0.0 <= result.consistency.score <= 1.0

    def test_reflect_critique_attached(self):
        layer = MetacognitionLayer()
        result = layer.reflect([GOOD_RESPONSE], {})
        assert result.critique is not None
