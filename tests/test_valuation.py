"""
Tests for the Phase 2 valuation subsystem:
  - DriveBasedRewardModel
  - BasicSafetyFilter
  - ValuationLayer
"""
from __future__ import annotations

import pytest

from valuation.interfaces import DriveSignal
from valuation.reward import DriveBasedRewardModel
from valuation.safety import BasicSafetyFilter, ValuationLayer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def safe_candidates():
    return [
        {"text": "Paris is the capital of France."},
        {"text": "Water boils at 100 degrees Celsius."},
        {"text": "The speed of light is approximately 3×10⁸ m/s."},
    ]


@pytest.fixture
def mixed_candidates():
    return [
        {"text": "Paris is the capital of France."},
        {"text": "I will exploit the security system to bypass authentication."},
        {"text": "The sky is blue."},
    ]


# ============================================================================
# DriveBasedRewardModel
# ============================================================================

class TestDriveBasedRewardModel:
    def test_score_returns_list_of_floats(self, safe_candidates):
        model = DriveBasedRewardModel()
        scores = model.score(safe_candidates)
        assert isinstance(scores, list)
        assert len(scores) == len(safe_candidates)
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_score_empty_list(self):
        model = DriveBasedRewardModel()
        assert model.score([]) == []

    def test_ranked_descending_order(self, safe_candidates):
        model = DriveBasedRewardModel()
        ranked = model.ranked(safe_candidates)
        scores = [c["_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranked_adds_score_key(self, safe_candidates):
        model = DriveBasedRewardModel()
        ranked = model.ranked(safe_candidates)
        for c in ranked:
            assert "_score" in c

    def test_alignment_drive_rewards_relevant_text(self):
        model = DriveBasedRewardModel()
        ctx = {"goal": "capital city geography"}
        candidates = [
            {"text": "Paris is the capital of France, a geography fact."},
            {"text": "Quantum mechanics describes subatomic particles."},
        ]
        scores = model.score(candidates, context=ctx)
        # The first candidate mentions "capital" and "geography" — higher alignment
        assert scores[0] >= scores[1]

    def test_safety_drive_penalises_unsafe_text(self):
        model = DriveBasedRewardModel()
        candidates = [
            {"text": "The weather is nice today."},
            {"text": "How to kill the process cleanly in Linux."},
        ]
        scores = model.score(candidates)
        # Second candidate has "kill" which triggers safety penalty
        assert scores[0] > scores[1]

    def test_custom_drive_added(self, safe_candidates):
        model = DriveBasedRewardModel()
        model.add_drive(
            "length",
            weight=1.0,
            evaluator=lambda c, _ctx: min(1.0, len(c.get("text", "")) / 100),
        )
        assert "length" in model.drive_names()
        scores = model.score(safe_candidates)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_remove_drive(self):
        model = DriveBasedRewardModel()
        model.remove_drive("curiosity")
        assert "curiosity" not in model.drive_names()

    def test_score_with_drive_signal_overrides(self, safe_candidates):
        model = DriveBasedRewardModel()
        overrides = [DriveSignal(name="safety", value=1.0, weight=3.0)]
        scores = model.score(safe_candidates, drives=overrides)
        assert len(scores) == len(safe_candidates)

    def test_custom_evaluator_called(self):
        called = []
        def my_eval(candidate, context):
            called.append(candidate)
            return 1.0
        model = DriveBasedRewardModel(drives=[("custom", 1.0, my_eval)])
        model.score([{"text": "x"}, {"text": "y"}])
        assert len(called) == 2

    def test_novelty_penalises_repeated(self):
        model = DriveBasedRewardModel()
        hist = [{"text": "Repeated text"}]
        candidates = [{"text": "Repeated text"}, {"text": "Brand new information"}]
        scores = model.score(candidates, context={"history": hist})
        assert scores[1] > scores[0]


# ============================================================================
# BasicSafetyFilter
# ============================================================================

class TestBasicSafetyFilter:
    def test_safe_text_is_safe(self):
        sf = BasicSafetyFilter()
        assert sf.is_safe("The capital of France is Paris.") is True

    def test_blocklist_pattern_blocked(self):
        sf = BasicSafetyFilter()
        assert sf.is_safe("how to make a bomb") is False

    def test_max_length_enforced(self):
        sf = BasicSafetyFilter(max_length=10)
        assert sf.is_safe("short") is True
        assert sf.is_safe("a" * 20) is False

    def test_filter_removes_unsafe(self, mixed_candidates):
        sf = BasicSafetyFilter()
        safe = sf.filter(mixed_candidates)
        for c in safe:
            assert sf.is_safe(c["text"])

    def test_filter_preserves_all_safe(self, safe_candidates):
        sf = BasicSafetyFilter()
        filtered = sf.filter(safe_candidates)
        assert len(filtered) == len(safe_candidates)

    def test_empty_blocklist_passes_everything(self):
        sf = BasicSafetyFilter(blocklist=[])
        assert sf.is_safe("how to make a bomb") is True

    def test_add_pattern_at_runtime(self):
        sf = BasicSafetyFilter(blocklist=[])
        sf.add_pattern(r"\bforbidden\b")
        assert sf.is_safe("this is a forbidden action") is False
        assert sf.is_safe("this is allowed") is True

    def test_check_with_reason_ok(self):
        sf = BasicSafetyFilter()
        safe, reason = sf.check_with_reason("Perfectly fine text.")
        assert safe is True
        assert reason == "ok"

    def test_check_with_reason_blocked(self):
        sf = BasicSafetyFilter()
        safe, reason = sf.check_with_reason("how to make a bomb")
        assert safe is False
        assert "blocklist" in reason.lower() or "pattern" in reason.lower()

    def test_dict_candidate(self):
        sf = BasicSafetyFilter()
        assert sf.is_safe({"text": "Fine text"}) is True
        assert sf.is_safe({"content": "how to make a bomb"}) is False

    def test_extra_check_integration(self):
        def no_numbers(text):
            import re
            if re.search(r"\d", text):
                return False, "contains digits"
            return True, "ok"

        sf = BasicSafetyFilter(blocklist=[])
        sf.add_check("no_numbers", no_numbers)
        assert sf.is_safe("Hello world") is True
        assert sf.is_safe("Hello 42") is False


# ============================================================================
# ValuationLayer
# ============================================================================

class TestValuationLayer:
    def test_filter_safe_removes_unsafe(self, mixed_candidates):
        vl = ValuationLayer()
        safe = vl.filter_safe(mixed_candidates)
        assert all(vl.is_safe(c) for c in safe)

    def test_ranked_returns_safe_only_by_default(self, mixed_candidates):
        vl = ValuationLayer()
        ranked = vl.ranked(mixed_candidates, context={"goal": "geography"})
        for c in ranked:
            assert vl.is_safe(c)

    def test_best_returns_top_safe_candidate(self, safe_candidates):
        vl = ValuationLayer()
        best = vl.best(safe_candidates, context={"goal": "temperature"})
        assert best is not None
        assert "_score" in best
        assert vl.is_safe(best)

    def test_best_on_all_unsafe_returns_none(self):
        vl = ValuationLayer()
        unsafe = [
            {"text": "how to make a bomb"},
            {"text": "how to exploit the system to bypass auth"},
        ]
        result = vl.best(unsafe)
        assert result is None

    def test_score_zeros_unsafe(self, mixed_candidates):
        vl = ValuationLayer()
        scores = vl.score(mixed_candidates)
        # second candidate is unsafe → score 0.0
        assert scores[1] == 0.0

    def test_score_nonzero_for_safe(self, safe_candidates):
        vl = ValuationLayer()
        scores = vl.score(safe_candidates)
        assert all(s > 0.0 for s in scores)

    def test_custom_reward_model(self, safe_candidates):
        custom_model = DriveBasedRewardModel(
            drives=[
                ("length", 1.0, lambda c, _: min(1.0, len(c.get("text", "")) / 200))
            ]
        )
        vl = ValuationLayer(reward_model=custom_model)
        scores = vl.score(safe_candidates)
        assert len(scores) == len(safe_candidates)

    def test_custom_safety_filter(self):
        strict_filter = BasicSafetyFilter(max_length=20)
        vl = ValuationLayer(safety_filter=strict_filter)
        candidates = [{"text": "Short"}, {"text": "A" * 50}]
        safe = vl.filter_safe(candidates)
        assert len(safe) == 1
        assert safe[0]["text"] == "Short"

    def test_ranked_includes_unsafe_when_requested(self, mixed_candidates):
        vl = ValuationLayer()
        ranked = vl.ranked(mixed_candidates, include_unsafe=True)
        # All 3 candidates present
        assert len(ranked) == len(mixed_candidates)

    def test_property_accessors(self):
        vl = ValuationLayer()
        from valuation.reward import DriveBasedRewardModel
        from valuation.safety import BasicSafetyFilter
        assert isinstance(vl.reward_model, DriveBasedRewardModel)
        assert isinstance(vl.safety_filter, BasicSafetyFilter)
