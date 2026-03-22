"""
Tests for the Hybrid Nawal Engine — Priority 1 coverage.

Covers:
  - ConfidenceScorer  (entropy, perplexity, length, language, composite)
  - IntelligentRouter (routing decisions, statistics, fallback logging)
  - HybridNawalEngine (end-to-end routing with mocked Nawal + teacher)

All Nawal model and DeepSeek teacher calls are mocked so these tests
run entirely in-process with no GPU or network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from hybrid.confidence import ConfidenceScorer
from hybrid.router import IntelligentRouter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def scorer() -> ConfidenceScorer:
    return ConfidenceScorer(confidence_threshold=0.75)


@pytest.fixture
def router(tmp_path) -> IntelligentRouter:
    log_file = str(tmp_path / "fallbacks.jsonl")
    return IntelligentRouter(
        confidence_threshold=0.75,
        log_fallbacks=True,
        fallback_log_path=log_file,
    )


def _peaked_logits(vocab_size: int = 100) -> torch.Tensor:
    """Logits where one token has a huge mass — high confidence."""
    logits = torch.zeros(1, vocab_size)
    logits[0, 0] = 50.0  # overwhelming mass on token 0
    return logits


def _uniform_logits(vocab_size: int = 100) -> torch.Tensor:
    """Uniform logits — maximum uncertainty."""
    return torch.zeros(1, vocab_size)


def _make_confidence_scores(overall: float) -> dict[str, float]:
    return {
        "overall": overall,
        "entropy": overall,
        "perplexity": overall,
        "length": overall,
        "language": overall,
        "should_use_nawal": overall >= 0.75,
    }


# ============================================================================
# ConfidenceScorer — Entropy
# ============================================================================


class TestConfidenceScorerEntropy:
    """Test entropy-based confidence computation."""

    def test_peaked_logits_high_confidence(self, scorer):
        """Highly peaked distribution → near-1 confidence."""
        logits = _peaked_logits()
        conf = scorer.compute_entropy(logits)
        assert (
            conf > 0.9
        ), f"Peaked logits should give very high confidence, got {conf:.4f}"

    def test_uniform_logits_low_confidence(self, scorer):
        """Uniform distribution → near-0 confidence (maximum entropy)."""
        logits = _uniform_logits()
        conf = scorer.compute_entropy(logits)
        assert conf < 0.1, f"Uniform logits should give low confidence, got {conf:.4f}"

    def test_entropy_bounded_0_1(self, scorer):
        """Entropy confidence must always be in [0, 1]."""
        for _ in range(10):
            logits = torch.randn(1, 50)
            conf = scorer.compute_entropy(logits)
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_3d_batch_input(self, scorer):
        """3-D logits [batch, seq, vocab] — use last token position."""
        logits_3d = torch.zeros(2, 10, 100)
        logits_3d[:, -1, 0] = 50.0  # peaked at last position
        conf = scorer.compute_entropy(logits_3d)
        assert conf > 0.9

    def test_higher_vocab_same_peaked(self, scorer):
        """Peak of same magnitude on larger vocab is still high confidence."""
        for vocab in (50, 200, 1000):
            logits = torch.zeros(1, vocab)
            logits[0, 0] = 50.0
            conf = scorer.compute_entropy(logits)
            assert (
                conf > 0.9
            ), f"Vocab={vocab}: expected high confidence, got {conf:.4f}"


# ============================================================================
# ConfidenceScorer — Perplexity
# ============================================================================


class TestConfidenceScorerPerplexity:
    """Test perplexity-based confidence computation."""

    def test_perfect_predictions_high_confidence(self, scorer):
        """When student perfectly predicts targets, confidence → 1."""
        vocab_size = 50
        batch, seq = 2, 8
        # Logits with large positive for the correct token
        logits = torch.zeros(batch, seq, vocab_size)
        targets = torch.zeros(batch, seq, dtype=torch.long)
        logits[:, :, 0] = 10.0  # strong prediction for token 0
        # targets already == 0

        conf = scorer.compute_perplexity(logits, targets)
        assert (
            conf > 0.8
        ), f"Perfect prediction should give high perplexity confidence, got {conf:.4f}"

    def test_wrong_predictions_lower_confidence(self, scorer):
        """When student predicts wrong tokens, confidence is lower."""
        vocab_size = 50
        batch, seq = 2, 8
        logits = torch.zeros(batch, seq, vocab_size)
        logits[:, :, 0] = 10.0  # strongly predicts token 0
        targets = torch.ones(batch, seq, dtype=torch.long)  # but target is token 1

        conf = scorer.compute_perplexity(logits, targets)
        assert (
            conf < 0.5
        ), f"Wrong predictions should give lower confidence, got {conf:.4f}"

    def test_perplexity_confidence_bounded(self, scorer):
        """Perplexity confidence must be in [0, 1]."""
        logits = torch.randn(2, 10, 30)
        targets = torch.randint(0, 30, (2, 10))
        conf = scorer.compute_perplexity(logits, targets)
        assert 0.0 <= conf <= 1.0


# ============================================================================
# ConfidenceScorer — Length
# ============================================================================


class TestConfidenceScorerLength:
    """Test length-based confidence computation."""

    def test_zero_length_full_confidence(self, scorer):
        assert scorer.compute_length_confidence(0) == 1.0

    def test_half_max_gives_0_75(self, scorer):
        # formula: 1.0 - length / (2 * max_length)
        conf = scorer.compute_length_confidence(512, max_length=1024)
        assert abs(conf - 0.75) < 1e-6

    def test_at_max_length_gives_0_5(self, scorer):
        conf = scorer.compute_length_confidence(1024, max_length=1024)
        assert abs(conf - 0.5) < 1e-6

    def test_over_max_length_zero(self, scorer):
        conf = scorer.compute_length_confidence(2000, max_length=1024)
        assert conf == 0.0

    def test_length_bounded_0_1(self, scorer):
        for length in [0, 100, 500, 1024, 2048]:
            conf = scorer.compute_length_confidence(length)
            assert 0.0 <= conf <= 1.0, f"length={length}: conf={conf}"


# ============================================================================
# ConfidenceScorer — Language
# ============================================================================


class TestConfidenceScorerLanguage:
    """Test language-based confidence computation."""

    def test_native_languages_full_confidence(self, scorer):
        for lang in ["en", "es", "bzj", "cab", "mop"]:
            assert scorer.compute_language_confidence(lang) == 1.0, f"lang={lang}"

    def test_unknown_language_reduced_confidence(self, scorer):
        conf = scorer.compute_language_confidence("zh")
        assert conf == 0.5

    def test_custom_native_languages(self):
        scorer = ConfidenceScorer()
        scorer.native_languages = ["fr", "de"]
        assert scorer.compute_language_confidence("fr") == 1.0
        assert scorer.compute_language_confidence("en") == 0.5


# ============================================================================
# ConfidenceScorer — Full composite score
# ============================================================================


class TestConfidenceScorerComposite:
    """Test full compute_confidence() output."""

    def test_returns_all_keys(self, scorer):
        logits = _peaked_logits()
        result = scorer.compute_confidence(logits, detected_language="en")
        for key in [
            "overall",
            "entropy",
            "perplexity",
            "length",
            "language",
            "should_use_nawal",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_overall_bounded(self, scorer):
        logits = torch.randn(1, 100)
        result = scorer.compute_confidence(logits)
        assert 0.0 <= result["overall"] <= 1.0

    def test_high_confidence_recommends_nawal(self, scorer):
        """Peaked logits + native language + short → should_use_nawal = True."""
        scorer_low = ConfidenceScorer(confidence_threshold=0.1)
        logits = _peaked_logits(vocab_size=100)
        result = scorer_low.compute_confidence(logits, detected_language="en")
        assert result["should_use_nawal"] is True

    def test_low_threshold_easy_to_pass(self, scorer):
        scorer_low = ConfidenceScorer(confidence_threshold=0.0)
        logits = _uniform_logits()
        result = scorer_low.compute_confidence(logits, detected_language="en")
        assert result["should_use_nawal"] is True

    def test_high_threshold_hard_to_pass(self, scorer):
        scorer_strict = ConfidenceScorer(confidence_threshold=1.0)
        logits = torch.randn(1, 100)
        result = scorer_strict.compute_confidence(logits, detected_language="en")
        assert result["should_use_nawal"] is False

    def test_with_input_ids_uses_perplexity(self, scorer):
        vocab_size = 50
        logits = torch.zeros(1, 8, vocab_size)
        logits[:, :, 0] = 5.0
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        result = scorer.compute_confidence(
            logits, input_ids=input_ids, detected_language="en"
        )
        # perplexity sub-score should be computed (not 0.5 default)
        assert result["perplexity"] != 0.5

    def test_update_threshold(self, scorer):
        scorer.update_threshold(0.9)
        assert scorer.threshold == 0.9
        scorer.update_threshold(1.5)  # clamped to 1.0
        assert scorer.threshold == 1.0
        scorer.update_threshold(-0.5)  # clamped to 0.0
        assert scorer.threshold == 0.0

    def test_weights_sum_to_1(self):
        """Default weights should sum to 1.0."""
        s = ConfidenceScorer()
        total = (
            s.entropy_weight + s.perplexity_weight + s.length_weight + s.language_weight
        )
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


# ============================================================================
# IntelligentRouter — Routing decisions
# ============================================================================


class TestIntelligentRouterRouting:
    """Test routing decisions."""

    def test_high_confidence_routes_to_nawal(self, router):
        scores = _make_confidence_scores(0.9)
        decision, meta = router.route("test query", torch.zeros(1, 10), scores)
        assert decision == "nawal"
        assert meta["decision"] == "nawal"

    def test_low_confidence_routes_to_deepseek(self, router):
        scores = _make_confidence_scores(0.3)
        decision, meta = router.route("test query", torch.zeros(1, 10), scores)
        assert decision == "deepseek"
        assert meta["decision"] == "deepseek"

    def test_exactly_at_threshold_routes_to_nawal(self, router):
        """Exactly at threshold (0.75) should route to Nawal (>=)."""
        scores = _make_confidence_scores(0.75)
        decision, _ = router.route("query", torch.zeros(1, 10), scores)
        assert decision == "nawal"

    def test_just_below_threshold_routes_to_deepseek(self, router):
        scores = _make_confidence_scores(0.749)
        decision, _ = router.route("query", torch.zeros(1, 10), scores)
        assert decision == "deepseek"

    def test_metadata_contains_required_fields(self, router):
        scores = _make_confidence_scores(0.8)
        _, meta = router.route("hello", torch.zeros(1, 10), scores)
        for key in ["timestamp", "decision", "confidence", "threshold", "query_length"]:
            assert key in meta, f"Missing metadata key: {key}"


# ============================================================================
# IntelligentRouter — Statistics
# ============================================================================


class TestIntelligentRouterStatistics:
    """Test statistics tracking."""

    def test_initial_stats_are_zero(self, router):
        stats = router.get_statistics()
        assert stats["total_queries"] == 0
        assert stats["nawal_handled"] == 0
        assert stats["deepseek_fallback"] == 0
        assert stats["sovereignty_rate"] == 0.0

    def test_nawal_count_increments(self, router):
        scores = _make_confidence_scores(0.9)
        router.route("q1", torch.zeros(1, 5), scores)
        router.route("q2", torch.zeros(1, 5), scores)
        stats = router.get_statistics()
        assert stats["nawal_handled"] == 2
        assert stats["total_queries"] == 2

    def test_deepseek_count_increments(self, router):
        scores = _make_confidence_scores(0.3)
        router.route("q1", torch.zeros(1, 5), scores)
        stats = router.get_statistics()
        assert stats["deepseek_fallback"] == 1

    def test_sovereignty_rate_calculation(self, router):
        """6 nawal + 4 deepseek = 60% sovereignty."""
        for _ in range(6):
            router.route("q", torch.zeros(1, 5), _make_confidence_scores(0.9))
        for _ in range(4):
            router.route("q", torch.zeros(1, 5), _make_confidence_scores(0.1))
        stats = router.get_statistics()
        assert abs(stats["sovereignty_rate"] - 0.6) < 1e-9

    def test_100_percent_sovereignty(self, router):
        for _ in range(5):
            router.route("q", torch.zeros(1, 5), _make_confidence_scores(0.9))
        assert router.get_statistics()["sovereignty_rate"] == 1.0

    def test_zero_percent_sovereignty(self, router):
        for _ in range(5):
            router.route("q", torch.zeros(1, 5), _make_confidence_scores(0.1))
        assert router.get_statistics()["sovereignty_rate"] == 0.0

    def test_reset_statistics(self, router):
        router.route("q", torch.zeros(1, 5), _make_confidence_scores(0.9))
        router.reset_statistics()
        stats = router.get_statistics()
        assert stats["total_queries"] == 0
        assert stats["nawal_handled"] == 0
        assert stats["sovereignty_rate"] == 0.0

    def test_update_threshold(self, router):
        router.update_threshold(0.9)
        assert router.threshold == 0.9

    def test_update_threshold_clamped(self, router):
        router.update_threshold(2.0)
        assert router.threshold == 1.0
        router.update_threshold(-1.0)
        assert router.threshold == 0.0

    def test_statistics_include_timestamp(self, router):
        stats = router.get_statistics()
        assert "timestamp" in stats


# ============================================================================
# IntelligentRouter — Fallback logging
# ============================================================================


class TestIntelligentRouterFallbackLogging:
    """Test that fallback queries are written to JSONL."""

    def test_fallback_creates_log_file(self, router, tmp_path):
        scores = _make_confidence_scores(0.2)
        router.route("I need help with something complex", torch.zeros(1, 5), scores)
        assert Path(router.fallback_log_path).exists()

    def test_fallback_log_contains_valid_json(self, router):
        scores = _make_confidence_scores(0.1)
        router.route("test fallback", torch.zeros(1, 5), scores)
        with open(router.fallback_log_path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert "query_hash" in entry
        assert "confidence" in entry
        assert len(entry["query_hash"]) == 16  # truncated sha256

    def test_fallback_does_not_log_pii(self, router):
        """Query content must be hashed — not stored in clear text."""
        query = "My SSN is 123-45-6789"
        scores = _make_confidence_scores(0.1)
        router.route(query, torch.zeros(1, 5), scores)
        content = Path(router.fallback_log_path).read_text()
        assert query not in content

    def test_nawal_queries_not_logged(self, router):
        """High-confidence (Nawal) queries should NOT appear in fallback log."""
        scores = _make_confidence_scores(0.95)
        router.route("simple query", torch.zeros(1, 5), scores)
        # Log file should not exist or be empty
        if Path(router.fallback_log_path).exists():
            assert Path(router.fallback_log_path).stat().st_size == 0

    def test_export_fallback_logs(self, router, tmp_path):
        scores = _make_confidence_scores(0.1)
        router.route("q1", torch.zeros(1, 5), scores)
        router.route("q2", torch.zeros(1, 5), scores)

        export_path = str(tmp_path / "export.jsonl")
        count = router.export_fallback_logs(export_path)
        assert count == 2
        assert Path(export_path).exists()

    def test_export_no_logs_returns_zero(self, router, tmp_path):
        """No fallbacks logged → export returns 0."""
        export_path = str(tmp_path / "export.jsonl")
        router2 = IntelligentRouter(
            fallback_log_path=str(tmp_path / "nonexistent.jsonl")
        )
        count = router2.export_fallback_logs(export_path)
        assert count == 0

    def test_logging_disabled(self, tmp_path):
        """When log_fallbacks=False, no file should be created."""
        router_nolog = IntelligentRouter(
            confidence_threshold=0.75,
            log_fallbacks=False,
            fallback_log_path=str(tmp_path / "should_not_exist.jsonl"),
        )
        scores = _make_confidence_scores(0.1)
        router_nolog.route("query", torch.zeros(1, 5), scores)
        assert not Path(router_nolog.fallback_log_path).exists()


# ============================================================================
# HybridNawalEngine — with mocked Nawal and teacher
# ============================================================================


def _make_mock_nawal(
    vocab_size: int = 100, seq_len: int = 5, confidence: str = "high"
) -> MagicMock:
    """Build a minimal mock Nawal that satisfies HybridNawalEngine's API."""
    mock = MagicMock()

    # Config
    mock.config.model_size = "small"
    mock.config.num_parameters = 117_000_000
    mock.config.num_layers = 4

    # Tokenizer
    mock.tokenizer.encode.return_value = torch.ones(1, seq_len, dtype=torch.long)

    # Forward pass — return peaked or uniform logits
    if confidence == "high":
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, :, 0] = 50.0  # very peaked → high confidence
    else:
        logits = torch.zeros(1, seq_len, vocab_size)  # uniform → low confidence

    mock.forward.return_value = {"logits": logits}

    # Generate
    mock.generate.return_value = ["Belmopan is the capital of Belize."]

    # No language detector
    mock.language_detector = None

    return mock


def _make_mock_teacher() -> MagicMock:
    teacher = MagicMock()
    teacher.generate.return_value = {"text": "DeepSeek response here.", "cached": False}
    return teacher


class TestHybridNawalEngineInit:
    """Test engine initialisation."""

    def test_init_without_teacher(self):
        """Engine starts without pre-loading teacher."""
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal, auto_load_teacher=False)
        assert engine.teacher is None
        assert engine.nawal is nawal

    def test_init_with_teacher(self):
        nawal = _make_mock_nawal()
        teacher = _make_mock_teacher()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            teacher_model=teacher,
        )
        assert engine.teacher is teacher

    def test_confidence_scorer_created(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal)
        assert isinstance(engine.confidence_scorer, ConfidenceScorer)

    def test_router_created(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal)
        assert isinstance(engine.router, IntelligentRouter)


class TestHybridNawalEngineGenerate:
    """Test generate routing logic."""

    def test_high_confidence_uses_nawal(self):
        """Peaked logits → routes to Nawal, Nawal.generate called."""
        nawal = _make_mock_nawal(confidence="high")
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.1,  # low threshold so peaked logits pass
            enable_distillation_logging=False,
        )
        result = engine.generate("What is DALLA?")
        assert result["model_used"] == "nawal"
        nawal.generate.assert_called_once()

    def test_low_confidence_uses_deepseek(self):
        """Uniform logits → below threshold → routes to DeepSeek teacher."""
        nawal = _make_mock_nawal(confidence="low")
        teacher = _make_mock_teacher()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            teacher_model=teacher,
            confidence_threshold=0.99,  # high threshold so uniform logits fail
            enable_distillation_logging=False,
        )
        result = engine.generate("Explain quantum entanglement in detail.")
        assert result["model_used"] == "deepseek"
        teacher.generate.assert_called_once()

    def test_response_contains_required_keys(self):
        nawal = _make_mock_nawal(confidence="high")
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.0,
            enable_distillation_logging=False,
        )
        result = engine.generate("hello")
        for key in ["text", "model_used", "confidence", "latency_ms", "metadata"]:
            assert key in result, f"Missing key: {key}"

    def test_confidence_is_float_in_range(self):
        nawal = _make_mock_nawal(confidence="high")
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.0,
            enable_distillation_logging=False,
        )
        result = engine.generate("test")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_latency_ms_positive(self):
        nawal = _make_mock_nawal(confidence="high")
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.0,
            enable_distillation_logging=False,
        )
        result = engine.generate("test")
        assert result["latency_ms"] >= 0.0


class TestHybridNawalEngineLazyTeacher:
    """Test lazy teacher loading."""

    def test_teacher_loaded_on_fallback(self, tmp_path):
        """Teacher is created on first fallback, not at init."""
        nawal = _make_mock_nawal(confidence="low")
        teacher = _make_mock_teacher()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.99,  # always fallback
            auto_load_teacher=False,
            enable_distillation_logging=False,
        )
        assert engine.teacher is None

        # Inject the mock teacher before the fallback happens
        engine.teacher = teacher
        result = engine.generate("complex query")
        assert result["model_used"] == "deepseek"


class TestHybridNawalEngineStatistics:
    """Test statistics and configuration."""

    def test_get_statistics_structure(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal)
        stats = engine.get_statistics()
        assert "routing" in stats
        assert "nawal_config" in stats
        assert "teacher_loaded" in stats

    def test_statistics_sovereignty_rate_starts_zero(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal)
        stats = engine.get_statistics()
        assert stats["routing"]["sovereignty_rate"] == 0.0

    def test_update_threshold_propagates_to_both(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal, confidence_threshold=0.75)
        engine.update_threshold(0.9)
        assert engine.confidence_scorer.threshold == 0.9
        assert engine.router.threshold == 0.9

    def test_teacher_loaded_flag_false(self):
        nawal = _make_mock_nawal()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal, auto_load_teacher=False)
        assert engine.get_statistics()["teacher_loaded"] is False

    def test_teacher_loaded_flag_true(self):
        nawal = _make_mock_nawal()
        teacher = _make_mock_teacher()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(nawal_model=nawal, teacher_model=teacher)
        assert engine.get_statistics()["teacher_loaded"] is True


class TestHybridNawalEngineSovereigntyTracking:
    """Verify sovereignty rate accumulates correctly in engine."""

    def test_sovereignty_improves_with_nawal_decisions(self):
        nawal = _make_mock_nawal(confidence="high")
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            confidence_threshold=0.0,  # always Nawal
            enable_distillation_logging=False,
        )
        for _ in range(5):
            engine.generate("test")
        stats = engine.get_statistics()
        assert stats["routing"]["sovereignty_rate"] == 1.0

    def test_mixed_routing_tracks_correctly(self):
        # Use high confidence for all → 100% sovereign
        nawal = _make_mock_nawal(confidence="high")
        teacher = _make_mock_teacher()
        from hybrid.engine import HybridNawalEngine

        engine = HybridNawalEngine(
            nawal_model=nawal,
            teacher_model=teacher,
            confidence_threshold=0.0,
            enable_distillation_logging=False,
        )
        for _ in range(3):
            engine.generate("test")

        # Now switch to always-fallback
        engine.update_threshold(1.0)
        for _ in range(2):
            engine.generate("test")

        stats = engine.get_statistics()
        # 3 nawal + 2 deepseek → 60%
        assert abs(stats["routing"]["sovereignty_rate"] - 0.6) < 1e-9
