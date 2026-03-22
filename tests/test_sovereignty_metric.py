"""
Tests for SovereigntyMetrics — Priority 1 coverage.

Covers:
  - record()
  - sovereignty_rate()       — sliding-window behaviour
  - total_sovereignty_rate() — lifetime behaviour
  - is_on_track()            — roadmap milestone checks
  - target_for_month()       — interpolation
  - snapshot()               — dict structure
  - reset()
  - error handling
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hybrid.sovereignty_metrics import SovereigntyMetrics

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def metrics() -> SovereigntyMetrics:
    return SovereigntyMetrics()


# ============================================================================
# Basic recording
# ============================================================================


class TestSovereigntyMetricsRecord:

    def test_empty_init_has_zero_rate(self, metrics):
        assert metrics.total_sovereignty_rate() == 0.0
        assert metrics._total_nawal == 0
        assert metrics._total_deepseek == 0

    def test_record_nawal_increments_nawal_count(self, metrics):
        metrics.record("nawal")
        assert metrics._total_nawal == 1
        assert metrics._total_deepseek == 0

    def test_record_deepseek_increments_deepseek_count(self, metrics):
        metrics.record("deepseek")
        assert metrics._total_nawal == 0
        assert metrics._total_deepseek == 1

    def test_record_case_insensitive(self, metrics):
        metrics.record("Nawal")
        metrics.record("DEEPSEEK")
        assert metrics._total_nawal == 1
        assert metrics._total_deepseek == 1

    def test_record_invalid_raises_value_error(self, metrics):
        with pytest.raises(ValueError, match="nawal"):
            metrics.record("gpt4")

    def test_record_empty_string_raises(self, metrics):
        with pytest.raises(ValueError):
            metrics.record("")

    def test_record_whitespace_raises(self, metrics):
        with pytest.raises(ValueError):
            metrics.record("   ")


# ============================================================================
# Total sovereignty rate
# ============================================================================


class TestTotalSovereigntyRate:

    def test_all_nawal_is_100_percent(self, metrics):
        for _ in range(10):
            metrics.record("nawal")
        assert metrics.total_sovereignty_rate() == 1.0

    def test_all_deepseek_is_0_percent(self, metrics):
        for _ in range(5):
            metrics.record("deepseek")
        assert metrics.total_sovereignty_rate() == 0.0

    def test_8_nawal_2_deepseek_is_80_percent(self, metrics):
        for _ in range(8):
            metrics.record("nawal")
        for _ in range(2):
            metrics.record("deepseek")
        assert abs(metrics.total_sovereignty_rate() - 0.8) < 1e-9

    def test_one_of_each_50_percent(self, metrics):
        metrics.record("nawal")
        metrics.record("deepseek")
        assert abs(metrics.total_sovereignty_rate() - 0.5) < 1e-9

    def test_returns_float(self, metrics):
        metrics.record("nawal")
        assert isinstance(metrics.total_sovereignty_rate(), float)


# ============================================================================
# Sliding-window sovereignty rate
# ============================================================================


class TestWindowSovereigntyRate:

    def test_empty_window_returns_zero(self, metrics):
        assert metrics.sovereignty_rate(window_minutes=30) == 0.0

    def test_all_events_within_window(self, metrics):
        # Patch monotonic so all events appear recent
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=1000.0):
            metrics.record("nawal")
            metrics.record("nawal")
            metrics.record("deepseek")

        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=1001.0):
            rate = metrics.sovereignty_rate(window_minutes=30)
        assert abs(rate - 2 / 3) < 1e-9

    def test_old_events_excluded_from_window(self, metrics):
        # Record 5 deep-seek events 2 hours ago
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=0.0):
            for _ in range(5):
                metrics.record("deepseek")

        # Record 8 nawal events "now"
        now_ts = 2 * 60 * 60 + 100  # 2h + 100s later
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=now_ts):
            for _ in range(8):
                metrics.record("nawal")

        # Query with 30-min window — old deepseek events should be excluded
        query_ts = now_ts + 5
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=query_ts):
            rate = metrics.sovereignty_rate(window_minutes=30)
        # Only the 8 recent nawal events are in window
        assert rate == 1.0

    def test_partial_window_only_counts_recent(self, metrics):
        # 2 nawal events 1 hour ago
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=0.0):
            metrics.record("nawal")
            metrics.record("nawal")
        # 3 deepseek events now
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=3600.0):
            metrics.record("deepseek")
            metrics.record("deepseek")
            metrics.record("deepseek")
        # Query with 30-min window → only see 3 deepseek events
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=3601.0):
            rate = metrics.sovereignty_rate(window_minutes=30)
        assert rate == 0.0

    def test_default_window_used_when_not_specified(self):
        """sovereignty_rate() uses ctor-specified window when no arg given."""
        m = SovereigntyMetrics(window_minutes=60)
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=1000.0):
            m.record("nawal")
        with patch("hybrid.sovereignty_metrics.time.monotonic", return_value=1001.0):
            rate = m.sovereignty_rate()  # should use 60-minute window
        assert rate == 1.0


# ============================================================================
# Roadmap milestone tracking
# ============================================================================


class TestIsOnTrack:

    def test_on_track_at_month_1_exactly(self, metrics):
        """50% rate at month 1 → on track."""
        for _ in range(5):
            metrics.record("nawal")
        for _ in range(5):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=1) is True

    def test_not_on_track_at_month_1_below_target(self, metrics):
        """40% rate at month 1 → below 50% target → not on track."""
        for _ in range(4):
            metrics.record("nawal")
        for _ in range(6):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=1) is False

    def test_on_track_at_month_6_exactly(self, metrics):
        """95% rate at month 6 → exactly on track."""
        for _ in range(95):
            metrics.record("nawal")
        for _ in range(5):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=6) is True

    def test_not_on_track_month_6_low_rate(self, metrics):
        """70% rate at month 6 → below 95% target."""
        for _ in range(7):
            metrics.record("nawal")
        for _ in range(3):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=6) is False

    def test_on_track_month_12_at_90_percent(self, metrics):
        for _ in range(9):
            metrics.record("nawal")
        for _ in range(1):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=12) is True

    def test_not_on_track_month_12_below_90(self, metrics):
        for _ in range(85):
            metrics.record("nawal")
        for _ in range(15):
            metrics.record("deepseek")
        assert metrics.is_on_track(month=12) is False

    def test_empty_metrics_not_on_track(self, metrics):
        assert metrics.is_on_track(month=1) is False

    def test_is_on_track_returns_bool(self, metrics):
        metrics.record("nawal")
        result = metrics.is_on_track(month=1)
        assert isinstance(result, bool)


# ============================================================================
# Roadmap target interpolation
# ============================================================================


class TestTargetForMonth:

    def test_exact_milestone_1(self, metrics):
        assert metrics.target_for_month(1) == 0.50

    def test_exact_milestone_3(self, metrics):
        assert metrics.target_for_month(3) == 0.70

    def test_exact_milestone_6(self, metrics):
        assert metrics.target_for_month(6) == 0.95

    def test_exact_milestone_12(self, metrics):
        assert metrics.target_for_month(12) == 0.90

    def test_interpolation_month_2(self, metrics):
        """Month 2 is between 1 (50%) and 3 (70%) → 60%."""
        target = metrics.target_for_month(2)
        assert abs(target - 0.60) < 1e-9

    def test_interpolation_month_4(self, metrics):
        """Month 4 is between 3 (70%) and 6 (95%):
        frac=(4-3)/(6-3)=1/3  →  70 + 1/3*(95-70) = 70 + 8.333 = 78.333%."""
        target = metrics.target_for_month(4)
        expected = 0.70 + (1 / 3) * (0.95 - 0.70)
        assert abs(target - expected) < 1e-9

    def test_pre_milestone_uses_first_target(self, metrics):
        """Month 0 → before first milestone → same as month 1."""
        assert metrics.target_for_month(0) == 0.50

    def test_beyond_last_milestone_uses_last_target(self, metrics):
        """Month 24 → beyond month-12 milestone → 90%."""
        assert metrics.target_for_month(24) == 0.90


# ============================================================================
# snapshot() dict structure
# ============================================================================


class TestSnapshot:

    def test_returns_dict(self, metrics):
        snap = metrics.snapshot()
        assert isinstance(snap, dict)

    def test_all_required_keys_present(self, metrics):
        snap = metrics.snapshot()
        for key in [
            "window_minutes",
            "window_sovereignty_rate",
            "total_sovereignty_rate",
            "total_nawal",
            "total_deepseek",
            "total_queries",
        ]:
            assert key in snap, f"Missing key: {key}"

    def test_total_queries_equals_nawal_plus_deepseek(self, metrics):
        for _ in range(3):
            metrics.record("nawal")
        for _ in range(2):
            metrics.record("deepseek")
        snap = metrics.snapshot()
        assert snap["total_queries"] == 5
        assert snap["total_nawal"] == 3
        assert snap["total_deepseek"] == 2

    def test_rates_bounded_0_1(self, metrics):
        metrics.record("nawal")
        snap = metrics.snapshot()
        assert 0.0 <= snap["window_sovereignty_rate"] <= 1.0
        assert 0.0 <= snap["total_sovereignty_rate"] <= 1.0

    def test_custom_window_in_snapshot(self, metrics):
        snap = metrics.snapshot(window_minutes=60)
        assert snap["window_minutes"] == 60


# ============================================================================
# reset()
# ============================================================================


class TestReset:

    def test_reset_clears_all_state(self, metrics):
        for _ in range(10):
            metrics.record("nawal")
        for _ in range(5):
            metrics.record("deepseek")
        metrics.reset()
        assert metrics._total_nawal == 0
        assert metrics._total_deepseek == 0
        assert len(metrics._events) == 0

    def test_reset_sovereignty_rate_zero(self, metrics):
        metrics.record("nawal")
        metrics.reset()
        assert metrics.total_sovereignty_rate() == 0.0

    def test_recording_after_reset_works(self, metrics):
        metrics.record("nawal")
        metrics.reset()
        metrics.record("deepseek")
        assert metrics._total_deepseek == 1
        assert metrics._total_nawal == 0
