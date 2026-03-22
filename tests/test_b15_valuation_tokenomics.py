"""
B15 — Valuation & Tokenomics Rewards audit tests.

Covers:
  C15.1 - Quality scoring weight consistency (40/30/30)
  C15.2 - Accuracy score computation
  C15.3 - Timeliness score computation
  C15.4 - Token denomination & overflow
"""

from __future__ import annotations

import inspect
import re

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# C15.1 — Weight Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestC151WeightConsistency:
    """Weights used in FitnessScores.calculate_overall match the constants."""

    def test_calculate_overall_uses_module_constants(self):
        """calculate_overall references QUALITY_WEIGHT / TIMELINESS_WEIGHT / HONESTY_WEIGHT."""
        from blockchain.rewards import FitnessScores

        src = inspect.getsource(FitnessScores.calculate_overall)
        assert "QUALITY_WEIGHT" in src
        assert "TIMELINESS_WEIGHT" in src
        assert "HONESTY_WEIGHT" in src

    def test_no_hardcoded_weights_in_calculate_overall(self):
        """No raw 0.4 / 0.3 literals in calculate_overall body."""
        from blockchain.rewards import FitnessScores

        src = inspect.getsource(FitnessScores.calculate_overall)
        # Strip the docstring to avoid false positives
        src_lines = src.split("\n")
        body = "\n".join(
            l for l in src_lines if not l.strip().startswith(('"""', "'''", "#"))
        )
        assert "0.4" not in body, "Hardcoded 0.4 found — should use QUALITY_WEIGHT"
        assert (
            "0.3" not in body
        ), "Hardcoded 0.3 found — should use TIMELINESS/HONESTY_WEIGHT"

    def test_valuation_module_has_no_tokenomics_weights(self):
        """valuation/ must NOT define its own 40/30/30 weights — those belong in blockchain/rewards."""
        import valuation.reward as vr

        src = inspect.getsource(vr)
        assert "QUALITY_WEIGHT" not in src
        assert "TIMELINESS_WEIGHT" not in src
        assert "HONESTY_WEIGHT" not in src


# ═══════════════════════════════════════════════════════════════════════════
# C15.2 — Accuracy Score Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestC152AccuracyComputation:
    """FitnessScores correctly computes and clamps quality/accuracy scores."""

    def test_calculate_overall_exact_formula(self):
        """Verify the weighted-sum formula with known values."""
        from blockchain.rewards import FitnessScores

        fs = FitnessScores(quality=100, timeliness=0, honesty=0)
        assert abs(fs.calculate_overall() - 40.0) < 1e-10

        fs = FitnessScores(quality=0, timeliness=100, honesty=0)
        assert abs(fs.calculate_overall() - 30.0) < 1e-10

        fs = FitnessScores(quality=0, timeliness=0, honesty=100)
        assert abs(fs.calculate_overall() - 30.0) < 1e-10

    def test_clamping_negative_quality(self):
        """Negative quality is clamped to 0."""
        from blockchain.rewards import FitnessScores

        fs = FitnessScores(quality=-50, timeliness=50, honesty=50)
        overall = fs.calculate_overall()
        assert overall >= 0

    def test_clamping_over_hundred_quality(self):
        """quality > 100 is clamped to 100."""
        from blockchain.rewards import FitnessScores

        fs = FitnessScores(quality=200, timeliness=50, honesty=50)
        overall = fs.calculate_overall()
        assert overall <= 100

    def test_validate_rejects_out_of_range(self):
        """validate() returns errors for any score outside [0, 100]."""
        from blockchain.rewards import FitnessScores

        fs = FitnessScores(quality=-1, timeliness=101, honesty=50)
        errors = fs.validate()
        assert len(errors) == 2


# ═══════════════════════════════════════════════════════════════════════════
# C15.3 — Timeliness Score Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestC153TimelinessComputation:
    """Timeliness score is validated and weighted correctly."""

    def test_timeliness_contributes_30_percent(self):
        from blockchain.rewards import FitnessScores

        # All scores 100 → overall = 100. Timeliness at 0 → overall = 70
        fs_full = FitnessScores(quality=100, timeliness=100, honesty=100)
        fs_no_t = FitnessScores(quality=100, timeliness=0, honesty=100)
        diff = fs_full.calculate_overall() - fs_no_t.calculate_overall()
        assert abs(diff - 30.0) < 1e-10

    def test_training_submission_validates_timeliness(self):
        """TrainingSubmission.validate() rejects timeliness outside [0, 100]."""
        from blockchain.staking_connector import TrainingSubmission

        sub = TrainingSubmission(
            participant_id="5GTest",
            round_number=1,
            genome_id="g1",
            samples_trained=100,
            training_time=10.0,
            quality_score=50,
            timeliness_score=150,  # out of range
            honesty_score=50,
            fitness_score=50,
            model_hash="abc123",
        )
        errors = sub.validate()
        assert any("timeliness_score" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# C15.4 — Token Denomination & Overflow
# ═══════════════════════════════════════════════════════════════════════════


class TestC154DenominationOverflow:
    """PLANCK_PER_DALLA is consistent across all modules and fits in int64."""

    def test_rewards_denomination_is_1e12(self):
        from blockchain.rewards import PLANCK_PER_DALLA, DALLA_DECIMALS

        assert DALLA_DECIMALS == 12
        assert PLANCK_PER_DALLA == 10**12

    def test_payroll_uses_own_denomination_constant(self):
        """payroll_connector defines PAYROLL_PLANCK_PER_DALLA (10^8, not 10^12)."""
        from blockchain.payroll_connector import PAYROLL_PLANCK_PER_DALLA

        assert PAYROLL_PLANCK_PER_DALLA == 10**8

    def test_payroll_tax_threshold_uses_constant(self):
        """Payroll tax threshold is expressed via PAYROLL_PLANCK_PER_DALLA, not a magic number."""
        import blockchain.payroll_connector as pc

        src = inspect.getsource(pc.PayrollConnector.calculate_tax_withholding)
        assert "PAYROLL_PLANCK_PER_DALLA" in src
        assert "2600000000000" not in src, "Hardcoded threshold found"

    def test_staking_connector_no_magic_1e12(self):
        """staking_connector must use PLANCK_PER_DALLA, not hardcoded 1e12."""
        import blockchain.staking_connector as sc

        src = inspect.getsource(sc)
        # Allow 1e12 only in comments, not in executable code
        lines = src.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "1e12" in line and "PLANCK_PER_DALLA" not in line:
                pytest.fail(
                    f"Line {i}: hardcoded 1e12 found — use PLANCK_PER_DALLA instead"
                )

    def test_max_reward_fits_int64(self):
        """Maximum possible reward per round fits comfortably in int64."""
        from blockchain.rewards import (
            BASE_REWARD_DALLA,
            MAX_STAKE_MULTIPLIER,
            PLANCK_PER_DALLA,
        )

        # Worst case: perfect fitness (1.0) × max stake multiplier
        max_reward_dalla = BASE_REWARD_DALLA * 1.0 * MAX_STAKE_MULTIPLIER
        max_reward_planck = int(max_reward_dalla * PLANCK_PER_DALLA)
        assert max_reward_planck < 2**63 - 1, "Reward exceeds int64"

    def test_dalla_to_planck_is_integer(self):
        """dalla_to_planck always returns an integer."""
        from blockchain.rewards import dalla_to_planck

        for v in [0.0, 0.001, 1.0, 999.999, 10_000.0]:
            result = dalla_to_planck(v)
            assert isinstance(result, int)

    def test_reward_calculation_planck_is_int(self):
        """RewardCalculator.calculate_reward produces integer Planck."""
        from blockchain.rewards import FitnessScores, RewardCalculator, PLANCK_PER_DALLA

        calc = RewardCalculator()
        fs = FitnessScores(quality=75, timeliness=80, honesty=85)
        rc = calc.calculate_reward("test", 1, fs, 5000 * PLANCK_PER_DALLA)
        assert isinstance(rc.total_reward_planck, int)

    def test_community_connector_uses_1e12(self):
        """community_connector uses 10**12 denomination (consistent with rewards)."""
        import blockchain.community_connector as cc

        src = inspect.getsource(cc)
        # Should use 10**12, not 10**8
        assert "100000000" not in src or "10**12" in src
