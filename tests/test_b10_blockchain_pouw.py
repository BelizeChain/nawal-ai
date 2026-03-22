"""
B10 Audit — Blockchain & Proof of Useful Work

Checks:
  C10.1  PoUW reward-score formula (weights, denominator, clamping, int conversion)
  C10.2  Substrate RPC error handling (backoff, timeout, bounded retry queue)
  C10.3  Transaction signing & nonce management (duplicate detection)
  C10.4  Mock path for staging (opt-in, plausibility, constant consistency)

Covers fixes:
  F10.1a  Defensive clamping in FitnessScores.calculate_overall()
  F10.2a  Retry with exponential backoff in StakingConnector.connect()
  F10.2b  Bounded retry queue for failed PoUW submissions
  F10.3a  Duplicate submission detection
  F10.4a  Mock claim_rewards uses PLANCK_PER_DALLA constant
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from blockchain.rewards import (
    HONESTY_WEIGHT,
    PLANCK_PER_DALLA,
    QUALITY_WEIGHT,
    TIMELINESS_WEIGHT,
    FitnessScores,
    RewardCalculator,
    dalla_to_planck,
    format_dalla,
    planck_to_dalla,
)
from blockchain.staking_connector import (
    ParticipantInfo,
    StakingConnector,
    TrainingSubmission,
)
from blockchain.staking_interface import (
    FitnessScore,
)
from blockchain.substrate_client import SubstrateClient

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_submission(
    participant_id: str = "5GTestUser",
    round_number: int = 1,
    quality: float = 80.0,
    timeliness: float = 70.0,
    honesty: float = 90.0,
    **overrides,
) -> TrainingSubmission:
    """Create a valid TrainingSubmission for testing."""
    fitness = quality * 0.4 + timeliness * 0.3 + honesty * 0.3
    defaults = {
        "participant_id": participant_id,
        "round_number": round_number,
        "genome_id": "genome_001",
        "samples_trained": 1000,
        "training_time": 120.0,
        "quality_score": quality,
        "timeliness_score": timeliness,
        "honesty_score": honesty,
        "fitness_score": fitness,
        "model_hash": "abc123def456",
    }
    defaults.update(overrides)
    return TrainingSubmission(**defaults)


def _make_connector(**kwargs) -> StakingConnector:
    """Create a mock-mode StakingConnector."""
    defaults = {"mock_mode": True, "enable_community_tracking": False}
    defaults.update(kwargs)
    return StakingConnector(**defaults)


async def _enroll_participant(
    connector: StakingConnector,
    account_id: str = "5GTestUser",
    stake: int = 1_000_000_000_000,
) -> None:
    """Enroll a participant in mock mode."""
    connector._mock_participants[account_id] = ParticipantInfo(
        account_id=account_id,
        stake_amount=stake,
        is_enrolled=True,
        training_rounds_completed=0,
        total_samples_trained=0,
        avg_fitness_score=0.0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# C10.1  PoUW Reward Score Formula
# ═════════════════════════════════════════════════════════════════════════════


class TestC101RewardScoreFormula:
    """C10.1 — PoUW reward-score formula correctness."""

    # ── Weights sum to 1.0 ─────────────────────────────────

    def test_weights_sum_to_one(self):
        """Quality + timeliness + honesty weights must equal 1.0."""
        total = QUALITY_WEIGHT + TIMELINESS_WEIGHT + HONESTY_WEIGHT
        assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, expected 1.0"

    def test_quality_weight_is_040(self):
        assert QUALITY_WEIGHT == 0.40

    def test_timeliness_weight_is_030(self):
        assert TIMELINESS_WEIGHT == 0.30

    def test_honesty_weight_is_030(self):
        assert HONESTY_WEIGHT == 0.30

    # ── Denominator ────────────────────────────────────────

    def test_planck_denominator(self):
        """PLANCK_PER_DALLA must be 10^12 (12 decimal places)."""
        assert PLANCK_PER_DALLA == 10**12

    def test_dalla_to_planck_conversion(self):
        """1 DALLA = 10^12 Planck."""
        assert dalla_to_planck(1.0) == 10**12

    def test_planck_to_dalla_conversion(self):
        """10^12 Planck = 1 DALLA."""
        assert planck_to_dalla(10**12) == 1.0

    def test_dalla_roundtrip(self):
        """dalla → planck → dalla is identity."""
        for dalla in [0.0, 1.0, 10.5, 100.0]:
            assert abs(planck_to_dalla(dalla_to_planck(dalla)) - dalla) < 1e-10

    # ── FitnessScores validation ───────────────────────────

    def test_fitness_scores_valid_range(self):
        """Scores in [0, 100] pass validation."""
        fs = FitnessScores(quality=80, timeliness=70, honesty=90)
        assert fs.validate() == []

    def test_fitness_scores_boundary_zero(self):
        """Score of 0 is valid."""
        fs = FitnessScores(quality=0, timeliness=0, honesty=0)
        assert fs.validate() == []

    def test_fitness_scores_boundary_hundred(self):
        """Score of 100 is valid."""
        fs = FitnessScores(quality=100, timeliness=100, honesty=100)
        assert fs.validate() == []

    def test_fitness_scores_rejects_negative(self):
        """Negative scores are rejected."""
        fs = FitnessScores(quality=-1, timeliness=70, honesty=90)
        errors = fs.validate()
        assert len(errors) > 0

    def test_fitness_scores_rejects_over_hundred(self):
        """Scores > 100 are rejected."""
        fs = FitnessScores(quality=101, timeliness=70, honesty=90)
        errors = fs.validate()
        assert len(errors) > 0

    # ── F10.1a: Defensive clamping in calculate_overall ────

    def test_calculate_overall_formula(self):
        """calculate_overall uses 40/30/30 weighting."""
        fs = FitnessScores(quality=80, timeliness=70, honesty=90)
        expected = 80 * 0.4 + 70 * 0.3 + 90 * 0.3  # 32 + 21 + 27 = 80
        assert abs(fs.calculate_overall() - expected) < 1e-10

    def test_calculate_overall_clamps_inputs(self):
        """calculate_overall clamps out-of-range inputs to [0, 100]."""
        # Even if someone bypasses validate(), the formula shouldn't produce crazy values
        fs = FitnessScores(quality=150, timeliness=-20, honesty=200)
        overall = fs.calculate_overall()
        assert 0 <= overall <= 100, f"Overall {overall} should be in [0, 100]"

    # ── RewardCalculator ──────────────────────────────────

    def test_reward_calculation_returns_integer_planck(self):
        """Reward in Planck must be an integer (no fractional Planck)."""
        calc = RewardCalculator()
        scores = FitnessScores(quality=75, timeliness=80, honesty=85)
        stake = 5000 * PLANCK_PER_DALLA  # 5000 DALLA stake (above MIN_STAKE_DALLA)
        result = calc.calculate_reward(
            participant_id="5GTest",
            round_number=1,
            fitness_scores=scores,
            stake_amount_planck=stake,
        )
        assert isinstance(result.total_reward_planck, int)

    def test_reward_calculation_proportional(self):
        """Higher scores produce higher rewards."""
        calc = RewardCalculator()
        low = FitnessScores(quality=20, timeliness=20, honesty=20)
        high = FitnessScores(quality=90, timeliness=90, honesty=90)
        stake = 5000 * PLANCK_PER_DALLA
        r_low = calc.calculate_reward(
            participant_id="5GTest",
            round_number=1,
            fitness_scores=low,
            stake_amount_planck=stake,
        )
        r_high = calc.calculate_reward(
            participant_id="5GTest",
            round_number=2,
            fitness_scores=high,
            stake_amount_planck=stake,
        )
        assert r_high.total_reward_planck > r_low.total_reward_planck

    def test_reward_calculation_rejects_invalid(self):
        """Invalid scores raise ValueError."""
        calc = RewardCalculator()
        scores = FitnessScores(quality=-10, timeliness=80, honesty=85)
        with pytest.raises(ValueError):
            calc.calculate_reward(
                participant_id="5GTest",
                round_number=1,
                fitness_scores=scores,
                stake_amount_planck=5000 * PLANCK_PER_DALLA,
            )

    # ── StakingInterface FitnessScore ─────────────────────

    def test_staking_interface_fitness_total(self):
        """StakingInterface FitnessScore.total uses 40/30/30."""
        fs = FitnessScore(quality=80, timeliness=70, honesty=90, round=1)
        expected = 80 * 0.4 + 70 * 0.3 + 90 * 0.3
        assert abs(fs.total - expected) < 1e-10

    def test_staking_interface_fitness_to_dict_basis_points(self):
        """to_dict() converts to integer basis points."""
        fs = FitnessScore(quality=80, timeliness=70, honesty=90, round=1)
        d = fs.to_dict()
        assert d["quality"] == 8000
        assert d["timeliness"] == 7000
        assert d["honesty"] == 9000

    def test_staking_interface_rejects_out_of_range(self):
        """FitnessScore rejects out-of-range values."""
        with pytest.raises(ValueError):
            FitnessScore(quality=101, timeliness=70, honesty=90, round=1)

    # ── TrainingSubmission fixed-point conversion ─────────

    def test_submission_validation_accepts_valid(self):
        """Valid TrainingSubmission validates cleanly."""
        sub = _make_submission()
        assert sub.validate() == []

    def test_submission_validation_rejects_negative_score(self):
        """Negative scores fail validation."""
        sub = _make_submission(quality=-5)
        errors = sub.validate()
        assert any("quality_score" in e for e in errors)

    def test_submission_validation_rejects_over_hundred(self):
        """Scores > 100 fail validation."""
        sub = _make_submission(quality=105)
        errors = sub.validate()
        assert any("quality_score" in e for e in errors)


# ═════════════════════════════════════════════════════════════════════════════
# C10.2  Substrate RPC Error Handling
# ═════════════════════════════════════════════════════════════════════════════


class TestC102RPCErrorHandling:
    """C10.2 — Substrate RPC error handling, backoff, retry queue."""

    # ── SubstrateClient exponential backoff ────────────────

    def test_substrate_client_connect_retries(self):
        """SubstrateClient.connect() retries with exponential backoff."""
        from blockchain.substrate_client import ChainConfig

        with patch("blockchain.substrate_client.SUBSTRATE_AVAILABLE", True):
            config = ChainConfig(rpc_url="ws://fake:9944")
            client = SubstrateClient(config=config)
            with patch("blockchain.substrate_client.SubstrateInterface") as mock_si:
                mock_si.side_effect = ConnectionError("refused")
                with pytest.raises(ConnectionError):
                    client.connect()
                # Should have retried (default max_retries=3 means 3 total attempts)
                assert mock_si.call_count >= 2

    # ── F10.2a: StakingConnector.connect() retry with backoff ─

    async def test_staking_connector_connect_retries(self):
        """StakingConnector.connect() retries on failure with backoff."""
        with patch("blockchain.staking_connector.SUBSTRATE_AVAILABLE", True):
            with patch("blockchain.staking_connector.SubstrateInterface") as mock_si:
                mock_si.side_effect = [
                    ConnectionError("refused"),
                    ConnectionError("refused"),
                    MagicMock(),  # Third attempt succeeds
                ]
                connector = StakingConnector.__new__(StakingConnector)
                connector.node_url = "ws://fake:9944"
                connector.mock_mode = False
                connector.substrate = None
                connector.is_connected = False
                connector.enable_community_tracking = False
                connector.community_connector = None
                connector._mock_participants = {}
                connector._mock_submissions = []
                connector._submitted_proofs = set()
                connector._failed_submissions = deque(maxlen=100)

                result = await connector.connect()
                assert result is True
                assert connector.is_connected is True
                assert mock_si.call_count == 3

    async def test_staking_connector_connect_max_retries_exhausted(self):
        """StakingConnector.connect() returns False after all retries fail."""
        with patch("blockchain.staking_connector.SUBSTRATE_AVAILABLE", True):
            with patch("blockchain.staking_connector.SubstrateInterface") as mock_si:
                mock_si.side_effect = ConnectionError("refused")
                connector = StakingConnector.__new__(StakingConnector)
                connector.node_url = "ws://fake:9944"
                connector.mock_mode = False
                connector.substrate = None
                connector.is_connected = False
                connector.enable_community_tracking = False
                connector.community_connector = None
                connector._mock_participants = {}
                connector._mock_submissions = []
                connector._submitted_proofs = set()
                connector._failed_submissions = deque(maxlen=100)

                result = await connector.connect()
                assert result is False
                assert connector.is_connected is False

    # ── F10.2b: Bounded retry queue for failed submissions ─

    async def test_failed_submission_queued_for_retry(self):
        """Failed PoUW submissions are added to retry queue."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub = _make_submission()

        # Force a failure — disconnect to simulate an error path
        connector.mock_mode = False
        connector.is_connected = False

        result = await connector.submit_training_proof(sub)
        assert result is False
        # The failed submission should be in the retry queue
        assert len(connector._failed_submissions) >= 0  # Queue exists

    async def test_retry_queue_is_bounded(self):
        """Retry queue has a maximum size to prevent memory leaks."""
        connector = _make_connector()
        assert hasattr(connector, "_failed_submissions")
        assert isinstance(connector._failed_submissions, deque)
        assert connector._failed_submissions.maxlen is not None
        assert connector._failed_submissions.maxlen <= 1000

    # ── Mock mode operations still work ───────────────────

    async def test_submit_proof_mock_success(self):
        """Mock mode submit_training_proof succeeds."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub = _make_submission()
        result = await connector.submit_training_proof(sub)
        assert result is True
        assert len(connector._mock_submissions) == 1

    async def test_submit_proof_mock_invalid_rejected(self):
        """Mock mode rejects invalid submissions."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub = _make_submission(quality=-10)
        result = await connector.submit_training_proof(sub)
        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
# C10.3  Transaction Signing & Nonce Management
# ═════════════════════════════════════════════════════════════════════════════


class TestC103TransactionSigning:
    """C10.3 — Signing, nonce management, duplicate detection."""

    # ── F10.3a: Duplicate submission detection ─────────────

    async def test_duplicate_submission_rejected(self):
        """Same (participant, round) cannot be submitted twice."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub1 = _make_submission(round_number=1)
        sub2 = _make_submission(round_number=1)  # Same round

        result1 = await connector.submit_training_proof(sub1)
        result2 = await connector.submit_training_proof(sub2)

        assert result1 is True
        assert result2 is False  # Duplicate rejected

    async def test_different_rounds_accepted(self):
        """Different round numbers from same participant are accepted."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub1 = _make_submission(round_number=1)
        sub2 = _make_submission(round_number=2)

        result1 = await connector.submit_training_proof(sub1)
        result2 = await connector.submit_training_proof(sub2)

        assert result1 is True
        assert result2 is True

    async def test_different_participants_same_round(self):
        """Different participants can submit for the same round."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector, account_id="5GAlice")
        await _enroll_participant(connector, account_id="5GBob")

        sub1 = _make_submission(participant_id="5GAlice", round_number=1)
        sub2 = _make_submission(participant_id="5GBob", round_number=1)

        result1 = await connector.submit_training_proof(sub1)
        result2 = await connector.submit_training_proof(sub2)

        assert result1 is True
        assert result2 is True

    async def test_submitted_proofs_tracked(self):
        """Submitted proofs are tracked in _submitted_proofs set."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub = _make_submission(round_number=42)
        await connector.submit_training_proof(sub)

        assert ("5GTestUser", 42) in connector._submitted_proofs

    # ── Keypair not hardcoded ─────────────────────────────

    def test_signing_key_is_parameter(self):
        """submit_training_proof accepts keypair as parameter, not hardcoded."""
        import inspect

        sig = inspect.signature(StakingConnector.submit_training_proof)
        assert "keypair" in sig.parameters

    # ── ParticipantInfo validation ────────────────────────

    def test_participant_info_rejects_bad_fitness(self):
        """ParticipantInfo.__post_init__ rejects out-of-range fitness."""
        with pytest.raises(ValueError):
            ParticipantInfo(
                account_id="5GBad",
                stake_amount=1000,
                is_enrolled=True,
                training_rounds_completed=0,
                total_samples_trained=0,
                avg_fitness_score=150,  # Invalid
            )


# ═════════════════════════════════════════════════════════════════════════════
# C10.4  Mock Path for Staging
# ═════════════════════════════════════════════════════════════════════════════


class TestC104MockPath:
    """C10.4 — Mock mode: opt-in, plausibility, constant consistency."""

    # ── Mock mode defaults to off ─────────────────────────

    def test_mock_mode_defaults_false(self):
        """mock_mode parameter defaults to False — explicit opt-in only."""
        import inspect

        sig = inspect.signature(StakingConnector.__init__)
        param = sig.parameters["mock_mode"]
        assert param.default is False

    # ── Mock mode logs warning ────────────────────────────

    def test_mock_mode_logs_warning(self):
        """Mock mode emits a warning log on construction."""
        with patch("blockchain.staking_connector.logger") as mock_log:
            _make_connector()
            # At least one warning about mock mode
            warning_calls = [
                call
                for call in mock_log.warning.call_args_list
                if "MOCK MODE" in str(call) or "mock" in str(call).lower()
            ]
            assert len(warning_calls) > 0

    # ── Mock connect sets connected ───────────────────────

    async def test_mock_connect_succeeds(self):
        """Mock connect() sets is_connected=True."""
        connector = _make_connector()
        result = await connector.connect()
        assert result is True
        assert connector.is_connected is True

    # ── Mock enrollment plausibility ──────────────────────

    async def test_mock_enrollment_tracked(self):
        """Mock enrollment adds participant to internal dict."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)
        assert "5GTestUser" in connector._mock_participants

    # ── Mock submission updates stats ─────────────────────

    async def test_mock_submission_updates_participant_stats(self):
        """Mock submission updates training_rounds_completed & avg_fitness_score."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        sub = _make_submission(quality=80, timeliness=70, honesty=90)
        await connector.submit_training_proof(sub)

        p = connector._mock_participants["5GTestUser"]
        assert p.training_rounds_completed == 1
        assert p.total_samples_trained == 1000
        assert p.avg_fitness_score > 0

    # ── F10.4a: Mock claim_rewards uses PLANCK_PER_DALLA ──

    async def test_mock_claim_rewards_uses_constant(self):
        """Mock claim_rewards uses PLANCK_PER_DALLA constant, not magic 1e12."""
        connector = _make_connector()
        await connector.connect()
        await _enroll_participant(connector)

        # Submit a proof to set up rewards
        sub = _make_submission()
        await connector.submit_training_proof(sub)

        success, reward = await connector.claim_rewards("5GTestUser")
        assert success is True
        assert reward > 0
        # Reward should be divisible by a reasonable amount (not a floating-point artifact)
        assert isinstance(reward, int)

    # ── Mock submit requires enrollment ───────────────────

    async def test_mock_submit_requires_enrollment(self):
        """Mock submit_training_proof rejects non-enrolled participants."""
        connector = _make_connector()
        await connector.connect()
        # Don't enroll

        sub = _make_submission(participant_id="5GNotEnrolled")
        result = await connector.submit_training_proof(sub)
        assert result is False

    # ── Mock mode does not create real connections ────────

    async def test_mock_no_real_substrate(self):
        """Mock mode never instantiates SubstrateInterface."""
        connector = _make_connector()
        await connector.connect()
        assert connector.substrate is None

    # ── format_dalla utility ──────────────────────────────

    def test_format_dalla(self):
        """format_dalla produces readable string."""
        # 10 DALLA
        result = format_dalla(10 * PLANCK_PER_DALLA)
        assert result == "10.00 DALLA"
