"""
tests/test_genome_trainer_domain_models.py

Professional test suite covering:
  - client/genome_trainer.py  -- TrainingConfig, GenomeTrainer (fitness scoring,
                                  optimizer/scheduler creation, private integrity
                                  checks, save/load checkpoint)
  - client/domain_models.py   -- ModelDomain, DomainDataConfig,
                                  ModelArchitecturePreferences, DomainModelFactory,
                                  all four domain model classes (forward, preprocess,
                                  calculate_improvement, update_training_stats)
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float32_bytes(*values: float) -> bytes:
    """Pack floats into little-endian float32 bytes (accepted by sensor methods)."""
    return struct.pack(f"<{len(values)}f", *values)


def _make_trainer(with_current_model: bool = True):
    """Return a ready-to-use GenomeTrainer with a simple linear model attached."""
    from client.genome_trainer import GenomeTrainer, TrainingConfig

    cfg = TrainingConfig(
        participant_id="test_p1",
        validator_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        staking_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        device="cpu",
        mixed_precision=False,
    )
    trainer = GenomeTrainer(cfg)
    # Attach a tiny Linear model so optimizer / checkpoint helpers work
    _linear = nn.Linear(8, 4)
    trainer.model = _linear
    if with_current_model:
        # Use a mock that supports get_memory_footprint() (not on plain nn.Linear)
        mock_model = MagicMock()
        mock_model.get_memory_footprint = MagicMock(return_value=5 * 1024 * 1024)
        mock_model.parameters = _linear.parameters
        mock_model.state_dict = _linear.state_dict
        mock_model.load_state_dict = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        trainer.current_model = mock_model
    return trainer


def _weight_dict(scale: float = 0.01) -> dict:
    """Create a small weight dict similar to model.state_dict()."""
    return {
        "weight_layer": torch.randn(4, 8) * scale,
        "bias_layer": torch.randn(4) * scale,
    }


# ===========================================================================
# Section 1: TrainingConfig
# ===========================================================================


class TestTrainingConfig:
    def test_required_fields(self):
        from client.genome_trainer import TrainingConfig

        cfg = TrainingConfig(
            participant_id="pid",
            validator_address="5ABC",
            staking_account="5DEF",
        )
        assert cfg.participant_id == "pid"
        assert cfg.learning_rate == 1e-4
        assert cfg.local_epochs == 3
        assert cfg.compliance_mode is True

    def test_custom_fields(self):
        from client.genome_trainer import TrainingConfig

        cfg = TrainingConfig(
            participant_id="p2",
            validator_address="5X",
            staking_account="5Y",
            optimizer="sgd",
            scheduler=None,
            privacy_epsilon=1.0,
            device="cpu",
        )
        assert cfg.optimizer == "sgd"
        assert cfg.scheduler is None
        assert cfg.privacy_epsilon == 1.0

    def test_metadata_defaults_to_empty_dict(self):
        from client.genome_trainer import TrainingConfig

        cfg = TrainingConfig(
            participant_id="p3",
            validator_address="5A",
            staking_account="5B",
        )
        assert cfg.metadata == {}


# ===========================================================================
# Section 2: GenomeTrainer - initialisation
# ===========================================================================


class TestGenomeTrainerInit:
    def test_init_cpu(self):
        from client.genome_trainer import GenomeTrainer, TrainingConfig

        cfg = TrainingConfig(
            participant_id="init_test",
            validator_address="5A",
            staking_account="5B",
            device="cpu",
        )
        trainer = GenomeTrainer(cfg)
        assert trainer.device == torch.device("cpu")
        assert trainer.current_model is None
        assert trainer.current_genome is None
        assert trainer.historical_updates == []

    def test_init_auto_device(self):
        from client.genome_trainer import GenomeTrainer, TrainingConfig

        cfg = TrainingConfig(
            participant_id="dev_test",
            validator_address="5A",
            staking_account="5B",
            device="auto",
        )
        trainer = GenomeTrainer(cfg)
        # Should be either cuda or cpu
        assert trainer.device.type in ("cuda", "cpu")

    def test_init_custom_model_builder(self):
        from client.genome_trainer import GenomeTrainer, TrainingConfig
        from genome.model_builder import ModelBuilder

        mb = ModelBuilder(vocab_size=100, max_seq_length=64)
        cfg = TrainingConfig(
            participant_id="mb_test",
            validator_address="5A",
            staking_account="5B",
            device="cpu",
        )
        trainer = GenomeTrainer(cfg, model_builder=mb)
        assert trainer.model_builder is mb


# ===========================================================================
# Section 3: GenomeTrainer - calculate_fitness (public method)
# ===========================================================================


class TestCalculateFitness:
    def setup_method(self):
        self.trainer = _make_trainer()

    def test_high_accuracy_low_loss(self):
        score = self.trainer.calculate_fitness(
            {"accuracy": 0.95, "loss": 0.1, "training_time": 1.0}
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_zero_accuracy(self):
        score = self.trainer.calculate_fitness({"accuracy": 0.0})
        assert score >= 0.0

    def test_clamp_upper_bound(self):
        score = self.trainer.calculate_fitness(
            {"accuracy": 1.0, "loss": 0.0, "training_time": 0.0}
        )
        assert score <= 1.0

    def test_custom_weights(self):
        score = self.trainer.calculate_fitness(
            {"accuracy": 1.0, "loss": 0.0},
            quality_weight=1.0,
            timeliness_weight=0.0,
            honesty_weight=0.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Section 4: Private score helpers
# ===========================================================================


class TestPrivateScoreHelpers:
    def setup_method(self):
        self.trainer = _make_trainer()

    # ---- _calculate_quality_score ----

    def test_quality_zero_loss(self):
        score = self.trainer._calculate_quality_score(0.0)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_quality_with_val_loss(self):
        score = self.trainer._calculate_quality_score(1.0, val_loss=0.5)
        score2 = self.trainer._calculate_quality_score(0.5)
        assert score == pytest.approx(score2, abs=0.1)

    def test_quality_high_loss(self):
        score = self.trainer._calculate_quality_score(10.0)
        assert score < 5.0

    def test_quality_clamped_0_to_100(self):
        for loss in (0.0, 0.5, 2.0, 5.0, 100.0):
            s = self.trainer._calculate_quality_score(loss)
            assert 0.0 <= s <= 100.0

    # ---- _calculate_timeliness_score ----

    def test_timeliness_very_fast(self):
        # training_time << deadline → 100
        score = self.trainer._calculate_timeliness_score(1.0)
        assert score == pytest.approx(100.0)

    def test_timeliness_at_deadline(self):
        deadline = self.trainer.config.submission_deadline
        score = self.trainer._calculate_timeliness_score(deadline)
        assert 60.0 < score <= 100.0

    def test_timeliness_late(self):
        deadline = self.trainer.config.submission_deadline
        score = self.trainer._calculate_timeliness_score(deadline * 1.5)
        assert 30.0 <= score < 70.0

    def test_timeliness_very_late(self):
        deadline = self.trainer.config.submission_deadline
        score = self.trainer._calculate_timeliness_score(deadline * 3.0)
        assert score <= 30.0

    def test_timeliness_extremely_late(self):
        deadline = self.trainer.config.submission_deadline
        score = self.trainer._calculate_timeliness_score(deadline * 100.0)
        assert score == pytest.approx(0.0)

    # ---- _calculate_honesty_score (no current model → 100.0) ----

    def test_honesty_no_model(self):
        trainer = _make_trainer()
        trainer.current_model = None
        score = trainer._calculate_honesty_score()
        assert score == pytest.approx(100.0)

    def test_honesty_with_empty_weights_model(self):
        # current_model has parameters → should run all checks
        score = self.trainer._calculate_honesty_score()
        assert 0.0 <= score <= 100.0

    # ---- _get_max_norm_for_layer ----

    def test_max_norm_embedding(self):
        norm = self.trainer._get_max_norm_for_layer("token_embedding.weight")
        assert norm == pytest.approx(50.0)

    def test_max_norm_output_head(self):
        norm = self.trainer._get_max_norm_for_layer("lm_head.weight")
        assert norm == pytest.approx(100.0)

    def test_max_norm_layer_norm(self):
        norm = self.trainer._get_max_norm_for_layer("transformer.layernorm.weight")
        assert norm == pytest.approx(10.0)

    def test_max_norm_default_hidden(self):
        norm = self.trainer._get_max_norm_for_layer("transformer.linear.weight")
        assert norm == pytest.approx(30.0)

    def test_max_norm_classifier_head(self):
        norm = self.trainer._get_max_norm_for_layer("classifier.weight")
        assert norm == pytest.approx(100.0)


# ===========================================================================
# Section 5: Gradient / weight integrity checks
# ===========================================================================


class TestWeightIntegrityChecks:
    def setup_method(self):
        self.trainer = _make_trainer()

    # ---- _verify_gradient_norms (no initial weights → baseline stored) ----

    def test_gradient_norms_no_initial(self):
        weights = _weight_dict()
        score = self.trainer._verify_gradient_norms(weights)
        assert score == pytest.approx(100.0)
        # initial_weights should now be set
        assert len(self.trainer.initial_weights) > 0

    def test_gradient_norms_small_change(self):
        # Set baseline first
        weights = _weight_dict(0.01)
        self.trainer.initial_weights = {k: v.clone() for k, v in weights.items()}
        # Tiny update — should score high
        updated = {k: v + torch.randn_like(v) * 1e-3 for k, v in weights.items()}
        score = self.trainer._verify_gradient_norms(updated)
        assert score >= 80.0

    def test_gradient_norms_huge_change(self):
        weights = _weight_dict(0.01)
        self.trainer.initial_weights = {k: v.clone() for k, v in weights.items()}
        # Huge update — should score low
        updated = {k: v + torch.randn_like(v) * 1000.0 for k, v in weights.items()}
        score = self.trainer._verify_gradient_norms(updated)
        # All weights exceed max_norm → score = 0
        assert score == pytest.approx(0.0)

    def test_gradient_norms_unknown_layer(self):
        # key not in initial_weights → skipped
        weights = _weight_dict()
        self.trainer.initial_weights = {}  # empty baseline
        self.trainer._verify_gradient_norms(weights)  # sets baseline
        # Now call with a new key that wasn't in initial weights
        extra = dict(weights)
        extra["brand_new_layer"] = torch.randn(3, 3)
        score = self.trainer._verify_gradient_norms(extra)
        # brand_new_layer skipped; original layers have 0 norm change → high score
        assert score >= 80.0

    # ---- _check_zero_update ----

    def test_zero_update_no_initial(self):
        weights = _weight_dict()
        score = self.trainer._check_zero_update(weights)
        assert score == pytest.approx(100.0)

    def test_zero_update_no_change(self):
        weights = _weight_dict(0.1)
        self.trainer.initial_weights = {k: v.clone() for k, v in weights.items()}
        score = self.trainer._check_zero_update(weights)  # Same weights → zero diff
        assert score == pytest.approx(0.0)

    def test_zero_update_small_change(self):
        weights = _weight_dict(0.0)
        self.trainer.initial_weights = {k: v.clone() for k, v in weights.items()}
        updated = {k: v + 1e-5 * torch.ones_like(v) for k, v in weights.items()}
        score = self.trainer._check_zero_update(updated)
        assert score in (25.0, 50.0)

    def test_zero_update_normal_change(self):
        weights = _weight_dict(0.0)
        self.trainer.initial_weights = {k: v.clone() for k, v in weights.items()}
        updated = {k: v + 1e-3 * torch.ones_like(v) for k, v in weights.items()}
        score = self.trainer._check_zero_update(updated)
        assert score == pytest.approx(100.0)

    # ---- _check_update_variance ----

    def test_variance_no_initial(self):
        score = self.trainer._check_update_variance(_weight_dict())
        assert score == pytest.approx(100.0)

    def test_variance_single_layer(self):
        weights = {"only": torch.randn(5, 5)}
        self.trainer.initial_weights = {"only": torch.zeros(5, 5)}
        score = self.trainer._check_update_variance(weights)
        assert score == pytest.approx(100.0)

    def test_variance_consistent(self):
        self.trainer.initial_weights = {}
        self.trainer._verify_gradient_norms(_weight_dict(0.01))  # set baseline
        # Small, consistent changes
        weights = {
            k: v + torch.randn_like(v) * 1e-3
            for k, v in self.trainer.initial_weights.items()
        }
        score = self.trainer._check_update_variance(weights)
        assert 0.0 <= score <= 100.0

    def test_variance_chaotic(self):
        w = _weight_dict(0.01)
        self.trainer.initial_weights = {k: v.clone() for k, v in w.items()}
        # One layer gets huge random noise, another stays the same
        updated = {
            "weight_layer": self.trainer.initial_weights["weight_layer"]
            + torch.randn(4, 8) * 100.0,
            "bias_layer": self.trainer.initial_weights["bias_layer"].clone(),
        }
        score = self.trainer._check_update_variance(updated)
        assert score < 100.0  # Mixed variances → less than perfect

    # ---- _detect_byzantine_behavior ----

    def test_byzantine_detection_no_history(self):
        weights = _weight_dict()
        score = self.trainer._detect_byzantine_behavior(weights)
        assert 0.0 <= score <= 100.0

    def test_byzantine_detection_stores_statistics(self):
        weights = _weight_dict()
        self.trainer._detect_byzantine_behavior(weights)
        # Should store update statistics
        assert len(self.trainer.update_statistics) >= 1
        assert len(self.trainer.historical_updates) >= 1

    # ---- _store_update_statistics ----

    def test_store_update_statistics(self):
        weights = _weight_dict()
        self.trainer._store_update_statistics(weights)
        assert len(self.trainer.update_statistics) == 1
        assert "mean" in self.trainer.update_statistics[0]
        assert "std" in self.trainer.update_statistics[0]

    def test_store_update_statistics_empty(self):
        # Empty dict should return without error
        self.trainer._store_update_statistics({})
        assert len(self.trainer.update_statistics) == 0

    def test_store_update_statistics_capped_at_50(self):
        for _ in range(55):
            self.trainer._store_update_statistics(_weight_dict())
        assert len(self.trainer.update_statistics) == 50

    # ---- _verify_weight_magnitudes ----

    def test_weight_magnitudes_no_initial(self):
        score = self.trainer._verify_weight_magnitudes(_weight_dict())
        assert score == pytest.approx(100.0)

    def test_weight_magnitudes_small_change(self):
        w = _weight_dict(0.1)
        self.trainer.initial_weights = {k: v.clone() for k, v in w.items()}
        updated = {k: v + torch.randn_like(v) * 1e-3 for k, v in w.items()}
        score = self.trainer._verify_weight_magnitudes(updated)
        assert score == pytest.approx(100.0)

    def test_weight_magnitudes_huge_change(self):
        w = _weight_dict(0.1)
        self.trainer.initial_weights = {k: v.clone() for k, v in w.items()}
        updated = {k: v + torch.randn_like(v) * 1e6 for k, v in w.items()}
        # All layers exceed 2x expected → score = 0
        score = self.trainer._verify_weight_magnitudes(updated)
        assert score == pytest.approx(0.0)


# ===========================================================================
# Section 6: validate_privacy_compliance
# ===========================================================================


class TestValidatePrivacyCompliance:
    def setup_method(self):
        # current_model=None so validate_privacy_compliance skips the get_memory_footprint check
        self.trainer = _make_trainer(with_current_model=False)

    def test_compliance_disabled(self):
        self.trainer.config.compliance_mode = False
        result = self.trainer.validate_privacy_compliance(
            {"g": torch.randn(4, 4) * 500}
        )
        assert result is True

    def test_small_gradient_compliant(self):
        grads = {"layer.weight": torch.randn(4, 4) * 0.01}
        result = self.trainer.validate_privacy_compliance(grads)
        assert result is True

    def test_large_gradient_not_compliant(self):
        grads = {"layer.weight": torch.randn(4, 4) * 200.0}
        result = self.trainer.validate_privacy_compliance(grads)
        assert result is False

    def test_none_gradient_skipped(self):
        # None gradient should be skipped without error
        grads = {"layer.weight": None, "layer.bias": torch.randn(4) * 0.01}
        result = self.trainer.validate_privacy_compliance(grads)
        assert isinstance(result, bool)


# ===========================================================================
# Section 7: Optimizer and Scheduler creation
# ===========================================================================


class TestOptimizerSchedulerCreation:
    def setup_method(self):
        self.trainer = _make_trainer()

    def test_create_adamw_optimizer(self):
        self.trainer.config.optimizer = "adamw"
        opt = self.trainer._create_optimizer()
        assert isinstance(opt, torch.optim.AdamW)

    def test_create_adam_optimizer(self):
        self.trainer.config.optimizer = "adam"
        opt = self.trainer._create_optimizer()
        assert isinstance(opt, torch.optim.Adam)

    def test_create_sgd_optimizer(self):
        self.trainer.config.optimizer = "sgd"
        opt = self.trainer._create_optimizer()
        assert isinstance(opt, torch.optim.SGD)

    def test_create_unknown_optimizer_falls_back_to_adamw(self):
        self.trainer.config.optimizer = "some_unknown_optimizer"
        opt = self.trainer._create_optimizer()
        assert isinstance(opt, torch.optim.AdamW)

    def test_create_cosine_scheduler(self):
        self.trainer.config.scheduler = "cosine"
        opt = self.trainer._create_optimizer()
        sched = self.trainer._create_scheduler(opt, num_training_steps=10)
        assert sched is not None

    def test_create_linear_scheduler(self):
        self.trainer.config.scheduler = "linear"
        opt = self.trainer._create_optimizer()
        sched = self.trainer._create_scheduler(opt, num_training_steps=10)
        assert sched is not None

    def test_no_scheduler(self):
        self.trainer.config.scheduler = None
        opt = self.trainer._create_optimizer()
        sched = self.trainer._create_scheduler(opt, num_training_steps=10)
        assert sched is None

    def test_unknown_scheduler_returns_none(self):
        self.trainer.config.scheduler = "some_unknown_scheduler"
        opt = self.trainer._create_optimizer()
        sched = self.trainer._create_scheduler(opt, num_training_steps=10)
        assert sched is None


# ===========================================================================
# Section 8: save_checkpoint and load_checkpoint
# ===========================================================================


class TestCheckpointSaveLoad:
    def setup_method(self):
        self.trainer = _make_trainer()

    def test_save_checkpoint(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        self.trainer.save_checkpoint(path, epoch=5, metrics={"loss": 0.42})
        assert path.exists()

    def test_load_checkpoint_roundtrip(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        self.trainer.save_checkpoint(path, epoch=7, metrics={"loss": 0.11})
        epoch, metrics = self.trainer.load_checkpoint(path)
        assert epoch == 7
        assert metrics["loss"] == pytest.approx(0.11)

    def test_save_requires_model(self):
        del self.trainer.model
        with pytest.raises((ValueError, AttributeError)):
            self.trainer.save_checkpoint("/tmp/no_model.pt", epoch=0, metrics={})

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.trainer.load_checkpoint("/tmp/definitely_missing_ckpt_xyz.pt")

    def test_load_requires_model(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        self.trainer.save_checkpoint(path, epoch=1, metrics={})
        del self.trainer.model
        with pytest.raises((ValueError, AttributeError)):
            self.trainer.load_checkpoint(path)


# ===========================================================================
# Section 9: set_genome with initial_weights validation
# ===========================================================================


class TestSetGenomeInitialWeights:
    def test_set_genome_rejects_nan_weights(self):
        from client.genome_trainer import GenomeTrainer, TrainingConfig
        from genome.dna import ArchitectureLayer, Genome, LayerType

        trainer = GenomeTrainer(
            TrainingConfig(
                participant_id="p",
                validator_address="5A",
                staking_account="5B",
                device="cpu",
            )
        )
        genome = Genome(
            genome_id="g1",
            generation=0,
            parent_genomes=[],
            encoder_layers=[
                ArchitectureLayer(
                    layer_type=LayerType.MULTIHEAD_ATTENTION,
                    hidden_size=64,
                    num_heads=2,
                )
            ],
            decoder_layers=[],
        )
        model = trainer.model_builder.build_model(genome)
        bad_weights = {
            k: torch.full_like(v, float("nan")) for k, v in model.state_dict().items()
        }
        with pytest.raises(ValueError, match="NaN"):
            trainer.set_genome(genome, initial_weights=bad_weights)


# ===========================================================================
# Section 10: ModelDomain enum
# ===========================================================================


class TestModelDomain:
    def test_all_domains_present(self):
        from client.domain_models import ModelDomain

        assert hasattr(ModelDomain, "AGRITECH")
        assert hasattr(ModelDomain, "MARINE")
        assert hasattr(ModelDomain, "EDUCATION")
        assert hasattr(ModelDomain, "TECH")
        assert hasattr(ModelDomain, "GENERAL")

    def test_reward_multiplier_agritech(self):
        from client.domain_models import ModelDomain

        mult = ModelDomain.AGRITECH.reward_multiplier()
        assert mult == pytest.approx(1.5)

    def test_reward_multiplier_general(self):
        from client.domain_models import ModelDomain

        assert ModelDomain.GENERAL.reward_multiplier() == pytest.approx(1.0)

    def test_to_index_returns_int(self):
        from client.domain_models import ModelDomain

        for domain in ModelDomain:
            idx = domain.to_index()
            assert isinstance(idx, int)

    def test_to_index_unique(self):
        from client.domain_models import ModelDomain

        indices = [d.to_index() for d in ModelDomain]
        assert len(indices) == len(set(indices))


# ===========================================================================
# Section 11: DomainDataConfig and ModelArchitecturePreferences
# ===========================================================================


class TestDomainDataConfig:
    def test_default_config(self):
        from client.domain_models import DomainDataConfig

        cfg = DomainDataConfig()
        assert cfg.image_size == (224, 224)
        assert cfg.sensor_window_size == 100
        assert cfg.tokenizer_name == "character"

    def test_custom_config(self):
        from client.domain_models import DomainDataConfig

        cfg = DomainDataConfig(sensor_window_size=50, sensor_normalization="zscore")
        assert cfg.sensor_window_size == 50
        assert cfg.sensor_normalization == "zscore"


class TestModelArchitecturePreferences:
    def test_defaults(self):
        from client.domain_models import ModelArchitecturePreferences

        prefs = ModelArchitecturePreferences()
        assert prefs.min_layers == 3
        assert prefs.gradient_clip == pytest.approx(1.0)

    def test_custom(self):
        from client.domain_models import ModelArchitecturePreferences

        prefs = ModelArchitecturePreferences(min_layers=5, max_layers=30)
        assert prefs.min_layers == 5
        assert prefs.max_layers == 30


# ===========================================================================
# Section 12: DomainModelFactory
# ===========================================================================


class TestDomainModelFactory:
    def test_create_agritech(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        m = DomainModelFactory.create_model(ModelDomain.AGRITECH, device="cpu")
        assert m.domain == ModelDomain.AGRITECH

    def test_create_marine(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        m = DomainModelFactory.create_model(ModelDomain.MARINE, device="cpu")
        assert m.domain == ModelDomain.MARINE

    def test_create_education(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        with patch("client.domain_models.BelizeChainLLM") as mock_llm:
            mock_llm.return_value = MagicMock(spec=nn.Module)
            mock_llm.return_value.to = MagicMock(return_value=MagicMock(spec=nn.Module))
            m = DomainModelFactory.create_model(ModelDomain.EDUCATION, device="cpu")
        assert m.domain == ModelDomain.EDUCATION

    def test_create_tech(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        m = DomainModelFactory.create_model(ModelDomain.TECH, device="cpu")
        assert m.domain == ModelDomain.TECH

    def test_create_general_falls_back_to_agritech(self):
        from client.domain_models import AgriTechModel, DomainModelFactory, ModelDomain

        m = DomainModelFactory.create_model(ModelDomain.GENERAL, device="cpu")
        assert isinstance(m, AgriTechModel)

    def test_unknown_domain_raises(self):
        from client.domain_models import DomainModelFactory

        with pytest.raises((ValueError, KeyError)):
            DomainModelFactory.create_model("nonexistent_domain")

    def test_list_available_domains(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        domains = DomainModelFactory.list_available_domains()
        assert ModelDomain.AGRITECH in domains
        assert len(domains) >= 5

    def test_get_domain_info_agritech(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        info = DomainModelFactory.get_domain_info(ModelDomain.AGRITECH)
        assert "name" in info
        assert info["reward_multiplier"] == pytest.approx(1.5)

    def test_get_domain_info_all_domains(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        for domain in ModelDomain:
            info = DomainModelFactory.get_domain_info(domain)
            assert isinstance(info, dict)


# ===========================================================================
# Section 13: AgriTechModel
# ===========================================================================


class TestAgriTechModel:
    def setup_method(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        self.model = DomainModelFactory.create_model(ModelDomain.AGRITECH, device="cpu")

    def test_get_architecture_preferences(self):
        prefs = self.model.get_architecture_preferences()
        assert hasattr(prefs, "min_layers")

    def test_update_training_stats(self):
        self.model.update_training_stats(accuracy=0.85)
        # Should not raise

    def test_update_training_stats_multiple(self):
        for acc in [0.6, 0.7, 0.8, 0.9]:
            self.model.update_training_stats(accuracy=acc)

    def test_preprocess_sensor_data(self):
        sensor_values = np.array([0.1, 0.5, 0.3, 0.7, 0.2] * 20, dtype=np.float32)
        raw = {"feed_type": "sensor_reading", "data": sensor_values.tobytes()}
        result = self.model.preprocess_data(raw)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2  # (1, window_size)

    def test_preprocess_weather_data(self):
        weather_values = np.array([25.0, 80.0, 1013.0, 15.0] * 10, dtype=np.float32)
        raw = {"feed_type": "weather_data", "data": weather_values.tobytes()}
        result = self.model.preprocess_data(raw)
        assert isinstance(result, torch.Tensor)

    def test_preprocess_unknown_feed_type_raises(self):
        with pytest.raises(ValueError, match="Unknown feed type"):
            self.model.preprocess_data({"feed_type": "totally_unknown", "data": b""})

    def test_forward_sensor_data(self):
        # Default CNN takes (B, 3, H, W) - outputs 2D, triggers sensor/else branch
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = self.model.forward(x)
        assert "predictions" in out
        assert "confidence" in out

    def test_calculate_improvement_basic(self):
        old = {
            "predictions": torch.tensor([0.6, 0.7]),
            "confidence": torch.tensor([0.6, 0.7]),
        }
        new = {
            "predictions": torch.tensor([0.8, 0.9]),
            "confidence": torch.tensor([0.8, 0.9]),
        }
        improvement = self.model.calculate_improvement(old, new)
        assert isinstance(improvement, float)


# ===========================================================================
# Section 14: MarineModel
# ===========================================================================


class TestMarineModel:
    def setup_method(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        self.model = DomainModelFactory.create_model(ModelDomain.MARINE, device="cpu")

    def test_preprocess_water_quality(self):
        wq = np.array([35.0, 7.5, 8.1, 27.0], dtype=np.float32)
        raw = {"feed_type": "sensor_reading", "data": wq.tobytes()}
        result = self.model.preprocess_data(raw)
        assert isinstance(result, torch.Tensor)

    def test_forward_sensor_path(self):
        # Default Marine CNN takes (B, 3, H, W) - outputs 2D → else/water_quality branch
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = self.model.forward(x)
        assert "predictions" in out

    def test_calculate_improvement(self):
        old = {"predictions": torch.tensor([0.5]), "confidence": torch.tensor([0.5])}
        new = {"predictions": torch.tensor([0.8]), "confidence": torch.tensor([0.8])}
        improvement = self.model.calculate_improvement(old, new)
        assert isinstance(improvement, float)

    def test_get_architecture_preferences(self):
        prefs = self.model.get_architecture_preferences()
        assert hasattr(prefs, "preferred_layers")

    def test_update_training_stats(self):
        self.model.update_training_stats(accuracy=0.75)


# ===========================================================================
# Section 15: EducationModel
# ===========================================================================


class TestEducationModel:
    def setup_method(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        # EducationModel loads BelizeChainLLM (which calls HuggingFace) — mock it
        with patch("client.domain_models.BelizeChainLLM") as mock_llm_class:
            mock_llm_class.return_value = MagicMock(spec=nn.Module)
            mock_llm_class.return_value.to = MagicMock(
                return_value=MagicMock(spec=nn.Module)
            )
            self.model = DomainModelFactory.create_model(
                ModelDomain.EDUCATION, device="cpu"
            )
        # Attach a simple model for forward pass
        self.model.model = nn.Linear(4, 5)  # 5 features for education output

    def test_preprocess_student_data(self):
        import json

        student_json = json.dumps(
            {
                "time_spent_minutes": 45,
                "completion_rate": 85,
                "average_quiz_score": 78,
                "interaction_count": 12,
            }
        ).encode("utf-8")
        raw = {"feed_type": "phone_collection", "data": student_json}
        result = self.model.preprocess_data(raw)
        assert isinstance(result, torch.Tensor)

    def test_forward_sensor_data(self):
        import json

        student_json = json.dumps(
            {
                "time_spent_minutes": 30,
                "completion_rate": 70,
                "average_quiz_score": 75,
                "interaction_count": 8,
            }
        ).encode()
        raw = {"feed_type": "phone_collection", "data": student_json}
        x = self.model.preprocess_data(raw)  # (1, 4)
        with torch.no_grad():
            out = self.model.forward(x)
        assert "predictions" in out

    def test_calculate_improvement(self):
        old = {
            "predictions": torch.tensor([0.5, 0.6]),
            "confidence": torch.tensor([0.5, 0.6]),
        }
        new = {
            "predictions": torch.tensor([0.7, 0.8]),
            "confidence": torch.tensor([0.7, 0.8]),
        }
        improvement = self.model.calculate_improvement(old, new)
        assert isinstance(improvement, float)

    def test_get_architecture_preferences(self):
        prefs = self.model.get_architecture_preferences()
        assert prefs is not None


# ===========================================================================
# Section 16: TechModel
# ===========================================================================


class TestTechModel:
    def setup_method(self):
        from client.domain_models import DomainModelFactory, ModelDomain

        self.model = DomainModelFactory.create_model(ModelDomain.TECH, device="cpu")

    def test_preprocess_metrics(self):
        metrics = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 10, dtype=np.float32)
        raw = {"feed_type": "sensor_reading", "data": metrics.tobytes()}
        result = self.model.preprocess_data(raw)
        assert isinstance(result, torch.Tensor)

    def test_forward(self):
        metrics = np.array([0.1] * 50, dtype=np.float32)
        raw = {"feed_type": "sensor_reading", "data": metrics.tobytes()}
        x = self.model.preprocess_data(raw)
        with torch.no_grad():
            out = self.model.forward(x)
        assert "predictions" in out

    def test_calculate_improvement(self):
        old = {"predictions": torch.tensor([0.4]), "confidence": torch.tensor([0.4])}
        new = {"predictions": torch.tensor([0.6]), "confidence": torch.tensor([0.6])}
        improvement = self.model.calculate_improvement(old, new)
        assert isinstance(improvement, float)

    def test_get_architecture_preferences(self):
        prefs = self.model.get_architecture_preferences()
        assert prefs is not None


# ===========================================================================
# Section 17: Base DomainModel helpers
# ===========================================================================


class TestDomainModelBase:
    def test_genome_layer_to_pytorch_with_real_layer(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import ArchitectureLayer, LayerType

        model = AgriTechModel(device="cpu")
        layer = ArchitectureLayer(
            layer_type=LayerType.LINEAR,
            hidden_size=64,
            num_heads=None,
        )
        result = model._genome_layer_to_pytorch(layer)
        # Returns nn.Module or None
        assert result is None or isinstance(result, nn.Module)

    def test_genome_layer_to_pytorch_transformer(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import ArchitectureLayer, LayerType

        model = AgriTechModel(device="cpu")
        layer = ArchitectureLayer(
            layer_type=LayerType.TRANSFORMER_ENCODER,
            hidden_size=64,
            num_heads=8,
        )
        result = model._genome_layer_to_pytorch(layer)
        assert result is None or isinstance(result, nn.Module)


# ===========================================================================
# Section 18: calculate_quality_score standalone function
# ===========================================================================


class TestCalculateQualityScoreFunction:
    def test_basic_call(self):
        from client.domain_models import calculate_quality_score

        score = calculate_quality_score(
            accuracy=80, timeliness=70, completeness=85, consistency=75, provenance=90
        )
        assert isinstance(score, (int, float))
        assert score > 0

    def test_perfect_scores(self):
        from client.domain_models import calculate_quality_score

        score = calculate_quality_score(
            accuracy=100,
            timeliness=100,
            completeness=100,
            consistency=100,
            provenance=100,
        )
        assert score == 1000  # max = (100*30+100*20+100*15+100*20+100*15)//10 = 1000

    def test_zero_scores(self):
        from client.domain_models import calculate_quality_score

        score = calculate_quality_score(
            accuracy=0, timeliness=0, completeness=0, consistency=0, provenance=0
        )
        assert score == 0
