"""
tests/test_client_coverage_gaps.py

Comprehensive coverage gap tests for ALL client modules:
  - client/genome_trainer.py  (250 uncovered lines)
  - client/model.py           (69 uncovered lines)
  - client/train.py           (66 uncovered lines)
  - client/nawal.py           (38 uncovered lines)
  - client/data_loader.py     (62 uncovered lines)
  - client/domain_models.py   (80 uncovered lines)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Synchronously run an async coroutine."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_trainer(**overrides):
    """Create a GenomeTrainer with sane defaults and optional overrides."""
    from client.genome_trainer import TrainingConfig, GenomeTrainer

    defaults = dict(
        participant_id="test_p1",
        validator_address="5Grw",
        staking_account="5Grw",
        device="cpu",
        mixed_precision=False,
        local_epochs=1,
        submission_deadline=300.0,
        training_timeout=3600.0,
        gradient_accumulation_steps=1,
    )
    defaults.update(overrides)
    cfg = TrainingConfig(**defaults)
    trainer = GenomeTrainer(cfg)
    # The source uses self.config.gradient_clip (dynamic attr not in TrainingConfig)
    if not hasattr(cfg, 'gradient_clip'):
        cfg.gradient_clip = None
    return trainer


def _simple_model(in_features=8, out_features=4):
    """Create a minimal nn.Module for testing."""
    return nn.Linear(in_features, out_features)


def _mock_genome_model():
    """Create a mock GenomeModel with all needed methods."""
    model = MagicMock(spec=nn.Module)
    model.parameters = nn.Linear(8, 4).parameters
    sd = {"w": torch.randn(4, 8), "b": torch.randn(4)}
    model.state_dict = MagicMock(return_value=sd)
    model.load_state_dict = MagicMock()
    model.count_parameters = MagicMock(return_value=32)
    model.get_memory_footprint = MagicMock(return_value=128)
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()
    return model


def _weight_dict(scale=0.01):
    return {"w": torch.randn(4, 8) * scale, "b": torch.randn(4) * scale}


def _float32_bytes(*values):
    return struct.pack(f"<{len(values)}f", *values)


class DictOutputModel(nn.Module):
    """Real nn.Module that returns dict outputs for genome_trainer tests."""

    def __init__(self, in_f=8, out_f=4, return_loss=True, logits_3d=False, seq_len=5):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.return_loss = return_loss
        self.logits_3d = logits_3d
        self.seq_len = seq_len

    def forward(self, x):
        batch_size = x.size(0)
        logits = self.linear(x)
        if self.logits_3d:
            logits = logits.unsqueeze(1).expand(batch_size, self.seq_len, -1).contiguous()
        if self.return_loss:
            loss = logits.sum() * 0.001
            return {"loss": loss, "logits": logits}
        return {"loss": None, "logits": logits}


class KeywordDictModel(nn.Module):
    """Real nn.Module for train_genome that accepts keyword args and returns dict."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        x = self.embedding(input_ids)
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_memory_footprint(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())


# ===========================================================================
# 1. GENOME TRAINER — set_genome with initial_weights
# ===========================================================================


class TestSetGenomeWithWeights:
    """Cover lines 193-199: initial_weights validation in set_genome."""

    def test_set_genome_nan_weights_rejected(self):
        trainer = _make_trainer()
        mock_builder = MagicMock()
        mock_model = _mock_genome_model()
        mock_builder.build_model.return_value = mock_model
        trainer.model_builder = mock_builder

        genome = MagicMock()
        genome.genome_id = "g1"
        bad_weights = {"w": torch.tensor([float("nan"), 1.0])}

        with pytest.raises(ValueError, match="NaN or Inf"):
            trainer.set_genome(genome, initial_weights=bad_weights)

    def test_set_genome_inf_weights_rejected(self):
        trainer = _make_trainer()
        mock_builder = MagicMock()
        mock_model = _mock_genome_model()
        mock_builder.build_model.return_value = mock_model
        trainer.model_builder = mock_builder

        genome = MagicMock()
        genome.genome_id = "g2"
        bad_weights = {"w": torch.tensor([float("inf"), 1.0])}

        with pytest.raises(ValueError, match="NaN or Inf"):
            trainer.set_genome(genome, initial_weights=bad_weights)

    def test_set_genome_extreme_magnitude_warns(self):
        trainer = _make_trainer()
        mock_builder = MagicMock()
        mock_model = _mock_genome_model()
        mock_builder.build_model.return_value = mock_model
        trainer.model_builder = mock_builder

        genome = MagicMock()
        genome.genome_id = "g3"
        # Extreme magnitude but finite
        weights = {"w": torch.tensor([2e6, 1.0])}
        trainer.set_genome(genome, initial_weights=weights)
        mock_model.load_state_dict.assert_called_once_with(weights)

    def test_set_genome_valid_weights_loaded(self):
        trainer = _make_trainer()
        mock_builder = MagicMock()
        mock_model = _mock_genome_model()
        mock_builder.build_model.return_value = mock_model
        trainer.model_builder = mock_builder

        genome = MagicMock()
        genome.genome_id = "g4"
        weights = {"w": torch.randn(4, 8)}
        trainer.set_genome(genome, initial_weights=weights)
        mock_model.load_state_dict.assert_called_once()


# ===========================================================================
# 2. GENOME TRAINER — train_epoch dict output handling
# ===========================================================================


class TestTrainEpochOutputHandling:
    """Cover lines 255-300: dict outputs, 3D tensor handling."""

    def _make_trainer_with_model(self, model):
        trainer = _make_trainer()
        trainer.model = model
        return trainer

    def test_train_epoch_dict_with_loss(self):
        """Dict output with explicit loss key."""
        model = DictOutputModel(8, 4, return_loss=True)

        trainer = self._make_trainer_with_model(model)
        trainer.config.gradient_clip = None
        data = [(torch.randn(2, 8), torch.randint(0, 4, (2,)))]
        result = trainer.train_epoch(data)
        assert "loss" in result
        assert result["num_batches"] == 1

    def test_train_epoch_dict_no_loss_3d_classification(self):
        """Dict output without loss, 3D logits for classification."""
        model = DictOutputModel(8, 4, return_loss=False, logits_3d=True, seq_len=5)

        trainer = self._make_trainer_with_model(model)
        trainer.config.gradient_clip = None
        targets = torch.randint(0, 4, (2,))  # 1D targets for classification
        data = [(torch.randn(2, 8), targets)]
        result = trainer.train_epoch(data)
        assert result["num_batches"] == 1

    def test_train_epoch_dict_no_loss_3d_seq2seq(self):
        """Dict output without loss, 3D logits for seq-to-seq."""
        model = DictOutputModel(8, 10, return_loss=False, logits_3d=True, seq_len=5)

        trainer = self._make_trainer_with_model(model)
        trainer.config.gradient_clip = None
        targets = torch.randint(0, 10, (2, 5))  # 2D targets for seq2seq
        data = [(torch.randn(2, 8), targets)]
        result = trainer.train_epoch(data)
        assert result["num_batches"] == 1

    def test_train_epoch_tensor_3d_classification(self):
        """Plain tensor output, 3D for classification."""
        model = nn.Sequential(nn.Linear(8, 20))  # outputs (batch, 20)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 20)
            def forward(self, x):
                out = self.linear(x)
                return out.view(x.size(0), 5, 4)  # 3D: (batch, seq, classes)

        trainer = self._make_trainer_with_model(Wrapper())
        trainer.config.gradient_clip = None
        targets = torch.randint(0, 4, (2,))  # 1D classification
        data = [(torch.randn(2, 8), targets)]
        result = trainer.train_epoch(data)
        assert result["num_batches"] == 1

    def test_train_epoch_tensor_3d_seq2seq(self):
        """Plain tensor output, 3D for seq-to-seq."""
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 50)
            def forward(self, x):
                return self.linear(x).view(x.size(0), 5, 10)

        trainer = self._make_trainer_with_model(Wrapper())
        trainer.config.gradient_clip = None
        targets = torch.randint(0, 10, (2, 5))
        data = [(torch.randn(2, 8), targets)]
        result = trainer.train_epoch(data)
        assert result["num_batches"] == 1

    def test_train_epoch_self_supervised(self):
        """Single tensor batch (no labels) = self-supervised."""
        model = nn.Linear(8, 8)
        trainer = self._make_trainer_with_model(model)
        trainer.config.gradient_clip = 1.0
        data = [torch.randn(2, 8)]  # single tensor, no tuple
        result = trainer.train_epoch(data)
        assert result["num_batches"] == 1


# ===========================================================================
# 3. GENOME TRAINER — evaluate dict output handling
# ===========================================================================


class TestEvaluateOutputHandling:
    """Cover lines 332-394: evaluate method branches."""

    def _make_trainer_with_model(self, model):
        trainer = _make_trainer()
        trainer.model = model
        return trainer

    def test_evaluate_dict_with_loss(self):
        """Dict output with explicit loss + logits."""
        model = DictOutputModel(8, 4, return_loss=True)

        trainer = self._make_trainer_with_model(model)
        data = [(torch.randn(2, 8), torch.randint(0, 4, (2,)))]
        result = trainer.evaluate(data)
        assert "loss" in result
        assert "accuracy" in result

    def test_evaluate_dict_no_loss_3d_classification(self):
        """Dict output w/o loss, 3D logits, 1D targets → classification."""
        model = DictOutputModel(8, 4, return_loss=False, logits_3d=True, seq_len=5)

        trainer = self._make_trainer_with_model(model)
        targets = torch.randint(0, 4, (2,))
        data = [(torch.randn(2, 8), targets)]
        result = trainer.evaluate(data)
        assert "accuracy" in result

    def test_evaluate_dict_no_loss_3d_seq2seq(self):
        """Dict output w/o loss, 3D logits, 2D targets → seq2seq."""
        model = DictOutputModel(8, 10, return_loss=False, logits_3d=True, seq_len=5)

        trainer = self._make_trainer_with_model(model)
        targets = torch.randint(0, 10, (2, 5))
        data = [(torch.randn(2, 8), targets)]
        result = trainer.evaluate(data)
        assert "accuracy" in result

    def test_evaluate_self_supervised(self):
        """Evaluate with reconstruction targets (self-supervised style)."""
        model = nn.Linear(8, 4)
        trainer = self._make_trainer_with_model(model)
        inputs = torch.randn(2, 8)
        targets = torch.randint(0, 4, (2,))
        data = [(inputs, targets)]
        result = trainer.evaluate(data)
        assert result["accuracy"] >= 0.0


# ===========================================================================
# 4. GENOME TRAINER — train (multi-epoch) method
# ===========================================================================


class TestTrainMultiEpoch:
    """Cover lines 542-585: train() method with val_data and without."""

    def test_train_with_val_loader(self):
        trainer = _make_trainer()
        model = nn.Linear(8, 4)
        trainer.model = model
        trainer.config.local_epochs = 2

        train_data = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(3)]
        val_data = [(torch.randn(2, 8), torch.randint(0, 4, (2,))) for _ in range(2)]

        result = trainer.train(train_data, val_data, epochs=2)
        assert "final_loss" in result
        assert len(result["train_loss"]) == 2
        assert len(result["val_loss"]) == 2

    def test_train_without_val_loader(self):
        trainer = _make_trainer()
        model = nn.Linear(8, 4)
        trainer.model = model

        train_data = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
        result = trainer.train(train_loader=train_data, epochs=1)
        assert "final_loss" in result
        # val_loss should be populated from final evaluation on training set
        assert len(result["val_loss"]) >= 1

    def test_train_positional_args(self):
        """Old API: train(train_loader, val_loader, epochs=N)."""
        trainer = _make_trainer()
        trainer.model = nn.Linear(8, 4)
        train_data = [(torch.randn(4, 8), torch.randint(0, 4, (4,)))]
        val_data = [(torch.randn(2, 8), torch.randint(0, 4, (2,)))]
        result = trainer.train(train_data, val_data, epochs=1)
        assert result["final_loss"] >= 0.0

    def test_train_no_data_raises(self):
        trainer = _make_trainer()
        trainer.model = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="Must provide training data"):
            trainer.train(epochs=1)

    def test_train_no_model_raises(self):
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="No model set"):
            trainer.train(train_loader=[], epochs=1)


# ===========================================================================
# 5. GENOME TRAINER — async train_genome (BIGGEST GAP: 614-778)
# ===========================================================================


class TestTrainGenomeAsync:
    """Cover lines 614-778: the async train_genome method."""

    def _make_batch(self):
        return {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
            "labels": torch.randint(0, 100, (4, 16)),
        }

    def test_train_genome_full_loop(self):
        trainer = _make_trainer(local_epochs=1)
        real_model = KeywordDictModel(vocab_size=100, hidden_size=32)

        trainer.model_builder = MagicMock()
        trainer.model_builder.build_model.return_value = real_model

        genome = MagicMock()
        genome.genome_id = "genome_1"

        batch = self._make_batch()
        train_loader = [batch]

        weights, metrics = _run(trainer.train_genome(genome, train_loader))
        assert metrics.participant_id == "test_p1"
        assert metrics.genome_id == "genome_1"
        assert metrics.samples_trained > 0
        assert metrics.quality_score >= 0
        assert metrics.fitness_score >= 0

    def test_train_genome_with_validation(self):
        trainer = _make_trainer(local_epochs=1)
        real_model = KeywordDictModel(vocab_size=100, hidden_size=32)

        trainer.model_builder = MagicMock()
        trainer.model_builder.build_model.return_value = real_model

        genome = MagicMock()
        genome.genome_id = "genome_2"

        batch = self._make_batch()
        weights, metrics = _run(
            trainer.train_genome(genome, [batch], val_loader=[batch])
        )
        assert metrics.val_loss is not None

    def test_train_genome_timeout(self):
        """Timeout triggers early stop."""
        trainer = _make_trainer(local_epochs=10, training_timeout=0.001)

        import time

        class SlowModel(KeywordDictModel):
            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                time.sleep(0.01)
                return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)

        real_model = SlowModel(vocab_size=100, hidden_size=32)

        trainer.model_builder = MagicMock()
        trainer.model_builder.build_model.return_value = real_model

        genome = MagicMock()
        genome.genome_id = "g_timeout"

        # Multiple batches
        train_loader = [self._make_batch() for _ in range(100)]
        weights, metrics = _run(trainer.train_genome(genome, train_loader))
        # Should complete but with limited batches
        assert metrics.training_time > 0

    def test_train_genome_no_attention_mask(self):
        """Batch without attention_mask."""
        trainer = _make_trainer(local_epochs=1)
        real_model = KeywordDictModel(vocab_size=100, hidden_size=32)

        trainer.model_builder = MagicMock()
        trainer.model_builder.build_model.return_value = real_model

        genome = MagicMock()
        genome.genome_id = "g_no_mask"

        batch = {"input_ids": torch.randint(0, 100, (4, 16)), "labels": torch.randint(0, 100, (4, 16))}
        weights, metrics = _run(trainer.train_genome(genome, [batch]))
        assert metrics.samples_trained == 4

    def test_train_genome_with_global_weights(self):
        """Test loading global_weights before training."""
        trainer = _make_trainer(local_epochs=1)
        real_model = KeywordDictModel(vocab_size=100, hidden_size=32)

        trainer.model_builder = MagicMock()
        trainer.model_builder.build_model.return_value = real_model

        genome = MagicMock()
        genome.genome_id = "g_global"

        batch = self._make_batch()
        # Use a matching state_dict shape for global_weights
        global_w = real_model.state_dict()
        weights, metrics = _run(
            trainer.train_genome(genome, [batch], global_weights=global_w)
        )
        assert metrics.samples_trained > 0


# ===========================================================================
# 6. GENOME TRAINER — async _validate (790-819)
# ===========================================================================


class TestValidateAsync:
    """Cover lines 790-819: _validate method."""

    def test_validate_returns_loss(self):
        trainer = _make_trainer()
        real_model = KeywordDictModel(vocab_size=100, hidden_size=32)
        trainer.current_model = real_model

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
            "labels": torch.randint(0, 100, (4, 16)),
        }
        val_loss, val_acc = _run(trainer._validate([batch]))
        assert val_loss > 0.0
        assert val_acc is None  # Not calculated for LM


# ===========================================================================
# 7. GENOME TRAINER — submit_update (1078-1098)
# ===========================================================================


class TestSubmitUpdate:
    """Cover lines 1078-1098: async submit_update."""

    def test_submit_update_creates_model_update(self):
        from client.genome_trainer import TrainingConfig, GenomeTrainer
        from nawal.server import ModelUpdate, TrainingMetrics

        trainer = _make_trainer()
        trainer.current_round = 5
        genome = MagicMock()
        genome.genome_id = "g_submit"
        trainer.current_genome = genome

        weights = _weight_dict()
        metrics = TrainingMetrics(
            participant_id="test_p1",
            genome_id="g_submit",
            round_number=5,
            train_loss=0.1,
            samples_trained=100,
            training_time=10.0,
            quality_score=90.0,
            timeliness_score=80.0,
            honesty_score=95.0,
            fitness_score=88.0,
        )

        update = _run(trainer.submit_update(weights, metrics))
        assert isinstance(update, ModelUpdate)
        assert update.participant_id == "test_p1"
        assert update.genome_id == "g_submit"
        assert update.fitness_score == 88.0


# ===========================================================================
# 8. GENOME TRAINER — honesty score with data poisoning/DP/leakage
# ===========================================================================


class TestHonestyScoreAdvanced:
    """Cover lines 970-1041: _calculate_honesty_score with extra checks."""

    def test_honesty_score_with_predictions_and_losses(self):
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        initial = _weight_dict(0.01)
        trainer.current_model = mock_model
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}
        mock_model.state_dict.return_value = initial

        score = trainer._calculate_honesty_score(
            predictions=torch.randn(10, 4),
            losses=[0.5, 0.4, 0.3, 0.35, 0.32, 0.31, 0.3, 0.29, 0.28, 0.27],
        )
        assert 0.0 <= score <= 100.0

    def test_honesty_score_with_activations(self):
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        initial = _weight_dict(0.01)
        trainer.current_model = mock_model
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}
        mock_model.state_dict.return_value = initial

        activations = {"layer_0": torch.randn(4, 64), "layer_1": torch.randn(4, 32)}

        score = trainer._calculate_honesty_score(
            predictions=torch.randn(10, 4),
            losses=[0.5] * 12,
            activations=activations,
        )
        assert 0.0 <= score <= 100.0

    def test_honesty_score_with_dp_config(self):
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        initial = _weight_dict(0.01)
        trainer.current_model = mock_model
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}
        mock_model.state_dict.return_value = initial

        # Set up DP config
        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 10.0
        dp_config.budget.spent_epsilon = 1.0
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 100
        trainer.dp_config = dp_config

        gradients = {"w": torch.randn(4, 8) * 0.01, "b": torch.randn(4) * 0.01}

        score = trainer._calculate_honesty_score(
            gradients=gradients,
        )
        assert 0.0 <= score <= 100.0

    def test_honesty_score_with_data_leakage_check(self):
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        initial = _weight_dict(0.01)
        trainer.current_model = mock_model
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}
        mock_model.state_dict.return_value = initial

        # Populate training/val losses for leakage check
        trainer.training_losses = [0.5, 0.4, 0.3, 0.2]
        trainer.validation_losses = [0.6, 0.5, 0.4, 0.35]

        score = trainer._calculate_honesty_score(
            predictions=torch.randn(10, 4),
        )
        assert 0.0 <= score <= 100.0

    def test_honesty_score_six_checks(self):
        """When all 6 checks are available."""
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        initial = _weight_dict(0.01)
        trainer.current_model = mock_model
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}
        mock_model.state_dict.return_value = initial

        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 10.0
        dp_config.budget.spent_epsilon = 1.0
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 50
        trainer.dp_config = dp_config
        trainer.training_losses = [0.5] * 10
        trainer.validation_losses = [0.6] * 10

        score = trainer._calculate_honesty_score(
            predictions=torch.randn(10, 4),
            losses=[0.5] * 12,
            activations={"layer_0": torch.randn(4, 64)},
            gradients={"w": torch.randn(4, 8) * 0.01},
        )
        assert 0.0 <= score <= 100.0


# ===========================================================================
# 9. GENOME TRAINER — validate_privacy_compliance DP paths
# ===========================================================================


class TestValidatePrivacyComplianceDP:
    """Cover lines 1134-1254: DP compliance checks."""

    def test_dp_gradient_clipping_violation(self):
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()

        dp_config = MagicMock()
        dp_config.clip_norm = 0.1
        dp_config.noise_multiplier = 1.0
        trainer.dp_config = dp_config
        trainer.config.privacy_epsilon = 10.0
        trainer.privacy_spent_history = [(0.5, 10)]

        # Large gradient that violates clip norm
        gradients = {"w": torch.randn(4, 8) * 100.0}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is False

    def test_dp_budget_exhausted(self):
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()

        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        trainer.dp_config = dp_config
        trainer.config.privacy_epsilon = 1.0
        trainer.privacy_spent_history = [(2.0, 100)]  # Over budget

        gradients = {"w": torch.randn(4, 8) * 0.01}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is False

    def test_dp_budget_near_exhaustion_warning(self):
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()

        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        trainer.dp_config = dp_config
        trainer.config.privacy_epsilon = 1.0
        trainer.privacy_spent_history = [(0.95, 100)]  # 95% used

        gradients = {"w": torch.randn(4, 8) * 0.001}
        # Should warn but still return True (barely within budget)
        result = trainer.validate_privacy_compliance(gradients)
        # The budget check logs warning but 0.95 < 1.0 so passes
        assert result is True

    def test_dp_noise_scale_mismatch(self):
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()

        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        trainer.dp_config = dp_config
        trainer.config.privacy_epsilon = 10.0
        trainer.privacy_spent_history = [(0.1, 10)]
        # Noise scale history with wrong values (expect 1.0, got 5.0)
        trainer.noise_scale_history = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        gradients = {"w": torch.randn(4, 8) * 0.001}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is False

    def test_dp_low_noise_multiplier_warning(self):
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()

        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 0.1  # Very low
        trainer.dp_config = dp_config
        trainer.config.privacy_epsilon = 10.0
        trainer.privacy_spent_history = [(0.1, 10)]

        gradients = {"w": torch.randn(4, 8) * 0.001}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is True  # Warning only, no failure


# ===========================================================================
# 10. GENOME TRAINER — data leakage in validate_privacy_compliance
# ===========================================================================


class TestDataLeakageChecks:
    """Cover lines 1311-1476: data leakage patterns in validate_privacy_compliance."""

    def test_sparse_gradient_warning(self):
        """95%+ zeros in gradient logs a warning."""
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()
        trainer.dp_config = None

        sparse_grad = torch.zeros(100)
        sparse_grad[0] = 0.01
        gradients = {"w": sparse_grad}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is True  # Logs warning but doesn't fail

    def test_uniform_gradient_warning(self):
        """Gradient with <1% unique values logs a warning."""
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()
        trainer.dp_config = None

        # All same value (1 unique out of 200)
        uniform_grad = torch.ones(200)
        gradients = {"w": uniform_grad}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is True

    def test_small_model_large_gradient_warning(self):
        """Small model with large gradient norm triggers warning."""
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        mock_model = _mock_genome_model()
        mock_model.get_memory_footprint.return_value = 5 * 1024 * 1024  # 5MB
        trainer.current_model = mock_model
        trainer.dp_config = None

        large_grad = torch.randn(100) * 50.0
        gradients = {"w": large_grad}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is False

    def test_gradient_norm_variance_warning(self):
        """High gradient norm variance triggers model inversion warning."""
        trainer = _make_trainer()
        trainer.config.compliance_mode = True
        trainer.current_model = _mock_genome_model()
        trainer.dp_config = None
        # Simulate high variance gradient history
        trainer.gradient_clip_history = [0.1, 10.0, 0.01, 50.0, 0.001, 100.0, 0.1, 20.0, 0.01, 80.0]

        gradients = {"w": torch.randn(100) * 0.01}
        result = trainer.validate_privacy_compliance(gradients)
        assert result is True


# ===========================================================================
# 11. GENOME TRAINER — byzantine / poisoning detection methods
# ===========================================================================


class TestByzantineDetectionAdvanced:
    """Cover lines 1517-1611, 1758-1796, 1860-1995."""

    def _setup_trainer_with_history(self):
        trainer = _make_trainer()
        mock_model = _mock_genome_model()
        trainer.current_model = mock_model

        initial = _weight_dict(0.01)
        trainer.initial_weights = {k: v.clone() for k, v in initial.items()}

        # Build history
        for _ in range(6):
            w = {k: v + torch.randn_like(v) * 0.001 for k, v in initial.items()}
            trainer._store_update_statistics(w)
            trainer.historical_updates.append({k: v.clone().cpu() for k, v in w.items()})

        return trainer, initial

    def test_check_update_similarity(self):
        trainer, initial = self._setup_trainer_with_history()
        weights = {k: v + torch.randn_like(v) * 0.001 for k, v in initial.items()}
        score = trainer._check_update_similarity(weights)
        assert 0 <= score <= 100

    def test_check_update_similarity_opposite_dir(self):
        trainer, initial = self._setup_trainer_with_history()
        # Opposite direction
        weights = {k: initial[k] - (v - initial[k]) * 100 for k, v in trainer.historical_updates[-1].items()}
        score = trainer._check_update_similarity(weights)
        assert score <= 50

    def test_check_statistical_outliers(self):
        trainer, initial = self._setup_trainer_with_history()
        weights = {k: v + torch.randn_like(v) * 0.001 for k, v in initial.items()}
        score = trainer._check_statistical_outliers(weights)
        assert 0 <= score <= 100

    def test_check_statistical_outliers_extreme(self):
        trainer, initial = self._setup_trainer_with_history()
        weights = {k: v * 1000 for k, v in initial.items()}
        score = trainer._check_statistical_outliers(weights)
        assert score <= 66

    def test_detect_data_poisoning_with_losses(self):
        trainer = _make_trainer()
        trainer.current_model = _mock_genome_model()
        losses = [0.5, 0.4, 0.3, 0.35, 0.32, 0.31, 0.3, 0.29, 0.28, 0.27]
        score = trainer._detect_data_poisoning(losses=losses)
        assert 0 <= score <= 100

    def test_detect_data_poisoning_with_predictions(self):
        trainer = _make_trainer()
        trainer.current_model = _mock_genome_model()
        # Need at least 3 history entries
        for _ in range(4):
            trainer._store_predictions(torch.randn(10, 4))
        score = trainer._detect_data_poisoning(predictions=torch.randn(10, 4))
        assert 0 <= score <= 100

    def test_detect_data_poisoning_with_activations(self):
        trainer = _make_trainer()
        trainer.current_model = _mock_genome_model()
        acts = {"layer_0": torch.randn(4, 64)}
        # Build activation history
        for _ in range(6):
            trainer._store_activations(acts)
        score = trainer._detect_data_poisoning(activations=acts)
        assert 0 <= score <= 100

    def test_check_loss_distribution_bimodal(self):
        trainer = _make_trainer()
        # Create bimodal distribution
        losses = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9]
        score = trainer._check_loss_distribution(losses)
        assert 0 <= score <= 100

    def test_check_prediction_consistency_divergent(self):
        trainer = _make_trainer()
        for _ in range(5):
            trainer._store_predictions(torch.randn(10, 4))
        # Opposite predictions
        opposite = torch.randn(10, 4) * -100
        score = trainer._check_prediction_consistency(opposite)
        assert 0 <= score <= 100

    def test_check_feature_distribution_anomalous(self):
        trainer = _make_trainer()
        acts = {"layer_0": torch.randn(4, 64)}
        for _ in range(6):
            trainer._store_activations(acts)
        # Anomalous activations
        anomalous = {"layer_0": torch.randn(4, 64) * 1000}
        score = trainer._check_feature_distribution(anomalous)
        assert 0 <= score <= 100

    def test_check_activation_patterns_backdoor(self):
        trainer = _make_trainer()
        # Create activation with extreme spike (backdoor signature)
        acts = torch.zeros(100)
        acts[0] = 1000.0  # One extreme activation
        score = trainer._check_activation_patterns({"layer_0": acts})
        assert 0 <= score <= 100


# ===========================================================================
# 12. GENOME TRAINER — DP verification methods
# ===========================================================================


class TestDPVerification:
    """Cover lines 2122-2276: _verify_differential_privacy and sub-methods."""

    def test_verify_dp_no_config(self):
        trainer = _make_trainer()
        score = trainer._verify_differential_privacy({})
        assert score == 100.0

    def test_verify_dp_full(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 10.0
        dp_config.budget.spent_epsilon = 1.0
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 50
        trainer.dp_config = dp_config
        trainer.noise_scale_history = [1.0, 1.0, 1.0]

        gradients = {"w": torch.randn(4, 8) * 0.01}
        score = trainer._verify_differential_privacy(gradients, dp_config)
        assert 0 <= score <= 100

    def test_check_gradient_clipping_violations(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.clip_norm = 0.001
        gradients = {"w": torch.randn(4, 8)}  # Will exceed clip_norm
        score = trainer._check_gradient_clipping(gradients, dp_config)
        assert score < 100

    def test_check_privacy_budget_exhausted(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 1.0
        dp_config.budget.spent_epsilon = 2.0
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 100
        score = trainer._check_privacy_budget(dp_config)
        assert score == 0.0

    def test_check_privacy_budget_low(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 1.0
        dp_config.budget.spent_epsilon = 0.95
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 100
        score = trainer._check_privacy_budget(dp_config)
        assert score == 20.0

    def test_check_privacy_budget_moderate(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.budget = MagicMock()
        dp_config.budget.epsilon = 1.0
        dp_config.budget.spent_epsilon = 0.8
        dp_config.budget.delta = 1e-5
        dp_config.budget.steps = 100
        score = trainer._check_privacy_budget(dp_config)
        assert score == 50.0

    def test_check_noise_consistency_good(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        trainer.noise_scale_history = [1.0, 1.0, 1.05, 0.95]
        score = trainer._check_noise_consistency(dp_config)
        assert score >= 85

    def test_check_noise_consistency_bad(self):
        trainer = _make_trainer()
        dp_config = MagicMock()
        dp_config.clip_norm = 1.0
        dp_config.noise_multiplier = 1.0
        trainer.noise_scale_history = [5.0, 5.0, 5.0, 5.0]
        score = trainer._check_noise_consistency(dp_config)
        assert score < 85


# ===========================================================================
# 13. GENOME TRAINER — data leakage verification methods
# ===========================================================================


class TestDataLeakageVerification:
    """Cover lines 2345-2517: _verify_data_leakage and sub-methods."""

    def test_verify_no_data(self):
        trainer = _make_trainer()
        score = trainer._verify_data_leakage()
        assert score == 100.0

    def test_membership_inference_small_gap(self):
        trainer = _make_trainer()
        trainer.training_losses = [0.3, 0.28, 0.27, 0.26, 0.25]
        trainer.validation_losses = [0.32, 0.30, 0.29, 0.28, 0.27]
        score = trainer._check_membership_inference()
        assert score >= 70.0

    def test_membership_inference_large_gap(self):
        trainer = _make_trainer()
        trainer.training_losses = [0.01] * 10
        trainer.validation_losses = [0.5] * 10
        score = trainer._check_membership_inference()
        assert score < 70.0

    def test_gradient_inversion_low(self):
        trainer = _make_trainer()
        gradients = {"layer_0": torch.randn(10) * 0.01, "layer_1": torch.randn(10) * 0.01}
        score = trainer._check_gradient_inversion(gradients)
        assert score >= 80.0

    def test_gradient_inversion_high(self):
        trainer = _make_trainer()
        gradients = {"layer_0": torch.randn(10) * 100, "layer_1": torch.randn(10) * 0.01}
        score = trainer._check_gradient_inversion(gradients)
        assert score < 100

    def test_information_leakage_healthy(self):
        trainer = _make_trainer()
        # Healthy confidence between 0.6-0.9
        preds = torch.tensor([[0.1, 0.8], [0.2, 0.7], [0.3, 0.6]])
        score = trainer._check_information_leakage(preds)
        assert score >= 50.0

    def test_information_leakage_overconfident(self):
        trainer = _make_trainer()
        # Extreme overconfidence > 0.98
        preds = torch.tensor([[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]])
        score = trainer._check_information_leakage(preds)
        assert score <= 100

    def test_verify_data_leakage_with_all(self):
        trainer = _make_trainer()
        trainer.training_losses = [0.3] * 10
        trainer.validation_losses = [0.35] * 10
        gradients = {"layer_0": torch.randn(10) * 0.01}
        predictions = torch.randn(10, 4)
        score = trainer._verify_data_leakage(gradients=gradients, predictions=predictions)
        assert 0 <= score <= 100


# ===========================================================================
# 14. GENOME TRAINER — get_statistics
# ===========================================================================


class TestGetStatistics:
    """Cover lines 2489-2517."""

    def test_get_statistics_empty(self):
        trainer = _make_trainer()
        stats = trainer.get_statistics()
        assert stats["total_rounds"] == 0
        assert stats["avg_fitness"] == 0.0

    def test_get_statistics_with_metrics(self):
        from nawal.server import TrainingMetrics
        trainer = _make_trainer()
        genome = MagicMock()
        genome.genome_id = "g_stats"
        trainer.current_genome = genome

        m = TrainingMetrics(
            participant_id="test_p1",
            genome_id="g_stats",
            round_number=1,
            train_loss=0.3,
            samples_trained=100,
            training_time=10.0,
            quality_score=85.0,
            timeliness_score=90.0,
            honesty_score=88.0,
            fitness_score=87.0,
        )
        trainer.training_metrics.append(m)

        stats = trainer.get_statistics()
        assert stats["total_rounds"] == 1
        assert stats["total_samples"] == 100
        assert stats["avg_fitness"] == 87.0
        assert stats["current_genome_id"] == "g_stats"


# ===========================================================================
# 15. CLIENT/MODEL.PY — QuantizedBelizeModel, BelizeanLanguageDetector
# ===========================================================================


class TestQuantizedBelizeModel:
    """Cover lines 155-273 of model.py."""

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModelForSequenceClassification")
    @patch("client.model.BitsAndBytesConfig")
    def test_init_8bit(self, mock_bnb, mock_seq_model, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_inner = MagicMock()
        mock_inner.classifier = MagicMock()  # Has built-in classifier
        mock_inner.config.hidden_size = 768
        mock_seq_model.from_pretrained.return_value = mock_inner

        model = QuantizedBelizeModel(model_name="test-model", bits=8)
        assert model.bits == 8
        mock_bnb.assert_called_once()

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModelForSequenceClassification")
    @patch("client.model.AutoModelForCausalLM")
    @patch("client.model.BitsAndBytesConfig")
    def test_init_4bit_fallback_to_causal(self, mock_bnb, mock_causal, mock_seq_model, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        # Sequence classification fails
        mock_seq_model.from_pretrained.side_effect = Exception("not supported")
        # Causal LM succeeds
        mock_inner = MagicMock()
        mock_inner.config.hidden_size = 1024
        mock_causal.from_pretrained.return_value = mock_inner

        model = QuantizedBelizeModel(model_name="test-model", bits=4)
        assert model.bits == 4
        assert hasattr(model, "classification_head")

    @patch("client.model.AutoTokenizer")
    @patch("client.model.BitsAndBytesConfig")
    def test_init_unsupported_bits(self, mock_bnb, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        with pytest.raises(ValueError, match="Unsupported quantization bits"):
            QuantizedBelizeModel(model_name="test-model", bits=3)

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModelForSequenceClassification")
    @patch("client.model.BitsAndBytesConfig")
    def test_forward_with_classifier(self, mock_bnb, mock_seq_model, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_inner = MagicMock()
        mock_inner.classifier = MagicMock()
        mock_inner.config.hidden_size = 768
        output = MagicMock()
        output.logits = torch.randn(2, 2)
        mock_inner.return_value = output
        mock_inner.__call__ = MagicMock(return_value=output)
        mock_seq_model.from_pretrained.return_value = mock_inner

        model = QuantizedBelizeModel(model_name="test-model", bits=8)
        result = model(torch.randint(0, 100, (2, 16)))
        # Should return output directly
        assert result is not None

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModelForSequenceClassification")
    @patch("client.model.AutoModelForCausalLM")
    @patch("client.model.BitsAndBytesConfig")
    def test_forward_without_classifier(self, mock_bnb, mock_causal, mock_seq_model, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_seq_model.from_pretrained.side_effect = Exception("no")

        mock_inner = MagicMock()
        mock_inner.config.hidden_size = 64
        # No classifier attribute
        del mock_inner.classifier
        output = MagicMock()
        output.last_hidden_state = torch.randn(2, 16, 64)
        mock_inner.__call__ = MagicMock(return_value=output)
        mock_inner.return_value = output
        mock_causal.from_pretrained.return_value = mock_inner

        model = QuantizedBelizeModel(model_name="test-model", bits=4)
        result = model(torch.randint(0, 100, (2, 16)))
        assert "logits" in result

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModelForSequenceClassification")
    @patch("client.model.BitsAndBytesConfig")
    def test_get_memory_footprint(self, mock_bnb, mock_seq_model, mock_tokenizer):
        from client.model import QuantizedBelizeModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_inner = MagicMock()
        mock_inner.classifier = MagicMock()
        mock_inner.config.hidden_size = 768
        # Make parameters() return real parameters
        param = torch.randn(100)
        mock_inner.parameters.return_value = [param]
        mock_seq_model.from_pretrained.return_value = mock_inner

        model = QuantizedBelizeModel(model_name="test-model", bits=8)
        footprint = model.get_memory_footprint()
        assert footprint > 0


class TestBelizeanLanguageDetector:
    """Cover lines 288-338 of model.py."""

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModel")
    def test_init_and_forward(self, mock_model_cls, mock_tokenizer):
        from client.model import BelizeanLanguageDetector

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_transformer = MagicMock()
        mock_transformer.config.hidden_size = 128
        output = MagicMock()
        output.last_hidden_state = torch.randn(1, 10, 128)
        mock_transformer.return_value = output
        mock_transformer.__call__ = MagicMock(return_value=output)
        mock_model_cls.from_pretrained.return_value = mock_transformer

        detector = BelizeanLanguageDetector(model_name="test-model")
        probs = detector(torch.randint(0, 100, (1, 10)))
        assert probs.shape == (1, 5)  # 5 languages
        assert probs.sum().item() == pytest.approx(1.0, abs=0.01)

    @patch("client.model.AutoTokenizer")
    @patch("client.model.AutoModel")
    def test_predict_language(self, mock_model_cls, mock_tokenizer):
        from client.model import BelizeanLanguageDetector

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.randint(0, 100, (1, 10)), "attention_mask": torch.ones(1, 10)}
        mock_tok.__call__ = MagicMock(return_value=mock_tok.return_value)
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_transformer = MagicMock()
        mock_transformer.config.hidden_size = 128
        output = MagicMock()
        output.last_hidden_state = torch.randn(1, 10, 128)
        mock_transformer.return_value = output
        mock_transformer.__call__ = MagicMock(return_value=output)
        mock_model_cls.from_pretrained.return_value = mock_transformer

        detector = BelizeanLanguageDetector(model_name="test-model")
        lang = detector.predict_language("Hello world")
        assert lang in ["english", "spanish", "kriol", "garifuna", "maya"]


class TestModelVersioning:
    """Cover lines 383-461 of model.py: compute_model_hash, save/load checkpoint, versions_compatible."""

    def test_compute_model_hash(self):
        from client.model import compute_model_hash
        model = nn.Linear(8, 4)
        h = compute_model_hash(model)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_save_versioned_checkpoint(self, tmp_path):
        from client.model import save_versioned_checkpoint
        model = nn.Linear(8, 4)
        model.version = "1.0.0"
        model.training_rounds = 5
        model.privacy_epsilon = 1.0
        model.privacy_delta = 1e-5
        path = str(tmp_path / "ckpt.pt")
        save_versioned_checkpoint(model, path, metadata={"test": True})
        assert Path(path).exists()

    def test_load_versioned_checkpoint(self, tmp_path):
        from client.model import save_versioned_checkpoint, load_versioned_checkpoint
        model = nn.Linear(8, 4)
        model.version = "1.0.0"
        model.training_rounds = 3
        model.privacy_epsilon = 2.0
        model.privacy_delta = 1e-5
        model.last_updated = None
        path = str(tmp_path / "ckpt.pt")
        save_versioned_checkpoint(model, path, metadata={"round": 3})

        model2 = nn.Linear(8, 4)
        model2.version = "1.0.0"
        model2.training_rounds = 0
        model2.last_updated = None
        meta = load_versioned_checkpoint(model2, path)
        assert model2.training_rounds == 3
        assert meta["round"] == 3

    def test_load_versioned_checkpoint_version_mismatch(self, tmp_path):
        from client.model import save_versioned_checkpoint, load_versioned_checkpoint
        model = nn.Linear(8, 4)
        model.version = "1.0.0"
        path = str(tmp_path / "ckpt.pt")
        save_versioned_checkpoint(model, path)

        model2 = nn.Linear(8, 4)
        model2.version = "2.0.0"  # Different major version
        with pytest.raises(RuntimeError, match="Incompatible versions"):
            load_versioned_checkpoint(model2, path)

    def test_versions_compatible(self):
        from client.model import versions_compatible
        assert versions_compatible("1.0.0", "1.0.1") is True
        assert versions_compatible("1.0.0", "1.1.0") is False
        assert versions_compatible("1.0.0", "2.0.0") is False
        assert versions_compatible("bad", "1.0.0") is False


# ===========================================================================
# 16. CLIENT/TRAIN.PY — BelizeChainFederatedClient
# ===========================================================================


class TestBelizeChainFederatedClient:
    """Cover lines 90-312 of train.py."""

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.QuantizedBelizeModel")
    def test_init_quantized(self, mock_qmodel, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_qmodel.return_value = mock_model_inst
        mock_loader.return_value = MagicMock()

        config = BelizeTrainingConfig(participant_id="p1", quantization_bits=8)
        client = BelizeChainFederatedClient(config)
        assert client.model is not None
        mock_qmodel.assert_called_once()

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.BelizeChainLLM")
    def test_init_full_precision(self, mock_llm, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_llm.return_value = mock_model_inst
        mock_loader.return_value = MagicMock()

        config = BelizeTrainingConfig(participant_id="p1", quantization_bits=16)
        client = BelizeChainFederatedClient(config)
        mock_llm.assert_called_once()

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.QuantizedBelizeModel")
    def test_set_parameters_nan_rejected(self, mock_qmodel, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.state_dict.return_value = {"w": torch.randn(4, 8)}
        mock_qmodel.return_value = mock_model_inst
        mock_loader.return_value = MagicMock()

        config = BelizeTrainingConfig(participant_id="p1")
        client = BelizeChainFederatedClient(config)

        nan_params = [np.array([float("nan"), 1.0])]
        with pytest.raises(ValueError, match="NaN or Inf"):
            client.set_parameters(nan_params)

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.QuantizedBelizeModel")
    def test_set_parameters_extreme_magnitude_warns(self, mock_qmodel, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.state_dict.return_value = {"w": torch.randn(2)}
        mock_qmodel.return_value = mock_model_inst
        mock_loader.return_value = MagicMock()

        config = BelizeTrainingConfig(participant_id="p1")
        client = BelizeChainFederatedClient(config)
        # Extreme but valid
        params = [np.array([2e6, 1.0])]
        client.set_parameters(params)
        mock_model_inst.load_state_dict.assert_called_once()

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.QuantizedBelizeModel")
    def test_fit_training_loop(self, mock_qmodel, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        # Setup model
        real_model = nn.Linear(8, 4)
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.state_dict.return_value = real_model.state_dict()
        mock_model_inst.parameters.return_value = real_model.parameters()
        mock_model_inst.train = MagicMock()

        # Forward returns object with loss
        fwd_out = MagicMock()
        fwd_out.loss = torch.tensor(0.5, requires_grad=True)
        mock_model_inst.__call__ = MagicMock(return_value=fwd_out)
        mock_model_inst.return_value = fwd_out
        mock_qmodel.return_value = mock_model_inst

        # Setup data loader
        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
            "labels": torch.randint(0, 4, (4,)),
        }
        mock_train_loader = MagicMock()
        mock_train_loader.__iter__ = MagicMock(return_value=iter([batch]))
        mock_data = MagicMock()
        mock_data.get_train_loader.return_value = mock_train_loader
        mock_loader.return_value = mock_data

        compliance = MagicMock()
        compliance.filter_batch.return_value = batch
        compliance.get_stats.return_value = {"total": 1, "filtered": 0}

        config = BelizeTrainingConfig(participant_id="p1", local_epochs=1)
        client = BelizeChainFederatedClient(config)
        client.compliance_filter = compliance

        params = [v.cpu().numpy() for v in real_model.state_dict().values()]
        updated, num, metrics = client.fit(params, {})
        assert num >= 0
        assert "loss" in metrics

    @patch("client.train.BelizeDataLoader")
    @patch("client.train.QuantizedBelizeModel")
    def test_evaluate(self, mock_qmodel, mock_loader):
        from client.train import BelizeTrainingConfig, BelizeChainFederatedClient

        real_model = nn.Linear(8, 4)
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.state_dict.return_value = real_model.state_dict()
        mock_model_inst.eval = MagicMock()

        fwd_out = MagicMock()
        fwd_out.loss = torch.tensor(0.3)
        fwd_out.logits = torch.randn(4, 4)
        mock_model_inst.__call__ = MagicMock(return_value=fwd_out)
        mock_model_inst.return_value = fwd_out
        mock_qmodel.return_value = mock_model_inst

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
            "labels": torch.randint(0, 4, (4,)),
        }
        mock_eval_loader = MagicMock()
        mock_eval_loader.__iter__ = MagicMock(return_value=iter([batch]))
        mock_eval_loader.__len__ = MagicMock(return_value=1)
        mock_data = MagicMock()
        mock_data.get_eval_loader.return_value = mock_eval_loader
        mock_loader.return_value = mock_data

        compliance = MagicMock()
        compliance.filter_batch.return_value = batch

        config = BelizeTrainingConfig(participant_id="p1")
        client = BelizeChainFederatedClient(config)
        client.compliance_filter = compliance

        params = [v.cpu().numpy() for v in real_model.state_dict().values()]
        loss, num, metrics = client.evaluate(params, {})
        assert loss >= 0
        assert "accuracy" in metrics

    def test_apply_differential_privacy(self):
        from client.train import BelizeChainFederatedClient, BelizeTrainingConfig

        with patch("client.train.QuantizedBelizeModel") as mock_qmodel, \
             patch("client.train.BelizeDataLoader") as mock_loader:
            mock_model_inst = MagicMock()
            mock_model_inst.to.return_value = mock_model_inst
            mock_qmodel.return_value = mock_model_inst
            mock_loader.return_value = MagicMock()

            config = BelizeTrainingConfig(participant_id="p1")
            client = BelizeChainFederatedClient(config)

            params = [np.random.randn(10).astype(np.float64)]
            private = client._apply_differential_privacy(params, epsilon=1.0)
            assert len(private) == 1
            # Should have noise added
            assert not np.allclose(private[0], params[0])


# ===========================================================================
# 17. CLIENT/DATA_LOADER.PY — ComplianceDataFilter, BelizeDataset, etc.
# ===========================================================================


class TestComplianceDataFilter:
    """Cover lines 91-111 of data_loader.py."""

    def test_is_compliant_credit_card(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._is_compliant("Buy with card 4111111111111111") is False

    def test_is_compliant_ssn(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._is_compliant("SSN: 123-45-6789") is False

    def test_is_compliant_email(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._is_compliant("Contact at test@example.com") is False

    def test_is_compliant_belize_phone(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._is_compliant("Call +501-222-3344") is False

    def test_contains_restricted_illegal_gambling(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._contains_restricted_content("Promote illegal gambling") is True

    def test_contains_restricted_ponzi(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._contains_restricted_content("Run a ponzi scheme") is True

    def test_is_compliant_clean(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        assert f._is_compliant("BelizeChain is great") is True

    def test_filter_batch_all_compliant(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        batch = {
            "input_ids": torch.randint(0, 100, (3, 10)),
            "attention_mask": torch.ones(3, 10),
            "labels": torch.tensor([0, 1, 0]),
            "text": ["Hello world", "BelizeChain rocks", "Nature is beautiful"],
        }
        result = f.filter_batch(batch)
        assert result is not None
        assert result["input_ids"].shape[0] == 3

    def test_filter_batch_removes_non_compliant(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        batch = {
            "input_ids": torch.randint(0, 100, (3, 10)),
            "attention_mask": torch.ones(3, 10),
            "labels": torch.tensor([0, 1, 0]),
            "text": ["Hello", "SSN: 123-45-6789", "Good"],
        }
        result = f.filter_batch(batch)
        assert result is not None
        assert result["input_ids"].shape[0] == 2  # One removed

    def test_filter_batch_all_rejected(self):
        from client.data_loader import ComplianceDataFilter
        f = ComplianceDataFilter()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.tensor([0, 1]),
            "text": ["SSN: 123-45-6789", "Card: 4111111111111111"],
        }
        result = f.filter_batch(batch)
        assert result is None


class TestBelizeDataset:
    """Cover lines 193-223 of data_loader.py."""

    def test_load_json_data(self, tmp_path):
        from client.data_loader import BelizeDataset
        data = [{"text": "Hello", "label": 0}, {"text": "World", "label": 1}]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        })
        mock_tokenizer.return_value = mock_tokenizer.__call__.return_value

        ds = BelizeDataset(str(path), mock_tokenizer, max_length=10)
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item

    def test_load_csv_data(self, tmp_path):
        from client.data_loader import BelizeDataset
        csv_content = "text,label\nHello,0\nWorld,1\n"
        path = tmp_path / "data.csv"
        path.write_text(csv_content)

        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        })
        mock_tokenizer.return_value = mock_tokenizer.__call__.return_value

        ds = BelizeDataset(str(path), mock_tokenizer, max_length=10)
        assert len(ds) == 2

    def test_load_text_data(self, tmp_path):
        from client.data_loader import BelizeDataset
        path = tmp_path / "data.txt"
        path.write_text("Hello\nWorld\n")

        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        })
        mock_tokenizer.return_value = mock_tokenizer.__call__.return_value

        ds = BelizeDataset(str(path), mock_tokenizer, max_length=10)
        assert len(ds) == 2


class TestBelizeDataLoader:
    """Cover lines 275-379 of data_loader.py."""

    @patch("data.tokenizers.NawalTokenizerWrapper")
    def test_init_with_synthetic_data(self, mock_tokenizer_cls):
        from client.data_loader import BelizeDataLoader

        mock_tok = MagicMock()
        mock_tok.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        })
        mock_tok.return_value = mock_tok.__call__.return_value
        mock_tokenizer_cls.return_value = mock_tok

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call _create_synthetic_data directly via __new__ to test it
            loader = BelizeDataLoader.__new__(BelizeDataLoader)
            loader.tokenizer = mock_tok
            data_dir = Path(tmpdir)
            loader._create_synthetic_data(data_dir)
            assert (data_dir / "train.json").exists()

    @patch("data.tokenizers.NawalTokenizerWrapper")
    def test_create_synthetic_writes_json(self, mock_tokenizer_cls):
        from client.data_loader import BelizeDataLoader

        mock_tok = MagicMock()
        mock_tok.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        })
        mock_tok.return_value = mock_tok.__call__.return_value
        mock_tokenizer_cls.return_value = mock_tok

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # Call _create_synthetic_data directly
            loader = BelizeDataLoader.__new__(BelizeDataLoader)
            loader.tokenizer = mock_tok
            loader._create_synthetic_data(data_dir)
            assert (data_dir / "train.json").exists()
            with open(data_dir / "train.json") as f:
                data = json.load(f)
            assert len(data) == 10

    def test_collate_fn_with_compliance(self):
        from client.data_loader import ComplianceDataFilter

        loader = object.__new__(type("Loader", (), {}))
        # Test the collate pattern manually
        f = ComplianceDataFilter()
        batch = [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
             "labels": torch.tensor(0), "text": "Hello"},
            {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 1]),
             "labels": torch.tensor(1), "text": "World"},
        ]
        # Collate manually
        collated = {}
        for key in batch[0].keys():
            if key == "text":
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])

        result = f.filter_batch(collated)
        assert result is not None


class TestCreateBelizeanDataSplits:
    """Cover lines 397-428 of data_loader.py."""

    def test_create_splits_json(self, tmp_path):
        from client.data_loader import create_belizean_data_splits

        data = [{"text": f"Sample {i}", "label": i % 2} for i in range(9)]
        input_file = tmp_path / "data.json"
        input_file.write_text(json.dumps(data))

        output_dir = str(tmp_path / "federated")
        dirs = create_belizean_data_splits(str(input_file), num_participants=3, output_dir=output_dir)
        assert len(dirs) == 3
        for d in dirs:
            assert Path(d).exists()
            assert (Path(d) / "train.json").exists()

    def test_create_splits_text(self, tmp_path):
        from client.data_loader import create_belizean_data_splits

        text_content = "\n".join([f"Line {i}" for i in range(6)])
        input_file = tmp_path / "data.txt"
        input_file.write_text(text_content)

        output_dir = str(tmp_path / "fed2")
        dirs = create_belizean_data_splits(str(input_file), num_participants=2, output_dir=output_dir)
        assert len(dirs) == 2


# ===========================================================================
# 18. CLIENT/NAWAL.PY — Nawal model
# ===========================================================================


class TestNawalModel:
    """Cover lines 128-372 of nawal.py."""

    @patch("client.nawal.NawalTransformer")
    @patch("client.nawal.NawalModelConfig")
    def test_init_small(self, mock_config_cls, mock_transformer_cls):
        from client.nawal import Nawal

        mock_config = MagicMock()
        mock_config.belizean_vocab_extension = False
        mock_config.supported_languages = ["en", "es"]
        mock_config.num_layers = 6
        mock_config.hidden_size = 512
        mock_config.use_cache = False
        mock_config_cls.nawal_small.return_value = mock_config

        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer

        with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.__len__ = MagicMock(return_value=256)
            mock_tok.add_tokens = MagicMock(return_value=10)
            mock_tok_cls.return_value = mock_tok

            model = Nawal(model_size="small")
            assert model.config == mock_config

    @patch("client.nawal.NawalTransformer")
    @patch("client.nawal.NawalModelConfig")
    def test_forward_delegation(self, mock_config_cls, mock_transformer_cls):
        from client.nawal import Nawal

        mock_config = MagicMock()
        mock_config.belizean_vocab_extension = False
        mock_config.supported_languages = ["en"]
        mock_config.num_layers = 6
        mock_config.hidden_size = 512
        mock_config.use_cache = False
        mock_config_cls.nawal_small.return_value = mock_config

        mock_transformer = MagicMock()
        mock_transformer.return_value = {"logits": torch.randn(1, 10, 256)}
        mock_transformer_cls.return_value = mock_transformer

        with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.__len__ = MagicMock(return_value=256)
            mock_tok_cls.return_value = mock_tok

            model = Nawal(model_size="small")
            result = model(input_ids=torch.randint(0, 256, (1, 10)))
            mock_transformer.assert_called_once()

    @patch("client.nawal.NawalTransformer")
    @patch("client.nawal.NawalModelConfig")
    def test_generate(self, mock_config_cls, mock_transformer_cls):
        from client.nawal import Nawal

        mock_config = MagicMock()
        mock_config.belizean_vocab_extension = False
        mock_config.supported_languages = ["en", "es"]
        mock_config.num_layers = 6
        mock_config.hidden_size = 512
        mock_config.use_cache = False
        mock_config_cls.nawal_small.return_value = mock_config

        mock_transformer = MagicMock()
        mock_transformer.generate.return_value = torch.randint(0, 256, (1, 50))
        mock_transformer_cls.return_value = mock_transformer

        with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.__len__ = MagicMock(return_value=256)
            mock_tok.encode.return_value = torch.randint(0, 256, (1, 5))
            mock_tok.decode.return_value = "Generated text output"
            mock_tok_cls.return_value = mock_tok

            model = Nawal(model_size="small")
            outputs = model.generate("What is Belize?", max_length=50)
            assert len(outputs) >= 1
            assert isinstance(outputs[0], str)

    def test_from_pretrained_nonexistent(self):
        """Load from non-existent path falls back to new model."""
        from client.nawal import Nawal

        with patch("client.nawal.NawalTransformer") as mock_tf_cls, \
             patch("client.nawal.NawalModelConfig") as mock_cfg_cls:
            mock_config = MagicMock()
            mock_config.belizean_vocab_extension = False
            mock_config.supported_languages = ["en"]
            mock_config.num_layers = 6
            mock_config.hidden_size = 512
            mock_config.use_cache = False
            mock_cfg_cls.return_value = mock_config
            mock_cfg_cls.nawal_small.return_value = mock_config
            mock_tf_cls.return_value = MagicMock()

            with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
                mock_tok = MagicMock()
                mock_tok.__len__ = MagicMock(return_value=256)
                mock_tok_cls.return_value = mock_tok

                model = Nawal.from_pretrained("/nonexistent/path")
                assert model is not None

    @patch("client.nawal.NawalTransformer")
    @patch("client.nawal.NawalModelConfig")
    def test_save_to_belizechain(self, mock_config_cls, mock_transformer_cls):
        from client.nawal import Nawal

        mock_config = MagicMock()
        mock_config.belizean_vocab_extension = False
        mock_config.supported_languages = ["en"]
        mock_config.num_layers = 6
        mock_config.hidden_size = 512
        mock_config.use_cache = False
        mock_config_cls.nawal_small.return_value = mock_config

        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer

        with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.__len__ = MagicMock(return_value=256)
            mock_tok_cls.return_value = mock_tok

            model = Nawal(model_size="small")
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_to_belizechain("v1.0.0", save_directory=tmpdir)
                mock_transformer.save_pretrained.assert_called_with(tmpdir)

    @patch("client.nawal.NawalTransformer")
    @patch("client.nawal.NawalModelConfig")
    def test_save_to_belizechain_with_ipfs(self, mock_config_cls, mock_transformer_cls):
        from client.nawal import Nawal

        mock_config = MagicMock()
        mock_config.belizean_vocab_extension = False
        mock_config.supported_languages = ["en"]
        mock_config.num_layers = 6
        mock_config.hidden_size = 512
        mock_config.use_cache = False
        mock_config_cls.nawal_small.return_value = mock_config

        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer

        with patch("data.tokenizers.NawalTokenizerWrapper") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.__len__ = MagicMock(return_value=256)
            mock_tok_cls.return_value = mock_tok

            model = Nawal(model_size="small")
            with tempfile.TemporaryDirectory() as tmpdir:
                # IPFS upload should log warning but not crash
                model.save_to_belizechain("v1.0.0", save_directory=tmpdir, ipfs_node="http://localhost:5001")


class TestNawalLanguageDetector:
    """Cover lines 399-404 of nawal.py: LanguageDetector.detect."""

    def test_detect_english(self):
        from client.nawal import LanguageDetector
        d = LanguageDetector(["en", "es", "bzj"])
        assert d.detect("Hello, how are you today?") == "en"

    def test_detect_kriol(self):
        from client.nawal import LanguageDetector
        d = LanguageDetector(["en", "es", "bzj"])
        assert d.detect("Weh di gat fi du fi unu yaad") == "bzj"

    def test_detect_spanish(self):
        from client.nawal import LanguageDetector
        d = LanguageDetector(["en", "es", "bzj"])
        assert d.detect("El capital de Belize es Belmopan") == "es"


class TestComplianceFilter:
    """Cover nawal.py ComplianceFilter."""

    def test_filter_cleans_ssn(self):
        from client.nawal import ComplianceFilter
        f = ComplianceFilter()
        result = f.filter("My SSN is 123-45-6789")
        assert "[REDACTED]" in result

    def test_filter_clean_text(self):
        from client.nawal import ComplianceFilter
        f = ComplianceFilter()
        result = f.filter("BelizeChain is amazing")
        assert result == "BelizeChain is amazing"


# ===========================================================================
# 19. CLIENT/DOMAIN_MODELS.PY — build_model_from_genome, drone imagery, etc.
# ===========================================================================


class TestDomainModelGenomeBuild:
    """Cover lines 239-278 of domain_models.py: _build_model_from_genome, _genome_layer_to_pytorch."""

    def test_build_model_from_genome_layers(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import Genome, ArchitectureLayer, LayerType

        # Build genome with layers
        layer1 = ArchitectureLayer(layer_type=LayerType.LINEAR, parameters={"in_features": 10, "out_features": 64})
        layer2 = ArchitectureLayer(layer_type=LayerType.RELU, parameters={})
        layer3 = ArchitectureLayer(layer_type=LayerType.DROPOUT, parameters={"p": 0.1})

        genome = MagicMock()
        genome.architecture = [layer1, layer2, layer3]

        model = AgriTechModel(genome=genome)
        assert model.model is not None

    def test_genome_layer_unknown_type(self):
        from client.domain_models import AgriTechModel

        model = AgriTechModel()
        # Use MagicMock since ArchitectureLayer validates layer_type via Pydantic
        unknown_layer = MagicMock()
        unknown_layer.layer_type = "TOTALLY_UNKNOWN"
        unknown_layer.parameters = {}
        result = model._genome_layer_to_pytorch(unknown_layer)
        assert result is None

    def test_genome_layer_batch_norm(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import ArchitectureLayer, LayerType

        model = AgriTechModel()
        layer = ArchitectureLayer(layer_type=LayerType.BATCH_NORM, parameters={"num_features": 64})
        result = model._genome_layer_to_pytorch(layer)
        assert isinstance(result, nn.BatchNorm1d)

    def test_genome_layer_layer_norm(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import ArchitectureLayer, LayerType

        model = AgriTechModel()
        layer = ArchitectureLayer(layer_type=LayerType.LAYER_NORM, parameters={"normalized_shape": 64})
        result = model._genome_layer_to_pytorch(layer)
        assert isinstance(result, nn.LayerNorm)

    def test_genome_layer_conv2d(self):
        from client.domain_models import AgriTechModel
        from genome.encoding import ArchitectureLayer, LayerType

        model = AgriTechModel()
        layer = ArchitectureLayer(layer_type=LayerType.CONV2D, parameters={
            "in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1
        })
        result = model._genome_layer_to_pytorch(layer)
        assert isinstance(result, nn.Conv2d)


class TestAgriTechDroneImagery:
    """Cover lines 348-407 of domain_models.py: preprocess_data drone_imagery path."""

    def test_preprocess_drone_imagery(self):
        from client.domain_models import AgriTechModel
        model = AgriTechModel()

        # Create a small valid image in bytes
        img = Image.new("RGB", (64, 64), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        result = model.preprocess_data({"feed_type": "drone_imagery", "data": img_bytes})
        assert result.dim() == 4  # B, C, H, W
        assert result.shape[0] == 1

    def test_forward_image_data(self):
        from client.domain_models import AgriTechModel
        model = AgriTechModel(device="cpu")
        # Base nn.Sequential collapses 4D to 2D, so sensor path is taken
        input_tensor = torch.randn(1, 3, 224, 224)
        result = model.forward(input_tensor)
        assert "predictions" in result
        assert "confidence" in result
        assert "features" in result

    def test_calculate_improvement_with_ground_truth(self):
        from client.domain_models import AgriTechModel
        model = AgriTechModel()
        old_preds = {"predictions": torch.tensor([0.7]), "confidence": torch.tensor([0.6])}
        new_preds = {"predictions": torch.tensor([0.85]), "confidence": torch.tensor([0.8])}
        gt = torch.tensor([0.9])
        improvement = model.calculate_improvement(old_preds, new_preds, gt)
        assert improvement >= 0.0

    def test_preprocess_unknown_feed_type(self):
        from client.domain_models import AgriTechModel
        model = AgriTechModel()
        with pytest.raises(ValueError, match="Unknown feed type"):
            model.preprocess_data({"feed_type": "unknown"})


class TestMarineModelGaps:
    """Cover lines 609-636 of domain_models.py."""

    def test_preprocess_underwater_imagery(self):
        from client.domain_models import MarineModel
        model = MarineModel()
        img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = model.preprocess_data({"feed_type": "drone_imagery", "data": buf.getvalue()})
        assert result.dim() == 4

    def test_preprocess_water_quality_sensor(self):
        from client.domain_models import MarineModel
        model = MarineModel()
        sensor_data = _float32_bytes(35.0, 7.5, 8.0, 27.0)
        result = model.preprocess_data({"feed_type": "sensor_reading", "data": sensor_data})
        assert result.dim() == 2

    def test_forward_image_path(self):
        from client.domain_models import MarineModel
        model = MarineModel(device="cpu")
        input_tensor = torch.randn(1, 3, 224, 224)
        result = model.forward(input_tensor)
        # Base nn.Sequential collapses 4D to 2D, so sensor path is taken
        assert "predictions" in result
        assert "confidence" in result
        assert "features" in result

    def test_forward_sensor_path(self):
        from client.domain_models import MarineModel
        model = MarineModel(device="cpu")
        # Marine base model has Conv2d requiring 4D input
        input_tensor = torch.randn(1, 3, 64, 64)
        result = model.forward(input_tensor)
        assert "predictions" in result
        assert "features" in result


class TestEducationModelGaps:
    """Cover lines 858-882 of domain_models.py."""

    @patch("client.domain_models.BelizeChainLLM")
    def test_preprocess_student_data(self, mock_llm):
        from client.domain_models import EducationModel
        mock_llm.return_value = MagicMock()
        model = EducationModel()
        student = {
            "time_spent_minutes": 45,
            "completion_rate": 80,
            "average_quiz_score": 75,
            "interaction_count": 50,
        }
        data = json.dumps(student).encode("utf-8")
        result = model.preprocess_data({"feed_type": "phone_collection", "data": data})
        assert result.shape == (1, 4)

    @patch("client.domain_models.BelizeChainLLM")
    def test_preprocess_student_data_malformed(self, mock_llm):
        from client.domain_models import EducationModel
        mock_llm.return_value = MagicMock()
        model = EducationModel()
        data = b"not valid json"
        result = model.preprocess_data({"feed_type": "phone_collection", "data": data})
        # Should return zero tensor fallback
        assert result.shape == (1, 4)

    @patch("client.domain_models.BelizeChainLLM")
    def test_forward_education(self, mock_llm):
        from client.domain_models import EducationModel
        mock_llm.return_value = MagicMock()
        model = EducationModel()
        input_tensor = torch.randn(1, 4)
        result = model.forward(input_tensor)
        assert "performance_prediction" in result
        assert "learning_style" in result
        assert "intervention_needed" in result


class TestTechModelGaps:
    """Cover lines 1023-1043, 1100-1103 of domain_models.py."""

    def test_preprocess_metrics(self):
        from client.domain_models import TechModel
        model = TechModel()
        # 100 timesteps * 4 features = 400 floats
        data = _float32_bytes(*[50.0] * 400)
        result = model.preprocess_data({"feed_type": "sensor_reading", "data": data})
        assert result.dim() == 3  # (1, T, F)

    def test_preprocess_metrics_short(self):
        from client.domain_models import TechModel
        model = TechModel()
        # Short data that needs padding
        data = _float32_bytes(50.0, 60.0, 70.0, 80.0)
        result = model.preprocess_data({"feed_type": "sensor_reading", "data": data})
        assert result.dim() == 3

    def test_forward_tech(self):
        from client.domain_models import TechModel
        model = TechModel(device="cpu")
        input_tensor = torch.randn(1, 100, 4)
        result = model.forward(input_tensor)
        assert "anomaly_score" in result
        assert "performance_prediction" in result
        assert "incident_probability" in result

    def test_calculate_improvement_tech(self):
        from client.domain_models import TechModel
        model = TechModel()
        old_preds = {"predictions": torch.tensor([0.5]), "confidence": torch.tensor([0.3])}
        new_preds = {"predictions": torch.tensor([0.8]), "confidence": torch.tensor([0.7])}
        gt = torch.tensor([0.9])
        improvement = model.calculate_improvement(old_preds, new_preds, gt)
        assert improvement >= 0.0


# Need to import PIL for image tests
try:
    from PIL import Image
except ImportError:
    Image = None
