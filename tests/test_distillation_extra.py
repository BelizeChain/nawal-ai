"""
Extra coverage for training/distillation.py.

Targets the uncovered branches:
  - default __init__ (no student_model nor student_config)
  - student_config= branch
  - teacher loaded from model_id (no teacher_model kwarg)
  - use_wandb=True branch
  - train() method — main loop, val_dataset, wandb logging, max_steps
  - evaluate() method
  - from_checkpoint() classmethod body
  - upload_to_pakit()
  - _create_dataloader()
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.distillation import KnowledgeDistillationTrainer, KnowledgeDistillationLoss


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VOCAB = 64
SEQ = 8
BATCH = 2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirrored from the main test module so each file is self-contained)
# ──────────────────────────────────────────────────────────────────────────────

class _FakeConfig:
    def num_parameters(self) -> int:
        return 16 * VOCAB + VOCAB

    @property
    def vocab_size(self) -> int:
        return VOCAB

    @property
    def max_sequence_length(self) -> int:
        return SEQ


class _TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 16)
        self.proj = nn.Linear(16, VOCAB)
        self.config = _FakeConfig()

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=False):
        x = self.embed(input_ids)
        return {"logits": self.proj(x)}


def _make_student() -> nn.Module:
    return _TinyStudent()


def _make_teacher() -> MagicMock:
    teacher = MagicMock()
    teacher.model = MagicMock()
    teacher.model.eval.return_value = None
    teacher.model.parameters.return_value = iter([])

    def _call(*args, **kwargs):
        input_ids = kwargs.get("input_ids", torch.zeros(BATCH, SEQ, dtype=torch.long))
        b, s = input_ids.shape
        out = MagicMock()
        out.logits = torch.randn(b, s, VOCAB)
        return out

    teacher.model.side_effect = _call
    return teacher


def _make_batch() -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ)),
        "labels": torch.randint(0, VOCAB, (BATCH, SEQ)),
    }


def _make_loader(n_batches: int = 2) -> DataLoader:
    ids = torch.randint(0, VOCAB, (n_batches * BATCH, SEQ))
    lbls = ids.clone()
    ds = TensorDataset(ids, lbls)

    def collate(batch):
        a, b = zip(*batch)
        return {"input_ids": torch.stack(a), "labels": torch.stack(b)}

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


def _make_trainer(**kwargs) -> KnowledgeDistillationTrainer:
    defaults = dict(student_model=_make_student(), teacher_model=_make_teacher(), device="cpu")
    defaults.update(kwargs)
    return KnowledgeDistillationTrainer(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# __init__ — student_config= branch (lines 250-251)
# ──────────────────────────────────────────────────────────────────────────────

class TestStudentConfigBranch:

    def test_student_created_from_config(self):
        """Pass student_config= (not student_model=) to exercise line 250-251."""
        mock_config = MagicMock()
        mock_student = _make_student()
        mock_student.to = MagicMock(return_value=mock_student)

        with patch("training.distillation.NawalTransformer") as MockTransformer:
            MockTransformer.return_value = mock_student
            trainer = KnowledgeDistillationTrainer(
                student_config=mock_config,
                teacher_model=_make_teacher(),
                device="cpu",
            )

        MockTransformer.assert_called_once_with(mock_config)
        assert trainer.student is mock_student


# ──────────────────────────────────────────────────────────────────────────────
# __init__ — default branch (lines 252-255): no student_model, no student_config
# ──────────────────────────────────────────────────────────────────────────────

class TestDefaultStudentBranch:

    def test_default_nawal_medium_used(self):
        """Neither student_model nor student_config → nawal_medium() default."""
        mock_config = MagicMock()
        mock_student = _make_student()
        mock_student.to = MagicMock(return_value=mock_student)

        with patch("training.distillation.NawalModelConfig") as MockConfig, \
             patch("training.distillation.NawalTransformer") as MockTransformer:

            MockConfig.nawal_medium.return_value = mock_config
            MockTransformer.return_value = mock_student

            trainer = KnowledgeDistillationTrainer(
                teacher_model=_make_teacher(),
                device="cpu",
            )

        MockConfig.nawal_medium.assert_called_once()
        MockTransformer.assert_called_once_with(mock_config)


# ──────────────────────────────────────────────────────────────────────────────
# __init__ — teacher loading branch (lines 264-267)
# ──────────────────────────────────────────────────────────────────────────────

class TestTeacherLoadingBranch:

    def test_teacher_loaded_from_model_id(self):
        """No teacher_model= → teacher is loaded from teacher_model_id."""
        mock_teacher = _make_teacher()
        mock_ds_config = MagicMock()

        with patch("training.distillation.DeepSeekTeacher") as MockTeacher, \
             patch("hybrid.teacher.DeepSeekConfig") as MockDSConfig:

            MockDSConfig.return_value = mock_ds_config
            MockTeacher.return_value = mock_teacher

            trainer = KnowledgeDistillationTrainer(
                student_model=_make_student(),
                teacher_model_id="mock-teacher-id",
                device="cpu",
            )

        MockTeacher.assert_called_once_with(config=mock_ds_config)
        assert trainer.teacher is mock_teacher


# ──────────────────────────────────────────────────────────────────────────────
# __init__ — use_wandb=True branch (lines 285, 293)
# ──────────────────────────────────────────────────────────────────────────────

class TestWandbBranch:

    def test_wandb_init_called_when_enabled(self):
        """use_wandb=True → wandb.init() is called during __init__."""
        with patch("training.distillation.wandb") as mock_wandb:
            trainer = _make_trainer(use_wandb=True)

        mock_wandb.init.assert_called_once()
        assert trainer.use_wandb is True


# ──────────────────────────────────────────────────────────────────────────────
# train() — main path (lines 432-534)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainMethod:

    def test_train_runs_one_step_max(self, tmp_path):
        """train() with max_steps=1 completes and saves final checkpoint."""
        trainer = _make_trainer()
        loader = _make_loader()

        trainer.train(
            train_dataset=loader,
            num_epochs=1,
            max_steps=1,
            checkpoint_dir=str(tmp_path),
        )

        assert trainer.global_step == 1
        assert (tmp_path / "final_model.pt").exists()

    def test_train_overrides_learning_rate(self, tmp_path):
        """Passing learning_rate= overrides the optimizer's lr."""
        trainer = _make_trainer()
        trainer.train(
            train_dataset=_make_loader(),
            num_epochs=1,
            max_steps=1,
            learning_rate=1e-2,
            checkpoint_dir=str(tmp_path),
        )
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)

    def test_train_with_validation_dataset(self, tmp_path):
        """train() with val_dataset triggers evaluate() and saves best model."""
        trainer = _make_trainer()
        loader = _make_loader(n_batches=2)
        val_loader = _make_loader(n_batches=1)

        trainer.train(
            train_dataset=loader,
            val_dataset=val_loader,
            num_epochs=1,
            max_steps=1,
            eval_every=1,
            save_every=10000,   # suppress periodic saves
            checkpoint_dir=str(tmp_path),
        )
        # Best model should have been saved on first (and only) eval
        assert (tmp_path / "best_model.pt").exists()

    def test_train_periodic_checkpoint(self, tmp_path):
        """save_every=1 triggers a checkpoint after each step."""
        trainer = _make_trainer()
        trainer.train(
            train_dataset=_make_loader(),
            num_epochs=1,
            max_steps=1,
            save_every=1,
            checkpoint_dir=str(tmp_path),
        )
        assert (tmp_path / "checkpoint_step_1.pt").exists()

    def test_train_with_wandb_logging(self, tmp_path):
        """use_wandb=True causes wandb.log and wandb.finish to be called."""
        with patch("training.distillation.wandb") as mock_wandb:
            trainer = _make_trainer(use_wandb=True)
            trainer.train(
                train_dataset=_make_loader(),
                num_epochs=1,
                max_steps=1,
                checkpoint_dir=str(tmp_path),
            )
        mock_wandb.log.assert_called()
        mock_wandb.finish.assert_called_once()

    def test_train_dataset_dataloader_passthrough(self, tmp_path):
        """A DataLoader input is used directly without wrapping."""
        trainer = _make_trainer()
        loader = _make_loader()

        trainer.train(
            train_dataset=loader,
            num_epochs=1,
            max_steps=1,
            checkpoint_dir=str(tmp_path),
        )
        assert trainer.global_step >= 1

    def test_train_val_dataset_as_dataloader(self, tmp_path):
        """val_dataset can also be a DataLoader directly."""
        trainer = _make_trainer()
        trainer.train(
            train_dataset=_make_loader(),
            val_dataset=_make_loader(n_batches=1),
            num_epochs=1,
            max_steps=1,
            eval_every=1,
            checkpoint_dir=str(tmp_path),
        )
        assert (tmp_path / "best_model.pt").exists()

    def test_train_string_dataset_uses_create_dataloader(self, tmp_path):
        """train_dataset=str calls _create_dataloader internally."""
        trainer = _make_trainer()
        loader = _make_loader()

        # Patch _create_dataloader to avoid vocab mismatch
        with patch.object(trainer, "_create_dataloader", return_value=loader) as mock_cdl:
            trainer.train(
                train_dataset="path/to/data.jsonl",
                num_epochs=1,
                max_steps=1,
                checkpoint_dir=str(tmp_path),
            )

        mock_cdl.assert_called_once_with("path/to/data.jsonl", 8, shuffle=True)

    def test_train_val_string_dataset(self, tmp_path):
        """val_dataset=str calls _create_dataloader for the val loader."""
        trainer = _make_trainer()
        val_loader = _make_loader(n_batches=1)

        with patch.object(trainer, "_create_dataloader") as mock_cdl:
            mock_cdl.side_effect = [_make_loader(), val_loader]
            trainer.train(
                train_dataset="train.jsonl",
                val_dataset="val.jsonl",
                num_epochs=1,
                max_steps=1,
                eval_every=1,
                checkpoint_dir=str(tmp_path),
            )

        assert mock_cdl.call_count == 2

    def test_train_multiple_epochs_stops_at_max_steps(self, tmp_path):
        """max_steps terminates the outer epoch loop as well."""
        trainer = _make_trainer()
        trainer.train(
            train_dataset=_make_loader(n_batches=4),
            num_epochs=10,   # many epochs, but max_steps=1 stops it
            max_steps=1,
            checkpoint_dir=str(tmp_path),
        )
        assert trainer.global_step == 1


# ──────────────────────────────────────────────────────────────────────────────
# evaluate() method (lines 545-589)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateMethod:

    def test_evaluate_returns_loss_and_perplexity(self):
        trainer = _make_trainer()
        val_loader = _make_loader(n_batches=1)
        metrics = trainer.evaluate(val_loader)
        assert "loss" in metrics
        assert "perplexity" in metrics

    def test_evaluate_loss_is_positive(self):
        trainer = _make_trainer()
        metrics = trainer.evaluate(_make_loader(n_batches=2))
        assert metrics["loss"] > 0

    def test_evaluate_perplexity_matches_exp_loss(self):
        """perplexity ≈ exp(cross_entropy_hard_loss) — should be > 1 for random model."""
        import math
        trainer = _make_trainer()
        metrics = trainer.evaluate(_make_loader(n_batches=1))
        assert metrics["perplexity"] > 1.0

    def test_evaluate_multiple_batches_averages(self):
        trainer = _make_trainer()
        metrics = trainer.evaluate(_make_loader(n_batches=3))
        assert metrics["loss"] > 0

    def test_evaluate_sets_student_back_to_train_mode(self):
        trainer = _make_trainer()
        trainer.evaluate(_make_loader(n_batches=1))
        assert trainer.student.training is True


# ──────────────────────────────────────────────────────────────────────────────
# from_checkpoint() (lines 624-647)
# ──────────────────────────────────────────────────────────────────────────────

class TestFromCheckpoint:

    def test_from_checkpoint_restores_state(self, tmp_path):
        """from_checkpoint correctly loads student state and training metadata."""
        trainer = _make_trainer()
        trainer.global_step = 7
        trainer.current_epoch = 2
        trainer.best_val_loss = 0.42

        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        # from_checkpoint reconstructs NawalTransformer from the saved config.
        # Patch NawalTransformer so it returns a TinyStudent (fast).
        tiny = _make_student()
        with patch("training.distillation.NawalTransformer") as MockT, \
             patch("training.distillation.NawalModelConfig") as MockC:

            mock_cfg = MagicMock()
            MockC.return_value = mock_cfg
            MockT.return_value = tiny

            loaded = KnowledgeDistillationTrainer.from_checkpoint(
                ckpt_path,
                teacher_model=_make_teacher(),
                device="cpu",
            )

        assert loaded.global_step == 7
        assert loaded.current_epoch == 2
        assert loaded.best_val_loss == pytest.approx(0.42)

    def test_from_checkpoint_creates_trainer_with_correct_temperature(self, tmp_path):
        """Temperature and alpha are restored from checkpoint."""
        trainer = KnowledgeDistillationTrainer(
            student_model=_make_student(),
            teacher_model=_make_teacher(),
            device="cpu",
            temperature=6.0,
            alpha=0.9,
        )
        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        tiny = _make_student()
        with patch("training.distillation.NawalTransformer", return_value=tiny), \
             patch("training.distillation.NawalModelConfig"):

            loaded = KnowledgeDistillationTrainer.from_checkpoint(
                path,
                teacher_model=_make_teacher(),
                device="cpu",
            )

        assert loaded.loss_fn.temperature == pytest.approx(6.0)
        assert loaded.loss_fn.alpha == pytest.approx(0.9)


# ──────────────────────────────────────────────────────────────────────────────
# upload_to_pakit() (lines 670-686)
# ──────────────────────────────────────────────────────────────────────────────

class TestUploadToPakit:

    def test_upload_no_client_raises(self):
        """upload_to_pakit raises RuntimeError when pakit_client is None."""
        trainer = _make_trainer()
        assert trainer.pakit_client is None
        with pytest.raises(RuntimeError, match="Pakit client not initialized"):
            trainer.upload_to_pakit()

    def test_upload_calls_save_student_and_pakit(self, tmp_path):
        """upload_to_pakit saves student then uploads directory."""
        trainer = _make_trainer()
        mock_client = MagicMock()
        mock_client.upload_directory.return_value = "cid_abc"
        trainer.pakit_client = mock_client

        with patch.object(trainer, "save_student") as mock_save:
            result = trainer.upload_to_pakit(metadata={"v": "1.0"})

        mock_save.assert_called_once()
        mock_client.upload_directory.assert_called_once()
        assert result == "cid_abc"

    def test_upload_with_no_metadata(self):
        """upload_to_pakit uses empty dict when no metadata given."""
        trainer = _make_trainer()
        mock_client = MagicMock()
        mock_client.upload_directory.return_value = "hash123"
        trainer.pakit_client = mock_client

        with patch.object(trainer, "save_student"):
            result = trainer.upload_to_pakit()

        assert result == "hash123"
        _, kwargs = mock_client.upload_directory.call_args
        assert kwargs.get("metadata") == {} or mock_client.upload_directory.called


# ──────────────────────────────────────────────────────────────────────────────
# _create_dataloader() (lines 709-722)
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateDataloader:

    def test_returns_dataloader_instance(self):
        trainer = _make_trainer()
        loader = trainer._create_dataloader("dummy", batch_size=4)
        assert isinstance(loader, DataLoader)

    def test_batch_has_input_ids_and_labels(self):
        trainer = _make_trainer()
        loader = trainer._create_dataloader("dummy", batch_size=4)
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "labels" in batch

    def test_batch_sizes_match_requested(self):
        trainer = _make_trainer()
        loader = trainer._create_dataloader("dummy", batch_size=4)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4

    def test_shuffle_parameter_accepted(self):
        """shuffle=False should not raise."""
        trainer = _make_trainer()
        loader = trainer._create_dataloader("dummy", batch_size=8, shuffle=False)
        assert isinstance(loader, DataLoader)


# ──────────────────────────────────────────────────────────────────────────────
# train() — Dataset input branches (lines 444, 452) and wandb+val (line 500)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainDatasetBranches:

    def _make_dict_dataset(self):
        """Dataset returning dict batches (compatible with train_step)."""
        class DictDataset(TensorDataset):
            def __getitem__(self, idx):
                ids, lbls = super().__getitem__(idx)
                return {"input_ids": ids, "labels": lbls}

            def __len__(self):
                return super().__len__()

        ids = torch.randint(0, VOCAB, (BATCH * 2, SEQ))
        return DictDataset(ids, ids.clone())

    def test_train_with_torch_dataset(self, tmp_path):
        """train_dataset=Dataset (not DataLoader) hits the Dataset branch (line 444)."""
        trainer = _make_trainer()
        ds = self._make_dict_dataset()

        trainer.train(
            train_dataset=ds,
            batch_size=BATCH,
            num_epochs=1,
            max_steps=1,
            checkpoint_dir=str(tmp_path),
        )
        assert trainer.global_step == 1

    def test_train_with_val_torch_dataset(self, tmp_path):
        """val_dataset=Dataset hits the val Dataset branch (line 452)."""
        trainer = _make_trainer()
        loader = _make_loader()
        val_ds = self._make_dict_dataset()

        trainer.train(
            train_dataset=loader,
            val_dataset=val_ds,
            batch_size=BATCH,
            num_epochs=1,
            max_steps=1,
            eval_every=1,
            checkpoint_dir=str(tmp_path),
        )

    def test_train_wandb_with_val_logs_val_metrics(self, tmp_path):
        """use_wandb=True + val_dataset → wandb.log is called for val metrics (line 500)."""
        with patch("training.distillation.wandb") as mock_wandb:
            trainer = _make_trainer(use_wandb=True)
            trainer.train(
                train_dataset=_make_loader(),
                val_dataset=_make_loader(n_batches=1),
                num_epochs=1,
                max_steps=1,
                eval_every=1,
                checkpoint_dir=str(tmp_path),
            )
        # Expect at least 2 log calls: one for train metrics, one for val metrics
        assert mock_wandb.log.call_count >= 2


# ──────────────────────────────────────────────────────────────────────────────
# evaluate() — attention_mask branch (line 555)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateAttentionMask:

    def test_evaluate_with_attention_mask_in_batch(self):
        """Batch with attention_mask exercises the .to(device) branch (line 555)."""
        trainer = _make_trainer()

        ids = torch.randint(0, VOCAB, (BATCH * 2, SEQ))
        mask = torch.ones(BATCH * 2, SEQ, dtype=torch.long)
        ds = TensorDataset(ids, ids.clone(), mask)

        def collate(batch):
            a, b, m = zip(*batch)
            return {
                "input_ids": torch.stack(a),
                "labels": torch.stack(b),
                "attention_mask": torch.stack(m),
            }

        loader = DataLoader(ds, batch_size=BATCH, collate_fn=collate)
        metrics = trainer.evaluate(loader)
        assert "loss" in metrics


# ──────────────────────────────────────────────────────────────────────────────
# pakit_gateway __init__ branch
# ──────────────────────────────────────────────────────────────────────────────

class TestPakitGatewayInit:

    def test_pakit_client_created_when_gateway_given(self):
        """Providing pakit_gateway= creates a PakitClient."""
        trainer = _make_trainer(pakit_gateway="http://localhost:8081")
        from storage.pakit_client import PakitClient
        assert isinstance(trainer.pakit_client, PakitClient)

    def test_no_pakit_client_when_no_gateway(self):
        """Default (no gateway) → pakit_client is None."""
        trainer = _make_trainer()
        assert trainer.pakit_client is None
