"""
Tests for storage/ — Priority 2 operational layer.

Covers:
  - storage/metrics_db.py    → MetricsStore
  - storage/pakit_client.py  → PakitClient (HTTP calls mocked)
  - storage/checkpoint_manager.py → CheckpointManager (Pakit calls mocked)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch

from storage.metrics_db import MetricsStore
from storage.pakit_client import PakitClient
from storage.checkpoint_manager import CheckpointManager

# ===========================================================================
# MetricsStore
# ===========================================================================


class TestMetricsStore:

    def test_init_with_no_db_path(self):
        store = MetricsStore()
        assert store.db_path is None
        assert store._metrics == []

    def test_init_with_db_path(self):
        store = MetricsStore(db_path="/tmp/test.db")
        assert store.db_path == "/tmp/test.db"

    def test_record_stores_metric(self):
        store = MetricsStore()
        store.record("training_loss", 0.5, epoch="1")
        assert len(store._metrics) == 1
        assert store._metrics[0]["name"] == "training_loss"
        assert store._metrics[0]["value"] == 0.5

    def test_record_stores_labels(self):
        store = MetricsStore()
        store.record("accuracy", 0.9, epoch="3", split="val")
        entry = store._metrics[0]
        assert entry["epoch"] == "3"
        assert entry["split"] == "val"

    def test_get_all_metrics_no_filter(self):
        store = MetricsStore()
        store.record("loss", 0.5)
        store.record("acc", 0.9)
        result = store.get_metrics()
        assert len(result) == 2

    def test_get_metrics_filtered_by_name(self):
        store = MetricsStore()
        store.record("loss", 0.5)
        store.record("acc", 0.9)
        store.record("loss", 0.3)
        result = store.get_metrics("loss")
        assert len(result) == 2
        assert all(m["name"] == "loss" for m in result)

    def test_get_metrics_unknown_name_returns_empty(self):
        store = MetricsStore()
        store.record("loss", 0.5)
        assert store.get_metrics("nonexistent") == []

    def test_returns_copy_not_reference(self):
        store = MetricsStore()
        store.record("loss", 0.5)
        result = store.get_metrics()
        result.clear()  # Mutate the returned list
        assert len(store._metrics) == 1  # Original unchanged

    def test_multiple_records_all_retrieved(self):
        store = MetricsStore()
        for i in range(10):
            store.record("step_loss", float(i))
        assert len(store.get_metrics("step_loss")) == 10


# ===========================================================================
# PakitClient — init and environment
# ===========================================================================


class TestPakitClientInit:

    def test_default_urls(self):
        client = PakitClient()
        assert client.pakit_api_url == "http://localhost:8080"
        assert client.dag_gateway_url == "http://localhost:8081"
        assert client.compression == "zstd"

    def test_custom_init(self):
        client = PakitClient(
            pakit_api_url="http://prod-api:9090",
            dag_gateway_url="http://prod-dag:9091",
            compression="lz4",
        )
        assert client.pakit_api_url == "http://prod-api:9090"
        assert client.compression == "lz4"

    def test_from_env_uses_env_vars(self):
        with patch.dict(
            "os.environ",
            {
                "PAKIT_API_URL": "http://env-api:7070",
                "PAKIT_DAG_GATEWAY_URL": "http://env-dag:7071",
                "PAKIT_COMPRESSION": "brotli",
            },
        ):
            client = PakitClient.from_env()
        assert client.pakit_api_url == "http://env-api:7070"
        assert client.dag_gateway_url == "http://env-dag:7071"
        assert client.compression == "brotli"

    def test_from_env_defaults_when_not_set(self):
        with patch.dict("os.environ", {}, clear=False):
            # Remove keys if present
            import os

            for k in ("PAKIT_API_URL", "PAKIT_DAG_GATEWAY_URL", "PAKIT_COMPRESSION"):
                os.environ.pop(k, None)
            client = PakitClient.from_env()
        assert client.pakit_api_url == "http://localhost:8080"


# ===========================================================================
# PakitClient — mock upload (offline)
# ===========================================================================


class TestPakitClientMockUpload:

    def test_mock_upload_returns_hex_string(self, tmp_path):
        client = PakitClient()
        path = str(tmp_path / "file.bin")
        hash_val = client._mock_upload(path)
        int(hash_val, 16)  # Should be valid hex

    def test_mock_upload_is_deterministic(self, tmp_path):
        client = PakitClient()
        path = str(tmp_path / "model.pt")
        h1 = client._mock_upload(path)
        h2 = client._mock_upload(path)
        assert h1 == h2

    def test_mock_upload_different_paths_give_different_hashes(self, tmp_path):
        client = PakitClient()
        h1 = client._mock_upload(str(tmp_path / "a.pt"))
        h2 = client._mock_upload(str(tmp_path / "b.pt"))
        assert h1 != h2

    def test_mock_upload_metadata_changes_hash(self, tmp_path):
        client = PakitClient()
        path = str(tmp_path / "same.pt")
        h1 = client._mock_upload(path, metadata={"v": "1"})
        h2 = client._mock_upload(path, metadata={"v": "2"})
        assert h1 != h2


# ===========================================================================
# PakitClient — HTTP paths (mocked requests)
# ===========================================================================


class TestPakitClientHTTP:

    def test_upload_success(self, tmp_path):
        """upload_file returns content_hash on 200 response."""
        # Create a real temp file to upload
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"fake model weights")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hash": "abc123", "content_hash": "abc123"}

        with patch("storage.pakit_client.requests.post", return_value=mock_response):
            client = PakitClient()
            result = client.upload_file(str(fpath))
        assert result == "abc123"

    def test_upload_server_error_raises(self, tmp_path):
        """upload_file raises RuntimeError on non-200 response."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"data")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("storage.pakit_client.requests.post", return_value=mock_response):
            client = PakitClient()
            with pytest.raises(RuntimeError, match="500"):
                client.upload_file(str(fpath))

    def test_upload_no_requests_raises_runtime_error(self):
        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", False),
            pytest.raises(RuntimeError, match="requests"),
        ):
            PakitClient().upload_file("/any/path.pt")

    def test_download_success(self, tmp_path):
        """download_file returns True and writes file on 200."""
        valid_cid = "ab" * 32  # 64-char hex
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        output_path = str(tmp_path / "downloaded.bin")

        with (
            patch("storage.pakit_client.requests.get", return_value=mock_response),
            patch.object(PakitClient, "_compute_file_hash", return_value=valid_cid),
        ):
            client = PakitClient()
            result = client.download_file(valid_cid, output_path)

        assert result is True
        assert Path(output_path).read_bytes() == b"chunk1chunk2"

    def test_download_not_found_returns_false(self, tmp_path):
        valid_cid = "cd" * 32
        mock_response = MagicMock()
        mock_response.status_code = 404
        output_path = str(tmp_path / "out.bin")

        with patch("storage.pakit_client.requests.get", return_value=mock_response):
            result = PakitClient().download_file(valid_cid, output_path)

        assert result is False

    def test_download_no_requests_returns_false(self):
        valid_cid = "ef" * 32
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", False):
            result = PakitClient().download_file(valid_cid, "/tmp/out.bin")
        assert result is False

    def test_pin_content_success(self):
        valid_cid = "ab" * 32
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("storage.pakit_client.requests.post", return_value=mock_response):
            result = PakitClient().pin_content(valid_cid)
        assert result is True

    def test_pin_content_failure(self):
        valid_cid = "cd" * 32
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("storage.pakit_client.requests.post", return_value=mock_response):
            result = PakitClient().pin_content(valid_cid)
        assert result is False

    def test_pin_content_no_requests_returns_true(self):
        """When requests not available, pin mocks success."""
        valid_cid = "ef" * 32
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", False):
            result = PakitClient().pin_content(valid_cid)
        assert result is True

    def test_get_metadata_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0", "size": 1024}

        with patch("storage.pakit_client.requests.get", return_value=mock_response):
            meta = PakitClient().get_metadata("ab" * 32)
        assert meta == {"version": "1.0", "size": 1024}

    def test_get_metadata_not_found_returns_none(self):
        valid_cid = "cd" * 32
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("storage.pakit_client.requests.get", return_value=mock_response):
            meta = PakitClient().get_metadata(valid_cid)
        assert meta is None

    def test_get_metadata_no_requests_returns_none(self):
        valid_cid = "ef" * 32
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", False):
            meta = PakitClient().get_metadata(valid_cid)
        assert meta is None

    def test_upload_network_error_raises(self, tmp_path):
        """RequestException wraps as RuntimeError."""
        import requests

        fpath = tmp_path / "file.bin"
        fpath.write_bytes(b"data")

        with (
            patch(
                "storage.pakit_client.requests.post",
                side_effect=requests.RequestException("connection refused"),
            ),
            pytest.raises(RuntimeError, match="Pakit upload failed"),
        ):
            PakitClient().upload_file(str(fpath))


# ===========================================================================
# CheckpointManager — local operations (no Pakit)
# ===========================================================================


@pytest.fixture
def ckpt_manager(tmp_path) -> CheckpointManager:
    return CheckpointManager(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        pakit_client=None,
        auto_upload=False,
    )


def _make_state_dict() -> dict:
    return {"layer1.weight": torch.tensor([1.0, 2.0, 3.0])}


class TestCheckpointManagerLocal:

    def test_init_creates_directory(self, tmp_path):
        ckpt_dir = tmp_path / "new_checkpoints"
        CheckpointManager(checkpoint_dir=str(ckpt_dir))
        assert ckpt_dir.exists()

    def test_init_empty_registry(self, ckpt_manager):
        assert ckpt_manager.registry == {}

    def test_save_checkpoint_creates_file(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "test.pt")
        assert (ckpt_manager.checkpoint_dir / "test.pt").exists()

    def test_save_checkpoint_adds_to_registry(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "test.pt")
        assert "test.pt" in ckpt_manager.registry

    def test_save_checkpoint_computes_hash(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "test.pt")
        info = ckpt_manager.registry["test.pt"]
        assert "content_hash" in info
        assert len(info["content_hash"]) == 64  # SHA-256 hex

    def test_save_checkpoint_returns_local_path(self, ckpt_manager):
        result = ckpt_manager.save_checkpoint(_make_state_dict(), "test.pt")
        assert "test.pt" in result

    def test_load_checkpoint_returns_state(self, ckpt_manager):
        state = _make_state_dict()
        ckpt_manager.save_checkpoint(state, "test.pt")
        loaded = ckpt_manager.load_checkpoint("test.pt")
        assert loaded is not None
        assert torch.allclose(loaded["layer1.weight"], state["layer1.weight"])

    def test_load_nonexistent_returns_none(self, ckpt_manager):
        result = ckpt_manager.load_checkpoint("does_not_exist.pt")
        assert result is None

    def test_load_verifies_integrity(self, ckpt_manager, tmp_path):
        """Tampered checkpoint file raises integrity check failure → returns None."""
        ckpt_manager.save_checkpoint(_make_state_dict(), "tampered.pt")
        # Tamper the file
        ckpt_path = ckpt_manager.checkpoint_dir / "tampered.pt"
        ckpt_path.write_bytes(b"corrupted data")
        result = ckpt_manager.load_checkpoint("tampered.pt")
        assert result is None

    def test_list_checkpoints_returns_all(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "a.pt")
        ckpt_manager.save_checkpoint(_make_state_dict(), "b.pt")
        ckpts = ckpt_manager.list_checkpoints()
        assert len(ckpts) == 2
        names = [c["name"] for c in ckpts]
        assert "a.pt" in names
        assert "b.pt" in names

    def test_delete_checkpoint_removes_file_and_registry(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "del_me.pt")
        assert (ckpt_manager.checkpoint_dir / "del_me.pt").exists()
        result = ckpt_manager.delete_checkpoint("del_me.pt")
        assert result is True
        assert not (ckpt_manager.checkpoint_dir / "del_me.pt").exists()
        assert "del_me.pt" not in ckpt_manager.registry

    def test_delete_nonexistent_returns_false(self, ckpt_manager):
        assert ckpt_manager.delete_checkpoint("nope.pt") is False

    def test_registry_persists_to_disk(self, tmp_path):
        """Verify registry.json is written and reloaded by a fresh instance."""
        ckpt_dir = tmp_path / "persist_checkpoints"
        mgr1 = CheckpointManager(checkpoint_dir=str(ckpt_dir))
        mgr1.save_checkpoint(_make_state_dict(), "persisted.pt")

        mgr2 = CheckpointManager(checkpoint_dir=str(ckpt_dir))
        assert "persisted.pt" in mgr2.registry

    def test_metadata_stored_in_registry(self, ckpt_manager):
        meta = {"epoch": 5, "accuracy": 0.92, "notes": "best run"}
        ckpt_manager.save_checkpoint(_make_state_dict(), "meta.pt", metadata=meta)
        stored_meta = ckpt_manager.registry["meta.pt"]["metadata"]
        assert stored_meta["epoch"] == 5

    def test_content_hash_is_stable(self, ckpt_manager):
        """Same file → same hash re-computed by _compute_file_hash."""
        ckpt_manager.save_checkpoint(_make_state_dict(), "stable.pt")
        file_path = ckpt_manager.checkpoint_dir / "stable.pt"
        h1 = CheckpointManager._compute_file_hash(file_path)
        h2 = CheckpointManager._compute_file_hash(file_path)
        assert h1 == h2

    def test_integrity_check_passes_for_correct_file(self, ckpt_manager):
        ckpt_manager.save_checkpoint(_make_state_dict(), "good.pt")
        info = ckpt_manager.registry["good.pt"]
        file_path = ckpt_manager.checkpoint_dir / "good.pt"
        assert ckpt_manager._verify_checkpoint_integrity(file_path, info) is True

    def test_integrity_check_passes_for_legacy_no_hash(self, ckpt_manager, tmp_path):
        """Legacy checkpoint with no content_hash skips check (returns True)."""
        ckpt_manager.save_checkpoint(_make_state_dict(), "legacy.pt")
        file_path = ckpt_manager.checkpoint_dir / "legacy.pt"
        # Remove hash to simulate legacy entry
        info_no_hash = {"name": "legacy.pt", "path": str(file_path)}
        assert (
            ckpt_manager._verify_checkpoint_integrity(file_path, info_no_hash) is True
        )


# ===========================================================================
# CheckpointManager — Pakit integration (mocked)
# ===========================================================================


class TestCheckpointManagerWithPakit:

    def test_auto_upload_calls_pakit(self, tmp_path):
        """When auto_upload=True and pakit_client is set, uploads to Pakit."""
        mock_pakit = MagicMock()
        mock_pakit.upload_file.return_value = "cid_abc123"

        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path / "ckpts"),
            pakit_client=mock_pakit,
            auto_upload=True,
        )
        result = mgr.save_checkpoint(_make_state_dict(), "auto.pt")
        mock_pakit.upload_file.assert_called_once()
        assert result == "cid_abc123"
        assert mgr.registry["auto.pt"]["pakit_cid"] == "cid_abc123"

    def test_auto_upload_false_does_not_call_pakit(self, tmp_path):
        mock_pakit = MagicMock()
        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path / "ckpts"),
            pakit_client=mock_pakit,
            auto_upload=False,
        )
        mgr.save_checkpoint(_make_state_dict(), "no_upload.pt")
        mock_pakit.upload_file.assert_not_called()

    def test_pakit_upload_failure_still_saves_locally(self, tmp_path):
        """If Pakit upload fails, checkpoint still saved locally."""
        mock_pakit = MagicMock()
        mock_pakit.upload_file.side_effect = RuntimeError("network error")

        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path / "ckpts"),
            pakit_client=mock_pakit,
            auto_upload=True,
        )
        result = mgr.save_checkpoint(_make_state_dict(), "fallback.pt")
        assert (mgr.checkpoint_dir / "fallback.pt").exists()
        # result should be local path since upload failed
        assert "fallback.pt" in result

    def test_load_from_pakit_calls_download(self, tmp_path):
        """When from_pakit=True and CID exists, downloads from Pakit."""
        mock_pakit = MagicMock()

        mgr = CheckpointManager(
            checkpoint_dir=str(tmp_path / "ckpts2"),
            pakit_client=mock_pakit,
            auto_upload=False,
        )

        # Manually inject a registry entry with a pakit_cid
        state = _make_state_dict()
        original_path = mgr.checkpoint_dir / "remote.pt"
        torch.save(state, original_path)
        content_hash = CheckpointManager._compute_file_hash(original_path)

        mgr.registry["remote.pt"] = {
            "name": "remote.pt",
            "path": str(original_path),
            "pakit_cid": "cid_xyz",
            "content_hash": content_hash,
        }

        # Mock download to copy the original file to the temp path
        def _mock_download(cid, output_path):
            import shutil

            shutil.copy(str(original_path), output_path)
            return True

        mock_pakit.download_file.side_effect = _mock_download
        loaded = mgr.load_checkpoint("remote.pt", from_pakit=True)

        mock_pakit.download_file.assert_called_once()
        assert loaded is not None
