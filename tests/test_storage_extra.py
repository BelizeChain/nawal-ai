"""
Extra coverage for storage/pakit_client.py and storage/checkpoint_manager.py.

Targets the specific uncovered branches reported by coverage:
  pakit_client  : upload_directory, download/pin/metadata exception paths
  checkpoint_mgr: from_pakit branches, integrity failure, registry exceptions
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from storage.checkpoint_manager import CheckpointManager
from storage.pakit_client import PakitClient

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _mock_response(status: int = 200, json_data=None, text: str = "ok"):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.iter_content = MagicMock(return_value=iter([b"data"]))
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# PakitClient — upload_directory
# ──────────────────────────────────────────────────────────────────────────────


class TestUploadDirectory:

    def test_upload_directory_success(self, tmp_path):
        """upload_directory creates tar, uploads, and returns a content hash."""
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir" / "file.txt").write_text("hello")

        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.post.return_value = _mock_response(200, {"hash": "abc123"})
            client = PakitClient(dag_gateway_url="http://localhost:8081")
            result = client.upload_directory(str(tmp_path / "dir"))

        assert result == "abc123"

    def test_upload_directory_with_metadata(self, tmp_path):
        """Metadata is forwarded when uploading a directory."""
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir" / "model.pt").write_bytes(b"\x00" * 8)

        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.post.return_value = _mock_response(200, {"content_hash": "xyz"})
            client = PakitClient()
            result = client.upload_directory(
                str(tmp_path / "dir"), metadata={"version": "1.0"}
            )

        assert isinstance(result, str)

    def test_upload_directory_no_requests_raises(self, tmp_path):
        """upload_directory raises RuntimeError when requests is not available."""
        (tmp_path / "dir").mkdir()
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", False):
            client = PakitClient()
            with pytest.raises(RuntimeError, match="requests library required"):
                client.upload_directory(str(tmp_path / "dir"))

    def test_upload_directory_propagates_upload_error(self, tmp_path):
        """If upload_file raises, upload_directory re-raises."""
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir" / "f.bin").write_bytes(b"x")

        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.post.return_value = _mock_response(500, text="server error")
            client = PakitClient()
            with pytest.raises(Exception):
                client.upload_directory(str(tmp_path / "dir"))


# ──────────────────────────────────────────────────────────────────────────────
# PakitClient — exception paths in download / pin / get_metadata
# ──────────────────────────────────────────────────────────────────────────────


class TestExceptionPaths:

    def test_download_generic_exception_returns_false(self, tmp_path):
        """Any non-requests exception in download_file returns False."""
        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.get.side_effect = Exception("unexpected error")
            client = PakitClient()
            result = client.download_file("ab" * 32, str(tmp_path / "out.bin"))

        assert result is False

    def test_pin_content_generic_exception_returns_false(self):
        """Any exception in pin_content returns False."""
        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.post.side_effect = Exception("boom")
            client = PakitClient()
            result = client.pin_content("ab" * 32)

        assert result is False

    def test_get_metadata_generic_exception_returns_none(self):
        """Any exception in get_metadata returns None."""
        with (
            patch("storage.pakit_client.REQUESTS_AVAILABLE", True),
            patch("storage.pakit_client.requests") as mock_req,
        ):

            mock_req.get.side_effect = Exception("network failure")
            client = PakitClient()
            result = client.get_metadata("ab" * 32)

        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager — load_checkpoint branch coverage
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ckpt_mgr(tmp_path):
    return CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))


def _save_simple_ckpt(manager: CheckpointManager, name: str = "test.pt") -> str:
    state = {"layer.weight": torch.tensor([1.0, 2.0])}
    return manager.save_checkpoint(state, name)


class TestLoadCheckpointFromPakit:

    def test_from_pakit_no_client_returns_none(self, ckpt_mgr):
        """from_pakit=True but no pakit_client configured → None."""
        _save_simple_ckpt(ckpt_mgr)
        # Manually add a pakit_cid to make the from_pakit branch trigger
        ckpt_mgr.registry["test.pt"]["pakit_cid"] = "fakecid"
        ckpt_mgr._save_registry()

        assert ckpt_mgr.pakit_client is None
        result = ckpt_mgr.load_checkpoint("test.pt", from_pakit=True)
        assert result is None

    def test_from_pakit_download_fails_returns_none(self, tmp_path):
        """download_file returns False → load_checkpoint returns None."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        _save_simple_ckpt(mgr)
        mgr.registry["test.pt"]["pakit_cid"] = "fakecid"
        mgr._save_registry()

        mock_client = MagicMock()
        mock_client.download_file.return_value = False
        mgr.pakit_client = mock_client

        result = mgr.load_checkpoint("test.pt", from_pakit=True)
        assert result is None

    def test_from_pakit_integrity_failure_returns_none(self, tmp_path):
        """Integrity check fails on pakit-downloaded file → return None."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path / "ckpts"))
        _save_simple_ckpt(mgr)
        mgr.registry["test.pt"]["pakit_cid"] = "cid123"
        # Store a wrong expected hash so integrity verification fails
        mgr.registry["test.pt"]["content_hash"] = "0" * 64
        mgr._save_registry()

        def fake_download(cid, output_path):
            # Write some bytes to the output path so the file exists
            Path(output_path).write_bytes(b"corrupted data")
            return True

        mock_client = MagicMock()
        mock_client.download_file.side_effect = fake_download
        mgr.pakit_client = mock_client

        result = mgr.load_checkpoint("test.pt", from_pakit=True)
        assert result is None


class TestLoadCheckpointLocalIntegrity:

    def test_local_integrity_failure_returns_none(self, ckpt_mgr):
        """Tampered local checkpoint → integrity check returns None."""
        _save_simple_ckpt(ckpt_mgr)
        # Tamper the stored hash
        ckpt_mgr.registry["test.pt"]["content_hash"] = "deadbeef" * 8
        path = Path(ckpt_mgr.registry["test.pt"]["path"])
        # Write garbage to the file to mismatch the stored hash
        path.write_bytes(b"garbage corrupt data")

        result = ckpt_mgr.load_checkpoint("test.pt")
        assert result is None


class TestLoadCheckpointLocalMissingFile:

    def test_local_file_missing_returns_none(self, ckpt_mgr):
        """load_checkpoint returns None when the local checkpoint file is deleted (lines 160-161)."""
        _save_simple_ckpt(ckpt_mgr)
        # Delete the actual file but keep the registry entry
        ckpt_path = Path(ckpt_mgr.registry["test.pt"]["path"])
        ckpt_path.unlink()

        result = ckpt_mgr.load_checkpoint("test.pt")
        assert result is None

    def test_load_registry_handles_corrupt_json(self, tmp_path):
        """_load_registry returns {} on invalid JSON in registry file."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        registry_path = ckpt_dir / "registry.json"
        registry_path.write_text("{ not valid json !!!")

        mgr = CheckpointManager(checkpoint_dir=str(ckpt_dir))
        assert mgr.registry == {}

    def test_save_registry_handles_write_error(self, ckpt_mgr):
        """_save_registry gracefully handles IO error (does not raise)."""
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise — just log the error
            ckpt_mgr._save_registry()


# ──────────────────────────────────────────────────────────────────────────────
# PakitClient — module-level import guard (REQUESTS_AVAILABLE=False, lines 19-21)
# ──────────────────────────────────────────────────────────────────────────────


class TestRequestsUnavailableBranch:

    def test_requests_available_false_when_requests_absent(self):
        """Reload pakit_client with requests hidden → REQUESTS_AVAILABLE=False (lines 19-21)."""
        import importlib
        import sys

        import storage.pakit_client as target_module

        saved_requests = sys.modules.pop("requests", None)
        sys.modules["requests"] = None  # type: ignore[assignment]

        try:
            reloaded = importlib.reload(target_module)
            assert reloaded.REQUESTS_AVAILABLE is False
        finally:
            if saved_requests is not None:
                sys.modules["requests"] = saved_requests
            else:
                sys.modules.pop("requests", None)
            # Reload back to True state
            importlib.reload(target_module)
