"""
B11 — Storage & Pakit audit tests.

Checks:
  C11.1  Content ID integrity verification
  C11.2  Quantum compression integration resilience
  C11.3  Merkle proof submission
  C11.4  Azure staging fallback
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storage.pakit_client import PakitClient


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _mock_response(
    status: int = 200,
    content: bytes = b"model-data",
    json_data: dict | None = None,
):
    """Build a minimal mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.text = "ok"
    resp.iter_content = MagicMock(return_value=iter([content]))
    return resp


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# C11.1 — Content ID integrity verification
# ═══════════════════════════════════════════════════════════════════════════

class TestC111ContentIDIntegrity:
    """Download must recompute hash and reject mismatches; CID format validated."""

    # ── F11.1a: hash verification on download ──

    def test_download_verifies_hash_match(self, tmp_path):
        """download_file succeeds when content hash matches CID."""
        content = b"valid model data"
        cid = _sha256(content)
        out = str(tmp_path / "model.bin")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.get.return_value = _mock_response(200, content)
            client = PakitClient()
            result = client.download_file(cid, out)

        assert result is True
        assert Path(out).read_bytes() == content

    def test_download_rejects_hash_mismatch(self, tmp_path):
        """download_file returns False when hash doesn't match CID."""
        content = b"tampered data"
        wrong_cid = _sha256(b"original data")
        out = str(tmp_path / "bad.bin")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.get.return_value = _mock_response(200, content)
            client = PakitClient()
            result = client.download_file(wrong_cid, out)

        assert result is False

    def test_download_mismatch_does_not_leave_file(self, tmp_path):
        """On mismatch the partially-written file must be cleaned up."""
        content = b"corrupt"
        wrong_cid = _sha256(b"expected")
        out = str(tmp_path / "partial.bin")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.get.return_value = _mock_response(200, content)
            client = PakitClient()
            client.download_file(wrong_cid, out)

        assert not Path(out).exists()

    def test_download_hash_is_sha256(self, tmp_path):
        """The verification uses SHA-256 specifically."""
        content = b"sha256 test content"
        expected = hashlib.sha256(content).hexdigest()
        out = str(tmp_path / "sha.bin")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.get.return_value = _mock_response(200, content)
            client = PakitClient()
            result = client.download_file(expected, out)

        assert result is True

    # ── F11.1b: CID format validation ──

    def test_download_rejects_empty_cid(self, tmp_path):
        """Empty string CID is rejected before network call."""
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            client = PakitClient()
            result = client.download_file("", str(tmp_path / "out.bin"))

        assert result is False
        mock_req.get.assert_not_called()

    def test_download_rejects_non_hex_cid(self, tmp_path):
        """Non-hex CID is rejected before network call."""
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            client = PakitClient()
            result = client.download_file("not-a-valid-hex!!!", str(tmp_path / "out.bin"))

        assert result is False
        mock_req.get.assert_not_called()

    def test_download_rejects_path_traversal_cid(self, tmp_path):
        """CID with path-traversal chars is rejected."""
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            client = PakitClient()
            result = client.download_file("../../etc/passwd", str(tmp_path / "out.bin"))

        assert result is False
        mock_req.get.assert_not_called()

    def test_download_accepts_valid_64char_hex(self, tmp_path):
        """Valid 64-char hex CID passes format check (still needs hash match)."""
        content = b"data"
        cid = _sha256(content)
        out = str(tmp_path / "ok.bin")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.get.return_value = _mock_response(200, content)
            client = PakitClient()
            result = client.download_file(cid, out)

        assert result is True
        mock_req.get.assert_called_once()

    def test_upload_returns_hash_from_server(self, tmp_path):
        """upload_file returns the hash/content_hash from server response."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"model data")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.post.return_value = _mock_response(
                200, json_data={"hash": "abc123"}
            )
            client = PakitClient()
            result = client.upload_file(str(fpath))

        assert result == "abc123"

    def test_pin_rejects_invalid_cid(self):
        """pin_content rejects invalid CID format."""
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            client = PakitClient()
            result = client.pin_content("")

        assert result is False
        mock_req.post.assert_not_called()

    def test_get_metadata_rejects_invalid_cid(self):
        """get_metadata rejects invalid CID format."""
        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            client = PakitClient()
            result = client.get_metadata("../../../etc/passwd")

        assert result is None
        mock_req.get.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# C11.2 — Quantum compression integration resilience
# ═══════════════════════════════════════════════════════════════════════════

class TestC112QuantumCompression:
    """Compression via Kinich should degrade gracefully."""

    def test_upload_succeeds_without_kinich(self, tmp_path):
        """Upload works even when KINICH_ENABLED is false."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"model")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req, \
             patch.dict(os.environ, {"KINICH_ENABLED": "false"}):
            mock_req.post.return_value = _mock_response(
                200, json_data={"hash": "h1"}
            )
            client = PakitClient()
            result = client.upload_file(str(fpath))

        assert result == "h1"

    def test_pakit_client_has_no_kinich_dependency(self):
        """PakitClient does not import or reference kinich."""
        import inspect
        source = inspect.getsource(PakitClient)
        assert "kinich" not in source.lower()

    def test_compression_param_is_configurable(self):
        """Compression algo can be set to 'none' to disable compression."""
        client = PakitClient(compression="none")
        assert client.compression == "none"

    def test_from_env_reads_compression(self):
        """from_env picks up PAKIT_COMPRESSION."""
        with patch.dict(os.environ, {"PAKIT_COMPRESSION": "lz4"}):
            client = PakitClient.from_env()
        assert client.compression == "lz4"

    def test_pre_compress_fn_called_on_upload(self, tmp_path):
        """pre_compress_fn hook is invoked during upload."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"raw content")
        compress_fn = MagicMock(return_value=b"compressed content")

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.post.return_value = _mock_response(200, json_data={"hash": "h1"})
            client = PakitClient(pre_compress_fn=compress_fn)
            result = client.upload_file(str(fpath))

        compress_fn.assert_called_once_with(b"raw content")
        assert result == "h1"

    def test_pre_compress_fn_error_degrades_gracefully(self, tmp_path):
        """Upload succeeds even when pre_compress_fn raises."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"raw content")
        compress_fn = MagicMock(side_effect=RuntimeError("compression failed"))

        with patch("storage.pakit_client.REQUESTS_AVAILABLE", True), \
             patch("storage.pakit_client.requests") as mock_req:
            mock_req.post.return_value = _mock_response(200, json_data={"hash": "h2"})
            client = PakitClient(pre_compress_fn=compress_fn)
            result = client.upload_file(str(fpath))

        assert result == "h2"


# ═══════════════════════════════════════════════════════════════════════════
# C11.3 — Merkle proof submission
# ═══════════════════════════════════════════════════════════════════════════

class TestC113MerkleProof:
    """Merkle proof should be submitted from actual CID with finality wait."""

    def test_submit_merkle_proof_exists(self):
        """PakitClient has submit_merkle_proof method."""
        assert hasattr(PakitClient, "submit_merkle_proof")
        assert callable(getattr(PakitClient, "submit_merkle_proof"))

    def test_submit_merkle_proof_returns_none(self):
        """submit_merkle_proof returns None (stub, not yet fully implemented)."""
        client = PakitClient()
        result = client.submit_merkle_proof("deadbeef" * 8)
        assert result is None

    def test_submit_merkle_proof_logs_warning(self, caplog):
        """submit_merkle_proof logs a warning about missing implementation."""
        import logging
        with caplog.at_level(logging.WARNING, logger="storage.pakit_client"):
            client = PakitClient()
            client.submit_merkle_proof("deadbeef" * 8)
        assert any("not yet implemented" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════
# C11.4 — Azure staging fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestC114AzureStagingFallback:
    """When PAKIT_ENABLED=false, uploads should fall back to local path."""

    def test_pakit_enabled_false_upload_falls_back_to_local(self, tmp_path):
        """When pakit_enabled=False, upload_file saves locally and returns local hash."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"model data here")
        fallback_dir = tmp_path / "fallback"

        client = PakitClient(pakit_enabled=False, fallback_dir=str(fallback_dir))
        result = client.upload_file(str(fpath))

        # Should return a hash
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

        # Should have been copied to the fallback directory
        assert fallback_dir.exists()

    def test_pakit_enabled_false_download_from_local(self, tmp_path):
        """When pakit_enabled=False, download retrieves from local fallback."""
        content = b"locally stored model"
        cid = _sha256(content)
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()
        (fallback_dir / cid).write_bytes(content)
        out = str(tmp_path / "retrieved.bin")

        client = PakitClient(pakit_enabled=False, fallback_dir=str(fallback_dir))
        result = client.download_file(cid, out)

        assert result is True
        assert Path(out).read_bytes() == content

    def test_pakit_enabled_true_by_default(self):
        """Default PakitClient has pakit_enabled=True."""
        client = PakitClient()
        assert client.pakit_enabled is True

    def test_from_env_reads_pakit_enabled(self):
        """from_env reads PAKIT_ENABLED env var."""
        with patch.dict(os.environ, {"PAKIT_ENABLED": "false"}):
            client = PakitClient.from_env()
        assert client.pakit_enabled is False

    def test_from_env_reads_fallback_dir(self):
        """from_env reads PAKIT_FALLBACK_DIR env var."""
        with patch.dict(os.environ, {"PAKIT_FALLBACK_DIR": "/tmp/nawal-fallback"}):
            client = PakitClient.from_env()
        assert client.fallback_dir == "/tmp/nawal-fallback"

    def test_fallback_dir_is_logged(self, tmp_path, caplog):
        """When falling back to local, the fallback path is logged."""
        import logging
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"data")
        fallback_dir = tmp_path / "fb"

        with caplog.at_level(logging.INFO, logger="storage.pakit_client"):
            client = PakitClient(pakit_enabled=False, fallback_dir=str(fallback_dir))
            client.upload_file(str(fpath))

        assert any("fallback" in r.message.lower() for r in caplog.records)

    def test_fallback_upload_creates_dir(self, tmp_path):
        """Fallback upload creates the fallback directory if it doesn't exist."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"data")
        fallback_dir = tmp_path / "new_dir" / "nested"

        client = PakitClient(pakit_enabled=False, fallback_dir=str(fallback_dir))
        client.upload_file(str(fpath))

        assert fallback_dir.exists()

    def test_fallback_no_hardcoded_credentials(self):
        """PakitClient source code has no hardcoded Azure credentials."""
        import inspect
        source = inspect.getsource(PakitClient)
        # No Azure connection string or SAS token patterns
        assert "DefaultEndpointsProtocol" not in source
        assert "AccountKey=" not in source
        assert "SharedAccessSignature=" not in source
        assert "sv=20" not in source  # SAS token version pattern

    def test_fallback_download_not_found(self, tmp_path):
        """Fallback download returns False for missing CID."""
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()

        client = PakitClient(pakit_enabled=False, fallback_dir=str(fallback_dir))
        result = client.download_file("deadbeef" * 8, str(tmp_path / "out.bin"))

        assert result is False

    def test_mock_upload_still_works(self):
        """_mock_upload remains functional for testing."""
        client = PakitClient()
        h = client._mock_upload("/some/path")
        assert isinstance(h, str) and len(h) == 64

    def test_azure_blob_fallback_upload(self, tmp_path):
        """When pakit disabled and Azure configured, upload goes to Azure Blob."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"azure model")

        with patch.object(PakitClient, "_azure_blob_upload", return_value=True) as mock_az:
            client = PakitClient(
                pakit_enabled=False,
                azure_blob_connection_string="conn",
                azure_blob_container="nawal",
            )
            result = client.upload_file(str(fpath))

        assert isinstance(result, str) and len(result) == 64
        mock_az.assert_called_once()

    def test_azure_blob_fallback_download(self, tmp_path):
        """When pakit disabled and Azure configured, download tries Azure Blob."""
        cid = "deadbeef" * 8
        out = str(tmp_path / "out.bin")

        with patch.object(PakitClient, "_azure_blob_download", return_value=True) as mock_az:
            client = PakitClient(
                pakit_enabled=False,
                azure_blob_connection_string="conn",
                azure_blob_container="nawal",
            )
            result = client.download_file(cid, out)

        assert result is True
        mock_az.assert_called_once()

    def test_azure_blob_falls_back_to_local(self, tmp_path):
        """When Azure Blob upload fails, falls back to local dir."""
        fpath = tmp_path / "model.bin"
        fpath.write_bytes(b"data")
        fallback_dir = tmp_path / "local_fallback"

        with patch.object(PakitClient, "_azure_blob_upload", return_value=False):
            client = PakitClient(
                pakit_enabled=False,
                azure_blob_connection_string="conn",
                azure_blob_container="c",
                fallback_dir=str(fallback_dir),
            )
            result = client.upload_file(str(fpath))

        assert isinstance(result, str) and len(result) == 64
        assert fallback_dir.exists()

    def test_from_env_reads_azure_config(self):
        """from_env reads AZURE_BLOB_CONNECTION_STRING and AZURE_BLOB_CONTAINER."""
        with patch.dict(os.environ, {
            "AZURE_BLOB_CONNECTION_STRING": "connstr",
            "AZURE_BLOB_CONTAINER": "mycontainer",
        }):
            client = PakitClient.from_env()
        assert client.azure_blob_connection_string == "connstr"
        assert client.azure_blob_container == "mycontainer"
