"""
Pakit Client for Nawal

Uploads Nawal AI models and datasets to Pakit DAG-based storage.
"""

import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available")


class PakitClient:
    """
    Client for uploading Nawal models to Pakit storage.

    Integrates with Pakit's DAG-based storage engine for:
    - Model checkpoint persistence
    - Training dataset archival
    - Genome evolution history
    - Federated learning aggregation results
    """

    # Regex: 1-128 lowercase hex characters
    _CID_PATTERN = re.compile(r"^[0-9a-f]{1,128}$")

    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8080",
        dag_gateway_url: str = "http://localhost:8081",
        compression: str = "zstd",
        pakit_enabled: bool = True,
        fallback_dir: str | None = None,
        pre_compress_fn: Callable[[bytes], bytes] | None = None,
        azure_blob_connection_string: str | None = None,
        azure_blob_container: str | None = None,
    ):
        """
        Initialize Pakit client.

        Args:
            pakit_api_url: Pakit API server URL
            dag_gateway_url: Pakit DAG gateway URL
            compression: Compression algorithm (zstd, lz4, brotli, none)
            pakit_enabled: When False, use local fallback instead of Pakit
            fallback_dir: Local directory for fallback storage
            pre_compress_fn: Optional hook applied to content before upload
            azure_blob_connection_string: Azure Blob connection string for cloud fallback
            azure_blob_container: Azure Blob container name for cloud fallback
        """
        self.pakit_api_url = pakit_api_url
        self.dag_gateway_url = dag_gateway_url
        self.compression = compression
        self.pakit_enabled = pakit_enabled
        self.fallback_dir = fallback_dir
        self.pre_compress_fn = pre_compress_fn
        self.azure_blob_connection_string = azure_blob_connection_string
        self.azure_blob_container = azure_blob_container

    @classmethod
    def from_env(cls) -> "PakitClient":
        """
        Create PakitClient from environment variables.

        Reads:
            PAKIT_API_URL: Pakit API endpoint (default: http://localhost:8080)
            PAKIT_DAG_GATEWAY_URL: DAG gateway endpoint (default: http://localhost:8081)
            PAKIT_COMPRESSION: Compression algorithm (default: zstd)

        Returns:
            PakitClient instance configured from environment
        """
        return cls(
            pakit_api_url=os.getenv("PAKIT_API_URL", "http://localhost:8080"),
            dag_gateway_url=os.getenv("PAKIT_DAG_GATEWAY_URL", "http://localhost:8081"),
            compression=os.getenv("PAKIT_COMPRESSION", "zstd"),
            pakit_enabled=os.getenv("PAKIT_ENABLED", "true").lower() in ("true", "1", "yes"),
            fallback_dir=os.getenv("PAKIT_FALLBACK_DIR"),
            azure_blob_connection_string=os.getenv("AZURE_BLOB_CONNECTION_STRING"),
            azure_blob_container=os.getenv("AZURE_BLOB_CONTAINER"),
        )

    @staticmethod
    def _validate_cid(content_hash: str) -> bool:
        """Validate that content_hash is a plausible hex CID."""
        return bool(content_hash and PakitClient._CID_PATTERN.match(content_hash))

    def upload_file(self, file_path: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Upload single file to Pakit DAG storage.

        Args:
            file_path: Path to file
            metadata: Optional metadata

        Returns:
            DAG content hash
        """
        if not self.pakit_enabled:
            return self._fallback_upload(file_path, metadata)

        if not REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests library required for Pakit uploads. Install: pip install requests"
            )

        try:
            # Read file
            with open(file_path, "rb") as f:
                content = f.read()

            # Apply pre-compression hook if configured
            if self.pre_compress_fn is not None:
                try:
                    content = self.pre_compress_fn(content)
                except Exception as e:
                    logger.warning(f"Pre-compression hook failed, using raw content: {e}")

            # Upload via Pakit DAG gateway
            # Note: cannot mix files= and json= in requests; send metadata as form data
            upload_metadata = json.dumps(
                {
                    "metadata": metadata or {},
                    "compression": self.compression,
                    "deduplicate": True,
                }
            )
            response = requests.post(
                f"{self.dag_gateway_url}/api/v1/upload",
                files={
                    "file": (os.path.basename(file_path), content),
                    "options": (None, upload_metadata, "application/json"),
                },
            )

            if response.status_code == 200:
                result = response.json()
                content_hash = result.get("hash") or result.get("content_hash")
                logger.info(f"✅ Uploaded {file_path} to Pakit DAG: {content_hash}")
                return content_hash
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Pakit upload failed with status {response.status_code}")

        except requests.RequestException as e:
            logger.error(f"Upload network error: {e}")
            raise RuntimeError(f"Pakit upload failed: {e}") from e

    def upload_directory(self, dir_path: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Upload entire directory to Pakit.

        Args:
            dir_path: Directory path
            metadata: Optional metadata

        Returns:
            Root content ID
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests library required for Pakit uploads. Install: pip install requests"
            )

        try:
            # Create tar archive
            import tarfile
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                with tarfile.open(tmp.name, "w:gz") as tar:
                    tar.add(dir_path, arcname=os.path.basename(dir_path))

                # Upload archive
                cid = self.upload_file(tmp.name, metadata)

                # Cleanup
                os.unlink(tmp.name)

                return cid

        except Exception as e:
            logger.error(f"Directory upload error: {e}")
            raise

    def download_file(self, content_hash: str, output_path: str) -> bool:
        """
        Download file from Pakit DAG storage.

        Args:
            content_hash: DAG content hash to retrieve
            output_path: Where to save file

        Returns:
            True if successful
        """
        if not self._validate_cid(content_hash):
            logger.error(f"Invalid CID format: {content_hash!r}")
            return False

        if not self.pakit_enabled:
            return self._fallback_download(content_hash, output_path)

        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, cannot download")
            return False

        try:
            response = requests.get(
                f"{self.dag_gateway_url}/api/v1/retrieve/{content_hash}", stream=True
            )

            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify content integrity
                actual_hash = self._compute_file_hash(output_path)
                if actual_hash != content_hash:
                    logger.error(
                        f"Content integrity check FAILED: "
                        f"expected {content_hash[:16]}..., got {actual_hash[:16]}..."
                    )
                    os.unlink(output_path)
                    return False

                logger.info(f"✅ Downloaded {content_hash} to {output_path}")
                return True
            else:
                logger.error(f"Download failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def pin_content(self, content_hash: str) -> bool:
        """
        Pin content in DAG to ensure it stays available.

        Args:
            content_hash: DAG content hash to pin

        Returns:
            True if pinned
        """
        if not self._validate_cid(content_hash):
            logger.error(f"Invalid CID format for pin: {content_hash!r}")
            return False

        if not REQUESTS_AVAILABLE:
            return True  # Mock success

        try:
            response = requests.post(f"{self.dag_gateway_url}/api/v1/pin/{content_hash}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Pin error: {e}")
            return False

    def get_metadata(self, content_hash: str) -> dict[str, Any] | None:
        """
        Get metadata for stored content.

        Args:
            content_hash: DAG content hash

        Returns:
            Metadata dict or None
        """
        if not self._validate_cid(content_hash):
            logger.error(f"Invalid CID format for metadata: {content_hash!r}")
            return None

        if not REQUESTS_AVAILABLE:
            return None

        try:
            response = requests.get(f"{self.dag_gateway_url}/api/v1/metadata/{content_hash}")

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Metadata retrieval error: {e}")
            return None

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def submit_merkle_proof(self, content_hash: str) -> str | None:
        """Submit a Merkle proof for the given content hash.

        Placeholder for future on-chain proof submission.

        Args:
            content_hash: DAG content hash to prove

        Returns:
            Transaction hash if submitted, None otherwise
        """
        logger.warning(
            "submit_merkle_proof is not yet implemented — "
            "on-chain proof submission requires ledger integration"
        )
        return None

    def _fallback_upload(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store file when Pakit is disabled — tries Azure Blob, then local."""
        content_hash = self._compute_file_hash(file_path)

        # Try Azure Blob if configured
        if self.azure_blob_connection_string and self.azure_blob_container:
            if self._azure_blob_upload(file_path, content_hash):
                return content_hash

        # Fall back to local directory
        fallback = Path(self.fallback_dir or "./storage_fallback")
        fallback.mkdir(parents=True, exist_ok=True)
        dest = fallback / content_hash
        shutil.copy2(file_path, dest)

        logger.info(f"📦 Fallback upload: {file_path} -> {dest} (hash={content_hash})")
        return content_hash

    def _fallback_download(self, content_hash: str, output_path: str) -> bool:
        """Retrieve file from fallback — tries Azure Blob, then local."""
        # Try Azure Blob if configured
        if self.azure_blob_connection_string and self.azure_blob_container:
            if self._azure_blob_download(content_hash, output_path):
                return True

        # Fall back to local directory
        fallback = Path(self.fallback_dir or "./storage_fallback")
        src = fallback / content_hash
        if not src.exists():
            logger.error(f"Fallback file not found: {src}")
            return False
        shutil.copy2(str(src), output_path)
        logger.info(f"📦 Fallback download: {content_hash} -> {output_path}")
        return True

    def _azure_blob_upload(self, file_path: str, content_hash: str) -> bool:
        """Upload file to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            logger.warning("azure-storage-blob not installed, skipping Azure fallback")
            return False

        try:
            blob_service = BlobServiceClient.from_connection_string(
                self.azure_blob_connection_string
            )
            container = blob_service.get_container_client(self.azure_blob_container)
            with open(file_path, "rb") as f:
                container.upload_blob(name=content_hash, data=f, overwrite=True)
            logger.info(f"☁️ Azure Blob upload: {content_hash}")
            return True
        except Exception as e:
            logger.warning(f"Azure Blob upload failed: {e}")
            return False

    def _azure_blob_download(self, content_hash: str, output_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            logger.warning("azure-storage-blob not installed, skipping Azure fallback")
            return False

        try:
            blob_service = BlobServiceClient.from_connection_string(
                self.azure_blob_connection_string
            )
            container = blob_service.get_container_client(self.azure_blob_container)
            blob_data = container.download_blob(content_hash).readall()
            with open(output_path, "wb") as f:
                f.write(blob_data)
            logger.info(f"☁️ Azure Blob download: {content_hash} -> {output_path}")
            return True
        except Exception as e:
            logger.warning(f"Azure Blob download failed: {e}")
            return False

    def _mock_upload(self, path: str, metadata: dict[str, Any] | None = None) -> str:
        """Mock upload when Pakit API unavailable."""
        # Generate deterministic hash
        hasher = hashlib.sha256()
        hasher.update(path.encode())
        if metadata:
            hasher.update(json.dumps(metadata, sort_keys=True).encode())

        mock_hash = hasher.hexdigest()
        logger.info(f"📦 MOCK upload: {path} -> {mock_hash}")
        return mock_hash
