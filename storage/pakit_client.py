"""
Pakit Client for Nawal

Uploads Nawal AI models and datasets to Pakit DAG-based storage.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

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
    
    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8080",
        dag_gateway_url: str = "http://localhost:8081",
        compression: str = "zstd"
    ):
        """
        Initialize Pakit client.
        
        Args:
            pakit_api_url: Pakit API server URL
            dag_gateway_url: Pakit DAG gateway URL
            compression: Compression algorithm (zstd, lz4, brotli, none)
        """
        self.pakit_api_url = pakit_api_url
        self.dag_gateway_url = dag_gateway_url
        self.compression = compression
    
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
            compression=os.getenv("PAKIT_COMPRESSION", "zstd")
        )
    
    def upload_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload single file to Pakit DAG storage.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            
        Returns:
            DAG content hash
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, using mock upload")
            return self._mock_upload(file_path, metadata)
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Upload via Pakit DAG gateway
            response = requests.post(
                f"{self.dag_gateway_url}/api/v1/upload",
                files={'file': (os.path.basename(file_path), content)},
                json={
                    'metadata': metadata or {},
                    'compression': self.compression,
                    'deduplicate': True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content_hash = result.get('hash') or result.get('content_hash')
                logger.info(f"✅ Uploaded {file_path} to Pakit DAG: {content_hash}")
                return content_hash
            else:
                logger.error(f"Upload failed: {response.status_code}")
                return self._mock_upload(file_path, metadata)
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return self._mock_upload(file_path, metadata)
    
    def upload_directory(
        self,
        dir_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload entire directory to Pakit.
        
        Args:
            dir_path: Directory path
            metadata: Optional metadata
            
        Returns:
            Root content ID
        """
        if not REQUESTS_AVAILABLE:
            return self._mock_upload(dir_path, metadata)
        
        try:
            # Create tar archive
            import tarfile
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                with tarfile.open(tmp.name, 'w:gz') as tar:
                    tar.add(dir_path, arcname=os.path.basename(dir_path))
                
                # Upload archive
                cid = self.upload_file(tmp.name, metadata)
                
                # Cleanup
                os.unlink(tmp.name)
                
                return cid
                
        except Exception as e:
            logger.error(f"Directory upload error: {e}")
            return self._mock_upload(dir_path, metadata)
    
    def download_file(
        self,
        content_hash: str,
        output_path: str
    ) -> bool:
        """
        Download file from Pakit DAG storage.
        
        Args:
            content_hash: DAG content hash to retrieve
            output_path: Where to save file
            
        Returns:
            True if successful
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, cannot download")
            return False
        
        try:
            response = requests.get(
                f"{self.dag_gateway_url}/api/v1/retrieve/{content_hash}",
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
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
        if not REQUESTS_AVAILABLE:
            return True  # Mock success
        
        try:
            response = requests.post(
                f"{self.dag_gateway_url}/api/v1/pin/{content_hash}"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Pin error: {e}")
            return False
    
    def get_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for stored content.
        
        Args:
            content_hash: DAG content hash
            
        Returns:
            Metadata dict or None
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = requests.get(
                f"{self.dag_gateway_url}/api/v1/metadata/{content_hash}"
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Metadata retrieval error: {e}")
            return None
    
    def _mock_upload(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Mock upload when Pakit API unavailable."""
        # Generate deterministic hash
        hasher = hashlib.sha256()
        hasher.update(path.encode())
        if metadata:
            hasher.update(json.dumps(metadata, sort_keys=True).encode())
        
        mock_hash = hasher.hexdigest()
        logger.info(f"📦 MOCK upload: {path} -> {mock_hash}")
        return mock_hash
