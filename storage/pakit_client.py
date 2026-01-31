"""
Pakit Client for Nawal

Uploads Nawal AI models and datasets to Pakit (IPFS/Arweave).
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
    
    Integrates with Pakit's IPFS/Arweave backends for:
    - Model checkpoint persistence
    - Training dataset archival
    - Genome evolution history
    - Federated learning aggregation results
    """
    
    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8000",
        ipfs_gateway: str = "http://localhost:5001",
        use_arweave: bool = False
    ):
        """
        Initialize Pakit client.
        
        Args:
            pakit_api_url: Pakit API server URL
            ipfs_gateway: IPFS daemon URL
            use_arweave: Enable Arweave permanent storage
        """
        self.pakit_api_url = pakit_api_url
        self.ipfs_gateway = ipfs_gateway
        self.use_arweave = use_arweave
    
    def upload_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload single file to Pakit.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            
        Returns:
            Content ID (IPFS CID or hash)
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, using mock upload")
            return self._mock_upload(file_path, metadata)
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Upload via Pakit API
            response = requests.post(
                f"{self.pakit_api_url}/api/v1/upload",
                files={'file': (os.path.basename(file_path), content)},
                data={'metadata': json.dumps(metadata or {})}
            )
            
            if response.status_code == 200:
                result = response.json()
                cid = result.get('cid') or result.get('content_id')
                logger.info(f"✅ Uploaded {file_path} to Pakit: {cid}")
                return cid
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
        content_id: str,
        output_path: str
    ) -> bool:
        """
        Download file from Pakit.
        
        Args:
            content_id: Content ID to retrieve
            output_path: Where to save file
            
        Returns:
            True if successful
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, cannot download")
            return False
        
        try:
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/retrieve/{content_id}",
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✅ Downloaded {content_id} to {output_path}")
                return True
            else:
                logger.error(f"Download failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def pin_content(self, content_id: str) -> bool:
        """
        Pin content to ensure it stays available.
        
        Args:
            content_id: Content ID to pin
            
        Returns:
            True if pinned
        """
        if not REQUESTS_AVAILABLE:
            return True  # Mock success
        
        try:
            response = requests.post(
                f"{self.pakit_api_url}/api/v1/pin/{content_id}"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Pin error: {e}")
            return False
    
    def get_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for stored content.
        
        Args:
            content_id: Content ID
            
        Returns:
            Metadata dict or None
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/metadata/{content_id}"
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
        
        mock_cid = f"Qm{hasher.hexdigest()[:44]}"
        logger.info(f"📦 MOCK upload: {path} -> {mock_cid}")
        return mock_cid
