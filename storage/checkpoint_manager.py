"""
Checkpoint Manager

Manages Nawal model checkpoints with Pakit integration.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from nawal.storage.pakit_client import PakitClient

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with Pakit storage.
    
    Features:
    - Local checkpoint saving
    - Pakit upload for redundancy
    - Checkpoint metadata tracking
    - Version management
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        pakit_client: Optional[PakitClient] = None,
        auto_upload: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Local checkpoint directory
            pakit_client: Pakit client for uploads
            auto_upload: Automatically upload to Pakit
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.pakit_client = pakit_client
        self.auto_upload = auto_upload
        
        # Checkpoint registry
        self.registry_path = self.checkpoint_dir / "registry.json"
        self.registry = self._load_registry()
    
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        checkpoint_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint locally and optionally to Pakit.
        
        Args:
            model_state: Model state dict (PyTorch format)
            checkpoint_name: Checkpoint filename
            metadata: Additional metadata
            
        Returns:
            Checkpoint path (or Pakit CID if uploaded)
        """
        # Save locally
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        import torch
        torch.save(model_state, checkpoint_path)
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Record in registry
        checkpoint_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'pakit_cid': None,
        }
        
        # Upload to Pakit if enabled
        if self.auto_upload and self.pakit_client:
            try:
                cid = self.pakit_client.upload_file(
                    str(checkpoint_path),
                    metadata=metadata
                )
                checkpoint_info['pakit_cid'] = cid
                logger.info(f"☁️  Uploaded to Pakit: {cid}")
            except Exception as e:
                logger.warning(f"Pakit upload failed: {e}")
        
        # Update registry
        self.registry[checkpoint_name] = checkpoint_info
        self._save_registry()
        
        return checkpoint_info.get('pakit_cid') or str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_name: str,
        from_pakit: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from local or Pakit.
        
        Args:
            checkpoint_name: Checkpoint to load
            from_pakit: Load from Pakit instead of local
            
        Returns:
            Model state dict or None
        """
        if checkpoint_name not in self.registry:
            logger.error(f"Checkpoint not found: {checkpoint_name}")
            return None
        
        checkpoint_info = self.registry[checkpoint_name]
        
        if from_pakit and checkpoint_info.get('pakit_cid'):
            # Download from Pakit
            if not self.pakit_client:
                logger.error("Pakit client not configured")
                return None
            
            cid = checkpoint_info['pakit_cid']
            temp_path = self.checkpoint_dir / f"temp_{checkpoint_name}"
            
            if self.pakit_client.download_file(cid, str(temp_path)):
                import torch
                state = torch.load(temp_path)
                os.unlink(temp_path)
                return state
            else:
                logger.error(f"Failed to download from Pakit: {cid}")
                return None
        
        else:
            # Load from local
            checkpoint_path = Path(checkpoint_info['path'])
            
            if not checkpoint_path.exists():
                logger.error(f"Local checkpoint missing: {checkpoint_path}")
                return None
            
            import torch
            return torch.load(checkpoint_path)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return list(self.registry.values())
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete checkpoint (local only, Pakit data persists).
        
        Args:
            checkpoint_name: Checkpoint to delete
            
        Returns:
            True if deleted
        """
        if checkpoint_name not in self.registry:
            return False
        
        checkpoint_info = self.registry[checkpoint_name]
        checkpoint_path = Path(checkpoint_info['path'])
        
        if checkpoint_path.exists():
            os.unlink(checkpoint_path)
        
        del self.registry[checkpoint_name]
        self._save_registry()
        
        logger.info(f"🗑️  Deleted checkpoint: {checkpoint_name}")
        return True
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load checkpoint registry."""
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save checkpoint registry."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
