"""
Genome Registry for BelizeChain.

Stores evolved AI model genomes on-chain with decentralized
storage backends (IPFS, Arweave).

Key Features:
- On-chain genome metadata
- IPFS/Arweave content storage
- Genome versioning
- Provenance tracking
- Reproduction history

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import json

from loguru import logger

from .substrate_client import SubstrateClient, ExtrinsicReceipt


class StorageBackend(Enum):
    """Decentralized storage backends."""
    IPFS = "ipfs"
    ARWEAVE = "arweave"
    LOCAL = "local"  # For testing


@dataclass
class GenomeMetadata:
    """
    Metadata for stored genome.
    
    Attributes:
        genome_id: Unique genome identifier (hash)
        owner: Creator SS58 address
        generation: Generation number
        fitness: Best fitness score achieved
        storage_backend: Where genome data is stored
        content_hash: IPFS CID or Arweave TX ID
        parent_ids: Parent genome IDs (for crossover)
        created_at: Creation timestamp
        size_bytes: Genome size in bytes
    """
    genome_id: str
    owner: str
    generation: int
    fitness: float
    storage_backend: StorageBackend
    content_hash: str
    parent_ids: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for chain storage."""
        return {
            'genome_id': self.genome_id,
            'owner': self.owner,
            'generation': self.generation,
            'fitness': int(self.fitness * 100),  # Basis points
            'storage_backend': self.storage_backend.value,
            'content_hash': self.content_hash,
            'parent_ids': self.parent_ids,
            'timestamp': int(self.created_at.timestamp()),
            'size_bytes': self.size_bytes,
        }


class GenomeRegistry:
    """
    Registry for evolved AI genomes.
    
    Manages:
    - On-chain genome metadata
    - Decentralized content storage (IPFS/Arweave)
    - Genome versioning
    - Provenance tracking
    
    Usage:
        registry = GenomeRegistry(client)
        
        # Store genome
        genome_data = {...}  # Genome dictionary
        metadata = registry.store_genome(
            keypair=keypair,
            genome=genome_data,
            fitness=95.0,
            generation=10,
            parent_ids=["parent1_id", "parent2_id"],
        )
        
        # Retrieve genome
        genome = registry.get_genome(metadata.genome_id)
    """
    
    def __init__(
        self,
        client: SubstrateClient,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
        ipfs_url: str = "http://127.0.0.1:5001",
        local_storage_dir: Optional[Path] = None,
    ):
        """
        Initialize GenomeRegistry.
        
        Args:
            client: Substrate client
            storage_backend: Storage backend to use
            ipfs_url: IPFS API endpoint
            local_storage_dir: Directory for local storage
        """
        self.client = client
        self.storage_backend = storage_backend
        self.ipfs_url = ipfs_url
        
        # Local storage directory
        if local_storage_dir:
            self.local_storage_dir = local_storage_dir
        else:
            self.local_storage_dir = Path("./genome_storage")
        
        self.local_storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"GenomeRegistry initialized: backend={storage_backend.value}"
        )
    
    def store_genome(
        self,
        keypair,
        genome: Dict,
        fitness: float,
        generation: int,
        parent_ids: Optional[List[str]] = None,
        wait_for_finalization: bool = False,
    ) -> GenomeMetadata:
        """
        Store genome on-chain and in decentralized storage.
        
        Args:
            keypair: Owner keypair
            genome: Genome data (dictionary)
            fitness: Best fitness achieved
            generation: Generation number
            parent_ids: Parent genome IDs
            wait_for_finalization: Wait for block finalization
        
        Returns:
            Genome metadata
        """
        logger.info(
            f"Storing genome: generation={generation}, "
            f"fitness={fitness:.2f}"
        )
        
        # Serialize genome
        genome_json = json.dumps(genome, sort_keys=True)
        genome_bytes = genome_json.encode('utf-8')
        
        # Generate genome ID (content hash)
        genome_id = hashlib.sha256(genome_bytes).hexdigest()
        
        # Store content in decentralized storage
        if self.storage_backend == StorageBackend.IPFS:
            content_hash = self._store_ipfs(genome_bytes)
        elif self.storage_backend == StorageBackend.ARWEAVE:
            content_hash = self._store_arweave(genome_bytes)
        else:
            # Local storage
            content_hash = self._store_local(genome_id, genome_bytes)
        
        # Create metadata
        metadata = GenomeMetadata(
            genome_id=genome_id,
            owner=keypair.ss58_address,
            generation=generation,
            fitness=fitness,
            storage_backend=self.storage_backend,
            content_hash=content_hash,
            parent_ids=parent_ids or [],
            size_bytes=len(genome_bytes),
        )
        
        # Store metadata on-chain
        receipt = self.client.submit_extrinsic(
            keypair=keypair,
            call_module="GenomeRegistry",
            call_function="register_genome",
            call_params=metadata.to_dict(),
            wait_for_inclusion=True,
            wait_for_finalization=wait_for_finalization,
        )
        
        if receipt.success:
            logger.success(
                f"Genome stored: ID={genome_id[:16]}..., "
                f"hash={content_hash[:16]}..."
            )
        else:
            logger.error(f"Genome storage failed: {receipt.error}")
            raise RuntimeError(f"Failed to store genome: {receipt.error}")
        
        return metadata
    
    def get_genome(self, genome_id: str) -> Optional[Dict]:
        """
        Retrieve genome from storage.
        
        Args:
            genome_id: Genome identifier
        
        Returns:
            Genome data or None if not found
        """
        # Get metadata from chain
        metadata = self.get_metadata(genome_id)
        if metadata is None:
            logger.warning(f"Genome not found: {genome_id}")
            return None
        
        # Retrieve content from storage
        if metadata.storage_backend == StorageBackend.IPFS:
            genome_bytes = self._retrieve_ipfs(metadata.content_hash)
        elif metadata.storage_backend == StorageBackend.ARWEAVE:
            genome_bytes = self._retrieve_arweave(metadata.content_hash)
        else:
            genome_bytes = self._retrieve_local(metadata.content_hash)
        
        # Parse genome
        genome = json.loads(genome_bytes.decode('utf-8'))
        
        logger.debug(f"Retrieved genome: {genome_id[:16]}...")
        return genome
    
    def get_metadata(self, genome_id: str) -> Optional[GenomeMetadata]:
        """
        Get genome metadata from chain.
        
        Args:
            genome_id: Genome identifier
        
        Returns:
            Genome metadata or None
        """
        try:
            data = self.client.query_storage(
                module="GenomeRegistry",
                storage_function="Genomes",
                params=[genome_id],
            )
            
            if data is None:
                return None
            
            metadata = GenomeMetadata(
                genome_id=genome_id,
                owner=data['owner'],
                generation=data['generation'],
                fitness=data['fitness'] / 100.0,
                storage_backend=StorageBackend(data['storage_backend']),
                content_hash=data['content_hash'],
                parent_ids=data.get('parent_ids', []),
                created_at=datetime.fromtimestamp(data['timestamp']),
                size_bytes=data['size_bytes'],
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get genome metadata: {e}")
            return None
    
    def get_lineage(self, genome_id: str, depth: int = 10) -> List[GenomeMetadata]:
        """
        Get genome lineage (ancestors).
        
        Args:
            genome_id: Starting genome ID
            depth: Maximum depth to traverse
        
        Returns:
            List of ancestor genomes (oldest first)
        """
        lineage = []
        current_id = genome_id
        
        for _ in range(depth):
            metadata = self.get_metadata(current_id)
            if metadata is None:
                break
            
            lineage.append(metadata)
            
            # Get first parent
            if not metadata.parent_ids:
                break
            
            current_id = metadata.parent_ids[0]
        
        # Reverse to get oldest first
        lineage.reverse()
        
        logger.debug(f"Retrieved lineage: {len(lineage)} generations")
        return lineage
    
    def get_by_owner(self, owner: str) -> List[GenomeMetadata]:
        """
        Get all genomes owned by address.
        
        Args:
            owner: Owner SS58 address
        
        Returns:
            List of genome metadata
        """
        try:
            # Query all genomes (this would need proper indexing in production)
            genome_ids = self.client.query_storage(
                module="GenomeRegistry",
                storage_function="GenomesByOwner",
                params=[owner],
            )
            
            if genome_ids is None:
                return []
            
            # Get metadata for each genome
            genomes = []
            for genome_id in genome_ids:
                metadata = self.get_metadata(genome_id)
                if metadata:
                    genomes.append(metadata)
            
            logger.debug(f"Found {len(genomes)} genomes for {owner}")
            return genomes
            
        except Exception as e:
            logger.error(f"Failed to get genomes by owner: {e}")
            return []
    
    def _store_ipfs(self, data: bytes) -> str:
        """
        Store data on IPFS.
        
        Args:
            data: Data to store
        
        Returns:
            IPFS CID
        """
        try:
            import requests
            response = requests.post(
                f"{self.ipfs_url}/api/v0/add",
                files={'file': data},
            )
            cid = response.json()['Hash']
            logger.debug(f"Stored on IPFS: {cid}")
            return cid
        except Exception as e:
            logger.error(f"IPFS storage failed: {e}")
            raise RuntimeError("IPFS storage not available")
    
    def _retrieve_ipfs(self, cid: str) -> bytes:
        """Retrieve data from IPFS."""
        try:
            import requests
            response = requests.post(
                f"{self.ipfs_url}/api/v0/cat",
                params={'arg': cid},
            )
            return response.content
        except Exception as e:
            logger.error(f"IPFS retrieval failed: {e}")
            raise
    
    def _store_arweave(self, data: bytes) -> str:
        """
        Store data on Arweave.
        
        Args:
            data: Data to store
        
        Returns:
            Arweave transaction ID
        """
        # Placeholder - would need Arweave client library
        logger.warning("Arweave storage not implemented, using local")
        return self._store_local(hashlib.sha256(data).hexdigest(), data)
    
    def _retrieve_arweave(self, tx_id: str) -> bytes:
        """Retrieve data from Arweave."""
        # Placeholder
        logger.warning("Arweave retrieval not implemented, using local")
        return self._retrieve_local(tx_id)
    
    def _store_local(self, genome_id: str, data: bytes) -> str:
        """
        Store data locally.
        
        Args:
            genome_id: Genome identifier
            data: Data to store
        
        Returns:
            File path (used as content hash)
        """
        file_path = self.local_storage_dir / f"{genome_id}.json"
        file_path.write_bytes(data)
        logger.debug(f"Stored locally: {file_path}")
        return str(file_path)
    
    def _retrieve_local(self, content_hash: str) -> bytes:
        """Retrieve data from local storage."""
        file_path = Path(content_hash)
        if not file_path.exists():
            raise FileNotFoundError(f"Genome not found: {content_hash}")
        return file_path.read_bytes()
