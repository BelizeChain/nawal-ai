"""
BelizeID verification for Nawal federated learning clients
Ensures only KYC-verified citizens can participate in training
"""

import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import hashlib

try:
    from substrateinterface import SubstrateInterface
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logging.warning("py-substrate-interface not installed. Run: pip install substrate-interface")

logger = logging.getLogger(__name__)


class BelizeIDVerifier:
    """
    Verify BelizeID credentials against blockchain pallet-identity
    Implements caching to reduce RPC calls
    """
    
    def __init__(
        self,
        rpc_url: str = "ws://127.0.0.1:9944",
        cache_ttl_seconds: int = 3600  # 1 hour cache
    ):
        if not SUBSTRATE_AVAILABLE:
            raise ImportError("substrate-interface required. Install with: pip install substrate-interface")
        
        self.rpc_url = rpc_url
        self.cache: Dict[str, tuple[bool, datetime]] = {}
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.substrate: Optional[SubstrateInterface] = None
    
    async def connect(self):
        """Connect to BelizeChain node"""
        loop = asyncio.get_event_loop()
        self.substrate = await loop.run_in_executor(
            None,
            lambda: SubstrateInterface(url=self.rpc_url)
        )
        logger.info(f"✅ Connected to BelizeChain: {self.rpc_url}")
    
    async def verify(self, belizeid: str) -> bool:
        """
        Verify BelizeID is registered and KYC-approved
        
        Args:
            belizeid: BelizeID string (format: "BZ-XXXXX-YYYY")
        
        Returns:
            True if valid and KYC-approved, False otherwise
        """
        # Check cache first
        if belizeid in self.cache:
            is_valid, cached_at = self.cache[belizeid]
            if datetime.utcnow() - cached_at < self.cache_ttl:
                return is_valid
        
        # Query blockchain
        try:
            is_valid = await self._query_blockchain(belizeid)
            
            # Update cache
            self.cache[belizeid] = (is_valid, datetime.utcnow())
            
            return is_valid
        
        except Exception as e:
            logger.error(f"BelizeID verification failed: {e}")
            return False
    
    async def _query_blockchain(self, belizeid: str) -> bool:
        """Query pallet-identity for BelizeID registration"""
        if self.substrate is None:
            await self.connect()
        
        loop = asyncio.get_event_loop()
        
        # Convert BelizeID to identity_id (hash of BelizeID string)
        identity_id = self._belizeid_to_identity_id(belizeid)
        
        # Query Identity storage map
        result = await loop.run_in_executor(
            None,
            lambda: self.substrate.query(
                module='Identity',
                storage_function='Identities',
                params=[identity_id]
            )
        )
        
        if result is None:
            logger.debug(f"BelizeID {belizeid} not registered")
            return False
        
        # Check KYC status
        identity_data = result.value
        kyc_approved = identity_data.get('kycApproved', False)
        
        if not kyc_approved:
            logger.debug(f"BelizeID {belizeid} not KYC-approved")
            return False
        
        logger.debug(f"✅ BelizeID {belizeid} verified")
        return True
    
    def _belizeid_to_identity_id(self, belizeid: str) -> int:
        """Convert BelizeID string to numeric identity_id"""
        # Simple hash-based conversion (adjust to match pallet-identity logic)
        hash_bytes = hashlib.sha256(belizeid.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder='big') % (2**64)
    
    async def get_identity_details(self, belizeid: str) -> Optional[Dict]:
        """
        Get full identity details for a BelizeID
        
        Returns:
            Dictionary with identity data or None if not found
        """
        if self.substrate is None:
            await self.connect()
        
        loop = asyncio.get_event_loop()
        identity_id = self._belizeid_to_identity_id(belizeid)
        
        result = await loop.run_in_executor(
            None,
            lambda: self.substrate.query(
                module='Identity',
                storage_function='Identities',
                params=[identity_id]
            )
        )
        
        if result is None:
            return None
        
        return result.value
    
    async def check_rate_limits(self, belizeid: str) -> bool:
        """
        Check if BelizeID has exceeded federated learning rate limits
        
        Returns:
            True if within limits, False if rate-limited
        """
        # Query blockchain for recent training submissions
        # (Implementation depends on pallet-staking PoUW tracking)
        
        # For now, always return True (no rate limiting)
        return True
    
    def clear_cache(self):
        """Clear verification cache (call on cache invalidation events)"""
        self.cache.clear()
        logger.info("BelizeID cache cleared")
    
    async def close(self):
        """Close blockchain connection"""
        if self.substrate:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.substrate.close
            )


class DummyBelizeIDVerifier:
    """
    Dummy verifier for development/testing (always returns True)
    NEVER use in production!
    """
    
    async def connect(self):
        logger.warning("⚠️ Using DummyBelizeIDVerifier - NOT FOR PRODUCTION")
    
    async def verify(self, belizeid: str) -> bool:
        logger.debug(f"Dummy verification: {belizeid} -> APPROVED (dev mode)")
        return True
    
    async def get_identity_details(self, belizeid: str) -> Optional[Dict]:
        return {
            "belizeId": belizeid,
            "kycApproved": True,
            "accountType": "Citizen"
        }
    
    async def check_rate_limits(self, belizeid: str) -> bool:
        return True
    
    async def close(self):
        pass


def create_verifier(mode: str = "production", **kwargs) -> BelizeIDVerifier:
    """
    Factory function to create BelizeID verifier
    
    Args:
        mode: "production" or "development"
        **kwargs: Passed to verifier constructor
    
    Returns:
        BelizeID verifier instance
    """
    if mode == "development":
        return DummyBelizeIDVerifier()
    
    if mode == "production":
        if not SUBSTRATE_AVAILABLE:
            raise RuntimeError("substrate-interface required for production mode")
        return BelizeIDVerifier(**kwargs)
    
    raise ValueError(f"Unknown mode: {mode}")


# Example usage
if __name__ == "__main__":
    async def main():
        verifier = create_verifier(mode="development")
        await verifier.connect()
        
        # Test verification
        result = await verifier.verify("BZ-12345-6789")
        print(f"Verification result: {result}")
        
        await verifier.close()
    
    asyncio.run(main())
