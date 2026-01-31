"""
Substrate Client for BelizeChain Integration.

Provides low-level connection to BelizeChain via Substrate RPC.

Key Features:
- WebSocket/HTTP RPC connection
- Query chain state (storage, runtime calls)
- Submit extrinsics (signed transactions)
- Event listening and filtering
- Block monitoring

References:
- Substrate Interface: https://github.com/polkascan/py-substrate-interface
- BelizeChain RPC: ws://localhost:9944 (dev)

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import time

from loguru import logger

# Optional substrate-interface library
try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logger.warning(
        "substrate-interface not available. "
        "Install: pip install substrate-interface"
    )


class NetworkType(Enum):
    """BelizeChain network types."""
    LOCAL = "local"
    TESTNET = "testnet"
    MAINNET = "mainnet"


@dataclass
class ChainConfig:
    """
    Configuration for BelizeChain connection.
    
    Attributes:
        network: Network type (local/testnet/mainnet)
        rpc_url: WebSocket RPC endpoint
        type_registry: Custom type definitions
        ss58_format: Address format (42 for Substrate default)
    """
    network: NetworkType = NetworkType.LOCAL
    rpc_url: str = "ws://127.0.0.1:9944"
    type_registry: Optional[Dict] = None
    ss58_format: int = 42
    
    @classmethod
    def local(cls) -> "ChainConfig":
        """Create config for local development."""
        return cls(
            network=NetworkType.LOCAL,
            rpc_url="ws://127.0.0.1:9944",
        )
    
    @classmethod
    def testnet(cls) -> "ChainConfig":
        """Create config for testnet."""
        return cls(
            network=NetworkType.TESTNET,
            rpc_url="wss://testnet-rpc.belizechain.io",
        )
    
    @classmethod
    def mainnet(cls) -> "ChainConfig":
        """Create config for mainnet."""
        return cls(
            network=NetworkType.MAINNET,
            rpc_url="wss://rpc.belizechain.io",
        )


@dataclass
class ExtrinsicReceipt:
    """
    Receipt from submitted extrinsic.
    
    Attributes:
        extrinsic_hash: Hash of the extrinsic
        block_hash: Hash of block containing extrinsic
        block_number: Block number
        success: Whether extrinsic succeeded
        events: List of events emitted
        error: Error message if failed
    """
    extrinsic_hash: str
    block_hash: Optional[str] = None
    block_number: Optional[int] = None
    success: bool = False
    events: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


class SubstrateClient:
    """
    Client for interacting with BelizeChain.
    
    Provides high-level interface for:
    - Chain state queries
    - Extrinsic submission
    - Event monitoring
    - Block tracking
    
    Usage:
        # Connect to local node
        client = SubstrateClient(ChainConfig.local())
        
        # Query storage
        value = client.query_storage(
            module="System",
            storage_function="Account",
            params=["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"],
        )
        
        # Submit extrinsic
        keypair = Keypair.create_from_uri("//Alice")
        receipt = client.submit_extrinsic(
            keypair=keypair,
            call_module="Staking",
            call_function="submit_fitness",
            call_params={"score": 95},
        )
    """
    
    def __init__(self, config: ChainConfig):
        """
        Initialize Substrate client.
        
        Args:
            config: Chain configuration
        """
        if not SUBSTRATE_AVAILABLE:
            raise RuntimeError(
                "substrate-interface required. "
                "Install: pip install substrate-interface"
            )
        
        self.config = config
        self.substrate: Optional[SubstrateInterface] = None
        self._connected = False
        
        logger.info(f"SubstrateClient initialized: {config.network.value}")
    
    def connect(self) -> None:
        """Connect to BelizeChain RPC."""
        try:
            self.substrate = SubstrateInterface(
                url=self.config.rpc_url,
                ss58_format=self.config.ss58_format,
                type_registry=self.config.type_registry or {},
            )
            
            # Test connection
            chain = self.substrate.chain
            runtime_version = self.substrate.runtime_version
            
            self._connected = True
            logger.success(
                f"Connected to {chain} "
                f"(runtime: {runtime_version})"
            )
            
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to BelizeChain: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from BelizeChain RPC."""
        if self.substrate:
            self.substrate.close()
            self._connected = False
            logger.info("Disconnected from BelizeChain")
    
    def is_connected(self) -> bool:
        """Check if connected to chain."""
        return self._connected and self.substrate is not None
    
    def query_storage(
        self,
        module: str,
        storage_function: str,
        params: Optional[List[Any]] = None,
        block_hash: Optional[str] = None,
    ) -> Any:
        """
        Query chain storage.
        
        Args:
            module: Pallet name (e.g., "System", "Staking")
            storage_function: Storage item name
            params: Optional storage key parameters
            block_hash: Optional block hash (None = latest)
        
        Returns:
            Storage value
        """
        if not self.is_connected():
            self.connect()
        
        try:
            result = self.substrate.query(
                module=module,
                storage_function=storage_function,
                params=params or [],
                block_hash=block_hash,
            )
            
            logger.debug(
                f"Queried {module}.{storage_function}: "
                f"{str(result)[:100]}"
            )
            
            return result.value if hasattr(result, 'value') else result
            
        except SubstrateRequestException as e:
            logger.error(f"Storage query failed: {e}")
            raise
    
    def query_map(
        self,
        module: str,
        storage_function: str,
        block_hash: Optional[str] = None,
    ) -> List[tuple]:
        """
        Query all entries in a storage map.
        
        Args:
            module: Pallet name
            storage_function: Storage map name
            block_hash: Optional block hash
        
        Returns:
            List of (key, value) tuples
        """
        if not self.is_connected():
            self.connect()
        
        try:
            result = self.substrate.query_map(
                module=module,
                storage_function=storage_function,
                block_hash=block_hash,
            )
            
            entries = [(k.value, v.value) for k, v in result]
            logger.debug(f"Queried {module}.{storage_function}: {len(entries)} entries")
            
            return entries
            
        except SubstrateRequestException as e:
            logger.error(f"Storage map query failed: {e}")
            raise
    
    def submit_extrinsic(
        self,
        keypair: "Keypair",
        call_module: str,
        call_function: str,
        call_params: Dict[str, Any],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicReceipt:
        """
        Submit signed extrinsic to chain.
        
        Args:
            keypair: Signing keypair
            call_module: Pallet name
            call_function: Extrinsic function name
            call_params: Extrinsic parameters
            wait_for_inclusion: Wait for block inclusion
            wait_for_finalization: Wait for block finalization
        
        Returns:
            Extrinsic receipt
        """
        if not self.is_connected():
            self.connect()
        
        try:
            # Create call
            call = self.substrate.compose_call(
                call_module=call_module,
                call_function=call_function,
                call_params=call_params,
            )
            
            # Create signed extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=keypair,
            )
            
            # Submit extrinsic
            receipt = ExtrinsicReceipt(
                extrinsic_hash=extrinsic.extrinsic_hash,
            )
            
            logger.info(
                f"Submitting extrinsic: {call_module}.{call_function} "
                f"from {keypair.ss58_address}"
            )
            
            result = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            
            # Update receipt
            if wait_for_inclusion or wait_for_finalization:
                receipt.block_hash = result.block_hash
                receipt.success = result.is_success
                receipt.error = result.error_message
                
                # Get block number
                block = self.substrate.get_block(block_hash=result.block_hash)
                receipt.block_number = block['header']['number']
                
                # Get events
                receipt.events = self._get_extrinsic_events(result)
                
                if receipt.success:
                    logger.success(
                        f"Extrinsic included in block #{receipt.block_number}"
                    )
                else:
                    logger.error(f"Extrinsic failed: {receipt.error}")
            
            return receipt
            
        except SubstrateRequestException as e:
            logger.error(f"Extrinsic submission failed: {e}")
            raise
    
    def _get_extrinsic_events(self, result) -> List[Dict]:
        """Extract events from extrinsic result."""
        events = []
        
        if hasattr(result, 'triggered_events'):
            for event in result.triggered_events:
                events.append({
                    'module': event.value['module_id'],
                    'event': event.value['event_id'],
                    'attributes': event.value.get('attributes', {}),
                })
        
        return events
    
    def get_block(self, block_hash: Optional[str] = None) -> Dict:
        """
        Get block information.
        
        Args:
            block_hash: Block hash (None = latest)
        
        Returns:
            Block data
        """
        if not self.is_connected():
            self.connect()
        
        return self.substrate.get_block(block_hash=block_hash)
    
    def get_block_number(self, block_hash: Optional[str] = None) -> int:
        """
        Get block number.
        
        Args:
            block_hash: Block hash (None = latest)
        
        Returns:
            Block number
        """
        block = self.get_block(block_hash=block_hash)
        return block['header']['number']
    
    def get_events(self, block_hash: Optional[str] = None) -> List[Dict]:
        """
        Get events from block.
        
        Args:
            block_hash: Block hash (None = latest)
        
        Returns:
            List of events
        """
        if not self.is_connected():
            self.connect()
        
        events = self.substrate.get_events(block_hash=block_hash)
        
        return [
            {
                'module': event.value['module_id'],
                'event': event.value['event_id'],
                'attributes': event.value.get('attributes', {}),
                'block_hash': block_hash,
            }
            for event in events
        ]
    
    def subscribe_events(
        self,
        callback: Callable[[Dict], None],
        module_filter: Optional[str] = None,
        event_filter: Optional[str] = None,
    ) -> None:
        """
        Subscribe to chain events.
        
        Args:
            callback: Function to call for each event
            module_filter: Optional module name filter
            event_filter: Optional event name filter
        """
        if not self.is_connected():
            self.connect()
        
        logger.info(
            f"Subscribing to events: "
            f"module={module_filter}, event={event_filter}"
        )
        
        def event_handler(obj, update_nr, subscription_id):
            """Handle incoming events."""
            block_hash = obj['header']['parentHash']
            events = self.get_events(block_hash=block_hash)
            
            for event in events:
                # Apply filters
                if module_filter and event['module'] != module_filter:
                    continue
                if event_filter and event['event'] != event_filter:
                    continue
                
                # Call user callback
                callback(event)
        
        # Subscribe to new block headers
        self.substrate.subscribe_block_headers(event_handler)
    
    def get_runtime_constant(
        self,
        module: str,
        constant_name: str,
    ) -> Any:
        """
        Get runtime constant value.
        
        Args:
            module: Pallet name
            constant_name: Constant name
        
        Returns:
            Constant value
        """
        if not self.is_connected():
            self.connect()
        
        result = self.substrate.get_constant(
            module_name=module,
            constant_name=constant_name,
        )
        
        return result.value if hasattr(result, 'value') else result
    
    def get_metadata(self) -> Dict:
        """Get runtime metadata."""
        if not self.is_connected():
            self.connect()
        
        return self.substrate.get_metadata()
    
    def get_account_info(self, address: str) -> Dict:
        """
        Get account information.
        
        Args:
            address: Account address (SS58 format)
        
        Returns:
            Account data (nonce, balance, etc.)
        """
        return self.query_storage(
            module="System",
            storage_function="Account",
            params=[address],
        )
    
    def get_balance(self, address: str) -> int:
        """
        Get account free balance.
        
        Args:
            address: Account address
        
        Returns:
            Free balance in plancks
        """
        account_info = self.get_account_info(address)
        return account_info['data']['free']
    
    @staticmethod
    def create_keypair(
        mnemonic: Optional[str] = None,
        seed: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> "Keypair":
        """
        Create keypair for signing.
        
        Args:
            mnemonic: BIP39 mnemonic phrase
            seed: Hex seed
            uri: URI (e.g., "//Alice")
        
        Returns:
            Keypair
        """
        if uri:
            return Keypair.create_from_uri(uri)
        elif mnemonic:
            return Keypair.create_from_mnemonic(mnemonic)
        elif seed:
            return Keypair.create_from_seed(seed)
        else:
            return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
