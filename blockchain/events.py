"""
Blockchain Event Listener

Listens for BelizeChain events related to AI training:
- New training rounds announced
- Validator enrollments
- Reward distributions
- Slashing events

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from loguru import logger

try:
    from substrateinterface import SubstrateInterface
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False


# =============================================================================
# Event Types
# =============================================================================


class EventType(Enum):
    """Types of blockchain events."""
    
    # Training events
    TRAINING_ROUND_STARTED = "training_round_started"
    TRAINING_PROOF_SUBMITTED = "training_proof_submitted"
    TRAINING_ROUND_COMPLETED = "training_round_completed"
    
    # Enrollment events
    TRAINER_ENROLLED = "trainer_enrolled"
    TRAINER_UNENROLLED = "trainer_unenrolled"
    
    # Reward events
    REWARDS_CALCULATED = "rewards_calculated"
    REWARDS_CLAIMED = "rewards_claimed"
    
    # Penalty events
    TRAINER_SLASHED = "trainer_slashed"
    TRAINER_REPUTATION_UPDATED = "trainer_reputation_updated"
    
    # Genome events
    GENOME_DEPLOYED = "genome_deployed"
    GENOME_EVOLVED = "genome_evolved"


@dataclass
class TrainingEvent:
    """Training-related blockchain event."""
    
    event_type: EventType
    block_number: int
    block_hash: str
    timestamp: str
    data: dict[str, Any]
    
    def __str__(self) -> str:
        return (
            f"{self.event_type.value} at block {self.block_number} "
            f"({self.timestamp}): {self.data}"
        )


# =============================================================================
# Event Handler Protocol
# =============================================================================


EventHandler = Callable[[TrainingEvent], Awaitable[None]]


# =============================================================================
# Blockchain Event Listener
# =============================================================================


class BlockchainEventListener:
    """
    Listen for blockchain events related to AI training.
    
    Provides async event stream and callback registration for
    monitoring training activities on-chain.
    """
    
    def __init__(
        self,
        node_url: str = "ws://127.0.0.1:9944",
        mock_mode: bool = False,
    ):
        """
        Initialize event listener.
        
        Args:
            node_url: WebSocket URL of BelizeChain node
            mock_mode: Use mock mode for testing
        """
        self.node_url = node_url
        self.mock_mode = mock_mode or not SUBSTRATE_AVAILABLE
        self.substrate: SubstrateInterface | None = None
        self.is_listening = False
        
        # Event handlers by type
        self.handlers: dict[EventType, list[EventHandler]] = {
            event_type: [] for event_type in EventType
        }
        
        # Event history (for mock mode and debugging)
        self.event_history: list[TrainingEvent] = []
        self.max_history_size = 1000
        
        logger.info(
            "Initialized BlockchainEventListener",
            node_url=node_url,
            mock_mode=self.mock_mode,
        )
    
    async def connect(self) -> bool:
        """
        Connect to blockchain node.
        
        Returns:
            True if connected successfully
        """
        if self.mock_mode:
            logger.info("Running in mock mode, skipping blockchain connection")
            return True
        
        try:
            self.substrate = SubstrateInterface(url=self.node_url)
            logger.info("Connected to BelizeChain for event listening")
            return True
        except Exception as e:
            logger.error("Failed to connect for events", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from blockchain node."""
        self.is_listening = False
        
        if self.substrate:
            self.substrate.close()
            self.substrate = None
        
        logger.info("Disconnected from BelizeChain")
    
    def register_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Register event handler callback.
        
        Args:
            event_type: Type of event to handle
            handler: Async callback function
        """
        self.handlers[event_type].append(handler)
        logger.debug(
            "Registered event handler",
            event_type=event_type.value,
            total_handlers=len(self.handlers[event_type]),
        )
    
    def unregister_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Unregister event handler callback."""
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            logger.debug(
                "Unregistered event handler",
                event_type=event_type.value,
            )
    
    async def _dispatch_event(self, event: TrainingEvent) -> None:
        """
        Dispatch event to registered handlers.
        
        Args:
            event: Event to dispatch
        """
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Call handlers
        handlers = self.handlers.get(event.event_type, [])
        if handlers:
            logger.debug(
                "Dispatching event",
                event_type=event.event_type.value,
                handlers=len(handlers),
            )
            
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(
                        "Handler error",
                        event_type=event.event_type.value,
                        error=str(e),
                    )
    
    async def _parse_event(
        self,
        block_number: int,
        block_hash: str,
        event_record: Any,
    ) -> TrainingEvent | None:
        """
        Parse blockchain event record.
        
        Args:
            block_number: Block number
            block_hash: Block hash
            event_record: Raw event record from substrate
        
        Returns:
            TrainingEvent or None if not relevant
        """
        module_id = event_record.value['module_id']
        event_id = event_record.value['event_id']
        attributes = event_record.value.get('attributes', {})
        
        # Map blockchain events to our event types
        event_map = {
            ('Staking', 'TrainingRoundStarted'): EventType.TRAINING_ROUND_STARTED,
            ('Staking', 'TrainingProofSubmitted'): EventType.TRAINING_PROOF_SUBMITTED,
            ('Staking', 'TrainingRoundCompleted'): EventType.TRAINING_ROUND_COMPLETED,
            ('Staking', 'TrainerEnrolled'): EventType.TRAINER_ENROLLED,
            ('Staking', 'TrainerUnenrolled'): EventType.TRAINER_UNENROLLED,
            ('Staking', 'RewardsCalculated'): EventType.REWARDS_CALCULATED,
            ('Staking', 'RewardsClaimed'): EventType.REWARDS_CLAIMED,
            ('Staking', 'TrainerSlashed'): EventType.TRAINER_SLASHED,
            ('Staking', 'ReputationUpdated'): EventType.TRAINER_REPUTATION_UPDATED,
            ('AIRegistry', 'GenomeDeployed'): EventType.GENOME_DEPLOYED,
            ('AIRegistry', 'GenomeEvolved'): EventType.GENOME_EVOLVED,
        }
        
        event_key = (module_id, event_id)
        if event_key not in event_map:
            return None  # Not an AI training event
        
        event_type = event_map[event_key]
        
        return TrainingEvent(
            event_type=event_type,
            block_number=block_number,
            block_hash=block_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=attributes,
        )
    
    async def start_listening(self) -> None:
        """Start listening for blockchain events."""
        if self.is_listening:
            logger.warning("Already listening for events")
            return
        
        if self.mock_mode:
            logger.info("Mock mode: Event listening simulated")
            self.is_listening = True
            return
        
        if not self.substrate:
            if not await self.connect():
                logger.error("Cannot start listening: not connected")
                return
        
        self.is_listening = True
        logger.info("Started listening for blockchain events")
        
        try:
            # Subscribe to new blocks
            async for block_header in self._subscribe_new_heads():
                if not self.is_listening:
                    break
                
                # Get block details
                block_hash = block_header['header']['hash']
                block_number = block_header['header']['number']
                
                # Get events for this block
                events = self.substrate.get_events(block_hash)
                
                # Parse and dispatch relevant events
                for event_record in events:
                    training_event = await self._parse_event(
                        block_number,
                        block_hash,
                        event_record,
                    )
                    
                    if training_event:
                        await self._dispatch_event(training_event)
        
        except Exception as e:
            logger.error("Event listening error", error=str(e))
        finally:
            self.is_listening = False
            logger.info("Stopped listening for events")
    
    async def _subscribe_new_heads(self):
        """Subscribe to new block headers."""
        # This is a simplified version
        # Real implementation would use substrate.subscribe_block_headers()
        while self.is_listening:
            try:
                # Poll for new blocks
                # In production, use proper subscription
                await asyncio.sleep(6)  # Assume 6 second block time
                
                # Get latest block
                block_hash = self.substrate.get_chain_head()
                block = self.substrate.get_block(block_hash)
                
                yield block
            
            except Exception as e:
                logger.error("Block subscription error", error=str(e))
                await asyncio.sleep(10)
    
    def stop_listening(self) -> None:
        """Stop listening for events."""
        self.is_listening = False
        logger.info("Stopping event listener")
    
    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[TrainingEvent]:
        """
        Get recent event history.
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return
        
        Returns:
            List of recent events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    async def emit_mock_event(
        self,
        event_type: EventType,
        data: dict[str, Any],
    ) -> None:
        """
        Emit mock event for testing.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.mock_mode:
            logger.warning("emit_mock_event called outside mock mode")
            return
        
        event = TrainingEvent(
            event_type=event_type,
            block_number=len(self.event_history) + 1,
            block_hash=f"0x{'0' * 64}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        
        await self._dispatch_event(event)
        logger.debug("Emitted mock event", event_type=event_type.value)


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_training_round_handler(
    on_round_started: Callable[[int, str], Awaitable[None]] | None = None,
    on_proof_submitted: Callable[[str, int], Awaitable[None]] | None = None,
    on_round_completed: Callable[[int], Awaitable[None]] | None = None,
) -> dict[EventType, EventHandler]:
    """
    Create convenience handlers for training round events.
    
    Args:
        on_round_started: Callback for round start (round_num, genome_id)
        on_proof_submitted: Callback for proof submission (participant_id, round_num)
        on_round_completed: Callback for round completion (round_num)
    
    Returns:
        Dictionary of event handlers
    """
    handlers = {}
    
    if on_round_started:
        async def handle_started(event: TrainingEvent):
            round_num = event.data.get('round_number')
            genome_id = event.data.get('genome_id')
            await on_round_started(round_num, genome_id)
        handlers[EventType.TRAINING_ROUND_STARTED] = handle_started
    
    if on_proof_submitted:
        async def handle_proof(event: TrainingEvent):
            participant_id = event.data.get('participant')
            round_num = event.data.get('round_number')
            await on_proof_submitted(participant_id, round_num)
        handlers[EventType.TRAINING_PROOF_SUBMITTED] = handle_proof
    
    if on_round_completed:
        async def handle_completed(event: TrainingEvent):
            round_num = event.data.get('round_number')
            await on_round_completed(round_num)
        handlers[EventType.TRAINING_ROUND_COMPLETED] = handle_completed
    
    return handlers


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "EventType",
    "TrainingEvent",
    "EventHandler",
    "BlockchainEventListener",
    "create_training_round_handler",
]
