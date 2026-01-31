"""
Training metrics persistence to Azure Cosmos DB
Audit trail for federated learning rounds
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging

try:
    from azure.cosmos.aio import CosmosClient
    from azure.cosmos import exceptions, PartitionKey
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False
    logging.warning("Azure Cosmos DB SDK not installed. Run: pip install azure-cosmos")

logger = logging.getLogger(__name__)


class MetricsStore:
    """
    Persist federated learning metrics to Cosmos DB
    Enables regulatory audit and model performance tracking
    """
    
    def __init__(
        self,
        endpoint: str,
        key: str,
        database_name: str = "belizechain",
        container_name: str = "nawal_metrics"
    ):
        if not COSMOS_AVAILABLE:
            raise ImportError("azure-cosmos package required. Install with: pip install azure-cosmos")
        
        self.endpoint = endpoint
        self.key = key
        self.database_name = database_name
        self.container_name = container_name
        self.client: Optional[CosmosClient] = None
        self.container = None
    
    async def initialize(self):
        """Initialize Cosmos DB connection"""
        self.client = CosmosClient(self.endpoint, self.key)
        
        # Create database if not exists
        database = await self.client.create_database_if_not_exists(self.database_name)
        
        # Create container with partition key on round_id for efficient queries
        self.container = await database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/round_id"),
            offer_throughput=400  # 400 RU/s for dev (increase for production)
        )
        
        logger.info(f"✅ Cosmos DB initialized: {self.database_name}/{self.container_name}")
    
    async def log_training_round(
        self,
        round_id: int,
        metrics: Dict[str, float],
        participating_clients: int,
        aggregated_weights_hash: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Log metrics for a completed training round
        
        Args:
            round_id: Federated learning round number
            metrics: Dictionary of metrics (accuracy, loss, etc.)
            participating_clients: Number of clients that contributed
            aggregated_weights_hash: Hash of aggregated model weights (integrity check)
            timestamp: Round completion timestamp (defaults to now)
        """
        if self.container is None:
            raise RuntimeError("MetricsStore not initialized. Call initialize() first.")
        
        timestamp = timestamp or datetime.utcnow()
        
        document = {
            "id": f"round_{round_id}_{timestamp.isoformat()}",
            "round_id": round_id,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "participating_clients": participating_clients,
            "aggregated_weights_hash": aggregated_weights_hash,
            "ttl": None,  # No auto-deletion (keep forever for audit)
        }
        
        try:
            await self.container.create_item(body=document)
            logger.info(f"📊 Logged round {round_id}: accuracy={metrics.get('accuracy', 'N/A')}")
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"Round {round_id} already logged (duplicate)")
        except Exception as e:
            logger.error(f"Failed to log round {round_id}: {e}")
            raise
    
    async def log_client_contribution(
        self,
        round_id: int,
        client_id: str,
        belizeid: Optional[str],
        local_metrics: Dict[str, float],
        num_samples: int,
        computation_time_seconds: float
    ):
        """
        Log individual client contribution for PoUW rewards
        
        Args:
            round_id: Training round number
            client_id: Client identifier (IP hash or BelizeID)
            belizeid: Authenticated BelizeID (None for anonymous clients)
            local_metrics: Client's local training metrics
            num_samples: Number of training samples contributed
            computation_time_seconds: Time spent on local training
        """
        if self.container is None:
            raise RuntimeError("MetricsStore not initialized")
        
        document = {
            "id": f"client_{client_id}_round_{round_id}_{datetime.utcnow().isoformat()}",
            "round_id": round_id,
            "client_id": client_id,
            "belizeid": belizeid,
            "timestamp": datetime.utcnow().isoformat(),
            "local_metrics": local_metrics,
            "num_samples": num_samples,
            "computation_time_seconds": computation_time_seconds,
            "ttl": 7776000,  # Auto-delete after 90 days (privacy)
        }
        
        try:
            await self.container.create_item(body=document)
        except Exception as e:
            logger.error(f"Failed to log client contribution: {e}")
            # Don't raise - client contribution logging is non-critical
    
    async def get_round_history(
        self,
        start_round: int,
        end_round: int
    ) -> List[Dict]:
        """
        Query training history for a range of rounds
        Used for FSC regulatory audits
        """
        if self.container is None:
            raise RuntimeError("MetricsStore not initialized")
        
        query = f"""
        SELECT * FROM c 
        WHERE c.round_id >= {start_round} 
        AND c.round_id <= {end_round}
        AND NOT IS_DEFINED(c.client_id)
        ORDER BY c.round_id ASC
        """
        
        items = []
        async for item in self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ):
            items.append(item)
        
        return items
    
    async def get_client_pouw_score(
        self,
        belizeid: str,
        start_round: int,
        end_round: int
    ) -> Dict[str, float]:
        """
        Calculate Proof of Useful Work score for a client
        
        Returns:
            Dict with quality, timeliness, and honesty scores
        """
        if self.container is None:
            raise RuntimeError("MetricsStore not initialized")
        
        query = f"""
        SELECT * FROM c 
        WHERE c.belizeid = '{belizeid}'
        AND c.round_id >= {start_round}
        AND c.round_id <= {end_round}
        """
        
        contributions = []
        async for item in self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ):
            contributions.append(item)
        
        if not contributions:
            return {"quality": 0.0, "timeliness": 0.0, "honesty": 0.0}
        
        # Calculate scores (simplified - implement full PoUW algorithm)
        avg_accuracy = sum(c["local_metrics"].get("accuracy", 0) for c in contributions) / len(contributions)
        avg_timeliness = 1.0  # Placeholder: calculate based on submission timestamps
        avg_honesty = 1.0     # Placeholder: calculate based on privacy compliance
        
        return {
            "quality": avg_accuracy,
            "timeliness": avg_timeliness,
            "honesty": avg_honesty,
            "total_contributions": len(contributions)
        }
    
    async def close(self):
        """Close Cosmos DB connection"""
        if self.client:
            await self.client.close()


# Alternative: PostgreSQL backend for self-hosted deployments
class PostgreSQLMetricsStore:
    """
    PostgreSQL-based metrics store (alternative to Cosmos DB)
    Use for on-premise deployments or cost optimization
    """
    
    def __init__(self, connection_string: str):
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg required. Install with: pip install asyncpg")
        
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Create database pool and tables"""
        import asyncpg
        
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            # Create training_rounds table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS training_rounds (
                    round_id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    metrics JSONB NOT NULL,
                    participating_clients INTEGER NOT NULL,
                    aggregated_weights_hash TEXT NOT NULL
                )
            """)
            
            # Create client_contributions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS client_contributions (
                    id SERIAL PRIMARY KEY,
                    round_id INTEGER NOT NULL,
                    client_id TEXT NOT NULL,
                    belizeid TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    local_metrics JSONB NOT NULL,
                    num_samples INTEGER NOT NULL,
                    computation_time_seconds REAL NOT NULL,
                    FOREIGN KEY (round_id) REFERENCES training_rounds(round_id)
                )
            """)
            
            # Create index for fast BelizeID queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contributions_belizeid 
                ON client_contributions(belizeid)
            """)
        
        logger.info("✅ PostgreSQL metrics store initialized")
    
    async def log_training_round(self, round_id: int, metrics: Dict, participating_clients: int, aggregated_weights_hash: str):
        """Log training round to PostgreSQL"""
        import json
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO training_rounds (round_id, timestamp, metrics, participating_clients, aggregated_weights_hash)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (round_id) DO NOTHING
            """, round_id, datetime.utcnow(), json.dumps(metrics), participating_clients, aggregated_weights_hash)
    
    async def close(self):
        """Close database pool"""
        if self.pool:
            await self.pool.close()


# Factory function to create appropriate metrics store
def create_metrics_store(backend: str = "cosmosdb", **kwargs) -> MetricsStore:
    """
    Create metrics store instance
    
    Args:
        backend: "cosmosdb" or "postgresql"
        **kwargs: Backend-specific configuration
    
    Returns:
        Initialized metrics store
    """
    if backend == "cosmosdb":
        if not COSMOS_AVAILABLE:
            logger.warning("Cosmos DB not available, falling back to PostgreSQL")
            backend = "postgresql"
        else:
            return MetricsStore(**kwargs)
    
    if backend == "postgresql":
        return PostgreSQLMetricsStore(**kwargs)
    
    raise ValueError(f"Unknown backend: {backend}")
