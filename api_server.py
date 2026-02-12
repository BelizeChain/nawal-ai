"""
Nawal Federated Learning API Server

Production-grade FastAPI server for Nawal federated learning orchestration.
Provides REST endpoints for FL round management, model submission, and PoUW integration.

Endpoints:
- POST /api/v1/fl/rounds - Start new FL round
- GET /api/v1/fl/rounds/{round_id} - Get round status
- POST /api/v1/fl/participants/enroll - Enroll participant
- POST /api/v1/fl/participants/submit - Submit model delta
- GET /api/v1/fl/participants/{account_id} - Get participant stats
- GET /api/v1/fl/metrics - Get system metrics

Author: BelizeChain AI Team
Date: October 2025
License: MIT
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Nawal imports
from nawal.config import NawalConfig
from nawal.blockchain.staking_connector import (
    StakingConnector,
    ParticipantInfo,
    TrainingSubmission,
)
from nawal.server.aggregator import FederatedAggregator, AggregationStrategy


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig(BaseModel):
    """API server configuration."""
    
    host: str = Field(
        default="127.0.0.1",
        description="Server host (use 0.0.0.0 for Docker/cloud, set via NAWAL_HOST env var)"
    )
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Blockchain connection
    blockchain_rpc: str = Field(
        default="ws://localhost:9944",
        description="BelizeChain RPC endpoint"
    )
    blockchain_enabled: bool = Field(
        default=True,
        description="Enable blockchain integration"
    )
    
    # FL configuration
    min_participants: int = Field(default=3, ge=1, description="Minimum FL participants")
    max_participants: int = Field(default=100, le=1000, description="Maximum FL participants")
    round_timeout: int = Field(default=3600, description="Round timeout in seconds")
    
    # Security
    enable_auth: bool = Field(default=False, description="Enable API authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    
    # Storage
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory for model checkpoints"
    )
    
    @field_validator('checkpoint_dir')
    @classmethod
    def create_checkpoint_dir(cls, v: Path) -> Path:
        """Create checkpoint directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# =============================================================================
# API Models
# =============================================================================

class EnrollRequest(BaseModel):
    """Request to enroll FL participant."""
    
    account_id: str = Field(..., description="Participant account ID (SS58 address)")
    stake_amount: int = Field(..., ge=1000, description="Stake amount in Mahogany")
    public_key: Optional[str] = Field(None, description="Encryption public key")


class EnrollResponse(BaseModel):
    """Response for enrollment."""
    
    success: bool
    participant_id: str
    message: str
    enrollment_time: str


class SubmitModelRequest(BaseModel):
    """Request to submit model delta."""
    
    participant_id: str = Field(..., description="Participant ID")
    round_id: str = Field(..., description="FL round ID")
    model_cid: str = Field(..., description="IPFS CID of model delta")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Model quality (0-100)")
    training_samples: int = Field(..., ge=1, description="Number of training samples")
    privacy_proof: Optional[str] = Field(None, description="Zero-knowledge privacy proof")


class SubmitModelResponse(BaseModel):
    """Response for model submission."""
    
    success: bool
    submission_id: str
    reward_eligible: bool
    estimated_reward: int
    message: str


class StartRoundRequest(BaseModel):
    """Request to start FL round."""
    
    dataset_name: str = Field(..., description="Dataset identifier")
    target_accuracy: float = Field(default=0.85, ge=0.0, le=1.0, description="Target accuracy")
    max_participants: int = Field(default=10, ge=1, le=100, description="Max participants")
    timeout: int = Field(default=3600, ge=60, description="Round timeout (seconds)")


class StartRoundResponse(BaseModel):
    """Response for starting round."""
    
    round_id: str
    status: str
    start_time: str
    expected_completion: str


class RoundStatus(BaseModel):
    """FL round status."""
    
    round_id: str
    status: str  # "pending", "active", "completed", "failed"
    participants: int
    submissions_received: int
    current_accuracy: Optional[float]
    start_time: str
    completion_time: Optional[str]


class ParticipantStats(BaseModel):
    """Participant statistics."""
    
    account_id: str
    total_rounds: int
    successful_rounds: int
    total_rewards: int
    average_quality: float
    last_submission: Optional[str]


class SystemMetrics(BaseModel):
    """System-wide FL metrics."""
    
    total_rounds: int
    active_rounds: int
    total_participants: int
    active_participants: int
    total_models_trained: int
    average_round_time: float
    blockchain_connected: bool


# =============================================================================
# Global State
# =============================================================================

class AppState:
    """Application state."""
    
    def __init__(self):
        self.config: Optional[ServerConfig] = None
        self.staking_connector: Optional[StakingConnector] = None
        self.fl_aggregator: Optional[FederatedAggregator] = None
        self.active_rounds: Dict[str, Dict[str, Any]] = {}
        self.round_counter: int = 0
        # Metrics tracking
        self.participant_submissions: Dict[str, float] = {}  # account_id -> last_submission_timestamp
        self.completed_rounds: List[Dict[str, Any]] = []  # Track completed rounds for stats
    
    async def initialize(self, config: ServerConfig):
        """Initialize application state."""
        self.config = config
        
        # Initialize blockchain connector
        if config.blockchain_enabled:
            self.staking_connector = StakingConnector(
                node_url=config.blockchain_rpc,  # Fixed: Changed rpc_url to node_url
                mock_mode=False  # Production mode
            )
            try:
                await self.staking_connector.connect()
                logger.info("✅ Connected to BelizeChain at {}", config.blockchain_rpc)
            except Exception as e:
                logger.error("❌ Failed to connect to blockchain: {}", e)
                logger.warning("Running in degraded mode (blockchain unavailable)")
        else:
            logger.info("Blockchain integration disabled")
        
        # Initialize FL aggregator (will be created per round)
        logger.info("✅ Nawal API server initialized")
    
    async def shutdown(self):
        """Cleanup resources."""
        if self.staking_connector:
            await self.staking_connector.disconnect()
        logger.info("✅ Nawal API server shutdown complete")


# Global app state
app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    config = ServerConfig(
        blockchain_rpc=os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944"),
        blockchain_enabled=os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true",
        port=int(os.getenv("PORT", "8080")),
    )
    await app_state.initialize(config)
    
    yield
    
    # Shutdown
    await app_state.shutdown()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Nawal Federated Learning API",
    description="Production API for BelizeChain federated learning orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "nawal-fl-api",
        "timestamp": datetime.utcnow().isoformat(),
        "blockchain_connected": app_state.staking_connector is not None,
    }


@app.get("/api/v1/status")
async def get_status():
    """Get API status and configuration."""
    return {
        "service": "Nawal Federated Learning",
        "version": "1.0.0",
        "blockchain": {
            "enabled": app_state.config.blockchain_enabled,
            "connected": app_state.staking_connector is not None,
            "rpc_url": app_state.config.blockchain_rpc,
        },
        "active_rounds": len(app_state.active_rounds),
        "total_rounds": app_state.round_counter,
    }


# =============================================================================
# FL Round Management
# =============================================================================

@app.post("/api/v1/fl/rounds", response_model=StartRoundResponse, status_code=status.HTTP_201_CREATED)
async def start_fl_round(request: StartRoundRequest):
    """
    Start new federated learning round.
    
    This endpoint:
    1. Creates new FL round with unique ID
    2. Initializes aggregator
    3. Waits for participant enrollment
    """
    try:
        # Generate round ID
        round_id = f"round_{app_state.round_counter}_{uuid.uuid4().hex[:8]}"
        app_state.round_counter += 1
        
        # Create round metadata
        start_time = datetime.utcnow()
        round_data = {
            "round_id": round_id,
            "dataset": request.dataset_name,
            "target_accuracy": request.target_accuracy,
            "max_participants": request.max_participants,
            "timeout": request.timeout,
            "status": "pending",
            "participants": [],
            "submissions": [],
            "start_time": start_time.isoformat(),
            "completion_time": None,
        }
        
        app_state.active_rounds[round_id] = round_data
        
        logger.info(
            "🚀 Started FL round {} for dataset '{}'",
            round_id,
            request.dataset_name
        )
        
        return StartRoundResponse(
            round_id=round_id,
            status="pending",
            start_time=start_time.isoformat(),
            expected_completion=(start_time.timestamp() + request.timeout).__str__(),
        )
        
    except Exception as e:
        logger.error("Failed to start FL round: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start FL round: {str(e)}"
        )


@app.get("/api/v1/fl/rounds/{round_id}", response_model=RoundStatus)
async def get_round_status(round_id: str):
    """Get FL round status."""
    if round_id not in app_state.active_rounds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Round {round_id} not found"
        )
    
    round_data = app_state.active_rounds[round_id]
    
    return RoundStatus(
        round_id=round_id,
        status=round_data["status"],
        participants=len(round_data["participants"]),
        submissions_received=len(round_data["submissions"]),
        current_accuracy=round_data.get("current_accuracy"),
        start_time=round_data["start_time"],
        completion_time=round_data.get("completion_time"),
    )


# =============================================================================
# Participant Management
# =============================================================================

@app.post("/api/v1/fl/participants/enroll", response_model=EnrollResponse)
async def enroll_participant(request: EnrollRequest):
    """
    Enroll participant in federated learning.
    
    This endpoint:
    1. Verifies participant KYC via Identity pallet
    2. Enrolls participant via Staking pallet
    3. Returns enrollment confirmation
    """
    try:
        if not app_state.staking_connector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Blockchain connector not available"
            )
        
        # Enroll via blockchain
        result = await app_state.staking_connector.enroll_participant(
            account_id=request.account_id,
            stake_amount=request.stake_amount,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        logger.info("✅ Enrolled participant: {}", request.account_id)
        
        return EnrollResponse(
            success=True,
            participant_id=request.account_id,
            message="Participant enrolled successfully",
            enrollment_time=datetime.utcnow().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enroll participant: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {str(e)}"
        )


@app.post("/api/v1/fl/participants/submit", response_model=SubmitModelResponse)
async def submit_model_delta(request: SubmitModelRequest):
    """
    Submit model delta for FL round.
    
    This endpoint:
    1. Validates participant enrollment
    2. Stores model delta (IPFS CID)
    3. Submits training proof to blockchain
    4. Calculates PoUW score and potential rewards
    """
    try:
        # Verify round exists
        if request.round_id not in app_state.active_rounds:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Round {request.round_id} not found"
            )
        
        round_data = app_state.active_rounds[request.round_id]
        
        # Verify round is active
        if round_data["status"] != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Round {request.round_id} is not active"
            )
        
        # Create submission
        submission = TrainingSubmission(
            participant_id=request.participant_id,
            round_id=request.round_id,
            model_cid=request.model_cid,
            quality_score=request.quality_score,
            sample_count=request.training_samples,
            privacy_proof=request.privacy_proof or "",
        )
        
        # Submit to blockchain if available
        if app_state.staking_connector:
            result = await app_state.staking_connector.submit_training_proof(submission)
            
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["message"]
                )
        
        # Track submission timestamp
        from time import time
        app_state.participant_submissions[request.participant_id] = time()
        
        # Add to round submissions
        round_data["submissions"].append({
            "participant_id": request.participant_id,
            "model_cid": request.model_cid,
            "quality_score": request.quality_score,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Calculate estimated reward (simplified)
        estimated_reward = int(request.quality_score * 1_000_000_000)  # Quality-based
        
        logger.info(
            "✅ Received model submission from {} for round {}",
            request.participant_id,
            request.round_id
        )
        
        return SubmitModelResponse(
            success=True,
            submission_id=f"{request.round_id}_{request.participant_id}",
            reward_eligible=request.quality_score >= 60.0,
            estimated_reward=estimated_reward,
            message="Model delta submitted successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit model: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model submission failed: {str(e)}"
        )


@app.get("/api/v1/fl/participants/{account_id}", response_model=ParticipantStats)
async def get_participant_stats(account_id: str):
    """Get participant statistics."""
    try:
        if not app_state.staking_connector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Blockchain connector not available"
            )
        
        # Get participant info from blockchain
        participant = await app_state.staking_connector.get_participant(account_id)
        
        if not participant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Participant {account_id} not found"
            )
        
        # Get last submission timestamp
        last_submission_ts = app_state.participant_submissions.get(account_id)
        last_submission_iso = None
        if last_submission_ts:
            from datetime import datetime
            last_submission_iso = datetime.fromtimestamp(last_submission_ts).isoformat()
        
        return ParticipantStats(
            account_id=account_id,
            total_rounds=participant.total_rounds,
            successful_rounds=participant.successful_rounds,
            total_rewards=participant.total_rewards_earned,
            average_quality=participant.avg_quality_score,
            last_submission=last_submission_iso,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get participant stats: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {str(e)}"
        )


# =============================================================================
# System Metrics
# =============================================================================

@app.get("/api/v1/fl/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get system-wide FL metrics."""
    try:
        active_rounds_count = sum(
            1 for r in app_state.active_rounds.values()
            if r["status"] == "active"
        )
        
        # Get blockchain stats if available
        total_participants = 0
        active_participants = 0
        
        if app_state.staking_connector:
            participants = await app_state.staking_connector.get_all_participants()
            total_participants = len(participants)
            active_participants = sum(1 for p in participants if p.is_enrolled)
        
        # Calculate average round time from completed rounds
        average_round_time_seconds = 0.0
        if app_state.completed_rounds:
            total_time = 0.0
            for round_info in app_state.completed_rounds:
                if round_info.get("start_time") and round_info.get("completion_time"):
                    start = datetime.fromisoformat(round_info["start_time"])
                    end = datetime.fromisoformat(round_info["completion_time"])
                    total_time += (end - start).total_seconds()
            
            if total_time > 0:
                average_round_time_seconds = total_time / len(app_state.completed_rounds)
        
        return SystemMetrics(
            total_rounds=app_state.round_counter,
            active_rounds=active_rounds_count,
            total_participants=total_participants,
            active_participants=active_participants,
            total_models_trained=sum(len(r["submissions"]) for r in app_state.active_rounds.values()),
            average_round_time=average_round_time_seconds,
            blockchain_connected=app_state.staking_connector is not None,
        )
        
    except Exception as e:
        logger.error("Failed to get metrics: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metrics: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run API server."""
    # Configure logging
    logger.add(
        "logs/nawal_api_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # Get configuration from environment
    # Priority: NAWAL_API_HOST → NAWAL_HOST → HOST → default
    # Use 0.0.0.0 for Docker/cloud deployments, 127.0.0.1 for local dev
    host = os.getenv("NAWAL_API_HOST", os.getenv("NAWAL_HOST", os.getenv("HOST", "0.0.0.0")))
    port = int(os.getenv("NAWAL_PORT", os.getenv("PORT", "8080")))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info("🚀 Starting Nawal FL API server on {}:{}", host, port)
    logger.info("   Blockchain RPC: {}", os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944"))
    logger.info("   Reload: {}", reload)
    logger.info("   Workers: {}", workers)
    
    # Run server
    uvicorn.run(
        "nawal.api_server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers=1 when reload=True
        log_level="info",
    )


if __name__ == "__main__":
    main()
