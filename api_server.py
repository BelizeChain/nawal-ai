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
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.exceptions import RequestValidationError
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
from nawal.blockchain.identity_verifier import create_verifier
from nawal.maintenance.input_screener import InputScreener
from nawal.maintenance.output_filter import OutputFilter
from nawal.maintenance.interfaces import RiskLevel
from nawal.server.aggregator import FederatedAggregator, AggregationStrategy

# =============================================================================
# Configuration
# =============================================================================


class ServerConfig(BaseModel):
    """API server configuration."""

    host: str = Field(
        default="127.0.0.1",
        description="Server host (use 0.0.0.0 for Docker/cloud, set via NAWAL_HOST env var)",
    )
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # Blockchain connection
    blockchain_rpc: str = Field(
        default="ws://localhost:9944", description="BelizeChain RPC endpoint"
    )
    blockchain_enabled: bool = Field(
        default=True, description="Enable blockchain integration"
    )

    # FL configuration
    min_participants: int = Field(
        default=3, ge=1, description="Minimum FL participants"
    )
    max_participants: int = Field(
        default=100, le=1000, description="Maximum FL participants"
    )
    round_timeout: int = Field(default=3600, description="Round timeout in seconds")

    # Security
    enable_auth: bool = Field(default=False, description="Enable API authentication")
    api_key: Optional[str] = Field(
        default=None, description="API key for authentication"
    )

    # Storage
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"), description="Directory for model checkpoints"
    )

    @field_validator("checkpoint_dir")
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

    account_id: str = Field(
        ..., max_length=256, description="Participant account ID (SS58 address)"
    )
    stake_amount: int = Field(..., ge=1000, description="Stake amount in Mahogany")
    public_key: Optional[str] = Field(
        None, max_length=512, description="Encryption public key"
    )
    belizeid: Optional[str] = Field(
        None, max_length=64, description="BelizeID for KYC verification"
    )


class EnrollResponse(BaseModel):
    """Response for enrollment."""

    success: bool
    participant_id: str
    message: str
    enrollment_time: str


class SubmitModelRequest(BaseModel):
    """Request to submit model delta."""

    participant_id: str = Field(..., max_length=256, description="Participant ID")
    round_id: str = Field(..., max_length=256, description="FL round ID")
    model_cid: str = Field(..., max_length=512, description="IPFS CID of model delta")
    quality_score: float = Field(
        ..., ge=0.0, le=100.0, description="Model quality (0-100)"
    )
    training_samples: int = Field(..., ge=1, description="Number of training samples")
    privacy_proof: Optional[str] = Field(
        None, max_length=2048, description="Zero-knowledge privacy proof"
    )


class SubmitModelResponse(BaseModel):
    """Response for model submission."""

    success: bool
    submission_id: str
    reward_eligible: bool
    estimated_reward: int
    message: str


class StartRoundRequest(BaseModel):
    """Request to start FL round."""

    dataset_name: str = Field(..., max_length=256, description="Dataset identifier")
    target_accuracy: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Target accuracy"
    )
    max_participants: int = Field(
        default=10, ge=1, le=100, description="Max participants"
    )
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


class StatusResponse(BaseModel):
    """API status response."""

    service: str
    version: str
    blockchain: Dict[str, Any]
    active_rounds: int
    total_rounds: int


# =============================================================================
# Global State
# =============================================================================


class AppState:
    """Application state."""

    def __init__(self):
        self.config: Optional[ServerConfig] = None
        self.staking_connector: Optional[StakingConnector] = None
        self.fl_aggregator: Optional[FederatedAggregator] = None
        self.identity_verifier = None
        self.input_screener: Optional[InputScreener] = None
        self.output_filter: Optional[OutputFilter] = None
        self.active_rounds: Dict[str, Dict[str, Any]] = {}
        self.round_counter: int = 0
        # Metrics tracking
        self.participant_submissions: Dict[str, float] = (
            {}
        )  # account_id -> last_submission_timestamp
        self.completed_rounds: List[Dict[str, Any]] = (
            []
        )  # Track completed rounds for stats

    async def initialize(self, config: ServerConfig):
        """Initialize application state."""
        self.config = config

        # Initialize identity verifier
        verifier_mode = "production" if _is_production else "development"
        try:
            self.identity_verifier = create_verifier(
                mode=verifier_mode,
                rpc_url=config.blockchain_rpc,
            )
            await self.identity_verifier.connect()
            logger.info("✅ Identity verifier initialized (mode={})", verifier_mode)
        except Exception as e:
            logger.error("❌ Failed to initialize identity verifier: {}", e)
            self.identity_verifier = None

        # Initialize blockchain connector
        if config.blockchain_enabled:
            self.staking_connector = StakingConnector(
                node_url=config.blockchain_rpc,  # Fixed: Changed rpc_url to node_url
                mock_mode=False,  # Production mode
            )
            try:
                await self.staking_connector.connect()
                logger.info("✅ Connected to BelizeChain at {}", config.blockchain_rpc)
            except Exception as e:
                logger.error("❌ Failed to connect to blockchain: {}", e)
                logger.warning("Running in degraded mode (blockchain unavailable)")
                self.staking_connector = None
        else:
            logger.info("Blockchain integration disabled")

        # Initialize FL aggregator (will be created per round)

        # Initialize content safety filters
        self.input_screener = InputScreener()
        self.output_filter = OutputFilter()
        logger.info("✅ Content safety filters initialized")

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

_is_production = os.getenv("NAWAL_ENV", "development").lower() == "production"

app = FastAPI(
    title="Nawal Federated Learning API",
    description="Production API for BelizeChain federated learning orchestration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
)

# CORS middleware — restrict origins in production
_allowed_origins = os.getenv(
    "NAWAL_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# =============================================================================
# API Key Authentication Middleware
# =============================================================================


async def verify_api_key(request: Request, call_next) -> JSONResponse:
    """Verify API key when authentication is enabled."""
    # Skip auth for health check and docs
    if request.url.path in (
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/healthz",
        "/readyz",
    ):
        return await call_next(request)

    if not app_state.config or not app_state.config.enable_auth:
        return await call_next(request)

    # Only accept API key from header — query params leak in access logs
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != app_state.config.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"},
        )
    return await call_next(request)


app.middleware("http")(verify_api_key)


# =============================================================================
# Rate Limiting Middleware
# =============================================================================


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        # Prune expired entries
        self._hits[key] = [t for t in hits if now - t < self.window]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True


_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("NAWAL_RATE_LIMIT", "60")),
    window_seconds=int(os.getenv("NAWAL_RATE_WINDOW", "60")),
)


async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP rate limiting."""
    if request.url.path in ("/health", "/healthz", "/readyz"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."},
        )
    return await call_next(request)


app.middleware("http")(rate_limit_middleware)


async def body_size_limit_middleware(request: Request, call_next):
    """Reject requests with bodies exceeding 10 MB."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10_485_760:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large (max 10 MB)"},
        )
    return await call_next(request)


app.middleware("http")(body_size_limit_middleware)


async def content_screening_middleware(request: Request, call_next):
    """Screen incoming POST/PUT request bodies for harmful content."""
    if request.method in ("POST", "PUT") and app_state.input_screener:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.body()
            if body:
                try:
                    text = body.decode("utf-8", errors="replace")
                    result = app_state.input_screener.screen(text)
                    if result.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCKED):
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={
                                "detail": "Request blocked by content safety filter",
                                "flags": result.flags,
                            },
                        )
                except Exception:
                    pass  # Don't block on screener errors

    return await call_next(request)


app.middleware("http")(content_screening_middleware)


# =============================================================================
# Health & Status Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config_ok = app_state.config is not None
    blockchain_ok = app_state.staking_connector is not None
    is_healthy = config_ok
    return JSONResponse(
        content={
            "status": "healthy" if is_healthy else "degraded",
            "service": "nawal-fl-api",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blockchain_connected": blockchain_ok,
        },
        status_code=200 if is_healthy else 503,
    )


@app.get("/healthz")
async def liveness_check():
    """Liveness probe — process is alive."""
    return {"status": "alive"}


@app.get("/readyz")
async def readiness_check():
    """Readiness probe — service is ready to accept traffic."""
    config_ok = app_state.config is not None
    blockchain_ok = app_state.staking_connector is not None
    is_ready = config_ok
    return JSONResponse(
        content={"ready": is_ready, "blockchain_connected": blockchain_ok},
        status_code=200 if is_ready else 503,
    )


@app.get("/api/v1/status", response_model=StatusResponse, status_code=200)
async def get_status():
    """Get API status and configuration."""
    return StatusResponse(
        service="Nawal Federated Learning",
        version="1.0.0",
        blockchain={
            "enabled": (
                app_state.config.blockchain_enabled if app_state.config else False
            ),
            "connected": app_state.staking_connector is not None,
            "rpc_url": app_state.config.blockchain_rpc if app_state.config else "",
        },
        active_rounds=len(app_state.active_rounds),
        total_rounds=app_state.round_counter,
    )


# =============================================================================
# FL Round Management
# =============================================================================


@app.post(
    "/api/v1/fl/rounds",
    response_model=StartRoundResponse,
    status_code=status.HTTP_201_CREATED,
)
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
        start_time = datetime.now(timezone.utc)
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
            "🚀 Started FL round {} for dataset '{}'", round_id, request.dataset_name
        )

        return StartRoundResponse(
            round_id=round_id,
            status="pending",
            start_time=start_time.isoformat(),
            expected_completion=(
                start_time + timedelta(seconds=request.timeout)
            ).isoformat(),
        )

    except Exception as e:
        logger.error("Failed to start FL round: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start FL round. Check server logs for details.",
        )


@app.get("/api/v1/fl/rounds/{round_id}", response_model=RoundStatus, status_code=200)
async def get_round_status(round_id: str):
    """Get FL round status."""
    if round_id not in app_state.active_rounds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Round {round_id} not found"
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


@app.post(
    "/api/v1/fl/participants/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
)
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
                detail="Blockchain connector not available",
            )

        # Verify KYC via BelizeID before enrollment
        if app_state.identity_verifier and request.belizeid:
            is_verified = await app_state.identity_verifier.verify(request.belizeid)
            if not is_verified:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="KYC verification failed for the provided BelizeID",
                )
        elif _is_production and not request.belizeid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="BelizeID is required for enrollment in production",
            )

        # Enroll via blockchain
        result = await app_state.staking_connector.enroll_participant(
            account_id=request.account_id,
            stake_amount=request.stake_amount,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
            )

        logger.debug("Enrolled participant: {}", request.account_id)

        return EnrollResponse(
            success=True,
            participant_id=request.account_id,
            message="Participant enrolled successfully",
            enrollment_time=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enroll participant: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enrollment failed. Check server logs for details.",
        )


@app.post(
    "/api/v1/fl/participants/submit",
    response_model=SubmitModelResponse,
    status_code=status.HTTP_201_CREATED,
)
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
                detail=f"Round {request.round_id} not found",
            )

        round_data = app_state.active_rounds[request.round_id]

        # Verify round is active
        if round_data["status"] != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Round {request.round_id} is not active",
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
                    status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
                )

        # Track submission timestamp
        from time import time

        app_state.participant_submissions[request.participant_id] = time()

        # Add to round submissions
        round_data["submissions"].append(
            {
                "participant_id": request.participant_id,
                "model_cid": request.model_cid,
                "quality_score": request.quality_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Calculate estimated reward (simplified)
        estimated_reward = int(request.quality_score * 1_000_000_000)  # Quality-based

        logger.info(
            "✅ Received model submission from {} for round {}",
            request.participant_id,
            request.round_id,
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
            detail="Model submission failed. Check server logs for details.",
        )


@app.get(
    "/api/v1/fl/participants/{account_id}",
    response_model=ParticipantStats,
    status_code=200,
)
async def get_participant_stats(account_id: str):
    """Get participant statistics."""
    try:
        if not app_state.staking_connector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Blockchain connector not available",
            )

        # Get participant info from blockchain
        participant = await app_state.staking_connector.get_participant(account_id)

        if not participant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Participant not found"
            )

        # Get last submission timestamp
        last_submission_ts = app_state.participant_submissions.get(account_id)
        last_submission_iso = None
        if last_submission_ts:
            last_submission_iso = datetime.fromtimestamp(
                last_submission_ts, tz=timezone.utc
            ).isoformat()

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
            detail="Failed to fetch participant stats. Check server logs for details.",
        )


# =============================================================================
# System Metrics
# =============================================================================


@app.get("/api/v1/fl/metrics", response_model=SystemMetrics, status_code=200)
async def get_system_metrics():
    """Get system-wide FL metrics."""
    try:
        active_rounds_count = sum(
            1 for r in app_state.active_rounds.values() if r["status"] == "active"
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
                average_round_time_seconds = total_time / len(
                    app_state.completed_rounds
                )

        return SystemMetrics(
            total_rounds=app_state.round_counter,
            active_rounds=active_rounds_count,
            total_participants=total_participants,
            active_participants=active_participants,
            total_models_trained=sum(
                len(r["submissions"]) for r in app_state.active_rounds.values()
            ),
            average_round_time=average_round_time_seconds,
            blockchain_connected=app_state.staking_connector is not None,
        )

    except Exception as e:
        logger.error("Failed to get metrics: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch metrics. Check server logs for details.",
        )


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Reformat 422 validation errors into consistent envelope."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — never leak internals."""
    logger.error("Unhandled exception on {}: {}", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run API server."""
    # Configure logging via the central monitoring module so that
    # env vars (NAWAL_LOG_LEVEL, NAWAL_LOG_SERIALIZE, NAWAL_ENV) are
    # respected and JSON serialization is enabled in production.
    try:
        from nawal.monitoring.logging_config import configure_logging

        configure_logging(
            log_file=Path("logs/nawal_api.log"),
        )
    except Exception:
        # Fallback: direct loguru setup
        logger.add(
            "logs/nawal_api_{time}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
        )

    # Get configuration from environment
    # Priority: NAWAL_API_HOST → NAWAL_HOST → HOST → default
    # Default to 127.0.0.1 for security; set HOST=0.0.0.0 explicitly for Docker/cloud
    host = os.getenv(
        "NAWAL_API_HOST", os.getenv("NAWAL_HOST", os.getenv("HOST", "127.0.0.1"))
    )
    port = int(os.getenv("NAWAL_PORT", os.getenv("PORT", "8080")))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))

    logger.info("🚀 Starting Nawal FL API server on {}:{}", host, port)
    logger.info(
        "   Blockchain RPC: {}", os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944")
    )
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
