"""
Production-grade inference API for Nawal BelizeChain LLM
FastAPI REST/gRPC endpoint for serving trained models
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, AsyncIterator
import torch
import asyncio
import logging
from datetime import datetime, timezone
import json

from nawal.client.model import BelizeChainLLM
from nawal.security.dp_inference import DPInferenceGuard
from nawal.blockchain.identity_verifier import BelizeIDVerifier
from nawal.monitoring.metrics_collector import InferenceMetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan (replaces deprecated @app.on_event)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    logger.info("Loading BelizeChain LLM model...")
    try:
        model = BelizeChainLLM.from_checkpoint(
            checkpoint_path="checkpoints/final_checkpoint.pt",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        logger.info(f"Model loaded successfully ({model.num_parameters():,} parameters)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

    app.state.model = model
    app.state.dp_guard = DPInferenceGuard(epsilon=2.0)
    yield
    # Shutdown cleanup
    app.state.model = None
    app.state.dp_guard = None


# Initialize FastAPI app
app = FastAPI(
    title="Nawal Inference API",
    description="Privacy-preserving LLM inference for BelizeChain",
    version="1.0.0",
    lifespan=lifespan,
)

belizeid_verifier = BelizeIDVerifier()
metrics = InferenceMetricsCollector()


# =============================================================================
# Rate Limiting
# =============================================================================

class _RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        self._hits[key] = [t for t in hits if now - t < self.window]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True


_rate_limiter = _RateLimiter(
    max_requests=int(os.getenv("NAWAL_INFERENCE_RATE_LIMIT", "30")),
    window_seconds=int(os.getenv("NAWAL_INFERENCE_RATE_WINDOW", "60")),
)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Enforce per-IP rate limiting on inference endpoints."""
    if request.url.path in ("/health", "/model/info"):
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )
    return await call_next(request)


class InferenceRequest(BaseModel):
    """Request schema for text generation"""
    prompt: str = Field(..., min_length=1, max_length=2048, description="Input text prompt")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    stream: bool = Field(False, description="Enable streaming response")
    belizeid: Optional[str] = Field(None, description="BelizeID for authenticated requests")


class InferenceResponse(BaseModel):
    """Response schema for text generation"""
    text: str
    tokens_generated: int
    inference_time_ms: float
    model_version: str
    timestamp: str


class ModelInfo(BaseModel):
    """Model metadata response"""
    model_name: str
    version: str
    parameters: int
    training_rounds: int
    last_updated: str
    privacy_budget: Dict[str, float]


async def verify_belizeid(belizeid: Optional[str] = Header(None)) -> Optional[str]:
    """Dependency: Verify BelizeID authentication"""
    if belizeid:
        is_valid = await belizeid_verifier.verify(belizeid)
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid BelizeID")
        return belizeid
    return None


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": request.app.state.model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(request: Request):
    """Get model metadata and statistics"""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name="BelizeChainLLM",
        version=model.version,
        parameters=model.num_parameters(),
        training_rounds=model.training_rounds,
        last_updated=model.last_updated.isoformat(),
        privacy_budget={
            "epsilon": model.privacy_epsilon,
            "delta": model.privacy_delta
        }
    )


@app.post("/infer", response_model=InferenceResponse)
async def generate_text(
    request: InferenceRequest,
    http_request: Request,
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Generate text from prompt (non-streaming)"""
    model = http_request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now(timezone.utc)

    try:
        # Apply differential privacy noise to input embeddings
        with http_request.app.state.dp_guard.inference_context():
            output = await asyncio.to_thread(
                model.generate,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

        inference_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Log metrics
        await metrics.log_inference(
            user_id=belizeid or "anonymous",
            prompt_length=len(request.prompt),
            output_length=len(output),
            inference_time=inference_time
        )

        return InferenceResponse(
            text=output,
            tokens_generated=len(output.split()),  # DESIGN NOTE: swap for tokenizer.encode(output) when tokenizer is available
            inference_time_ms=inference_time,
            model_version=model.version,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")


@app.post("/infer/stream")
async def generate_text_stream(
    request: InferenceRequest,
    http_request: Request,
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Generate text with streaming response"""
    model = http_request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def token_generator() -> AsyncIterator[str]:
        """Stream tokens as they're generated"""
        try:
            for token in model.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.01)  # Small delay for streaming
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"error": "Streaming failed"}) + "\n"

    return StreamingResponse(
        token_generator(),
        media_type="application/x-ndjson"
    )


@app.post("/batch/infer")
async def batch_inference(
    requests: List[InferenceRequest],
    http_request: Request,
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Process multiple inference requests in batch"""
    model = http_request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 32:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (32)")

    results = []
    for req in requests:
        try:
            response = await generate_text(req, http_request, belizeid)
            results.append({"status": "success", "data": response})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {"results": results, "total": len(requests)}


@app.get("/metrics")
async def get_metrics():
    """Get inference metrics (Prometheus format)"""
    return await metrics.export_prometheus()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",  # Security: bind to localhost only (use reverse proxy for external access)
        port=8000,
        log_level="info"
    )
