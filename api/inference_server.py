"""
Production-grade inference API for Nawal BelizeChain LLM
FastAPI REST/gRPC endpoint for serving trained models
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, AsyncIterator
import torch
import asyncio
import logging
from datetime import datetime
import json

from nawal.client.model import BelizeChainLLM
from nawal.security.differential_privacy import DPInferenceGuard
from nawal.blockchain.identity_verifier import BelizeIDVerifier
from nawal.monitoring.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nawal Inference API",
    description="Privacy-preserving LLM inference for BelizeChain",
    version="1.0.0"
)

# Global model instance (loaded on startup)
global_model: Optional[BelizeChainLLM] = None
dp_guard = DPInferenceGuard(epsilon=2.0)  # ε=2.0 for inference (looser than training)
belizeid_verifier = BelizeIDVerifier()
metrics = MetricsCollector()


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


@app.on_event("startup")
async def load_model():
    """Load trained model on server startup"""
    global global_model
    
    logger.info("Loading BelizeChain LLM model...")
    try:
        global_model = BelizeChainLLM.from_checkpoint(
            checkpoint_path="checkpoints/final_checkpoint.pt",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        global_model.eval()  # Set to evaluation mode
        logger.info(f"✅ Model loaded successfully ({global_model.num_parameters():,} parameters)")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": global_model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model metadata and statistics"""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="BelizeChainLLM",
        version=global_model.version,
        parameters=global_model.num_parameters(),
        training_rounds=global_model.training_rounds,
        last_updated=global_model.last_updated.isoformat(),
        privacy_budget={
            "epsilon": global_model.privacy_epsilon,
            "delta": global_model.privacy_delta
        }
    )


@app.post("/infer", response_model=InferenceResponse)
async def generate_text(
    request: InferenceRequest,
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Generate text from prompt (non-streaming)"""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.utcnow()
    
    try:
        # Apply differential privacy noise to input embeddings
        with dp_guard.inference_context():
            output = await asyncio.to_thread(
                global_model.generate,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
        
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Log metrics
        await metrics.log_inference(
            user_id=belizeid or "anonymous",
            prompt_length=len(request.prompt),
            output_length=len(output),
            inference_time=inference_time
        )
        
        return InferenceResponse(
            text=output,
            tokens_generated=len(output.split()),
            inference_time_ms=inference_time,
            model_version=global_model.version,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/stream")
async def generate_text_stream(
    request: InferenceRequest,
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Generate text with streaming response"""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def token_generator() -> AsyncIterator[str]:
        """Stream tokens as they're generated"""
        try:
            for token in global_model.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.01)  # Small delay for streaming
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
    
    return StreamingResponse(
        token_generator(),
        media_type="application/x-ndjson"
    )


@app.post("/batch/infer")
async def batch_inference(
    requests: List[InferenceRequest],
    belizeid: Optional[str] = Depends(verify_belizeid)
):
    """Process multiple inference requests in batch"""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 32:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (32)")
    
    results = []
    for req in requests:
        try:
            response = await generate_text(req, belizeid)
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
