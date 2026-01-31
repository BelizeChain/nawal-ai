"""
Metrics collection for inference API
Exports Prometheus-compatible metrics
"""

import time
from typing import Dict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


class MetricsCollector:
    """Collect and export inference metrics"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Request counters
        self.requests_total = Counter(
            'nawal_inference_requests_total',
            'Total inference requests',
            ['user_id', 'status'],
            registry=self.registry
        )
        
        # Latency histogram
        self.inference_duration = Histogram(
            'nawal_inference_duration_seconds',
            'Inference duration in seconds',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Token metrics
        self.tokens_generated = Counter(
            'nawal_tokens_generated_total',
            'Total tokens generated',
            registry=self.registry
        )
        
        # Model info gauge
        self.model_info = Gauge(
            'nawal_model_info',
            'Model information',
            ['version', 'parameters'],
            registry=self.registry
        )
    
    async def log_inference(self, user_id: str, prompt_length: int, output_length: int, inference_time: float):
        """Log inference metrics"""
        self.requests_total.labels(user_id=user_id, status='success').inc()
        self.inference_duration.observe(inference_time / 1000.0)  # Convert ms to seconds
        self.tokens_generated.inc(output_length)
    
    async def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)
