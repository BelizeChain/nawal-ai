"""
Prometheus Exporter for Nawal AI.

Exports metrics in Prometheus format via HTTP endpoint.

Author: BelizeChain Team
License: MIT
"""

from typing import Optional
import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# Optional Prometheus client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusExporter:
    """
    Prometheus metrics exporter for Nawal AI.
    
    Provides HTTP endpoint for Prometheus scraping.
    """
    
    def __init__(self, port: int = 9090, registry: Optional['CollectorRegistry'] = None):
        """
        Initialize Prometheus exporter.
        
        Args:
            port: HTTP port for metrics endpoint
            registry: Optional custom registry
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client required. Install: pip install prometheus-client")
        
        self.port = port
        self.registry = registry or CollectorRegistry()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        
        # Define metrics
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        
        # Training metrics
        self.training_loss = Gauge(
            'nawal_training_loss',
            'Training loss value',
            ['epoch'],
            registry=self.registry
        )
        
        self.training_accuracy = Gauge(
            'nawal_training_accuracy',
            'Training accuracy percentage',
            ['epoch'],
            registry=self.registry
        )
        
        self.validation_loss = Gauge(
            'nawal_validation_loss',
            'Validation loss value',
            ['epoch'],
            registry=self.registry
        )
        
        self.validation_accuracy = Gauge(
            'nawal_validation_accuracy',
            'Validation accuracy percentage',
            ['epoch'],
            registry=self.registry
        )
        
        self.epoch_time = Histogram(
            'nawal_epoch_duration_seconds',
            'Time per epoch in seconds',
            registry=self.registry
        )
        
        # Evolution metrics
        self.fitness_score = Gauge(
            'nawal_fitness_score',
            'Individual fitness score',
            ['generation'],
            registry=self.registry
        )
        
        self.best_fitness = Gauge(
            'nawal_best_fitness',
            'Best fitness in generation',
            ['generation'],
            registry=self.registry
        )
        
        self.average_fitness = Gauge(
            'nawal_average_fitness',
            'Average fitness in generation',
            ['generation'],
            registry=self.registry
        )
        
        self.generation_time = Histogram(
            'nawal_generation_duration_seconds',
            'Time per generation in seconds',
            registry=self.registry
        )
        
        # Federated learning metrics
        self.round_time = Histogram(
            'nawal_federated_round_duration_seconds',
            'Time per federated round in seconds',
            registry=self.registry
        )
        
        self.client_count = Gauge(
            'nawal_federated_clients',
            'Number of participating clients',
            ['round'],
            registry=self.registry
        )
        
        self.aggregation_time = Histogram(
            'nawal_aggregation_duration_seconds',
            'Time for model aggregation in seconds',
            registry=self.registry
        )
        
        self.communication_cost = Counter(
            'nawal_communication_bytes_total',
            'Total communication cost in bytes',
            registry=self.registry
        )
        
        # Blockchain metrics
        self.transactions_total = Counter(
            'nawal_blockchain_transactions_total',
            'Total blockchain transactions',
            ['status'],
            registry=self.registry
        )
        
        self.transaction_time = Histogram(
            'nawal_transaction_duration_seconds',
            'Transaction submission time in seconds',
            registry=self.registry
        )
        
        self.block_number = Gauge(
            'nawal_blockchain_block_number',
            'Current block number',
            registry=self.registry
        )
        
        self.finalization_time = Histogram(
            'nawal_finalization_duration_seconds',
            'Block finalization time in seconds',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'nawal_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'nawal_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'nawal_gpu_usage_percent',
            'GPU usage percentage',
            ['device'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'nawal_gpu_memory_percent',
            'GPU memory usage percentage',
            ['device'],
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'nawal_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
    
    def update_from_collector(self, collector: 'MetricsCollector') -> None:
        """
        Update Prometheus metrics from MetricsCollector.
        
        Args:
            collector: MetricsCollector instance
        """
        from .metrics import MetricType
        
        # Training metrics
        for metric_type in [MetricType.TRAINING_LOSS, MetricType.TRAINING_ACCURACY,
                           MetricType.VALIDATION_LOSS, MetricType.VALIDATION_ACCURACY]:
            latest = collector.get_latest(metric_type)
            if latest:
                epoch = latest.labels.get('epoch', '0')
                
                if metric_type == MetricType.TRAINING_LOSS:
                    self.training_loss.labels(epoch=epoch).set(latest.value)
                elif metric_type == MetricType.TRAINING_ACCURACY:
                    self.training_accuracy.labels(epoch=epoch).set(latest.value)
                elif metric_type == MetricType.VALIDATION_LOSS:
                    self.validation_loss.labels(epoch=epoch).set(latest.value)
                elif metric_type == MetricType.VALIDATION_ACCURACY:
                    self.validation_accuracy.labels(epoch=epoch).set(latest.value)
        
        # Epoch time
        epoch_time = collector.get_latest(MetricType.EPOCH_TIME)
        if epoch_time:
            self.epoch_time.observe(epoch_time.value)
        
        # Fitness metrics
        for metric_type in [MetricType.FITNESS_SCORE, MetricType.BEST_FITNESS,
                           MetricType.AVERAGE_FITNESS]:
            latest = collector.get_latest(metric_type)
            if latest:
                generation = latest.labels.get('generation', '0')
                
                if metric_type == MetricType.FITNESS_SCORE:
                    self.fitness_score.labels(generation=generation).set(latest.value)
                elif metric_type == MetricType.BEST_FITNESS:
                    self.best_fitness.labels(generation=generation).set(latest.value)
                elif metric_type == MetricType.AVERAGE_FITNESS:
                    self.average_fitness.labels(generation=generation).set(latest.value)
        
        # System metrics
        cpu = collector.get_latest(MetricType.CPU_USAGE)
        if cpu:
            self.cpu_usage.set(cpu.value)
        
        memory = collector.get_latest(MetricType.MEMORY_USAGE)
        if memory:
            self.memory_usage.set(memory.value)
        
        disk = collector.get_latest(MetricType.DISK_USAGE)
        if disk:
            self.disk_usage.set(disk.value)
        
        # GPU metrics
        gpu_usage = collector.get_latest(MetricType.GPU_USAGE)
        if gpu_usage:
            device = gpu_usage.labels.get('device', '0')
            self.gpu_usage.labels(device=device).set(gpu_usage.value)
        
        gpu_memory = collector.get_latest(MetricType.GPU_MEMORY)
        if gpu_memory:
            device = gpu_memory.labels.get('device', '0')
            self.gpu_memory.labels(device=device).set(gpu_memory.value)
    
    def start(self) -> None:
        """Start HTTP server for metrics endpoint."""
        if self._server is not None:
            return  # Already running
        
        registry = self.registry
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(generate_latest(registry))
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK')
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        # Bind to localhost by default for security (set NAWAL_METRICS_HOST=0.0.0.0 for cloud)
        bind_host = os.getenv('NAWAL_METRICS_HOST', '127.0.0.1')
        self._server = HTTPServer((bind_host, self.port), MetricsHandler)
        
        # Run in background thread
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
