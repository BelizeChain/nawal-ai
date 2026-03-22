"""
Kinich Quantum Connector

Connects Nawal's classical ML models to Kinich's quantum processors.
"""

import numpy as np
from loguru import logger
from typing import Optional, Dict, Any, List, Tuple
import asyncio

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - Kinich connector will be limited")


class KinichQuantumConnector:
    """
    Connector between Nawal (classical ML) and Kinich (quantum ML).

    Enables Nawal models to offload feature processing to Kinich's
    quantum neural networks for quantum-enhanced inference.

    Workflow:
    1. Nawal extracts classical features from data
    2. Connector encodes features for quantum processing
    3. Kinich processes via QNN (VQC, QSVM, etc.)
    4. Connector decodes quantum results
    5. Nawal uses quantum-enhanced predictions

    Example:
        >>> connector = KinichQuantumConnector(
        ...     kinich_endpoint="http://localhost:8002",
        ...     classical_dim=768,
        ...     quantum_dim=8
        ... )
        >>>
        >>> # Extract features from Nawal model
        >>> features = nawal_model.encode(text_data)  # [batch, 768]
        >>>
        >>> # Process via Kinich quantum
        >>> quantum_enhanced = await connector.quantum_process(features)
        >>>
        >>> # Use in Nawal predictions
        >>> predictions = nawal_model.decode(quantum_enhanced)
    """

    def __init__(
        self,
        kinich_endpoint: str = "http://localhost:8002",
        classical_dim: int = 768,
        quantum_dim: int = 8,
        enable_caching: bool = True,
        fallback_to_classical: bool = True,
        request_timeout: float = 30.0,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize Kinich quantum connector.

        Args:
            kinich_endpoint: URL of Kinich quantum API
            classical_dim: Dimension of Nawal's classical features
            quantum_dim: Dimension of quantum features (number of qubits)
            enable_caching: Cache quantum results for repeated inputs
            fallback_to_classical: Fall back to classical if quantum unavailable
            request_timeout: HTTP request timeout in seconds
            circuit_breaker_threshold: Consecutive failures before disabling quantum path
        """
        self.kinich_endpoint = kinich_endpoint
        self.classical_dim = classical_dim
        self.quantum_dim = quantum_dim
        self.enable_caching = enable_caching
        self.fallback_to_classical = fallback_to_classical
        self.request_timeout = request_timeout

        # Lazy import Kinich components
        self.bridge = None
        self.kinich_available = False

        # Circuit breaker state
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False

        # Cache for quantum results (bounded to prevent memory leaks)
        self._cache_max_size: int = 1024
        self.result_cache: Dict[str, np.ndarray] = {}

        # Statistics
        self.stats = {
            "quantum_calls": 0,
            "cache_hits": 0,
            "fallback_calls": 0,
            "total_latency": 0.0,
        }

        # Initialize connection
        self._init_kinich_connection()

        logger.info(
            f"Initialized KinichQuantumConnector: "
            f"classical_dim={classical_dim}, quantum_dim={quantum_dim}"
        )

    def _init_kinich_connection(self) -> None:
        """Initialize connection to Kinich quantum backend (sync-safe)."""
        try:
            import os
            import urllib.request
            import urllib.error

            kinich_api_url = os.getenv("KINICH_API_URL", self.kinich_endpoint)

            # Synchronous health check — safe to call from any context
            try:
                req = urllib.request.Request(
                    f"{kinich_api_url}/api/v1/health",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.kinich_available = resp.status == 200
                if self.kinich_available:
                    logger.info(
                        f"Connected to Kinich quantum backend at {kinich_api_url}"
                    )
                else:
                    logger.warning("Kinich health check failed")
            except (urllib.error.URLError, OSError) as health_error:
                logger.warning(f"Kinich health check error: {health_error}")
                self.kinich_available = False

        except Exception as e:
            logger.warning(f"Kinich not available: {e}")
            self.kinich_available = False
            if not self.fallback_to_classical:
                raise

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate and sanitize input features."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D [batch, {self.classical_dim}], got shape {features.shape}"
            )
        if features.shape[1] != self.classical_dim:
            raise ValueError(
                f"features dim {features.shape[1]} != classical_dim {self.classical_dim}"
            )
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("features contain NaN or Inf values")
        return features

    def _is_quantum_available(self) -> bool:
        """Check if quantum path is available (health + circuit breaker)."""
        return self.kinich_available and not self._circuit_open

    def _record_quantum_success(self) -> None:
        """Record a successful quantum call, resetting the circuit breaker."""
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            logger.info("Circuit breaker closed — Kinich recovered")

    def _record_quantum_failure(self) -> None:
        """Record a quantum failure, potentially opening the circuit breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            self._circuit_open = True
            logger.warning(
                f"Circuit breaker OPEN after {self._consecutive_failures} "
                f"consecutive failures — skipping Kinich calls"
            )

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (e.g. after Kinich restart)."""
        self._consecutive_failures = 0
        self._circuit_open = False
        logger.info("Circuit breaker manually reset")

    async def quantum_process(
        self, features: np.ndarray, model_type: str = "vqc", **kwargs
    ) -> np.ndarray:
        """
        Process features through Kinich quantum neural networks.

        Args:
            features: Classical features from Nawal [batch_size, classical_dim]
            model_type: Quantum model type ('vqc', 'qsvm', 'qnn')
            **kwargs: Additional model-specific parameters

        Returns:
            Quantum-enhanced features [batch_size, classical_dim]
        """
        import time

        start_time = time.time()

        # Validate input
        features = self._validate_features(features)

        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(features)
            if cache_key in self.result_cache:
                self.stats["cache_hits"] += 1
                logger.debug("Cache hit - returning cached result")
                return self.result_cache[cache_key]

        # Process via Kinich (circuit breaker guards the quantum path)
        if self._is_quantum_available():
            try:
                result = await self._quantum_forward(features, model_type, **kwargs)
                self.stats["quantum_calls"] += 1
                self._record_quantum_success()
            except Exception as e:
                logger.warning(f"Quantum processing failed: {e}")
                self._record_quantum_failure()
                if self.fallback_to_classical:
                    result = self._classical_fallback(features)
                    self.stats["fallback_calls"] += 1
                else:
                    raise
        else:
            # Fallback to classical
            result = self._classical_fallback(features)
            self.stats["fallback_calls"] += 1

        # Update statistics
        latency = time.time() - start_time
        self.stats["total_latency"] += latency

        # Cache result
        if self.enable_caching:
            # Evict oldest entries if cache is full
            if len(self.result_cache) >= self._cache_max_size:
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            self.result_cache[cache_key] = result

        return result

    async def _quantum_forward(
        self, features: np.ndarray, model_type: str, **kwargs
    ) -> np.ndarray:
        """
        Execute quantum forward pass via Kinich HTTP API.

        Steps:
        1. Send features to Kinich API
        2. Kinich processes via quantum neural network
        3. Receive quantum-enhanced results
        """
        import aiohttp

        # Prepare request payload
        payload = {
            "features": features.tolist(),
            "model_type": model_type,
            "classical_dim": self.classical_dim,
            "quantum_dim": self.quantum_dim,
            **kwargs,
        }

        # Send to Kinich API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.kinich_endpoint}/api/v1/qml/process",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Kinich API error: {error_text}")

                result = await resp.json()

                if "quantum_enhanced_features" not in result:
                    raise RuntimeError(
                        f"Kinich response missing 'quantum_enhanced_features' key, "
                        f"got keys: {list(result.keys())}"
                    )

                classical_results = np.array(result["quantum_enhanced_features"])

                # Validate response shape matches input
                if classical_results.shape != features.shape:
                    raise RuntimeError(
                        f"Kinich response shape {classical_results.shape} != "
                        f"input shape {features.shape}"
                    )

        return classical_results

    async def _vqc_forward(self, quantum_features: np.ndarray, **kwargs) -> np.ndarray:
        """Process via Variational Quantum Classifier (HTTP API)."""
        # VQC is handled by _quantum_forward HTTP call
        # This method is kept for backward compatibility
        logger.warning("_vqc_forward called directly - use _quantum_forward instead")
        return await self._quantum_forward(quantum_features, "vqc", **kwargs)

    async def _qsvm_forward(self, quantum_features: np.ndarray, **kwargs) -> np.ndarray:
        """Process via Quantum Support Vector Machine."""
        # Placeholder for QSVM (implemented in Phase 3)
        logger.warning("QSVM not yet implemented - using mock")

        # Mock QSVM output
        batch_size = quantum_features.shape[0]
        return np.random.rand(batch_size, kwargs.get("num_classes", 2))

    async def _qnn_forward(self, quantum_features: np.ndarray, **kwargs) -> np.ndarray:
        """Process via Quantum Neural Network (HTTP API)."""
        # QNN is handled by _quantum_forward HTTP call
        # This method is kept for backward compatibility
        logger.warning("_qnn_forward called directly - use _quantum_forward instead")
        return await self._quantum_forward(quantum_features, "qnn", **kwargs)

    def _classical_fallback(self, features: np.ndarray) -> np.ndarray:
        """
        Fallback to classical processing when quantum unavailable.

        Uses simple classical transformation to maintain pipeline.
        """
        logger.debug("Using classical fallback")

        # Simple classical transformation (PCA-like)
        # In production, this would use trained classical model
        if not hasattr(self, "_fallback_matrix"):
            rng = np.random.default_rng(seed=42)
            self._fallback_matrix = (
                rng.standard_normal((self.classical_dim, self.classical_dim)) * 0.01
            )

        result = features @ self._fallback_matrix.T
        return result

    def _get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key for features."""
        # Use hash of feature values
        return str(hash(features.tobytes()))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get connector statistics.

        Returns:
            Dictionary with usage statistics
        """
        total_calls = self.stats["quantum_calls"] + self.stats["fallback_calls"]

        return {
            "total_calls": total_calls,
            "quantum_calls": self.stats["quantum_calls"],
            "cache_hits": self.stats["cache_hits"],
            "fallback_calls": self.stats["fallback_calls"],
            "quantum_ratio": (
                self.stats["quantum_calls"] / total_calls if total_calls > 0 else 0.0
            ),
            "cache_hit_ratio": (
                self.stats["cache_hits"] / total_calls if total_calls > 0 else 0.0
            ),
            "avg_latency": (
                self.stats["total_latency"] / total_calls if total_calls > 0 else 0.0
            ),
            "kinich_available": self.kinich_available,
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
        }

    def clear_cache(self) -> None:
        """Clear result cache."""
        self.result_cache.clear()
        logger.info("Cache cleared")

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.stats = {
            "quantum_calls": 0,
            "cache_hits": 0,
            "fallback_calls": 0,
            "total_latency": 0.0,
        }
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"KinichQuantumConnector("
            f"endpoint={self.kinich_endpoint}, "
            f"dims={self.classical_dim}→{self.quantum_dim}, "
            f"quantum_calls={stats['quantum_calls']}, "
            f"available={self.kinich_available})"
        )


class QuantumEnhancedLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch layer that uses Kinich quantum processing.

    Can be inserted into Nawal's neural network architectures
    to add quantum enhancement.

    Example:
        >>> class HybridModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.classical_encoder = nn.Linear(768, 768)
        ...         self.quantum_layer = QuantumEnhancedLayer(768, 8)
        ...         self.classifier = nn.Linear(768, 10)
        ...
        ...     def forward(self, x):
        ...         x = self.classical_encoder(x)
        ...         x = self.quantum_layer(x)  # Quantum enhancement
        ...         return self.classifier(x)
    """

    def __init__(self, classical_dim: int, quantum_dim: int, model_type: str = "vqc"):
        """
        Initialize quantum-enhanced layer.

        Args:
            classical_dim: Input/output feature dimension
            quantum_dim: Quantum feature dimension (number of qubits)
            model_type: Quantum model type
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for QuantumEnhancedLayer")

        super().__init__()

        self.classical_dim = classical_dim
        self.quantum_dim = quantum_dim
        self.model_type = model_type

        # Create connector
        self.connector = KinichQuantumConnector(
            classical_dim=classical_dim, quantum_dim=quantum_dim
        )

        logger.info(
            f"Initialized QuantumEnhancedLayer: "
            f"{classical_dim}→{quantum_dim}→{classical_dim}"
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass with quantum enhancement.

        Args:
            x: Input tensor [batch_size, classical_dim]

        Returns:
            Quantum-enhanced tensor [batch_size, classical_dim]
        """
        # Convert to numpy
        x_np = x.detach().cpu().numpy()

        # Process via Kinich — run coroutine safely from sync or async context
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop — schedule in a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result_np = pool.submit(
                    asyncio.run,
                    self.connector.quantum_process(x_np, model_type=self.model_type),
                ).result()
        else:
            result_np = asyncio.run(
                self.connector.quantum_process(x_np, model_type=self.model_type)
            )

        # Convert back to tensor
        result = torch.from_numpy(result_np).to(x.device).to(x.dtype)

        return result

    def extra_repr(self) -> str:
        return f"classical_dim={self.classical_dim}, quantum_dim={self.quantum_dim}"
