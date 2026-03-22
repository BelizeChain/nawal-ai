"""
QuantumAnomalyDetector — Quantum kernel–based anomaly detection.

Three-tier routing
------------------
1. **Quantum (Kinich live)**
   Encodes telemetry as a quantum kernel feature map and runs a quantum
   kernel SVM on the Kinich node.

2. **Simulated quantum** (simulation_mode=True)
   Approximates a quantum kernel via a random Fourier feature map
   (Rahimi & Recht, 2007) — a recognised classical approximation to
   RBF quantum kernels.

3. **Classical fallback**
   Mahalanobis z-score on the fitted normal distribution.
   Falls back to per-feature z-score when covariance is singular.

Public API
----------
::

    from quantum.quantum_anomaly import QuantumAnomalyDetector
    import numpy as np

    det = QuantumAnomalyDetector(contamination=0.05)
    det.fit(normal_telemetry)          # np.ndarray shape (N, D)
    flags  = det.predict(new_samples)  # List[bool]  True = anomaly
    scores = det.score(new_samples)    # np.ndarray  [0, 1]

PhaseHook (Phase 5 Kinich live)
---------------------------------
Replace ``_quantum_kernel_score()`` body with a real Kinich HTTP call
that returns per-sample kernel distances.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# --------------------------------------------------------------------------- #
# QuantumAnomalyDetector                                                       #
# --------------------------------------------------------------------------- #


class QuantumAnomalyDetector:
    """
    Quantum kernel–based anomaly detector.

    Args:
        connector            : KinichQuantumConnector.
        fallback_to_classical: Degrade gracefully (default True).
        simulation_mode      : Use RFF quantum-kernel approximation
                               (default False).
        contamination        : Expected fraction of anomalies in test
                               data; used to set the score threshold
                               (default 0.05).
        n_qubits             : Quantum feature dimension (default 8).
        rff_components       : Random Fourier Feature count used in
                               simulated mode (default 256).
        random_state         : Reproducibility seed (default 42).
    """

    def __init__(
        self,
        connector: Optional[Any] = None,
        fallback_to_classical: bool = True,
        simulation_mode: bool = False,
        contamination: float = 0.05,
        n_qubits: int = 8,
        rff_components: int = 256,
        random_state: int = 42,
    ) -> None:
        self._connector = connector
        self.fallback_to_classical = fallback_to_classical
        self.simulation_mode = simulation_mode
        self.contamination = contamination
        self.n_qubits = n_qubits
        self.rff_components = rff_components

        self._rng = np.random.default_rng(random_state)
        self._fitted = False

        # Fit artefacts (classical path)
        self._mean_vec: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None  # precision matrix
        self._stds: Optional[np.ndarray] = None  # per-feature stds (fallback)
        self._threshold: float = 0.5

        # Fit artefacts (RFF path)
        self._rff_weights: Optional[np.ndarray] = None  # (D, rff_components)
        self._rff_biases: Optional[np.ndarray] = None  # (rff_components,)
        self._rff_mean: Optional[np.ndarray] = None  # mean RFF of normal data
        self._rff_gamma: float = 1.0

        self.stats: Dict[str, int] = {
            "quantum_calls": 0,
            "simulated_calls": 0,
            "classical_calls": 0,
        }

        logger.info(
            f"QuantumAnomalyDetector ready: simulation_mode={simulation_mode} "
            f"contamination={contamination} n_qubits={n_qubits}"
        )

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, normal_telemetry: np.ndarray) -> "QuantumAnomalyDetector":
        """
        Fit the detector on normal (non-anomalous) telemetry.

        Args:
            normal_telemetry: np.ndarray of shape (N, D).

        Returns:
            self
        """
        X = np.asarray(normal_telemetry, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] == 0:
            raise ValueError("normal_telemetry must not be empty")

        n, d = X.shape

        # Classical statistics
        self._mean_vec = X.mean(axis=0)
        self._stds = X.std(axis=0) + 1e-9  # per-feature fallback

        if n > d:
            cov = np.cov(X, rowvar=False)
            try:
                self._cov_inv = np.linalg.inv(cov + np.eye(d) * 1e-6)
            except np.linalg.LinAlgError:
                self._cov_inv = None
        else:
            self._cov_inv = None  # Not enough samples for full covariance

        # RFF artefacts for simulated quantum kernel
        gamma = 1.0 / d
        self._rff_gamma = gamma
        self._rff_weights = self._rng.normal(
            0, np.sqrt(2 * gamma), (d, self.rff_components)
        )
        self._rff_biases = self._rng.uniform(0, 2 * np.pi, self.rff_components)
        phi_X = self._rff_transform(X)
        self._rff_mean = phi_X.mean(axis=0)

        # Calibrate threshold so approximately contamination fraction are flagged
        raw_scores = self._compute_scores(X, mode="auto")
        thresh_idx = max(0, int(np.floor(n * (1.0 - self.contamination))) - 1)
        sorted_scores = np.sort(raw_scores)
        self._threshold = float(sorted_scores[thresh_idx])

        self._fitted = True

        logger.info(
            f"QuantumAnomalyDetector.fit: n={n} d={d} "
            f"threshold={self._threshold:.4f}"
        )
        return self

    # ------------------------------------------------------------------ #
    # Predict / Score                                                      #
    # ------------------------------------------------------------------ #

    def predict(self, telemetry: np.ndarray) -> List[bool]:
        """
        Classify samples as anomalous (True) or normal (False).

        Args:
            telemetry: np.ndarray of shape (N, D) or (D,).

        Returns:
            List[bool] of length N.
        """
        scores = self.score(telemetry)
        return [float(s) > self._threshold for s in scores]

    def score(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, ∞).  Higher = more anomalous.

        Args:
            telemetry: np.ndarray of shape (N, D) or (D,).

        Returns:
            np.ndarray of shape (N,).
        """
        if not self._fitted:
            raise RuntimeError("Detector is not fitted — call fit() first")

        X = np.asarray(telemetry, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        t0 = time.perf_counter()

        if self._should_use_quantum():
            scores = self._quantum_score(X)
            self.stats["quantum_calls"] += len(X)
            mode = "quantum"
        elif self.simulation_mode:
            scores = self._simulated_kernel_score(X)
            self.stats["simulated_calls"] += len(X)
            mode = "simulated_rff"
        else:
            scores = self._mahalanobis_score(X)
            self.stats["classical_calls"] += len(X)
            mode = "mahalanobis"

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"QuantumAnomalyDetector.score: mode={mode} n={len(X)} "
            f"elapsed={elapsed*1000:.1f}ms"
        )
        return scores

    # ------------------------------------------------------------------ #
    # Score implementations                                                #
    # ------------------------------------------------------------------ #

    def _compute_scores(self, X: np.ndarray, mode: str = "auto") -> np.ndarray:
        """Helper used during fit for threshold calibration."""
        if mode == "simulated" or (mode == "auto" and self.simulation_mode):
            return self._simulated_kernel_score(X)
        return self._mahalanobis_score(X)

    def _mahalanobis_score(self, X: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance–based anomaly score.
        Falls back to normalised z-score when covariance is unavailable.
        """
        diff = X - self._mean_vec

        if self._cov_inv is not None:
            # Mahalanobis distance squared
            maha = np.einsum("nd,dd,nd->n", diff, self._cov_inv, diff)
            return np.sqrt(np.maximum(maha, 0.0))
        else:
            # Per-feature z-score (L2 norm of standardised residuals)
            z = diff / self._stds
            return np.linalg.norm(z, axis=1)

    def _simulated_kernel_score(self, X: np.ndarray) -> np.ndarray:
        """
        Simulated quantum kernel score via Random Fourier Features.

        Score = distance in RFF space from the normal-data centroid.
        """
        phi_X = self._rff_transform(X)
        diff = phi_X - self._rff_mean
        return np.linalg.norm(diff, axis=1)

    def _quantum_score(self, X: np.ndarray) -> np.ndarray:
        """
        Real quantum kernel score via Kinich connector.

        PhaseHook — Phase 5: replace with Kinich HTTP call.
        """
        logger.debug(
            "QuantumAnomalyDetector: Kinich live — using RFF proxy (Phase 5 TBD)"
        )
        return self._simulated_kernel_score(X)

    # ------------------------------------------------------------------ #
    # RFF kernel approximation                                             #
    # ------------------------------------------------------------------ #

    def _rff_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Random Fourier Feature transform: φ(x) = √(2/D) · cos(xW + b).
        Approximates an RBF kernel k(x,y) ≈ φ(x)·φ(y).
        """
        projection = X @ self._rff_weights + self._rff_biases  # (N, rff_components)
        return np.sqrt(2.0 / self.rff_components) * np.cos(projection)

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def _should_use_quantum(self) -> bool:
        if self._connector is None:
            return False
        return bool(getattr(self._connector, "kinich_available", False))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.stats.values())
        return {
            **self.stats,
            "total_calls": total,
            "threshold": self._threshold,
            "fitted": self._fitted,
            "quantum_ratio": (
                self.stats["quantum_calls"] / total if total > 0 else 0.0
            ),
        }
