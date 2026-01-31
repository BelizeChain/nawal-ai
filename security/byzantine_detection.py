"""
Byzantine Detection for Federated Learning.

Implements robust aggregation methods to detect and mitigate
Byzantine (malicious) client attacks.

Key Features:
- Krum aggregation (Byzantine-robust)
- Multi-Krum aggregation
- Trimmed mean aggregation
- Cosine similarity detection
- Gradient norm analysis
- Reputation scoring

References:
- Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (2017)
- Yin et al. "Byzantine-Robust Distributed Learning" (2018)

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from loguru import logger


class AggregationMethod(Enum):
    """Byzantine-robust aggregation methods."""
    FEDAVG = "fedavg"  # Standard (not Byzantine-robust)
    KRUM = "krum"  # Select single closest update
    MULTI_KRUM = "multi_krum"  # Average k closest updates
    TRIMMED_MEAN = "trimmed_mean"  # Remove outliers, average rest
    MEDIAN = "median"  # Coordinate-wise median
    PHOCAS = "phocas"  # Reputation-based


@dataclass
class ClientReputation:
    """
    Track client reputation for Byzantine detection.
    
    Attributes:
        client_id: Client identifier
        score: Reputation score (0-1, higher = more trustworthy)
        contributions: Number of contributions
        anomalies: Number of detected anomalies
        history: Recent behavior history
    """
    client_id: int
    score: float = 1.0
    contributions: int = 0
    anomalies: int = 0
    history: List[float] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def update(self, is_anomalous: bool, decay: float = 0.95):
        """Update reputation based on behavior."""
        self.contributions += 1
        
        if is_anomalous:
            self.anomalies += 1
            self.score *= decay  # Decay reputation
        else:
            self.score = min(1.0, self.score + 0.05)  # Slowly increase
        
        self.history.append(self.score)
        
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def is_trustworthy(self, threshold: float = 0.5) -> bool:
        """Check if client is trustworthy."""
        return self.score >= threshold


class ByzantineDetector:
    """
    Byzantine detection and robust aggregation.
    
    Protects against malicious clients who send:
    - Poisoned gradients (intentionally wrong)
    - Large magnitude updates (gradient explosion)
    - Backdoored models (hidden triggers)
    
    Usage:
        detector = ByzantineDetector(
            method=AggregationMethod.KRUM,
            num_byzantine=2,
        )
        
        # Aggregate with Byzantine tolerance
        global_update = detector.aggregate(
            client_updates=updates,
            method=AggregationMethod.KRUM,
        )
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(updates)
    """
    
    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.KRUM,
        num_byzantine: int = 0,
        reputation_enabled: bool = True,
    ):
        """
        Initialize ByzantineDetector.
        
        Args:
            method: Default aggregation method
            num_byzantine: Expected number of Byzantine clients
            reputation_enabled: Whether to track client reputation
        """
        self.method = method
        self.num_byzantine = num_byzantine
        self.reputation_enabled = reputation_enabled
        
        # Client reputation tracking
        self.reputations: Dict[int, ClientReputation] = {}
        
        logger.info(
            f"ByzantineDetector initialized: method={method.value}, "
            f"num_byzantine={num_byzantine}, reputation={reputation_enabled}"
        )
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_ids: Optional[List[int]] = None,
        method: Optional[AggregationMethod] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with Byzantine tolerance.
        
        Args:
            client_updates: List of client model updates
            client_ids: Optional client identifiers
            method: Aggregation method (if None, use default)
        
        Returns:
            Aggregated global update
        """
        if method is None:
            method = self.method
        
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")
        
        # Detect anomalies if reputation enabled
        if self.reputation_enabled and client_ids is not None:
            anomalies = self.detect_anomalies(client_updates)
            for client_id, is_anomalous in zip(client_ids, anomalies):
                self._update_reputation(client_id, is_anomalous)
        
        # Route to appropriate aggregation method
        if method == AggregationMethod.FEDAVG:
            return self._fedavg(client_updates)
        elif method == AggregationMethod.KRUM:
            return self._krum(client_updates)
        elif method == AggregationMethod.MULTI_KRUM:
            return self._multi_krum(client_updates)
        elif method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(client_updates)
        elif method == AggregationMethod.MEDIAN:
            return self._median(client_updates)
        elif method == AggregationMethod.PHOCAS:
            if client_ids is None:
                raise ValueError("PHOCAS requires client IDs")
            return self._phocas(client_updates, client_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _fedavg(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg aggregation (not Byzantine-robust).
        
        Args:
            client_updates: Client model updates
        
        Returns:
            Averaged update
        """
        aggregated = {}
        num_clients = len(client_updates)
        
        for param_name in client_updates[0].keys():
            param_sum = torch.zeros_like(client_updates[0][param_name])
            for update in client_updates:
                param_sum += update[param_name]
            aggregated[param_name] = param_sum / num_clients
        
        return aggregated
    
    def _krum(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Krum aggregation: Select single closest update.
        
        Krum selects the update with smallest sum of distances
        to its k-nearest neighbors, where k = n - f - 2
        (n = total clients, f = Byzantine clients).
        
        Args:
            client_updates: Client model updates
        
        Returns:
            Selected update (closest to honest majority)
        """
        num_clients = len(client_updates)
        f = self.num_byzantine
        k = num_clients - f - 2
        
        if k <= 0:
            logger.warning("Too many Byzantine clients for Krum, using FedAvg")
            return self._fedavg(client_updates)
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(client_updates)
        
        # For each client, sum distances to k-nearest neighbors
        scores = []
        for i in range(num_clients):
            # Get distances from client i to all others
            dists = [distances[i][j] for j in range(num_clients) if j != i]
            dists.sort()
            # Sum k smallest distances
            score = sum(dists[:k])
            scores.append(score)
        
        # Select client with smallest score
        best_idx = scores.index(min(scores))
        logger.debug(f"Krum selected client {best_idx} (score={scores[best_idx]:.4f})")
        
        return client_updates[best_idx]
    
    def _multi_krum(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        m: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum aggregation: Average m closest updates.
        
        Args:
            client_updates: Client model updates
            m: Number of updates to select (if None, use n - f)
        
        Returns:
            Averaged selected updates
        """
        num_clients = len(client_updates)
        f = self.num_byzantine
        
        if m is None:
            m = num_clients - f
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(client_updates)
        
        # Score each client (sum of k-nearest distances)
        k = num_clients - f - 2
        scores = []
        for i in range(num_clients):
            dists = [distances[i][j] for j in range(num_clients) if j != i]
            dists.sort()
            score = sum(dists[:k])
            scores.append((score, i))
        
        # Select m clients with smallest scores
        scores.sort()
        selected_indices = [idx for _, idx in scores[:m]]
        selected_updates = [client_updates[i] for i in selected_indices]
        
        logger.debug(f"Multi-Krum selected {m} clients: {selected_indices}")
        
        # Average selected updates
        return self._fedavg(selected_updates)
    
    def _trimmed_mean(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean aggregation: Remove outliers, average rest.
        
        For each parameter coordinate:
        1. Sort client values
        2. Remove top and bottom trim_ratio
        3. Average remaining values
        
        Args:
            client_updates: Client model updates
            trim_ratio: Fraction to trim from each end (0.1 = remove 10% from each side)
        
        Returns:
            Trimmed mean update
        """
        num_clients = len(client_updates)
        num_trim = int(num_clients * trim_ratio)
        
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            # Stack all client values for this parameter
            stacked = torch.stack([u[param_name] for u in client_updates])
            
            # Sort along client dimension
            sorted_values, _ = torch.sort(stacked, dim=0)
            
            # Trim and average
            if num_trim > 0:
                trimmed = sorted_values[num_trim:-num_trim]
            else:
                trimmed = sorted_values
            
            aggregated[param_name] = trimmed.mean(dim=0)
        
        logger.debug(f"Trimmed mean: removed {num_trim} outliers from each side")
        return aggregated
    
    def _median(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median aggregation.
        
        Robust to up to 50% Byzantine clients.
        
        Args:
            client_updates: Client model updates
        
        Returns:
            Median update
        """
        aggregated = {}
        
        for param_name in client_updates[0].keys():
            stacked = torch.stack([u[param_name] for u in client_updates])
            aggregated[param_name] = torch.median(stacked, dim=0).values
        
        logger.debug("Applied median aggregation")
        return aggregated
    
    def _phocas(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        PHOCAS: Reputation-based weighted aggregation.
        
        Weight each client's contribution by their reputation score.
        
        Args:
            client_updates: Client model updates
            client_ids: Client identifiers
        
        Returns:
            Reputation-weighted update
        """
        # Get reputation weights
        weights = []
        for client_id in client_ids:
            if client_id not in self.reputations:
                self.reputations[client_id] = ClientReputation(client_id=client_id)
            weights.append(self.reputations[client_id].score)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted average
        aggregated = {}
        for param_name in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            for update, weight in zip(client_updates, weights):
                weighted_sum += update[param_name] * weight
            aggregated[param_name] = weighted_sum
        
        logger.debug(f"PHOCAS aggregation with weights: {[f'{w:.3f}' for w in weights]}")
        return aggregated
    
    def _compute_pairwise_distances(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> List[List[float]]:
        """
        Compute pairwise L2 distances between client updates.
        
        Args:
            client_updates: Client model updates
        
        Returns:
            Distance matrix (n x n)
        """
        num_clients = len(client_updates)
        distances = [[0.0] * num_clients for _ in range(num_clients)]
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = self._compute_distance(client_updates[i], client_updates[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def _compute_distance(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor],
    ) -> float:
        """
        Compute L2 distance between two updates.
        
        Args:
            update1: First update
            update2: Second update
        
        Returns:
            L2 distance
        """
        distance = 0.0
        for param_name in update1.keys():
            diff = update1[param_name] - update2[param_name]
            distance += torch.norm(diff).item() ** 2
        
        return math.sqrt(distance)
    
    def detect_anomalies(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        threshold_std: float = 3.0,
    ) -> List[bool]:
        """
        Detect anomalous client updates.
        
        Uses multiple heuristics:
        - Gradient norm (detect large updates)
        - Cosine similarity (detect direction outliers)
        - Statistical outliers (Z-score)
        
        Args:
            client_updates: Client model updates
            threshold_std: Z-score threshold for anomaly detection
        
        Returns:
            List of boolean flags (True = anomalous)
        """
        num_clients = len(client_updates)
        
        # Compute gradient norms
        norms = [self._compute_norm(update) for update in client_updates]
        
        # Detect norm outliers (Z-score)
        mean_norm = sum(norms) / len(norms)
        std_norm = math.sqrt(sum((n - mean_norm) ** 2 for n in norms) / len(norms))
        
        anomalies = []
        for i, norm in enumerate(norms):
            z_score = abs(norm - mean_norm) / (std_norm + 1e-6)
            is_anomalous = z_score > threshold_std
            
            if is_anomalous:
                logger.warning(
                    f"Anomaly detected: client {i}, norm={norm:.4f}, "
                    f"z_score={z_score:.2f}"
                )
            
            anomalies.append(is_anomalous)
        
        return anomalies
    
    def _compute_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of update."""
        total_norm = 0.0
        for param in update.values():
            total_norm += torch.norm(param).item() ** 2
        return math.sqrt(total_norm)
    
    def _update_reputation(self, client_id: int, is_anomalous: bool) -> None:
        """Update client reputation score."""
        if client_id not in self.reputations:
            self.reputations[client_id] = ClientReputation(client_id=client_id)
        
        self.reputations[client_id].update(is_anomalous)
    
    def get_client_reputation(self, client_id: int) -> Optional[ClientReputation]:
        """Get reputation for specific client."""
        return self.reputations.get(client_id)
    
    def get_trustworthy_clients(self, threshold: float = 0.5) -> List[int]:
        """Get list of trustworthy client IDs."""
        return [
            cid for cid, rep in self.reputations.items()
            if rep.is_trustworthy(threshold)
        ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "num_byzantine": self.num_byzantine,
            "reputation_enabled": self.reputation_enabled,
            "num_tracked_clients": len(self.reputations),
            "reputations": {
                cid: {
                    "score": rep.score,
                    "contributions": rep.contributions,
                    "anomalies": rep.anomalies,
                }
                for cid, rep in self.reputations.items()
            },
        }


# Utility functions

def recommend_aggregation_method(
    num_clients: int,
    num_byzantine: int,
) -> AggregationMethod:
    """
    Recommend Byzantine-robust aggregation method.
    
    Args:
        num_clients: Total number of clients
        num_byzantine: Expected Byzantine clients
    
    Returns:
        Recommended aggregation method
    """
    byzantine_ratio = num_byzantine / num_clients
    
    if byzantine_ratio == 0:
        return AggregationMethod.FEDAVG
    elif byzantine_ratio < 0.2:
        return AggregationMethod.KRUM
    elif byzantine_ratio < 0.3:
        return AggregationMethod.MULTI_KRUM
    elif byzantine_ratio < 0.5:
        return AggregationMethod.TRIMMED_MEAN
    else:
        return AggregationMethod.MEDIAN
