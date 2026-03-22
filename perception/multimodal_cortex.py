"""
MultimodalCortex — fuse embeddings from multiple sensory cortices into a
single world-state vector.

Fusion strategies
-----------------
"mean"       : Mean-pool all present embeddings (no projection needed).
"weighted"   : Weighted mean using per-modality importance weights.
"projection" : Learned linear projection (requires training). Falls back to
               "mean" if not trained.

The output always has the same dimensionality (``hidden_dim``) regardless
of which modalities are present.  Absent modalities contribute nothing.

PhaseHook
---------
Phase 5a: train the cross-attention projection on Belize dataset.
Replace ``_project_fuse()`` — the public ``fuse()`` / ``encode()`` API
is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from perception.interfaces import WorldState
from perception.text_cortex import _l2_normalize, _project

# --------------------------------------------------------------------------- #
# Learned projector (optional, used only if weights loaded)                   #
# --------------------------------------------------------------------------- #


class _ModalityProjector(nn.Module):
    """
    Small MLP that projects any-dim embedding into ``hidden_dim``.

    Architecture: Linear(in_dim, hidden_dim) → LayerNorm → GELU → Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# MultimodalCortex                                                             #
# --------------------------------------------------------------------------- #


class MultimodalCortex:
    """
    Fuse text, image, and audio embeddings into a single world-state vector.

    Args:
        hidden_dim     : Output dimension (should match Nawal transformer `d_model`).
        fusion_strategy: "mean" | "weighted" | "projection".
        text_weight    : Relative importance of textual embedding  (weighted mode).
        image_weight   : Relative importance of image embedding    (weighted mode).
        audio_weight   : Relative importance of audio embedding    (weighted mode).
        device         : "cpu", "cuda", or "auto".
    """

    STRATEGIES = {"mean", "weighted", "projection"}

    def __init__(
        self,
        hidden_dim: int = 256,
        fusion_strategy: str = "weighted",
        text_weight: float = 1.0,
        image_weight: float = 0.8,
        audio_weight: float = 0.7,
        device: str = "auto",
    ) -> None:
        if fusion_strategy not in self.STRATEGIES:
            raise ValueError(
                f"fusion_strategy must be one of {self.STRATEGIES}, " f"got {fusion_strategy!r}"
            )
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.audio_weight = audio_weight
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        # Lazy-initialised projectors (used only in "projection" mode)
        self._projectors: dict[str, _ModalityProjector | None] = {
            "text": None,
            "image": None,
            "audio": None,
        }
        self._projectors_trained = False

        logger.debug(f"MultimodalCortex init: dim={hidden_dim} " f"strategy={fusion_strategy}")

    # ------------------------------------------------------------------ #
    # Primary API                                                          #
    # ------------------------------------------------------------------ #

    def fuse(self, world_state: WorldState) -> list[float]:
        """
        Fuse all embedding fields of *world_state* into a single vector.

        Args:
            world_state : WorldState with any combination of
                          text_embedding, image_embedding, audio_embedding.

        Returns:
            List[float] of length ``hidden_dim``, L2-normalised.
            Returns zero vector if no embeddings are present.
        """
        embeddings: dict[str, list[float]] = {}
        weights: dict[str, float] = {}

        if world_state.text_embedding is not None:
            embeddings["text"] = world_state.text_embedding
            weights["text"] = self.text_weight
        if world_state.image_embedding is not None:
            embeddings["image"] = world_state.image_embedding
            weights["image"] = self.image_weight
        if world_state.audio_embedding is not None:
            embeddings["audio"] = world_state.audio_embedding
            weights["audio"] = self.audio_weight

        if not embeddings:
            logger.warning("MultimodalCortex.fuse: no embeddings available")
            return [0.0] * self.hidden_dim

        if self.fusion_strategy == "projection" and self._projectors_trained:
            fused = self._project_fuse(embeddings)
        elif self.fusion_strategy == "weighted":
            fused = self._weighted_fuse(embeddings, weights)
        else:
            fused = self._mean_fuse(embeddings)

        return _l2_normalize(fused)

    def update_world_state(self, world_state: WorldState) -> WorldState:
        """
        Fuse and store the result in ``world_state.fused_embedding``.

        Returns the same WorldState object (mutated in-place).
        """
        world_state.fused_embedding = self.fuse(world_state)
        return world_state

    # ------------------------------------------------------------------ #
    # Fusion implementations                                               #
    # ------------------------------------------------------------------ #

    def _mean_fuse(self, embeddings: dict[str, list[float]]) -> list[float]:
        """Simple mean of all embeddings, projected to hidden_dim."""
        acc = [0.0] * self.hidden_dim
        for vec in embeddings.values():
            projected = _maybe_project(vec, self.hidden_dim)
            for i, v in enumerate(projected):
                acc[i] += v
        n = len(embeddings)
        return [x / n for x in acc]

    def _weighted_fuse(
        self,
        embeddings: dict[str, list[float]],
        weights: dict[str, float],
    ) -> list[float]:
        """Weighted mean, normalised by total weight of present modalities."""
        total_w = sum(weights.get(k, 1.0) for k in embeddings)
        acc = [0.0] * self.hidden_dim
        for key, vec in embeddings.items():
            w = weights.get(key, 1.0) / total_w
            projected = _maybe_project(vec, self.hidden_dim)
            for i, v in enumerate(projected):
                acc[i] += v * w
        return acc

    def _project_fuse(self, embeddings: dict[str, list[float]]) -> list[float]:
        """
        Learned cross-attention projection fusion.

        Each modality is projected separately then summed.
        Falls back to weighted mean if projectors not initialised.
        """
        self._init_projectors(embeddings)
        acc_tensor = torch.zeros(self.hidden_dim, device=self.device)
        for key, vec in embeddings.items():
            proj = self._projectors.get(key)
            if proj is None:
                projected = _maybe_project(vec, self.hidden_dim)
                t = torch.tensor(projected, device=self.device)
            else:
                in_dim = len(vec)
                if proj.net[0].in_features != in_dim:
                    # Dimension mismatch — recreate projector
                    self._projectors[key] = _ModalityProjector(in_dim, self.hidden_dim).to(
                        self.device
                    )
                    proj = self._projectors[key]
                t = proj(torch.tensor(vec, device=self.device).unsqueeze(0)).squeeze(0)
            acc_tensor += t
        return acc_tensor.detach().cpu().tolist()

    def _init_projectors(self, embeddings: dict[str, list[float]]) -> None:
        """Lazily initialise projectors from input dimensions."""
        for key, vec in embeddings.items():
            if self._projectors[key] is None:
                in_dim = len(vec)
                self._projectors[key] = _ModalityProjector(in_dim, self.hidden_dim).to(self.device)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _maybe_project(vec: list[float], dim: int) -> list[float]:
    """Project or truncate *vec* to *dim* dimensions."""
    if len(vec) == dim:
        return vec
    return _project(vec, dim)
