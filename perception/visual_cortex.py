"""
VisualCortex — Encode images into dense embeddings using CLIP.

Architecture
------------
Backbone : ``openai/clip-vit-base-patch32`` via HuggingFace Transformers.
           (open-clip-torch can replace this in Phase 5a fine-tuning.)
Output   : 512-dim L2-normalised CLIP visual embedding.

Stub mode
---------
If ``open_clip`` / ``transformers.CLIPModel`` is unavailable (CI, edge
deploy, test runner without GPU) the cortex falls back to a deterministic
hash-pixel stub that produces a valid embedding vector without any model
weights.  Stub mode is activated automatically on import failure.

PhaseHook
---------
Phase 5a: fine-tune ``CLIPModel`` on Belize aerial / marine imagery.
Replace ``_clip_embed()`` internals — the public ``encode()`` API is
unchanged.

Dependency
----------
    pip install transformers Pillow
    # OR for fine-tuning:
    pip install open-clip-torch
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger

from perception.interfaces import AbstractCortex, WorldState
from perception.text_cortex import _l2_normalize, _project

# Try to import PIL — required for image loading
try:
    from PIL import Image as PILImage  # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed — VisualCortex will accept np.ndarray only.")

# Optional numpy
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# CLIP model availability flag — checked lazily
_CLIP_ATTEMPTED = False
_CLIP_AVAILABLE = False


def _check_stub_hash(pixel_data: bytes, dim: int) -> List[float]:
    """
    Produce a deterministic embedding from raw pixel bytes (stub mode).
    Not semantically meaningful — used only when no model is available.
    """
    digest = hashlib.sha256(pixel_data[:4096]).digest()
    out: List[float] = []
    for i in range(dim):
        byte_idx = (i * 4) % len(digest)
        val = int.from_bytes(digest[byte_idx : byte_idx + 4], "big", signed=True)
        out.append(float(val) / 2**31)
    return _l2_normalize(out)


class VisualCortex(AbstractCortex):
    """
    Image-modality sensory cortex.

    Args:
        model_name  : HuggingFace CLIP model ID.
                      Set to None to force stub mode (useful for tests).
        embed_dim   : Output dimension (CLIP base = 512; set to project).
        device      : "cpu", "cuda", or "auto".
        stub_mode   : Force stub mode regardless of model availability.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL,
        embed_dim: int = 512,
        device: str = "auto",
        stub_mode: bool = False,
    ) -> None:
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.stub_mode = stub_mode or (model_name is None)
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

        self._processor: Any = None
        self._model: Any     = None
        self._loaded         = False

        logger.debug(
            f"VisualCortex init: model={model_name or 'STUB'} "
            f"dim={embed_dim} device={self.device}"
        )

    # ------------------------------------------------------------------ #
    # AbstractCortex interface                                             #
    # ------------------------------------------------------------------ #

    def preprocess(self, raw_input: Any) -> Any:
        """
        Accept PIL Image, numpy array, file path, or bytes.
        Returns a PIL Image object (or the raw object in stub mode).
        """
        if not PIL_AVAILABLE:
            return raw_input  # pass through for stub pixel hash

        if isinstance(raw_input, str):
            return PILImage.open(raw_input).convert("RGB")

        if NUMPY_AVAILABLE and isinstance(raw_input, np.ndarray):
            return PILImage.fromarray(raw_input.astype("uint8")).convert("RGB")

        if isinstance(raw_input, bytes):
            import io
            return PILImage.open(io.BytesIO(raw_input)).convert("RGB")

        # Assume PIL Image already
        return raw_input

    def encode(self, raw_input: Any) -> List[float]:
        """
        Encode an image to a 512-dim CLIP visual embedding.

        Args:
            raw_input : PIL Image, np.ndarray [H,W,C], file path str, or bytes.

        Returns:
            List[float] of length ``embed_dim``, L2-normalised.
        """
        image = self.preprocess(raw_input)

        if self.stub_mode or not self.model_name:
            return self._stub_embed(image)

        self._load_model()
        if self.stub_mode:  # load failed — fallback engaged
            return self._stub_embed(image)

        return self._clip_embed(image)

    def _to_world_state(self, embedding: List[float], raw_input: Any) -> WorldState:
        return WorldState(
            image_embedding=embedding,
            metadata={
                "cortex": "VisualCortex",
                "mode": "stub" if self.stub_mode else "clip",
                "model": self.model_name,
            },
        )

    # ------------------------------------------------------------------ #
    # CLIP mode (lazy-loaded)                                              #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # avoid retry on failure
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
            logger.info(f"VisualCortex: loading '{self.model_name}'")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        except Exception as exc:
            logger.warning(
                f"VisualCortex: CLIP load failed ({exc}). "
                "Activating stub mode."
            )
            self.stub_mode = True

    def _clip_embed(self, image: Any) -> List[float]:
        """Run CLIP vision encoder and return the image embedding."""
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)  # [1, 512]
        vec = feats[0].cpu().tolist()
        if len(vec) != self.embed_dim:
            vec = _project(vec, self.embed_dim)
        return _l2_normalize(vec)

    # ------------------------------------------------------------------ #
    # Stub mode (no model)                                                 #
    # ------------------------------------------------------------------ #

    def _stub_embed(self, image: Any) -> List[float]:
        """
        Deterministic hash-pixel stub — produces a valid unit vector
        without any model weights.  Used in tests and offline deploys.
        """
        if PIL_AVAILABLE and isinstance(image, PILImage.Image):
            # Downscale to 8×8 and hash pixels
            thumb = image.resize((8, 8)).convert("RGB")
            pixel_bytes = bytes(thumb.tobytes())
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            pixel_bytes = image.tobytes()[:4096]
        elif isinstance(image, bytes):
            pixel_bytes = image
        else:
            pixel_bytes = str(id(image)).encode()

        return _check_stub_hash(pixel_bytes, self.embed_dim)
