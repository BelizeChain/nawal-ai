"""
SensoryHub — Unified perception entry point.

Accepts any combination of text / image / audio inputs and returns a
single ``WorldState`` with all available embeddings populated and a
``fused_embedding`` in Nawal's hidden space.

Typical usage
-------------
::

    hub = SensoryHub()                        # lazy-load sub-cortices

    ws = hub.encode(text="Hello Belize!")     # text-only
    ws = hub.encode(image=pil_img)            # vision-only
    ws = hub.encode(text="caption", image=pil_img)  # multimodal

    text = hub.transcribe(audio_array)        # ASR shortcut
    desc = hub.describe_image(pil_img)        # image caption shortcut

Architecture
------------
::

    SensoryHub
    ├── TextCortex      (hash-ngram or BERT sentence embedding)
    ├── VisualCortex    (CLIP ViT-B/32  — stub if not available)
    ├── AuditoryCortex  (Whisper-base   — stub if not available)
    └── MultimodalCortex (weighted-mean fusion → fused_embedding)

Integration with IoT Oracle Pipeline
-------------------------------------
In ``integration/oracle_pipeline.py``::

    world_state = sensory_hub.encode(
        image=drone_image,
        text=metadata_text,
    )
    prediction = nawal.predict_from_embedding(world_state.fused_embedding)

PhaseHook
---------
Phase 5a: swap ``VisualCortex`` backbone for fine-tuned CLIP on Belize data.
Phase 5b: swap ``AuditoryCortex`` backbone for Kriol-fine-tuned Whisper.
Phase 6d: connect ``fused_embedding`` to ``QuantumImagination`` for richer
          environment modelling.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from loguru import logger

from perception.interfaces import WorldState
from perception.text_cortex import TextCortex
from perception.visual_cortex import VisualCortex
from perception.auditory_cortex import AuditoryCortex
from perception.multimodal_cortex import MultimodalCortex


class SensoryHub:
    """
    Unified perception gateway.

    Args:
        text_cortex      : TextCortex instance (default: TextCortex()).
        visual_cortex    : VisualCortex instance (default: slug stub).
        auditory_cortex  : AuditoryCortex instance (default: stub).
        multimodal_cortex: MultimodalCortex instance (default: weighted).
        hidden_dim       : Embedding dimension across all cortices.
        device           : "cpu", "cuda", or "auto".
        strict           : If True, raise on missing modalities.
                           If False (default), silently skip absent ones.
    """

    def __init__(
        self,
        text_cortex:       Optional[TextCortex]       = None,
        visual_cortex:     Optional[VisualCortex]     = None,
        auditory_cortex:   Optional[AuditoryCortex]   = None,
        multimodal_cortex: Optional[MultimodalCortex] = None,
        hidden_dim: int  = 256,
        device:     str  = "auto",
        strict:     bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.device     = device
        self.strict     = strict

        self._text  = text_cortex      or TextCortex(embed_dim=hidden_dim)
        self._vis   = visual_cortex    or VisualCortex(
            embed_dim=hidden_dim, stub_mode=True
        )
        self._aud   = auditory_cortex  or AuditoryCortex(
            embed_dim=hidden_dim, stub_mode=True
        )
        self._mm    = multimodal_cortex or MultimodalCortex(
            hidden_dim=hidden_dim, fusion_strategy="weighted"
        )

        logger.info(
            f"SensoryHub ready: dim={hidden_dim} "
            f"vis_stub={self._vis.stub_mode} "
            f"aud_stub={self._aud.stub_mode}"
        )

    # ------------------------------------------------------------------ #
    # Primary API                                                          #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        text:       Optional[str]  = None,
        image:      Optional[Any]  = None,   # PIL Image, np.ndarray, path
        audio:      Optional[Any]  = None,   # np.ndarray [samples] at 16 kHz
        audio_path: Optional[str]  = None,
    ) -> WorldState:
        """
        Fuse any combination of modalities into a single WorldState.

        At least one modality must be provided, otherwise a warning is
        logged and an empty WorldState is returned.

        Args:
            text       : Raw text string.
            image      : PIL Image / ndarray / file-path string.
            audio      : Float32 mono array at ``sample_rate`` Hz.
            audio_path : Path to audio file (alternative to *audio*).

        Returns:
            WorldState with populated embeddings and ``fused_embedding``.
        """
        if text is None and image is None and audio is None and audio_path is None:
            if self.strict:
                raise ValueError("SensoryHub.encode: at least one modality required")
            logger.warning("SensoryHub.encode called with no inputs")
            return WorldState()

        ws = WorldState()

        if text is not None:
            try:
                ws.text_embedding = self._text.encode(text)
                ws.raw_text = text
                ws.metadata["text_len"] = len(text)
            except Exception as exc:
                self._handle_error("TextCortex", exc)

        if image is not None:
            try:
                ws.image_embedding = self._vis.encode(image)
                ws.metadata["has_image"] = True
            except Exception as exc:
                self._handle_error("VisualCortex", exc)

        audio_input = audio if audio is not None else audio_path
        if audio_input is not None:
            try:
                ws.audio_embedding = self._aud.encode(audio_input)
                ws.metadata["has_audio"] = True
            except Exception as exc:
                self._handle_error("AuditoryCortex", exc)

        # Fuse all present embeddings
        self._mm.update_world_state(ws)

        logger.debug(
            "SensoryHub.encode: "
            f"text={'yes' if ws.text_embedding else 'no'} "
            f"image={'yes' if ws.image_embedding else 'no'} "
            f"audio={'yes' if ws.audio_embedding else 'no'} "
            f"fused_dim={len(ws.fused_embedding) if ws.fused_embedding else 0}"
        )
        return ws

    # Alias: process() is a more intuitive name for multi-modal encoding
    def process(
        self,
        text:       Optional[str]  = None,
        image:      Optional[Any]  = None,
        audio:      Optional[Any]  = None,
        audio_path: Optional[str]  = None,
    ) -> WorldState:
        """Alias for :meth:`encode` — fuse any combination of modalities."""
        return self.encode(text=text, image=image, audio=audio, audio_path=audio_path)

    def transcribe(self, audio: Any) -> str:
        """
        Convenience shortcut: speech-to-text.

        Args:
            audio : np.ndarray [samples] or file path.

        Returns:
            Transcription string.
        """
        return self._aud.transcribe(audio)

    def describe_image(self, image: Any, prompt: str = "Describe this image:") -> str:
        """
        Produce a text description of an image by:
        1. Encoding the image to a visual embedding.
        2. Encoding the prompt to a text embedding.
        3. Fusing them and returning a description placeholder.

        Note: This method returns a fixed-format string in Phase 4.
        In a full production deploy, the fused embedding would be fed
        into Nawal's decoder for actual language generation.

        Args:
            image  : PIL Image / ndarray / path.
            prompt : Text prompt to guide description.

        Returns:
            Description string.
        """
        ws = self.encode(text=prompt, image=image)
        if ws.fused_embedding is None:
            return "[Image description unavailable — no embeddings produced]"

        # Stub description: summarise embedding statistics
        feats = ws.fused_embedding
        mean_v = sum(feats) / len(feats) if feats else 0.0
        mode = "stub" if self._vis.stub_mode else "clip"
        return (
            f"[Image encoded via {mode.upper()} · "
            f"emb_dim={len(feats)} · mean={mean_v:.4f}]"
        )

    # ------------------------------------------------------------------ #
    # Sub-system accessors                                                 #
    # ------------------------------------------------------------------ #

    @property
    def text_cortex(self) -> TextCortex:
        return self._text

    @property
    def visual_cortex(self) -> VisualCortex:
        return self._vis

    @property
    def auditory_cortex(self) -> AuditoryCortex:
        return self._aud

    @property
    def multimodal_cortex(self) -> MultimodalCortex:
        return self._mm

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _handle_error(self, name: str, exc: Exception) -> None:
        msg = f"SensoryHub: {name} error: {exc}"
        if self.strict:
            raise RuntimeError(msg) from exc
        logger.warning(msg)
