"""
Perception Module — Nawal Brain Architecture (Sensory Cortices)

Sub-systems:
    text       — tokenization → embedding (hash-ngram or BERT sentence encoder)
    visual     — image / video understanding (CLIP ViT-B/32; stub if unavailable)
    auditory   — speech-to-text + audio embeddings (Whisper; stub if unavailable)
    multimodal — weighted fusion → world-state vector
    hub        — SensoryHub: unified entry point (accepts text | image | audio)

Phase 0: skeleton + interfaces only.
Phase 4: TextCortex + VisualCortex + AuditoryCortex + SensoryHub (this phase).
Phase 5: fine-tune VisualCortex on Belize imagery; fine-tune AuditoryCortex on Kriol.
Phase 6d: connect fused_embedding to QuantumImagination engine.

Canonical import:
    from nawal.perception import SensoryHub
"""

from perception.interfaces import AbstractCortex, WorldState
from perception.text_cortex import TextCortex
from perception.visual_cortex import VisualCortex
from perception.auditory_cortex import AuditoryCortex
from perception.multimodal_cortex import MultimodalCortex
from perception.sensory_hub import SensoryHub

__all__ = [
    # interfaces
    "AbstractCortex",
    "WorldState",
    # cortices
    "TextCortex",
    "VisualCortex",
    "AuditoryCortex",
    "MultimodalCortex",
    # hub
    "SensoryHub",
]
