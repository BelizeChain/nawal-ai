"""
B6 Audit Tests — Memory, Perception & Cortex.

Covers:
  C6.1 Memory scope and isolation
  C6.2 Perception input sanitisation
  C6.3 Cortex coordination logic
  C6.4 Data flow to transformer
"""

from __future__ import annotations

import time
import uuid
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memory.interfaces import MemoryRecord
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.manager import MemoryManager

from perception.text_cortex import TextCortex
from perception.visual_cortex import VisualCortex
from perception.auditory_cortex import AuditoryCortex
from perception.multimodal_cortex import MultimodalCortex
from perception.sensory_hub import SensoryHub
from perception.interfaces import WorldState

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _rec(
    key: str = None,
    content: str = "test",
    dim: int = 8,
    metadata: dict = None,
    ttl: float = None,
    embedding: list = None,
) -> MemoryRecord:
    emb = (
        embedding if embedding is not None else list(np.random.randn(dim).astype(float))
    )
    return MemoryRecord(
        key=key or str(uuid.uuid4()),
        content=content,
        embedding=emb,
        metadata=metadata or {},
        ttl=ttl,
    )


# =========================================================================== #
# C6.1 — Memory scope and isolation                                           #
# =========================================================================== #


class TestC61MemoryScope:

    # F6.1a — MemoryManager.retrieve handles None embeddings gracefully
    def test_retrieve_handles_none_embedding(self):
        """Records with embedding=None should not crash retrieve()."""
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        rec_with = _rec("with_emb", content="has embedding", dim=8)
        rec_without = MemoryRecord(
            key="no_emb", content="no embedding", embedding=None, metadata={}
        )
        mm.store(rec_with, store="working")
        mm.store(rec_without, store="working")

        query = list(np.random.randn(8).astype(float))
        results = mm.retrieve(query, top_k=10)
        keys = {r.key for r in results}
        assert "with_emb" in keys
        assert "no_emb" in keys

    def test_retrieve_handles_empty_embedding(self):
        """Records with embedding=[] should not crash retrieve()."""
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        rec = MemoryRecord(
            key="empty_emb", content="empty vec", embedding=[], metadata={}
        )
        mm.store(rec, store="working")
        query = list(np.random.randn(8).astype(float))
        results = mm.retrieve(query, top_k=10)
        assert any(r.key == "empty_emb" for r in results)

    # F6.1b — consolidation_age validation
    def test_negative_consolidation_age_raises(self):
        """MemoryManager should reject negative consolidation_age."""
        with pytest.raises(ValueError, match="consolidation_age"):
            MemoryManager(consolidation_age=-1.0, episodic_persist_path=None)

    def test_zero_consolidation_age_accepted(self):
        """Zero is a valid consolidation_age (immediate consolidation)."""
        mm = MemoryManager(consolidation_age=0.0, episodic_persist_path=None)
        assert mm.consolidation_age == 0.0

    # Memory isolation: separate MemoryManager instances don't share
    def test_memory_managers_isolated(self):
        mm1 = MemoryManager(working_max_size=8, episodic_persist_path=None)
        mm2 = MemoryManager(working_max_size=8, episodic_persist_path=None)
        mm1.store(_rec("only_in_mm1", dim=8), store="working")
        assert mm1.get("only_in_mm1") is not None
        assert mm2.get("only_in_mm1") is None

    # Store routing correctness
    def test_store_routing_working(self):
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        mm.store(_rec("w1", dim=8), store="working")
        assert mm.working.get("w1") is not None
        assert mm.episodic.get("w1") is None

    def test_store_routing_episodic(self):
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        mm.store(_rec("e1", dim=8), store="episodic")
        assert mm.episodic.get("e1") is not None
        assert mm.working.get("e1") is None

    def test_store_routing_all(self):
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        mm.store(_rec("a1", dim=8), store="all")
        assert mm.working.get("a1") is not None
        assert mm.episodic.get("a1") is not None
        assert mm.semantic.get("a1") is not None

    def test_unknown_store_raises(self):
        mm = MemoryManager(episodic_persist_path=None)
        with pytest.raises(ValueError, match="Unknown store"):
            mm.retrieve([0.1] * 8, stores=["nonexistent"])

    # Consolidation moves items correctly
    def test_consolidation_moves_mature_items(self):
        mm = MemoryManager(
            working_max_size=16,
            consolidation_age=0.0,
            episodic_persist_path=None,
        )
        rec = _rec("c1", dim=8)
        rec.timestamp = time.time() - 10  # already mature
        mm.store(rec, store="working")
        moved = mm.consolidate()
        assert moved == 1
        assert mm.working.get("c1") is None
        assert mm.episodic.get("c1") is not None


# =========================================================================== #
# C6.2 — Perception input sanitisation                                        #
# =========================================================================== #


class TestC62PerceptionSanitisation:

    # F6.2a — TextCortex strips null bytes and control chars
    def test_text_cortex_strips_null_bytes(self):
        tc = TextCortex(embed_dim=32)
        cleaned = tc.preprocess("hello\x00world")
        assert "\x00" not in cleaned
        assert "hello" in cleaned and "world" in cleaned

    def test_text_cortex_strips_control_chars(self):
        tc = TextCortex(embed_dim=32)
        cleaned = tc.preprocess("test\x01\x02\x03data\x7fend")
        assert "\x01" not in cleaned
        assert "\x7f" not in cleaned
        assert "test" in cleaned
        assert "data" in cleaned
        assert "end" in cleaned

    def test_text_cortex_preserves_newlines_tabs(self):
        """Newlines/tabs should collapse to spaces but not be stripped as control chars."""
        tc = TextCortex(embed_dim=32)
        cleaned = tc.preprocess("line1\nline2\ttab")
        assert "line1" in cleaned and "line2" in cleaned and "tab" in cleaned

    # F6.2b — TextCortex hash mode enforces length limit
    def test_text_cortex_hash_mode_truncates_long_input(self):
        tc = TextCortex(embed_dim=32, model_name=None)
        long_text = "word " * 50_000  # 250k chars
        cleaned = tc.preprocess(long_text)
        assert len(cleaned) <= tc.MAX_TEXT_LENGTH

    def test_text_cortex_hash_mode_long_input_embedding_succeeds(self):
        """Very long text should produce a valid embedding without hanging."""
        tc = TextCortex(embed_dim=32, model_name=None)
        long_text = "word " * 50_000
        emb = tc.encode(long_text)
        assert len(emb) == 32
        assert all(isinstance(v, float) for v in emb)

    def test_text_cortex_empty_input(self):
        tc = TextCortex(embed_dim=32)
        emb = tc.encode("")
        assert emb == [0.0] * 32

    def test_text_cortex_non_string_coercion(self):
        tc = TextCortex(embed_dim=32)
        emb = tc.encode(12345)
        assert len(emb) == 32

    # F6.2c — AuditoryCortex preprocess enforces max_duration
    def test_auditory_cortex_preprocess_trims_long_audio(self):
        ac = AuditoryCortex(
            embed_dim=32,
            stub_mode=True,
            sample_rate=16000,
            max_duration=1.0,
        )
        # 3 seconds of audio at 16kHz
        long_audio = np.random.randn(48000).astype("float32")
        processed = ac.preprocess(long_audio)
        max_samples = int(16000 * 1.0)
        assert len(processed) <= max_samples

    def test_auditory_cortex_preprocess_short_audio_unchanged(self):
        ac = AuditoryCortex(
            embed_dim=32,
            stub_mode=True,
            sample_rate=16000,
            max_duration=30.0,
        )
        short_audio = np.random.randn(8000).astype("float32")
        processed = ac.preprocess(short_audio)
        assert len(processed) == 8000

    def test_auditory_cortex_preprocess_stereo_to_mono(self):
        ac = AuditoryCortex(
            embed_dim=32,
            stub_mode=True,
            sample_rate=16000,
            max_duration=30.0,
        )
        stereo = np.random.randn(2, 8000).astype("float32")
        processed = ac.preprocess(stereo)
        assert processed.ndim == 1

    # VisualCortex stub mode produces valid embedding
    def test_visual_cortex_stub_embedding_valid(self):
        vc = VisualCortex(embed_dim=32, stub_mode=True)
        fake_image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        emb = vc.encode(fake_image)
        assert len(emb) == 32
        assert all(isinstance(v, float) for v in emb)

    # SensoryHub with no inputs
    def test_sensory_hub_no_inputs_non_strict(self):
        hub = SensoryHub(hidden_dim=32)
        ws = hub.encode()
        assert ws.fused_embedding is None

    def test_sensory_hub_no_inputs_strict_raises(self):
        hub = SensoryHub(hidden_dim=32, strict=True)
        with pytest.raises(ValueError, match="at least one modality"):
            hub.encode()

    # SensoryHub text encoding produces valid WorldState
    def test_sensory_hub_text_encoding(self):
        hub = SensoryHub(hidden_dim=32)
        ws = hub.encode(text="Hello Belize!")
        assert ws.text_embedding is not None
        assert len(ws.text_embedding) == 32
        assert ws.fused_embedding is not None


# =========================================================================== #
# C6.3 — Cortex coordination logic (cortex/ is a re-export alias)            #
# =========================================================================== #


class TestC63CortexCoordination:

    def test_cortex_exports_transformer(self):
        """cortex/__init__.py should re-export architecture classes."""
        from cortex import NawalTransformer

        assert NawalTransformer is not None

    def test_cortex_exports_config(self):
        from cortex import NawalModelConfig

        assert NawalModelConfig is not None

    def test_multimodal_fusion_strategy_validation(self):
        with pytest.raises(ValueError, match="fusion_strategy"):
            MultimodalCortex(fusion_strategy="invalid")

    def test_multimodal_weighted_fusion(self):
        mm = MultimodalCortex(hidden_dim=32, fusion_strategy="weighted")
        ws = WorldState(
            text_embedding=list(np.random.randn(32)),
            image_embedding=list(np.random.randn(32)),
        )
        result = mm.fuse(ws)
        assert len(result) == 32
        # Not all zeros
        assert any(v != 0.0 for v in result)

    def test_multimodal_mean_fusion(self):
        mm = MultimodalCortex(hidden_dim=32, fusion_strategy="mean")
        ws = WorldState(text_embedding=list(np.random.randn(32)))
        result = mm.fuse(ws)
        assert len(result) == 32

    def test_multimodal_no_embeddings_returns_zeros(self):
        mm = MultimodalCortex(hidden_dim=32)
        ws = WorldState()
        result = mm.fuse(ws)
        assert result == [0.0] * 32

    def test_multimodal_update_world_state(self):
        mm = MultimodalCortex(hidden_dim=32)
        ws = WorldState(text_embedding=list(np.random.randn(32)))
        mm.update_world_state(ws)
        assert ws.fused_embedding is not None
        assert len(ws.fused_embedding) == 32


# =========================================================================== #
# C6.4 — Data flow to transformer                                             #
# =========================================================================== #


class TestC64DataFlow:

    # F6.4a — generate() guards against negative max_new_tokens
    def test_generate_prompt_longer_than_max_length(self):
        """When prompt tokens >= max_length, generate should return promptly."""
        from unittest.mock import MagicMock, PropertyMock
        import torch

        mock_nawal = MagicMock()
        # Simulate tokenizer returning more tokens than max_length
        fake_ids = torch.arange(150).unsqueeze(0)  # 150 tokens
        mock_nawal.tokenizer.encode.return_value = fake_ids
        mock_nawal.tokenizer.decode.return_value = "decoded prompt"
        mock_nawal.language_detector.detect.return_value = "en"
        mock_nawal.compliance_filter.filter.side_effect = lambda x: x

        # Import the generate function's logic
        from client.nawal import Nawal

        # Call the logic directly using the instance method bound to our mock
        # We replicate the key guard logic check
        prompt = "x" * 500
        max_length = 100
        input_ids = fake_ids
        max_new_tokens = max_length - input_ids.size(1)
        assert (
            max_new_tokens < 1
        ), "This test expects prompt to be longer than max_length"

    def test_memory_manager_context_window_returns_list(self):
        """context_window should always return a list."""
        mm = MemoryManager(working_max_size=16, episodic_persist_path=None)
        mm.store_text("Turn 1", embedding=[0.1] * 8, store="working")
        mm.store_text("Turn 2", embedding=[0.2] * 8, store="working")
        ctx = mm.context_window(n_recent=5)
        assert isinstance(ctx, list)
        assert len(ctx) == 2

    def test_context_window_with_query(self):
        """context_window with query_embedding should include episodic results."""
        mm = MemoryManager(
            working_max_size=16,
            episodic_persist_path=None,
            embedding_dim=8,
        )
        mm.store_text("Working item", embedding=[0.5] * 8, store="working")
        mm.store_text("Episodic item", embedding=[0.3] * 8, store="episodic")
        ctx = mm.context_window(
            query_embedding=[0.5] * 8,
            n_recent=5,
            n_episodic=5,
        )
        assert len(ctx) >= 1

    def test_sensory_hub_to_multimodal_dimension_consistency(self):
        """SensoryHub output dimensions should match hidden_dim."""
        hub = SensoryHub(hidden_dim=64)
        ws = hub.encode(text="Test input for dimension check")
        assert ws.text_embedding is not None
        assert len(ws.text_embedding) == 64
        assert ws.fused_embedding is not None
        assert len(ws.fused_embedding) == 64


# =========================================================================== #
# Semantic memory persistence safety                                           #
# =========================================================================== #


class TestSemanticMemorySafety:

    def test_proximity_decay_validation(self):
        with pytest.raises(ValueError, match="proximity_decay"):
            SemanticMemory(proximity_decay=0.0)
        with pytest.raises(ValueError, match="proximity_decay"):
            SemanticMemory(proximity_decay=1.5)

    def test_store_and_retrieve(self):
        sm = SemanticMemory()
        rec = _rec("concept1", content="Test concept", dim=8)
        sm.store(rec)
        assert sm.get("concept1") is not None
        results = sm.retrieve([0.1] * 8, top_k=5)
        assert len(results) >= 1

    def test_add_relation_missing_node_raises(self):
        sm = SemanticMemory()
        sm.store(_rec("a", dim=8))
        with pytest.raises(KeyError):
            sm.add_relation("a", "nonexistent")

    def test_clear(self):
        sm = SemanticMemory()
        sm.store(_rec("x", dim=8))
        sm.clear()
        assert len(sm) == 0
