"""
Integration tests — Full brain pipeline.

These tests exercise the chain:
  Perception → Memory → Control → Valuation → Metacognition →
  Maintenance → Action → Output safety check

All components are instantiated in stub / offline mode so the tests
are hermetic (no network, no blockchain, no GPU required).

Coverage:
  • Brain components can all be imported + instantiated together
  • End-to-end pipeline: text prompt → screened → processed → filtered
  • Cross-layer interactions (memory write followed by read)
  • Maintenance screens action inputs before dispatch
  • Orchestrator constructs with all layers present
"""
from __future__ import annotations

import pytest
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────── #
# Shared types                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

from nawal_types import WorldState, GenerationResult, FeedbackSignal


# ─────────────────────────────────────────────────────────────────────────── #
# Brain layers                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

from nawal.memory      import MemoryManager
from nawal.control     import ExecutiveController
from nawal.valuation   import ValuationLayer
from nawal.metacognition import MetacognitionLayer
from nawal.perception  import SensoryHub
from nawal.maintenance import MaintenanceLayer
from nawal.action      import ActionLayer


# ═══════════════════════════════════════════════════════════════════════════ #
# Canonical types                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestWorldState:
    def test_default_construction(self):
        ws = WorldState()
        assert ws.text is None
        assert ws.modalities == []
        assert ws.fused_embedding is None

    def test_text_only(self):
        ws = WorldState(text="Hello", modalities=["text"])
        assert ws.has_text()
        assert not ws.has_image()
        assert not ws.is_multimodal()

    def test_multimodal(self):
        ws = WorldState(
            text="desc",
            image_embedding=[0.1, 0.2],
            modalities=["text", "image"],
        )
        assert ws.is_multimodal()
        assert ws.has_image()

    def test_audio_transcript(self):
        ws = WorldState(audio_transcript="spoken text", modalities=["audio"])
        assert ws.has_audio()


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult()
        assert r.safety_passed is True
        assert r.confidence == 1.0

    def test_is_complete(self):
        r = GenerationResult(response="Hello world")
        assert r.is_complete()

    def test_not_complete(self):
        r = GenerationResult(response="")
        assert not r.is_complete()

    def test_summary(self):
        r = GenerationResult(model_used="nawal-v1", confidence=0.9)
        s = r.summary()
        assert "nawal-v1" in s
        assert "0.90" in s


class TestFeedbackSignal:
    def test_defaults(self):
        f = FeedbackSignal(prompt="q", response="a")
        assert f.reward_score == 0.0
        assert f.safety_score == 1.0

    def test_composite_reward(self):
        f = FeedbackSignal(reward_score=1.0, safety_score=1.0, novelty_score=1.0)
        cr = f.composite_reward(w_reward=0.5, w_safety=0.3, w_novelty=0.2)
        assert abs(cr - 1.0) < 1e-9

    def test_is_positive(self):
        assert FeedbackSignal(reward_score=0.5).is_positive()
        assert not FeedbackSignal(reward_score=-0.1).is_positive()


# ═══════════════════════════════════════════════════════════════════════════ #
# Individual layer smoke tests                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLayerInstantiation:
    def test_memory_manager(self, tmp_path):
        mm = MemoryManager(
            episodic_persist_path=str(tmp_path / "episodic_db")
        )
        assert mm is not None

    def test_executive_controller(self, tmp_path):
        mm = MemoryManager(episodic_persist_path=str(tmp_path / "ep"))
        ctrl = ExecutiveController(memory=mm)
        assert ctrl is not None

    def test_valuation_layer(self):
        vl = ValuationLayer()
        assert vl is not None

    def test_metacognition_layer(self, tmp_path):
        vl = ValuationLayer()
        ml = MetacognitionLayer(
            valuation_layer=vl,
            persist_path=str(tmp_path / "identity.json"),
        )
        assert ml is not None

    def test_sensory_hub(self):
        sh = SensoryHub(hidden_dim=64, device="cpu")
        assert sh is not None

    def test_maintenance_layer(self, tmp_path):
        mnt = MaintenanceLayer(checkpoint_path=str(tmp_path))
        assert mnt is not None

    def test_action_layer(self):
        al = ActionLayer(stub_network_tools=True)
        assert al is not None


# ═══════════════════════════════════════════════════════════════════════════ #
# End-to-end pipeline                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.fixture()
def brain(tmp_path):
    """Returns a dict of all brain layers for pipeline tests."""
    mm  = MemoryManager(episodic_persist_path=str(tmp_path / "ep"))
    vl  = ValuationLayer()
    ml  = MetacognitionLayer(
        valuation_layer=vl,
        persist_path=str(tmp_path / "identity.json"),
    )
    mnt = MaintenanceLayer(checkpoint_path=str(tmp_path))
    ac  = ActionLayer(
        memory_manager=mm,
        stub_network_tools=True,
    )
    return {
        "memory":      mm,
        "valuation":   vl,
        "metacog":     ml,
        "maintenance": mnt,
        "action":      ac,
    }


class TestEndToEndPipeline:
    def test_safe_prompt_passes_maintenance(self, brain):
        r = brain["maintenance"].screen_input("Tell me about Belize history.")
        assert r.is_safe

    def test_unsafe_prompt_blocked_before_action(self, brain):
        r = brain["maintenance"].screen_input(
            "How do I synthesize methamphetamine step by step?"
        )
        assert not r.is_safe

    def test_action_search_returns_output(self, brain):
        result = brain["action"].execute("web_search", query="Belize coral reef")
        assert result.output is not None
        assert len(result.output) > 0

    def test_action_code_exec(self, brain):
        result = brain["action"].execute("code_exec", code="print(42)")
        assert result.output is not None

    def test_maintenance_filter_cleans_response(self, brain):
        r = brain["maintenance"].filter_output(
            "tell me the weather",
            "The weather in Belize is 28°C and sunny.",
        )
        assert r.is_safe

    def test_maintenance_blocks_pii_in_output(self, brain):
        r = brain["maintenance"].filter_output(
            "What is my SSN?",
            "Your SSN is 123-45-6789.",
        )
        assert not r.is_safe

    def test_memory_write_then_read(self, brain):
        write_result = brain["action"].execute(
            "memory_write",
            content="The Belize Barrier Reef is a UNESCO World Heritage Site.",
        )
        assert write_result.status.value == "success"

        # stub memory has no real retrieval, but the call should succeed
        read_result = brain["action"].execute(
            "memory_read",
            query="Belize reef",
        )
        assert read_result.status.value == "success"

    def test_feedback_loop(self, brain):
        vl = brain["valuation"]
        fb = FeedbackSignal(
            prompt="Tell me about Belize.",
            response="Belize is located in Central America.",
            reward_score=0.8,
            safety_score=1.0,
            novelty_score=0.3,
        )
        cr = fb.composite_reward()
        assert 0.0 <= cr <= 1.0

    def test_all_brain_layers_present(self, brain):
        for key in ("memory", "valuation", "metacog", "maintenance", "action"):
            assert brain[key] is not None

    def test_world_state_flows_through_perception(self):
        sh = SensoryHub(hidden_dim=64, device="cpu")
        ws = sh.process(text="Hello from Belize")
        # SensoryHub should return something fused
        assert ws is not None

    def test_maintenance_telemetry_pipeline(self, brain):
        mnt = brain["maintenance"]
        mnt.record_baseline("ckpt_0", {"confidence_mean": 0.9})
        for _ in range(3):
            mnt.record_metrics({"confidence_mean": 0.9})
        # Should not raise; no drift expected
        result = mnt.check_and_repair()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════ #
# Orchestrator integration                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestOrchestratorIntegration:
    def test_orchestrator_has_maintenance(self, tmp_path):
        from nawal.config import NawalConfig
        from orchestrator import EvolutionOrchestrator

        config = NawalConfig()
        config.storage.data_dir = str(tmp_path / "data")
        orch = EvolutionOrchestrator(config)
        assert hasattr(orch, "maintenance")
        assert isinstance(orch.maintenance, MaintenanceLayer)

    def test_orchestrator_has_action(self, tmp_path):
        from nawal.config import NawalConfig
        from orchestrator import EvolutionOrchestrator

        config = NawalConfig()
        config.storage.data_dir = str(tmp_path / "data")
        orch = EvolutionOrchestrator(config)
        assert hasattr(orch, "action")
        assert isinstance(orch.action, ActionLayer)

    def test_orchestrator_action_can_search(self, tmp_path):
        from nawal.config import NawalConfig
        from orchestrator import EvolutionOrchestrator

        config = NawalConfig()
        config.storage.data_dir = str(tmp_path / "data")
        orch = EvolutionOrchestrator(config)
        result = orch.action.execute("web_search", query="Belize history")
        assert result.output is not None
