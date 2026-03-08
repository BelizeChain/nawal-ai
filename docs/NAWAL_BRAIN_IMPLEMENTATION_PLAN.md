# Nawal AI — Full Brain Architecture Implementation Plan
**BelizeChain AI Team | March 2026**
**Classification: Internal Technical Roadmap**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Audit](#2-current-state-audit)
3. [Target Architecture](#3-target-architecture)
4. [Brain-to-Code Mapping](#4-brain-to-code-mapping)
5. [Gap Analysis](#5-gap-analysis)
6. [Phase 0 — Modular Refactor](#phase-0--modular-refactor-weeks-14)
7. [Phase 1 — Memory System (Hippocampus)](#phase-1--memory-system-hippocampus-weeks-510)
8. [Phase 2 — Executive Controller (Prefrontal Cortex)](#phase-2--executive-controller-prefrontal-cortex-weeks-1118)
9. [Phase 3 — Valuation & Safety (Limbic + Immune)](#phase-3--valuation--safety-limbic--immune-weeks-1924)
10. [Phase 4 — Metacognitive Layer (Default Mode Network)](#phase-4--metacognitive-layer-default-mode-network-weeks-2530)
11. [Phase 5 — Perception Layer (Sensory Cortices)](#phase-5--perception-layer-sensory-cortices-weeks-3140)
12. [Phase 6 — Quantum Integration (Kinich Full Activation)](#phase-6--quantum-integration-kinich-full-activation-weeks-4152)
13. [Module Interface Contracts](#13-module-interface-contracts)
14. [Dependency & Technology Decisions](#14-dependency--technology-decisions)
15. [Testing Strategy Per Phase](#15-testing-strategy-per-phase)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Success Metrics Summary](#17-success-metrics-summary)

---

## 1. Executive Summary

Nawal is Belize's sovereign AI system. The goal of this plan is to evolve it from a capable
**language model with genome evolution and federated learning** into a full **artificial brain**
with all major cognitive systems represented in code.

**What Nawal already is** (strong foundation):
- A custom from-scratch transformer (117M / 350M / 1.3B params)
- NEAT-style genetic algorithm driving architecture evolution
- Federated learning across validator nodes (Flower + Substrate)
- Blockchain-integrated rewards (DALLA), staking (PoUW), genome registry (on-chain)
- Hybrid quantum-classical LLM architecture with Kinich bridge
- Byzantine-resistant federated security (Krum, DP-SGD, Paillier encryption)
- Domain-specific models for AgriTech, Marine, Education, Tech, General
- IoT oracle pipeline (drones, sensors, buoys, cameras, weather stations)

**What Nawal is not yet** (the gaps this plan closes):
- Cannot remember past conversations beyond the context window
- Cannot perceive images or audio — text only
- Cannot plan multi-step tasks autonomously
- Has no self-critique or self-correction loop
- Has no explicit reward/preference model (RLHF)
- Has no output safety filter
- Has no metacognitive identity module

**Plan duration**: 12 months (52 weeks), 6 phases.
**Team size assumption**: 3–5 engineers.
**Quantum activation** (Phase 6) requires a live Kinich quantum node.

---

## 2. Current State Audit

### 2.1 File Inventory with Brain Mapping

| File / Module | Lines | Brain Region | Status |
|---|---|---|---|
| `architecture/transformer.py` | 386 | Cerebrum | ✅ Production |
| `architecture/attention.py` | 209 | Cerebrum | ✅ Production |
| `architecture/feedforward.py` | ~150 | Cerebrum | ✅ Production |
| `architecture/embeddings.py` | ~200 | Cerebrum | ✅ Production |
| `architecture/config.py` | 173 | Cerebrum | ✅ Production |
| `client/nawal.py` | 539 | Cerebrum | ✅ Production |
| `models/hybrid_llm.py` | 323 | Cerebrum + Quantum | ✅ Implemented |
| `genome/encoding.py` | 569 | DNA | ✅ Production |
| `genome/operators.py` | 888 | DNA | ✅ Production |
| `genome/population.py` | 595 | DNA | ✅ Production |
| `genome/fitness.py` | 591 | DNA / Limbic | ✅ Production |
| `genome/history.py` | ~300 | DNA | ✅ Production |
| `genome/model_builder.py` | ~400 | DNA | ✅ Production |
| `genome/dna.py` | 330 | DNA (compat) | ✅ Stable |
| `orchestrator.py` | 607 | Prefrontal Cortex | 🟡 Partial |
| `hybrid/engine.py` | 329 | Prefrontal Cortex | 🟡 Partial |
| `hybrid/router.py` | 197 | Prefrontal Cortex | 🟡 Partial |
| `hybrid/confidence.py` | 229 | Prefrontal + Metacog | 🟡 Partial |
| `hybrid/teacher.py` | ~200 | Prefrontal | ✅ Stable |
| `training/distillation.py` | 722 | Basal Ganglia | 🟡 Partial |
| `security/byzantine_detection.py` | 593 | Immune System | ✅ Production |
| `security/differential_privacy.py` | ~400 | Immune System | ✅ Production |
| `security/secure_aggregation.py` | 762 | Immune System | ✅ Production |
| `security/dp_inference.py` | ~300 | Immune System | ✅ Stable |
| `server/aggregator.py` | 722 | Motor (FL coord) | ✅ Production |
| `server/participant_manager.py` | ~400 | Motor | ✅ Production |
| `api_server.py` | 744 | Motor | ✅ Production |
| `api/inference_server.py` | ~400 | Motor | ✅ Production |
| `blockchain/staking_connector.py` | 709 | Motor | ✅ Production |
| `blockchain/rewards.py` | 451 | Limbic | ✅ Production |
| `blockchain/genome_registry.py` | 469 | DNA | ✅ Production |
| `blockchain/community_connector.py` | ~400 | Limbic | ✅ Stable |
| `blockchain/mesh_network.py` | ~500 | Motor | ✅ Stable |
| `blockchain/identity_verifier.py` | ~300 | Immune | ✅ Stable |
| `integration/kinich_connector.py` | 445 | Quantum | 🟡 Embryonic |
| `integration/oracle_pipeline.py` | 679 | Motor | ✅ Production |
| `integration/oracle_pipeline.py` (IoT) | 679 | Sensory (Input) | 🟡 Data only |
| `client/domain_models.py` | 1357 | Cerebellum | 🟡 Partial |
| `monitoring/metrics.py` | 408 | Immune | ✅ Stable |
| `monitoring/prometheus_exporter.py` | ~200 | Immune | ✅ Stable |
| `storage/checkpoint_manager.py` | 260 | Memory (weights) | ✅ Stable |
| `storage/pakit_client.py` | ~300 | Memory (storage) | ✅ Stable |
| `data/data_manager.py` | 535 | Sensory (training) | ✅ Stable |
| `data/tokenizers.py` | ~400 | Sensory | ✅ Stable |

### 2.2 Architecture Summary (What We Have Built)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NAWAL-AI CURRENT ARCHITECTURE                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │            CEREBRUM (TRANSFORMER BACKBONE)                   │    │
│  │  NawalTransformer (117M / 350M / 1.3B)                      │    │
│  │  ├── MultiHeadAttention (RoPE, KV-cache)                    │    │
│  │  ├── FeedForward (GeLU, pre-norm)                           │    │
│  │  ├── NawalEmbeddings (token + positional)                   │    │
│  │  └── 5-language vocab: EN / ES / Kriol / Garifuna / Maya    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          DNA — GENOME EVOLUTION ENGINE                        │   │
│  │  Genome (encoding.py) → GenomeEncoder                        │   │
│  │  ├── MutationOperators (12 mutation types)                   │   │
│  │  ├── CrossoverOperators (5 crossover types)                  │   │
│  │  ├── PopulationManager (tournament / roulette / rank)        │   │
│  │  ├── FitnessEvaluator (Quality 40% / Time 30% / Honesty 30%) │  │
│  │  └── GenomeRegistry (on-chain via Substrate)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │       PREFRONTAL CORTEX — HYBRID ROUTING ENGINE               │   │
│  │  HybridNawalEngine                                            │   │
│  │  ├── ConfidenceScorer (entropy + perplexity + length + lang) │   │
│  │  ├── IntelligentRouter (sovereign path vs teacher path)       │   │
│  │  ├── DeepSeekTeacher (33B fallback)                          │   │
│  │  └── KnowledgeDistillation (nightly batch improvement)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │       FEDERATED LEARNING ORCHESTRATION                        │   │
│  │  EvolutionOrchestrator                                        │   │
│  │  ├── FederatedAggregator (FedAvg / Byzantine-robust)         │   │
│  │  ├── ParticipantManager (validator tracking)                  │   │
│  │  ├── ByzantineDetector (Krum / trimmed mean / PHOCAS)        │   │
│  │  ├── SecureAggregation (Paillier homomorphic encryption)      │   │
│  │  └── DifferentialPrivacy (DP-SGD / DP inference)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │       MOTOR CORTEX — ACTION LAYER                             │   │
│  │  ├── API Server (FastAPI FL endpoints)                        │   │
│  │  ├── Inference Server (text generation)                       │   │
│  │  ├── OraclePipeline (IoT → inference → PoUW submission)      │   │
│  │  ├── StakingConnector (Substrate extrinsics)                  │   │
│  │  ├── RewardDistributor (DALLA distribution)                   │   │
│  │  └── MeshNetwork (P2P validator communication)               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │       QUANTUM BRIDGE (KINICH)                                 │   │
│  │  KinichQuantumConnector (classical → encode → QNN → decode)  │   │
│  │  HybridQuantumClassicalLLM (quantum injection at layer N/2)  │   │
│  │  Status: IMPLEMENTED but no live Kinich endpoint             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ❌ MISSING: Memory | Perception | Metacognition | Safety Filter     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 What the Domain Models Already Know

`client/domain_models.py` has 1357 lines of domain specialist models.
These are the **cerebellum** — specialized fast-skill modules:

| Domain | Model Type | Input | Output | Reward Multiplier |
|---|---|---|---|---|
| `AGRITECH` | Crop/soil predictor | Sensor arrays | Yield forecast | 1.5× |
| `MARINE` | Ocean health predictor | Buoy + drone data | Ecosystem status | 1.4× |
| `EDUCATION` | Personalized tutor | Student performance | Learning path | 1.3× |
| `TECH` | Infrastructure monitor | System telemetry | Anomaly / action | 1.1× |
| `GENERAL` | Multi-purpose LLM | Any text | Any text | 1.0× |

These domain models are already hooked into the IoT Oracle pipeline.
They feed PoUW submissions back to the staking pallet.
**This is a complete Cerebellum stub — the specialized fast-skill layer exists.**

---

## 3. Target Architecture

The target state is a full artificial brain with all 11 systems active:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NAWAL-AI TARGET ARCHITECTURE (12 months)                   │
│                                                                               │
│  SENSORY HUB ──────────────────────────────────────────────────────────┐    │
│  [nawal/perception/]                                                    │    │
│  VisualCortex (ViT) + AuditoryCortex (Whisper) + MultimodalFusion      │    │
│                   │ world_state_embedding                               │    │
│                   ▼                                                     │    │
│  CEREBRUM (transformer) ─────────────────────────────────────────────┐ │    │
│  [nawal/cortex/]                                                      │ │    │
│  NawalTransformer + HybridQuantumClassicalLLM                        │ │    │
│  SpecializedCortices: MathReasoner, CodeReasoner, SocialReasoner     │ │    │
│             │ candidate_thoughts / plans / responses                  │ │    │
│             ▼                                                          │ │    │
│  HIPPOCAMPUS / MEMORY ─────────────────────────────────────────────┐ │ │    │
│  [nawal/memory/]                                                    │ │ │    │
│  WorkingMemory (scratchpad) + EpisodicMemory + SemanticMemory(RAG)  │ │ │    │
│  QuantumMemory (Grover search — Phase 6)                           │ │ │    │
│             │ retrieved_context                                      │ │ │    │
│             ▼                                                        │ │ │    │
│  PREFRONTAL CORTEX / EXECUTIVE ───────────────────────────────────┐│ │ │    │
│  [nawal/control/]                                                  ││ │ │    │
│  GoalStack + TaskPlanner + ToolRegistry + ActionExecutor           ││ │ │    │
│  QuantumPlanSelector (QAOA — Phase 6)                              ││ │ │    │
│             │ best_plan | tool_call                                 ││ │ │    │
│             ▼                                                       ││ │ │    │
│  CEREBELLUM / DOMAIN SKILLS ───────────────────────────────────┐  ││ │ │    │
│  [nawal/client/domain_models.py] (already exists)              │  ││ │ │    │
│  AgriTech + Marine + Education + Tech + General                 │  ││ │ │    │
│             │ skill_output                                      │  ││ │ │    │
│             ▼                                                   │  ││ │ │    │
│  MOTOR CORTEX / ACTION ──────────────────────────────────────┐ │  ││ │ │    │
│  [nawal/action/]                                             │ │  ││ │ │    │
│  APIAgent + BlockchainActions + OraclePipeline + CodeSandbox │ │  ││ │ │    │
│             │ world_change                                    │ │  ││ │ │    │
│             ▼                                                 │ │  ││ │ │    │
│  BASAL GANGLIA / DNA ─────────────────────────────────────┐  │ │  ││ │ │    │
│  [nawal/genome/] (already exists)                         │  │ │  ││ │ │    │
│  GenomeEvolution + DistillationTrainer + RewardModel       │  │ │  ││ │ │    │
│             │ weight_update | genome_evolution              │  │ │  ││ │ │    │
│             ▼                                               │  │ │  ││ │ │    │
│  LIMBIC SYSTEM / VALUATION ─────────────────────────────┐  │  │ │  ││ │ │    │
│  [nawal/valuation/]                                      │  │  │ │  ││ │ │    │
│  RewardModel + SafetyFilter + IntrinsicReward            │  │  │ │  ││ │ │    │
│  ValuationAggregator + QuantumValueEvaluator (Phase 6)   │  │  │ │  ││ │ │    │
│             │ good_bad_signal                              │  │  │ │  ││ │ │    │
│             ▼                                             │  │  │ │  ││ │ │    │
│  METACOGNITIVE LAYER ───────────────────────────────────┐│  │  │ │  ││ │ │    │
│  [nawal/metacognition/]                                 ││  │  │ │  ││ │ │    │
│  SelfCritic + ConsistencyChecker + ConfidenceCalibrator ││  │  │ │  ││ │ │    │
│  InternalSimulator + IdentityModule                     ││  │  │ │  ││ │ │    │
│  QuantumImagination (parallel rollouts — Phase 6)       ││  │  │ │  ││ │ │    │
│             │ refined_output                              ││  │  │ │  ││ │ │    │
│             ▼                                            ││  │  │ │  ││ │ │    │
│  IMMUNE SYSTEM / MAINTENANCE ──────────────────────────┐││  │  │ │  ││ │ │    │
│  [nawal/maintenance/] ← [security/ + monitoring/]      │││  │  │ │  ││ │ │    │
│  ByzantineDetector + DifferentialPrivacy                │││  │  │ │  ││ │ │    │
│  InputSafetyScreener + OutputSafetyFilter               │││  │  │ │  ││ │ │    │
│  DriftDetector + SelfRepairProtocol                     │││  │  │ │  ││ │ │    │
│  QuantumAnomalyDetector (Phase 6)                       │││  │  │ │  ││ │ │    │
└─────────────────────────────────────────────────────────┘└┘  └  └ └  └└ └ └────┘
```

---

## 4. Brain-to-Code Mapping

### Definitive mapping of brain region → Python module

| Brain Region  | Cognitive Function | Python Module | Key Classes |
|---|---|---|---|
| 🧠 Cerebrum | Language, reasoning, abstraction | `nawal/cortex/` | `NawalTransformer`, `NawalTransformerBlock` |
| 🧬 DNA | Genome, evolution, heredity | `nawal/genome/` | `Genome`, `PopulationManager`, `FitnessScore` |
| 🧩 Prefrontal Cortex | Planning, goals, working memory | `nawal/control/` | `ExecutiveController`, `GoalStack`, `TaskPlanner` |
| 🧠 Hippocampus | Episodic + semantic memory, RAG | `nawal/memory/` | `MemoryManager`, `EpisodicMemory`, `SemanticMemory` |
| 👁️ Sensory Cortices | Vision, audio, multimodal | `nawal/perception/` | `SensoryHub`, `VisualCortex`, `AuditoryCortex` |
| 🏃 Motor Cortex | Tool use, API calls, actions | `nawal/action/` | `ToolRegistry`, `ActionExecutor`, `APIAgent` |
| 🧠 Cerebellum | Fast domain skills | `nawal/client/domain_models.py` | `AgriTechModel`, `MarineModel`, `EduModel` |
| ❤️ Limbic System | Reward, preferences, safety | `nawal/valuation/` | `RewardModel`, `SafetyFilter`, `IntrinsicReward` |
| 🧩 Default Mode Net | Metacognition, self-reflection | `nawal/metacognition/` | `SelfCritic`, `ConsistencyChecker`, `IdentityModule` |
| 🧠 Basal Ganglia | RL, habit, distillation | `nawal/training/distillation.py` | `KnowledgeDistillationTrainer` |
| 🧬 Immune System | Safety, anomaly, self-repair | `nawal/maintenance/` | `InputScreener`, `OutputFilter`, `DriftDetector` |
| ⚛️ Quantum | Kinich integration | `nawal/quantum/` | `KinichQuantumConnector`, `HybridQuantumClassicalLLM` |

---

## 5. Gap Analysis

### 5.1 Status by System

| System | What Exists | What is Missing | Severity |
|---|---|---|---|
| **Cerebrum** | Full transformer, 3 model sizes, KV-cache | Specialized cortex routing (math/code/social/physical) | LOW |
| **DNA** | Complete NEAT genome engine + on-chain registry | RLHF signal into fitness; user preference feedback loop | MEDIUM |
| **Prefrontal Cortex** | Confidence-based routing, multi-gen orchestration | Goal stack, task decomposition, working memory buffer, tool use | CRITICAL |
| **Hippocampus** | Checkpoint storage, Pakit IPFS | Vector DB, RAG pipeline, episodic store, semantic memory | CRITICAL |
| **Sensory Cortices** | BPE tokenizer (training only), IoT data parsers | Vision encoder, audio encoder, multimodal fusion, runtime perception | HIGH |
| **Motor Cortex** | FL API, blockchain actions, oracle IoT pipeline | General tool-use agent, code sandbox, web search | HIGH |
| **Cerebellum** | 5 domain models (AgriTech / Marine / Edu / Tech / General) | Fast reflexive caching, low-latency response store | LOW |
| **Limbic / Valuation** | PoUW fitness, DALLA rewards, confidence scorer | Reward model (RLHF/DPO), output safety filter, input injection detector | CRITICAL |
| **Metacognition** | Single confidence scalar | Self-critique loop, consistency sampling, internal simulation, identity | HIGH |
| **Basal Ganglia** | Distillation trainer (batch), genome evolution | Online RL (PPO/GRPO), preference model, habit cache | MEDIUM |
| **Immune System** | Byzantine detection, DP-SGD, Paillier, monitoring | Input safety screener, output classifier, model drift detector, self-repair | HIGH |
| **Quantum** | Kinich connector + Hybrid LLM architecture | Live Kinich endpoint, quantum memory, quantum optimization, quantum anomaly | MEDIUM |

### 5.2 Dependency Graph (build this before that)

```
Memory ──────────┐
                 ▼
Perception ──► Executive Controller ──► Metacognition
                 │                          │
                 ▼                          ▼
          Valuation/Safety ──────► Full Response Pipeline
                 │
                 ▼
          Genome Fitness Loop (RLHF → DNA)
                 │
                 ▼
          Quantum Modules (requires live Kinich node)
```

**Critical dependency**: Memory must be built before Executive Controller.
**Critical dependency**: Valuation/Safety filter must be built before public deployment.
**Quantum** is independent of all other phases — it enhances existing modules rather than blocking them.

---

## Phase 0 — Modular Refactor (Weeks 1–4)

### Goal
Reorganize existing code into the brain-layout directory structure.
Zero new features. All existing tests must still pass after refactoring.

### Why This Matters
Currently, there is no `nawal/memory/`, `nawal/control/`, `nawal/perception/`,
`nawal/valuation/`, or `nawal/metacognition/`. Without the directory skeleton,
future phases have no clear home, and import paths will collide.

### Proposed Directory Structure (Target)

```
nawal/
├── __init__.py
├── cortex/                     ← RENAMED from architecture/
│   ├── __init__.py
│   ├── transformer.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── embeddings.py
│   └── config.py
├── genome/                     ← UNCHANGED (already clean)
│   ├── dna.py
│   ├── encoding.py
│   ├── fitness.py
│   ├── operators.py
│   ├── population.py
│   ├── history.py
│   └── model_builder.py
├── memory/                     ← NEW (Phase 1)
│   ├── __init__.py
│   ├── interfaces.py           ← ABC definitions only (Phase 0 placeholder)
│   ├── working_memory.py
│   ├── episodic_memory.py
│   ├── semantic_memory.py
│   ├── rag_pipeline.py
│   └── memory_manager.py
├── control/                    ← NEW (Phase 2), absorbs hybrid/
│   ├── __init__.py
│   ├── interfaces.py           ← ABC definitions only (Phase 0 placeholder)
│   ├── goal_stack.py
│   ├── task_planner.py
│   ├── tool_registry.py
│   ├── action_executor.py
│   └── executive_controller.py
├── perception/                 ← NEW (Phase 5)
│   ├── __init__.py
│   ├── interfaces.py           ← ABC definitions only (Phase 0 placeholder)
│   ├── visual_cortex.py
│   ├── auditory_cortex.py
│   ├── multimodal_cortex.py
│   └── sensory_hub.py
├── valuation/                  ← NEW (Phase 3)
│   ├── __init__.py
│   ├── interfaces.py           ← ABC definitions only (Phase 0 placeholder)
│   ├── reward_model.py
│   ├── preference_store.py
│   ├── safety_filter.py
│   ├── intrinsic_reward.py
│   └── value_aggregator.py
├── metacognition/              ← NEW (Phase 4)
│   ├── __init__.py
│   ├── interfaces.py           ← ABC definitions only (Phase 0 placeholder)
│   ├── self_critic.py
│   ├── consistency_checker.py
│   ├── confidence_calibrator.py
│   ├── internal_simulator.py
│   └── identity_module.py
├── action/                     ← NEW (Phase 2), absorbs api/ + blockchain/ actions
│   ├── __init__.py
│   ├── tool_registry.py
│   ├── api_agent.py
│   ├── code_sandbox.py
│   ├── web_search.py
│   └── blockchain_tool.py
├── maintenance/                ← NEW (Phase 3), absorbs security/ + monitoring/
│   ├── __init__.py
│   ├── input_screener.py       ← NEW
│   ├── output_filter.py        ← NEW
│   ├── drift_detector.py       ← NEW
│   ├── self_repair.py          ← NEW
│   └── health_monitor.py       ← wraps monitoring/
├── quantum/                    ← NEW, absorbs integration/kinich + models/hybrid_llm
│   ├── __init__.py
│   ├── kinich_connector.py
│   ├── hybrid_llm.py
│   ├── quantum_memory.py       ← NEW (Phase 6)
│   ├── quantum_optimizer.py    ← NEW (Phase 6)
│   ├── quantum_anomaly.py      ← NEW (Phase 6)
│   └── quantum_imagination.py  ← NEW (Phase 6)
├── client/                     ← UNCHANGED (Nawal + domain_models)
├── server/                     ← UNCHANGED (FL aggregator)
├── blockchain/                 ← UNCHANGED
├── storage/                    ← UNCHANGED
├── training/                   ← UNCHANGED
├── data/                       ← UNCHANGED
└── monitoring/                 ← UNCHANGED (but also aliased in maintenance/)
```

### Phase 0 Tasks

| # | Task | Owner | Estimate | Notes |
|---|---|---|---|---|
| 0.1 | Create all new `__init__.py` files for empty modules | Any | 0.5 day | Run tests after each |
| 0.2 | Create `interfaces.py` ABCs in memory/, control/, perception/, valuation/, metacognition/ | 1 engineer | 1 day | Defines contracts before implementation |
| 0.3 | Add backward-compat aliases: `nawal.architecture` → `nawal.cortex` | Any | 0.5 day | Keeps existing imports working |
| 0.4 | Move `integration/kinich_connector.py` → `nawal/quantum/kinich_connector.py` | Any | 0.5 day | Update all imports |
| 0.5 | Move `models/hybrid_llm.py` → `nawal/quantum/hybrid_llm.py` | Any | 0.5 day | Update all imports |
| 0.6 | Run full test suite — all tests must still pass | Any | 1 day | — |
| 0.7 | Update `pyproject.toml` module list | Any | 0.5 day | — |

### Phase 0 Success Criteria
- `pytest tests/` passes with 0 regressions
- All new directories exist with placeholder `interfaces.py`
- `import nawal.memory`, `import nawal.control` etc. all resolve (even if empty)
- CI pipeline remains green

---

## Phase 1 — Memory System (Hippocampus) (Weeks 5–10)

### Goal
Give Nawal persistent memory beyond its 1024/2048-token context window.
After this phase, Nawal can remember past conversations, retrieve relevant knowledge,
and inject context into generation — without retraining.

### Why Memory First
Everything else depends on memory:
- Executive controller cannot plan across sessions without episodic memory
- Metacognition cannot track its own decisions without working memory
- Perception outputs are useless if they cannot be stored and recalled
- RAG is the single highest-leverage add: immediately improves answer quality

### Architecture

```
nawal/memory/
│
├── WorkingMemory           — in-process scratchpad for current task
│   • dict[str, any]  + stack trace of intermediate steps
│   • max 50 items, FIFO eviction
│   • cleared between conversation sessions
│
├── EpisodicMemory          — append-only log of past events
│   • SQLite store (conversations, decisions, outcomes)
│   • Each entry: timestamp, actor, action, context, embedding
│   • Queried by recency or semantic similarity
│   • Backed up to Pakit (IPFS) on session end
│
├── SemanticMemory          — vector database of knowledge
│   • ChromaDB (Phase 1) → Qdrant (production scale)
│   • Documents chunked to 512 tokens, embedded with Nawal encoder
│   • Collections: belize_corpus, laws, general, domain_specific
│   • Supports: exact match, semantic search, hybrid (BM25 + vector)
│
├── RAGPipeline             — retrieval-augmented generation
│   • Wraps SemanticMemory + query routing
│   • Injects top-k retrieved chunks into prompt context
│   • Deduplication + re-ranking before injection
│   • Respects max_position_embeddings budget
│
└── MemoryManager           — unified interface for all memory types
    • Used by ExecutiveController, HybridNawalEngine, InferencServer
```

### New Dependencies to Add

```toml
# pyproject.toml additions
chromadb>=0.5.0           # Vector DB (Phase 1)
qdrant-client>=1.12.0     # Production vector DB (Phase 1, optional)
rank-bm25>=0.2.2          # BM25 keyword retrieval for hybrid search
sentence-transformers>=3.0.0  # Embedding model for memory (sovereign after Phase 2)
```

### Implementation Detail

**`nawal/memory/interfaces.py`** (written in Phase 0):
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class MemoryRecord:
    id: str
    content: str
    embedding: list[float] | None
    metadata: dict[str, Any]
    timestamp: str
    source: str  # "episodic" | "semantic" | "working"
    score: float = 0.0  # Relevance score when retrieved

class AbstractMemory(ABC):
    @abstractmethod
    def store(self, content: str, metadata: dict) -> str: ...
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]: ...
    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...
    @abstractmethod
    def clear(self) -> None: ...
```

**`nawal/memory/working_memory.py`**:
```python
class WorkingMemory:
    """
    In-process scratchpad for multi-step task reasoning.
    Analogous to the brain's immediate working memory (< 1 minute timescale).
    """
    def __init__(self, max_items: int = 50):
        self._store: dict[str, Any] = {}
        self._history: list[dict] = []  # ordered trace
        self.max_items = max_items

    def set(self, key: str, value: Any) -> None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def push_step(self, step: str, result: Any) -> None:
        """Record a reasoning step for chain-of-thought tracing."""
    def get_trace(self) -> list[dict]: ...      # Return full reasoning trace
    def clear(self) -> None: ...               # Called at session end
    def summarize(self) -> str: ...            # Compress trace to a summary string
```

**`nawal/memory/episodic_memory.py`**:
```python
class EpisodicMemory:
    """
    Episodic + spatial memory — analogous to hippocampus.
    Records events with timestamps. Supports:
    - Replay (look back N sessions)
    - Semantic search over past experiences
    - Pakit backup for long-term persistence
    """
    def __init__(self, db_path: str = "./storage/episodic.db",
                 pakit_client=None,
                 embedding_fn: Callable = None): ...

    def record_event(self, actor: str, action: str,
                     content: str, context: dict = None) -> str: ...
    def get_recent(self, n: int = 10) -> list[EpisodicRecord]: ...
    def search(self, query: str, top_k: int = 5) -> list[EpisodicRecord]: ...
    def export_to_pakit(self) -> str: ...       # Returns IPFS CID
    def load_from_pakit(self, cid: str) -> int: ...  # Returns records loaded
```

**`nawal/memory/semantic_memory.py`**:
```python
class SemanticMemory:
    """
    Vector database backed semantic memory — world knowledge retrieval.
    Phase 1: ChromaDB.
    Phase 3: Migrate to Qdrant for production scale.
    """
    COLLECTIONS = ["belize_corpus", "laws", "government", "science",
                   "agritech", "marine", "education", "general"]

    def __init__(self, persist_dir: str = "./storage/vector_db",
                 embedding_fn: Callable = None,
                 collection: str = "general"): ...

    def ingest(self, documents: list[str],
               metadatas: list[dict] = None,
               collection: str = "general") -> int: ...   # Returns docs ingested
    def retrieve(self, query: str, top_k: int = 5,
                 collection: str = "general",
                 filter: dict = None) -> list[MemoryRecord]: ...
    def hybrid_search(self, query: str, top_k: int = 5) -> list[MemoryRecord]: ...
    def delete_collection(self, collection: str) -> bool: ...
```

**`nawal/memory/rag_pipeline.py`**:
```python
class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Injects retrieved context into the model prompt before generation.
    """
    def __init__(self, semantic_memory: SemanticMemory,
                 episodic_memory: EpisodicMemory,
                 max_context_tokens: int = 512,
                 top_k: int = 5): ...

    def build_context(self, query: str,
                      include_episodic: bool = True) -> str:
        """
        Retrieve relevant documents and format them as context string.
        Respects max_context_tokens budget.
        Deduplicates overlapping chunks.
        Ranks by relevance score.
        """

    def augment_prompt(self, original_prompt: str, query: str = None) -> str:
        """
        Insert retrieved context before the user prompt.
        Format: [CONTEXT]\n{context}\n[/CONTEXT]\n{original_prompt}
        """
```

**`nawal/memory/memory_manager.py`**:
```python
class MemoryManager:
    """
    Unified interface. The only import other modules need.
    """
    def __init__(self, config: MemoryConfig): ...

    # Write
    def remember(self, content: str, source: str = "conversation",
                 metadata: dict = None) -> str: ...

    # Read
    def recall(self, query: str, top_k: int = 5,
               memory_types: list[str] = None) -> list[MemoryRecord]: ...

    # Working memory (current task)
    def scratch(self, key: str, value: Any = None) -> Any: ...
    def get_trace(self) -> list[dict]: ...

    # Session lifecycle
    def begin_session(self, session_id: str) -> None: ...
    def end_session(self) -> None: ...   # Flush to episodic, backup to Pakit

    # Ingestion
    def ingest_documents(self, docs: list[str],
                         collection: str = "general") -> int: ...
    def ingest_belize_corpus(self) -> int: ...  # One-time corpus load
```

### Integration Points

1. **`hybrid/engine.py` (HybridNawalEngine)**: Before generating, call
   `rag_pipeline.augment_prompt(prompt)` to inject retriever context.
2. **`api/inference_server.py`**: Wrap with `memory_manager.begin_session()` /
   `memory_manager.end_session()`.
3. **`genome/fitness.py`**: Add a `memory_hit_rate` metric to FitnessScore.
   Genomes that make better use of retrieved context score higher.

### Phase 1 Tasks

| # | Task | Estimate | Dependencies |
|---|---|---|---|
| 1.1 | Implement `WorkingMemory` + tests | 2 days | None |
| 1.2 | Implement `EpisodicMemory` (SQLite + embeddings + Pakit) | 3 days | 1.1 |
| 1.3 | Implement `SemanticMemory` (ChromaDB) | 3 days | None |
| 1.4 | Ingest Belize corpus (`data/belize_corpus/`) into SemanticMemory | 1 day | 1.3 |
| 1.5 | Implement `RAGPipeline` | 2 days | 1.2, 1.3 |
| 1.6 | Implement `MemoryManager` (unifed API) | 1 day | 1.1–1.5 |
| 1.7 | Integrate RAG into `HybridNawalEngine.generate()` | 2 days | 1.5, 1.6 |
| 1.8 | Integrate `begin_session` / `end_session` into inference server | 1 day | 1.6, 1.7 |
| 1.9 | Add `memory_hit_rate` to FitnessScore | 0.5 day | 1.6 |
| 1.10 | Write integration tests (`test_memory.py`) | 2 days | 1.1–1.8 |

### Phase 1 Success Criteria
- RAG retrieval hits relevant Belize corpus document for > 80% of Belize-domain queries
- Nawal answers questions about previous sessions (episodic recall)
- Working memory trace is accessible and accurate for multi-step tasks
- Memory overhead < 100ms added latency to inference
- `pytest tests/test_memory.py` passes 100%

---

## Phase 2 — Executive Controller (Prefrontal Cortex) (Weeks 11–18)

### Goal
Give Nawal the ability to plan multi-step tasks, manage goals, decompose complex
requests, and use tools — all without human hand-holding per step.

### Architecture

```
nawal/control/
│
├── GoalStack               — priority queue of active objectives
│   • Priority levels: CRITICAL / HIGH / MEDIUM / LOW
│   • Handles: add_goal, pop_goal, peek_goal, list_active
│   • Persists active goals to working memory between steps
│
├── TaskPlanner             — break a goal into executable steps
│   • Strategy 1: Chain-of-Thought decomposition (default)
│   • Strategy 2: MCTS (tree search, for complex multi-branch tasks)
│   • Output: Plan (ordered list of Step objects)
│
├── ToolRegistry            — register and call external tools
│   • Tools: memory_search, web_search, run_code, blockchain_action,
│             oracle_query, domain_model_inference, math_solver
│   • Each tool: name, description, input_schema, output_schema, fn
│   • Auto-selects tool based on step description
│
├── ActionExecutor          — execute a Plan step by step
│   • Calls tools in sequence
│   • On failure: retry (max 3), fallback, or escalate
│   • Records each step result in WorkingMemory
│
└── ExecutiveController     — ties GoalStack + TaskPlanner + ActionExecutor
    • Main interface for all planning + execution
    • Integrates with MemoryManager for context injection
    • Integrates with HybridNawalEngine for reasoning steps
```

### Key Data Structures

```python
@dataclass
class Goal:
    id: str
    description: str
    priority: int         # 1 (CRITICAL) to 5 (LOW)
    created_at: str
    deadline: str | None
    parent_goal_id: str | None    # For sub-goals
    status: str           # "pending" | "active" | "completed" | "failed"

@dataclass
class Step:
    id: str
    description: str
    tool: str | None      # Tool to use, or None if just reasoning
    tool_args: dict
    expected_output: str  # Natural language description of expected result
    status: str
    result: Any | None

@dataclass
class Plan:
    goal_id: str
    steps: list[Step]
    created_at: str
    estimated_cost: float  # DALLA cost estimate
    estimated_time: float  # seconds

@dataclass
class ExecutionResult:
    plan: Plan
    success: bool
    final_answer: str
    steps_completed: int
    total_steps: int
    duration: float
    error: str | None
```

### Tool Implementations (nawal/action/)

```python
# nawal/action/tool_registry.py
class ToolRegistry:
    """
    Central registry for all tools Nawal can call.
    Each tool is a callable with a defined schema.
    """
    built_in_tools = [
        "memory_search",      # → MemoryManager.recall()
        "memory_store",       # → MemoryManager.remember()
        "run_code",           # → CodeSandbox.execute()
        "web_search",         # → WebSearchTool.search()
        "blockchain_action",  # → BlockchainTool.*
        "oracle_query",       # → OraclePipeline.query()
        "domain_inference",   # → DomainModelFactory.run()
        "math_solver",        # → sympy or wolframalpha
    ]

    def register(self, name: str, fn: Callable,
                 description: str, schema: dict) -> None: ...
    def call(self, tool_name: str, **kwargs) -> ToolResult: ...
    def list_tools(self) -> list[ToolDescription]: ...
    def get_schema_for_planner(self) -> str: ...  # JSON schema for prompt injection
```

```python
# nawal/action/code_sandbox.py
class CodeSandbox:
    """
    Sandboxed Python code execution.
    Uses RestrictedPython (Phase 2) → Docker container (Phase 3 production).
    """
    ALLOWED_MODULES = ["math", "json", "statistics", "datetime",
                       "re", "itertools", "functools"]
    BLOCKED_MODULES = ["os", "subprocess", "sys", "socket", "importlib"]
    MAX_EXECUTION_TIME = 10  # seconds

    def execute(self, code: str, context: dict = None) -> CodeResult: ...
    def is_safe(self, code: str) -> tuple[bool, list[str]]: ...
```

### Chain-of-Thought Task Planning

The `TaskPlanner` uses Nawal itself to decompose goals:

```
SYSTEM: You are Nawal's planning module. Given a goal, decompose it into
        ordered steps. For each step, specify which tool to use.

        Available tools:
        {tool_registry.get_schema_for_planner()}

USER: Goal: {goal.description}

NAWAL OUTPUT (structured):
Step 1: search_memory for context on "{topic}"
Step 2: run_code to compute {calculation}
Step 3: memory_store result with metadata
Step 4: blockchain_action submit result to oracle pallet
```

The planner parses this output into a typed `Plan` object.

### Integration Points

1. **`HybridNawalEngine`**: After routing decision, before returning, check if goal
   requires multi-step execution. If yes, hand off to `ExecutiveController`.
2. **`MemoryManager`**: Each `ActionExecutor` step writes to `WorkingMemory`.
   After plan completion, key results written to `EpisodicMemory`.
3. **`OraclePipeline`**: Exposed as a tool in `ToolRegistry`.
4. **`blockchain/`**: All staking, rewards, genome registry operations exposed as tools.

### Phase 2 Tasks

| # | Task | Estimate | Dependencies |
|---|---|---|---|
| 2.1 | Implement `GoalStack` + tests | 1 day | Phase 0 |
| 2.2 | Implement `TaskPlanner` (CoT decomposition) | 3 days | 2.1, Phase 1 |
| 2.3 | Implement `ToolRegistry` + built-in tool wrappers | 3 days | Phase 1 |
| 2.4 | Implement `CodeSandbox` (RestrictedPython) | 2 days | 2.3 |
| 2.5 | Implement `WebSearchTool` (DuckDuckGo / SearX) | 1 day | 2.3 |
| 2.6 | Implement `BlockchainTool` (wraps existing blockchain/) | 1 day | 2.3 |
| 2.7 | Implement `ActionExecutor` (step-by-step with retry) | 3 days | 2.2, 2.3 |
| 2.8 | Implement `ExecutiveController` (GoalStack + Planner + Executor) | 2 days | 2.1–2.7 |
| 2.9 | Integrate `ExecutiveController` into `HybridNawalEngine` | 2 days | 2.8 |
| 2.10 | Write `test_control.py` integration tests | 2 days | 2.1–2.9 |

### Phase 2 Success Criteria
- Nawal solves a 3-step task (search memory → compute → blockchain submit) autonomously
- `TaskPlanner` decomposes 90% of test goals into valid, executable plans
- `CodeSandbox` blocks all dangerous code patterns (injection tests pass)
- Response latency for 3-step plan < 10 seconds end-to-end
- All existing tests still pass (no regressions)

---

## Phase 3 — Valuation & Safety (Limbic + Immune) (Weeks 19–24)

### Goal
Give Nawal explicit good/bad signals beyond PoUW fitness, and protect users with
input and output safety layers. This phase is **required before any public deployment.**

### Architecture

```
nawal/valuation/
│
├── PreferenceStore         — log human A/B preference pairs for DPO training
├── RewardModel             — trained on preferences, scores any response
├── IntrinsicReward         — curiosity + novelty + consistency drives
└── ValuationAggregator     — combine reward + intrinsic + fitness → final score

nawal/maintenance/
│
├── InputScreener           — detect prompt injection, jailbreaks, unsafe inputs
├── OutputFilter            — post-generation safety classifier
├── DriftDetector           — detect model behavior change from baseline
└── SelfRepair              — rollback to last safe checkpoint on anomaly
```

### Reward Model (DPO Approach)

**Why DPO over PPO**: Direct Preference Optimization trains a policy directly from
preference pairs without a separate reward model training loop. Simpler, more stable.

```python
# nawal/valuation/reward_model.py
class RewardModel:
    """
    Scores prompt+response pairs. Trained via DPO on human preference data.
    Initially: use a small cross-encoder fine-tuned on synthetic Belize preferences.
    Later: replace with DPO-trained Nawal weights.
    """
    def score(self, prompt: str, response: str) -> float: ...       # → 0.0 to 1.0
    def rank(self, prompt: str, responses: list[str]) -> list[str]: ...
    def log_preference(self, prompt: str, chosen: str, rejected: str): ...
    def train_dpo(self, num_steps: int = 1000) -> TrainingResult: ...
```

**Connecting Reward Model to Genome Fitness**:

In `genome/fitness.py`, the `FitnessScore.quality` component is currently based
on training loss only. After Phase 3:

```python
# NEW: quality = 0.6 * training_loss_score + 0.4 * reward_model_score
fitness.quality = (
    0.6 * compute_training_quality(model, dataset) +
    0.4 * reward_model.score(sample_prompt, model.generate(sample_prompt))
)
```

This closes the loop: **human preferences → reward signal → genome fitness → evolution**.

### Safety Layers

```python
# nawal/maintenance/input_screener.py
class InputScreener:
    """
    Screens all incoming prompts before they reach the model.
    Detects:
    - Prompt injection (attempts to override system instructions)
    - Jailbreaks (attempts to bypass safety guidelines)
    - Malicious code (if code execution is requested)
    - PII leakage risk (accidentally routing private data)
    """
    BLOCKED_PATTERNS = [...]   # Regex + semantic patterns

    def screen(self, prompt: str) -> ScreeningResult:
        """
        Returns:
            ScreeningResult(
                is_safe: bool,
                risk_level: str,    # "none" | "low" | "medium" | "high" | "blocked"
                flags: list[str],   # What was detected
                sanitized: str,     # Cleaned version if safe enough
            )
        """

# nawal/maintenance/output_filter.py
class OutputFilter:
    """
    Screens all generated outputs before returning to user.
    Detects:
    - Harmful content (violence, self-harm, illegal instructions)
    - PII exposure (names, addresses, credentials)
    - Hallucinated facts (cross-check against memory/knowledge base)
    - Confidential data leakage
    """
    def filter(self, prompt: str, response: str) -> FilterResult: ...
    def is_safe(self, response: str) -> bool: ...
```

### Drift Detector + Self-Repair

```python
# nawal/maintenance/drift_detector.py
class DriftDetector:
    """
    Monitors model behavior over time.
    Compares current model outputs against baseline (last known good checkpoint).
    Alerts if:
    - Average confidence score drops > 15%
    - Safety filter block rate increases > 5%
    - Response perplexity increases > 20% vs baseline
    - Byzantine detection anomaly rate spikes
    """
    def record_baseline(self, checkpoint_id: str) -> None: ...
    def check(self) -> DriftReport: ...
    def is_drifted(self) -> bool: ...

# nawal/maintenance/self_repair.py
class SelfRepair:
    """
    Triggered when DriftDetector.is_drifted() returns True.
    Actions:
    1. Alert operators via monitoring system
    2. Log drift event to EpisodicMemory
    3. If auto_repair=True: rollback to last known good checkpoint
    4. If auto_repair=False: pause inference, wait for human intervention
    """
    def repair(self, strategy: str = "rollback") -> RepairResult: ...
    def rollback(self, checkpoint_id: str) -> bool: ...
```

### Integration Points

1. **`api_server.py`**: Wrap all `/inference` endpoints with
   `InputScreener.screen()` (before) and `OutputFilter.filter()` (after).
2. **`genome/fitness.py`**: Add `reward_model.score()` to quality component.
3. **`monitoring/`**: `DriftDetector` feeds into Prometheus metrics.
4. **`storage/checkpoint_manager.py`**: `SelfRepair.rollback()` uses existing
   `CheckpointManager` to restore weights.

### Phase 3 Tasks

| # | Task | Estimate | Dependencies |
|---|---|---|---|
| 3.1 | Implement `PreferenceStore` (SQLite for A/B pairs) | 1 day | Phase 0 |
| 3.2 | Build synthetic Belize preference dataset (200 pairs) | 2 days | 3.1 |
| 3.3 | Implement `RewardModel` (cross-encoder, train on 3.2) | 4 days | 3.1, 3.2 |
| 3.4 | Implement `IntrinsicReward` (curiosity + novelty scores) | 2 days | Phase 1 (memory) |
| 3.5 | Implement `ValuationAggregator` + connect to FitnessScore | 1 day | 3.3, 3.4 |
| 3.6 | Implement `InputScreener` (regex + semantic patterns) | 3 days | None |
| 3.7 | Implement `OutputFilter` (harm classifier) | 3 days | 3.3 |
| 3.8 | Implement `DriftDetector` + Prometheus integration | 2 days | Phase 0 |
| 3.9 | Implement `SelfRepair` (rollback via CheckpointManager) | 1 day | 3.8 |
| 3.10 | Integrate `InputScreener` and `OutputFilter` into API | 1 day | 3.6, 3.7 |
| 3.11 | Write `test_valuation.py` + `test_safety.py` | 3 days | 3.1–3.10 |

### Phase 3 Success Criteria
- `InputScreener` blocks > 95% of prompt injection test vectors (standard red-team set)
- `OutputFilter` catches > 90% of synthetic harmful outputs
- `RewardModel` agrees with human preference raters > 80% (held-out 50 pair test)
- `DriftDetector` triggers correctly on injected model corruption
- `SelfRepair` rolls back successfully to last good checkpoint in all test cases
- SafetyFilter adds < 50ms overhead to inference

---

## Phase 4 — Metacognitive Layer (Default Mode Network) (Weeks 25–30)

### Goal
Give Nawal the ability to evaluate its own outputs, catch errors before returning them,
simulate consequences of plans before executing, and maintain a stable identity over time.

### Architecture

```
nawal/metacognition/
│
├── SelfCritic              — critique own outputs before returning
├── ConsistencyChecker      — sample N outputs, measure agreement
├── ConfidenceCalibrator    — calibrate confidence scores vs real accuracy
├── InternalSimulator       — simulate plan consequences before executing
└── IdentityModule          — persistent self-description and capability registry
```

### Self-Critique Loop

```
User query
    │
    ▼
HybridNawalEngine.generate() → draft_response
    │
    ▼
SelfCritic.critique(prompt, draft_response) → Critique(score, issues, suggestions)
    │
    ├── If critique.score >= 0.8: return draft_response
    │
    └── If critique.score < 0.8:
            │
            ▼
        HybridNawalEngine.generate(
            prompt + CRITIQUE_CONTEXT + critique.suggestions
        ) → refined_response
            │
            ▼
        Return refined_response (max 2 critique cycles to prevent loops)
```

```python
# nawal/metacognition/self_critic.py
@dataclass
class Critique:
    score: float               # 0.0 to 1.0
    issues: list[str]          # Identified problems
    suggestions: list[str]     # How to improve
    factual_errors: list[str]  # Potential hallucinations
    reasoning_gaps: list[str]  # Missing logic steps

class SelfCritic:
    """
    Uses Nawal itself (with a special system prompt) to critique its outputs.
    Cost: doubles inference calls when triggered.
    Mode: "off" | "on_low_confidence" (default) | "always"
    """
    def __init__(self, engine: HybridNawalEngine,
                 mode: str = "on_low_confidence",
                 threshold: float = 0.6): ...

    def critique(self, prompt: str, response: str) -> Critique: ...
    def refine(self, prompt: str, response: str,
               critique: Critique) -> str: ...
    def critique_and_refine(self, prompt: str,
                             initial_response: str,
                             max_cycles: int = 2) -> str: ...
```

### Consistency Checker

For high-stakes decisions (plan selection, factual claims), sample N outputs
and measure how much they agree:

```python
class ConsistencyChecker:
    """
    Samples N independently generated responses to the same prompt.
    Measures agreement. High disagreement = low confidence.
    Used by ExecutiveController for plan selection.
    """
    def check(self, prompt: str, n_samples: int = 5,
              temperature: float = 0.8) -> ConsistencyResult: ...
    # Returns: majority_answer, confidence, disagreement_examples
```

### Internal Simulator (Pre-Action Rollout)

Before `ActionExecutor` executes a `Plan`, optionally simulate it:

```python
class InternalSimulator:
    """
    Simulates plan execution without actually executing it.
    Uses Nawal as a world model: "If I do step X, what likely happens?"
    """
    def simulate(self, current_state: dict,
                 plan: Plan,
                 steps_ahead: int = 3) -> list[SimulatedState]: ...
    def estimate_risk(self, plan: Plan) -> RiskAssessment: ...
```

### Identity Module

```python
class IdentityModule:
    """
    Maintains Nawal's persistent self-model.
    Tracks: capabilities, limitations, past decisions, values.
    Used in system prompts to ground Nawal's responses in its actual identity.
    """
    DEFAULT_IDENTITY = {
        "name": "Nawal",
        "origin": "Belize",
        "purpose": "Sovereign AI for the Belizean people",
        "languages": ["en", "es", "bzj", "cab", "mop"],
        "capabilities": [],        # Populated dynamically from ToolRegistry
        "limitations": [],         # Updated when errors/failures occur
        "values": ["sovereignty", "transparency", "service", "honesty"],
        "version": "1.0.0",
    }

    def get_system_prompt(self) -> str: ...     # Nawal's self-description for prompts
    def record_capability(self, cap: str) -> None: ...
    def record_limitation(self, lim: str) -> None: ...
    def get_capability_matrix(self) -> dict: ...
```

### Integration Points

1. **`HybridNawalEngine.generate()`**: Add optional `self_critique=True` argument.
   When True: runs `SelfCritic.critique_and_refine()`.
2. **`ExecutiveController.execute()`**: Before running high-cost plans, call
   `InternalSimulator.estimate_risk()`. Abort if risk > threshold.
3. **`api/inference_server.py`**: Inject `IdentityModule.get_system_prompt()`
   into the system prompt for all requests.
4. **`hybrid/confidence.py`**: Replace simple scalar with `ConfidenceCalibrator`
   calibrated output.

### Phase 4 Tasks

| # | Task | Estimate | Dependencies |
|---|---|---|---|
| 4.1 | Implement `SelfCritic` + system prompt design | 3 days | Phase 2 |
| 4.2 | Implement `ConsistencyChecker` | 2 days | Phase 2 |
| 4.3 | Implement `ConfidenceCalibrator` (isotonic regression) | 2 days | Phase 3 |
| 4.4 | Implement `InternalSimulator` | 3 days | Phase 2 |
| 4.5 | Implement `IdentityModule` + system prompt integration | 2 days | None |
| 4.6 | Integrate `SelfCritic` into `HybridNawalEngine` | 1 day | 4.1 |
| 4.7 | Integrate `InternalSimulator` into `ExecutiveController` | 1 day | 4.4 |
| 4.8 | Integrate `IdentityModule` into inference server | 1 day | 4.5 |
| 4.9 | Write `test_metacognition.py` | 2 days | 4.1–4.8 |

### Phase 4 Success Criteria
- Self-critique loop improves answer score > 10% on a held-out Belize eval set
- `ConsistencyChecker` correctly flags contradictory multi-sample outputs
- `IdentityModule` system prompt is used in 100% of inference requests
- `InternalSimulator` aborts > 95% of plans that would fail in real execution
- Inference latency with `mode="on_low_confidence"` adds < 2s on average

---

## Phase 5 — Perception Layer (Sensory Cortices) (Weeks 31–40)

### Goal
Nawal perceives images and processes audio, not just text.
After this phase, it can answer questions about Belize satellite imagery, drone footage,
document scans, and spoken queries in Kriol.

### Architecture

```
nawal/perception/
│
├── VisualCortex            — encode images into embeddings
│   • Backbone: CLIP ViT-B/32 (open MIT + Apache)
│   • Optional: fine-tune on Belize imagery dataset
│   • Output: torch.Tensor [hidden_dim]
│
├── AuditoryCortex          — encode audio / transcribe speech
│   • Backbone: Whisper-base (MIT license)
│   • Fine-tune: Kriol + Garifuna + Mopan speech data
│   • Output: transcription (str) + audio embeddings
│
├── MultimodalCortex        — fuse text + image + audio embeddings
│   • Cross-attention projection layer (trained from scratch)
│   • Aligns visual/audio embeddings to Nawal's text embedding space
│   • Output: world_state_embedding [hidden_dim] — same space as Nawal's tokens
│
└── SensoryHub              — unified entry point
    • Accepts any combination of text/image/audio
    • Routes to relevant cortex
    • Returns fused world_state_embedding
```

### Key Interface

```python
# nawal/perception/sensory_hub.py
from PIL import Image
import numpy as np

class SensoryHub:
    """
    Unified perception interface.
    Accepts text, image, audio in any combination.
    Returns a single world-state embedding in Nawal's hidden space.
    """
    def __init__(self, config: PerceptionConfig,
                 device: str = "auto"): ...

    def encode(
        self,
        text: str | None = None,
        image: Image.Image | np.ndarray | None = None,
        audio: np.ndarray | None = None,   # 16kHz mono float32
        audio_path: str | None = None,
    ) -> torch.Tensor:                     # → [hidden_dim] world-state embedding
        """Fuse all available modalities into a single embedding."""

    def transcribe(self, audio: np.ndarray) -> str:
        """Speech-to-text (Whisper backend)."""

    def describe_image(self, image: Image.Image) -> str:
        """Caption an image using visual cortex + Nawal decoder."""
```

### New Dependencies

```toml
# pyproject.toml additions for Phase 5
Pillow>=11.0.0                        # Image loading
openai-whisper>=20231117              # Audio transcription (MIT)
open-clip-torch>=2.26.1              # CLIP ViT (open source)
torchaudio>=2.4.0                    # Audio processing
soundfile>=0.12.1                    # Audio file I/O
```

### Integration with IoT Oracle Pipeline

`integration/oracle_pipeline.py` already ingests data from:
- Drones → camera images
- Sensors → numerical arrays
- Cameras → image streams

After Phase 5, these inputs flow through `SensoryHub`:
```python
# In oracle_pipeline.py (existing), add:
world_state = sensory_hub.encode(
    image=drone_image,
    text=metadata_description
)
prediction = domain_model.predict_from_embedding(world_state)
```

### Phase 5a: Vision (Weeks 31–35)
### Phase 5b: Audio (Weeks 36–40)

| # | Task | Estimate | Phase |
|---|---|---|---|
| 5.1 | Implement `VisualCortex` (CLIP backbone) + tests | 3 days | 5a |
| 5.2 | Implement `MultimodalCortex` (cross-attn fusion) | 4 days | 5a |
| 5.3 | Implement `SensoryHub` (image + text paths) | 2 days | 5a |
| 5.4 | Fine-tune VisualCortex on Belize aerial/marine imagery | 5 days | 5a |
| 5.5 | Integrate `SensoryHub` into `OraclePipeline` | 2 days | 5a |
| 5.6 | Implement `AuditoryCortex` (Whisper backbone) | 3 days | 5b |
| 5.7 | Fine-tune Whisper on Kriol speech | 5 days | 5b |
| 5.8 | Add audio path to `SensoryHub` | 2 days | 5b |
| 5.9 | Add `/api/v1/perceive` multimodal endpoint to API server | 2 days | 5b |
| 5.10 | Write `test_perception.py` | 2 days | 5a + 5b |

### Phase 5 Success Criteria
- `SensoryHub.encode(image=belize_aerial)` returns valid embeddings
- Nawal answers questions about Belize satellite images with > 70% accuracy
- Whisper fine-tuned on Kriol achieves < 25% WER on Kriol speech test set
- `OraclePipeline` successfully processes drone image inputs end-to-end
- Vision overhead < 500ms per image on GPU

---

## Phase 6 — Quantum Integration (Kinich Full Activation) (Weeks 41–52)

### Goal
Activate the quantum layer for real computation.
Connect to a live Kinich quantum node and replace classical fallbacks
with quantum-accelerated alternatives in 4 areas:
memory retrieval, plan optimization, anomaly detection, and imagination/simulation.

### Prerequisites
- Live Kinich quantum node available at a real endpoint (not localhost:8002)
- Kinich API documentation / SDK received from Kinich team
- `KinichQuantumConnector` tested end-to-end with real quantum hardware

### 6a — Quantum Memory Search (Weeks 41–44)

Replaces classical FAISS top-k search with Grover-inspired quantum search
for very large memory stores (> 1 million records).

```python
# nawal/quantum/quantum_memory.py
class QuantumMemory:
    """
    Quantum-accelerated memory retrieval.
    Uses Kinich QNN to find similar vectors faster than classical methods
    for large corpora. Falls back to classical FAISS if Kinich unavailable.
    """
    def search(
        self,
        query_vector: np.ndarray,  # [dim] query embedding
        top_k: int = 5,
        corpus_vectors: np.ndarray = None,  # [N, dim] if small enough to send
    ) -> list[int]:  # Returns indices of top-k matches
        if kinich_available and len(corpus) > QUANTUM_THRESHOLD:
            return self._quantum_search(query_vector, top_k)
        return self._classical_search(query_vector, top_k)
```

### 6b — Quantum Plan Optimizer (Weeks 45–47)

Replaces greedy plan selection in `TaskPlanner` with QAOA-based optimization
when there are many competing candidate plans.

```python
# nawal/quantum/quantum_optimizer.py
class QuantumPlanOptimizer:
    """
    Uses QAOA (Quantum Approximate Optimization Algorithm) via Kinich
    to select the best plan from many candidates, optimizing multiple objectives:
    - Minimize steps
    - Maximize expected reward
    - Minimize DALLA cost
    - Minimize risk
    """
    def select_best_plan(
        self,
        candidate_plans: list[Plan],
        objectives: list[Objective],
        constraints: list[Constraint] = None,
    ) -> Plan: ...

    def rank_plans(
        self,
        candidate_plans: list[Plan],
        objectives: list[Objective],
    ) -> list[Plan]:  # Sorted best → worst
        ...
```

### 6c — Quantum Anomaly Detection (Weeks 48–50)

Enhances `DriftDetector` in `maintenance/` with a quantum kernel SVM
for detecting out-of-distribution system behavior in high-dimensional telemetry.

```python
# nawal/quantum/quantum_anomaly.py
class QuantumAnomalyDetector:
    """
    Quantum kernel SVM for anomaly detection on high-dimensional telemetry.
    Classical fallback: Isolation Forest.
    """
    def fit(self, normal_telemetry: np.ndarray) -> None: ...
    def predict(self, telemetry: np.ndarray) -> list[bool]: ...  # True = anomaly
    def score(self, telemetry: np.ndarray) -> np.ndarray: ...   # Anomaly scores
```

### 6d — Quantum Imagination Engine (Weeks 51–52, Research Track)

Enhances `InternalSimulator` with quantum parallel rollouts —
different plan branches simulated in superposition.

```python
# nawal/quantum/quantum_imagination.py
class QuantumImagination:
    """
    Research-grade. Uses quantum sampling to explore many plan rollouts
    in parallel rather than sequentially. Results used by InternalSimulator
    to estimate plan risk and diversity of outcomes.
    """
    def sample_futures(
        self,
        current_state: dict,
        possible_actions: list[Step],
        n_samples: int = 10,
    ) -> list[SimulatedState]: ...
```

### Phase 6 Tasks

| # | Task | Estimate | Gate |
|---|---|---|---|
| 6.0 | Verify Kinich live endpoint + integration test | 3 days | Kinich node available |
| 6.1 | Implement `QuantumMemory.search()` with classical fallback | 3 days | 6.0 |
| 6.2 | Validate quantum vs classical search agreement on test corpus | 2 days | 6.1 |
| 6.3 | Implement `QuantumPlanOptimizer` with classical fallback | 4 days | 6.0 |
| 6.4 | Implement `QuantumAnomalyDetector` | 3 days | 6.0 |
| 6.5 | Integrate `QuantumAnomalyDetector` into `DriftDetector` | 1 day | 6.4 |
| 6.6 | Implement `QuantumImagination` (research track) | 5 days | 6.0 |
| 6.7 | Benchmark quantum vs classical for all 4 modules | 3 days | 6.1–6.6 |
| 6.8 | Write `test_quantum.py` | 2 days | 6.1–6.6 |

### Phase 6 Success Criteria
- Quantum memory search returns same top-k as FAISS on test corpus (> 95% agreement)
- Quantum plan optimizer selects plans rated by human reviewers as better than greedy > 70% of time
- Quantum anomaly detector matches classical Isolation Forest recall (baseline)
- All quantum modules degrade gracefully to classical when Kinich unavailable
- No quantum module adds > 5× classical latency at relevant corpus sizes

---

## 13. Module Interface Contracts

### The five golden rules for all new modules:

1. **Every module has `interfaces.py` with ABCs defined before implementation**
2. **All quantum operations are behind classical Python APIs** — caller never knows if result is classical or quantum
3. **Every module has a `from_config(config: dict)` factory classmethod**
4. **All I/O types are Pydantic v2 models** — no bare dicts at module boundaries
5. **All modules respect `fallback_to_classical: bool = True`** — quantum is always optional

### Cross-module data flow (canonical types)

```python
# nawal/types.py — canonical shared types
from pydantic import BaseModel

class WorldState(BaseModel):
    """Output of SensoryHub — input to Cerebrum."""
    text: str | None = None
    image_embedding: list[float] | None = None
    audio_transcript: str | None = None
    fused_embedding: list[float] | None = None
    modalities: list[str] = []  # which modalities are present
    timestamp: str

class GenerationResult(BaseModel):
    """Output of HybridNawalEngine — flows through all layers."""
    prompt: str
    response: str
    model_used: str           # "nawal" | "deepseek"
    confidence: float
    latency_ms: float
    memory_context_used: bool
    critique_applied: bool
    safety_passed: bool
    metadata: dict = {}

class MemoryRecord(BaseModel):
    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict = {}
    timestamp: str
    source: str
    score: float = 0.0

class Plan(BaseModel):
    goal_id: str
    steps: list[Step]
    estimated_cost_dalla: float
    estimated_time_seconds: float
    risk_score: float = 0.0

class FeedbackSignal(BaseModel):
    """Used by RewardModel and genome FitnessScore."""
    prompt: str
    response: str
    reward_score: float      # 0.0 to 1.0 from RewardModel
    human_rating: float | None = None  # If available
    safety_score: float
    consistency_score: float
    memory_utilized: bool
```

---

## 14. Dependency & Technology Decisions

| Decision | Choice | Rationale | Alternative |
|---|---|---|---|
| Vector DB (Phase 1) | **ChromaDB** | Zero-ops, local, pure Python, BSD license | Qdrant (migrate at production scale) |
| Vector DB (Production) | **Qdrant** | Scalable server, Rust core, gRPC API, Docker | Pinecone (cloud, non-sovereign) |
| Embedding model | **Nawal encoder** (own hidden states) | Maximum sovereignty — no external embedding call | sentence-transformers/all-MiniLM-L6-v2 (Phase 1 fallback) |
| RLHF strategy | **DPO** (Direct Preference Optimization) | Simpler than PPO, no reward model collapse risk | PPO (more powerful but unstable) |
| Safety classifier | **Custom fine-tuned** distilbert-base | Small, fast, can be trained on Belizean norms | OpenAI moderation API (non-sovereign) |
| Vision backbone | **CLIP ViT-B/32** (OpenCLIP) | Apache license, well-tested, good zero-shot | Train from scratch (sovereignty — 12 months extra) |
| Audio backbone | **Whisper-base** (OpenAI) | MIT license, multilingual, proven | Wav2Vec2 |
| Code sandbox | **RestrictedPython** → **Docker** | RestrictedPython for Phase 2 speed; Docker for production safety | Firecracker microVMs (later) |
| Web search | **SearXNG** (self-hosted) | Sovereign, no Google API dependency, FOSS | DuckDuckGo API |
| Quantum SDK | **PennyLane** (Kinich SDK) | Framework-agnostic, supports Kinich backend | Qiskit (IBM-centric) |
| Database (episodic) | **SQLite → PostgreSQL** | SQLite for Phase 1 simplicity; migrate to Postgres at production | MongoDB |

---

## 15. Testing Strategy Per Phase

### Test Structure (all phases)

```
tests/
├── test_memory.py              ← Phase 1
├── test_control.py             ← Phase 2
├── test_valuation.py           ← Phase 3
├── test_safety.py              ← Phase 3
├── test_metacognition.py       ← Phase 4
├── test_perception.py          ← Phase 5
├── test_quantum.py             ← Phase 6
├── test_integration_brain.py   ← End-to-end brain pipeline test
└── (existing tests unchanged)
    test_genome.py
    test_evolution.py
    test_byzantine_detection.py
    test_differential_privacy.py
    test_federation.py
    test_blockchain.py
    test_training.py
    ...
```

### Key Integration Test: `test_integration_brain.py`

This test runs the **full brain pipeline** end-to-end for a sample task:

```python
def test_full_brain_pipeline():
    """
    Tests the full Nawal brain pipeline:
    1. Input goes through InputScreener (immune)
    2. Sensory Hub encodes it (perception)
    3. Memory Manager augments with retrieved context (hippocampus)
    4. Executive Controller plans (prefrontal)
    5. HybridNawalEngine generates (cerebrum)
    6. SelfCritic evaluates (metacognition)
    7. OutputFilter screens result (immune)
    8. RewardModel scores, logs to preference store (limbic)
    9. Result returned to user
    """
    # Input: "What are the best crops for the Cayo District given recent rainfall?"
    # Expected: Retrieves AgriTech domain knowledge, generates relevant answer
    ...
```

### Per-Phase Test Requirements

| Phase | Minimum Coverage | Key Tests |
|---|---|---|
| Phase 0 | 100% import coverage | All modules import cleanly |
| Phase 1 | 85% | RAG retrieval accuracy, episodic recall, working memory trace |
| Phase 2 | 80% | 3-step plan execution, tool calls, code sandbox blocking |
| Phase 3 | 90% | Safety filter injection tests (OWASP LLM top-10), reward model accuracy |
| Phase 4 | 80% | Self-critique improvement rate, identity module in prompts |
| Phase 5 | 80% | Image encoding correctness, Whisper Kriol WER |
| Phase 6 | 85% | Classical fallback works when Kinich unavailable |

---

## 16. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Kinich quantum node not available by Phase 6 | HIGH | LOW | All quantum modules have classical fallback. Phase 6 can be pushed. |
| Whisper fine-tuning on Kriol data underperforms | MEDIUM | MEDIUM | Collect more Kriol audio. Use Whisper-small as starting point instead of base. |
| Self-critique loop doubles latency and users complain | HIGH | MEDIUM | Default mode: `on_low_confidence` only. Full critique only for high-stakes queries. |
| DPO training on synthetic preferences introduces bias | MEDIUM | HIGH | Human review of all synthetic preference pairs before training. Add red-team review step. |
| ChromaDB hits scaling limit in production | HIGH (eventual) | MEDIUM | Plan Qdrant migration from day 1. Write a migration script in Phase 1. |
| Genome fitness + reward model loop produces reward hacking | MEDIUM | HIGH | Cap reward model contribution to 40% of quality metric. Include diverse eval set. Monitor FitnessScore distribution. |
| Memory RAG retrieval injects irrelevant context and degrades quality | MEDIUM | MEDIUM | Re-ranker step before injection. Confidence threshold on retrieval. Ablation test with/without RAG. |
| Belize corpus too small for meaningful semantic memory | HIGH | MEDIUM | Augment from legal documents, gov datasets, Wikipedia Belize, Belizean news archives. |
| Tool-use agent takes dangerous blockchain actions | LOW | CRITICAL | All `blockchain_action` tool calls require explicit human confirmation flag unless in approved list. Default: read-only tools only. |
| Code sandbox escape (RestrictedPython bypass) | MEDIUM | HIGH | Move to Docker sandbox for Phase 3. Add output content scanning on all sandbox results. |

---

## 17. Success Metrics Summary

### By Phase — Quantified Targets

| Phase | Metric | Target |
|---|---|---|
| **P0** | Import test pass rate | 100% |
| **P0** | Existing test regression rate | 0% |
| **P1** | RAG retrieval precision (Belize domain) | > 80% |
| **P1** | Episodic memory recall accuracy | > 85% |
| **P1** | RAG inference latency overhead | < 100ms |
| **P2** | 3-step task autonomous completion rate | > 90% |
| **P2** | Code sandbox injection blocking rate | 100% |
| **P2** | Tool-use inference latency (3-step) | < 10s |
| **P3** | Prompt injection blocking rate | > 95% |
| **P3** | Output harm detection rate | > 90% |
| **P3** | Reward model agreement with humans | > 80% |
| **P3** | Safety filter latency overhead | < 50ms |
| **P4** | Self-critique quality improvement | > 10% on eval set |
| **P4** | InternalSimulator plan failure prediction accuracy | > 95% |
| **P5** | Belize aerial image Q&A accuracy | > 70% |
| **P5** | Kriol speech WER | < 25% |
| **P5** | Vision inference latency | < 500ms on GPU |
| **P6** | Quantum vs classical memory search agreement | > 95% |
| **P6** | Classical fallback rate (when Kinich down) | 100% |
| **System** | End-to-end sovereign response rate (Nawal handles, not DeepSeek) | > 90% (month 12) |
| **System** | Average end-to-end inference latency | < 3s (month 12) |

---

## Appendix A — Week-by-Week Schedule

```
Weeks 1–4:   Phase 0: Refactor
Weeks 5–10:  Phase 1: Memory
Weeks 11–18: Phase 2: Executive Controller + Tool Use
Weeks 19–24: Phase 3: Valuation + Safety
Weeks 25–30: Phase 4: Metacognition
Weeks 31–35: Phase 5a: Vision
Weeks 36–40: Phase 5b: Audio
Weeks 41–44: Phase 6a: Quantum Memory
Weeks 45–47: Phase 6b: Quantum Optimizer
Weeks 48–50: Phase 6c: Quantum Anomaly
Weeks 51–52: Phase 6d: Quantum Imagination (research)
```

---

## Appendix B — Brain Capability Matrix (End State)

| Brain Region | Capability | Implemented After |
|---|---|---|
| 🧠 Cerebrum | Language, reasoning, generation | Already done |
| 🧬 DNA | Architecture evolution, genome heredity | Already done |
| 🧠 Hippocampus | Episodic memory, semantic RAG, working scratchpad | Phase 1 |
| 🧩 Prefrontal | Multi-step planning, goal management, tool use | Phase 2 |
| 🏃 Motor Cortex | API calls, code execution, blockchain actions, web search | Phase 2 |
| ❤️ Limbic | RLHF reward signal, preferences, intrinsic drives | Phase 3 |
| 🧬 Immune | Input safety, output filtering, drift detection, self-repair | Phase 3 |
| 🧩 Default Mode | Self-critique, consistency, identity, internal simulation | Phase 4 |
| 👁️ Visual Cortex | Image encoding, scene understanding, Belize aerial imagery | Phase 5a |
| 👂 Auditory Cortex | Speech transcription, Kriol/Garifuna/Maya speech | Phase 5b |
| 🧠 Cerebellum | AgriTech / Marine / Edu / Tech domain specialists | Already done |
| ⚛️ Quantum | Quantum memory, plan optimizer, anomaly detector, imagination | Phase 6 |

---

*Document maintained by BelizeChain AI Team.*
*Next review: After Phase 0 completion.*
*Version: 1.0 — March 1, 2026*
