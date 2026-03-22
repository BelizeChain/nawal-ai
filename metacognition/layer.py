"""
MetacognitionLayer — unified facade for the nawal self-reflection system.

Coordinates:
  - SelfCritic           : reviews candidate responses for quality issues
  - ConsistencyChecker   : detects contradictions across multiple candidates
  - ConfidenceCalibrator : aggregates signals into a calibrated confidence
  - InternalSimulator    : imagines action outcomes before execution
  - IdentityModule       : maintains agent persona and decision history

The primary entry-point is ``reflect()``, called after the core LLM
generates candidates and before the action layer emits a response.

Typical flow::

    layer = MetacognitionLayer(persist_path="./data/identity.json")
    layer.load()

    result = layer.reflect(
        candidates=["Paris is the capital of France.", "Lyon is the capital."],
        context={"goal": "What is the capital of France?"},
    )
    print(result.best_candidate)   # "Paris is the capital of France."
    print(result.confidence.value) # 0.83
    print(result.approved)         # True

    layer.identity.record_decision(
        goal=context["goal"], outcome="success", confidence=result.confidence.value
    )

PhaseHook:
    Phase 5 replaces ``InternalSimulator.simulate()`` with
    ``QuantumImagination.simulate()`` — the MetacognitionLayer interface is
    unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from metacognition.confidence_calibrator import ConfidenceCalibrator
from metacognition.consistency_checker import ConsistencyChecker, ConsistencyResult
from metacognition.identity_module import IdentityModule
from metacognition.interfaces import ConfidenceScore, CritiqueResult
from metacognition.internal_simulator import InternalSimulator
from metacognition.self_critic import SelfCritic

# --------------------------------------------------------------------------- #
# Reflection result                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class ReflectionResult:
    """
    Full metacognitive reflection output.

    Attributes:
        best_candidate    : The highest-quality approved response, or the
                            least-bad candidate if none approved.
        approved          : True iff best_candidate passed all critic checks.
        confidence        : Calibrated confidence score.
        critique          : CritiqueResult for best_candidate.
        consistency       : ConsistencyResult across all candidates.
        issues            : Aggregated list of issue strings.
        self_description  : Brief identity string (injected into responses).
    """

    best_candidate: str
    approved: bool
    confidence: ConfidenceScore
    critique: CritiqueResult | None = None
    consistency: ConsistencyResult | None = None
    issues: list[str] = field(default_factory=list)
    self_description: str = ""


# --------------------------------------------------------------------------- #
# MetacognitionLayer                                                           #
# --------------------------------------------------------------------------- #


class MetacognitionLayer:
    """
    Orchestrates all self-reflection sub-systems.

    Args:
        critic              : SelfCritic instance (default: SelfCritic()).
        consistency_checker : ConsistencyChecker (default: ConsistencyChecker()).
        calibrator          : ConfidenceCalibrator (default: ConfidenceCalibrator()).
        simulator           : InternalSimulator (default: InternalSimulator()).
        identity            : IdentityModule (default: IdentityModule()).
        valuation_layer     : Optional valuation.ValuationLayer for scoring states
                              inside the simulator.
        persist_path        : Path for IdentityModule JSON persistence.
    """

    def __init__(
        self,
        critic: SelfCritic | None = None,
        consistency_checker: ConsistencyChecker | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        simulator: InternalSimulator | None = None,
        identity: IdentityModule | None = None,
        valuation_layer: Any | None = None,
        persist_path: str | None = None,
    ) -> None:
        self._critic = critic or SelfCritic()
        self._checker = consistency_checker or ConsistencyChecker()
        self._calib = calibrator or ConfidenceCalibrator()
        self._sim = simulator or InternalSimulator(valuation_layer=valuation_layer)
        self._identity = identity or IdentityModule(persist_path=persist_path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reflect(
        self,
        candidates: list[str],
        context: dict[str, Any] | None = None,
        plan_score: float | None = None,
        memory_relevance: float | None = None,
        safety_score: float | None = None,
    ) -> ReflectionResult:
        """
        Run the full metacognitive reflection cycle.

        Steps:
          1. Consistency check across all candidates.
          2. Critique each candidate; collect approved ones preferentially.
          3. Calibrate confidence using all available signals.
          4. Return the best candidate with full reflection result.

        Args:
            candidates       : List of candidate response strings (≥1).
            context          : Active goal, memory context, etc.
            plan_score       : Score from the classical planner (0–1).
            memory_relevance : Best memory retrieval similarity (0–1).
            safety_score     : Safety score from ValuationLayer (0–1).

        Returns:
            ReflectionResult with best_candidate and diagnostics.
        """
        ctx = context or {}
        if not candidates:
            return ReflectionResult(
                best_candidate="",
                approved=False,
                confidence=ConfidenceScore(value=0.0, method="no_candidates"),
                issues=["No candidates provided"],
            )

        logger.info(
            f"MetacognitionLayer.reflect: {len(candidates)} candidate(s), "
            f"goal={str(ctx.get('goal', ''))[:60]!r}"
        )

        try:
            # Step 1: Consistency
            consistency = self._checker.check(candidates, ctx)

            # Step 2: Critique each candidate
            critiques: list[CritiqueResult] = [self._critic.critique(c, ctx) for c in candidates]

            # Step 3: Select best
            # Prefer: approved + most-consistent + highest critic confidence
            best_idx, best_critique = self._select_best(candidates, critiques, consistency)
            best_candidate = candidates[best_idx]
            approved = critiques[best_idx].approved if critiques else False

            # Step 4: Calibration
            verbal_signal = best_candidate if best_candidate else ""
            signals: dict[str, Any] = {
                "verbal": verbal_signal,
                "consistency": consistency.score,
            }
            if best_critique is not None:
                crit_conf = best_critique.confidence
                if crit_conf is not None:
                    signals["critic_score"] = crit_conf.value
            if plan_score is not None:
                signals["plan_score"] = plan_score
            if memory_relevance is not None:
                signals["memory"] = memory_relevance
            if safety_score is not None:
                signals["safety"] = safety_score

            confidence = self._calib.calibrate(signals, ctx)

            # Step 5: Aggregate issues
            all_issues = [iss for cr in critiques for iss in cr.issues]

            result = ReflectionResult(
                best_candidate=best_candidate,
                approved=approved,
                confidence=confidence,
                critique=best_critique,
                consistency=consistency,
                issues=all_issues,
                self_description=self._identity.self_description(brief=True),
            )

            logger.info(
                f"MetacognitionLayer result: approved={approved} "
                f"confidence={confidence.value:.3f} issues={len(all_issues)}"
            )
            return result

        except Exception as exc:
            # Metacognition must never block inference — return the first
            # candidate with a degraded-confidence marker.
            logger.error(
                f"MetacognitionLayer.reflect failed ({type(exc).__name__}: "
                f"{exc}) — passing first candidate through"
            )
            return ReflectionResult(
                best_candidate=candidates[0],
                approved=False,
                confidence=ConfidenceScore(value=0.0, method="metacognition_error"),
                issues=[f"Metacognition error: {exc}"],
            )

    def simulate_actions(
        self,
        current_state: dict[str, Any],
        possible_actions: list[dict[str, Any]],
        horizon: int = 3,
        n_samples: int = 4,
    ) -> dict[str, Any]:
        """
        Wrapper around InternalSimulator.simulate() + best_action().

        Returns the best action dict based on imagined rollouts.
        """
        scenarios = self._sim.simulate(
            current_state, possible_actions, horizon=horizon, n_samples=n_samples
        )
        return self._sim.best_action(scenarios)

    def load(self) -> bool:
        """Load persisted identity state."""
        return self._identity.load()

    def save(self) -> None:
        """Persist identity state to disk."""
        self._identity.save()

    # ------------------------------------------------------------------ #
    # Sub-system accessors                                                 #
    # ------------------------------------------------------------------ #

    @property
    def identity(self) -> IdentityModule:
        return self._identity

    @property
    def critic(self) -> SelfCritic:
        return self._critic

    @property
    def checker(self) -> ConsistencyChecker:
        return self._checker

    @property
    def calibrator(self) -> ConfidenceCalibrator:
        return self._calib

    @property
    def simulator(self) -> InternalSimulator:
        return self._sim

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _select_best(
        self,
        candidates: list[str],
        critiques: list[CritiqueResult],
        consistency: ConsistencyResult,
    ) -> tuple[int, CritiqueResult | None]:
        """
        Pick the best (index, critique) from candidates given critiques
        and consistency result.

        Priority:
          1. Approved candidates only (if any exist).
          2. Most-consistent candidate (lowest contradiction count).
          3. Highest critic confidence.
        """
        n = len(candidates)
        approved_indices = [i for i, cr in enumerate(critiques) if cr.approved]

        pool = approved_indices if approved_indices else list(range(n))

        # Among the pool, prefer most-supported (fewest contradictions)
        most_supported = consistency.most_supported
        if most_supported is not None and most_supported in pool:
            idx = most_supported
        else:
            # Fall back to highest critic confidence within pool
            idx = max(
                pool,
                key=lambda i: (
                    critiques[i].confidence.value if critiques[i].confidence is not None else 0.5
                ),
            )

        critique = critiques[idx] if idx < len(critiques) else None
        return idx, critique
