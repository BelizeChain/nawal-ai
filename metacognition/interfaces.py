"""
Metacognition interfaces — Abstract Base Classes for the Default Mode Network.

The Metacognitive layer "thinks about thinking":
  - Critic       : self-reviews a candidate output before it is emitted
  - Simulator    : imagines future rollouts of possible actions
  - Confidence   : estimates uncertainty per token / plan / claim

Phase 3: classical critic + confidence estimation.
Phase 5: Quantum Imagination Engine overrides AbstractSimulator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfidenceScore:
    """
    Uncertainty estimate attached to a model output.

    Attributes:
        value       : Scalar confidence in [0.0, 1.0].
        method      : How it was computed ("logprob", "ensemble", "verbal", …).
        explanation : Optional natural-language justification.
        metadata    : Extra calibration details.
    """
    value: float
    method: str = "unknown"
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """
    Output of an AbstractCritic review pass.

    Attributes:
        approved         : True if the response should be emitted as-is.
        issues           : List of identified problems (empty if approved).
        revised_response : Critic's rewrite (if approved=False and rewrite possible).
        confidence       : Confidence in the critique itself.
    """
    approved: bool
    issues: List[str] = field(default_factory=list)
    revised_response: Optional[str] = None
    confidence: Optional[ConfidenceScore] = None


class AbstractCritic(ABC):
    """
    Reviews the model's own candidate output before emission.

    Classical implementation (Phase 3): chain-of-thought self-critique
    loop using the same base transformer.
    """

    @abstractmethod
    def critique(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> CritiqueResult:
        """
        Evaluate *response* and either approve or flag issues.

        Args:
            response : Candidate model output (text).
            context  : Active goal, memory snapshot, safety constraints.

        Returns:
            CritiqueResult with approval flag and optional revision.
        """

    @abstractmethod
    def estimate_confidence(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> ConfidenceScore:
        """
        Return a confidence estimate for *response*.

        May use log-probability introspection or verbal uncertainty probing.
        """


class AbstractSimulator(ABC):
    """
    Imagines future rollouts before the agent acts.

    Classical implementation (Phase 3): autoregressive "what-if" sampling.
    Quantum override (Phase 5): quantum rollout sampler explores many
    futures in superposition before collapsing to the highest-value branch.
    """

    @abstractmethod
    def simulate(
        self,
        current_state: Dict[str, Any],
        possible_actions: List[Dict[str, Any]],
        horizon: int = 3,
        n_samples: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generate imagined future scenarios.

        Args:
            current_state    : World-state snapshot + memory context.
            possible_actions : Candidate next actions to branch from.
            horizon          : How many steps to simulate forward.
            n_samples        : Number of rollout samples per action branch.

        Returns:
            List of scenario dicts, each containing:
              - "action"       : The triggering action.
              - "trajectory"   : Sequence of predicted world states.
              - "value"        : Estimated cumulative reward.
        """

    @abstractmethod
    def best_action(
        self,
        simulations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select the highest-value action from completed simulations.

        Args:
            simulations : Output of :meth:`simulate`.

        Returns:
            The action descriptor with the highest estimated value.
        """
