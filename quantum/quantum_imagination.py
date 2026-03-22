"""
QuantumImagination — Quantum parallel future rollouts.

Extends (or replaces) ``InternalSimulator`` from the Metacognition
layer with quantum-parallel sampling.  The core insight is that a
quantum superposition over actions naturally generates diverse
trajectories that a sequential classical simulator cannot.

Three-tier routing
------------------
1. **Quantum (Kinich live)**
   Encodes action embeddings into a superposition and samples
   collapsed states, yielding diverse trajectories in O(√N) shots.

2. **Simulated quantum** (simulation_mode=True)
   Applies a stochastic perturbation kernel to diversify classical
   rollouts, approximating the spread of quantum outcomes.

3. **Classical fallback**
   Wraps ``InternalSimulator.simulate()`` (if provided), otherwise
   uses a simple one-step heuristic.

Data classes
------------
``SimulatedState``
    Carries the result of a single imagined trajectory.

Public API
----------
::

    from quantum.quantum_imagination import QuantumImagination, SimulatedState
    from metacognition.internal_simulator import InternalSimulator

    qi = QuantumImagination(
        internal_simulator=InternalSimulator(),
        simulation_mode=True,
    )
    futures = qi.sample_futures(
        current_state={"task": "summarise"},
        possible_actions=[{"model": "gpt"}, {"model": "local"}],
        n_samples=8,
    )
    best = qi.best_action(futures)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# --------------------------------------------------------------------------- #
# Data classes                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class SimulatedState:
    """
    The result of a single imagined future trajectory.

    Attributes:
        action      : The action that was simulated.
        trajectory  : Sequence of (state, action, next_state) dicts.
        value       : Estimated cumulative reward / utility in [0, 1].
        uncertainty : Spread of quantum outcomes (0 = classical certainty).
        n_samples   : Number of quantum samples this state is derived from.
        metadata    : Arbitrary extra data from the rollout.
    """

    action: Dict[str, Any]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    value: float = 0.0
    uncertainty: float = 0.0
    n_samples: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# QuantumImagination                                                           #
# --------------------------------------------------------------------------- #


class QuantumImagination:
    """
    Quantum parallel future imagination.

    Args:
        internal_simulator   : Optional InternalSimulator from
                               ``metacognition.internal_simulator``.
                               Used on the classical fallback path.
        connector            : KinichQuantumConnector.
        fallback_to_classical: Degrade gracefully (default True).
        simulation_mode      : Enable stochastic quantum simulation
                               (default False).
        n_qubits             : Circuit width — determines diversity of
                               sampled futures (default 8).
        n_samples            : Default number of trajectory samples
                               per action (default 8).
        noise_scale          : Perturbation noise magnitude in
                               simulated mode (default 0.05).
        random_state         : RNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        internal_simulator: Optional[Any] = None,
        connector: Optional[Any] = None,
        fallback_to_classical: bool = True,
        simulation_mode: bool = False,
        n_qubits: int = 8,
        n_samples: int = 8,
        noise_scale: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._simulator = internal_simulator
        self._connector = connector
        self.fallback_to_classical = fallback_to_classical
        self.simulation_mode = simulation_mode
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(random_state)

        self.stats: Dict[str, int] = {
            "quantum_calls": 0,
            "simulated_calls": 0,
            "classical_calls": 0,
        }

        logger.info(
            f"QuantumImagination ready: simulation_mode={simulation_mode} "
            f"n_qubits={n_qubits} n_samples={n_samples} "
            f"connector={'yes' if connector else 'none'}"
        )

    # ------------------------------------------------------------------ #
    # Primary public methods                                               #
    # ------------------------------------------------------------------ #

    def sample_futures(
        self,
        current_state: Dict[str, Any],
        possible_actions: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
    ) -> List[SimulatedState]:
        """
        Generate imagined future trajectories for each possible action.

        Args:
            current_state   : Current world / agent state dict.
            possible_actions: List of candidate action dicts.
            n_samples       : Trajectories per action; defaults to
                              ``self.n_samples``.

        Returns:
            List of SimulatedState (one per action × sample).
            The list is ordered by estimated value descending.
        """
        if not possible_actions:
            return []

        n = n_samples if n_samples is not None else self.n_samples
        t0 = time.perf_counter()

        if self._should_use_quantum():
            futures = self._quantum_sample(current_state, possible_actions, n)
            self.stats["quantum_calls"] += len(possible_actions) * n
            mode = "quantum"
        elif self.simulation_mode:
            futures = self._simulated_sample(current_state, possible_actions, n)
            self.stats["simulated_calls"] += len(possible_actions) * n
            mode = "simulated"
        else:
            futures = self._classical_sample(current_state, possible_actions, n)
            self.stats["classical_calls"] += len(possible_actions) * n
            mode = "classical"

        futures.sort(key=lambda s: s.value, reverse=True)
        elapsed = time.perf_counter() - t0
        logger.debug(
            f"QuantumImagination.sample_futures: mode={mode} "
            f"actions={len(possible_actions)} samples={n} "
            f"futures={len(futures)} elapsed={elapsed*1000:.1f}ms"
        )
        return futures

    def best_action(self, futures: List[SimulatedState]) -> Optional[Dict[str, Any]]:
        """
        Return the action with the highest average value across futures.

        Args:
            futures: Output of ``sample_futures()``.

        Returns:
            Best action dict, or None if *futures* is empty.
        """
        if not futures:
            return None

        # Aggregate by action identity
        action_scores: Dict[str, List[float]] = {}
        action_map: Dict[str, Dict[str, Any]] = {}

        for state in futures:
            key = str(state.action)
            action_scores.setdefault(key, []).append(state.value)
            action_map[key] = state.action

        best_key = max(action_scores, key=lambda k: float(np.mean(action_scores[k])))
        return action_map[best_key]

    # ------------------------------------------------------------------ #
    # Sampling implementations                                             #
    # ------------------------------------------------------------------ #

    # ·· Classical ·······················································
    def _classical_sample(
        self,
        state: Dict[str, Any],
        actions: List[Dict[str, Any]],
        n: int,
    ) -> List[SimulatedState]:
        """
        Classical deterministic simulation.
        Wraps InternalSimulator if available, else uses heuristics.
        """
        results: List[SimulatedState] = []
        for action in actions:
            for _ in range(n):
                traj, value = self._simulate_one(state, action, jitter=False)
                results.append(
                    SimulatedState(
                        action=action,
                        trajectory=traj,
                        value=value,
                        uncertainty=0.0,
                        n_samples=1,
                    )
                )
        return results

    # ·· Simulated quantum ···············································
    def _simulated_sample(
        self,
        state: Dict[str, Any],
        actions: List[Dict[str, Any]],
        n: int,
    ) -> List[SimulatedState]:
        """
        Stochastic perturbation simulation that approximates quantum
        superposition diversity.

        Each sample adds Gaussian noise to the action embedding before
        evaluation, producing a spread of outcomes analogous to quantum
        measurement outcomes.
        """
        results: List[SimulatedState] = []
        for action in actions:
            sample_values: List[float] = []
            traj_0, base_value = self._simulate_one(state, action, jitter=False)

            for _ in range(n):
                _, val = self._simulate_one(state, action, jitter=True)
                sample_values.append(val)

            mean_val = float(np.mean(sample_values))
            uncertainty = float(np.std(sample_values))

            results.append(
                SimulatedState(
                    action=action,
                    trajectory=traj_0,
                    value=mean_val,
                    uncertainty=uncertainty,
                    n_samples=n,
                    metadata={"sample_values": sample_values},
                )
            )
        return results

    # ·· Quantum (Kinich) ················································
    def _quantum_sample(
        self,
        state: Dict[str, Any],
        actions: List[Dict[str, Any]],
        n: int,
    ) -> List[SimulatedState]:
        """
        Quantum trajectory sampling via Kinich connector.

        PhaseHook — Phase 5: replace with Kinich superposition circuit.
        Falls back to stochastic simulation.
        """
        logger.debug(
            "QuantumImagination: Kinich live — using stochastic proxy (Phase 5 TBD)"
        )
        return self._simulated_sample(state, actions, n)

    # ------------------------------------------------------------------ #
    # Single-step simulation                                               #
    # ------------------------------------------------------------------ #

    def _simulate_one(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        jitter: bool = False,
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Simulate one trajectory step (classical heuristic).

        If an InternalSimulator is available, delegates to it.

        Returns:
            (trajectory, estimated_value)
        """
        if self._simulator is not None:
            try:
                result = self._simulator.simulate(state=state, action=action)
                traj = getattr(
                    result, "trajectory", [{"state": state, "action": action}]
                )
                value = float(getattr(result, "value", 0.5))
                if jitter:
                    value += float(self._rng.normal(0, self.noise_scale))
                    value = max(0.0, min(1.0, value))
                return traj, value
            except Exception as exc:
                logger.debug(f"QuantumImagination: InternalSimulator failed: {exc}")

        # Built-in heuristic: score by how many "positive" keys the action has
        positive_keys = {"speed", "accuracy", "safety", "quality", "efficiency"}
        raw_score = sum(
            1.0 for k, v in action.items() if k in positive_keys and v
        ) / max(len(positive_keys), 1)

        if jitter:
            raw_score += float(self._rng.normal(0, self.noise_scale))
            raw_score = max(0.0, min(1.0, raw_score))

        traj = [{"state": state, "action": action, "next_state": {**state, **action}}]
        return traj, raw_score

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def _should_use_quantum(self) -> bool:
        if self._connector is None:
            return False
        return bool(getattr(self._connector, "kinich_available", False))

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.stats.values())
        return {
            **self.stats,
            "total_calls": total,
            "quantum_ratio": (
                self.stats["quantum_calls"] / total if total > 0 else 0.0
            ),
        }
