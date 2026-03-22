"""
Control interfaces — Abstract Base Classes for the Executive Controller.

The executive controller (Prefrontal Cortex) is responsible for:
  - Maintaining a prioritized goal stack
  - Generating candidate action plans from the current world state
  - Selecting the best plan (classically or via quantum optimizer)
  - Dispatching the chosen plan to the Action layer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Goal:
    """
    A single objective in the agent's goal stack.

    Attributes:
        goal_id     : Unique identifier.
        description : Human-readable goal statement.
        priority    : 0.0–1.0 (higher = more urgent).
        context     : Arbitrary context dict (tools available, constraints, …).
        status      : Current lifecycle state.
        sub_goals   : Optional decomposition into child goals.
    """

    goal_id: str
    description: str
    priority: float = 0.5
    context: dict[str, Any] = field(default_factory=dict)
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: list[Goal] = field(default_factory=list)


@dataclass
class Plan:
    """
    An ordered sequence of action steps for achieving a Goal.

    Attributes:
        plan_id    : Unique identifier.
        goal_id    : The goal this plan addresses.
        steps      : Ordered list of action descriptors (dicts with 'tool', 'args').
        score      : Estimated value / feasibility (0.0–1.0).
        metadata   : Planner-specific annotations.
    """

    plan_id: str
    goal_id: str
    steps: list[dict[str, Any]]
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractPlanner(ABC):
    """
    Generates one or more candidate Plans for a given Goal.

    The classical implementation (Phase 2) uses LLM-based chain-of-thought
    planning.  The Phase 4 "Quantum Basal Ganglia" wraps this ABC and
    overrides ``select_plan`` with a QAOA-based combinatorial optimizer.
    """

    @abstractmethod
    def generate_plans(
        self,
        goal: Goal,
        world_state: dict[str, Any],
        n_candidates: int = 3,
    ) -> list[Plan]:
        """
        Produce up to *n_candidates* Plans for *goal* given *world_state*.

        Args:
            goal         : The active goal to plan for.
            world_state  : Current perception snapshot + memory context.
            n_candidates : How many alternative plans to generate.

        Returns:
            List of Plan objects, unordered (caller picks best).
        """

    @abstractmethod
    def select_plan(
        self,
        plans: list[Plan],
        constraints: dict[str, Any] | None = None,
    ) -> Plan:
        """
        Choose the best Plan from a ranked list.

        Classical default: pick the highest ``score``.
        Quantum override (Phase 4): run QAOA / quantum annealing.

        Args:
            plans       : Candidate plans produced by generate_plans.
            constraints : Hard requirements (budget, latency, safety tags).

        Returns:
            The selected Plan.
        """


class AbstractExecutor(ABC):
    """
    Dispatches a chosen Plan to the appropriate Action sub-systems.
    """

    @abstractmethod
    def execute(
        self,
        plan: Plan,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Execute *plan* step-by-step.

        Args:
            plan    : The Plan returned by AbstractPlanner.select_plan.
            dry_run : If True, validate steps without actually running them.

        Returns:
            Execution result dict with keys:
              - "status"  : "success" | "partial" | "failed"
              - "outputs" : per-step output list
              - "error"   : error message if failed (else None)
        """

    @abstractmethod
    def interrupt(self, plan_id: str) -> bool:
        """
        Gracefully interrupt a running plan.

        Returns True if the plan was found and stopped.
        """
