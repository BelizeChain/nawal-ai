"""
Goal Stack — priority-ordered objective tracker for the Executive Controller.

The GoalStack is the agent's "to-do list": a thread-safe, priority-sorted
collection of Goals that drives all planning and execution decisions.

Behaviours mirror the prefrontal-cortex goal-maintenance role:
  - Goals are sorted by descending priority at all times.
  - Only one Goal is ACTIVE at a time (the highest-priority PENDING goal).
  - Completed / Failed goals are retained in history for memory consolidation.
  - Sub-goals can be pushed independently; they reference their parent via
    ``goal.context["parent_id"]``.

Usage::

    gs = GoalStack()
    g  = gs.push("Summarise the quarterly report", priority=0.9)
    gs.activate(g.goal_id)
    active = gs.active()          # returns the Goal
    gs.complete(g.goal_id)
    print(gs.history())           # [completed Goal]
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, List, Optional

from loguru import logger

from control.interfaces import Goal, GoalStatus


class GoalStack:
    """
    Thread-safe, priority-ordered goal manager.

    Goals are kept in a sorted list (highest priority first).
    completed / failed goals are moved to a separate history list.

    Args:
        max_active_goals : Soft cap on concurrent PENDING/ACTIVE goals.
                           A warning is emitted when exceeded (not enforced).
    """

    def __init__(self, max_active_goals: int = 16) -> None:
        self.max_active_goals = max_active_goals
        self._goals: Dict[str, Goal] = {}  # live goals
        self._history: List[Goal] = []  # completed / failed
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Write API                                                            #
    # ------------------------------------------------------------------ #

    def push(
        self,
        goal_or_description,
        priority: float = 1.0,
        context: Optional[Dict] = None,
        goal_id: Optional[str] = None,
        sub_goals: Optional[List[Goal]] = None,
    ) -> Goal:
        """
        Queue a goal.

        Accepts either:
        - A pre-built ``Goal`` instance — stored as-is (all other args ignored).
        - A ``str`` description — a new Goal is created with the given kwargs.

        Args:
            goal_or_description : Goal object **or** human-readable objective string.
            priority    : Urgency weight (higher = more urgent). Only used when
                          creating a Goal from a string description.
            context     : Arbitrary metadata. Only used for string-based creation.
            goal_id     : Explicit ID (auto UUID if omitted). String creation only.
            sub_goals   : Child goals. String creation only.

        Returns:
            The stored Goal.
        """
        if isinstance(goal_or_description, Goal):
            goal = goal_or_description
        else:
            goal = Goal(
                goal_id=goal_id or str(uuid.uuid4()),
                description=goal_or_description,
                priority=priority,
                context=context or {},
                status=GoalStatus.PENDING,
                sub_goals=sub_goals or [],
            )
        with self._lock:
            live = len(self._goals)
            if live >= self.max_active_goals:
                logger.warning(
                    f"GoalStack has {live} live goals (max_active={self.max_active_goals})"
                )
            self._goals[goal.goal_id] = goal
        logger.debug(
            f"GoalStack push goal_id={goal.goal_id!r} "
            f"priority={goal.priority:.2f} desc={goal.description!r}"
        )
        return goal

    def activate(self, goal_id: str) -> bool:
        """
        Mark a goal ACTIVE (only one should be active at a time).

        Demotes any currently ACTIVE goal back to PENDING first.

        Returns:
            True on success, False if the goal_id is not found or the goal
            is in a terminal state that cannot be re-activated.
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                logger.debug(f"GoalStack activate: goal_id={goal_id!r} not found")
                return False
            if goal.status not in (GoalStatus.PENDING, GoalStatus.BLOCKED):
                logger.debug(
                    f"GoalStack activate: goal {goal_id!r} in status "
                    f"{goal.status.value!r} — cannot activate"
                )
                return False
            # Demote current active
            for g in self._goals.values():
                if g.status == GoalStatus.ACTIVE:
                    g.status = GoalStatus.PENDING
            goal.status = GoalStatus.ACTIVE
        logger.info(
            f"GoalStack activated goal_id={goal_id!r} desc={goal.description!r}"
        )
        return True

    def complete(self, goal_id: str) -> Goal:
        """Mark a goal COMPLETED and move it to history."""
        return self._finalise(goal_id, GoalStatus.COMPLETED)

    def fail(self, goal_id: str, reason: str = "") -> Goal:
        """Mark a goal FAILED and move it to history."""
        goal = self._finalise(goal_id, GoalStatus.FAILED)
        if reason:
            goal.context["failure_reason"] = reason
        return goal

    def block(self, goal_id: str, reason: str = "") -> Goal:
        """Mark a goal BLOCKED (waiting on external resource)."""
        with self._lock:
            goal = self._require(goal_id)
            goal.status = GoalStatus.BLOCKED
            if reason:
                goal.context["block_reason"] = reason
        logger.debug(f"GoalStack blocked goal_id={goal_id!r}")
        return goal

    def update_priority(self, goal_id: str, priority: float) -> Goal:
        """Adjust the priority of an existing goal (no range clamping)."""
        with self._lock:
            goal = self._require(goal_id)
            goal.priority = priority
        logger.debug(
            f"GoalStack updated priority goal_id={goal_id!r} "
            f"new_priority={priority:.2f}"
        )
        return goal

    # ------------------------------------------------------------------ #
    # Read API                                                             #
    # ------------------------------------------------------------------ #

    def active(self) -> Optional[Goal]:
        """Return the currently ACTIVE goal, or None."""
        with self._lock:
            for g in self._sorted():
                if g.status == GoalStatus.ACTIVE:
                    return g
        return None

    def next_pending(self) -> Optional[Goal]:
        """Return the highest-priority PENDING goal without activating it."""
        with self._lock:
            for g in self._sorted():
                if g.status == GoalStatus.PENDING:
                    return g
        return None

    def peek(self) -> Optional[Goal]:
        """
        Return the highest-priority actionable goal (ACTIVE first,
        then PENDING) without modifying state.
        """
        with self._lock:
            for g in self._sorted():
                if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING):
                    return g
        return None

    def get(self, goal_id: str) -> Optional[Goal]:
        """Return a live goal by ID, or None if not found."""
        with self._lock:
            return self._goals.get(goal_id)

    def all_live(self) -> List[Goal]:
        """Return all non-terminated goals, sorted by descending priority."""
        with self._lock:
            return self._sorted()

    def history(self, last_n: int = 50) -> List[Goal]:
        """Return the most-recent completed/failed goals (newest first)."""
        with self._lock:
            return list(reversed(self._history[-last_n:]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._goals)

    def __repr__(self) -> str:
        with self._lock:
            live = len(self._goals)
            hist = len(self._history)
        return f"GoalStack(live={live}, history={hist})"

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _sorted(self) -> List[Goal]:
        """Return live goals sorted by descending priority (caller holds lock)."""
        return sorted(self._goals.values(), key=lambda g: g.priority, reverse=True)

    def _require(self, goal_id: str) -> Goal:
        """Return goal or raise KeyError (caller holds lock)."""
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal not found: {goal_id!r}")
        return goal

    def _finalise(self, goal_id: str, status: GoalStatus) -> Goal:
        """Move goal to terminal state and archive it."""
        with self._lock:
            goal = self._require(goal_id)
            goal.status = status
            del self._goals[goal_id]
            self._history.append(goal)
        logger.info(
            f"GoalStack {status.value} goal_id={goal_id!r} "
            f"desc={goal.description!r}"
        )
        return goal
