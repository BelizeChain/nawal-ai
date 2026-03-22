"""
Executive Controller — the prefrontal cortex of the nawal brain.

Orchestrates GoalStack, ClassicalPlanner, and ToolExecutor into a single
``tick()``-driven control loop.  Each tick:

  1. Peeks the highest-priority live goal.
  2. Tries to activate it (demoting any lower-priority active goal first).
  3. Generates candidate plans (ClassicalPlanner).
  4. Selects the best plan (hook for Phase 4 QAOA selector).
  5. Executes the plan (ToolExecutor).
  6. Updates goal status based on execution result.
  7. Stores the result summary in memory.

Usage::

    mm         = MemoryManager(...)
    controller = ExecutiveController(memory=mm)
    controller.register_tool("respond", my_respond_fn)
    controller.register_tool("search",  my_search_fn)

    controller.add_goal("Summarise recent news", priority=1.0)
    result = controller.tick(world_state={"context": "..."})
    print(result)

Thread safety: tick() acquires the GoalStack's internal lock transitively;
  it is NOT re-entrant — do not call tick() concurrently on the same instance.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from loguru import logger

from control.executor import ToolExecutor
from control.goal_stack import GoalStack
from control.interfaces import Goal, GoalStatus, Plan
from control.planner import ClassicalPlanner

# --------------------------------------------------------------------------- #
# Default available tool set                                                   #
# --------------------------------------------------------------------------- #

_DEFAULT_TOOLS = [
    "noop",
    "log",
    "memory_read",
    "memory_write",
    "respond",
    "reason",
    "validate",
    "search",
    "execute",
]


# --------------------------------------------------------------------------- #
# ExecutiveController                                                          #
# --------------------------------------------------------------------------- #


class ExecutiveController:
    """
    High-level executive controller.

    Args:
        memory             : MemoryManager instance (optional; enables built-in
                             memory tool wiring and result storage).
        available_tools    : Tool names the planner is allowed to schedule.
        planner_confidence : Base confidence passed to ClassicalPlanner.
        executor_timeout   : Per-step timeout for ToolExecutor.
        max_plans          : Maximum candidate plans generated per tick.
    """

    def __init__(
        self,
        memory: Any | None = None,
        available_tools: list[str] | None = None,
        planner_confidence: float = 0.7,
        executor_timeout: float = 30.0,
        max_plans: int = 3,
    ) -> None:
        self._memory = memory
        self._max_plans = max_plans
        self._lock = threading.Lock()

        _tools = available_tools if available_tools is not None else _DEFAULT_TOOLS

        self._goal_stack = GoalStack()
        self._planner = ClassicalPlanner(
            available_tools=_tools,
            base_confidence=planner_confidence,
        )
        self._executor = ToolExecutor(
            memory=memory,
            timeout=executor_timeout,
        )

        # Execution history
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_goal(
        self,
        description: str,
        priority: float = 1.0,
        context: dict[str, Any] | None = None,
        sub_goals: list[Goal] | None = None,
    ) -> Goal:
        """
        Push a new goal onto the stack.

        Returns the created Goal.
        """
        goal = Goal(
            goal_id=str(uuid.uuid4()),
            description=description,
            priority=priority,
            context=context or {},
            status=GoalStatus.PENDING,
            sub_goals=sub_goals or [],
        )
        self._goal_stack.push(goal)
        logger.info(
            f"ExecutiveController goal enqueued: {description!r} "
            f"(priority={priority}, id={goal.goal_id})"
        )
        return goal

    def register_tool(self, name: str, fn: Callable, overwrite: bool = True) -> None:
        """Register a callable tool with the executor."""
        self._executor.register(name, fn, overwrite=overwrite)

    def interrupt_current(self) -> bool:
        """
        Interrupt the currently executing plan (if any).

        Returns True if a running plan was found.
        """
        goal = self._goal_stack.peek()
        if goal is None:
            return False
        # Find the plan_id associated with the current goal in history
        for entry in reversed(self._history):
            if entry.get("goal_id") == goal.goal_id and entry.get("plan_id"):
                return self._executor.interrupt(entry["plan_id"])
        return False

    def tick(
        self,
        world_state: dict[str, Any] | None = None,
        plan_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Single control-loop iteration.

        Returns a summary dict, or None if no goal is ready.

        Summary format::

            {
                "goal_id":     str,
                "description": str,
                "plan_id":     str,
                "status":      "success" | "partial" | "failed",
                "outputs":     [...],
                "elapsed":     float,   # seconds
            }
        """
        ws = world_state or {}
        constraints = plan_constraints or {}

        # ---- Select next goal ---------------------------------------- #
        goal = self._goal_stack.next_pending()
        if goal is None:
            goal = self._goal_stack.peek()  # might be already-active
        if goal is None:
            logger.debug("ExecutiveController tick — no pending goals")
            return None

        # ---- Activate goal ------------------------------------------- #
        if goal.status != GoalStatus.ACTIVE:
            activated = self._goal_stack.activate(goal.goal_id)
            if not activated:
                logger.warning(f"Could not activate goal {goal.goal_id!r}; skipping tick.")
                return None

        logger.info(f"ExecutiveController tick: goal={goal.description!r} " f"(id={goal.goal_id})")
        t0 = time.monotonic()

        # ---- Generate candidate plans -------------------------------- #
        try:
            plans = self._planner.generate_plans(
                goal=goal,
                world_state=ws,
                n_candidates=self._max_plans,
            )
        except Exception as exc:
            logger.error(f"Planner failure: {exc}")
            self._goal_stack.fail(goal.goal_id)
            return self._build_summary(goal, None, "failed", [], 0.0, str(exc))

        if not plans:
            logger.warning("Planner returned no plans; failing goal.")
            self._goal_stack.fail(goal.goal_id)
            return self._build_summary(goal, None, "failed", [], 0.0, "no plans generated")

        # ---- Select best plan ---------------------------------------- #
        plan = self._planner.select_plan(plans, constraints=constraints)

        logger.info(
            f"Selected plan_id={plan.plan_id!r} " f"score={plan.score:.3f} steps={len(plan.steps)}"
        )

        # ---- Execute plan -------------------------------------------- #
        try:
            exec_result = self._executor.execute(plan)
        except Exception as exc:
            logger.error(f"Executor raised unexpectedly: {exc}")
            self._goal_stack.fail(goal.goal_id)
            elapsed = time.monotonic() - t0
            return self._build_summary(goal, plan, "failed", [], elapsed, str(exc))

        elapsed = time.monotonic() - t0
        exec_status = exec_result["status"]

        # ---- Update goal status -------------------------------------- #
        if exec_status == "success":
            self._goal_stack.complete(goal.goal_id)
        elif exec_status == "failed":
            self._goal_stack.fail(goal.goal_id)
        # "partial" → goal stays active; caller may re-tick

        # ---- Persist result to memory -------------------------------- #
        self._store_result_in_memory(goal, plan, exec_result, elapsed)

        summary = self._build_summary(goal, plan, exec_status, exec_result["outputs"], elapsed)
        self._history.append(summary)
        return summary

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the controller's state."""
        live = self._goal_stack.all_live()
        hist = self._goal_stack.history()
        return {
            "pending_goals": sum(1 for g in live if g.status == GoalStatus.PENDING),
            "active_goals": sum(1 for g in live if g.status == GoalStatus.ACTIVE),
            "total_live": len(live),
            "completed_goals": sum(1 for g in hist if g.status == GoalStatus.COMPLETED),
            "failed_goals": sum(1 for g in hist if g.status == GoalStatus.FAILED),
            "ticks_run": len(self._history),
            "registered_tools": self._executor.available_tools(),
        }

    @property
    def goal_stack(self) -> GoalStack:
        return self._goal_stack

    @property
    def planner(self) -> ClassicalPlanner:
        return self._planner

    @property
    def executor(self) -> ToolExecutor:
        return self._executor

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        goal: Goal,
        plan: Plan | None,
        status: str,
        outputs: list,
        elapsed: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "goal_id": goal.goal_id,
            "description": goal.description,
            "plan_id": plan.plan_id if plan else None,
            "plan_score": plan.score if plan else 0.0,
            "status": status,
            "outputs": outputs,
            "elapsed": round(elapsed, 4),
            "error": error,
        }

    def _store_result_in_memory(
        self,
        goal: Goal,
        plan: Plan,
        exec_result: dict[str, Any],
        elapsed: float,
    ) -> None:
        if self._memory is None:
            return
        try:
            content = (
                f"Goal: {goal.description}\n"
                f"Plan score: {plan.score:.3f}\n"
                f"Status: {exec_result['status']}\n"
                f"Steps run: {len(exec_result['outputs'])}\n"
                f"Elapsed: {elapsed:.3f}s"
            )
            self._memory.store_text(
                content,
                metadata={
                    "source": "executive_controller",
                    "goal_id": goal.goal_id,
                    "plan_id": plan.plan_id,
                    "status": exec_result["status"],
                },
                store="episodic",
            )
        except Exception as exc:
            logger.warning(f"Failed to store execution result in memory: {exc}")
