"""
Control Module — Nawal Brain Architecture (Prefrontal Cortex / Executive Function)

Sub-systems:
    planner       — generates candidate action plans from goal + context
    executor      — dispatches chosen plan to action modules
    goal_stack    — priority-ordered objective tracker
    orchestrator  — decides which brain service to call next

Phase 0: skeleton + interfaces only.
Phase 2: classical planner + goal-stack implementation.
Phase 4: Quantum Basal Ganglia plan-selector (QAOA-based).

Canonical import:
    from nawal.control.interfaces import AbstractPlanner, AbstractExecutor
"""

from control.interfaces import AbstractPlanner, AbstractExecutor, Goal, GoalStatus, Plan
from control.goal_stack import GoalStack
from control.planner import ClassicalPlanner
from control.executor import ToolExecutor
from control.controller import ExecutiveController

__all__ = [
    # Interfaces
    "AbstractPlanner",
    "AbstractExecutor",
    "Goal",
    "GoalStatus",
    "Plan",
    # Implementations (Phase 2)
    "GoalStack",
    "ClassicalPlanner",
    "ToolExecutor",
    "ExecutiveController",
]
