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

from control.controller import ExecutiveController
from control.executor import ToolExecutor
from control.goal_stack import GoalStack
from control.interfaces import AbstractExecutor, AbstractPlanner, Goal, GoalStatus, Plan
from control.planner import ClassicalPlanner

__all__ = [
    "AbstractExecutor",
    # Interfaces
    "AbstractPlanner",
    "ClassicalPlanner",
    "ExecutiveController",
    "Goal",
    # Implementations (Phase 2)
    "GoalStack",
    "GoalStatus",
    "Plan",
    "ToolExecutor",
]
