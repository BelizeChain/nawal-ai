"""
Classical Planner — template-based plan generation for the Executive Controller.

Implements AbstractPlanner using a goal-type-aware step template library.
No LLM required at Phase 2 — the templates handle common agent task patterns.
Phase 3 will replace template generation with LLM chain-of-thought planning while
keeping the same AbstractPlanner interface.

Plan selection uses weighted scoring across three dimensions:
  - feasibility  : does the plan fit within known tool constraints?
  - efficiency   : estimated number of steps / tool calls
  - confidence   : how well-matched the goal is to the template

Phase 4 note: ``select_plan`` is the hook for the Quantum Basal Ganglia (QAOA)
optimizer — it overrides only this one method, keeping generation classical.

Usage::

    planner = ClassicalPlanner(available_tools=["search", "memory_read", "respond"])
    plans   = planner.generate_plans(goal, world_state, n_candidates=3)
    best    = planner.select_plan(plans)
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from control.interfaces import AbstractPlanner, Goal, GoalStatus, Plan


# --------------------------------------------------------------------------- #
# Step templates                                                               #
# --------------------------------------------------------------------------- #

# Each template is a list of step dicts.  Variables like {goal} are filled
# at plan-generation time via str.format_map on each step's "args" values.
_TEMPLATES: Dict[str, List[List[Dict[str, Any]]]] = {
    "retrieve": [
        [
            {"tool": "memory_read",  "args": {"query": "{goal}", "top_k": 5}},
            {"tool": "respond",      "args": {"source": "memory"}},
        ],
        [
            {"tool": "search",       "args": {"query": "{goal}"}},
            {"tool": "memory_write", "args": {"content": "{goal}"}},
            {"tool": "respond",      "args": {"source": "search"}},
        ],
    ],
    "analyse": [
        [
            {"tool": "memory_read",  "args": {"query": "{goal}", "top_k": 8}},
            {"tool": "reason",       "args": {"task": "{goal}", "mode": "chain_of_thought"}},
            {"tool": "respond",      "args": {"source": "reasoning"}},
        ],
    ],
    "generate": [
        [
            {"tool": "reason",    "args": {"task": "{goal}", "mode": "generative"}},
            {"tool": "validate",  "args": {"content": "{goal}"}},
            {"tool": "respond",   "args": {"source": "generation"}},
        ],
        [
            {"tool": "memory_read", "args": {"query": "{goal}", "top_k": 4}},
            {"tool": "reason",      "args": {"task": "{goal}", "mode": "generative"}},
            {"tool": "respond",     "args": {"source": "rag_generation"}},
        ],
    ],
    "act": [
        [
            {"tool": "validate",  "args": {"action": "{goal}"}},
            {"tool": "execute",   "args": {"command": "{goal}"}},
            {"tool": "memory_write", "args": {"content": "executed: {goal}"}},
        ],
    ],
    "default": [
        [
            {"tool": "memory_read", "args": {"query": "{goal}", "top_k": 5}},
            {"tool": "reason",      "args": {"task": "{goal}", "mode": "chain_of_thought"}},
            {"tool": "respond",     "args": {"source": "default"}},
        ],
    ],
}

# Keywords that trigger each template type
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "retrieve": ["retrieve", "find", "get", "fetch", "look up", "search", "what is", "who is", "where is", "tell me", "recall", "remember"],
    "analyse":  ["analyse", "analyze", "explain", "summarise", "summarize", "compare", "evaluate", "why"],
    "generate": ["write", "create", "generate", "draft", "compose", "make", "build", "design"],
    "act":      ["execute", "run", "call", "send", "post", "deploy", "update", "delete"],
}


def _detect_intent(description: str) -> str:
    """Map a goal description to a template intent key."""
    lower = description.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "default"


def _fill_template(template: List[Dict[str, Any]], goal_desc: str) -> List[Dict[str, Any]]:
    """Substitute {goal} placeholders in a template copy."""
    filled = []
    for raw_step in template:
        step = {"tool": raw_step["tool"], "args": {}}
        for k, v in raw_step.get("args", {}).items():
            step["args"][k] = v.format_map({"goal": goal_desc}) if isinstance(v, str) else v
        filled.append(step)
    return filled


# --------------------------------------------------------------------------- #
# ClassicalPlanner                                                             #
# --------------------------------------------------------------------------- #

class ClassicalPlanner(AbstractPlanner):
    """
    Template-driven classical planner.

    Generates candidate Plans by:
      1. Detecting goal intent (retrieve / analyse / generate / act / default).
      2. Selecting the matching template variants.
      3. Filtering out templates that require unavailable tools.
      4. Scoring each plan on feasibility + efficiency.

    Args:
        available_tools : Set of tool names the executor can call.
                          If None, all template tools are assumed available.
        base_confidence : Default confidence applied to all scored plans.
    """

    def __init__(
        self,
        available_tools: Optional[List[str]] = None,
        base_confidence: float = 0.7,
    ) -> None:
        self.available_tools: Optional[set[str]] = (
            set(available_tools) if available_tools is not None else None
        )
        self.base_confidence = base_confidence

    # ------------------------------------------------------------------ #
    # AbstractPlanner implementation                                       #
    # ------------------------------------------------------------------ #

    def generate_plans(
        self,
        goal: Goal,
        world_state: Dict[str, Any],
        n_candidates: int = 3,
    ) -> List[Plan]:
        """
        Produce up to *n_candidates* Plans from the template library.

        Each template variant for the detected intent becomes one Plan.
        Plans are scored immediately so the caller can pick the best one.
        """
        if goal.status not in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            logger.warning(
                f"Planner asked to plan for goal in status "
                f"{goal.status.value!r} — returning empty list"
            )
            return []

        intent   = _detect_intent(goal.description)
        variants = _TEMPLATES.get(intent, _TEMPLATES["default"])
        logger.debug(
            f"Planner detected intent={intent!r} for goal={goal.goal_id!r}"
        )

        plans: List[Plan] = []
        for template in variants:
            steps = _fill_template(template, goal.description)

            # Filter: skip if any step requires a tool we don't have
            if self.available_tools is not None:
                if any(s["tool"] not in self.available_tools for s in steps):
                    continue

            plan = Plan(
                plan_id=str(uuid.uuid4()),
                goal_id=goal.goal_id,
                steps=steps,
                score=0.0,
                metadata={
                    "intent": intent,
                    "template_len": len(steps),
                },
            )
            plan.score = self._score(plan, goal, world_state)
            plans.append(plan)

            if len(plans) >= n_candidates:
                break

        # Fall back: always provide at least one plan
        if not plans:
            fallback_steps = _fill_template(
                _TEMPLATES["default"][0], goal.description
            )
            plans.append(Plan(
                plan_id=str(uuid.uuid4()),
                goal_id=goal.goal_id,
                steps=fallback_steps,
                score=self.base_confidence * 0.5,
                metadata={"intent": "fallback"},
            ))

        logger.info(
            f"Planner generated {len(plans)} plan(s) "
            f"for goal={goal.goal_id!r} intent={intent!r}"
        )
        return plans

    def select_plan(
        self,
        plans: List[Plan],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """
        Select the highest-scoring plan.

        Respects hard constraints:
          - ``constraints["max_steps"]`` : discard plans with more steps.
          - ``constraints["required_tools"]`` : discard plans missing tools.

        Phase 4: override this method with QAOA-based selector.
        """
        if not plans:
            raise ValueError("No plans to select from")

        candidates = list(plans)

        if constraints:
            max_steps = constraints.get("max_steps")
            required  = set(constraints.get("required_tools") or [])

            candidates = [
                p for p in candidates
                if (max_steps is None or len(p.steps) <= max_steps)
                and (not required or required.issubset({s["tool"] for s in p.steps}))
            ]

            if not candidates:
                # Truncate the best scoring plan to satisfy max_steps
                if max_steps is not None and max_steps > 0:
                    fallback = max(plans, key=lambda p: p.score)
                    import dataclasses
                    truncated = dataclasses.replace(
                        fallback,
                        plan_id=str(uuid.uuid4()),
                        steps=fallback.steps[:max_steps],
                    )
                    logger.warning(
                        f"All plans exceeded max_steps={max_steps}; "
                        f"returning truncated plan with {len(truncated.steps)} steps"
                    )
                    return truncated
                logger.warning(
                    "All plans were filtered out by constraints — "
                    "returning highest-score original plan"
                )
                candidates = plans

        best = max(candidates, key=lambda p: p.score)
        logger.info(
            f"Planner selected plan_id={best.plan_id!r} "
            f"score={best.score:.3f} steps={len(best.steps)}"
        )
        return best

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #

    def _score(
        self,
        plan: Plan,
        goal: Goal,
        world_state: Dict[str, Any],
    ) -> float:
        """
        Composite score in [0, 1]:
          - feasibility  40 % : all tools available, state has useful context
          - efficiency   30 % : shorter plans score higher
          - confidence   30 % : base_confidence × goal priority
        """
        # Feasibility: deduct 0.1 per unknown tool
        unknown = 0
        if self.available_tools is not None:
            unknown = sum(
                1 for s in plan.steps
                if s["tool"] not in self.available_tools
            )
        feasibility = max(0.0, 1.0 - unknown * 0.1)

        # Efficiency: prefer fewer steps (normalise against max template length = 4)
        max_len = 4
        efficiency = max(0.0, 1.0 - (len(plan.steps) - 1) / max(max_len, 1))

        # Confidence: base × goal priority
        confidence = self.base_confidence * goal.priority

        score = 0.4 * feasibility + 0.3 * efficiency + 0.3 * confidence
        return round(score, 4)
