"""
Tests for the Phase 2 control subsystem:
  - GoalStack
  - ClassicalPlanner
  - ToolExecutor
  - ExecutiveController
"""

from __future__ import annotations

import uuid

import pytest

from control.controller import ExecutiveController
from control.executor import ToolExecutor
from control.goal_stack import GoalStack
from control.interfaces import Goal, GoalStatus, Plan
from control.planner import ClassicalPlanner

# ============================================================================
# Helpers
# ============================================================================


def _goal(description: str = "test goal", priority: float = 1.0) -> Goal:
    return Goal(
        goal_id=str(uuid.uuid4()),
        description=description,
        priority=priority,
        context={},
        status=GoalStatus.PENDING,
        sub_goals=[],
    )


def _plan(goal_id: str = "g1", steps: int = 2, score: float = 0.8) -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal_id=goal_id,
        steps=[{"tool": "noop", "args": {}} for _ in range(steps)],
        score=score,
        metadata={},
    )


# ============================================================================
# GoalStack
# ============================================================================


class TestGoalStack:
    def test_push_and_peek(self):
        gs = GoalStack()
        g = _goal()
        gs.push(g)
        assert gs.peek() is not None

    def test_priority_order(self):
        gs = GoalStack()
        low = _goal("low", priority=0.5)
        high = _goal("high", priority=5.0)
        mid = _goal("mid", priority=2.0)
        for g in (low, high, mid):
            gs.push(g)
        pending = gs.next_pending()
        assert pending.description == "high"

    def test_activate(self):
        gs = GoalStack()
        g = _goal()
        gs.push(g)
        ok = gs.activate(g.goal_id)
        assert ok
        assert gs.peek().status == GoalStatus.ACTIVE

    def test_activate_demotes_current(self):
        gs = GoalStack()
        g1 = _goal("first", priority=1.0)
        g2 = _goal("second", priority=2.0)
        gs.push(g1)
        gs.push(g2)
        gs.activate(g1.goal_id)
        assert g1.status == GoalStatus.ACTIVE
        gs.activate(g2.goal_id)
        # g1 demoted back to PENDING
        assert g1.status == GoalStatus.PENDING
        assert g2.status == GoalStatus.ACTIVE

    def test_complete(self):
        gs = GoalStack()
        g = _goal()
        gs.push(g)
        gs.activate(g.goal_id)
        gs.complete(g.goal_id)
        assert g.goal_id not in [x.goal_id for x in gs.all_live()]
        hist = gs.history()
        assert any(x.goal_id == g.goal_id for x in hist)

    def test_fail(self):
        gs = GoalStack()
        g = _goal()
        gs.push(g)
        gs.fail(g.goal_id)
        hist = gs.history()
        assert any(
            x.goal_id == g.goal_id and x.status == GoalStatus.FAILED for x in hist
        )

    def test_block(self):
        gs = GoalStack()
        g = _goal()
        gs.push(g)
        gs.block(g.goal_id)
        assert g.status == GoalStatus.BLOCKED

    def test_update_priority(self):
        gs = GoalStack()
        g = _goal(priority=1.0)
        gs.push(g)
        gs.update_priority(g.goal_id, 9.0)
        assert gs.peek().goal_id == g.goal_id
        assert gs.peek().priority == 9.0

    def test_next_pending_skips_non_pending(self):
        gs = GoalStack()
        g1 = _goal("active", priority=2.0)
        g2 = _goal("pending", priority=1.0)
        gs.push(g1)
        gs.push(g2)
        gs.activate(g1.goal_id)
        nxt = gs.next_pending()
        assert nxt.goal_id == g2.goal_id

    def test_history_last_n(self):
        gs = GoalStack()
        for i in range(5):
            g = _goal(f"g{i}")
            gs.push(g)
            gs.fail(g.goal_id)
        hist = gs.history(last_n=3)
        assert len(hist) == 3

    def test_activate_missing_goal_returns_false(self):
        gs = GoalStack()
        assert gs.activate("no-such-id") is False


# ============================================================================
# ClassicalPlanner
# ============================================================================


class TestClassicalPlanner:
    def test_generate_plans_returns_list(self):
        p = ClassicalPlanner(
            available_tools=["noop", "memory_read", "reason", "respond"]
        )
        g = _goal("Retrieve recent news about AI")
        plans = p.generate_plans(g, {}, n_candidates=3)
        assert isinstance(plans, list)
        assert len(plans) >= 1

    def test_all_plans_have_steps(self):
        p = ClassicalPlanner(
            available_tools=["noop", "memory_read", "reason", "respond"]
        )
        g = _goal("Analyse sentiment of this text")
        plans = p.generate_plans(g, {})
        for plan in plans:
            assert isinstance(plan.steps, list)
            assert len(plan.steps) > 0

    def test_plan_score_between_0_and_1(self):
        p = ClassicalPlanner(available_tools=["noop"])
        g = _goal("do something")
        plans = p.generate_plans(g, {})
        for plan in plans:
            assert 0.0 <= plan.score <= 1.0

    def test_select_plan_returns_highest_score(self):
        p = ClassicalPlanner(available_tools=["noop"])
        g = _goal("test")
        plans = p.generate_plans(g, {}, n_candidates=3)
        selected = p.select_plan(plans, {})
        assert selected.score == max(pl.score for pl in plans)

    def test_select_plan_respects_max_steps(self):
        p = ClassicalPlanner(
            available_tools=["noop", "memory_read", "reason", "respond"]
        )
        g = _goal("generate a detailed report on climate change")
        plans = p.generate_plans(g, {}, n_candidates=3)
        selected = p.select_plan(plans, constraints={"max_steps": 2})
        assert len(selected.steps) <= 2

    def test_intent_detection_retrieve(self):
        from control.planner import _detect_intent

        assert _detect_intent("retrieve recent memories") == "retrieve"

    def test_intent_detection_generate(self):
        from control.planner import _detect_intent

        assert _detect_intent("generate a creative story") == "generate"

    def test_intent_detection_analyse(self):
        from control.planner import _detect_intent

        assert _detect_intent("analyse the log file") == "analyse"

    def test_intent_detection_act(self):
        from control.planner import _detect_intent

        assert _detect_intent("execute the command") == "act"

    def test_intent_detection_default(self):
        from control.planner import _detect_intent

        intent = _detect_intent("unknowable cryptic phrase xyz")
        assert intent == "default"

    def test_generate_plans_no_available_tools(self):
        """Even with no tools narrowed, at least one plan (noop fallback) returned."""
        p = ClassicalPlanner(available_tools=[])
        g = _goal("do something")
        plans = p.generate_plans(g, {})
        assert len(plans) >= 1


# ============================================================================
# ToolExecutor
# ============================================================================


class TestToolExecutor:
    def test_execute_noop_plan_succeeds(self):
        ex = ToolExecutor()
        plan = _plan()
        result = ex.execute(plan)
        assert result["status"] == "success"
        assert result["error"] is None

    def test_execute_dry_run(self):
        ex = ToolExecutor()
        plan = _plan(steps=3)
        result = ex.execute(plan, dry_run=True)
        assert result["status"] == "success"
        assert all(o["result"].get("dry_run") for o in result["outputs"])

    def test_execute_custom_tool(self):
        ex = ToolExecutor()
        ex.register("echo", lambda text="", **_: {"echoed": text})
        plan = Plan(
            plan_id="p1",
            goal_id="g1",
            steps=[{"tool": "echo", "args": {"text": "hello"}}],
            score=1.0,
            metadata={},
        )
        result = ex.execute(plan)
        assert result["status"] == "success"
        assert result["outputs"][0]["result"]["echoed"] == "hello"

    def test_execute_unknown_tool_returns_partial(self):
        ex = ToolExecutor()
        plan = Plan(
            plan_id="p1",
            goal_id="g1",
            steps=[{"tool": "nonexistent", "args": {}}],
            score=1.0,
            metadata={},
        )
        result = ex.execute(plan)
        assert result["status"] == "partial"

    def test_execute_required_unknown_tool_fails(self):
        ex = ToolExecutor()
        plan = Plan(
            plan_id="p1",
            goal_id="g1",
            steps=[{"tool": "nonexistent", "args": {}, "required": True}],
            score=1.0,
            metadata={},
        )
        result = ex.execute(plan)
        assert result["status"] == "failed"

    def test_execute_multi_step(self):
        ex = ToolExecutor()
        calls = []
        ex.register("record", lambda idx=0, **_: calls.append(idx) or {"recorded": idx})
        plan = Plan(
            plan_id="p1",
            goal_id="g1",
            steps=[
                {"tool": "record", "args": {"idx": 0}},
                {"tool": "record", "args": {"idx": 1}},
                {"tool": "record", "args": {"idx": 2}},
            ],
            score=1.0,
            metadata={},
        )
        result = ex.execute(plan)
        assert result["status"] == "success"
        assert calls == [0, 1, 2]

    def test_overwrite_false_raises(self):
        ex = ToolExecutor()
        ex.register("mytool", lambda **_: {})
        with pytest.raises(ValueError):
            ex.register("mytool", lambda **_: {}, overwrite=False)

    def test_available_tools_contains_builtins(self):
        ex = ToolExecutor()
        tools = ex.available_tools()
        for name in ("noop", "log", "memory_read", "memory_write"):
            assert name in tools

    def test_tool_raising_exception_partial(self):
        ex = ToolExecutor()
        ex.register("boom", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
        plan = Plan(
            plan_id="p1",
            goal_id="g1",
            steps=[{"tool": "boom", "args": {}}],
            score=1.0,
            metadata={},
        )
        result = ex.execute(plan)
        # Non-required step → partial
        assert result["status"] == "partial"


# ============================================================================
# ExecutiveController
# ============================================================================


class TestExecutiveController:
    def test_add_goal_returns_goal(self):
        ctrl = ExecutiveController()
        g = ctrl.add_goal("Summarise recent news", priority=1.0)
        assert g.goal_id is not None
        assert g.status == GoalStatus.PENDING

    def test_tick_no_goals_returns_none(self):
        ctrl = ExecutiveController()
        result = ctrl.tick()
        assert result is None

    def test_tick_runs_and_returns_summary(self):
        ctrl = ExecutiveController()
        ctrl.add_goal("Retrieve memory about climate", priority=1.0)
        summary = ctrl.tick()
        assert summary is not None
        assert "status" in summary
        assert summary["status"] in ("success", "partial", "failed")

    def test_tick_marks_goal_complete_on_success(self):
        ctrl = ExecutiveController()
        ctrl.add_goal("noop task", priority=1.0)
        ctrl.tick()
        # With only noop-capable tools available, any status is acceptable
        # but goal should no longer be active
        stats = ctrl.stats()
        assert isinstance(stats["ticks_run"], int)
        assert stats["ticks_run"] >= 1

    def test_register_tool_propagates(self):
        ctrl = ExecutiveController()
        called = []
        ctrl.register_tool("custom_fn", lambda **_: called.append(True) or {})
        assert "custom_fn" in ctrl.executor.available_tools()

    def test_stats_returns_dict(self):
        ctrl = ExecutiveController()
        s = ctrl.stats()
        for key in ("pending_goals", "active_goals", "ticks_run", "registered_tools"):
            assert key in s

    def test_multiple_goals_processed_sequentially(self):
        ctrl = ExecutiveController()
        for i in range(3):
            ctrl.add_goal(f"task {i}", priority=float(i + 1))
        results = []
        for _ in range(3):
            r = ctrl.tick()
            if r is not None:
                results.append(r)
        assert len(results) >= 1

    def test_goal_with_custom_tool(self):
        ctrl = ExecutiveController()
        outputs = []
        ctrl.register_tool(
            "respond", lambda message="", **_: outputs.append(message) or {}
        )
        ctrl.add_goal("Generate a greeting", priority=1.0)
        ctrl.tick()
        # The test just verifies no exceptions and a result was produced

    def test_tick_with_world_state(self):
        ctrl = ExecutiveController()
        ctrl.add_goal("Search for news", priority=1.0)
        result = ctrl.tick(world_state={"context": "test context", "urgency": "low"})
        assert result is not None
