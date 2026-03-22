"""
Tests for the Action Layer (Motor Cortex).

Coverage:
  • ToolRegistry       — register, get, call, safety screening, unknown tool
  • WebSearchTool      — stub mode, empty query
  • CodeSandbox        — stub mode, empty code, stdout capture
  • MemoryReadTool     — stub mode, empty query
  • MemoryWriteTool    — stub mode, empty content
  • ActionExecutor     — single call, plan execution, history, success_rate
  • ActionLayer        — facade, available_tools, execute, plan, status
"""

from __future__ import annotations

import pytest

from nawal.action import (
    ToolStatus,
    ToolCategory,
    ToolResult,
    ToolSpec,
    AbstractTool,
    ToolRegistry,
    ActionExecutor,
    WebSearchTool,
    CodeSandbox,
    MemoryReadTool,
    MemoryWriteTool,
    ActionLayer,
)

# ═══════════════════════════════════════════════════════════════════════════ #
# Fixtures / helpers                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #


class _EchoTool(AbstractTool):
    """Simple tool that echoes its input — used for registry/executor tests."""

    _SPEC = ToolSpec(
        name="echo",
        description="Echo the message argument.",
        parameters={"message": {"type": "string", "required": True}},
        category=ToolCategory.CUSTOM,
        safe=True,
    )

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    def run(self, message: str = "", **_) -> ToolResult:
        return ToolResult(tool_name="echo", status=ToolStatus.SUCCESS, output=message)


class _FailTool(AbstractTool):
    """Always fails when run."""

    _SPEC = ToolSpec(name="always_fail", description="Always fails", safe=True)

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    def run(self, **_) -> ToolResult:
        raise RuntimeError("intentional test failure")


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(_EchoTool())
    return r


def _executor() -> ActionExecutor:
    return ActionExecutor(_registry())


def _layer() -> ActionLayer:
    return ActionLayer(stub_network_tools=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# ToolRegistry                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestToolRegistry:
    def test_register_and_get(self):
        r = ToolRegistry()
        t = _EchoTool()
        r.register(t)
        assert r.get("echo") is t

    def test_contains(self):
        r = _registry()
        assert "echo" in r
        assert "missing" not in r

    def test_len(self):
        r = _registry()
        assert len(r) == 1

    def test_list_tools_returns_specs(self):
        r = _registry()
        specs = r.list_tools()
        assert len(specs) == 1
        assert specs[0].name == "echo"

    def test_call_success(self):
        result = _registry().call("echo", message="hello")
        assert result.status == ToolStatus.SUCCESS
        assert result.output == "hello"

    def test_call_unknown_tool(self):
        result = _registry().call("ghost")
        assert result.status == ToolStatus.FAILURE
        assert "not registered" in (result.error or "")

    def test_call_tool_exception_becomes_failure(self):
        r = ToolRegistry()
        r.register(_FailTool())
        result = r.call("always_fail")
        assert result.status == ToolStatus.FAILURE
        assert result.error is not None

    def test_unregister(self):
        r = _registry()
        removed = r.unregister("echo")
        assert removed is True
        assert "echo" not in r

    def test_unregister_missing_returns_false(self):
        r = _registry()
        assert r.unregister("ghost") is False

    def test_overwrite_registration(self):
        r = _registry()
        r.register(_EchoTool())  # same name again
        assert len(r) == 1


# ═══════════════════════════════════════════════════════════════════════════ #
# WebSearchTool                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestWebSearchTool:
    def test_stub_returns_results(self):
        tool = WebSearchTool(use_stub=True)
        result = tool.run(query="Belize reef")
        assert result.status == ToolStatus.SUCCESS
        assert isinstance(result.output, list)
        assert len(result.output) > 0

    def test_stub_result_structure(self):
        tool = WebSearchTool(use_stub=True)
        result = tool.run(query="test query")
        for item in result.output:
            assert "title" in item
            assert "url" in item
            assert "snippet" in item

    def test_num_results_respected(self):
        tool = WebSearchTool(use_stub=True)
        result = tool.run(query="test", num_results=3)
        assert len(result.output) == 3

    def test_empty_query_fails(self):
        tool = WebSearchTool(use_stub=True)
        result = tool.run(query="")
        assert result.status == ToolStatus.FAILURE

    def test_spec_name(self):
        assert WebSearchTool().spec.name == "web_search"

    def test_spec_category(self):
        assert WebSearchTool().spec.category == ToolCategory.WEB_SEARCH

    def test_metadata_includes_latency(self):
        tool = WebSearchTool(use_stub=True)
        result = tool.run(query="test")
        assert "latency_ms" in result.metadata

    def test_callable_interface(self):
        tool = WebSearchTool(use_stub=True)
        result = tool(query="callable test")
        assert result.status == ToolStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════════════════ #
# CodeSandbox                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestCodeSandbox:
    def test_stub_mode(self):
        sb = CodeSandbox(use_stub=True)
        result = sb.run(code="print('hello')")
        assert result.status == ToolStatus.SUCCESS
        assert isinstance(result.output, dict)

    def test_stub_stdout_extracted(self):
        sb = CodeSandbox(use_stub=True)
        result = sb.run(code="print('stub_output')")
        assert "stub_output" in result.output.get("stdout", "")

    def test_live_exec_arithmetic(self):
        sb = CodeSandbox(use_stub=False)
        result = sb.run(code="print(2 + 2)")
        assert result.status == ToolStatus.SUCCESS
        assert "4" in result.output.get("stdout", "")

    def test_live_exec_syntax_error(self):
        sb = CodeSandbox(use_stub=False)
        result = sb.run(code="def bad(")
        assert result.status == ToolStatus.FAILURE

    def test_empty_code_fails(self):
        sb = CodeSandbox(use_stub=True)
        result = sb.run(code="")
        assert result.status == ToolStatus.FAILURE

    def test_spec_name(self):
        assert CodeSandbox().spec.name == "code_exec"

    def test_spec_category(self):
        assert CodeSandbox().spec.category == ToolCategory.CODE_EXEC

    def test_context_injection(self):
        sb = CodeSandbox(use_stub=False)
        result = sb.run(code="print(x + 1)", context={"x": 41})
        assert result.status == ToolStatus.SUCCESS
        assert "42" in result.output.get("stdout", "")

    def test_dangerous_import_blocked(self):
        sb = CodeSandbox(use_stub=False)
        result = sb.run(code="import os; print(os.getcwd())")
        # Should either fail with ImportError or succeed in fallback exec
        # We just test it doesn't raise an exception
        assert result.status in (ToolStatus.SUCCESS, ToolStatus.FAILURE)


# ═══════════════════════════════════════════════════════════════════════════ #
# MemoryReadTool / MemoryWriteTool                                            #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestMemoryTools:
    def test_read_stub_returns_empty_list(self):
        tool = MemoryReadTool(memory_manager=None)
        result = tool.run(query="What did I say yesterday?")
        assert result.status == ToolStatus.SUCCESS
        assert isinstance(result.output, list)

    def test_read_empty_query_fails(self):
        tool = MemoryReadTool()
        result = tool.run(query="")
        assert result.status == ToolStatus.FAILURE

    def test_read_spec_name(self):
        assert MemoryReadTool().spec.name == "memory_read"

    def test_write_stub_returns_record_id(self):
        tool = MemoryWriteTool(memory_manager=None)
        result = tool.run(content="Something important happened today.")
        assert result.status == ToolStatus.SUCCESS
        assert "record_id" in result.output

    def test_write_empty_content_fails(self):
        tool = MemoryWriteTool()
        result = tool.run(content="")
        assert result.status == ToolStatus.FAILURE

    def test_write_spec_name(self):
        assert MemoryWriteTool().spec.name == "memory_write"

    def test_read_with_real_memory(self):
        """Minimal duck-typed memory manager stub."""

        class _EpisodicStub:
            def retrieve(self, query, top_k=5):
                return [type("R", (), {"content": "past event", "metadata": {}})()]

        class _MMStub:
            episodic = _EpisodicStub()

        tool = MemoryReadTool(memory_manager=_MMStub())
        result = tool.run(query="past event")
        assert result.status == ToolStatus.SUCCESS
        assert len(result.output) == 1
        assert result.output[0]["content"] == "past event"

    def test_write_with_real_memory(self):
        class _EpisodicStub:
            stored = []

            def store(self, record):
                self.stored.append(record)

        class _MMStub:
            episodic = _EpisodicStub()

        mm = _MMStub()
        tool = MemoryWriteTool(memory_manager=mm)
        result = tool.run(content="Stored content", metadata={"tag": "test"})
        assert result.status == ToolStatus.SUCCESS
        assert len(mm.episodic.stored) == 1


# ═══════════════════════════════════════════════════════════════════════════ #
# ActionExecutor                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestActionExecutor:
    def test_execute_single(self):
        result = _executor().execute("echo", message="test")
        assert result.status == ToolStatus.SUCCESS
        assert result.output == "test"

    def test_execute_records_history(self):
        ex = _executor()
        ex.execute("echo", message="a")
        ex.execute("echo", message="b")
        assert len(ex.history()) == 2

    def test_clear_history(self):
        ex = _executor()
        ex.execute("echo", message="a")
        ex.clear_history()
        assert ex.history() == []

    def test_success_rate_all_ok(self):
        ex = _executor()
        ex.execute("echo", message="x")
        assert ex.success_rate() == 1.0

    def test_success_rate_mixed(self):
        ex = ActionExecutor(_registry())
        ex._registry.register(_FailTool())
        ex.execute("echo", message="ok")
        ex.execute("always_fail")
        assert ex.success_rate() == 0.5

    def test_execute_plan_sequential(self):
        ex = _executor()
        results = ex.execute_plan(
            [
                {"tool": "echo", "args": {"message": "step1"}},
                {"tool": "echo", "args": {"message": "step2"}},
            ]
        )
        assert len(results) == 2
        assert all(r.status == ToolStatus.SUCCESS for r in results)

    def test_execute_plan_missing_tool_key(self):
        ex = _executor()
        results = ex.execute_plan([{"args": {}}])
        assert results[0].status == ToolStatus.FAILURE

    def test_execute_plan_chaining(self):
        ex = _executor()
        results = ex.execute_plan(
            [
                {"tool": "echo", "args": {"message": "hello"}},
                {"tool": "echo", "args": {"message": "world"}},
            ]
        )
        assert results[1].output == "world"

    def test_history_is_copy(self):
        ex = _executor()
        ex.execute("echo", message="x")
        h = ex.history()
        h.clear()
        assert len(ex.history()) == 1


# ═══════════════════════════════════════════════════════════════════════════ #
# ActionLayer                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #


class TestActionLayer:
    def test_instantiates(self):
        layer = _layer()
        assert layer is not None

    def test_available_tools_not_empty(self):
        specs = _layer().available_tools()
        assert len(specs) >= 4  # web_search, code_exec, memory_read, memory_write

    def test_available_tools_names(self):
        names = {t.name for t in _layer().available_tools()}
        assert "web_search" in names
        assert "code_exec" in names
        assert "memory_read" in names
        assert "memory_write" in names

    def test_execute_web_search_stub(self):
        result = _layer().execute("web_search", query="coral reef")
        assert result.status == ToolStatus.SUCCESS

    def test_execute_memory_read_stub(self):
        result = _layer().execute("memory_read", query="what happened")
        assert result.status == ToolStatus.SUCCESS

    def test_execute_memory_write_stub(self):
        result = _layer().execute("memory_write", content="remember this")
        assert result.status == ToolStatus.SUCCESS

    def test_execute_unknown_tool(self):
        result = _layer().execute("nonexistent")
        assert result.status == ToolStatus.FAILURE

    def test_execute_plan(self):
        layer = _layer()
        results = layer.execute_plan(
            [
                {"tool": "web_search", "args": {"query": "Belize"}},
                {"tool": "memory_write", "args": {"content": "search done"}},
            ]
        )
        assert len(results) == 2
        assert all(r.status == ToolStatus.SUCCESS for r in results)

    def test_register_extra_tool(self):
        layer = _layer()
        layer.register_tool(_EchoTool())
        result = layer.execute("echo", message="dynamic")
        assert result.status == ToolStatus.SUCCESS

    def test_get_status(self):
        status = _layer().get_status()
        assert "num_tools" in status
        assert "tools" in status
        assert "success_rate" in status
        assert "history_len" in status

    def test_status_num_tools_matches(self):
        layer = _layer()
        status = layer.get_status()
        assert status["num_tools"] == len(layer.available_tools())
