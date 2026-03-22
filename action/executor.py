"""
ActionExecutor — dispatches ordered sequences of tool calls.

The executor coordinates calling one or many tools in sequence,
propagates outputs as ``context`` to subsequent calls (chaining),
and records per-call telemetry.

Usage::

    executor = ActionExecutor(registry)
    result = executor.execute("web_search", query="Belize Garifuna history")
    print(result.output)

    # Multi-step plan
    results = executor.execute_plan([
        {"tool": "web_search",  "args": {"query": "coral bleaching 2024"}},
        {"tool": "memory_write","args": {"content": "<result from step 1>"}},
    ])
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from action.interfaces import ToolResult, ToolStatus
from action.tool_registry import ToolRegistry


class ActionExecutor:
    """
    Dispatches tool calls through the ToolRegistry.

    Args:
        registry       : Populated ToolRegistry.
        chain_outputs  : Whether to inject the previous result's output as
                         ``_prev_output`` in subsequent calls (default True).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        chain_outputs: bool = True,
    ) -> None:
        self._registry = registry
        self._chain = chain_outputs
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Single tool call                                                     #
    # ------------------------------------------------------------------ #

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a single tool and record the call in history."""
        t0 = time.time()
        result = self._registry.call(tool_name, **kwargs)
        elapsed = (time.time() - t0) * 1000

        self._history.append(
            {
                "tool": tool_name,
                "kwargs": {k: v for k, v in kwargs.items() if k != "code"},  # skip large blobs
                "status": result.status,
                "latency_ms": elapsed,
            }
        )

        logger.debug(
            f"ActionExecutor: '{tool_name}' → {result.status.value} " f"({elapsed:.1f} ms)"
        )
        return result

    # ------------------------------------------------------------------ #
    # Plan execution (list of steps)                                       #
    # ------------------------------------------------------------------ #

    def execute_plan(self, steps: list[dict[str, Any]]) -> list[ToolResult]:
        """
        Execute an ordered plan.

        Each step is a dict::

            {"tool": "<name>", "args": {<kwargs>}}

        If ``chain_outputs=True``, the string representation of the previous
        step's ``output`` is injected as ``_prev_output`` into the next step's
        args (unless the next step already provides that key).

        Returns: list of ToolResult, one per step.
        """
        results: list[ToolResult] = []
        prev_output: Any = None

        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            args = dict(step.get("args", {}))

            if self._chain and prev_output is not None and "_prev_output" not in args:
                args["_prev_output"] = prev_output

            if not tool_name:
                r = ToolResult(
                    tool_name="",
                    status=ToolStatus.FAILURE,
                    error=f"Step {i}: missing 'tool' key",
                )
                results.append(r)
                continue

            result = self.execute(tool_name, **args)
            results.append(result)
            prev_output = result.output if result.status == ToolStatus.SUCCESS else None

        return results

    # ------------------------------------------------------------------ #
    # Telemetry / introspection                                            #
    # ------------------------------------------------------------------ #

    def history(self) -> list[dict[str, Any]]:
        """Return a copy of the execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def success_rate(self) -> float:
        """Fraction of calls that returned SUCCESS."""
        if not self._history:
            return 1.0
        ok = sum(1 for h in self._history if h["status"] == ToolStatus.SUCCESS)
        return ok / len(self._history)
