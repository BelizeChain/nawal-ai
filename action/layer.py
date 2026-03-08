"""
ActionLayer — Motor Cortex facade.

Combines:
  • ToolRegistry        — catalogue of available tools
  • ActionExecutor      — dispatches tool calls
  • Default built-in tools (web_search, code_exec, memory_read, memory_write)

Usage::

    from nawal.action import ActionLayer

    layer = ActionLayer()
    result = layer.execute("web_search", query="Belize marine ecology")
    print(result.output)

    print([t.name for t in layer.available_tools()])
"""
from __future__ import annotations

from typing import Any, List, Optional

from loguru import logger

from action.interfaces import AbstractActionLayer, ToolResult, ToolSpec
from action.tool_registry import ToolRegistry
from action.executor import ActionExecutor
from action.tools.web_search import WebSearchTool
from action.tools.code_sandbox import CodeSandbox
from action.tools.memory_tool import MemoryReadTool, MemoryWriteTool


class ActionLayer(AbstractActionLayer):
    """
    High-level Motor Cortex facade.

    During construction the layer:
      1. Creates a ToolRegistry (optionally with a safety screener).
      2. Registers built-in tools (web_search, code_exec, memory_read,
         memory_write).
      3. Creates an ActionExecutor backed by that registry.

    Args:
        memory_manager    : MemoryManager — wired into MemoryReadTool /
                            MemoryWriteTool.  Pass None for stub mode.
        safety_screener   : InputScreener from MaintenanceLayer — screens
                            non-safe tool calls before execution.
        extra_tools       : Additional AbstractTool instances to register.
        stub_network_tools: Force network tools into stub/offline mode.
    """

    def __init__(
        self,
        memory_manager:     Optional[Any] = None,
        safety_screener:    Optional[Any] = None,
        extra_tools:        Optional[list] = None,
        stub_network_tools: bool = False,
    ) -> None:
        self._registry = ToolRegistry(safety_screener=safety_screener)
        self._executor = ActionExecutor(self._registry)

        # Register built-in tools
        self._registry.register(
            WebSearchTool(use_stub=stub_network_tools)
        )
        self._registry.register(
            CodeSandbox(use_stub=stub_network_tools)
        )
        self._registry.register(
            MemoryReadTool(memory_manager=memory_manager)
        )
        self._registry.register(
            MemoryWriteTool(memory_manager=memory_manager)
        )

        # Register any caller-supplied extras
        for tool in (extra_tools or []):
            self._registry.register(tool)

        logger.info(
            f"ActionLayer ready — {len(self._registry)} tools: "
            f"{[t.name for t in self._registry.list_tools()]}"
        )

    # ------------------------------------------------------------------ #
    # AbstractActionLayer interface                                        #
    # ------------------------------------------------------------------ #

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a named tool action."""
        return self._executor.execute(tool_name, **kwargs)

    def available_tools(self) -> List[ToolSpec]:
        """Return specs of all registered tools."""
        return self._registry.list_tools()

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    def execute_plan(self, steps: list) -> list:
        """Execute an ordered plan (see ActionExecutor.execute_plan)."""
        return self._executor.execute_plan(steps)

    def register_tool(self, tool: Any) -> None:
        """Dynamically add a tool at runtime."""
        self._registry.register(tool)

    def get_status(self) -> dict:
        """Return runtime telemetry."""
        return {
            "num_tools":    len(self._registry),
            "tools":        [t.name for t in self._registry.list_tools()],
            "success_rate": self._executor.success_rate(),
            "history_len":  len(self._executor.history()),
        }
