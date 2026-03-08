"""
ToolRegistry — Central catalogue of available tools.

All tools registered here are discoverable by the ExecutiveController
for LLM-style function-calling (tool-use agent).

Usage::

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(CodeSandbox())

    result = registry.call("web_search", query="Belize reef coral bleaching")
    print(result.output)

The registry is thread-safe for concurrent reads; concurrent registration
is protected by a lock.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from action.interfaces import (
    AbstractTool,
    AbstractToolRegistry,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


class ToolRegistry(AbstractToolRegistry):
    """
    Thread-safe tool catalogue.

    Args:
        safety_screener : Optional InputScreener.  When provided, the
                          ``query`` / ``code`` / ``input`` keyword argument
                          of each call is screened before execution.
    """

    def __init__(self, safety_screener: Optional[Any] = None) -> None:
        self._tools:    Dict[str, AbstractTool] = {}
        self._screener  = safety_screener
        self._lock      = threading.RLock()

        logger.info("ToolRegistry initialised")

    # ------------------------------------------------------------------ #
    # Registry management                                                  #
    # ------------------------------------------------------------------ #

    def register(self, tool: AbstractTool) -> None:
        """Register a tool.  Overwrites any existing tool with the same name."""
        with self._lock:
            self._tools[tool.spec.name] = tool
        logger.info(f"ToolRegistry: registered '{tool.spec.name}'")

    def unregister(self, name: str) -> bool:
        """Remove a tool.  Returns True if found and removed."""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.info(f"ToolRegistry: unregistered '{name}'")
                return True
        return False

    def get(self, name: str) -> Optional[AbstractTool]:
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        with self._lock:
            return [t.spec for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Look up and execute a tool by name.

        Returns:
            ToolResult — status=FAILURE if tool not found,
                         status=BLOCKED if screener rejects.
        """
        tool = self.get(tool_name)
        if tool is None:
            logger.warning(f"ToolRegistry: unknown tool '{tool_name}'")
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILURE,
                error=f"Tool '{tool_name}' is not registered",
            )

        # Optional safety screening of user-provided string arguments
        if self._screener is not None and not tool.spec.safe:
            for arg_key in ("query", "code", "input", "prompt"):
                val = kwargs.get(arg_key)
                if isinstance(val, str):
                    screen = self._screener.screen(val)
                    if not screen.is_safe:
                        logger.warning(
                            f"ToolRegistry: screener blocked call to "
                            f"'{tool_name}' on arg '{arg_key}': {screen.flags}"
                        )
                        return ToolResult(
                            tool_name=tool_name,
                            status=ToolStatus.BLOCKED,
                            error=f"Blocked by safety screener: {screen.flags}",
                        )

        try:
            return tool.run(**kwargs)
        except Exception as exc:
            logger.error(f"ToolRegistry: '{tool_name}' raised {exc}")
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILURE,
                error=str(exc),
            )
