"""
Action Module — Nawal Brain Architecture (Motor Cortex / Cerebellum)

Sub-systems:
    interfaces    — ABCs + ToolResult / ToolSpec data-classes
    tool_registry — ToolRegistry (register / lookup / call)
    executor      — ActionExecutor (single + plan dispatch)
    tools/        — built-in tools: WebSearchTool, CodeSandbox, MemoryReadTool, MemoryWriteTool
    layer         — ActionLayer facade (single import point)

Usage::

    from nawal.action import ActionLayer, ToolResult, ToolStatus

    layer = ActionLayer()
    result = layer.execute("web_search", query="Belize reef")
    print(result.output)
"""

from action.interfaces import (
    ToolStatus,
    ToolCategory,
    ToolResult,
    ToolSpec,
    AbstractTool,
    AbstractToolRegistry,
    AbstractActionLayer,
)
from action.tool_registry import ToolRegistry
from action.executor import ActionExecutor
from action.tools.web_search import WebSearchTool
from action.tools.code_sandbox import CodeSandbox
from action.tools.memory_tool import MemoryReadTool, MemoryWriteTool
from action.layer import ActionLayer

__all__ = [
    # Enums
    "ToolStatus",
    "ToolCategory",
    # Data-classes
    "ToolResult",
    "ToolSpec",
    # ABCs
    "AbstractTool",
    "AbstractToolRegistry",
    "AbstractActionLayer",
    # Concrete classes
    "ToolRegistry",
    "ActionExecutor",
    "WebSearchTool",
    "CodeSandbox",
    "MemoryReadTool",
    "MemoryWriteTool",
    "ActionLayer",
]
