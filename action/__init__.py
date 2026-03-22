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

from action.executor import ActionExecutor
from action.interfaces import (
    AbstractActionLayer,
    AbstractTool,
    AbstractToolRegistry,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from action.layer import ActionLayer
from action.tool_registry import ToolRegistry
from action.tools.code_sandbox import CodeSandbox
from action.tools.memory_tool import MemoryReadTool, MemoryWriteTool
from action.tools.web_search import WebSearchTool

__all__ = [
    "AbstractActionLayer",
    # ABCs
    "AbstractTool",
    "AbstractToolRegistry",
    "ActionExecutor",
    "ActionLayer",
    "CodeSandbox",
    "MemoryReadTool",
    "MemoryWriteTool",
    "ToolCategory",
    # Concrete classes
    "ToolRegistry",
    # Data-classes
    "ToolResult",
    "ToolSpec",
    # Enums
    "ToolStatus",
    "WebSearchTool",
]
