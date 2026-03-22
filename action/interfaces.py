"""
Action interfaces — ABCs and data-classes for the Motor Cortex.

All concrete action/tool implementations must satisfy these contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Enumerations                                                                  #
# --------------------------------------------------------------------------- #


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Safety filter prevented execution


class ToolCategory(str, Enum):
    WEB_SEARCH = "web_search"
    CODE_EXEC = "code_execution"
    MEMORY = "memory"
    BLOCKCHAIN = "blockchain"
    IO = "io"
    CUSTOM = "custom"


# --------------------------------------------------------------------------- #
# Data classes                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class ToolResult:
    """
    Standardised result returned by every tool call.

    Attributes:
        tool_name : Name of the tool that was called.
        status    : Execution outcome.
        output    : Primary output (string, JSON-serialisable).
        error     : Error message if status != SUCCESS.
        metadata  : Optional extra data (latency, source URLs, etc.).
        cost      : Estimated cost in DALLA tokens (0.0 if free).
    """

    tool_name: str
    status: ToolStatus = ToolStatus.SUCCESS
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class ToolSpec:
    """
    Declarative description of a tool (used for LLM function-calling).

    Attributes:
        name        : Unique tool identifier.
        description : What the tool does (shown to the LLM).
        parameters  : JSON-schema-style parameter descriptions.
        category    : Tool category.
        safe        : Whether the tool can run without a safety pre-check.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: ToolCategory = ToolCategory.CUSTOM
    safe: bool = True


# --------------------------------------------------------------------------- #
# Abstract base classes                                                         #
# --------------------------------------------------------------------------- #


class AbstractTool(ABC):
    """Base class for all callable tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the declarative ToolSpec."""

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments."""

    def __call__(self, **kwargs: Any) -> ToolResult:
        return self.run(**kwargs)


class AbstractToolRegistry(ABC):
    """Manages a catalogue of available tools."""

    @abstractmethod
    def register(self, tool: AbstractTool) -> None:
        """Add *tool* to the registry."""

    @abstractmethod
    def get(self, name: str) -> Optional[AbstractTool]:
        """Return tool by name, or None."""

    @abstractmethod
    def list_tools(self) -> List[ToolSpec]:
        """Return specs of all registered tools."""

    @abstractmethod
    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Look up and execute a tool by name."""


class AbstractActionLayer(ABC):
    """High-level Motor Cortex facade."""

    @abstractmethod
    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a named tool action."""

    @abstractmethod
    def available_tools(self) -> List[ToolSpec]:
        """List all available tools."""
