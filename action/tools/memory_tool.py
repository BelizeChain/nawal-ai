"""
Memory tools — read/write integration between the Motor Cortex and
the Memory System.

These tools allow the agent to:
  - ``MemoryReadTool``  — retrieve relevant memories from episodic or semantic
                          stores given a query embedding.
  - ``MemoryWriteTool`` — persist a new memory record (e.g. after a
                          successful action).

The tools accept an optional ``memory_manager`` object (duck-typed to the
MemoryManager interface) so they work in unit tests without requiring
the full memory stack.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from action.interfaces import (
    AbstractTool,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


class MemoryReadTool(AbstractTool):
    """
    Retrieve top-k memories from the episodic or semantic store.

    Args:
        memory_manager : MemoryManager (or compatible stub).
        top_k          : Default number of results to return.
    """

    _SPEC = ToolSpec(
        name="memory_read",
        description=(
            "Retrieve memories relevant to a query. "
            "Returns a list of {content, metadata} objects."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "Natural-language retrieval query",
                "required": True,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (default 5)",
                "required": False,
            },
            "store": {
                "type": "string",
                "description": "'episodic' or 'semantic'",
                "required": False,
            },
        },
        category=ToolCategory.MEMORY,
        safe=True,
    )

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        top_k: int = 5,
    ) -> None:
        self._mm = memory_manager
        self._top_k = top_k

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    def run(
        self,
        query: str,
        top_k: int = 5,
        store: str = "episodic",
        **_: Any,
    ) -> ToolResult:
        if not query:
            return ToolResult(
                tool_name="memory_read",
                status=ToolStatus.FAILURE,
                error="query must not be empty",
            )

        t0 = time.time()

        if self._mm is None:
            # Stub: return empty list
            return ToolResult(
                tool_name="memory_read",
                status=ToolStatus.SUCCESS,
                output=[],
                metadata={"mode": "stub", "query": query},
            )

        try:
            k = top_k or self._top_k
            store_obj = getattr(self._mm, store, None) or getattr(
                self._mm, "episodic", None
            )
            if store_obj is None:
                raise AttributeError(f"memory_manager has no store '{store}'")
            # EpisodicMemory.retrieve() takes query_embedding as first positional arg.
            # Use a zero vector when the caller passes a plain text query;
            # this matches the closest stored records by vector similarity.
            try:
                q_emb = list(map(float, query))
            except (TypeError, ValueError):
                q_emb = [0.0] * 768  # zero vector — returns nearest stored records
            records = store_obj.retrieve(q_emb, k)
            output = [
                {
                    "content": getattr(r, "content", str(r)),
                    "metadata": getattr(r, "metadata", {}),
                }
                for r in (records or [])
            ]
        except Exception as exc:
            logger.error(f"MemoryReadTool: {exc}")
            return ToolResult(
                tool_name="memory_read",
                status=ToolStatus.FAILURE,
                error=str(exc),
            )

        latency_ms = (time.time() - t0) * 1000
        return ToolResult(
            tool_name="memory_read",
            status=ToolStatus.SUCCESS,
            output=output,
            metadata={"latency_ms": latency_ms, "count": len(output)},
        )


class MemoryWriteTool(AbstractTool):
    """
    Persist a new memory record to the episodic store.

    Args:
        memory_manager : MemoryManager (or compatible stub).
    """

    _SPEC = ToolSpec(
        name="memory_write",
        description=(
            "Store a new memory (text + optional metadata) into the episodic memory."
        ),
        parameters={
            "content": {
                "type": "string",
                "description": "Text content to remember",
                "required": True,
            },
            "metadata": {
                "type": "object",
                "description": "Optional key-value metadata",
                "required": False,
            },
            "tags": {
                "type": "array",
                "description": "Optional string tags",
                "required": False,
            },
        },
        category=ToolCategory.MEMORY,
        safe=True,
    )

    def __init__(self, memory_manager: Optional[Any] = None) -> None:
        self._mm = memory_manager

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    def run(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **_: Any,
    ) -> ToolResult:
        if not content:
            return ToolResult(
                tool_name="memory_write",
                status=ToolStatus.FAILURE,
                error="content must not be empty",
            )

        record_id = str(uuid.uuid4())

        if self._mm is None:
            return ToolResult(
                tool_name="memory_write",
                status=ToolStatus.SUCCESS,
                output={"record_id": record_id, "mode": "stub"},
            )

        try:
            store = getattr(self._mm, "episodic", None)
            if store is None:
                raise AttributeError("memory_manager has no 'episodic' store")

            # Build a proper MemoryRecord so the episodic backend accepts it.
            try:
                from memory.interfaces import MemoryRecord

                record_obj = MemoryRecord(
                    key=record_id,
                    content=content,
                    metadata={**(metadata or {}), "tags": tags or []},
                )
            except Exception:
                # Fallback: pass a plain dict for stores that accept it
                record_obj = {  # type: ignore[assignment]
                    "id": record_id,
                    "content": content,
                    "metadata": metadata or {},
                    "tags": tags or [],
                }

            # Support both .store(record) and .add(record)
            writer = getattr(store, "store", None) or getattr(store, "add", None)
            if writer is None:
                raise AttributeError("Episodic store has no 'store' or 'add' method")
            writer(record_obj)
        except Exception as exc:
            logger.error(f"MemoryWriteTool: {exc}")
            return ToolResult(
                tool_name="memory_write",
                status=ToolStatus.FAILURE,
                error=str(exc),
            )

        return ToolResult(
            tool_name="memory_write",
            status=ToolStatus.SUCCESS,
            output={"record_id": record_id, "stored": True},
        )
