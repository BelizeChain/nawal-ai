"""
Tool Executor — dispatches Plan steps to registered callable tools.

Implements AbstractExecutor. Each "tool" is a plain Python callable
registered by name before the first ``execute()`` call.

Design principles:
  - Tools are sync or async callables; the executor handles both.
  - Execution is step-by-step; any failure is caught, logged, and returned
    as a partial result (unless the step is marked ``required=True``).
  - ``interrupt()`` sets a per-plan flag checked between steps.
  - Results are accumulated in an ``ExecutionResult`` structure and logged
    into memory (if a MemoryManager is provided).

Built-in tools (always available):
  - ``noop``         : does nothing, returns {}
  - ``memory_read``  : reads from MemoryManager (if provided)
  - ``memory_write`` : writes to MemoryManager (if provided)
  - ``log``          : emits a log message

Usage::

    executor = ToolExecutor(memory=mm)
    executor.register("respond", my_respond_fn)
    executor.register("search",  my_search_fn)

    result = executor.execute(plan)
    print(result["status"])    # "success" | "partial" | "failed"
    print(result["outputs"])   # per-step output list
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from control.interfaces import AbstractExecutor, Plan


# --------------------------------------------------------------------------- #
# ToolExecutor                                                                 #
# --------------------------------------------------------------------------- #

class ToolExecutor(AbstractExecutor):
    """
    Step-by-step plan dispatcher.

    Args:
        memory  : Optional MemoryManager for the built-in memory_read/write tools.
        timeout : Per-step timeout in seconds (0 = no timeout).
    """

    def __init__(
        self,
        memory: Optional[Any] = None,   # MemoryManager — avoid circular import
        timeout: float = 30.0,
    ) -> None:
        self._memory = memory
        self.timeout = timeout
        self._tools: Dict[str, Callable] = {}
        self._interrupt_flags: Dict[str, threading.Event] = {}

        # Register built-in tools
        self._register_builtins()

    # ------------------------------------------------------------------ #
    # Tool registration                                                    #
    # ------------------------------------------------------------------ #

    def register(self, name: str, fn: Callable, overwrite: bool = False) -> None:
        """
        Register a callable under *name*.

        Args:
            name      : Tool identifier used in plan step ``{"tool": name}``.
            fn        : Sync or async callable.  Receives step ``args`` as kwargs.
            overwrite : If False and tool already registered, raises ValueError.
        """
        if name in self._tools and not overwrite:
            raise ValueError(
                f"Tool {name!r} is already registered. Pass overwrite=True to replace."
            )
        self._tools[name] = fn
        logger.debug(f"ToolExecutor registered tool={name!r}")

    def available_tools(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------ #
    # AbstractExecutor implementation                                      #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        plan: Plan,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute *plan* step-by-step.

        Returns::

            {
                "status":  "success" | "partial" | "failed",
                "outputs": [ {step_index: int, tool: str, result: Any}, … ],
                "error":   str | None,
            }
        """
        flag = threading.Event()
        self._interrupt_flags[plan.plan_id] = flag

        outputs: List[Dict[str, Any]] = []
        error: Optional[str] = None
        status = "success"

        logger.info(
            f"ToolExecutor executing plan_id={plan.plan_id!r} "
            f"steps={len(plan.steps)} dry_run={dry_run}"
        )

        for idx, step in enumerate(plan.steps):
            # Check interrupt
            if flag.is_set():
                logger.info(f"ToolExecutor interrupted at step {idx}")
                status = "partial"
                break

            tool_name = step.get("tool", "noop")
            args      = step.get("args", {})
            required  = step.get("required", False)

            if dry_run:
                outputs.append({
                    "step": idx,
                    "tool": tool_name,
                    "result": {"dry_run": True, "args": args},
                })
                continue

            if tool_name not in self._tools:
                msg = f"Unknown tool {tool_name!r} at step {idx}"
                logger.warning(msg)
                outputs.append({"step": idx, "tool": tool_name, "result": None, "error": msg})
                if required:
                    status, error = "failed", msg
                    break
                status = "partial"
                continue

            try:
                result = self._invoke(tool_name, args)
                outputs.append({"step": idx, "tool": tool_name, "result": result})
                logger.debug(f"Step {idx} {tool_name!r} ok")
            except Exception as exc:
                msg = f"Step {idx} {tool_name!r} raised: {exc}"
                logger.error(msg)
                outputs.append({"step": idx, "tool": tool_name, "result": None, "error": str(exc)})
                if required:
                    status, error = "failed", msg
                    break
                status = "partial"

        # Clean up interrupt flag
        self._interrupt_flags.pop(plan.plan_id, None)

        logger.info(
            f"ToolExecutor plan_id={plan.plan_id!r} status={status!r} "
            f"steps_run={len(outputs)}"
        )
        return {"status": status, "outputs": outputs, "error": error}

    def interrupt(self, plan_id: str) -> bool:
        """
        Signal a running plan to stop after the current step.

        Returns True if the plan was found (flag set), False otherwise.
        """
        flag = self._interrupt_flags.get(plan_id)
        if flag is not None:
            flag.set()
            logger.info(f"ToolExecutor interrupt requested for plan_id={plan_id!r}")
            return True
        return False

    # ------------------------------------------------------------------ #
    # Invocation                                                           #
    # ------------------------------------------------------------------ #

    def _invoke(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call the tool, handling both sync and async callables."""
        fn = self._tools[tool_name]
        if asyncio.iscoroutinefunction(fn):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, fn(**args)).result(
                        timeout=self.timeout if self.timeout > 0 else None
                    )
            else:
                return asyncio.run(fn(**args))
        else:
            return fn(**args)

    # ------------------------------------------------------------------ #
    # Built-in tools                                                       #
    # ------------------------------------------------------------------ #

    def _register_builtins(self) -> None:
        self._tools["noop"] = lambda **_: {}

        self._tools["log"] = lambda message="", level="info", **_: (
            getattr(logger, level, logger.info)(f"[tool:log] {message}") or {}
        )

        def _memory_read(query: str = "", top_k: int = 5, **_) -> Dict:
            if self._memory is None:
                return {"records": []}
            emb = _text_to_mock_embedding(query)
            records = self._memory.retrieve(emb, top_k=top_k)
            return {"records": [{"key": r.key, "content": r.content} for r in records]}

        def _memory_write(content: str = "", metadata: Optional[Dict] = None, **_) -> Dict:
            if self._memory is None:
                return {"stored": False}
            rec = self._memory.store_text(
                content,
                metadata=metadata or {"source": "executor"},
                store="episodic",
            )
            return {"stored": True, "key": rec.key}

        self._tools["memory_read"]  = _memory_read
        self._tools["memory_write"] = _memory_write

        # Stub tools that Phase 3+ will replace with real implementations
        for stub in ("respond", "reason", "validate", "search", "execute"):
            _name = stub  # capture loop variable

            def _stub(**kwargs: Any) -> Dict[str, Any]:
                logger.debug(f"Stub tool {_name!r} called with {kwargs}")
                return {"stub": _name, "kwargs": kwargs}

            # Re-bind so each lambda captures the correct stub name
            self._tools[stub] = (lambda n: lambda **kw: {"stub": n, "kwargs": kw})(stub)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _text_to_mock_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Deterministic mock embedding from text hash.
    Replaced in Phase 3 with a real encoder call.
    """
    import hashlib
    import struct
    h = hashlib.sha256(text.encode()).digest()
    # Repeat hash bytes to fill dim floats
    repeated = (h * ((dim * 4 // len(h)) + 1))[: dim * 4]
    floats = [struct.unpack_from("f", repeated, i * 4)[0] for i in range(dim)]
    norm = sum(x * x for x in floats) ** 0.5 or 1.0
    return [x / norm for x in floats]
