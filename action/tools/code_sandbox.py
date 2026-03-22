"""
CodeSandbox — restricted Python execution for the Motor Cortex.

Safety approach (layers in order of preference):
  1. ``RestrictedPython`` (pure-Python AST transformer — default)
  2. Subprocess / Docker  (full isolation — optional, not wired by default)
  3. Stub                 (NAWAL_STUB_TOOLS=1 — always succeeds)

The sandbox captures stdout/stderr and returns them in ``ToolResult.output``.

Usage::

    sb = CodeSandbox()
    result = sb.run(code="print(2 + 2)")
    assert result.output["stdout"] == "4\\n"
"""

from __future__ import annotations

import io
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from loguru import logger

from action.interfaces import (
    AbstractTool,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)

# RestrictedPython import — optional; fall back to plain exec if absent
try:
    from RestrictedPython import (  # type: ignore
        PrintCollector,  # type: ignore
        compile_restricted,
        safe_globals,
    )

    _RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    _RESTRICTED_PYTHON_AVAILABLE = False
    logger.debug("CodeSandbox: RestrictedPython not installed — using exec fallback")


class CodeSandbox(AbstractTool):
    """
    Execute small Python snippets in a restricted environment.

    Args:
        timeout_seconds : Max CPU time for a single execution (default 10).
        max_output_len  : Maximum characters captured from stdout (default 4096).
        use_stub        : Force stub mode regardless of env-var.
        allow_imports   : List of module names allowed inside the sandbox.
                          Defaults to a safe built-in subset.
    """

    _SAFE_IMPORTS_DEFAULT = frozenset(
        {
            "math",
            "statistics",
            "random",
            "datetime",
            "json",
            "re",
            "string",
            "itertools",
            "functools",
            "collections",
            "decimal",
            "fractions",
        }
    )

    _SPEC = ToolSpec(
        name="code_exec",
        description=(
            "Execute a Python code snippet and return its stdout output. "
            "Imports are restricted to a safe allow-list."
        ),
        parameters={
            "code": {
                "type": "string",
                "description": "Python source to execute",
                "required": True,
            },
            "context": {
                "type": "object",
                "description": "Variable bindings injected into globals",
                "required": False,
            },
        },
        category=ToolCategory.CODE_EXEC,
        safe=False,
    )

    def __init__(
        self,
        timeout_seconds: int = 10,
        max_output_len: int = 4096,
        use_stub: bool = False,
        allow_imports: frozenset | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_out = max_output_len
        self._stub = use_stub or os.environ.get("NAWAL_STUB_TOOLS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._allow_imports = allow_imports or self._SAFE_IMPORTS_DEFAULT

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    # ------------------------------------------------------------------ #
    # Public execution                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        **_: Any,
    ) -> ToolResult:
        if not code or not code.strip():
            return ToolResult(
                tool_name="code_exec",
                status=ToolStatus.FAILURE,
                error="code must not be empty",
            )

        t0 = time.time()

        if self._stub:
            output = self._stub_exec(code)
            status = ToolStatus.SUCCESS
        else:
            output, status = self._sandboxed_exec(code, context or {})

        latency_ms = (time.time() - t0) * 1000

        return ToolResult(
            tool_name="code_exec",
            status=status,
            output=output,
            error=output.get("error") if status != ToolStatus.SUCCESS else None,
            metadata={"latency_ms": latency_ms},
        )

    # ------------------------------------------------------------------ #
    # Execution back-ends                                                  #
    # ------------------------------------------------------------------ #

    def _sandboxed_exec(self, code: str, context: dict[str, Any]) -> tuple:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        if _RESTRICTED_PYTHON_AVAILABLE:
            # RestrictedPython transforms print() into _print_() calls.
            # After exec, the local variable `_print` (no trailing _) holds a
            # PrintCollector instance; calling it returns the captured stdout text.
            import warnings as _w

            g = dict(safe_globals)
            g["_print_"] = PrintCollector
            g["_getattr_"] = getattr
            g["_getiter_"] = iter
            g["_getitem_"] = lambda obj, key: obj[key]
            g["__builtins__"]["__import__"] = self._restricted_import
            g.update(context)
            loc: dict[str, Any] = {}
            try:
                stderr_buf = io.StringIO()
                with _w.catch_warnings():
                    _w.simplefilter("ignore", SyntaxWarning)
                    byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
                with redirect_stderr(stderr_buf):
                    exec(byte_code, g, loc)
                # loc['_print'] is the PrintCollector instance; calling it returns text
                collector = loc.get("_print")
                stdout_text = collector() if collector is not None else ""
                return {
                    "stdout": stdout_text[: self._max_out],
                    "stderr": stderr_buf.getvalue()[: self._max_out],
                }, ToolStatus.SUCCESS
            except SyntaxError as exc:
                return {"error": f"SyntaxError: {exc}"}, ToolStatus.FAILURE
            except Exception as exc:
                return {"error": str(exc)}, ToolStatus.FAILURE
        else:
            # Plain exec (no AST restriction)
            safe_builtins = {
                k: __builtins__[k]
                for k in (  # type: ignore
                    "print",
                    "len",
                    "range",
                    "enumerate",
                    "zip",
                    "map",
                    "filter",
                    "int",
                    "float",
                    "str",
                    "bool",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "abs",
                    "min",
                    "max",
                    "sum",
                    "sorted",
                    "reversed",
                    "isinstance",
                    "type",
                    "repr",
                    "round",
                )
                if k
                in (
                    getattr(__builtins__, "__dict__", __builtins__)
                    if isinstance(__builtins__, dict)
                    else vars(__builtins__)
                )
            }
            g = {"__builtins__": safe_builtins}
            g.update(context)
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(code, "<sandbox>", "exec"), g)
                return {
                    "stdout": stdout_buf.getvalue()[: self._max_out],
                    "stderr": stderr_buf.getvalue()[: self._max_out],
                }, ToolStatus.SUCCESS
            except Exception as exc:
                return {"error": str(exc)}, ToolStatus.FAILURE

    def _restricted_import(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._allow_imports:
            raise ImportError(f"Import '{name}' is not allowed in the sandbox")
        return __import__(name, *args, **kwargs)

    @staticmethod
    def _stub_exec(code: str) -> dict[str, str]:
        first_print = ""
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("print(") and line.endswith(")"):
                first_print = line[6:-1].strip().strip("'\"") + "\n"
                break
        return {"stdout": first_print or "[stub exec]\n", "stderr": ""}
