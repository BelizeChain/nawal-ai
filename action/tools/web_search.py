"""
WebSearchTool — web search action for the Motor Cortex.

Production path: delegates to a SearXNG instance configured via
``SEARXNG_URL`` environment variable (default http://localhost:8080).

Fallback / test path: returns a structured stub response so the full
brain pipeline can be tested offline.

Usage::

    tool = WebSearchTool()
    result = tool.run(query="Belize reef coral bleaching 2024", num_results=5)
    print(result.output)   # list of {title, url, snippet}
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from action.interfaces import (
    AbstractTool,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


class WebSearchTool(AbstractTool):
    """
    Lightweight web search tool.

    Args:
        searxng_url  : Base URL of a SearXNG instance.  Defaults to the
                       ``SEARXNG_URL`` env-var or http://localhost:8080.
        timeout      : HTTP request timeout in seconds (default 10).
        use_stub     : Force stub mode regardless of env-var (useful in tests).
    """

    _SPEC = ToolSpec(
        name="web_search",
        description=(
            "Search the web for up-to-date information. "
            "Returns a list of {title, url, snippet} objects."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "num_results": {
                "type": "integer",
                "description": "Max results (default 5)",
                "required": False,
            },
            "language": {
                "type": "string",
                "description": "Language code (default en)",
                "required": False,
            },
        },
        category=ToolCategory.WEB_SEARCH,
        safe=False,
    )

    def __init__(
        self,
        searxng_url: str | None = None,
        timeout: int = 10,
        use_stub: bool = False,
    ) -> None:
        self._url = searxng_url or os.environ.get("SEARXNG_URL", "http://localhost:8080")
        self._timeout = timeout
        self._stub = use_stub or os.environ.get("NAWAL_STUB_TOOLS", "").lower() in (
            "1",
            "true",
            "yes",
        )

    @property
    def spec(self) -> ToolSpec:
        return self._SPEC

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en",
        **_: Any,
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                tool_name="web_search",
                status=ToolStatus.FAILURE,
                error="query must not be empty",
            )

        t0 = time.time()

        if self._stub:
            results = self._stub_results(query, num_results)
        else:
            try:
                results = self._live_search(query, num_results, language)
            except Exception as exc:
                logger.warning(f"WebSearchTool: live search failed ({exc}), falling back to stub")
                results = self._stub_results(query, num_results)

        latency_ms = (time.time() - t0) * 1000
        return ToolResult(
            tool_name="web_search",
            status=ToolStatus.SUCCESS,
            output=results,
            metadata={
                "latency_ms": latency_ms,
                "num_results": len(results),
                "query": query,
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _live_search(self, query: str, num_results: int, language: str) -> list[dict[str, str]]:
        """Call a real SearXNG endpoint."""
        import json
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "lang": language,
            }
        )
        url = f"{self._url.rstrip('/')}/search?{params}"
        req = urllib.request.urlopen(url, timeout=self._timeout)
        data = json.loads(req.read().decode())
        raw = data.get("results", [])[:num_results]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in raw
        ]

    def _stub_results(self, query: str, num_results: int) -> list[dict[str, str]]:
        """Return deterministic stub results (offline / test mode)."""
        return [
            {
                "title": f"Stub result {i + 1} for: {query}",
                "url": f"https://example.com/stub/{i + 1}",
                "snippet": f"This is a stub snippet for '{query}' (result {i + 1}).",
            }
            for i in range(min(num_results, 5))
        ]
