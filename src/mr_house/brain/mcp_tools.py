"""MCP (Model Context Protocol) tool integration.

Launches the MCP servers listed in ``config.yaml`` over stdio, discovers their
tools, and exposes them to the LLM in the Ollama/OpenAI ``tools`` schema. When
the model emits a tool call we route it back to the owning server and return the
textual result.

The MCP SDK is async; to keep the rest of the app simple and synchronous we run
a dedicated asyncio loop in a background thread and marshal calls onto it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _HAVE_MCP = True
except Exception as exc:  # pragma: no cover
    _HAVE_MCP = False
    log.warning("mcp SDK unavailable (%s); MCP tools disabled.", exc)


@dataclass
class _Tool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified(self) -> str:
        # Namespaced so two servers can share a tool name.
        return f"{self.server}__{self.name}"


class MCPToolManager:
    """Owns MCP server connections and exposes tools to the brain."""

    def __init__(self, servers_cfg: dict[str, Any]) -> None:
        self._servers_cfg = servers_cfg or {}
        self._tools: dict[str, _Tool] = {}
        self._sessions: dict[str, "ClientSession"] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._exit_stack = None

    # -- lifecycle ---------------------------------------------------------- #
    @property
    def available(self) -> bool:
        return _HAVE_MCP and bool(self._tools)

    def start(self) -> None:
        if not _HAVE_MCP or not self._servers_cfg:
            log.info("MCP disabled or no servers configured.")
            return
        self._thread = threading.Thread(target=self._run_loop, name="MCP", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._tools:
            log.info("MCP tools available: %s", sorted(self._tools))
        else:
            log.warning("No MCP tools were discovered.")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_all())
            self._ready.set()
            self._loop.run_forever()
        except Exception as exc:
            log.error("MCP loop crashed: %s", exc)
            self._ready.set()

    async def _connect_all(self) -> None:
        from contextlib import AsyncExitStack

        self._exit_stack = AsyncExitStack()
        for name, spec in self._servers_cfg.items():
            try:
                await self._connect_one(name, spec)
            except Exception as exc:
                log.error("Failed to connect MCP server '%s': %s", name, exc)

    async def _connect_one(self, name: str, spec: dict[str, Any]) -> None:
        params = StdioServerParameters(
            command=spec["command"],
            args=spec.get("args", []),
            env=spec.get("env") or None,
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session

        listed = await session.list_tools()
        for t in listed.tools:
            tool = _Tool(
                server=name,
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            self._tools[tool.qualified] = tool
        log.info("Connected MCP server '%s' (%d tools).", name, len(listed.tools))

    # -- schema for the LLM ------------------------------------------------- #
    def openai_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in the OpenAI/Ollama function-calling format."""
        out = []
        for tool in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.qualified,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return out

    # -- invocation --------------------------------------------------------- #
    def call(self, qualified_name: str, arguments: dict[str, Any], timeout: float = 30.0) -> str:
        """Synchronously invoke a tool; returns a text result for the model."""
        tool = self._tools.get(qualified_name)
        if tool is None or self._loop is None:
            return f"Error: unknown tool '{qualified_name}'."
        fut = asyncio.run_coroutine_threadsafe(
            self._call_async(tool, arguments), self._loop
        )
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            log.error("Tool '%s' failed: %s", qualified_name, exc)
            return f"Error calling {qualified_name}: {exc}"

    async def _call_async(self, tool: _Tool, arguments: dict[str, Any]) -> str:
        session = self._sessions[tool.server]
        result = await session.call_tool(tool.name, arguments)
        parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(tool returned no content)"

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            async def _cleanup():
                if self._exit_stack is not None:
                    await self._exit_stack.aclose()

            try:
                fut = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)

