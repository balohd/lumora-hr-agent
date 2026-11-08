"""MCP client used by the agent orchestrator.

Transport is chosen by environment:
  - MCP_SERVER_URL set  -> Streamable HTTP client to a separately deployed MCP server
  - otherwise           -> stdio: spawns `python -m mcp_server.server` as a child process

The client discovers tools with `tools/list` at connect time and every tool invocation goes
through `tools/call`. Failures are converted into structured error results so the agent can
degrade gracefully (e.g. the MCP server is down or a tool is disabled).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("agent.mcp_client")
ROOT = Path(__file__).resolve().parents[1]


class MCPToolError(RuntimeError):
    pass


class MCPClientManager:
    def __init__(self, server_url: str | None = None, disabled_tools: set[str] | None = None,
                 call_timeout: float = float(os.environ.get("MCP_CALL_TIMEOUT", "30"))):
        self.server_url = server_url if server_url is not None else os.environ.get("MCP_SERVER_URL") or None
        self.disabled_tools = disabled_tools if disabled_tools is not None else {
            t.strip() for t in os.environ.get("MCP_DISABLED_TOOLS", "").split(",") if t.strip()}
        self.call_timeout = call_timeout
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.tools: dict[str, Any] = {}
        self.connected = False
        self.last_error: str | None = None
        self.server_info: dict = {}
        self._lock = asyncio.Lock()

    @property
    def transport(self) -> str:
        return "streamable-http" if self.server_url else "stdio"

    async def connect(self) -> None:
        async with self._lock:
            await self._close_unlocked()
            stack = AsyncExitStack()
            try:
                if self.server_url:
                    from mcp.client.streamable_http import streamable_http_client

                    read, write, _ = await stack.enter_async_context(streamable_http_client(self.server_url))
                else:
                    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                                   cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
                    read, write = await stack.enter_async_context(stdio_client(params, errlog=sys.stderr))
                session = await stack.enter_async_context(ClientSession(read, write))
                init = await asyncio.wait_for(session.initialize(), timeout=120)
                listed = await session.list_tools()
                self._stack, self.session = stack, session
                self.tools = {t.name: t for t in listed.tools}
                self.server_info = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                    "protocol_version": init.protocolVersion}
                self.connected, self.last_error = True, None
                log.info("MCP connected via %s; tools: %s", self.transport, sorted(self.tools))
            except BaseException as exc:  # noqa: BLE001 - anyio may raise ExceptionGroup
                self.connected, self.last_error = False, f"{type(exc).__name__}: {exc}"
                log.error("MCP connection failed: %s", self.last_error)
                try:
                    await stack.aclose()
                except BaseException:  # noqa: BLE001
                    pass

    async def _close_unlocked(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException as exc:  # noqa: BLE001
                log.warning("error closing MCP session: %s", exc)
        self._stack, self.session, self.connected = None, None, False

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    def available_tools(self) -> list[Any]:
        return [t for n, t in self.tools.items() if n not in self.disabled_tools]

    def tool_schemas(self) -> list[dict]:
        return [{"name": t.name, "description": (t.description or "").strip(), "input_schema": t.inputSchema}
                for t in self.available_tools()]

    async def ping(self) -> bool:
        if not self.session:
            return False
        try:
            await asyncio.wait_for(self.session.send_ping(), timeout=5)
            return True
        except BaseException:  # noqa: BLE001
            return False

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return its JSON result as a dict (or a structured error)."""
        if name in self.disabled_tools:
            return {"error": "tool_unavailable", "message": f"Tool '{name}' is disabled in this deployment."}
        if not self.connected or not self.session:
            await self.connect()
            if not self.connected:
                return {"error": "mcp_unavailable", "message": f"MCP server unavailable: {self.last_error}"}
        if name not in self.tools:
            return {"error": "unknown_tool", "message": f"Tool '{name}' not exposed by the MCP server."}
        try:
            res = await asyncio.wait_for(self.session.call_tool(name, arguments), timeout=self.call_timeout)
        except asyncio.TimeoutError:
            return {"error": "tool_timeout", "message": f"Tool '{name}' timed out after {self.call_timeout}s."}
        except BaseException as exc:  # noqa: BLE001 - broken pipe, closed stream...
            self.connected, self.last_error = False, f"{type(exc).__name__}: {exc}"
            return {"error": "mcp_call_failed", "message": self.last_error}
        if res.structuredContent is not None:
            payload = res.structuredContent
            payload = payload.get("result", payload) if isinstance(payload, dict) and set(payload) == {"result"} else payload
        else:
            text = "".join(getattr(c, "text", "") for c in res.content)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"text": text}
        if res.isError:
            return {"error": "tool_error", "message": str(payload)[:500]}
        return payload if isinstance(payload, dict) else {"result": payload}

    def status(self) -> dict:
        return {"connected": self.connected, "transport": self.transport, "server_url": self.server_url,
                "server": self.server_info, "tool_count": len(self.available_tools()),
                "tools": sorted(t.name for t in self.available_tools()),
                "disabled_tools": sorted(self.disabled_tools), "last_error": self.last_error,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
