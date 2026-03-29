"""
MCP Client — connects to Model Context Protocol servers and converts
their tools into LangChain-compatible StructuredTools.

Supports both **stdio** (local process) and **SSE** (remote HTTP) transports.

Usage:
    # In config or .env, define MCP servers:
    MCP_SERVERS = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        },
        "remote-api": {
            "url": "http://localhost:8080/sse"
        }
    }

    tools = load_mcp_tools(MCP_SERVERS)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from langchain_core.tools import StructuredTool

from .config import MCP_SERVERS

# ── Persistent event loop for all MCP I/O ─────────────────────────────
# One loop shared across all tool invocations avoids the "new loop per call"
# resource leak while keeping the sync/async bridge clean.
_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_loop_lock = threading.Lock()


def _get_mcp_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop for MCP operations, creating it if needed."""
    global _mcp_loop
    with _mcp_loop_lock:
        if _mcp_loop is None or _mcp_loop.is_closed():
            _mcp_loop = asyncio.new_event_loop()
            t = threading.Thread(target=_mcp_loop.run_forever, daemon=True)
            t.start()
        return _mcp_loop


def _run_async(coro) -> Any:
    """Submit a coroutine to the shared MCP loop and block until it completes."""
    loop = _get_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


async def _call_stdio_tool(server_config: dict, tool_name: str, kwargs: dict) -> str:
    """Open a fresh stdio session, call one tool, close the session."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        return f"[MCP] 'mcp' package not installed"

    params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        env=server_config.get("env"),
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=kwargs)
                parts = []
                for item in result.content:
                    parts.append(item.text if hasattr(item, "text") else str(item))
                return "\n".join(parts)
    except Exception as exc:
        return f"[MCP stdio error calling {tool_name}] {exc}"


async def _call_sse_tool(server_config: dict, tool_name: str, kwargs: dict) -> str:
    """Open a fresh SSE session, call one tool, close the session."""
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        return f"[MCP] 'mcp' package not installed"

    url = server_config["url"]
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=kwargs)
                parts = []
                for item in result.content:
                    parts.append(item.text if hasattr(item, "text") else str(item))
                return "\n".join(parts)
    except Exception as exc:
        return f"[MCP SSE error calling {tool_name}] {exc}"


def _wrap_mcp_tool(server_name: str, server_config: dict, mcp_tool) -> StructuredTool:
    """Convert an MCP tool descriptor into a LangChain StructuredTool.

    Each call to the returned tool opens a fresh MCP session so that the
    tool remains callable after the discovery session has closed.  All async
    I/O runs on the shared _mcp_loop thread — no per-call event loop creation.
    """
    tool_name = mcp_tool.name
    description = mcp_tool.description or tool_name
    _config = server_config
    _name = tool_name

    is_sse = "url" in server_config

    def call_mcp_tool(**kwargs: Any) -> str:
        try:
            if is_sse:
                return _run_async(_call_sse_tool(_config, _name, kwargs))
            else:
                return _run_async(_call_stdio_tool(_config, _name, kwargs))
        except Exception as exc:
            return f"[MCP tool {_name} error] {exc}"

    call_mcp_tool.__name__ = tool_name
    call_mcp_tool.__doc__ = description

    return StructuredTool.from_function(
        func=call_mcp_tool,
        name=tool_name,
        description=description,
    )


async def _discover_tools(name: str, config: dict) -> list[StructuredTool]:
    """Connect to a server once to list available tools, then wrap each one."""
    is_sse = "url" in config

    try:
        if is_sse:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            cm = sse_client(config["url"])
        else:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )
            cm = stdio_client(params)
    except ImportError:
        print(f"  [MCP] 'mcp' package not installed — skipping server '{name}'")
        return []

    tools: list[StructuredTool] = []
    try:
        async with cm as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for mcp_tool in result.tools:
                    tool = _wrap_mcp_tool(name, config, mcp_tool)
                    tools.append(tool)
                    print(f"  [MCP] {name}/{mcp_tool.name} loaded")
    except Exception as exc:
        print(f"  [MCP] Failed to connect to '{name}': {exc}")

    return tools


async def _load_all_mcp_tools(servers: dict) -> list[StructuredTool]:
    """Load tool descriptors from all configured MCP servers."""
    all_tools: list[StructuredTool] = []
    for name, config in servers.items():
        if "url" not in config and "command" not in config:
            print(f"  [MCP] Unknown server config for '{name}' — skipping")
            continue
        tools = await _discover_tools(name, config)
        all_tools.extend(tools)
    return all_tools


def load_mcp_tools(servers: dict | None = None) -> list[StructuredTool]:
    """Synchronous entry point: load tools from configured MCP servers.

    Returns an empty list if no servers are configured or if the mcp
    package is not installed.  Uses the shared MCP event loop so that
    tool discovery and tool execution share the same threading model.
    """
    servers = servers or MCP_SERVERS
    if not servers:
        return []

    print("\n[MCP] Loading tools from configured servers ...")
    try:
        tools = _run_async(_load_all_mcp_tools(servers))
    except Exception as exc:
        print(f"[MCP] Error loading tools: {exc}")
        tools = []

    print(f"[MCP] {len(tools)} tool(s) loaded from {len(servers)} server(s)")
    return tools
