"""Lightweight MCP Server for Qontinui Runner.

This server provides a minimal MCP interface that forwards all requests
to the Qontinui Runner via HTTP. It's designed to be fast to start and
have minimal dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .client import QontinuiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MCP Server instance
server = Server("qontinui-mcp")
client: QontinuiClient | None = None


def get_client() -> QontinuiClient:
    """Get or create the Qontinui client."""
    global client
    if client is None:
        client = QontinuiClient()
    return client


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------

TOOLS = [
    types.Tool(
        name="get_executor_status",
        description="Get the current status of the Qontinui Runner.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="list_monitors",
        description="List available monitors with position information (left, middle, right, primary).",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="load_config",
        description="Load a JSON workflow configuration file into the runner. Use this before running workflows.",
        inputSchema={
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Absolute path to the JSON configuration file",
                },
            },
            "required": ["config_path"],
        },
    ),
    types.Tool(
        name="ensure_config_loaded",
        description="Ensure a specific config file is loaded. Loads it if not already loaded, skips if already loaded.",
        inputSchema={
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Absolute path to the JSON configuration file",
                },
            },
            "required": ["config_path"],
        },
    ),
    types.Tool(
        name="get_loaded_config",
        description="Get information about the currently loaded configuration, including available workflows.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="run_workflow",
        description="Run a workflow by name from the currently loaded configuration.",
        inputSchema={
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to run",
                },
                "monitor": {
                    "type": "string",
                    "description": "Monitor to run on: 'left', 'right', 'middle', 'primary', or monitor index (0, 1, 2)",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Maximum execution time in seconds (default: 300)",
                    "default": 300,
                },
            },
            "required": ["workflow_name"],
        },
    ),
    types.Tool(
        name="stop_execution",
        description="Stop the currently running workflow execution.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls by forwarding to the runner."""
    qontinui = get_client()

    try:
        if name == "get_executor_status":
            response = await qontinui.status()
            return [types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))]

        elif name == "list_monitors":
            response = await qontinui.list_monitors()
            return [types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))]

        elif name == "load_config":
            config_path = arguments.get("config_path", "")
            response = await qontinui.load_config(config_path)
            return [types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))]

        elif name == "ensure_config_loaded":
            config_path = arguments.get("config_path", "")
            if qontinui.is_config_loaded(config_path):
                return [types.TextContent(type="text", text=json.dumps({
                    "success": True,
                    "data": {"already_loaded": True, "config_path": config_path},
                }, indent=2))]
            response = await qontinui.load_config(config_path)
            return [types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))]

        elif name == "get_loaded_config":
            info = qontinui.get_loaded_config_info()
            return [types.TextContent(type="text", text=json.dumps({
                "success": True,
                "data": info,
            }, indent=2))]

        elif name == "run_workflow":
            workflow_name = arguments.get("workflow_name", "")
            monitor = arguments.get("monitor")
            timeout = arguments.get("timeout_seconds", 300)

            result = await qontinui.run_workflow(workflow_name, monitor=monitor, timeout=timeout)
            return [types.TextContent(type="text", text=json.dumps({
                "success": result.success,
                "execution_id": result.execution_id,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "events": result.events,
            }, indent=2))]

        elif name == "stop_execution":
            response = await qontinui.stop_execution()
            return [types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))]

        else:
            return [types.TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Unknown tool: {name}",
            }, indent=2))]

    except Exception as e:
        logger.exception(f"Error calling tool {name}")
        return [types.TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e),
        }, indent=2))]


async def main() -> None:
    """Run the MCP server."""
    logger.info("Starting Qontinui MCP Server (lightweight)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run() -> None:
    """Entry point for the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
