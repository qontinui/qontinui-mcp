"""Lightweight MCP Server for Qontinui Runner.

This server provides a minimal MCP interface that forwards all requests
to the Qontinui Runner via HTTP. It's designed to be fast to start and
have minimal dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    types.Tool(
        name="get_task_runs",
        description="Get all task runs from the runner database. Optionally filter by status.",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'running', 'complete', 'failed', 'stopped'. Default: all.",
                    "enum": ["running", "complete", "failed", "stopped"],
                },
            },
        },
    ),
    types.Tool(
        name="get_task_run",
        description="Get a specific task run with full details including execution_steps_json and output_log.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task run ID to retrieve",
                },
            },
            "required": ["task_id"],
        },
    ),
    types.Tool(
        name="list_screenshots",
        description="List available screenshots in the .dev-logs/screenshots directory. Returns file paths that can be read with Claude's Read tool.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="read_runner_logs",
        description="Read runner JSONL log files from .dev-logs directory. Use this to access detailed execution logs including image recognition results, action events, and Playwright test results.",
        inputSchema={
            "type": "object",
            "properties": {
                "log_type": {
                    "type": "string",
                    "description": "Type of logs to read: 'general' (executor events), 'actions' (workflow execution), 'image-recognition' (match results with annotated screenshots), 'playwright' (test results), or 'all' (all types).",
                    "enum": [
                        "general",
                        "actions",
                        "image-recognition",
                        "playwright",
                        "all",
                    ],
                    "default": "all",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries per log type (default: 100). Most recent entries are returned.",
                    "default": 100,
                },
            },
        },
    ),
    types.Tool(
        name="get_automation_runs",
        description="Get recent automation runs from the runner database. These are GUI automation workflow executions with detailed action/state/transition data.",
        inputSchema={
            "type": "object",
            "properties": {
                "config_id": {
                    "type": "string",
                    "description": "Optional config ID to filter runs by.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of runs to return (default: 20).",
                    "default": 20,
                },
            },
        },
    ),
    types.Tool(
        name="get_automation_run",
        description="Get a specific automation run with full details including actions_summary, states_visited, transitions_executed, template_matches, and anomalies.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The automation run ID to retrieve.",
                },
            },
            "required": ["run_id"],
        },
    ),
]


@server.list_tools()  # type: ignore[untyped-decorator,no-untyped-call]
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return TOOLS


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls by forwarding to the runner."""
    qontinui = get_client()

    try:
        if name == "get_executor_status":
            response = await qontinui.status()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "list_monitors":
            response = await qontinui.list_monitors()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "load_config":
            config_path = arguments.get("config_path", "")
            response = await qontinui.load_config(config_path)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "ensure_config_loaded":
            config_path = arguments.get("config_path", "")
            # Use verify_config_loaded to check BOTH local cache AND runner state
            if await qontinui.verify_config_loaded(config_path):
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": True,
                                "data": {
                                    "already_loaded": True,
                                    "config_path": config_path,
                                },
                            },
                            indent=2,
                        ),
                    )
                ]
            # Config not loaded on runner, load it now
            response = await qontinui.load_config(config_path)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_loaded_config":
            info = qontinui.get_loaded_config_info()
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": True,
                            "data": info,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "run_workflow":
            workflow_name = arguments.get("workflow_name", "")
            monitor = arguments.get("monitor")
            timeout = arguments.get("timeout_seconds", 300)

            result = await qontinui.run_workflow(
                workflow_name, monitor=monitor, timeout=timeout
            )
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": result.success,
                            "execution_id": result.execution_id,
                            "duration_ms": result.duration_ms,
                            "error": result.error,
                            "events": result.events,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "stop_execution":
            response = await qontinui.stop_execution()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_task_runs":
            status = arguments.get("status")
            response = await qontinui.get_task_runs(status=status)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_task_run":
            task_id = arguments.get("task_id", "")
            if not task_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "task_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.get_task_run(task_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "list_screenshots":
            response = await qontinui.list_screenshots()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "read_runner_logs":
            log_type = arguments.get("log_type", "all")
            limit = arguments.get("limit", 100)
            response = await qontinui.read_runner_logs(log_type=log_type, limit=limit)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_automation_runs":
            config_id = arguments.get("config_id")
            limit = arguments.get("limit", 20)
            response = await qontinui.get_automation_runs(
                config_id=config_id, limit=limit
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_automation_run":
            run_id = arguments.get("run_id", "")
            response = await qontinui.get_automation_run(run_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        else:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": f"Unknown tool: {name}",
                        },
                        indent=2,
                    ),
                )
            ]

    except Exception as e:
        logger.exception(f"Error calling tool {name}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": str(e),
                    },
                    indent=2,
                ),
            )
        ]


async def main() -> None:
    """Run the MCP server."""
    logger.info("Starting Qontinui MCP Server (lightweight)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def run() -> None:
    """Entry point for the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
