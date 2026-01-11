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
    # Test Management Tools
    types.Tool(
        name="list_tests",
        description="List all verification tests. Filter by type (playwright_cdp, qontinui_vision, python_script, repository_test) or category.",
        inputSchema={
            "type": "object",
            "properties": {
                "enabled_only": {
                    "type": "boolean",
                    "description": "Only return enabled tests (default: false)",
                    "default": False,
                },
                "test_type": {
                    "type": "string",
                    "description": "Filter by test type",
                    "enum": [
                        "playwright_cdp",
                        "qontinui_vision",
                        "python_script",
                        "repository_test",
                    ],
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category (visual, dom, network, data, log, layout, unit, integration, custom)",
                },
            },
        },
    ),
    types.Tool(
        name="get_test",
        description="Get a specific verification test by ID with full details including code and configuration.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "The test ID to retrieve",
                },
            },
            "required": ["test_id"],
        },
    ),
    types.Tool(
        name="execute_test",
        description="Execute a verification test by ID. Returns execution result including pass/fail status, output, assertions, and screenshots.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "The test ID to execute",
                },
                "task_run_id": {
                    "type": "string",
                    "description": "Optional task run ID to link results to",
                },
            },
            "required": ["test_id"],
        },
    ),
    types.Tool(
        name="list_test_results",
        description="List test results with optional filtering by test, task run, or status.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "Filter results by test ID",
                },
                "task_run_id": {
                    "type": "string",
                    "description": "Filter results by task run ID",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status",
                    "enum": ["passed", "failed", "error", "timeout", "skipped"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 100)",
                    "default": 100,
                },
            },
        },
    ),
    types.Tool(
        name="get_test_history",
        description="Get test history summary with aggregated statistics including pass rate, total runs, and recent results.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "Optional test ID to filter history for a specific test",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to aggregate (default: 1000)",
                    "default": 1000,
                },
            },
        },
    ),
    types.Tool(
        name="create_test",
        description="Create a new verification test. Use this to define automated tests that verify application behavior. Supports Playwright CDP (browser DOM assertions), Python scripts (custom verification logic), and repository tests (pytest, Jest, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable test name",
                },
                "test_type": {
                    "type": "string",
                    "description": "Type of test to create",
                    "enum": [
                        "playwright_cdp",
                        "qontinui_vision",
                        "python_script",
                        "repository_test",
                    ],
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the test verifies",
                },
                "category": {
                    "type": "string",
                    "description": "Test category for organization",
                    "enum": [
                        "visual",
                        "dom",
                        "network",
                        "data",
                        "log",
                        "layout",
                        "unit",
                        "integration",
                        "custom",
                    ],
                },
                "playwright_code": {
                    "type": "string",
                    "description": "TypeScript/JavaScript code for playwright_cdp tests. Should use Playwright expect assertions.",
                },
                "python_code": {
                    "type": "string",
                    "description": "Python code for python_script tests. Should return JSON with status, assertions, output fields.",
                },
                "repo_test_command": {
                    "type": "string",
                    "description": "Command to run for repository_test (e.g., 'pytest tests/test_api.py -v')",
                },
                "repo_test_working_directory": {
                    "type": "string",
                    "description": "Working directory for repository tests (default: project root)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Test timeout in seconds (default: 60)",
                    "default": 60,
                },
                "is_critical": {
                    "type": "boolean",
                    "description": "If true, test failure fails the entire task (default: true)",
                    "default": True,
                },
                "success_criteria": {
                    "type": "string",
                    "description": "Natural language description of what success looks like (for documentation)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for organization and filtering",
                },
            },
            "required": ["name", "test_type"],
        },
    ),
    types.Tool(
        name="update_test",
        description="Update an existing verification test by ID. Only provided fields will be updated.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "ID of the test to update",
                },
                "name": {
                    "type": "string",
                    "description": "New test name",
                },
                "description": {
                    "type": "string",
                    "description": "New description",
                },
                "playwright_code": {
                    "type": "string",
                    "description": "New Playwright code (for playwright_cdp tests)",
                },
                "python_code": {
                    "type": "string",
                    "description": "New Python code (for python_script tests)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "New timeout in seconds",
                },
                "is_critical": {
                    "type": "boolean",
                    "description": "Whether test failure fails the task",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Whether test is enabled",
                },
            },
            "required": ["test_id"],
        },
    ),
    types.Tool(
        name="delete_test",
        description="Delete a verification test by ID. This is permanent and cannot be undone.",
        inputSchema={
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "ID of the test to delete",
                },
            },
            "required": ["test_id"],
        },
    ),
    # DOM Capture Tools
    types.Tool(
        name="list_dom_captures",
        description="List DOM captures (HTML snapshots) from browser pages. Use this to find captured page HTML for debugging UI/styling issues.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "Filter by task run ID to get captures from a specific task",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by capture source",
                    "enum": ["playwright", "extension"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of captures to return (default: 50)",
                    "default": 50,
                },
            },
        },
    ),
    types.Tool(
        name="get_dom_capture",
        description="Get metadata for a specific DOM capture by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "capture_id": {
                    "type": "string",
                    "description": "The DOM capture ID",
                },
            },
            "required": ["capture_id"],
        },
    ),
    types.Tool(
        name="get_dom_capture_html",
        description="Get the full HTML content of a DOM capture. Use this to analyze page structure for debugging UI/styling issues.",
        inputSchema={
            "type": "object",
            "properties": {
                "capture_id": {
                    "type": "string",
                    "description": "The DOM capture ID",
                },
            },
            "required": ["capture_id"],
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

        # Test Management Tools
        elif name == "list_tests":
            enabled_only = arguments.get("enabled_only", False)
            test_type = arguments.get("test_type")
            category = arguments.get("category")
            response = await qontinui.list_tests(
                enabled_only=enabled_only,
                test_type=test_type,
                category=category,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_test":
            test_id = arguments.get("test_id", "")
            if not test_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "test_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.get_test(test_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "execute_test":
            test_id = arguments.get("test_id", "")
            if not test_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "test_id is required"},
                            indent=2,
                        ),
                    )
                ]
            task_run_id = arguments.get("task_run_id")
            response = await qontinui.execute_test(test_id, task_run_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "list_test_results":
            test_id = arguments.get("test_id")
            task_run_id = arguments.get("task_run_id")
            status = arguments.get("status")
            limit = arguments.get("limit", 100)
            response = await qontinui.list_test_results(
                test_id=test_id,
                task_run_id=task_run_id,
                status=status,
                limit=limit,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_test_history":
            test_id = arguments.get("test_id")
            limit = arguments.get("limit", 1000)
            response = await qontinui.get_test_history(test_id=test_id, limit=limit)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "create_test":
            test_name = arguments.get("name", "")
            test_type = arguments.get("test_type", "")
            if not test_name or not test_type:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "name and test_type are required",
                            },
                            indent=2,
                        ),
                    )
                ]
            # Build repo_test_config if applicable
            repo_test_config = None
            if test_type == "repository_test":
                command = arguments.get("repo_test_command")
                if command:
                    repo_test_config = {
                        "command": command,
                        "working_directory": arguments.get(
                            "repo_test_working_directory", "${PROJECT_ROOT}"
                        ),
                        "parse_format": "generic",
                    }
            response = await qontinui.create_test(
                name=test_name,
                test_type=test_type,
                description=arguments.get("description"),
                category=arguments.get("category"),
                playwright_code=arguments.get("playwright_code"),
                python_code=arguments.get("python_code"),
                repo_test_config=repo_test_config,
                timeout_seconds=arguments.get("timeout_seconds", 60),
                is_critical=arguments.get("is_critical", True),
                success_criteria=arguments.get("success_criteria"),
                tags=arguments.get("tags"),
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "update_test":
            test_id = arguments.get("test_id", "")
            if not test_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "test_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.update_test(
                test_id=test_id,
                name=arguments.get("name"),
                description=arguments.get("description"),
                playwright_code=arguments.get("playwright_code"),
                python_code=arguments.get("python_code"),
                timeout_seconds=arguments.get("timeout_seconds"),
                is_critical=arguments.get("is_critical"),
                enabled=arguments.get("enabled"),
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "delete_test":
            test_id = arguments.get("test_id", "")
            if not test_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "test_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.delete_test(test_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        # DOM Capture Tools
        elif name == "list_dom_captures":
            task_run_id = arguments.get("task_run_id")
            source = arguments.get("source")
            limit = arguments.get("limit", 50)
            response = await qontinui.list_dom_captures(
                task_run_id=task_run_id,
                source=source,
                limit=limit,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_dom_capture":
            capture_id = arguments.get("capture_id", "")
            if not capture_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "capture_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.get_dom_capture(capture_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_dom_capture_html":
            capture_id = arguments.get("capture_id", "")
            if not capture_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "capture_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.get_dom_capture_html(capture_id)
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
