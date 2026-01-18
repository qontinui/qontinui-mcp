"""Lightweight MCP Server for Qontinui Runner.

This server provides a minimal MCP interface that forwards all requests
to the Qontinui Runner via HTTP. It's designed to be fast to start and
have minimal dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import prompts as prompt_builders
from .client import QontinuiClient
from .permissions import (
    check_permission,
    get_tool_permission_level,
    permission_denied_response,
)

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
# Tool Caching with Version Tracking
# -----------------------------------------------------------------------------


@dataclass
class ToolCache:
    """Cache for MCP tools with version tracking."""

    version: str
    tools: list[types.Tool]
    cached_at: float


# Global tool cache
_tool_cache: ToolCache | None = None


async def get_tools_with_cache() -> list[types.Tool]:
    """Get tools with caching based on runner's tool version.

    The cache is invalidated when:
    - The runner's tool version changes (config loaded, tests added/removed)
    - Cache is older than 5 minutes (fallback)
    """
    global _tool_cache

    # Check if cache is valid (not older than 5 minutes)
    if _tool_cache is not None:
        cache_age = time.time() - _tool_cache.cached_at
        if cache_age < 300:  # 5 minutes
            # Try to check version with runner
            try:
                qontinui = get_client()
                version_response = await qontinui.get_tool_version()
                if version_response.success and version_response.data:
                    current_version = version_response.data.get("version", "")
                    if current_version == _tool_cache.version:
                        logger.debug(f"Tool cache hit (version={current_version})")
                        return _tool_cache.tools
                    else:
                        logger.info(
                            f"Tool cache invalidated: version changed from "
                            f"{_tool_cache.version} to {current_version}"
                        )
            except Exception as e:
                logger.warning(f"Failed to check tool version: {e}")
                # If we can't check version, use cached tools if recent
                if cache_age < 60:  # Use cache if less than 1 minute old
                    return _tool_cache.tools

    # Rebuild tool cache
    try:
        qontinui = get_client()
        version_response = await qontinui.get_tool_version()
        current_version = ""
        if version_response.success and version_response.data:
            current_version = version_response.data.get("version", "")
    except Exception:
        current_version = ""

    _tool_cache = ToolCache(
        version=current_version,
        tools=TOOLS,
        cached_at=time.time(),
    )
    logger.info(f"Tool cache rebuilt (version={current_version}, tools={len(TOOLS)})")
    return _tool_cache.tools


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
        name="get_task_run_events",
        description="Get events for a task run from SQLite database. Use this for historical queries on past task runs. For real-time events during execution, use read_runner_logs().",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "The task run ID to get events for.",
                },
                "event_type": {
                    "type": "string",
                    "description": "Filter by event type: 'general', 'action', 'image_recognition', 'ai_output'.",
                    "enum": ["general", "action", "image_recognition", "ai_output"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return.",
                },
            },
            "required": ["task_run_id"],
        },
    ),
    types.Tool(
        name="get_task_run_screenshots",
        description="Get screenshots for a task run from SQLite database. Returns screenshot metadata including file paths, template names, confidence scores, and match locations.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "The task run ID to get screenshots for.",
                },
            },
            "required": ["task_run_id"],
        },
    ),
    types.Tool(
        name="get_task_run_playwright_results",
        description="Get Playwright test results for a task run from SQLite database. Returns test outcomes, durations, errors, and failure screenshots.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "The task run ID to get Playwright results for.",
                },
            },
            "required": ["task_run_id"],
        },
    ),
    types.Tool(
        name="migrate_task_run_logs",
        description="Migrate JSONL logs to SQLite for a task run. This persists the current .dev-logs/*.jsonl files to the database linked to the task run. Call this after task completion to preserve logs.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "The task run ID to migrate logs for.",
                },
            },
            "required": ["task_run_id"],
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
    # AWAS (AI Web Action Standard) Tools
    types.Tool(
        name="awas_discover",
        description="Discover AWAS manifest for a website. AWAS (AI Web Action Standard) enables AI agents to interact with web applications through standardized API definitions. This tool fetches the manifest from /.well-known/ai-actions.json and returns the available actions.",
        inputSchema={
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Base URL of the website (e.g., https://example.com)",
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "If true, bypass cache and fetch fresh manifest (default: false)",
                    "default": False,
                },
            },
            "required": ["base_url"],
        },
    ),
    types.Tool(
        name="awas_check_support",
        description="Check if a website supports AWAS (AI Web Action Standard). Returns a summary of the website's AWAS capabilities without fetching the full manifest details.",
        inputSchema={
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Base URL of the website to check (e.g., https://example.com)",
                },
            },
            "required": ["base_url"],
        },
    ),
    types.Tool(
        name="awas_list_actions",
        description="List available AWAS actions for a website. Returns the action definitions including their parameters, HTTP methods, and descriptions. Use awas_discover first to fetch the manifest.",
        inputSchema={
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Base URL of the website",
                },
                "read_only_only": {
                    "type": "boolean",
                    "description": "If true, only return read-only (safe) actions that don't modify data (default: false)",
                    "default": False,
                },
            },
            "required": ["base_url"],
        },
    ),
    types.Tool(
        name="awas_execute",
        description="Execute an AWAS action on a website. The manifest must be discovered first using awas_discover. This tool makes the HTTP request defined by the action and returns the response.",
        inputSchema={
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Base URL of the website (manifest must be discovered first)",
                },
                "action_id": {
                    "type": "string",
                    "description": "ID of the action to execute (from the manifest)",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters to pass to the action (path, query, body, header params)",
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": "Authentication credentials. For bearer_token: {token: 'xxx'}. For api_key: {api_key: 'xxx'}. For basic: {username: 'xxx', password: 'xxx'}",
                    "additionalProperties": True,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Override default timeout in seconds (default: 30)",
                },
            },
            "required": ["base_url", "action_id"],
        },
    ),
    # Inline Python Execution
    types.Tool(
        name="execute_python",
        description="Execute inline Python code. Returns stdout, stderr, and optional return value. Use this for quick data analysis, custom verification logic, or any Python scripting needs.",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. The code becomes the body of a function - use 'return' to return a JSON-serializable value.",
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pip packages to install (uses uvx for isolation). Example: ['requests', 'pandas']",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 30)",
                    "default": 30,
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for execution (default: temp directory)",
                },
            },
            "required": ["code"],
        },
    ),
    # Agent Spawning
    types.Tool(
        name="spawn_sub_agent",
        description="Spawn a sub-agent with a specific task and optionally scoped tools. The sub-agent runs autonomously and returns when complete. Use this for complex tasks that benefit from hierarchical decomposition.",
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the sub-agent. Be specific about what needs to be accomplished.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool names the sub-agent can use. If not specified, all tools are available.",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum turns/iterations for the sub-agent (default: 10)",
                    "default": 10,
                },
                "context": {
                    "type": "string",
                    "description": "Additional context to provide to the sub-agent.",
                },
            },
            "required": ["task"],
        },
    ),
    # Workflow Generation
    types.Tool(
        name="generate_workflow",
        description="Generate a UnifiedWorkflow from a natural language description using AI. "
        "The AI will create a complete workflow with appropriate setup, verification, agentic, "
        "and completion steps based on the description. The generated workflow can then be "
        "loaded into the Workflow Builder for editing or saved directly.",
        inputSchema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of what the workflow should do. "
                    "Be specific about the task, e.g., 'Run TypeScript type checking and fix errors' "
                    "or 'Build a React app and run Playwright tests, fixing any failures'.",
                },
                "category": {
                    "type": "string",
                    "description": "Category for the workflow (e.g., 'testing', 'development', 'deployment')",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the workflow (e.g., ['typescript', 'react', 'testing'])",
                },
            },
            "required": ["description"],
        },
    ),
]


@server.list_tools()  # type: ignore[untyped-decorator,no-untyped-call]
async def list_tools() -> list[types.Tool]:
    """List all available tools with caching.

    Uses version-based caching to avoid rebuilding tool list on every request.
    Cache is invalidated when runner's config or tests change.
    """
    return await get_tools_with_cache()


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls by forwarding to the runner.

    All tool calls go through permission checks before execution.
    Permission levels:
    - READ_ONLY: Safe operations (logs, status, queries)
    - EXECUTE: Run workflows, tests, capture screenshots
    - MODIFY: Create/update/delete resources
    - DANGEROUS: Stop execution, restart runner
    """
    # Check permissions before executing
    if not await check_permission(name, arguments):
        level = get_tool_permission_level(name)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(permission_denied_response(name, level), indent=2),
            )
        ]

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

        elif name == "get_task_run_events":
            task_run_id = arguments.get("task_run_id", "")
            event_type = arguments.get("event_type")
            limit = arguments.get("limit")
            response = await qontinui.get_task_run_events(
                task_run_id=task_run_id, event_type=event_type, limit=limit
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_task_run_screenshots":
            task_run_id = arguments.get("task_run_id", "")
            response = await qontinui.get_task_run_screenshots(task_run_id=task_run_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_task_run_playwright_results":
            task_run_id = arguments.get("task_run_id", "")
            response = await qontinui.get_task_run_playwright_results(
                task_run_id=task_run_id
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "migrate_task_run_logs":
            task_run_id = arguments.get("task_run_id", "")
            response = await qontinui.migrate_task_run_logs(task_run_id=task_run_id)
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

        # AWAS (AI Web Action Standard) Tools
        elif name == "awas_discover":
            base_url = arguments.get("base_url", "")
            if not base_url:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "base_url is required"},
                            indent=2,
                        ),
                    )
                ]
            force_refresh = arguments.get("force_refresh", False)
            response = await qontinui.awas_discover(base_url, force_refresh)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "awas_check_support":
            base_url = arguments.get("base_url", "")
            if not base_url:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "base_url is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.awas_check_support(base_url)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "awas_list_actions":
            base_url = arguments.get("base_url", "")
            if not base_url:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "base_url is required"},
                            indent=2,
                        ),
                    )
                ]
            read_only_only = arguments.get("read_only_only", False)
            response = await qontinui.awas_list_actions(base_url, read_only_only)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "awas_execute":
            base_url = arguments.get("base_url", "")
            action_id = arguments.get("action_id", "")
            if not base_url or not action_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "base_url and action_id are required",
                            },
                            indent=2,
                        ),
                    )
                ]
            params = arguments.get("params")
            credentials = arguments.get("credentials")
            timeout_seconds = arguments.get("timeout_seconds")
            response = await qontinui.awas_execute(
                base_url=base_url,
                action_id=action_id,
                params=params,
                credentials=credentials,
                timeout_seconds=timeout_seconds,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "execute_python":
            code = arguments.get("code", "")
            if not code:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "code is required"},
                            indent=2,
                        ),
                    )
                ]
            dependencies = arguments.get("dependencies")
            timeout_seconds = arguments.get("timeout_seconds", 30)
            working_directory = arguments.get("working_directory")
            response = await qontinui.execute_python(
                code=code,
                dependencies=dependencies,
                timeout_seconds=timeout_seconds,
                working_directory=working_directory,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "spawn_sub_agent":
            task = arguments.get("task", "")
            if not task:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "task is required"},
                            indent=2,
                        ),
                    )
                ]
            tools = arguments.get("tools")
            max_iterations = arguments.get("max_iterations", 10)
            context = arguments.get("context")
            response = await qontinui.spawn_sub_agent(
                task=task,
                tools=tools,
                max_iterations=max_iterations,
                context=context,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "generate_workflow":
            description = arguments.get("description", "")
            if not description:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "description is required"},
                            indent=2,
                        ),
                    )
                ]
            category = arguments.get("category")
            tags = arguments.get("tags")
            response = await qontinui.generate_workflow(
                description=description,
                category=category,
                tags=tags,
            )
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


# -----------------------------------------------------------------------------
# MCP Resources - Read-only data access via URI scheme
# -----------------------------------------------------------------------------


@server.list_resources()  # type: ignore[untyped-decorator,no-untyped-call]
async def list_resources() -> list[types.Resource]:
    """List available resources.

    Resources provide read-only access to:
    - Configuration: Current workflow config
    - Screenshots: Captured screenshots from automation
    - Logs: JSONL event logs
    - Tests: Verification test definitions
    - DOM Captures: HTML snapshots from browser pages
    """
    qontinui = get_client()
    resources: list[types.Resource] = []

    try:
        # Configuration resource - if config is loaded
        status = await qontinui.status()
        if status.success and status.data and status.data.get("config_loaded"):
            config_path = status.data.get("config_path", "unknown")
            resources.append(
                types.Resource(
                    uri="qontinui://config/current",  # type: ignore[arg-type]
                    name="Current Configuration",
                    description=f"Currently loaded workflow configuration from {config_path}",
                    mimeType="application/json",
                )
            )

        # Log resources - always available
        log_types = [
            ("general", "General executor events"),
            ("actions", "Workflow action/tree events"),
            ("image-recognition", "Image recognition results with match details"),
            ("playwright", "Playwright test execution results"),
        ]
        for log_type, description in log_types:
            resources.append(
                types.Resource(
                    uri=f"qontinui://logs/{log_type}",  # type: ignore[arg-type]
                    name=f"Runner Logs: {log_type}",
                    description=description,
                    mimeType="application/jsonl",
                )
            )

        # Screenshot resources - list available screenshots
        screenshots_response = await qontinui.list_screenshots()
        if screenshots_response.success and screenshots_response.data:
            screenshots = screenshots_response.data.get("screenshots", [])
            for s in screenshots[:20]:  # Limit to first 20
                screenshot_id = s.get("id") or s.get("filename", "unknown")
                timestamp = s.get("timestamp", "")
                resources.append(
                    types.Resource(
                        uri=f"qontinui://screenshots/{screenshot_id}",  # type: ignore[arg-type]
                        name=f"Screenshot: {timestamp or screenshot_id}",
                        description=s.get("description", "Screenshot capture"),
                        mimeType="image/png",
                    )
                )

        # Test resources - list verification tests
        tests_response = await qontinui.list_tests()
        if tests_response.success and tests_response.data:
            tests = tests_response.data.get("tests", [])
            for test in tests[:20]:  # Limit to first 20
                test_id = test.get("id", "unknown")
                test_name = test.get("name", "Unnamed Test")
                test_type = test.get("test_type", "unknown")
                resources.append(
                    types.Resource(
                        uri=f"qontinui://tests/{test_id}",  # type: ignore[arg-type]
                        name=f"Test: {test_name}",
                        description=f"{test_type} - {test.get('description', 'No description')}",
                        mimeType="application/json",
                    )
                )

        # DOM capture resources
        dom_response = await qontinui.list_dom_captures()
        if dom_response.success and dom_response.data:
            captures = dom_response.data.get("captures", [])
            for capture in captures[:10]:  # Limit to first 10
                capture_id = capture.get("id", "unknown")
                url = capture.get("url", "unknown")
                source = capture.get("source", "unknown")
                resources.append(
                    types.Resource(
                        uri=f"qontinui://dom/{capture_id}",  # type: ignore[arg-type]
                        name=f"DOM: {url[:50]}{'...' if len(url) > 50 else ''}",
                        description=f"HTML snapshot from {source}",
                        mimeType="text/html",
                    )
                )

        # Task run resources - list recent task runs
        task_runs_response = await qontinui.get_task_runs()
        if task_runs_response.success and task_runs_response.data:
            task_runs = task_runs_response.data.get("task_runs", [])
            for task_run in task_runs[:10]:  # Limit to first 10
                task_id = task_run.get("id", "unknown")
                task_name = task_run.get("task_name", "Unnamed Task")
                status = task_run.get("status", "unknown")
                resources.append(
                    types.Resource(
                        uri=f"qontinui://task-runs/{task_id}",  # type: ignore[arg-type]
                        name=f"Task Run: {task_name}",
                        description=f"Status: {status}",
                        mimeType="application/json",
                    )
                )

    except Exception:
        logger.exception("Error listing resources")
        # Return empty list on error - resources are optional

    return resources


@server.read_resource()  # type: ignore[untyped-decorator,no-untyped-call]
async def read_resource(uri: str) -> str:
    """Read a resource by URI.

    URI scheme: qontinui://{type}/{id}

    Supported types:
    - config/current: Current workflow configuration
    - logs/{type}: JSONL log files (general, actions, image-recognition, playwright)
    - screenshots/{id}: Screenshot data (base64 encoded)
    - tests/{id}: Test definition JSON
    - dom/{id}: DOM capture HTML content
    - task-runs/{id}: Task run details
    """
    qontinui = get_client()

    # Parse URI - remove scheme and split
    if not uri.startswith("qontinui://"):
        raise ValueError(f"Invalid URI scheme. Expected qontinui://, got: {uri}")

    path = uri.replace("qontinui://", "")
    parts = path.split("/", 1)  # Split into type and id

    if len(parts) < 1:
        raise ValueError(f"Invalid URI format: {uri}")

    resource_type = parts[0]
    resource_id = parts[1] if len(parts) > 1 else None

    try:
        if resource_type == "config":
            if resource_id == "current":
                info = qontinui.get_loaded_config_info()
                if info.get("loaded"):
                    return json.dumps(info, indent=2)
                else:
                    return json.dumps(
                        {"error": "No configuration loaded", "loaded": False}, indent=2
                    )
            else:
                raise ValueError(f"Unknown config resource: {resource_id}")

        elif resource_type == "logs":
            if resource_id not in [
                "general",
                "actions",
                "image-recognition",
                "playwright",
            ]:
                raise ValueError(
                    f"Unknown log type: {resource_id}. "
                    "Valid types: general, actions, image-recognition, playwright"
                )
            response = await qontinui.read_runner_logs(log_type=resource_id, limit=1000)
            if response.success:
                return json.dumps(response.data, indent=2)
            else:
                return json.dumps({"error": response.error}, indent=2)

        elif resource_type == "screenshots":
            if not resource_id:
                raise ValueError("Screenshot ID is required")
            # For screenshots, return the metadata and file path
            # (actual image reading is done via Claude's Read tool)
            response = await qontinui.list_screenshots()
            if response.success and response.data:
                screenshots = response.data.get("screenshots", [])
                for s in screenshots:
                    if s.get("id") == resource_id or s.get("filename") == resource_id:
                        return json.dumps(s, indent=2)
            return json.dumps(
                {"error": f"Screenshot not found: {resource_id}"}, indent=2
            )

        elif resource_type == "tests":
            if not resource_id:
                raise ValueError("Test ID is required")
            response = await qontinui.get_test(resource_id)
            if response.success:
                return json.dumps(response.data, indent=2)
            else:
                return json.dumps({"error": response.error}, indent=2)

        elif resource_type == "dom":
            if not resource_id:
                raise ValueError("DOM capture ID is required")
            response = await qontinui.get_dom_capture_html(resource_id)
            if response.success and response.data:
                # Return the HTML content directly
                html_content: str = response.data.get("html", "")
                return html_content
            else:
                return json.dumps(
                    {"error": response.error or "DOM capture not found"}, indent=2
                )

        elif resource_type == "task-runs":
            if not resource_id:
                raise ValueError("Task run ID is required")
            response = await qontinui.get_task_run(resource_id)
            if response.success:
                return json.dumps(response.data, indent=2)
            else:
                return json.dumps({"error": response.error}, indent=2)

        else:
            raise ValueError(
                f"Unknown resource type: {resource_type}. "
                "Valid types: config, logs, screenshots, tests, dom, task-runs"
            )

    except Exception as e:
        logger.exception(f"Error reading resource: {uri}")
        return json.dumps({"error": str(e)}, indent=2)


# -----------------------------------------------------------------------------
# MCP Prompts - Parameterized templates for common tasks
# -----------------------------------------------------------------------------

PROMPTS = [
    types.Prompt(
        name="debug_test_failure",
        description="Analyze a test failure with structured debugging approach. Aggregates test details, recent results, and optionally screenshots to diagnose issues.",
        arguments=[
            types.PromptArgument(
                name="test_id", description="Test ID to debug", required=True
            ),
            types.PromptArgument(
                name="include_screenshots",
                description="Include screenshot analysis (true/false)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="analyze_screenshot",
        description="Visual analysis of a screenshot for UI verification. Provides context for analyzing captured screenshots from automation.",
        arguments=[
            types.PromptArgument(
                name="screenshot_id",
                description="Screenshot ID or filename",
                required=True,
            ),
            types.PromptArgument(
                name="focus_area",
                description="Region or element to focus analysis on (optional)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="fix_playwright_failure",
        description="Structured workflow to fix a failing Playwright test. Provides debugging workflow for CDP-based browser tests.",
        arguments=[
            types.PromptArgument(
                name="spec_name",
                description="Playwright spec file or test name",
                required=True,
            ),
            types.PromptArgument(
                name="error_message",
                description="Error message from the failure (optional)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="verify_workflow_state",
        description="Verify current GUI state matches expected workflow state. Guides verification that the application is in the correct state.",
        arguments=[
            types.PromptArgument(
                name="state_name",
                description="Expected state name to verify",
                required=True,
            ),
            types.PromptArgument(
                name="workflow_name",
                description="Workflow context (optional)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="create_verification_test",
        description="Generate a verification test for a UI behavior. Guides creation of new tests with appropriate type and structure.",
        arguments=[
            types.PromptArgument(
                name="behavior_description",
                description="What behavior to verify",
                required=True,
            ),
            types.PromptArgument(
                name="test_type",
                description="Preferred test type: playwright_cdp, python_script, or qontinui_vision",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="analyze_automation_run",
        description="Review automation run results and identify issues. Provides comprehensive analysis of workflow execution.",
        arguments=[
            types.PromptArgument(
                name="run_id",
                description="Automation run ID (optional, defaults to most recent)",
                required=False,
            ),
            types.PromptArgument(
                name="focus_on_failures",
                description="Focus analysis on failures (true/false, default: true)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="debug_image_recognition",
        description="Debug template matching and image recognition issues. Helps diagnose why visual pattern matching is failing.",
        arguments=[
            types.PromptArgument(
                name="template_name",
                description="Template name to debug (optional, shows all if not specified)",
                required=False,
            ),
            types.PromptArgument(
                name="last_n_attempts",
                description="Number of recent attempts to analyze (default: 10)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="summarize_task_progress",
        description="Task status summary with execution progress, test results, and recent events.",
        arguments=[
            types.PromptArgument(
                name="task_run_id",
                description="Task run ID (optional, defaults to most recent)",
                required=False,
            ),
        ],
    ),
    # Verification-specific prompts
    types.Prompt(
        name="analyze_verification_failure",
        description="Analyze why a verification criterion failed. Helps diagnose test failures, environment issues, and verification plan problems.",
        arguments=[
            types.PromptArgument(
                name="task_id",
                description="Task run ID with the verification failure",
                required=True,
            ),
            types.PromptArgument(
                name="criterion_id",
                description="Specific criterion ID that failed (optional)",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="create_verification_plan",
        description="Generate a verification plan for a feature. Includes success criteria, test types, and execution strategy.",
        arguments=[
            types.PromptArgument(
                name="feature_description",
                description="Description of the feature to verify",
                required=True,
            ),
            types.PromptArgument(
                name="strategy",
                description="Verification strategy: exhaustive, smoke, or targeted",
                required=False,
            ),
        ],
    ),
]


@server.list_prompts()  # type: ignore[untyped-decorator,no-untyped-call]
async def list_prompts() -> list[types.Prompt]:
    """List available prompts."""
    return PROMPTS


@server.get_prompt()  # type: ignore[untyped-decorator,no-untyped-call]
async def get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Get a prompt by name with the provided arguments.

    Prompts aggregate context from the runner to build structured prompts
    for common automation tasks.
    """
    qontinui = get_client()

    try:
        if name == "debug_test_failure":
            return await prompt_builders.build_debug_test_failure_prompt(
                qontinui, arguments
            )
        elif name == "analyze_screenshot":
            return await prompt_builders.build_analyze_screenshot_prompt(
                qontinui, arguments
            )
        elif name == "fix_playwright_failure":
            return await prompt_builders.build_fix_playwright_failure_prompt(
                qontinui, arguments
            )
        elif name == "verify_workflow_state":
            return await prompt_builders.build_verify_workflow_state_prompt(
                qontinui, arguments
            )
        elif name == "create_verification_test":
            return await prompt_builders.build_create_verification_test_prompt(
                qontinui, arguments
            )
        elif name == "analyze_automation_run":
            return await prompt_builders.build_analyze_automation_run_prompt(
                qontinui, arguments
            )
        elif name == "debug_image_recognition":
            return await prompt_builders.build_debug_image_recognition_prompt(
                qontinui, arguments
            )
        elif name == "summarize_task_progress":
            return await prompt_builders.build_summarize_task_progress_prompt(
                qontinui, arguments
            )
        elif name == "analyze_verification_failure":
            return await prompt_builders.build_analyze_verification_failure_prompt(
                qontinui, arguments
            )
        elif name == "create_verification_plan":
            return await prompt_builders.build_create_verification_plan_prompt(
                qontinui, arguments
            )
        else:
            return types.GetPromptResult(
                description=f"Unknown prompt: {name}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Error: Unknown prompt '{name}'. Available prompts: "
                            + ", ".join(p.name for p in PROMPTS),
                        ),
                    )
                ],
            )

    except Exception as e:
        logger.exception(f"Error building prompt {name}")
        return types.GetPromptResult(
            description=f"Error: {name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text", text=f"Error building prompt: {e}"
                    ),
                )
            ],
        )


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
