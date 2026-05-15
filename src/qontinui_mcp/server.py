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
from urllib.parse import quote

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
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum iterations for agentic phase (default: 10). "
                    "This controls how many verification/agentic loops can occur.",
                    "default": 10,
                },
                "provider": {
                    "type": "string",
                    "description": "AI provider override for the workflow. Options: "
                    "'claude_cli', 'anthropic_api', 'openai_api', 'gemini_api'. "
                    "Leave empty to use the default from Settings.",
                    "enum": ["claude_cli", "anthropic_api", "openai_api", "gemini_api"],
                },
                "model": {
                    "type": "string",
                    "description": "Model override (depends on provider). Examples: "
                    "'claude-sonnet-4-20250514', 'gpt-4o', 'gemini-3-flash-preview'. "
                    "Leave empty to use the provider's default.",
                },
                "skip_ai_summary": {
                    "type": "boolean",
                    "description": "Skip AI summary generation at the end of workflow execution. "
                    "Default: false (summary is generated).",
                    "default": False,
                },
                "log_source_selection": {
                    "type": "string",
                    "description": "Log source selection mode: 'default' (use global profile), "
                    "'ai' (let AI select), 'all' (use all sources), or a specific profile_id.",
                    "default": "default",
                },
                "prompt_template": {
                    "type": "string",
                    "description": "Custom developer prompt template for the workflow's agentic phase. "
                    "Supports variables: {{SESSION_ID}}, {{ITERATION}}, {{MAX_ITERATIONS}}, "
                    "{{GOAL}}, {{EXECUTION_STEPS}}, {{WORKSPACE_ESCAPED}}.",
                },
                "auto_include_contexts": {
                    "type": "boolean",
                    "description": "Whether to auto-include AI contexts based on task mentions. "
                    "Default: true.",
                    "default": True,
                },
            },
            "required": ["description"],
        },
    ),
    # Plan Execution
    types.Tool(
        name="execute_plan",
        description="Execute a structured implementation plan with sequential AI phases. Each phase runs as a separate AI session with full context from prior phases.",
        inputSchema={
            "type": "object",
            "properties": {
                "plan_name": {
                    "type": "string",
                    "description": "Name of the plan",
                },
                "plan_overview": {
                    "type": "string",
                    "description": "Overview of what the plan accomplishes",
                },
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Phase name",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Instructions for this phase",
                            },
                        },
                        "required": ["name", "prompt"],
                    },
                    "minItems": 1,
                },
                "next_steps_sweep": {
                    "type": "boolean",
                    "description": "Run a sweep after all phases (default: true)",
                    "default": True,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional timeout per AI session",
                },
            },
            "required": ["plan_name", "plan_overview", "phases"],
        },
    ),
    # Visual Context for AI
    types.Tool(
        name="get_annotated_screenshot",
        description="Get an annotated screenshot with element IDs and bounding boxes for AI consumption. "
        "Elements are labeled with IDs (E001, E002, etc.) and color-coded by type (button=orange, "
        "input=green, link=cyan, etc.). Useful for AI to understand current GUI state and reference "
        "specific elements by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "screenshot_base64": {
                    "type": "string",
                    "description": "Optional base64-encoded screenshot. If not provided, captures current screen.",
                },
                "auto_detect": {
                    "type": "boolean",
                    "description": "Automatically detect UI elements if not provided (default: true).",
                    "default": True,
                },
                "include_legend": {
                    "type": "boolean",
                    "description": "Include a color legend in the output image (default: true).",
                    "default": True,
                },
            },
        },
    ),
    types.Tool(
        name="get_visual_diff",
        description="Get a visual diff highlighting changes between two screenshots. "
        "Shows regions that appeared (green) or disappeared (red), with change statistics. "
        "Useful for AI to understand what changed after an action.",
        inputSchema={
            "type": "object",
            "properties": {
                "before_base64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot before the change.",
                },
                "after_base64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot after the change.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Pixel difference threshold (0-255) for change detection (default: 30).",
                    "default": 30.0,
                },
                "min_region_area": {
                    "type": "integer",
                    "description": "Minimum pixel area for a region to be reported (default: 100).",
                    "default": 100,
                },
            },
            "required": ["before_base64", "after_base64"],
        },
    ),
    types.Tool(
        name="get_interaction_heatmap",
        description="Get an interaction heatmap showing clickable/interactive regions. "
        "Warmer colors indicate higher interactivity potential. Useful for AI to identify "
        "where actions can be taken on the current screen.",
        inputSchema={
            "type": "object",
            "properties": {
                "screenshot_base64": {
                    "type": "string",
                    "description": "Optional base64-encoded screenshot. If not provided, captures current screen.",
                },
                "auto_detect": {
                    "type": "boolean",
                    "description": "Automatically detect UI elements if not provided (default: true).",
                    "default": True,
                },
                "alpha": {
                    "type": "number",
                    "description": "Heatmap transparency (0-1, higher = more visible). Default: 0.6.",
                    "default": 0.6,
                },
            },
        },
    ),
    # State Machine tools
    types.Tool(
        name="load_state_machine",
        description="Load a UI Bridge state machine configuration into the runner. "
        "The config should be the JSON export from the web UI's state machine builder, "
        "compatible with UIBridgeRuntime.from_dict().",
        inputSchema={
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "The state machine configuration JSON (exported from web UI).",
                },
            },
            "required": ["config"],
        },
    ),
    types.Tool(
        name="get_state_machine_status",
        description="Get the status and statistics of the currently loaded state machine. "
        "Returns whether a state machine is loaded, and if so, counts of states, "
        "transitions, active states, and complexity metrics.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="get_active_states",
        description="Get the currently active states in the loaded state machine. "
        "Active states are determined by querying the UI Bridge for visible elements "
        "and mapping them to registered states.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="execute_state_transition",
        description="Execute a specific transition by ID in the loaded state machine. "
        "This runs the transition's action sequence (clicks, typing, etc.) "
        "and updates the state machine's active states.",
        inputSchema={
            "type": "object",
            "properties": {
                "transition_id": {
                    "type": "string",
                    "description": "The ID of the transition to execute.",
                },
            },
            "required": ["transition_id"],
        },
    ),
    types.Tool(
        name="navigate_to_states",
        description="Navigate to target states using pathfinding. "
        "The state machine will find the optimal path from current active states "
        "to the target states and execute each transition along the way.",
        inputSchema={
            "type": "object",
            "properties": {
                "target_states": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of target state IDs to navigate to.",
                },
            },
            "required": ["target_states"],
        },
    ),
    types.Tool(
        name="get_available_transitions",
        description="Get all transitions that are currently available from the active states. "
        "Returns the list of transitions that can be executed right now.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    # Scenario Projection tools (UI Bridge IR doc projections)
    types.Tool(
        name="project_scenarios",
        description=(
            "Get a deterministic static scenario projection of an IR document - "
            "the states, their outbound transitions, and required-element counts. "
            "Useful for understanding the structure of a UI's state machine without "
            "executing it. Returns byte-identical output for the same IR. "
            "For runtime-aware projection (current state, available transitions), "
            "use project_current_scenario instead."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "ir_doc_id": {
                    "type": "string",
                    "description": (
                        "Identifier of the IR document to project. "
                        "Same id space as /spec/page/{id}."
                    ),
                },
            },
            "required": ["ir_doc_id"],
        },
    ),
    types.Tool(
        name="project_current_scenario",
        description=(
            "Get a runtime-aware scenario projection - like project_scenarios, but "
            "augmented with the live page state: which states are currently active, "
            "which transitions resolve in the live registry (available), and which "
            "do not (blocked, with the resolution failure cause). Non-deterministic "
            "by design (depends on live page state). Use this to answer 'what can "
            "the user do right now?'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "ir_doc_id": {
                    "type": "string",
                    "description": (
                        "Identifier of the IR document. The runtime view is "
                        "computed against this IR's structure plus the live "
                        "UI Bridge registry."
                    ),
                },
            },
            "required": ["ir_doc_id"],
        },
    ),
    # Spec-Check tools (B-style spec verification against the live UI)
    types.Tool(
        name="check_page_spec",
        description=(
            "Run B-style spec verification against the live UI. Loads the page "
            "spec at `page_id`, fetches the current UI Bridge snapshot, and "
            "reports per-state match outcomes. Returns just the summary + "
            "recommended state when `mode='summary'`; returns the full "
            "SpecCheckResult when `mode='full'`."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": (
                        "The spec page id (matches the directory under "
                        "qontinui-runner/specs/pages/)."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": (
                        "summary returns SpecCheckSummary + RecommendedState "
                        "only; full returns the entire SpecCheckResult."
                    ),
                    "default": "summary",
                },
            },
            "required": ["page_id"],
        },
    ),
    types.Tool(
        name="list_page_specs",
        description=(
            "Enumerate every page spec registered under "
            "qontinui-runner/specs/pages/. Returns one entry per spec with id, "
            "app name, and high-level config metadata (version, description, "
            "group count). Useful for picking a page_id to pass to "
            "check_page_spec."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="describe_page_spec",
        description=(
            "Read the bundled-page projection of a single spec, identified by "
            "`page_id`. Returns the same shape the legacy "
            "`*.spec.uibridge.json` files used: version, description, groups[], "
            "metadata. Distinct from check_page_spec, which evaluates the spec "
            "against the live UI."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The spec page id.",
                },
            },
            "required": ["page_id"],
        },
    ),
    types.Tool(
        name="validate_page_spec",
        description=(
            "Validate the on-disk IR for a page spec against the G2 "
            "distinctness rules: no empty criteria, no identical state "
            "element-sets, no subset domination. Does NOT require a live app "
            "- purely an IR-level check. Returns the DistinctnessReport: "
            "`{ ok: bool, violations: [{ reason, ... }] }` "
            "(reason in emptyCriteria | identicalStates | subsetDomination)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The spec page id.",
                },
            },
            "required": ["page_id"],
        },
    ),
    # GUI Config Pipeline tools (element-to-image + config bridge)
    types.Tool(
        name="capture_gui_elements",
        description="Capture element images from the runner's current UI page. "
        "Gets a UI Bridge snapshot (element positions) and a screenshot, then "
        "crops each element into a base64 PNG image. Returns element images with "
        "metadata (id, label, type, dimensions, base64 data). "
        "Use this to build visual GUI automation configs from live UI.",
        inputSchema={
            "type": "object",
            "properties": {
                "window_offset_x": {
                    "type": "integer",
                    "description": "X offset from monitor origin to webview content area. Default: 0.",
                    "default": 0,
                },
                "window_offset_y": {
                    "type": "integer",
                    "description": "Y offset from monitor origin to webview content area. Default: 0.",
                    "default": 0,
                },
                "scale_factor": {
                    "type": "number",
                    "description": "DPI scale factor (e.g. 2.0 for Retina). Default: 1.0.",
                    "default": 1.0,
                },
                "category_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include elements in these categories (e.g. ['interactive']).",
                },
                "min_element_size": {
                    "type": "integer",
                    "description": "Minimum element width/height in pixels. Default: 4.",
                    "default": 4,
                },
                "padding": {
                    "type": "integer",
                    "description": "Extra pixels around each element crop. Default: 0.",
                    "default": 0,
                },
            },
        },
    ),
    types.Tool(
        name="capture_multi_state_gui_config",
        description="Capture a complete multi-state GUI automation config in one call. "
        "Walks through a sequence of interactions (click, scroll), capturing "
        "screenshots and element snapshots at each step. Automatically diffs "
        "element sets to assign only NEW elements to each state, crops images, "
        "builds transitions, and produces a complete QontinuiConfig. "
        "Much simpler than calling capture_gui_elements + build_gui_config manually.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Config name.",
                },
                "interactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "description": "Action: 'initial' (no action, just capture), "
                                "'click', or 'scroll'.",
                                "enum": ["initial", "click", "scroll"],
                            },
                            "target": {
                                "type": "string",
                                "description": "Element ID to act on (not needed for 'initial').",
                            },
                            "state_name": {
                                "type": "string",
                                "description": "Human-readable name for this state.",
                            },
                            "wait_seconds": {
                                "type": "number",
                                "description": "Seconds to wait after action before capture. Default: 1.0.",
                                "default": 1.0,
                            },
                        },
                        "required": ["action_type", "state_name"],
                    },
                    "description": "Ordered list of interactions. First should be 'initial'.",
                },
                "min_element_size": {
                    "type": "integer",
                    "description": "Minimum element width/height in pixels. Default: 4.",
                    "default": 4,
                },
                "description": {
                    "type": "string",
                    "description": "Optional config description.",
                },
                "similarity": {
                    "type": "number",
                    "description": "Default similarity threshold for pattern matching. Default: 0.85.",
                    "default": 0.85,
                },
            },
            "required": ["name", "interactions"],
        },
    ),
    types.Tool(
        name="build_gui_config",
        description="Build a visual GUI automation config (QontinuiConfig) from captured "
        "element images and state/transition definitions. The output is a complete "
        "JSON config with base64-encoded template images, ready for import into "
        "the web's State Machine page at /automation-builder/states. "
        "Use capture_gui_elements first to get element images, then define states "
        "and transitions referencing those element IDs.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Config name.",
                },
                "states": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "element_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "description": {"type": "string"},
                            "is_initial": {"type": "boolean"},
                            "is_final": {"type": "boolean"},
                        },
                        "required": ["id", "name", "element_ids"],
                    },
                    "description": "State definitions. Each state lists the element IDs that must be visible.",
                },
                "transitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "from_states": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "activate_states": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "exit_states": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "stays_visible": {"type": "boolean"},
                        },
                        "required": [
                            "id",
                            "name",
                            "from_states",
                            "activate_states",
                            "exit_states",
                        ],
                    },
                    "description": "Transition definitions.",
                },
                "element_images": {
                    "type": "object",
                    "description": "Mapping of element_id -> {base64_png, width, height, sha256, label}. "
                    "Typically the output of capture_gui_elements.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional config description.",
                },
                "similarity": {
                    "type": "number",
                    "description": "Default similarity threshold for pattern matching. Default: 0.85.",
                    "default": 0.85,
                },
            },
            "required": ["name", "states", "transitions", "element_images"],
        },
    ),
    # -------------------------------------------------------------------------
    # Observation Memory Tools (Engram-inspired persistent cross-session memory)
    # -------------------------------------------------------------------------
    types.Tool(
        name="mem_save",
        description="Save a persistent observation (decision, architecture, bugfix, pattern, learning, discovery). "
        "Supports topic_key for evolving knowledge (upserts existing observation with same key). "
        "Content within <private>...</private> tags is automatically redacted.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the observation.",
                },
                "content": {
                    "type": "string",
                    "description": "Full observation content (markdown supported).",
                },
                "observation_type": {
                    "type": "string",
                    "enum": [
                        "decision",
                        "architecture",
                        "bugfix",
                        "pattern",
                        "learning",
                        "discovery",
                    ],
                    "description": "Type classification for the observation.",
                },
                "topic_key": {
                    "type": "string",
                    "description": "Optional stable key for upsert semantics (e.g. 'architecture/auth-model'). "
                    "Saving with an existing topic_key updates the observation.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["project", "personal", "global"],
                    "description": "Scope of the observation. Default: project.",
                    "default": "project",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID for scoping. Optional.",
                },
                "workflow_id": {
                    "type": "string",
                    "description": "Related workflow ID. Optional.",
                },
                "task_run_id": {
                    "type": "string",
                    "description": "Related task run ID. Optional.",
                },
            },
            "required": ["title", "content", "observation_type"],
        },
    ),
    types.Tool(
        name="mem_search",
        description="Full-text search over persistent observations. Returns 300-char previews "
        "with relevance ranking. Use mem_get for full content.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language, supports stemming).",
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional: scope search to a project.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results. Default: 20.",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="mem_get",
        description="Get the full content of an observation by ID. Use after mem_search "
        "to retrieve complete content (progressive disclosure).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Observation ID.",
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="mem_context",
        description="Get relevant observations for a project. Returns recent observations "
        "scoped to the project plus any global observations.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project ID to get context for.",
                },
                "observation_type": {
                    "type": "string",
                    "enum": [
                        "decision",
                        "architecture",
                        "bugfix",
                        "pattern",
                        "learning",
                        "discovery",
                    ],
                    "description": "Optional: filter by observation type.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results. Default: 20.",
                    "default": 20,
                },
            },
            "required": ["project_id"],
        },
    ),
    types.Tool(
        name="mem_delete",
        description="Soft-delete an observation by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Observation ID to delete.",
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="mem_stats",
        description="Get observation statistics grouped by type.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="mem_update",
        description="Update an existing observation's title, content, or type.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Observation ID to update.",
                },
                "title": {
                    "type": "string",
                    "description": "New title (optional).",
                },
                "content": {
                    "type": "string",
                    "description": "New content (optional).",
                },
                "observation_type": {
                    "type": "string",
                    "enum": [
                        "decision",
                        "architecture",
                        "bugfix",
                        "pattern",
                        "learning",
                        "discovery",
                    ],
                    "description": "New type classification (optional).",
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="mem_by_task_run",
        description="Get all observations linked to a specific task run.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_run_id": {
                    "type": "string",
                    "description": "Task run ID to query observations for.",
                },
            },
            "required": ["task_run_id"],
        },
    ),
    types.Tool(
        name="mem_export",
        description="Export all observations as JSON for backup or sharing.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="mem_cleanup",
        description="Run retention policy to archive old, low-value observations.",
        inputSchema={
            "type": "object",
            "properties": {
                "retention_days": {
                    "type": "integer",
                    "description": "Days to retain. Default: 90.",
                    "default": 90,
                },
                "max_revision_count": {
                    "type": "integer",
                    "description": "Max revision count for cleanup candidates. Default: 1.",
                    "default": 1,
                },
            },
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
            response = await qontinui.generate_workflow(
                description=description,
                category=arguments.get("category"),
                tags=arguments.get("tags"),
                max_iterations=arguments.get("max_iterations"),
                provider=arguments.get("provider"),
                model=arguments.get("model"),
                skip_ai_summary=arguments.get("skip_ai_summary"),
                log_source_selection=arguments.get("log_source_selection"),
                prompt_template=arguments.get("prompt_template"),
                auto_include_contexts=arguments.get("auto_include_contexts"),
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "execute_plan":
            plan_name = arguments.get("plan_name", "")
            plan_overview = arguments.get("plan_overview", "")
            phases = arguments.get("phases", [])
            if not plan_name or not plan_overview or not phases:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "plan_name, plan_overview, and phases are required",
                            },
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.execute_plan(
                plan_name=plan_name,
                plan_overview=plan_overview,
                phases=phases,
                next_steps_sweep=arguments.get("next_steps_sweep", True),
                timeout_seconds=arguments.get("timeout_seconds"),
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        # Visual Context for AI
        elif name == "get_annotated_screenshot":
            screenshot_base64 = arguments.get("screenshot_base64")
            auto_detect = arguments.get("auto_detect", True)
            include_legend = arguments.get("include_legend", True)
            response = await qontinui.get_annotated_screenshot(
                screenshot_base64=screenshot_base64,
                auto_detect=auto_detect,
                include_legend=include_legend,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_visual_diff":
            before_base64 = arguments.get("before_base64", "")
            after_base64 = arguments.get("after_base64", "")
            if not before_base64 or not after_base64:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": "before_base64 and after_base64 are required",
                            },
                            indent=2,
                        ),
                    )
                ]
            threshold = arguments.get("threshold", 30.0)
            min_region_area = arguments.get("min_region_area", 100)
            response = await qontinui.get_visual_diff(
                before_base64=before_base64,
                after_base64=after_base64,
                threshold=threshold,
                min_region_area=min_region_area,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_interaction_heatmap":
            screenshot_base64 = arguments.get("screenshot_base64")
            auto_detect = arguments.get("auto_detect", True)
            alpha = arguments.get("alpha", 0.6)
            response = await qontinui.get_interaction_heatmap(
                screenshot_base64=screenshot_base64,
                auto_detect=auto_detect,
                alpha=alpha,
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        # State Machine tools
        elif name == "load_state_machine":
            config = arguments.get("config")
            if not config:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "config is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.load_state_machine(config)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_state_machine_status":
            response = await qontinui.get_state_machine_status()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_active_states":
            response = await qontinui.get_sm_active_states()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "execute_state_transition":
            transition_id = arguments.get("transition_id", "")
            if not transition_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "transition_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.execute_state_transition(transition_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "navigate_to_states":
            target_states = arguments.get("target_states", [])
            if not target_states:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "target_states is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.navigate_to_states(target_states)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "get_available_transitions":
            response = await qontinui.get_sm_available_transitions()
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        # Scenario Projection tools
        elif name == "project_scenarios":
            ir_doc_id = arguments.get("ir_doc_id", "")
            if not ir_doc_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "ir_doc_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.project_scenarios(ir_doc_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "project_current_scenario":
            ir_doc_id = arguments.get("ir_doc_id", "")
            if not ir_doc_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "ir_doc_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.project_current_scenario(ir_doc_id)
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        # Spec-Check tools
        elif name == "check_page_spec":
            page_id = arguments.get("page_id", "")
            if not page_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "page_id is required"},
                            indent=2,
                        ),
                    )
                ]
            mode = arguments.get("mode", "summary")
            response = await qontinui.check_page_spec(page_id)
            body = response.data
            if not response.success or not isinstance(body, dict):
                # Surface the SpecError envelope (e.g. spec-not-found) or
                # the transport error verbatim.
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            body
                            or {
                                "ok": False,
                                "reason": "request-failed",
                                "error": response.error,
                            },
                            indent=2,
                        ),
                    )
                ]
            # An un-spec'd page returns 2xx-or-not with an {ok: false, ...}
            # envelope — pass it through unchanged, do not paper over it.
            if body.get("ok") is False:
                return [types.TextContent(type="text", text=json.dumps(body, indent=2))]
            if mode == "summary":
                # SpecCheckSummary already carries `recommended_state` as a
                # nested field, so dropping `state_results` (the heavy
                # per-state breakdown) is sufficient for the summary contract.
                summary_only = {
                    "ok": True,
                    "result_schema_version": body.get("result_schema_version"),
                    "page_id": body.get("page_id"),
                    "summary": body.get("summary"),
                    "warnings": body.get("warnings", []),
                }
                return [
                    types.TextContent(
                        type="text", text=json.dumps(summary_only, indent=2)
                    )
                ]
            return [types.TextContent(type="text", text=json.dumps(body, indent=2))]

        elif name == "list_page_specs":
            response = await qontinui.list_page_specs()
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        response.data
                        if response.data is not None
                        else {
                            "ok": False,
                            "reason": "request-failed",
                            "error": response.error,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "describe_page_spec":
            page_id = arguments.get("page_id", "")
            if not page_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "page_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.describe_page_spec(page_id)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        response.data
                        if response.data is not None
                        else {
                            "ok": False,
                            "reason": "request-failed",
                            "error": response.error,
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "validate_page_spec":
            page_id = arguments.get("page_id", "")
            if not page_id:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": "page_id is required"},
                            indent=2,
                        ),
                    )
                ]
            response = await qontinui.validate_page_spec(page_id)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        response.data
                        if response.data is not None
                        else {
                            "ok": False,
                            "reason": "request-failed",
                            "error": response.error,
                        },
                        indent=2,
                    ),
                )
            ]

        # GUI Config Pipeline tools
        elif name == "capture_gui_elements":
            return await _handle_capture_gui_elements(qontinui, arguments)

        elif name == "capture_multi_state_gui_config":
            return await _handle_capture_multi_state_gui_config(qontinui, arguments)

        elif name == "build_gui_config":
            return await _handle_build_gui_config(qontinui, arguments)

        # ----- Observation Memory Tools -----

        elif name == "mem_save":
            payload: dict[str, Any] = {
                "title": arguments["title"],
                "content": arguments["content"],
                "observationType": arguments["observation_type"],
                "scope": arguments.get("scope", "project"),
            }
            if arguments.get("topic_key"):
                payload["topicKey"] = arguments["topic_key"]
            if arguments.get("project_id"):
                payload["projectId"] = arguments["project_id"]
            if arguments.get("workflow_id"):
                payload["workflowId"] = arguments["workflow_id"]
            if arguments.get("task_run_id"):
                payload["taskRunId"] = arguments["task_run_id"]
            response = await qontinui._request(
                "POST", "/observations", json_data=payload
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_search":
            params = f"?q={quote(arguments['query'])}&max_results={arguments.get('max_results', 20)}"
            if arguments.get("project_id"):
                params += f"&project_id={quote(arguments['project_id'])}"
            response = await qontinui._request("GET", f"/observations/search{params}")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_get":
            obs_id = arguments["id"]
            response = await qontinui._request("GET", f"/observations/{obs_id}")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_context":
            params = f"?project_id={quote(arguments['project_id'])}&max_results={arguments.get('max_results', 20)}"
            if arguments.get("observation_type"):
                params += f"&observation_type={quote(arguments['observation_type'])}"
            response = await qontinui._request("GET", f"/observations/context{params}")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_delete":
            obs_id = arguments["id"]
            response = await qontinui._request("DELETE", f"/observations/{obs_id}")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_stats":
            response = await qontinui._request("GET", "/observations/stats")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_update":
            obs_id = arguments["id"]
            update_payload: dict[str, Any] = {}
            if arguments.get("title"):
                update_payload["title"] = arguments["title"]
            if arguments.get("content"):
                update_payload["content"] = arguments["content"]
            if arguments.get("observation_type"):
                update_payload["observationType"] = arguments["observation_type"]
            response = await qontinui._request(
                "PUT", f"/observations/{obs_id}", json_data=update_payload
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_by_task_run":
            task_run_id = quote(arguments["task_run_id"])
            response = await qontinui._request(
                "GET", f"/observations/by-task-run/{task_run_id}"
            )
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_export":
            response = await qontinui._request("GET", "/observations/export")
            return [
                types.TextContent(
                    type="text", text=json.dumps(response.__dict__, indent=2)
                )
            ]

        elif name == "mem_cleanup":
            params = f"?retention_days={arguments.get('retention_days', 90)}&max_revision_count={arguments.get('max_revision_count', 1)}"
            response = await qontinui._request("POST", f"/observations/cleanup{params}")
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


# ---------------------------------------------------------------------------
# GUI Config Pipeline handlers
# ---------------------------------------------------------------------------


async def _handle_capture_gui_elements(
    qontinui: QontinuiClient, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Capture element images from the runner's current UI page."""
    try:
        window_offset_x = int(arguments.get("window_offset_x", 0))
        window_offset_y = int(arguments.get("window_offset_y", 0))
        scale_factor = float(arguments.get("scale_factor", 1.0))
        min_element_size = int(arguments.get("min_element_size", 4))
        padding = int(arguments.get("padding", 0))
    except (ValueError, TypeError) as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "error": f"Invalid numeric argument: {e}"}
                ),
            )
        ]
    response = await qontinui.capture_gui_elements(
        window_offset_x=window_offset_x,
        window_offset_y=window_offset_y,
        scale_factor=scale_factor,
        category_filter=arguments.get("category_filter"),
        min_element_size=min_element_size,
        padding=padding,
    )
    return [
        types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))
    ]


async def _handle_capture_multi_state_gui_config(
    qontinui: QontinuiClient, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Capture a complete multi-state GUI config in one call."""
    name = arguments.get("name", "Untitled Multi-State Config")
    interactions = arguments.get("interactions", [])
    if not interactions:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "error": "interactions list is required"}
                ),
            )
        ]

    response = await qontinui.capture_multi_state_gui_config(
        name=name,
        interactions=interactions,
        min_element_size=int(arguments.get("min_element_size", 4)),
        description=arguments.get("description", ""),
        similarity=float(arguments.get("similarity", 0.85)),
    )
    return [
        types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))
    ]


async def _handle_build_gui_config(
    qontinui: QontinuiClient,
    arguments: dict[str, Any],
) -> list[types.TextContent]:
    """Build a QontinuiConfig from element images and state/transition definitions."""
    name = arguments.get("name", "Untitled Config")
    states = arguments.get("states", [])
    transitions = arguments.get("transitions", [])
    element_images = arguments.get("element_images", {})
    description = arguments.get("description", "")
    similarity = float(arguments.get("similarity", 0.85))

    if not states:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"success": False, "error": "states is required"}),
            )
        ]
    if not element_images:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "error": "element_images is required"}
                ),
            )
        ]

    response = await qontinui.build_gui_config(
        name=name,
        states=states,
        transitions=transitions,
        element_images=element_images,
        description=description,
        similarity=similarity,
    )
    return [
        types.TextContent(type="text", text=json.dumps(response.__dict__, indent=2))
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
