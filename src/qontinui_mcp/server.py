"""Qontinui MCP Server Implementation.

Provides AI-powered workflow generation, node discovery, and automation execution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from qontinui_mcp.database.loader import get_node, initialize_database
from qontinui_mcp.database.search import (
    get_all_categories,
    search_nodes,
    search_nodes_by_action_type,
    search_nodes_by_category,
    search_workflows,
)
from qontinui_mcp.tools.execution import (
    assert_state_visible,
    capture_screen,
    compare_screenshots,
    is_execution_available,
    run_automation,
    wait_for_state,
)
from qontinui_mcp.tools.generator import WorkflowGenerator
from qontinui_mcp.utils.validation import validate_workflow_structure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_DIR = Path.home() / ".qontinui" / "mcp"
DB_PATH = DB_DIR / "qontinui.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection, initializing if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return initialize_database(str(DB_PATH))


# Initialize MCP server
server = Server("qontinui-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    tools = [
        # Knowledge/Search Tools
        Tool(
            name="search_nodes",
            description="Search for Qontinui action nodes using natural language. Returns matching nodes with descriptions and parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'click button', 'find image', 'type text')",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_workflows",
            description="Search for Qontinui workflow templates using natural language. Returns matching workflow templates with use cases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'login workflow', 'data entry', 'form filling')",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_nodes_by_category",
            description="Get all nodes in a specific category (e.g., 'mouse', 'keyboard', 'vision').",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name",
                    },
                },
                "required": ["category"],
            },
        ),
        Tool(
            name="get_nodes_by_action_type",
            description="Get all nodes of a specific action type (e.g., 'CLICK', 'FIND', 'TYPE').",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": "Action type (e.g., CLICK, FIND, TYPE)",
                    },
                },
                "required": ["action_type"],
            },
        ),
        Tool(
            name="list_categories",
            description="List all available categories for nodes and workflows.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_action_details",
            description="Get detailed information about a specific action node including parameters, examples, and usage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The ID of the action node",
                    },
                },
                "required": ["action_id"],
            },
        ),
        # Workflow Tools
        Tool(
            name="validate_workflow",
            description="Validate a workflow JSON structure. Checks schema, connections, detects cycles, and finds unreachable steps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "The workflow JSON to validate",
                    },
                },
                "required": ["workflow"],
            },
        ),
        Tool(
            name="create_workflow",
            description="Create a workflow from a list of action steps. Automatically generates UUIDs and connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the workflow",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the workflow does",
                    },
                    "steps": {
                        "type": "array",
                        "description": "Array of workflow steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "description": "Action type (e.g., CLICK, FIND, TYPE)",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Step description",
                                },
                                "options": {
                                    "type": "object",
                                    "description": "Action options",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                "required": ["name", "description", "steps"],
            },
        ),
        Tool(
            name="generate_workflow",
            description="Generate a complete workflow from a natural language description. AI-powered workflow creation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the workflow (e.g., 'Login to website by clicking login button and typing credentials')",
                    },
                },
                "required": ["description"],
            },
        ),
        # Execution Tools
        Tool(
            name="run_automation",
            description="Execute a Qontinui automation script and return results. Requires qontinui library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The automation script in DSL format or JSON workflow",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default: 30)",
                        "default": 30,
                    },
                    "capture_screenshot": {
                        "type": "boolean",
                        "description": "Whether to capture final screenshot (default: true)",
                        "default": True,
                    },
                },
                "required": ["script"],
            },
        ),
        Tool(
            name="capture_screenshot",
            description="Capture the current screen and return as base64-encoded image.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="assert_state_visible",
            description="Assert that a visual state/element is visible on screen. Returns match confidence and location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "state_image_path": {
                        "type": "string",
                        "description": "Path to the reference image to find",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "How long to wait for the state (default: 10)",
                        "default": 10,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity for a match, 0-1 (default: 0.9)",
                        "default": 0.9,
                    },
                },
                "required": ["state_image_path"],
            },
        ),
        Tool(
            name="wait_for_state",
            description="Wait until a visual state appears on screen or timeout.",
            inputSchema={
                "type": "object",
                "properties": {
                    "state_image_path": {
                        "type": "string",
                        "description": "Path to the reference image to wait for",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum time to wait in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["state_image_path"],
            },
        ),
        Tool(
            name="compare_screenshots",
            description="Compare two screenshots and return difference analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image1_path": {
                        "type": "string",
                        "description": "Path to first image",
                    },
                    "image2_path": {
                        "type": "string",
                        "description": "Path to second image",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Maximum difference percentage to consider a match (default: 0.05)",
                        "default": 0.05,
                    },
                },
                "required": ["image1_path", "image2_path"],
            },
        ),
        Tool(
            name="is_execution_available",
            description="Check if execution tools are available (qontinui library installed).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    conn = get_db_connection()
    generator = WorkflowGenerator(conn)

    try:
        result: Any = None

        # Knowledge/Search Tools
        if name == "search_nodes":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not query:
                raise ValueError("Query parameter is required")
            result = search_nodes(conn, query, limit)

        elif name == "search_workflows":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not query:
                raise ValueError("Query parameter is required")
            result = search_workflows(conn, query, limit)

        elif name == "get_nodes_by_category":
            category = arguments.get("category", "")
            if not category:
                raise ValueError("Category parameter is required")
            result = search_nodes_by_category(conn, category)

        elif name == "get_nodes_by_action_type":
            action_type = arguments.get("action_type", "")
            if not action_type:
                raise ValueError("Action type parameter is required")
            result = search_nodes_by_action_type(conn, action_type)

        elif name == "list_categories":
            result = get_all_categories(conn)

        elif name == "get_action_details":
            action_id = arguments.get("action_id", "")
            if not action_id:
                raise ValueError("action_id parameter is required")
            node = get_node(conn, action_id)
            if not node:
                result = {"error": f"Action not found: {action_id}"}
            else:
                result = node

        # Workflow Tools
        elif name == "validate_workflow":
            workflow = arguments.get("workflow")
            if not workflow:
                raise ValueError("workflow parameter is required")
            validation = validate_workflow_structure(workflow)
            result = validation.model_dump()

        elif name == "create_workflow":
            wf_name = arguments.get("name", "")
            description = arguments.get("description", "")
            steps = arguments.get("steps", [])
            if not wf_name or not description or not steps:
                raise ValueError("name, description, and steps parameters are required")
            workflow = generator.create_workflow(wf_name, description, steps)
            validation = validate_workflow_structure(workflow)
            result = {
                "workflow": workflow,
                "validation": validation.model_dump(),
            }

        elif name == "generate_workflow":
            description = arguments.get("description", "")
            if not description:
                raise ValueError("description parameter is required")
            gen_result = generator.generate_from_description(description)
            result = {
                "success": gen_result.success,
                "workflow": gen_result.workflow,
                "error": gen_result.error,
                "suggestions": gen_result.suggestions,
            }

        # Execution Tools
        elif name == "run_automation":
            script = arguments.get("script", "")
            timeout = arguments.get("timeout_seconds", 30)
            capture = arguments.get("capture_screenshot", True)
            if not script:
                raise ValueError("script parameter is required")
            exec_result = run_automation(script, timeout, capture)
            result = {
                "success": exec_result.success,
                "duration_ms": exec_result.duration_ms,
                "errors": exec_result.errors,
                "output": exec_result.output,
                "screenshot_base64": exec_result.screenshot_base64,
            }

        elif name == "capture_screenshot":
            screenshot = capture_screen()
            result = {
                "success": screenshot.success,
                "image_base64": screenshot.image_base64,
                "width": screenshot.width,
                "height": screenshot.height,
                "error": screenshot.error,
            }

        elif name == "assert_state_visible":
            state_path = arguments.get("state_image_path", "")
            timeout = arguments.get("timeout_seconds", 10)
            threshold = arguments.get("similarity_threshold", 0.9)
            if not state_path:
                raise ValueError("state_image_path parameter is required")
            assertion = assert_state_visible(state_path, timeout, threshold)
            result = {
                "visible": assertion.visible,
                "confidence": assertion.confidence,
                "location": assertion.location,
                "duration_ms": assertion.duration_ms,
                "error": assertion.error,
            }

        elif name == "wait_for_state":
            state_path = arguments.get("state_image_path", "")
            timeout = arguments.get("timeout_seconds", 30)
            if not state_path:
                raise ValueError("state_image_path parameter is required")
            wait_result = wait_for_state(state_path, timeout)
            result = {
                "found": wait_result.visible,
                "confidence": wait_result.confidence,
                "location": wait_result.location,
                "duration_ms": wait_result.duration_ms,
                "error": wait_result.error,
            }

        elif name == "compare_screenshots":
            image1 = arguments.get("image1_path", "")
            image2 = arguments.get("image2_path", "")
            threshold = arguments.get("threshold", 0.05)
            if not image1 or not image2:
                raise ValueError("image1_path and image2_path parameters are required")
            comparison = compare_screenshots(image1, image2, threshold)
            result = {
                "match": comparison.match,
                "diff_percent": comparison.diff_percent,
                "diff_regions": comparison.diff_regions,
                "diff_image_base64": comparison.diff_image_base64,
                "error": comparison.error,
            }

        elif name == "is_execution_available":
            result = {"available": is_execution_available()}

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        raise

    finally:
        conn.close()


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting Qontinui MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    """Entry point."""
    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
