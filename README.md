# qontinui-mcp

Lightweight MCP server for [Qontinui Runner](https://github.com/qontinui/qontinui-runner) - enables AI-driven visual automation.

## Installation

```bash
pip install qontinui-mcp
```

## Quick Start

1. **Start the Qontinui Runner** (desktop application)

2. **Configure your AI client** (Claude Desktop, Claude Code, Cursor, etc.)

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "qontinui-mcp",
      "args": []
    }
  }
}
```

3. **Run workflows via AI**

The AI can now:
- Load workflow configuration files
- Run visual automation workflows
- Monitor execution status
- Control which monitor to use

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QONTINUI_RUNNER_HOST` | Runner host address | Auto-detected (WSL-aware) |
| `QONTINUI_RUNNER_PORT` | Runner HTTP port | `9876` |
| `QONTINUI_RESULTS_DIR` | Directory for automation results | `.automation-results` |
| `QONTINUI_DEV_LOGS_DIR` | Directory for dev logs | `.dev-logs` |

## Features

### Area A: SSE Event Streaming

Real-time event streaming via Server-Sent Events (SSE) for monitoring workflow execution.

**Endpoint:** `/sse/events`

**Client Usage:**

```python
from qontinui_mcp.client import QontinuiClient

client = QontinuiClient()

def handle_event(event: dict):
    print(f"Event: {event['event_type']} - {event}")

await client.subscribe_events(callback=handle_event, timeout=60)
```

**Event Types:**
- `qontinui/execution_started` - Workflow begins
- `qontinui/execution_progress` - Step completion
- `qontinui/execution_completed` - Workflow ends
- `qontinui/test_started` - Test begins
- `qontinui/test_completed` - Test ends
- `qontinui/image_recognition` - Match found/failed
- `qontinui/error` - Error occurs
- `qontinui/warning` - Non-fatal issue

### Area B: MCP Prompts

Parameterized prompt templates for common automation tasks. Prompts aggregate context from the runner to provide structured debugging, analysis, and verification workflows.

| Prompt | Description | Arguments |
|--------|-------------|-----------|
| `debug_test_failure` | Analyze a test failure with structured debugging approach | `test_id` (required), `include_screenshots` |
| `analyze_screenshot` | Visual analysis of a screenshot for UI verification | `screenshot_id` (required), `focus_area` |
| `fix_playwright_failure` | Structured workflow to fix a failing Playwright test | `spec_name` (required), `error_message` |
| `verify_workflow_state` | Verify current GUI state matches expected workflow state | `state_name` (required), `workflow_name` |
| `create_verification_test` | Generate a verification test for a UI behavior | `behavior_description` (required), `test_type` |
| `analyze_automation_run` | Review automation run results and identify issues | `run_id`, `focus_on_failures` |
| `debug_image_recognition` | Debug template matching and image recognition issues | `template_name`, `last_n_attempts` |
| `summarize_task_progress` | Task status summary with execution progress | `task_run_id` |
| `analyze_verification_failure` | Analyze why a verification criterion failed | `task_id` (required), `criterion_id` |
| `create_verification_plan` | Generate a verification plan for a feature | `feature_description` (required), `strategy` |

### Area C: Tool Caching

Version-based tool caching to optimize MCP tool list requests.

**Endpoint:** `/tool-version`

**Response:**
```json
{
  "version": "abc123...",
  "tool_count": 35,
  "test_count": 12
}
```

The MCP server caches tools and invalidates the cache when:
- The runner's tool version changes (config loaded, tests added/removed)
- Cache is older than 5 minutes (fallback)

### Area E: Permission System

Fine-grained permission control for tool calls inspired by OpenCode's permission system.

**Permission Levels:**

| Level | Description | Example Tools |
|-------|-------------|---------------|
| `READ_ONLY` | Safe operations that only read data | `get_executor_status`, `list_monitors`, `read_runner_logs` |
| `EXECUTE` | Operations that run workflows or tests | `run_workflow`, `execute_test`, `execute_python` |
| `MODIFY` | Operations that change data | `create_test`, `update_test`, `load_config` |
| `DANGEROUS` | Operations that can disrupt execution | `stop_execution`, `restart_runner` |

**Configuration:**

```python
from qontinui_mcp.permissions import get_permission_service, PermissionLevel

service = get_permission_service()

# Auto-approve only read operations (default)
service.configure(auto_approve_levels={PermissionLevel.READ_ONLY})

# Auto-approve all operations (trusted context)
service.auto_approve_all()

# Custom permission handler
service.on_request = lambda req: input(f"Allow {req.tool_name}? (y/n)") == "y"
```

### Area F: MCP Resources

Read-only data access via URI scheme for accessing runner data.

**URI Scheme:** `qontinui://{type}/{id}`

**Resource Types:**

| URI Pattern | Description | MIME Type |
|-------------|-------------|-----------|
| `qontinui://config/current` | Currently loaded workflow configuration | `application/json` |
| `qontinui://logs/{type}` | JSONL log files (general, actions, image-recognition, playwright) | `application/jsonl` |
| `qontinui://screenshots/{id}` | Screenshot metadata and file paths | `image/png` |
| `qontinui://tests/{id}` | Verification test definitions | `application/json` |
| `qontinui://dom/{id}` | DOM capture HTML content | `text/html` |
| `qontinui://task-runs/{id}` | Task run details | `application/json` |

### Area G: Inline Python Execution

Execute arbitrary Python code with optional dependency isolation via uvx.

**Tool:** `execute_python`

**Parameters:**
- `code` (required): Python code to execute
- `dependencies`: List of pip packages to install
- `timeout_seconds`: Execution timeout (default: 30)
- `working_directory`: Working directory for execution

**Example:**

```python
# Simple calculation
result = await client.execute_python(
    code="return {'sum': 1 + 2, 'product': 3 * 4}"
)
# result.data["return_value"] == {"sum": 3, "product": 12}

# With dependencies
result = await client.execute_python(
    code="""
    import requests
    resp = requests.get('https://api.example.com/data')
    return resp.json()
    """,
    dependencies=["requests"],
)
```

### Area H: Agent Spawning

Hierarchical task decomposition by spawning sub-agents with focused tasks.

**Tool:** `spawn_sub_agent`

**Parameters:**
- `task` (required): Task description for the sub-agent
- `tools`: List of tool names to restrict the sub-agent to
- `max_iterations`: Maximum turns/iterations (default: 10)
- `context`: Additional context to provide

**Example:**

```python
result = await client.spawn_sub_agent(
    task="Verify that the login form works correctly",
    tools=["run_workflow", "capture_screenshot", "execute_test"],
    max_iterations=5,
    context="The login page is at /login with username and password fields."
)
```

## Available Tools

### Core Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `get_executor_status` | READ_ONLY | Get runner status |
| `list_monitors` | READ_ONLY | List available monitors |
| `load_config` | MODIFY | Load a workflow configuration file |
| `ensure_config_loaded` | MODIFY | Load config if not already loaded |
| `get_loaded_config` | READ_ONLY | Get loaded configuration info |
| `run_workflow` | EXECUTE | Run a workflow by name |
| `stop_execution` | DANGEROUS | Stop current execution |

### Task Management Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `get_task_runs` | READ_ONLY | Get all task runs |
| `get_task_run` | READ_ONLY | Get specific task run details |
| `get_task_run_events` | READ_ONLY | Get events for a task run |
| `get_task_run_screenshots` | READ_ONLY | Get screenshots for a task run |
| `get_task_run_playwright_results` | READ_ONLY | Get Playwright results for a task run |
| `migrate_task_run_logs` | EXECUTE | Migrate JSONL logs to SQLite |

### Automation Run Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `get_automation_runs` | READ_ONLY | Get recent automation runs |
| `get_automation_run` | READ_ONLY | Get specific automation run details |

### Test Management Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `list_tests` | READ_ONLY | List all verification tests |
| `get_test` | READ_ONLY | Get test by ID |
| `execute_test` | EXECUTE | Execute a verification test |
| `list_test_results` | READ_ONLY | List test results |
| `get_test_history` | READ_ONLY | Get test history summary |
| `create_test` | MODIFY | Create a new verification test |
| `update_test` | MODIFY | Update an existing test |
| `delete_test` | MODIFY | Delete a verification test |

**Test Types:**
- `playwright_cdp` - Browser DOM assertions using Playwright
- `qontinui_vision` - Visual verification using image recognition
- `python_script` - Custom Python verification logic
- `repository_test` - Run pytest, Jest, or other test frameworks

### Log Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `list_screenshots` | READ_ONLY | List available screenshots |
| `read_runner_logs` | READ_ONLY | Read runner JSONL log files |

**Log Types:**
- `general` - General executor events
- `actions` - Workflow action/tree events
- `image-recognition` - Image recognition results with match details
- `playwright` - Playwright test execution results

### DOM Capture Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `list_dom_captures` | READ_ONLY | List DOM captures |
| `get_dom_capture` | READ_ONLY | Get DOM capture metadata |
| `get_dom_capture_html` | READ_ONLY | Get DOM capture HTML content |

### AWAS (AI Web Action Standard) Tools

Tools for interacting with websites that support the AWAS standard.

| Tool | Permission | Description |
|------|------------|-------------|
| `awas_discover` | EXECUTE | Discover AWAS manifest for a website |
| `awas_check_support` | READ_ONLY | Check if a website supports AWAS |
| `awas_list_actions` | READ_ONLY | List available AWAS actions |
| `awas_execute` | EXECUTE | Execute an AWAS action |

### Advanced Tools

| Tool | Permission | Description |
|------|------------|-------------|
| `execute_python` | EXECUTE | Execute inline Python code |
| `spawn_sub_agent` | EXECUTE | Spawn a sub-agent with a specific task |

## Example Usage

### Basic Workflow Execution

```python
# In an AI conversation:
"Load the config at /path/to/workflow.json and run the 'login_test' workflow on the left monitor"
```

### Test-Driven Verification

```python
# Create a verification test
"Create a Playwright test that verifies the login button is visible and enabled"

# Execute the test
"Run the login_button_visible test and show me the results"

# Debug failures
"Use the debug_test_failure prompt for test abc123 with screenshots"
```

### Automation Analysis

```python
# Analyze a failed automation run
"Analyze the most recent automation run and identify why it failed"

# Debug image recognition
"Debug the template matching for the 'submit_button' template"
```

## Development

```bash
# Clone
git clone https://github.com/qontinui/qontinui-mcp
cd qontinui-mcp

# Install dependencies
poetry install

# Run server locally
poetry run qontinui-mcp

# Run type checking
poetry run mypy src/

# Run linting
poetry run ruff check src/
```

## Architecture

```
qontinui-mcp (MCP Server)
    |
    v
QontinuiClient (HTTP Client)
    |
    v
qontinui-runner (Desktop App, port 9876)
    |
    v
Python Subprocess (Qontinui Execution)
```

The MCP server is a thin wrapper that:
1. Exposes runner capabilities via MCP protocol
2. Provides permission control for tool calls
3. Caches tool definitions for performance
4. Aggregates context for structured prompts
5. Streams events via SSE for real-time monitoring

## License

MIT
