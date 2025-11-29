# Qontinui MCP Server (Python)

MCP server for Qontinui visual automation - enables AI-powered workflow generation and execution.

## Features

### Knowledge Tools
- **search_nodes** - Search for action nodes using natural language
- **search_workflows** - Search for workflow templates
- **get_nodes_by_category** - Filter nodes by category
- **get_nodes_by_action_type** - Filter by action type (CLICK, FIND, TYPE, etc.)
- **list_categories** - List all available categories
- **get_action_details** - Get detailed node information

### Workflow Tools
- **validate_workflow** - Validate workflow structure with cycle detection
- **create_workflow** - Create workflow from action steps
- **generate_workflow** - Generate workflow from natural language description

### Execution Tools (requires qontinui library)
- **run_automation** - Execute automation scripts
- **capture_screenshot** - Capture current screen
- **assert_state_visible** - Assert visual state is on screen
- **wait_for_state** - Wait until state appears
- **compare_screenshots** - Visual diff between images
- **is_execution_available** - Check if execution is available

## Installation

```bash
cd qontinui-mcp-py
poetry install
```

## Usage

### Run the MCP server

```bash
poetry run qontinui-mcp
```

### Configure with Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (Linux) or equivalent:

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "poetry",
      "args": ["run", "qontinui-mcp"],
      "cwd": "/path/to/qontinui-mcp-py"
    }
  }
}
```

Or using the installed script:

```json
{
  "mcpServers": {
    "qontinui": {
      "command": "/path/to/qontinui-mcp-py/.venv/bin/qontinui-mcp"
    }
  }
}
```

## Development

```bash
# Install dependencies
poetry install

# Run in development
poetry run python -m qontinui_mcp.server

# Run linting
poetry run black .
poetry run isort .
poetry run ruff check .
poetry run mypy src/
```

## Architecture

```
qontinui-mcp-py/
├── src/qontinui_mcp/
│   ├── __init__.py
│   ├── server.py           # MCP server implementation
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql      # SQLite + FTS5 schema
│   │   ├── loader.py       # DB initialization
│   │   └── search.py       # FTS5 search functions
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── generator.py    # Workflow generation
│   │   └── execution.py    # Automation execution
│   ├── types/
│   │   ├── __init__.py
│   │   └── models.py       # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── validation.py   # Workflow validation
├── pyproject.toml
└── README.md
```

## Action Types Supported

### Basic Actions
- FIND, CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK
- DEFINE, TYPE, MOVE, HOVER
- VANISH, WAIT_VANISH, HIGHLIGHT
- SCROLL_MOUSE_WHEEL, SCROLL_UP, SCROLL_DOWN
- MOUSE_DOWN, MOUSE_UP, KEY_DOWN, KEY_UP
- CLASSIFY

### Composite Actions
- CLICK_UNTIL, DRAG, RUN_PROCESS

## Database

The server uses SQLite with FTS5 for full-text search:
- Location: `~/.qontinui/mcp/qontinui.db`
- Auto-created on first run
- Supports fuzzy search with ranking

## License

MIT License
