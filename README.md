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

## Available Tools

| Tool | Description |
|------|-------------|
| `get_executor_status` | Get runner status |
| `list_monitors` | List available monitors |
| `load_config` | Load a workflow configuration file |
| `ensure_config_loaded` | Load config if not already loaded |
| `get_loaded_config` | Get loaded configuration info |
| `run_workflow` | Run a workflow by name |
| `stop_execution` | Stop current execution |

## Example Usage

```python
# In an AI conversation:
"Load the config at /path/to/workflow.json and run the 'login_test' workflow on the left monitor"
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
```

## License

MIT
