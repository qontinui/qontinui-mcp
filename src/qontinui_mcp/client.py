"""HTTP client for Qontinui Runner API."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_PORT = 9876
DEFAULT_TIMEOUT = 30.0
EXECUTION_TIMEOUT = 300.0

# Results directory for QA feedback loop
# This is the same location used by qontinui-runner-mcp
AUTOMATION_RESULTS_DIR = Path(
    "/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.automation-results"
)
DEV_LOGS_DIR = Path("/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.dev-logs")
MAX_HISTORY_RUNS = 10


def get_windows_host() -> str:
    """Get the Windows host IP address from WSL.

    In WSL2, the Windows host is accessible via the IP in /etc/resolv.conf.
    Falls back to localhost for native Windows/Mac/Linux.
    """
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except (FileNotFoundError, IndexError):
        pass
    return "localhost"


def convert_wsl_path(wsl_path: str) -> str:
    """Convert a WSL path to a Windows path.

    /mnt/c/Users/... -> C:\\Users\\...
    """
    if wsl_path.startswith("/mnt/"):
        parts = wsl_path.split("/")
        if len(parts) >= 3:
            drive = parts[2].upper()
            rest = "/".join(parts[3:])
            return f"{drive}:\\{rest.replace('/', '\\')}"
    return wsl_path


def _save_automation_results(
    execution_id: str,
    config_path: str,
    workflow_name: str,
    success: bool,
    duration_ms: int,
    error: str | None,
    events: list[dict[str, Any]],
    monitor: int | str | None = None,
) -> Path:
    """Save automation results to filesystem for QA feedback loop.

    This mirrors the functionality in qontinui-runner-mcp to ensure
    execution.json is always created for the /analyze-automation command.
    """
    latest_dir = AUTOMATION_RESULTS_DIR / "latest"
    history_dir = AUTOMATION_RESULTS_DIR / "history"
    latest_logs_dir = latest_dir / "logs"
    latest_screenshots_dir = latest_dir / "screenshots"

    for d in [latest_dir, history_dir, latest_logs_dir, latest_screenshots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Archive previous latest to history (if exists)
    existing_execution_file = latest_dir / "execution.json"
    if existing_execution_file.exists():
        try:
            with open(existing_execution_file) as f:
                prev_data = json.load(f)
                prev_id = prev_data.get("execution_id", "unknown")
                prev_timestamp = (
                    prev_data.get("timestamp", "unknown")
                    .replace(":", "-")
                    .replace(".", "-")
                )

            history_entry_name = f"{prev_timestamp}_{prev_id[:8]}"
            history_entry_dir = history_dir / history_entry_name

            if not history_entry_dir.exists():
                shutil.copytree(latest_dir, history_entry_dir)
                logger.info(f"Archived previous run to history: {history_entry_name}")

            # Clean up old history entries
            entries = sorted(
                [d for d in history_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_entry in entries[MAX_HISTORY_RUNS:]:
                try:
                    shutil.rmtree(old_entry)
                except Exception as e:
                    logger.warning(f"Failed to remove old history entry: {e}")
        except Exception as e:
            logger.warning(f"Failed to archive previous results: {e}")

    # Clear latest directory
    for item in latest_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Recreate subdirectories
    latest_logs_dir.mkdir(exist_ok=True)
    latest_screenshots_dir.mkdir(exist_ok=True)

    # Capture log snapshots from .dev-logs
    log_files = ["backend.log", "frontend.log", "qontinui-api.log", "runner.log"]
    for log_file in log_files:
        src_log = DEV_LOGS_DIR / log_file
        if src_log.exists():
            try:
                with open(src_log, "r", errors="ignore") as f:
                    lines = f.readlines()
                    last_lines = lines[-500:] if len(lines) > 500 else lines

                dst_log = latest_logs_dir / log_file
                with open(dst_log, "w") as f:
                    f.writelines(last_lines)
            except Exception as e:
                logger.warning(f"Failed to capture log {log_file}: {e}")

    # Copy AI output log (complete file, not truncated)
    ai_output_log = DEV_LOGS_DIR / "ai-output.jsonl"
    if ai_output_log.exists():
        try:
            shutil.copy2(ai_output_log, latest_logs_dir / "ai-output.jsonl")
            logger.info("Copied AI output log to automation results")
        except Exception as e:
            logger.warning(f"Failed to copy AI output log: {e}")

    # Build execution results JSON
    timestamp = datetime.now().isoformat()

    execution_result = {
        "execution_id": execution_id,
        "config_path": config_path,
        "workflow_name": workflow_name,
        "monitor": monitor,
        "success": success,
        "duration_ms": duration_ms,
        "timestamp": timestamp,
        "error": error,
        "summary": {
            "total_events": len(events),
            "test_results_count": 0,
            "console_errors_count": 0,
            "network_failures_count": 0,
        },
        "test_results": [],
        "console_errors": [],
        "network_failures": [],
        "screenshots": [],
        "log_snapshots": {
            "backend": (
                str(latest_logs_dir / "backend.log")
                if (latest_logs_dir / "backend.log").exists()
                else None
            ),
            "frontend": (
                str(latest_logs_dir / "frontend.log")
                if (latest_logs_dir / "frontend.log").exists()
                else None
            ),
            "api": (
                str(latest_logs_dir / "qontinui-api.log")
                if (latest_logs_dir / "qontinui-api.log").exists()
                else None
            ),
            "runner": (
                str(latest_logs_dir / "runner.log")
                if (latest_logs_dir / "runner.log").exists()
                else None
            ),
            "ai_output": (
                str(latest_logs_dir / "ai-output.jsonl")
                if (latest_logs_dir / "ai-output.jsonl").exists()
                else None
            ),
        },
    }

    # Write execution.json
    execution_file = latest_dir / "execution.json"
    with open(execution_file, "w") as f:
        json.dump(execution_result, f, indent=2)

    logger.info(f"Saved automation results to {execution_file}")
    return execution_file


@dataclass
class RunnerResponse:
    """Response from the Qontinui Runner API."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ExecutionResult:
    """Result of a workflow execution."""

    execution_id: str
    success: bool
    duration_ms: int = 0
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class QontinuiClient:
    """HTTP client for the Qontinui Runner API.

    This is a lightweight client that forwards requests to the runner.
    All heavy lifting (execution, vision, etc.) happens in the runner.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = DEFAULT_RUNNER_PORT,
    ) -> None:
        """Initialize the client.

        Args:
            host: Runner host. Auto-detected from WSL if None.
            port: Runner port. Defaults to 9876.
        """
        self.host = host or os.environ.get("QONTINUI_RUNNER_HOST") or get_windows_host()
        self.port = int(os.environ.get("QONTINUI_RUNNER_PORT", port))
        self.base_url = f"http://{self.host}:{self.port}"
        self._client: httpx.AsyncClient | None = None
        self._loaded_config_path: str | None = None
        self._loaded_config: dict[str, Any] | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> RunnerResponse:
        """Make an HTTP request to the runner API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = await client.get(url, timeout=timeout)
            elif method == "POST":
                response = await client.post(url, json=json_data, timeout=timeout)
            else:
                return RunnerResponse(
                    success=False, error=f"Unsupported method: {method}"
                )

            response.raise_for_status()
            data = response.json()
            return RunnerResponse(
                success=data.get("success", False),
                data=data.get("data"),
                error=data.get("error"),
            )
        except httpx.ConnectError as e:
            return RunnerResponse(
                success=False,
                error=f"Cannot connect to runner at {url}. Is qontinui-runner running? Error: {e}",
            )
        except httpx.HTTPStatusError as e:
            return RunnerResponse(
                success=False,
                error=f"Runner API error: {e.response.status_code} - {e.response.text}",
            )
        except Exception as e:
            return RunnerResponse(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Core API Methods
    # -------------------------------------------------------------------------

    async def health(self) -> RunnerResponse:
        """Check runner health."""
        return await self._request("GET", "/health")

    async def status(self) -> RunnerResponse:
        """Get runner status."""
        return await self._request("GET", "/status")

    async def load_config(self, config_path: str) -> RunnerResponse:
        """Load a workflow configuration file.

        Args:
            config_path: Path to the JSON config file (WSL or Windows path).
        """
        path = Path(config_path)
        if not path.exists():
            return RunnerResponse(
                success=False, error=f"Config file not found: {config_path}"
            )

        # Cache the config locally
        try:
            with open(path) as f:
                self._loaded_config = json.load(f)
            self._loaded_config_path = str(path.resolve())
        except json.JSONDecodeError as e:
            return RunnerResponse(success=False, error=f"Invalid JSON: {e}")

        # Convert to Windows path if needed
        windows_path = convert_wsl_path(str(path.resolve()))
        return await self._request(
            "POST", "/load-config", {"config_path": windows_path}
        )

    async def run_workflow(
        self,
        workflow_name: str,
        monitor: int | str | None = None,
        timeout: float = EXECUTION_TIMEOUT,
    ) -> ExecutionResult:
        """Run a workflow by name.

        Args:
            workflow_name: Name of the workflow to run.
            monitor: Monitor to run on ('left', 'right', 'primary', or index).
            timeout: Execution timeout in seconds.
        """
        import uuid

        execution_id = str(uuid.uuid4())

        if self._loaded_config is None:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                error="No configuration loaded. Use load_config first.",
            )

        request_data: dict[str, Any] = {"workflow_name": workflow_name}
        if monitor is not None:
            # Resolve monitor descriptor
            monitor_index = await self._resolve_monitor(monitor)
            print(
                f"[MCP_CLIENT] Monitor resolution: '{monitor}' -> index {monitor_index}"
            )
            if monitor_index is not None:
                request_data["monitor_index"] = monitor_index

        print(f"[MCP_CLIENT] Sending run-workflow request: {request_data}")
        response = await self._request(
            "POST", "/run-workflow", request_data, timeout=timeout
        )
        print(f"[MCP_CLIENT] run-workflow response success={response.success}")

        if response.success and response.data:
            result = ExecutionResult(
                execution_id=execution_id,
                success=response.data.get("success", False),
                duration_ms=response.data.get("duration_ms", 0),
                error=response.data.get("error"),
                events=response.data.get("events", []),
            )
        else:
            result = ExecutionResult(
                execution_id=execution_id,
                success=False,
                error=response.error or "Unknown error",
            )

        # Save execution results for QA feedback loop
        try:
            _save_automation_results(
                execution_id=execution_id,
                config_path=self._loaded_config_path or "unknown",
                workflow_name=workflow_name,
                success=result.success,
                duration_ms=result.duration_ms,
                error=result.error,
                events=result.events,
                monitor=monitor,
            )
        except Exception as e:
            logger.warning(f"Failed to save automation results: {e}")

        return result

    async def stop_execution(self) -> RunnerResponse:
        """Stop the current workflow execution."""
        return await self._request("POST", "/stop-execution")

    async def list_monitors(self) -> RunnerResponse:
        """List available monitors."""
        return await self._request("GET", "/monitors")

    async def _resolve_monitor(self, monitor: int | str) -> int | None:
        """Resolve a monitor descriptor to an index."""
        if isinstance(monitor, int):
            return monitor

        # Get monitor list from runner
        response = await self.list_monitors()
        if not response.success or not response.data:
            return None

        monitors = response.data.get("monitors", [])
        monitor_lower = monitor.lower()

        # Match by position - return the monitor's actual index, not the list position
        for m in monitors:
            if m.get("position", "").lower() == monitor_lower:
                index = m.get("index")
                return int(index) if index is not None else None
            if monitor_lower == "primary" and m.get("is_primary"):
                index = m.get("index")
                return int(index) if index is not None else None

        # Try parsing as int
        try:
            return int(monitor)
        except ValueError:
            return None

    def get_loaded_config_info(self) -> dict[str, Any]:
        """Get information about the currently loaded config."""
        if self._loaded_config_path is None:
            return {"loaded": False, "config_path": None, "workflows": []}

        workflows = []
        if self._loaded_config:
            if "workflows" in self._loaded_config:
                workflows = [
                    {"name": w.get("name", f"workflow_{i}"), "id": w.get("id")}
                    for i, w in enumerate(self._loaded_config.get("workflows", []))
                ]
            elif "name" in self._loaded_config:
                workflows = [{"name": self._loaded_config.get("name")}]

        return {
            "loaded": True,
            "config_path": self._loaded_config_path,
            "workflows": workflows,
        }

    def is_config_loaded(self, config_path: str) -> bool:
        """Check if a specific config is loaded (local cache only).

        WARNING: This only checks the local cache, NOT the runner's actual state.
        Use verify_config_loaded() for a reliable check that queries the runner.
        """
        if self._loaded_config_path is None:
            return False
        return Path(self._loaded_config_path).resolve() == Path(config_path).resolve()

    async def verify_config_loaded(self, config_path: str) -> bool:
        """Verify that the runner actually has a config loaded.

        This queries the runner's /status endpoint to check if config_loaded is true.
        More reliable than is_config_loaded() which only checks local cache.

        Args:
            config_path: Path to verify is loaded.

        Returns:
            True if local cache matches AND runner confirms config is loaded.
        """
        # First check local cache
        if not self.is_config_loaded(config_path):
            return False

        # Then verify with runner
        response = await self.status()
        if not response.success or not response.data:
            # Can't verify, assume not loaded
            return False

        config_loaded = response.data.get("config_loaded", False)
        return bool(config_loaded)
