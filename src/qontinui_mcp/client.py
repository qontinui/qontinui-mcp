"""HTTP client for Qontinui Runner API."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from qontinui_schemas.common import utc_now

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_PORT = 9876
DEFAULT_TIMEOUT = 30.0
EXECUTION_TIMEOUT = 300.0

# Results directory for QA feedback loop
# This is the same location used by qontinui-runner-mcp
# Paths are configurable via environment variables with WSL-style defaults
AUTOMATION_RESULTS_DIR = Path(
    os.environ.get(
        "QONTINUI_RESULTS_DIR",
        "/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.automation-results",
    )
)
DEV_LOGS_DIR = Path(
    os.environ.get(
        "QONTINUI_DEV_LOGS_DIR",
        "/mnt/c/Users/Joshua/Documents/qontinui_parent_directory/.dev-logs",
    )
)
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
            return "{}:\\{}".format(drive, rest.replace("/", "\\"))
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
    timestamp = utc_now().isoformat()

    execution_result: dict[str, Any] = {
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
            elif method == "PUT":
                response = await client.put(url, json=json_data, timeout=timeout)
            elif method == "DELETE":
                response = await client.delete(url, timeout=timeout)
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

        # If we don't have a local config cache, check if runner has one loaded
        if self._loaded_config is None:
            status_response = await self.status()
            if status_response.success and status_response.data:
                runner_has_config = status_response.data.get("config_loaded", False)
                if runner_has_config:
                    # Runner has a config loaded - we can proceed
                    # Set _loaded_config to empty dict to indicate config exists
                    # (we don't have the full config, but that's okay for execution)
                    self._loaded_config = {}
                    self._loaded_config_path = status_response.data.get("config_path")
                    logger.info(
                        f"Runner has config loaded at: {self._loaded_config_path}"
                    )
                else:
                    return ExecutionResult(
                        execution_id=execution_id,
                        success=False,
                        error="No configuration loaded. Use load_config first.",
                    )
            else:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=False,
                    error=f"Failed to check runner status: {status_response.error}",
                )

        request_data: dict[str, Any] = {"workflow_name": workflow_name}
        if monitor is not None:
            # Resolve monitor descriptor
            monitor_index = await self._resolve_monitor(monitor)
            logger.debug(f"Monitor resolution: '{monitor}' -> index {monitor_index}")
            if monitor_index is not None:
                request_data["monitor_index"] = monitor_index

        logger.debug(f"Sending run-workflow request: {request_data}")
        response = await self._request(
            "POST", "/run-workflow", request_data, timeout=timeout
        )
        logger.debug(f"run-workflow response success={response.success}")

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

    async def get_task_runs(self, status: str | None = None) -> RunnerResponse:
        """Get all task runs, optionally filtered by status.

        Args:
            status: Optional status filter ('running', 'complete', 'failed', 'stopped')
        """
        endpoint = "/task-runs"
        if status == "running":
            endpoint = "/task-runs/running"
        return await self._request("GET", endpoint)

    async def get_task_run(self, task_id: str) -> RunnerResponse:
        """Get a specific task run with full details including execution_steps_json.

        Args:
            task_id: The task run ID to retrieve.
        """
        return await self._request("GET", f"/task-runs/{task_id}")

    async def get_automation_runs(
        self, config_id: str | None = None, limit: int = 20
    ) -> RunnerResponse:
        """Get recent automation runs (from run_details table).

        Args:
            config_id: Optional config ID to filter by.
            limit: Maximum number of runs to return (default: 20).

        Returns:
            RunnerResponse with list of RunDetails objects containing:
            - actions_summary: Summary of actions executed
            - states_visited: List of state names visited
            - transitions_executed: List of transitions with timing
            - template_matches: List of template match results
            - anomalies: Any detected anomalies
        """
        params = []
        if config_id:
            params.append(f"config_id={config_id}")
        if limit != 20:
            params.append(f"limit={limit}")
        endpoint = "/runs"
        if params:
            endpoint += "?" + "&".join(params)
        return await self._request("GET", endpoint)

    async def get_automation_run(self, run_id: str) -> RunnerResponse:
        """Get a specific automation run with full details.

        Args:
            run_id: The automation run ID to retrieve.

        Returns:
            RunnerResponse with RunDetails object containing all run data.
        """
        return await self._request("GET", f"/runs/{run_id}")

    async def list_screenshots(self) -> RunnerResponse:
        """List available screenshots in the .dev-logs/screenshots directory."""
        # This reads files directly since the runner might not have a dedicated endpoint
        screenshots_dir = DEV_LOGS_DIR / "screenshots"
        if not screenshots_dir.exists():
            return RunnerResponse(success=True, data={"screenshots": [], "count": 0})

        screenshots = []
        for f in sorted(
            screenshots_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            screenshots.append(
                {
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )

        return RunnerResponse(
            success=True, data={"screenshots": screenshots, "count": len(screenshots)}
        )

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

    # -------------------------------------------------------------------------
    # Test API Methods
    # -------------------------------------------------------------------------

    async def list_tests(
        self,
        enabled_only: bool = False,
        test_type: str | None = None,
        category: str | None = None,
    ) -> RunnerResponse:
        """List all verification tests.

        Args:
            enabled_only: Only return enabled tests.
            test_type: Filter by test type (playwright_cdp, qontinui_vision, python_script, repository_test).
            category: Filter by category.

        Returns:
            RunnerResponse with list of tests.
        """
        params = []
        if enabled_only:
            params.append("enabled_only=true")
        if test_type:
            params.append(f"test_type={test_type}")
        if category:
            params.append(f"category={category}")

        endpoint = "/tests"
        if params:
            endpoint += "?" + "&".join(params)

        return await self._request("GET", endpoint)

    async def get_test(self, test_id: str) -> RunnerResponse:
        """Get a verification test by ID.

        Args:
            test_id: The test ID.

        Returns:
            RunnerResponse with test data.
        """
        return await self._request("GET", f"/tests/{test_id}")

    async def execute_test(
        self,
        test_id: str,
        task_run_id: str | None = None,
    ) -> RunnerResponse:
        """Execute a verification test by ID.

        Args:
            test_id: The test ID to execute.
            task_run_id: Optional task run ID to link results to.

        Returns:
            RunnerResponse with execution result.
        """
        return await self._request(
            "POST",
            f"/tests/{test_id}/execute",
            {"task_run_id": task_run_id},
            timeout=EXECUTION_TIMEOUT,
        )

    async def execute_test_suite(
        self,
        tests: list[dict[str, Any]],
        parallel: bool = False,
        stop_on_failure: bool = False,
    ) -> RunnerResponse:
        """Execute multiple tests as a suite.

        Args:
            tests: List of test definitions.
            parallel: Run tests in parallel.
            stop_on_failure: Stop on first failure.

        Returns:
            RunnerResponse with suite results.
        """
        return await self._request(
            "POST",
            "/tests/execute-suite",
            {
                "tests": tests,
                "parallel": parallel,
                "stop_on_failure": stop_on_failure,
            },
            timeout=EXECUTION_TIMEOUT,
        )

    async def list_test_results(
        self,
        test_id: str | None = None,
        task_run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> RunnerResponse:
        """List test results with optional filtering.

        Args:
            test_id: Filter by test ID.
            task_run_id: Filter by task run ID.
            status: Filter by status (passed, failed, error, timeout, skipped).
            limit: Maximum results to return.

        Returns:
            RunnerResponse with list of test results.
        """
        params = [f"limit={limit}"]
        if test_id:
            params.append(f"test_id={test_id}")
        if task_run_id:
            params.append(f"task_run_id={task_run_id}")
        if status:
            params.append(f"status={status}")

        endpoint = "/test-results?" + "&".join(params)
        return await self._request("GET", endpoint)

    async def get_test_result(self, result_id: str) -> RunnerResponse:
        """Get a specific test result by ID.

        Args:
            result_id: The result ID.

        Returns:
            RunnerResponse with test result.
        """
        return await self._request("GET", f"/test-results/{result_id}")

    async def get_test_history(
        self,
        test_id: str | None = None,
        limit: int = 1000,
    ) -> RunnerResponse:
        """Get test history summary with aggregated stats.

        Args:
            test_id: Optional test ID to filter history.
            limit: Maximum results to aggregate.

        Returns:
            RunnerResponse with history summary including pass rate, totals, etc.
        """
        params = [f"limit={limit}"]
        if test_id:
            params.append(f"test_id={test_id}")

        endpoint = "/tests/history?" + "&".join(params)
        return await self._request("GET", endpoint)

    async def create_test(
        self,
        name: str,
        test_type: str,
        description: str | None = None,
        category: str | None = None,
        playwright_code: str | None = None,
        python_code: str | None = None,
        repo_test_config: dict[str, Any] | None = None,
        timeout_seconds: int = 60,
        is_critical: bool = True,
        success_criteria: str | None = None,
        tags: list[str] | None = None,
    ) -> RunnerResponse:
        """Create a new verification test.

        Args:
            name: Human-readable test name.
            test_type: Type of test (playwright_cdp, qontinui_vision, python_script, repository_test).
            description: Description of what the test verifies.
            category: Test category (visual, dom, network, data, log, layout, unit, integration, custom).
            playwright_code: TypeScript/JavaScript code for playwright_cdp tests.
            python_code: Python code for python_script tests.
            repo_test_config: Configuration dict for repository_test (command, working_directory, parse_format).
            timeout_seconds: Test timeout in seconds.
            is_critical: If true, test failure fails the entire task.
            success_criteria: Natural language description of success criteria.
            tags: List of tags for organization.

        Returns:
            RunnerResponse with created test.
        """
        payload: dict[str, Any] = {
            "name": name,
            "test_type": test_type,
            "timeout_seconds": timeout_seconds,
            "is_critical": is_critical,
            "enabled": True,
            "ai_generated": True,  # Mark as AI-generated when created via MCP
            "tags": tags or [],
        }
        if description:
            payload["description"] = description
        if category:
            payload["category"] = category
        if playwright_code:
            payload["playwright_code"] = playwright_code
        if python_code:
            payload["python_code"] = python_code
        if repo_test_config:
            payload["repo_test_config"] = repo_test_config
        if success_criteria:
            payload["success_criteria"] = success_criteria

        return await self._request("POST", "/tests", payload)

    async def update_test(
        self,
        test_id: str,
        name: str | None = None,
        description: str | None = None,
        playwright_code: str | None = None,
        python_code: str | None = None,
        timeout_seconds: int | None = None,
        is_critical: bool | None = None,
        enabled: bool | None = None,
    ) -> RunnerResponse:
        """Update an existing verification test.

        Args:
            test_id: ID of the test to update.
            name: New test name.
            description: New description.
            playwright_code: New Playwright code.
            python_code: New Python code.
            timeout_seconds: New timeout in seconds.
            is_critical: Whether test failure fails the task.
            enabled: Whether test is enabled.

        Returns:
            RunnerResponse with updated test.
        """
        # First get the existing test to preserve fields
        existing = await self._request("GET", f"/tests/{test_id}")
        if not existing.success or not existing.data:
            return existing

        # Build update payload starting from existing data
        payload = existing.data.copy() if isinstance(existing.data, dict) else {}

        # Update provided fields
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if playwright_code is not None:
            payload["playwright_code"] = playwright_code
        if python_code is not None:
            payload["python_code"] = python_code
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if is_critical is not None:
            payload["is_critical"] = is_critical
        if enabled is not None:
            payload["enabled"] = enabled

        return await self._request("PUT", f"/tests/{test_id}", payload)

    async def delete_test(self, test_id: str) -> RunnerResponse:
        """Delete a verification test.

        Args:
            test_id: ID of the test to delete.

        Returns:
            RunnerResponse with deletion confirmation.
        """
        return await self._request("DELETE", f"/tests/{test_id}")

    # -------------------------------------------------------------------------
    # Log API Methods
    # -------------------------------------------------------------------------

    async def read_runner_logs(
        self,
        log_type: str = "all",
        limit: int = 100,
    ) -> RunnerResponse:
        """Read runner JSONL log files from .dev-logs directory.

        Args:
            log_type: Type of logs to read. Options:
                - 'general': General executor events (runner-general.jsonl)
                - 'actions': Tree/action events from workflow execution (runner-actions.jsonl)
                - 'image-recognition': Image recognition results with match details (runner-image-recognition.jsonl)
                - 'playwright': Playwright test execution results (runner-playwright.jsonl)
                - 'all': All log types combined
            limit: Maximum number of entries to return per log type (default: 100)

        Returns:
            RunnerResponse with log entries, or error if logs cannot be read.
        """
        log_files = {
            "general": "runner-general.jsonl",
            "actions": "runner-actions.jsonl",
            "image-recognition": "runner-image-recognition.jsonl",
            "playwright": "runner-playwright.jsonl",
        }

        if log_type == "all":
            types_to_read = list(log_files.keys())
        elif log_type in log_files:
            types_to_read = [log_type]
        else:
            return RunnerResponse(
                success=False,
                error=f"Unknown log type: {log_type}. Valid options: {', '.join(log_files.keys())}, all",
            )

        result: dict[str, Any] = {"logs": {}, "summary": {}}

        for ltype in types_to_read:
            log_file = DEV_LOGS_DIR / log_files[ltype]
            entries = []

            if log_file.exists():
                try:
                    with open(log_file, "r", errors="ignore") as f:
                        lines = f.readlines()
                        # Read last N lines (most recent entries)
                        recent_lines = lines[-limit:] if len(lines) > limit else lines
                        for line in recent_lines:
                            line = line.strip()
                            if line:
                                try:
                                    entries.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to read log file {log_file}: {e}")

            result["logs"][ltype] = entries
            result["summary"][ltype] = {
                "count": len(entries),
                "file_exists": log_file.exists(),
                "file_path": str(log_file),
            }

        return RunnerResponse(success=True, data=result)
