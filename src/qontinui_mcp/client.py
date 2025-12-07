"""HTTP client for Qontinui Runner API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_PORT = 9876
DEFAULT_TIMEOUT = 30.0
EXECUTION_TIMEOUT = 300.0


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
            return ExecutionResult(
                execution_id=execution_id,
                success=response.data.get("success", False),
                duration_ms=response.data.get("duration_ms", 0),
                error=response.data.get("error"),
                events=response.data.get("events", []),
            )
        else:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                error=response.error or "Unknown error",
            )

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
                return m.get("index")
            if monitor_lower == "primary" and m.get("is_primary"):
                return m.get("index")

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

        return response.data.get("config_loaded", False)
