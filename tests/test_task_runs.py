"""Tests for the /task-runs and /task-runs/running response-shape handling.

Context (2026-08-29): `GET /task-runs/running` on the runner changed from a
bare JSON array to an envelope: ``{"scope": "...", "task_runs": [...]}``. The
`scope` string exists because an operator once read this endpoint's `[]` and
nearly restarted a runner holding 23 live agent sessions -- the endpoint is a
port-filtered *workflow task-run ledger*, not a session census.

These tests cover:

1. ``QontinuiClient._request``'s shape handling -- it must not assume every
   response body is the ``{success, data, error}`` envelope, and must never
   silently produce ``RunnerResponse(success=False, data=None, error=None)``
   for a response that parsed without an exception.
2. ``QontinuiClient.get_task_runs`` -- both the `status="running"` path (new
   envelope) and the default path (`/task-runs`, which stays the plain
   `{success, data: [...]}` shape) must normalize to the same caller-facing
   shape: ``{"task_runs": [...], "scope": str | None}``.
3. The `get_task_runs` MCP tool (server.call_tool) -- it must surface the
   `scope` string to the calling AI agent rather than parsing it out and
   dropping it.

Since pytest-asyncio is not installed in this project, async coroutines are
driven via asyncio.run() (mirroring the other test modules here).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from qontinui_mcp.client import QontinuiClient, RunnerResponse
from qontinui_mcp.server import call_tool

# ---------------------------------------------------------------------------
# _request shape handling
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Minimal stand-in for an httpx.Response used by _request."""

    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")

    def json(self) -> Any:
        return self._body


def _patch_get(client: QontinuiClient, body: Any, status_code: int = 200) -> Any:
    fake_http = MagicMock()

    async def fake_get(url: str, timeout: float = 30.0) -> _FakeHttpResponse:
        return _FakeHttpResponse(body, status_code)

    fake_http.get = AsyncMock(side_effect=fake_get)
    return patch.object(client, "_get_client", AsyncMock(return_value=fake_http))


def test_request_unwraps_standard_envelope() -> None:
    """A body with a top-level "success" key is unwrapped as before."""
    client = QontinuiClient(host="localhost", port=9876)
    with _patch_get(client, {"success": True, "data": [1, 2, 3], "error": None}):
        result = asyncio.run(client._request("GET", "/task-runs"))

    assert result.success is True
    # `data` stays typed dict[str, Any] | None, but the runtime
    # value here is the raw list from the envelope's "data" field
    # (pre-existing behavior of the envelope-unwrap branch) --
    # cast for the comparison rather than widen the field type.
    assert cast(Any, result.data) == [1, 2, 3]
    assert result.error is None


def test_request_treats_dict_without_success_key_as_payload() -> None:
    """A dict body with no "success" key (the new /task-runs/running shape)
    is treated as the payload itself, not fabricated into a blank failure.
    """
    client = QontinuiClient(host="localhost", port=9876)
    body = {
        "scope": "workflow task-runs on API port 9876; NOT a session census",
        "task_runs": [{"id": "t1", "status": "running"}],
    }
    with _patch_get(client, body):
        result = asyncio.run(client._request("GET", "/task-runs/running"))

    assert result.success is True
    assert result.data == body
    assert result.error is None


def test_request_wraps_bare_list_body() -> None:
    """A bare JSON array body is wrapped so `data` stays a dict, and is
    never turned into an unexplained failure.
    """
    client = QontinuiClient(host="localhost", port=9876)
    with _patch_get(client, [{"id": "t1"}]):
        result = asyncio.run(client._request("GET", "/task-runs/running"))

    assert result.success is True
    assert result.data == {"result": [{"id": "t1"}]}
    assert result.error is None


def test_request_never_produces_blank_failure_for_non_envelope_body() -> None:
    """Regression guard for the exact regression described in the plan:
    dict.get("success", False) silently returning False with no exception
    and no error message, discarding the real payload.
    """
    client = QontinuiClient(host="localhost", port=9876)
    bodies: list[Any] = [
        {"scope": "x", "task_runs": []},
        [],
        {"anything": "else"},
    ]
    for body in bodies:
        with _patch_get(client, body):
            result = asyncio.run(client._request("GET", "/task-runs/running"))
        assert not (
            result.success is False and result.data is None and result.error is None
        ), f"blank failure produced for body={body!r}"


# ---------------------------------------------------------------------------
# QontinuiClient.get_task_runs normalization
# ---------------------------------------------------------------------------


def _fake_request_returning(
    endpoint_to_response: dict[str, RunnerResponse],
) -> Any:
    async def fake_request(
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> RunnerResponse:
        assert method == "GET"
        assert endpoint in endpoint_to_response, f"unexpected endpoint {endpoint}"
        return endpoint_to_response[endpoint]

    return fake_request


def test_get_task_runs_running_surfaces_scope_and_task_runs() -> None:
    """status="running" hits /task-runs/running and normalizes the new
    {"scope", "task_runs"} envelope into the caller-facing shape.
    """
    client = QontinuiClient(host="localhost", port=9876)
    scope_text = "workflow task-runs on API port 9876; NOT a session census"
    fake = _fake_request_returning(
        {
            "/task-runs/running": RunnerResponse(
                success=True,
                data={
                    "scope": scope_text,
                    "task_runs": [{"id": "t1", "status": "running"}],
                },
            )
        }
    )
    with patch.object(client, "_request", side_effect=fake):
        result = asyncio.run(client.get_task_runs(status="running"))

    assert result.success is True
    assert result.data is not None
    assert result.data["task_runs"] == [{"id": "t1", "status": "running"}]
    assert result.data["scope"] == scope_text


def test_get_task_runs_running_empty_list_still_carries_scope() -> None:
    """The exact incident scenario: an empty task_runs list must still carry
    the scope caution rather than looking like an unqualified "idle" signal.
    """
    client = QontinuiClient(host="localhost", port=9876)
    scope_text = "workflow task-runs on API port 9876; NOT a session census"
    fake = _fake_request_returning(
        {
            "/task-runs/running": RunnerResponse(
                success=True, data={"scope": scope_text, "task_runs": []}
            )
        }
    )
    with patch.object(client, "_request", side_effect=fake):
        result = asyncio.run(client.get_task_runs(status="running"))

    assert result.success is True
    assert result.data is not None
    assert result.data["task_runs"] == []
    assert result.data["scope"] == scope_text


def test_get_task_runs_default_status_hits_plain_task_runs_endpoint() -> None:
    """status=None hits /task-runs, whose {success, data: [...]} shape has
    no scope -- the caller-facing shape must still be coherent with the
    "running" path: {"task_runs": [...], "scope": None}.
    """
    client = QontinuiClient(host="localhost", port=9876)
    fake = _fake_request_returning(
        {
            "/task-runs": RunnerResponse(
                success=True,
                data=cast(Any, [{"id": "t2", "status": "complete"}]),
            )
        }
    )
    with patch.object(client, "_request", side_effect=fake):
        result = asyncio.run(client.get_task_runs())

    assert result.success is True
    assert result.data == {
        "task_runs": [{"id": "t2", "status": "complete"}],
        "scope": None,
    }


def test_get_task_runs_non_running_status_also_hits_plain_endpoint() -> None:
    """Any status other than "running" routes to /task-runs (the runner has
    no server-side status filter there), normalized the same way.
    """
    client = QontinuiClient(host="localhost", port=9876)
    fake = _fake_request_returning(
        {"/task-runs": RunnerResponse(success=True, data=cast(Any, []))}
    )
    with patch.object(client, "_request", side_effect=fake):
        result = asyncio.run(client.get_task_runs(status="complete"))

    assert result.data == {"task_runs": [], "scope": None}


def test_get_task_runs_passes_through_request_failure() -> None:
    """A transport failure must not be swallowed into a normalized-but-empty
    success -- the original error is returned untouched.
    """
    client = QontinuiClient(host="localhost", port=9876)
    fake = _fake_request_returning(
        {
            "/task-runs/running": RunnerResponse(
                success=False, error="Cannot connect to runner"
            )
        }
    )
    with patch.object(client, "_request", side_effect=fake):
        result = asyncio.run(client.get_task_runs(status="running"))

    assert result.success is False
    assert result.error == "Cannot connect to runner"
    assert result.data is None


# ---------------------------------------------------------------------------
# server.call_tool("get_task_runs", ...) surfaces scope
# ---------------------------------------------------------------------------


def _build_mock_client(get_task_runs_response: RunnerResponse) -> MagicMock:
    mock = MagicMock(spec=QontinuiClient)
    mock.get_task_runs = AsyncMock(return_value=get_task_runs_response)
    return mock


def test_call_tool_get_task_runs_surfaces_scope_when_present() -> None:
    scope_text = "workflow task-runs on API port 9876; NOT a session census"
    mock_client = _build_mock_client(
        RunnerResponse(
            success=True, data={"task_runs": [], "scope": scope_text}
        )
    )

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("get_task_runs", {"status": "running"}))

    mock_client.get_task_runs.assert_awaited_once_with(status="running")
    body = json.loads(result[0].text)
    assert body["success"] is True
    assert body["task_runs"] == []
    assert body["scope"] == scope_text


def test_call_tool_get_task_runs_omits_scope_when_absent() -> None:
    mock_client = _build_mock_client(
        RunnerResponse(
            success=True,
            data={"task_runs": [{"id": "t1", "status": "complete"}], "scope": None},
        )
    )

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("get_task_runs", {}))

    body = json.loads(result[0].text)
    assert body["success"] is True
    assert body["task_runs"] == [{"id": "t1", "status": "complete"}]
    assert "scope" not in body


def test_call_tool_get_task_runs_surfaces_error_with_no_data() -> None:
    mock_client = _build_mock_client(
        RunnerResponse(success=False, error="Cannot connect to runner")
    )

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("get_task_runs", {"status": "running"}))

    body = json.loads(result[0].text)
    assert body["success"] is False
    assert body["error"] == "Cannot connect to runner"
    assert "task_runs" not in body
    assert "scope" not in body
