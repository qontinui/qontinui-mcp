"""Tests for the scenario projection MCP tools.

These tests verify that the two new MCP tools (project_scenarios and
project_current_scenario) correctly:
1. Forward to the right QontinuiClient method with the right arguments
2. Hit the right HTTP endpoint with the right query string
3. Return the serialized RunnerResponse as TextContent

Since pytest-asyncio is not installed in this project, async coroutines
are driven via asyncio.run().
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from qontinui_mcp.client import QontinuiClient, RunnerResponse
from qontinui_mcp.permissions import (
    TOOL_PERMISSIONS,
    PermissionLevel,
)
from qontinui_mcp.server import call_tool

# ---------------------------------------------------------------------------
# Permission level assignment
# ---------------------------------------------------------------------------


def test_project_scenarios_is_read_only() -> None:
    assert TOOL_PERMISSIONS["project_scenarios"] is PermissionLevel.READ_ONLY


def test_project_current_scenario_is_read_only() -> None:
    assert TOOL_PERMISSIONS["project_current_scenario"] is PermissionLevel.READ_ONLY


# ---------------------------------------------------------------------------
# QontinuiClient.project_scenarios / project_current_scenario routing
# ---------------------------------------------------------------------------


def test_project_scenarios_calls_correct_endpoint() -> None:
    """Verify project_scenarios hits GET /scenarios/projection?ir_doc_id=..."""
    client = QontinuiClient(host="localhost", port=9876)

    async def fake_request(
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> RunnerResponse:
        assert method == "GET"
        assert endpoint == "/scenarios/projection?ir_doc_id=ir-123"
        assert json_data is None
        return RunnerResponse(success=True, data={"states": []})

    with patch.object(client, "_request", side_effect=fake_request):
        result = asyncio.run(client.project_scenarios("ir-123"))

    assert result.success is True
    assert result.data == {"states": []}


def test_project_current_scenario_calls_correct_endpoint() -> None:
    """Verify project_current_scenario hits GET /scenarios/current?ir_doc_id=..."""
    client = QontinuiClient(host="localhost", port=9876)

    async def fake_request(
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> RunnerResponse:
        assert method == "GET"
        assert endpoint == "/scenarios/current?ir_doc_id=ir-456"
        assert json_data is None
        return RunnerResponse(
            success=True, data={"active_states": ["s1"], "available": []}
        )

    with patch.object(client, "_request", side_effect=fake_request):
        result = asyncio.run(client.project_current_scenario("ir-456"))

    assert result.success is True
    assert result.data is not None
    assert result.data["active_states"] == ["s1"]


def test_project_scenarios_quotes_special_characters() -> None:
    """ir_doc_id with characters that need URL encoding must be quoted."""
    client = QontinuiClient(host="localhost", port=9876)
    seen_endpoint: dict[str, str] = {}

    async def fake_request(
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> RunnerResponse:
        seen_endpoint["endpoint"] = endpoint
        return RunnerResponse(success=True, data={})

    with patch.object(client, "_request", side_effect=fake_request):
        asyncio.run(client.project_scenarios("ir doc/with spaces&chars"))

    # Spaces must be percent-encoded; '/' and '&' too.
    assert seen_endpoint["endpoint"].startswith("/scenarios/projection?ir_doc_id=")
    # urllib.parse.quote default safe='/' so '/' is NOT encoded but ' ' and '&' are.
    assert "%20" in seen_endpoint["endpoint"]
    assert "%26" in seen_endpoint["endpoint"]


# ---------------------------------------------------------------------------
# server.call_tool dispatch
# ---------------------------------------------------------------------------


def _build_mock_client(
    project_scenarios_response: RunnerResponse | None = None,
    project_current_scenario_response: RunnerResponse | None = None,
) -> MagicMock:
    """Build a MagicMock QontinuiClient with the two relevant methods mocked."""
    mock = MagicMock(spec=QontinuiClient)
    mock.project_scenarios = AsyncMock(
        return_value=project_scenarios_response
        or RunnerResponse(success=True, data={"states": [], "transitions": []})
    )
    mock.project_current_scenario = AsyncMock(
        return_value=project_current_scenario_response
        or RunnerResponse(
            success=True, data={"active_states": [], "available": [], "blocked": []}
        )
    )
    return mock


def test_call_tool_project_scenarios_dispatches_correctly() -> None:
    expected = RunnerResponse(
        success=True,
        data={"states": [{"id": "home"}], "transitions": [{"id": "go-settings"}]},
    )
    mock_client = _build_mock_client(project_scenarios_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool("project_scenarios", {"ir_doc_id": "doc-1"})
        )

    mock_client.project_scenarios.assert_awaited_once_with("doc-1")
    mock_client.project_current_scenario.assert_not_called()

    assert len(result) == 1
    assert result[0].type == "text"
    payload = json.loads(result[0].text)
    assert payload["success"] is True
    assert payload["data"] == expected.data


def test_call_tool_project_current_scenario_dispatches_correctly() -> None:
    expected = RunnerResponse(
        success=True,
        data={
            "active_states": ["home"],
            "available": [{"transition_id": "go-settings"}],
            "blocked": [],
        },
    )
    mock_client = _build_mock_client(project_current_scenario_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool("project_current_scenario", {"ir_doc_id": "doc-2"})
        )

    mock_client.project_current_scenario.assert_awaited_once_with("doc-2")
    mock_client.project_scenarios.assert_not_called()

    assert len(result) == 1
    assert result[0].type == "text"
    payload = json.loads(result[0].text)
    assert payload["success"] is True
    assert payload["data"] == expected.data


def test_call_tool_project_scenarios_missing_ir_doc_id_returns_error() -> None:
    mock_client = _build_mock_client()

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("project_scenarios", {}))

    # Should NOT have called the client method
    mock_client.project_scenarios.assert_not_called()

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["success"] is False
    assert "ir_doc_id" in payload["error"]


def test_call_tool_project_current_scenario_missing_ir_doc_id_returns_error() -> None:
    mock_client = _build_mock_client()

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("project_current_scenario", {}))

    mock_client.project_current_scenario.assert_not_called()

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["success"] is False
    assert "ir_doc_id" in payload["error"]


# ---------------------------------------------------------------------------
# Tools are present in the tool list
# ---------------------------------------------------------------------------


def test_tools_are_registered() -> None:
    from qontinui_mcp.server import TOOLS

    names = {t.name for t in TOOLS}
    assert "project_scenarios" in names
    assert "project_current_scenario" in names


def test_tool_schemas_require_ir_doc_id() -> None:
    from qontinui_mcp.server import TOOLS

    by_name = {t.name: t for t in TOOLS}

    proj = by_name["project_scenarios"]
    assert proj.inputSchema["required"] == ["ir_doc_id"]
    assert "ir_doc_id" in proj.inputSchema["properties"]

    cur = by_name["project_current_scenario"]
    assert cur.inputSchema["required"] == ["ir_doc_id"]
    assert "ir_doc_id" in cur.inputSchema["properties"]
