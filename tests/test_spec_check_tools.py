"""Tests for the spec-check MCP tools.

These tests verify that the four new MCP tools (check_page_spec,
list_page_specs, describe_page_spec, validate_page_spec) correctly:
1. Forward to the right QontinuiClient method with the right arguments
2. Hit the right HTTP endpoint with the right body / query string
3. Apply the summary-mode transform (drop the heavy `state_results`)
4. Return the serialized response body as TextContent

Contract facts verified during the Plan 03 vet (corrected vs. the plan's
illustrative examples):

- ``SpecCheckResult.summary.match_outcome`` ∈ {full_match, partial_match,
  no_match} — snake_case, NOT "Match".
- The heavy per-state field is ``state_results`` (NOT
  ``state_match_results``).
- ``recommended_state`` is nested under ``summary``, not top-level.
- ``validate_page_spec`` MUST post ``{"pageId": ...}`` (camelCase) — the
  runner's ``ValidateBody`` is ``#[serde(rename_all="camelCase")]`` with no
  snake_case alias; a ``page_id`` body silently 400s.
- ``/spec/validate`` returns a ``DistinctnessReport`` shaped
  ``{ok, violations:[{reason,...}]}`` — NOT ``{ok, reasons}``.

Since pytest-asyncio is not installed in this project, async coroutines
are driven via asyncio.run() (mirroring test_scenario_projection_tools.py).
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


def test_check_page_spec_is_read_only() -> None:
    assert TOOL_PERMISSIONS["check_page_spec"] is PermissionLevel.READ_ONLY


def test_list_page_specs_is_read_only() -> None:
    assert TOOL_PERMISSIONS["list_page_specs"] is PermissionLevel.READ_ONLY


def test_describe_page_spec_is_read_only() -> None:
    assert TOOL_PERMISSIONS["describe_page_spec"] is PermissionLevel.READ_ONLY


def test_validate_page_spec_is_read_only() -> None:
    assert TOOL_PERMISSIONS["validate_page_spec"] is PermissionLevel.READ_ONLY


# ---------------------------------------------------------------------------
# QontinuiClient routing — endpoints, methods, request bodies
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Minimal stand-in for an httpx.Response used by _raw_request."""

    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        return self._body


def _patch_raw_http(
    client: QontinuiClient,
    capture: dict[str, Any],
    body: Any,
    status_code: int = 200,
) -> Any:
    """Patch the client's underlying httpx client used by _raw_request.

    Captures the method, url and json body so tests can assert on them.
    """
    fake_http = MagicMock()

    async def fake_get(url: str, timeout: float = 30.0) -> _FakeHttpResponse:
        capture["method"] = "GET"
        capture["url"] = url
        capture["json"] = None
        return _FakeHttpResponse(body, status_code)

    async def fake_post(
        url: str, json: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> _FakeHttpResponse:
        capture["method"] = "POST"
        capture["url"] = url
        capture["json"] = json
        return _FakeHttpResponse(body, status_code)

    fake_http.get = AsyncMock(side_effect=fake_get)
    fake_http.post = AsyncMock(side_effect=fake_post)

    return patch.object(client, "_get_client", AsyncMock(return_value=fake_http))


def test_check_page_spec_posts_camel_case_page_id() -> None:
    """check_page_spec → POST /spec-check with {"pageId": ...}.

    The runner's /spec-check endpoint accepts camelCase since
    2026-05-17 spec-check remediation (matching /spec/derive and
    /spec/validate). The MCP tool's argument name is still snake_case
    by Python convention.
    """
    client = QontinuiClient(host="localhost", port=9876)
    capture: dict[str, Any] = {}
    with _patch_raw_http(client, capture, {"pageId": "settings-general"}):
        result = asyncio.run(client.check_page_spec("settings-general"))

    assert capture["method"] == "POST"
    assert capture["url"].endswith("/spec-check")
    assert capture["json"] == {"pageId": "settings-general"}
    assert result.success is True
    assert result.data == {"pageId": "settings-general"}


def test_list_page_specs_hits_spec_list() -> None:
    """list_page_specs → GET /spec/list (pure forwarder)."""
    client = QontinuiClient(host="localhost", port=9876)
    capture: dict[str, Any] = {}
    with _patch_raw_http(client, capture, {"specs": []}):
        result = asyncio.run(client.list_page_specs())

    assert capture["method"] == "GET"
    assert capture["url"].endswith("/spec/list")
    assert result.data == {"specs": []}


def test_describe_page_spec_url_encodes_id() -> None:
    """describe_page_spec → GET /spec/page/{quote(id, safe='')}."""
    client = QontinuiClient(host="localhost", port=9876)
    capture: dict[str, Any] = {}
    with _patch_raw_http(client, capture, {"version": 1}):
        asyncio.run(client.describe_page_spec("weird/id with spaces"))

    assert capture["method"] == "GET"
    # safe='' means '/' IS encoded (to %2F), spaces → %20
    assert capture["url"].endswith("/spec/page/weird%2Fid%20with%20spaces")


def test_validate_page_spec_posts_camel_case_pageid() -> None:
    """validate_page_spec MUST post {"pageId": ...} (camelCase), not page_id.

    The runner's ValidateBody is #[serde(rename_all="camelCase")] with no
    snake_case alias — a page_id body silently 400s.
    """
    client = QontinuiClient(host="localhost", port=9876)
    capture: dict[str, Any] = {}
    with _patch_raw_http(client, capture, {"ok": True, "violations": []}):
        result = asyncio.run(client.validate_page_spec("settings-general"))

    assert capture["method"] == "POST"
    assert capture["url"].endswith("/spec/validate")
    assert capture["json"] == {"pageId": "settings-general"}
    assert "page_id" not in (capture["json"] or {})
    assert result.data == {"ok": True, "violations": []}


def test_raw_request_surfaces_404_body_verbatim() -> None:
    """An un-spec'd page returns a non-2xx SpecError body — surface it."""
    client = QontinuiClient(host="localhost", port=9876)
    capture: dict[str, Any] = {}
    envelope = {"ok": False, "reason": "spec-not-found", "pageId": "nope"}
    with _patch_raw_http(client, capture, envelope, status_code=404):
        result = asyncio.run(client.check_page_spec("nope"))

    # success reflects HTTP status, but the structured body is preserved.
    assert result.success is False
    assert result.data == envelope


# ---------------------------------------------------------------------------
# server.call_tool dispatch
# ---------------------------------------------------------------------------


def _build_mock_client(
    check_response: RunnerResponse | None = None,
    list_response: RunnerResponse | None = None,
    describe_response: RunnerResponse | None = None,
    validate_response: RunnerResponse | None = None,
) -> MagicMock:
    """Build a MagicMock QontinuiClient with the four relevant methods mocked."""
    mock = MagicMock(spec=QontinuiClient)
    mock.check_page_spec = AsyncMock(
        return_value=check_response
        or RunnerResponse(success=True, data={"summary": {}})
    )
    mock.list_page_specs = AsyncMock(
        return_value=list_response or RunnerResponse(success=True, data={"specs": []})
    )
    mock.describe_page_spec = AsyncMock(
        return_value=describe_response
        or RunnerResponse(success=True, data={"version": 1})
    )
    mock.validate_page_spec = AsyncMock(
        return_value=validate_response
        or RunnerResponse(success=True, data={"ok": True, "violations": []})
    )
    return mock


_FULL_SPEC_CHECK_RESULT: dict[str, Any] = {
    "result_schema_version": 1,
    "page_id": "settings-general",
    "summary": {
        "match_outcome": "full_match",
        "overall_match_rate": 1.0,
        "recommended_state": {
            "state_id": "default",
            "confidence": "high",
            "reason": "all required elements present",
        },
    },
    # Heavy per-state breakdown — MUST be dropped in summary mode.
    "state_results": [
        {"state_id": "default", "match_rate": 1.0, "match_outcome": "full_match"}
    ],
    "warnings": [],
}


def test_call_tool_check_page_spec_summary_mode_strips_state_results() -> None:
    expected = RunnerResponse(success=True, data=dict(_FULL_SPEC_CHECK_RESULT))
    mock_client = _build_mock_client(check_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool(
                "check_page_spec",
                {"page_id": "settings-general", "mode": "summary"},
            )
        )

    mock_client.check_page_spec.assert_awaited_once_with("settings-general")

    assert len(result) == 1
    assert result[0].type == "text"
    body = json.loads(result[0].text)
    assert body["ok"] is True
    assert body["result_schema_version"] == 1
    assert body["page_id"] == "settings-general"
    assert "summary" in body
    # recommended_state is nested under summary (not top-level).
    assert body["summary"]["recommended_state"]["state_id"] == "default"
    assert body["summary"]["match_outcome"] == "full_match"
    # summary mode strips the heavy per-state breakdown.
    assert "state_results" not in body
    assert body["warnings"] == []


def test_call_tool_check_page_spec_defaults_to_summary_mode() -> None:
    """mode omitted → summary mode (state_results dropped)."""
    expected = RunnerResponse(success=True, data=dict(_FULL_SPEC_CHECK_RESULT))
    mock_client = _build_mock_client(check_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool("check_page_spec", {"page_id": "settings-general"})
        )

    body = json.loads(result[0].text)
    assert "state_results" not in body
    assert body["summary"]["recommended_state"] is not None


def test_call_tool_check_page_spec_full_mode_keeps_state_results() -> None:
    expected = RunnerResponse(success=True, data=dict(_FULL_SPEC_CHECK_RESULT))
    mock_client = _build_mock_client(check_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool(
                "check_page_spec",
                {"page_id": "settings-general", "mode": "full"},
            )
        )

    body = json.loads(result[0].text)
    # Full mode returns the entire SpecCheckResult unchanged.
    assert body == _FULL_SPEC_CHECK_RESULT
    assert body["state_results"][0]["match_outcome"] == "full_match"


def test_call_tool_check_page_spec_unknown_page_surfaces_envelope() -> None:
    """Unknown page_id → {ok:false, reason:"spec-not-found"} verbatim."""
    envelope = {
        "ok": False,
        "reason": "spec-not-found",
        "pageId": "no-such-page",
    }
    expected = RunnerResponse(success=False, data=envelope)
    mock_client = _build_mock_client(check_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("check_page_spec", {"page_id": "no-such-page"}))

    body = json.loads(result[0].text)
    # The MCP tool does NOT paper over the envelope.
    assert body == envelope


def test_call_tool_check_page_spec_missing_page_id_returns_error() -> None:
    mock_client = _build_mock_client()

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("check_page_spec", {}))

    mock_client.check_page_spec.assert_not_called()
    body = json.loads(result[0].text)
    assert body["success"] is False
    assert "page_id" in body["error"]


def test_call_tool_list_page_specs_is_pure_forwarder() -> None:
    expected = RunnerResponse(
        success=True,
        data={"specs": [{"id": "settings-general", "app": "qontinui-web"}]},
    )
    mock_client = _build_mock_client(list_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("list_page_specs", {}))

    mock_client.list_page_specs.assert_awaited_once_with()
    body = json.loads(result[0].text)
    # No client-side transformation.
    assert body == expected.data


def test_call_tool_describe_page_spec_forwards_id() -> None:
    expected = RunnerResponse(
        success=True,
        data={"version": 1, "description": "general settings", "groups": []},
    )
    mock_client = _build_mock_client(describe_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool("describe_page_spec", {"page_id": "weird/id with spaces"})
        )

    mock_client.describe_page_spec.assert_awaited_once_with("weird/id with spaces")
    body = json.loads(result[0].text)
    assert body == expected.data


def test_call_tool_describe_page_spec_missing_id_returns_error() -> None:
    mock_client = _build_mock_client()

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("describe_page_spec", {}))

    mock_client.describe_page_spec.assert_not_called()
    body = json.loads(result[0].text)
    assert body["success"] is False
    assert "page_id" in body["error"]


def test_call_tool_validate_page_spec_returns_distinctness_report() -> None:
    report = {
        "ok": False,
        "violations": [
            {"reason": "emptyCriteria", "stateId": "blank"},
            {
                "reason": "subsetDomination",
                "subsetStateId": "a",
                "supersetStateId": "b",
            },
        ],
    }
    expected = RunnerResponse(success=True, data=report)
    mock_client = _build_mock_client(validate_response=expected)

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(
            call_tool("validate_page_spec", {"page_id": "settings-general"})
        )

    mock_client.validate_page_spec.assert_awaited_once_with("settings-general")
    body = json.loads(result[0].text)
    # DistinctnessReport shape: {ok, violations:[{reason,...}]} — not reasons.
    assert body == report
    assert {v["reason"] for v in body["violations"]} == {
        "emptyCriteria",
        "subsetDomination",
    }


def test_call_tool_validate_page_spec_missing_id_returns_error() -> None:
    mock_client = _build_mock_client()

    with patch("qontinui_mcp.server.get_client", return_value=mock_client):
        result = asyncio.run(call_tool("validate_page_spec", {}))

    mock_client.validate_page_spec.assert_not_called()
    body = json.loads(result[0].text)
    assert body["success"] is False
    assert "page_id" in body["error"]


# ---------------------------------------------------------------------------
# Tools are present in the tool list with the right schemas
# ---------------------------------------------------------------------------


def test_spec_check_tools_are_registered() -> None:
    from qontinui_mcp.server import TOOLS

    names = {t.name for t in TOOLS}
    assert "check_page_spec" in names
    assert "list_page_specs" in names
    assert "describe_page_spec" in names
    assert "validate_page_spec" in names


def test_check_page_spec_schema() -> None:
    from qontinui_mcp.server import TOOLS

    by_name = {t.name: t for t in TOOLS}
    schema = by_name["check_page_spec"].inputSchema
    assert schema["required"] == ["page_id"]
    assert "page_id" in schema["properties"]
    assert schema["properties"]["mode"]["enum"] == ["summary", "full"]
    assert schema["properties"]["mode"]["default"] == "summary"


def test_describe_and_validate_require_page_id() -> None:
    from qontinui_mcp.server import TOOLS

    by_name = {t.name: t for t in TOOLS}
    assert by_name["describe_page_spec"].inputSchema["required"] == ["page_id"]
    assert by_name["validate_page_spec"].inputSchema["required"] == ["page_id"]
    # list_page_specs takes no arguments.
    assert by_name["list_page_specs"].inputSchema.get("properties") == {}
