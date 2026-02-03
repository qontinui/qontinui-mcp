"""Permission system for MCP tool calls.

This module provides a permission layer that wraps tool calls with
authorization checks. Inspired by OpenCode's permission system.

Permission Levels:
- READ_ONLY: Safe operations that only read data (logs, status, screenshots)
- EXECUTE: Operations that run workflows or tests (may affect external systems)
- MODIFY: Operations that change data (create/update/delete tests, configs)
- DANGEROUS: Operations that can disrupt execution (stop, restart)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for tool operations."""

    READ_ONLY = "read_only"
    """Read-only operations: logs, status, screenshots, list queries."""

    EXECUTE = "execute"
    """Execution operations: run workflows, execute tests, capture screenshots."""

    MODIFY = "modify"
    """Modification operations: create/update/delete tests, load configs."""

    DANGEROUS = "dangerous"
    """Dangerous operations: stop execution, restart runner."""


@dataclass
class PermissionRequest:
    """A request for permission to execute a tool."""

    tool_name: str
    """Name of the tool being called."""

    level: PermissionLevel
    """Required permission level for this tool."""

    description: str
    """Human-readable description of the operation."""

    params: dict[str, Any]
    """Parameters being passed to the tool."""


@dataclass
class PermissionService:
    """Service for managing tool permissions.

    The permission service controls which tools can be executed based on
    permission levels. It supports:

    - Auto-approving safe (READ_ONLY) operations
    - Tracking granted permissions for the session
    - Requesting user approval for sensitive operations
    - Custom permission handlers for UI integration
    """

    auto_approve_levels: set[PermissionLevel] = field(
        default_factory=lambda: {PermissionLevel.READ_ONLY}
    )
    """Permission levels that are automatically approved without user interaction."""

    granted_permissions: list[PermissionRequest] = field(default_factory=list)
    """Permissions that have been granted during this session."""

    on_request: Callable[[PermissionRequest], bool] | None = None
    """Optional synchronous callback for permission requests."""

    on_request_async: Callable[[PermissionRequest], Any] | None = None
    """Optional async callback for permission requests (returns awaitable)."""

    _deny_by_default: bool = False
    """If True, deny all non-auto-approved permissions by default."""

    def configure(
        self,
        auto_approve_levels: set[PermissionLevel] | None = None,
        deny_by_default: bool | None = None,
    ) -> None:
        """Configure permission settings.

        Args:
            auto_approve_levels: Set of levels to auto-approve (default: READ_ONLY)
            deny_by_default: If True, deny non-auto-approved without handler
        """
        if auto_approve_levels is not None:
            self.auto_approve_levels = auto_approve_levels
        if deny_by_default is not None:
            self._deny_by_default = deny_by_default

    def auto_approve_all(self) -> None:
        """Auto-approve all permission levels (useful for trusted contexts)."""
        self.auto_approve_levels = {
            PermissionLevel.READ_ONLY,
            PermissionLevel.EXECUTE,
            PermissionLevel.MODIFY,
            PermissionLevel.DANGEROUS,
        }

    def reset_granted(self) -> None:
        """Clear all granted permissions (useful for new sessions)."""
        self.granted_permissions.clear()

    async def request(self, req: PermissionRequest) -> bool:
        """Request permission for a tool operation.

        Args:
            req: The permission request

        Returns:
            True if permission is granted, False otherwise
        """
        # Auto-approve based on permission level
        if req.level in self.auto_approve_levels:
            logger.debug(f"Auto-approved {req.tool_name} (level={req.level.value})")
            return True

        # Check if already granted this session (same tool + level)
        for granted in self.granted_permissions:
            if granted.tool_name == req.tool_name and granted.level == req.level:
                logger.debug(
                    f"Already granted {req.tool_name} (level={req.level.value})"
                )
                return True

        # Try async handler first
        if self.on_request_async is not None:
            try:
                result = self.on_request_async(req)
                if asyncio.iscoroutine(result):
                    approved: bool = await result
                else:
                    approved = bool(result)
                if approved:
                    self.granted_permissions.append(req)
                    logger.info(
                        f"Permission granted: {req.tool_name} ({req.level.value})"
                    )
                else:
                    logger.warning(
                        f"Permission denied: {req.tool_name} ({req.level.value})"
                    )
                return approved
            except Exception as e:
                logger.exception(f"Error in async permission handler: {e}")
                return False

        # Try sync handler
        if self.on_request is not None:
            try:
                approved = self.on_request(req)
                if approved:
                    self.granted_permissions.append(req)
                    logger.info(
                        f"Permission granted: {req.tool_name} ({req.level.value})"
                    )
                else:
                    logger.warning(
                        f"Permission denied: {req.tool_name} ({req.level.value})"
                    )
                return approved
            except Exception as e:
                logger.exception(f"Error in permission handler: {e}")
                return False

        # No handler - check deny_by_default
        if self._deny_by_default:
            logger.warning(
                f"Permission denied (no handler, deny_by_default): "
                f"{req.tool_name} ({req.level.value})"
            )
            return False

        # Default: approve if no handler (for backwards compatibility)
        logger.info(
            f"Permission auto-granted (no handler): {req.tool_name} ({req.level.value})"
        )
        self.granted_permissions.append(req)
        return True


# Tool permission mappings
TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    # READ_ONLY tools - safe operations that only read data
    "get_executor_status": PermissionLevel.READ_ONLY,
    "list_monitors": PermissionLevel.READ_ONLY,
    "get_loaded_config": PermissionLevel.READ_ONLY,
    "get_task_runs": PermissionLevel.READ_ONLY,
    "get_task_run": PermissionLevel.READ_ONLY,
    "list_screenshots": PermissionLevel.READ_ONLY,
    "read_runner_logs": PermissionLevel.READ_ONLY,
    "get_task_run_events": PermissionLevel.READ_ONLY,
    "get_task_run_screenshots": PermissionLevel.READ_ONLY,
    "get_task_run_playwright_results": PermissionLevel.READ_ONLY,
    "get_automation_runs": PermissionLevel.READ_ONLY,
    "get_automation_run": PermissionLevel.READ_ONLY,
    "list_tests": PermissionLevel.READ_ONLY,
    "get_test": PermissionLevel.READ_ONLY,
    "list_test_results": PermissionLevel.READ_ONLY,
    "get_test_history": PermissionLevel.READ_ONLY,
    "list_dom_captures": PermissionLevel.READ_ONLY,
    "get_dom_capture": PermissionLevel.READ_ONLY,
    "get_dom_capture_html": PermissionLevel.READ_ONLY,
    "awas_check_support": PermissionLevel.READ_ONLY,
    "awas_list_actions": PermissionLevel.READ_ONLY,
    # Visual context tools - generate AI-friendly visual context
    "get_annotated_screenshot": PermissionLevel.READ_ONLY,
    "get_visual_diff": PermissionLevel.READ_ONLY,
    "get_interaction_heatmap": PermissionLevel.READ_ONLY,
    # EXECUTE tools - run workflows, tests, capture screenshots
    "run_workflow": PermissionLevel.EXECUTE,
    "execute_test": PermissionLevel.EXECUTE,
    "capture_screenshot": PermissionLevel.EXECUTE,
    "migrate_task_run_logs": PermissionLevel.EXECUTE,
    "awas_discover": PermissionLevel.EXECUTE,
    "awas_execute": PermissionLevel.EXECUTE,
    "execute_python": PermissionLevel.EXECUTE,
    "spawn_sub_agent": PermissionLevel.EXECUTE,
    # MODIFY tools - create, update, delete resources
    "load_config": PermissionLevel.MODIFY,
    "ensure_config_loaded": PermissionLevel.MODIFY,
    "create_test": PermissionLevel.MODIFY,
    "update_test": PermissionLevel.MODIFY,
    "delete_test": PermissionLevel.MODIFY,
    # DANGEROUS tools - can disrupt execution
    "stop_execution": PermissionLevel.DANGEROUS,
    "restart_runner": PermissionLevel.DANGEROUS,
}


# Global permission service instance
_permission_service: PermissionService | None = None


def get_permission_service() -> PermissionService:
    """Get or create the global permission service."""
    global _permission_service
    if _permission_service is None:
        _permission_service = PermissionService()
    return _permission_service


def get_tool_permission_level(tool_name: str) -> PermissionLevel:
    """Get the permission level for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Permission level (defaults to EXECUTE if not explicitly mapped)
    """
    return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.EXECUTE)


async def check_permission(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Check if permission is granted for a tool call.

    Args:
        tool_name: Name of the tool being called
        arguments: Arguments being passed to the tool

    Returns:
        True if permission is granted, False otherwise
    """
    service = get_permission_service()
    level = get_tool_permission_level(tool_name)

    # Truncate params for description
    params_str = str(arguments)[:100]
    if len(str(arguments)) > 100:
        params_str += "..."

    req = PermissionRequest(
        tool_name=tool_name,
        level=level,
        description=f"Execute {tool_name} ({level.value}): {params_str}",
        params=arguments,
    )

    return await service.request(req)


def permission_denied_response(
    tool_name: str, level: PermissionLevel
) -> dict[str, Any]:
    """Generate a permission denied response.

    Args:
        tool_name: Name of the tool
        level: Required permission level

    Returns:
        Error response dict
    """
    return {
        "success": False,
        "error": f"Permission denied for {tool_name} (requires {level.value} permission)",
        "permission_required": level.value,
    }
