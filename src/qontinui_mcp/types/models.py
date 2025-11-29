"""Type definitions for Qontinui nodes, actions, and workflows."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """All available action types in Qontinui."""

    # Basic Actions - Atomic operations
    FIND = "FIND"
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    MIDDLE_CLICK = "MIDDLE_CLICK"
    DEFINE = "DEFINE"
    TYPE = "TYPE"
    MOVE = "MOVE"
    HOVER = "HOVER"
    VANISH = "VANISH"
    WAIT_VANISH = "WAIT_VANISH"
    HIGHLIGHT = "HIGHLIGHT"
    SCROLL_MOUSE_WHEEL = "SCROLL_MOUSE_WHEEL"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    MOUSE_DOWN = "MOUSE_DOWN"
    MOUSE_UP = "MOUSE_UP"
    KEY_DOWN = "KEY_DOWN"
    KEY_UP = "KEY_UP"
    CLASSIFY = "CLASSIFY"

    # Composite Actions - Complex operations
    CLICK_UNTIL = "CLICK_UNTIL"
    DRAG = "DRAG"
    RUN_PROCESS = "RUN_PROCESS"


class Region(BaseModel):
    """Region on screen."""

    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Location(BaseModel):
    """Location (point) on screen."""

    x: int
    y: int


class StateImage(BaseModel):
    """StateImage represents a visual pattern to search for."""

    name: str
    path: str
    similarity: float | None = Field(default=None, ge=0, le=1)
    fixed_region: Region | None = None


class Match(BaseModel):
    """Match result from pattern finding."""

    region: Region
    similarity: float
    image_name: str | None = None


class ObjectCollection(BaseModel):
    """Target object for actions (images, regions, locations)."""

    images: list[StateImage] | None = None
    regions: list[Region] | None = None
    locations: list[Location] | None = None
    strings: list[str] | None = None
    match_objects: list[Match] | None = None


class ActionConfig(BaseModel):
    """Configuration for an action."""

    type: ActionType
    description: str | None = None
    options: dict[str, Any] | None = None
    subsequent_actions: list[ActionConfig] | None = None


class ActionResult(BaseModel):
    """Result of an action execution."""

    success: bool
    matches: list[Match] | None = None
    output_text: str | None = None
    error: str | None = None
    duration_ms: int | None = None


class NodeParameter(BaseModel):
    """Parameter definition for a node."""

    name: str
    type: Literal["string", "number", "boolean", "image", "region", "location", "object"]
    required: bool
    description: str
    default: Any | None = None


class NodeMetadata(BaseModel):
    """Node metadata for search and documentation."""

    id: str
    name: str
    category: str
    description: str
    action_type: ActionType
    parameters: list[NodeParameter]
    examples: list[str] | None = None
    tags: list[str] | None = None


class NodeSearchResult(BaseModel):
    """Search result for nodes."""

    node: NodeMetadata
    score: float
    matched_fields: list[str]


class StepCondition(BaseModel):
    """Condition for step execution."""

    type: Literal["always", "if_success", "if_failure", "if_found", "if_not_found"]
    expression: str | None = None


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_attempts: int = Field(gt=0)
    delay_ms: int = Field(ge=0)
    backoff_multiplier: float | None = Field(default=None, gt=0)


class WorkflowStep(BaseModel):
    """A single step in a workflow."""

    id: str
    name: str
    description: str | None = None
    action: ActionConfig
    condition: StepCondition | None = None
    on_success: str | None = None  # Next step ID
    on_failure: str | None = None  # Next step ID or "exit"
    retry: RetryConfig | None = None
    timeout_ms: int | None = Field(default=None, gt=0)


class WorkflowMetadata(BaseModel):
    """Workflow metadata."""

    created_at: str | None = None
    updated_at: str | None = None
    complexity: Literal["simple", "medium", "complex"] | None = None
    estimated_duration_ms: int | None = None
    category: str | None = None
    prerequisites: list[str] | None = None


class Workflow(BaseModel):
    """A workflow is a named sequence of steps."""

    id: str
    name: str
    description: str
    version: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    steps: list[WorkflowStep]
    variables: dict[str, Any] | None = None
    metadata: WorkflowMetadata | None = None


class WorkflowResult(BaseModel):
    """Result of workflow execution."""

    workflow_id: str
    success: bool
    steps_executed: list[StepResult]
    duration_ms: int
    error: str | None = None
    stopped_at_step: str | None = None


class StepResult(BaseModel):
    """Result of a single step execution."""

    step_id: str
    action_result: ActionResult
    duration_ms: int
    attempts: int


class WorkflowTemplate(BaseModel):
    """Workflow template for search and generation."""

    id: str
    name: str
    description: str
    category: str
    tags: list[str]
    complexity: Literal["simple", "medium", "complex"]
    template: Workflow
    use_cases: list[str]
    customization_points: list[str]


class WorkflowSearchResult(BaseModel):
    """Search result for workflows."""

    workflow: WorkflowTemplate
    score: float
    matched_fields: list[str]
