"""MCP Prompts for Qontinui Runner.

This module provides parameterized prompt templates for common automation tasks.
Prompts aggregate context from the runner to help AI assistants with structured
debugging, analysis, and verification workflows.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp import types

if TYPE_CHECKING:
    from .client import QontinuiClient


def _format_test_results(results: list[dict[str, Any]], limit: int = 5) -> str:
    """Format test results for prompt context."""
    if not results:
        return "No recent test results."

    lines = []
    for result in results[:limit]:
        status = result.get("status", "unknown")
        duration = result.get("duration_ms", 0)
        timestamp = result.get("created_at", "")
        error = result.get("error_message", "")

        status_emoji = {
            "passed": "[PASS]",
            "failed": "[FAIL]",
            "error": "[ERROR]",
            "timeout": "[TIMEOUT]",
            "skipped": "[SKIP]",
        }.get(status, f"[{status.upper()}]")

        line = f"  {status_emoji} {timestamp} ({duration}ms)"
        if error:
            line += f"\n    Error: {error[:200]}{'...' if len(error) > 200 else ''}"
        lines.append(line)

    return "\n".join(lines)


def _format_screenshots(screenshots: list[dict[str, Any]], limit: int = 5) -> str:
    """Format screenshot metadata for prompt context."""
    if not screenshots:
        return "No screenshots available."

    lines = []
    for s in screenshots[:limit]:
        path = s.get("path", s.get("filename", "unknown"))
        timestamp = s.get("timestamp", "")
        template = s.get("template_name", "")
        confidence = s.get("confidence", 0)

        line = f"  - {path}"
        if timestamp:
            line += f" ({timestamp})"
        if template:
            line += f" | Template: {template}"
        if confidence:
            line += f" | Confidence: {confidence:.2%}"
        lines.append(line)

    return "\n".join(lines)


def _format_events(events: list[dict[str, Any]], limit: int = 10) -> str:
    """Format events for prompt context."""
    if not events:
        return "No events recorded."

    lines = []
    for event in events[:limit]:
        event_type = event.get("event_type", "unknown")
        timestamp = event.get("timestamp", "")
        data = event.get("data", {})

        # Extract key information based on event type
        summary = ""
        if event_type == "action":
            summary = data.get("action_name", "")
        elif event_type == "image_recognition":
            summary = (
                f"Template: {data.get('template_name', '')} - {data.get('result', '')}"
            )
        elif event_type == "error":
            summary = data.get("message", "")[:100]
        else:
            summary = str(data)[:100]

        lines.append(f"  [{event_type}] {timestamp}: {summary}")

    return "\n".join(lines)


async def build_debug_test_failure_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for debugging test failures.

    Aggregates test details, recent results, and optionally screenshots
    to help diagnose why a test is failing.
    """
    arguments = arguments or {}
    test_id = arguments.get("test_id", "")
    include_screenshots = (
        arguments.get("include_screenshots", "false").lower() == "true"
    )

    if not test_id:
        return types.GetPromptResult(
            description="Debug test failure (missing test_id)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: test_id argument is required for debug_test_failure prompt.",
                    ),
                )
            ],
        )

    # Fetch test details
    test_response = await client.get_test(test_id)
    if not test_response.success:
        return types.GetPromptResult(
            description=f"Debug test failure: {test_id} (not found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Error: Test not found: {test_id}. Error: {test_response.error}",
                    ),
                )
            ],
        )

    test = test_response.data or {}
    test_name = test.get("name", "Unknown Test")
    test_type = test.get("test_type", "unknown")

    # Fetch recent results
    results_response = await client.list_test_results(test_id=test_id, limit=5)
    recent_results = []
    if results_response.success and results_response.data:
        recent_results = results_response.data.get("results", [])

    # Build context parts
    context_parts = [
        "## Test Details",
        f"- **Name:** {test_name}",
        f"- **Type:** {test_type}",
        f"- **ID:** {test_id}",
        f"- **Description:** {test.get('description', 'No description')}",
        f"- **Category:** {test.get('category', 'unknown')}",
        f"- **Critical:** {test.get('is_critical', False)}",
        f"- **Timeout:** {test.get('timeout_seconds', 60)}s",
        "",
        "## Test Code",
    ]

    # Add test code based on type
    if test_type == "playwright_cdp":
        code = test.get("playwright_code", "No code available")
        context_parts.append(f"```typescript\n{code}\n```")
    elif test_type == "python_script":
        code = test.get("python_code", "No code available")
        context_parts.append(f"```python\n{code}\n```")
    elif test_type == "repository_test":
        config = test.get("repo_test_config", {})
        context_parts.append(f"Command: `{config.get('command', 'N/A')}`")
        context_parts.append(
            f"Working Directory: `{config.get('working_directory', 'N/A')}`"
        )
    else:
        context_parts.append("Unknown test type - code not available")

    # Add recent results
    context_parts.extend(
        [
            "",
            "## Recent Results",
            _format_test_results(recent_results),
        ]
    )

    # Add screenshots if requested
    if include_screenshots:
        # Get screenshots from the most recent task run
        if recent_results:
            latest_result = recent_results[0]
            task_run_id = latest_result.get("task_run_id")
            if task_run_id:
                screenshots_response = await client.get_task_run_screenshots(
                    task_run_id
                )
                if screenshots_response.success and screenshots_response.data:
                    screenshots = screenshots_response.data.get("screenshots", [])
                    context_parts.extend(
                        [
                            "",
                            "## Screenshots",
                            _format_screenshots(screenshots),
                        ]
                    )

    # Build the prompt
    prompt_text = f"""Analyze this failing test and identify the root cause:

{chr(10).join(context_parts)}

## Instructions

1. **Review the test code** and recent failure output
2. **Identify whether the failure is:**
   - A test bug (selector changed, timing issue, incorrect assertion)
   - An application bug (real regression in the application)
   - An environment issue (server not running, data missing, network error)
3. **Provide a specific fix recommendation** with code changes if applicable
4. **If it's an application bug**, describe what code needs to change in the application

Focus on actionable fixes rather than generic debugging advice."""

    return types.GetPromptResult(
        description=f"Debug test failure: {test_name}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_analyze_screenshot_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for analyzing screenshots.

    Provides context for visual analysis of UI screenshots from automation.
    """
    arguments = arguments or {}
    screenshot_id = arguments.get("screenshot_id", "")
    focus_area = arguments.get("focus_area", "")

    if not screenshot_id:
        return types.GetPromptResult(
            description="Analyze screenshot (missing screenshot_id)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: screenshot_id argument is required for analyze_screenshot prompt.",
                    ),
                )
            ],
        )

    # Get screenshot list to find metadata
    screenshots_response = await client.list_screenshots()
    screenshot_info = None
    if screenshots_response.success and screenshots_response.data:
        screenshots = screenshots_response.data.get("screenshots", [])
        for s in screenshots:
            if s.get("id") == screenshot_id or s.get("filename") == screenshot_id:
                screenshot_info = s
                break

    if not screenshot_info:
        # Screenshot might still exist by path
        screenshot_info = {"path": screenshot_id, "id": screenshot_id}

    screenshot_path = screenshot_info.get(
        "path", screenshot_info.get("filename", screenshot_id)
    )

    context_parts = [
        "## Screenshot Information",
        f"- **ID/Path:** {screenshot_id}",
    ]

    if screenshot_info.get("timestamp"):
        context_parts.append(f"- **Captured:** {screenshot_info['timestamp']}")
    if screenshot_info.get("template_name"):
        context_parts.append(f"- **Template:** {screenshot_info['template_name']}")
    if screenshot_info.get("confidence"):
        context_parts.append(
            f"- **Match Confidence:** {screenshot_info['confidence']:.2%}"
        )
    if screenshot_info.get("match_location"):
        context_parts.append(
            f"- **Match Location:** {screenshot_info['match_location']}"
        )

    focus_instruction = ""
    if focus_area:
        focus_instruction = f"\n**Focus Area:** {focus_area}\n"

    prompt_text = f"""Analyze this screenshot for UI verification:

{chr(10).join(context_parts)}
{focus_instruction}
**Screenshot Path:** `{screenshot_path}`

Please read the screenshot file using the Read tool and then:

## Instructions

1. **Describe what you see** in the screenshot
2. **Identify UI elements** visible (buttons, forms, text, navigation, etc.)
3. **Note any issues** such as:
   - Visual bugs (misalignment, overlapping elements, wrong colors)
   - Missing elements that should be present
   - Error messages or unexpected states
   - Loading states or incomplete renders
4. **Compare to expected state** if context is provided
5. **Suggest verification assertions** that could validate this UI state"""

    return types.GetPromptResult(
        description=f"Analyze screenshot: {screenshot_id}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_fix_playwright_failure_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for fixing Playwright test failures.

    Provides a structured workflow for diagnosing and fixing Playwright CDP tests.
    """
    arguments = arguments or {}
    spec_name = arguments.get("spec_name", "")
    error_message = arguments.get("error_message", "")

    if not spec_name:
        return types.GetPromptResult(
            description="Fix Playwright failure (missing spec_name)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: spec_name argument is required for fix_playwright_failure prompt.",
                    ),
                )
            ],
        )

    # Search for tests matching the spec name
    tests_response = await client.list_tests(test_type="playwright_cdp")
    matching_tests = []
    if tests_response.success and tests_response.data:
        tests = tests_response.data.get("tests", [])
        for test in tests:
            test_name = test.get("name", "").lower()
            if spec_name.lower() in test_name or test_name in spec_name.lower():
                matching_tests.append(test)

    # Read Playwright logs
    logs_response = await client.read_runner_logs(log_type="playwright", limit=50)
    playwright_logs = []
    if logs_response.success and logs_response.data:
        playwright_logs = logs_response.data.get("entries", [])

    context_parts = [
        f"## Playwright Spec: {spec_name}",
        "",
    ]

    if error_message:
        context_parts.extend(
            [
                "## Error Message",
                f"```\n{error_message}\n```",
                "",
            ]
        )

    if matching_tests:
        context_parts.append("## Matching Tests in Database")
        for test in matching_tests[:3]:
            context_parts.extend(
                [
                    f"### {test.get('name', 'Unknown')} (ID: {test.get('id', 'N/A')})",
                    f"```typescript\n{test.get('playwright_code', 'No code')}\n```",
                    "",
                ]
            )

    if playwright_logs:
        context_parts.extend(
            [
                "## Recent Playwright Execution Logs",
                "```json",
                json.dumps(playwright_logs[:10], indent=2),
                "```",
            ]
        )

    prompt_text = f"""Fix this failing Playwright test:

{chr(10).join(context_parts)}

## Debugging Workflow

1. **Analyze the error** - Is it a timeout, selector not found, assertion failure, or network issue?

2. **Check for common issues:**
   - **Selector problems:** Element may have changed class/id, use more stable selectors
   - **Timing issues:** Add waits for dynamic content, network requests
   - **State problems:** Previous test may not have cleaned up properly
   - **Network issues:** API responses may be slow or failing

3. **Identify the fix:**
   - If selector changed: Update to use data-testid, role, or text content
   - If timing issue: Add appropriate waits (waitForSelector, waitForLoadState)
   - If assertion wrong: Update expected value or make assertion more flexible
   - If test logic flawed: Restructure the test approach

4. **Provide the fixed code** with explanation of changes

5. **Suggest prevention** - How to make the test more resilient"""

    return types.GetPromptResult(
        description=f"Fix Playwright failure: {spec_name}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_verify_workflow_state_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for verifying workflow state.

    Helps verify that the current GUI state matches the expected workflow state.
    """
    arguments = arguments or {}
    state_name = arguments.get("state_name", "")
    workflow_name = arguments.get("workflow_name", "")

    if not state_name:
        return types.GetPromptResult(
            description="Verify workflow state (missing state_name)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: state_name argument is required for verify_workflow_state prompt.",
                    ),
                )
            ],
        )

    # Get current executor status
    status_response = await client.status()
    status_data: dict[str, Any] = (
        status_response.data if status_response.success and status_response.data else {}
    )

    # Get loaded config info
    config_info = client.get_loaded_config_info()

    # Get recent screenshots
    screenshots_response = await client.list_screenshots()
    recent_screenshots = []
    if screenshots_response.success and screenshots_response.data:
        recent_screenshots = screenshots_response.data.get("screenshots", [])[:5]

    context_parts = [
        f"## Expected State: {state_name}",
    ]

    if workflow_name:
        context_parts.append(f"## Workflow Context: {workflow_name}")

    context_parts.extend(
        [
            "",
            "## Current Executor Status",
            f"- **Running:** {status_data.get('running', False)}",
            f"- **Config Loaded:** {status_data.get('config_loaded', False)}",
        ]
    )

    if config_info.get("loaded"):
        context_parts.extend(
            [
                "",
                "## Loaded Configuration",
                f"- **Config Path:** {config_info.get('config_path', 'N/A')}",
                f"- **Workflows:** {', '.join(config_info.get('workflow_names', []))}",
            ]
        )

    if recent_screenshots:
        context_parts.extend(
            [
                "",
                "## Recent Screenshots (for visual verification)",
                _format_screenshots(recent_screenshots),
            ]
        )

    prompt_text = f"""Verify that the current GUI state matches the expected workflow state:

{chr(10).join(context_parts)}

## Verification Steps

1. **Capture current state:**
   - Take a screenshot using `capture_screenshot` tool
   - Optionally capture DOM using browser extension

2. **Compare to expected state "{state_name}":**
   - What visual elements should be present?
   - What text/labels should be visible?
   - What should NOT be visible (e.g., loading indicators, error messages)?

3. **Run verification checks:**
   - Use pattern matching to verify expected elements
   - Check for unexpected elements or error states
   - Verify interactive elements are in correct state

4. **Report findings:**
   - Does the current state match "{state_name}"?
   - List any discrepancies found
   - Suggest corrective actions if needed"""

    return types.GetPromptResult(
        description=f"Verify workflow state: {state_name}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_create_verification_test_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for creating verification tests.

    Guides the creation of new verification tests based on behavior descriptions.
    """
    arguments = arguments or {}
    behavior_description = arguments.get("behavior_description", "")
    test_type = arguments.get("test_type", "")

    if not behavior_description:
        return types.GetPromptResult(
            description="Create verification test (missing behavior_description)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: behavior_description argument is required for create_verification_test prompt.",
                    ),
                )
            ],
        )

    # Get existing tests for context
    tests_response = await client.list_tests()
    existing_tests = []
    if tests_response.success and tests_response.data:
        existing_tests = tests_response.data.get("tests", [])

    test_type_info = {
        "playwright_cdp": {
            "name": "Playwright CDP",
            "description": "Browser-based tests using Playwright with CDP connection. Best for DOM assertions, form interactions, and browser state verification.",
            "example": """// Example Playwright CDP test
const { expect } = require('@playwright/test');

// Get the page from CDP connection (pre-connected)
const page = context.page;

// Wait for element and verify
await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
await expect(page.locator('h1')).toHaveText('Welcome');

// Return result
return { status: 'passed', assertions: ['Login form visible', 'Header text correct'] };""",
        },
        "python_script": {
            "name": "Python Script",
            "description": "Custom Python verification logic. Best for data validation, API checks, file verification, and complex logic.",
            "example": """# Example Python verification script
import json

def verify():
    # Your verification logic here
    assertions = []

    # Example: Check file exists
    import os
    if os.path.exists('output.json'):
        assertions.append({'name': 'Output file exists', 'passed': True})
    else:
        assertions.append({'name': 'Output file exists', 'passed': False})

    # Return structured result
    passed = all(a['passed'] for a in assertions)
    return {
        'status': 'passed' if passed else 'failed',
        'assertions': assertions,
        'output': 'Verification complete'
    }

result = verify()
print(json.dumps(result))""",
        },
        "qontinui_vision": {
            "name": "Qontinui Vision",
            "description": "Visual pattern matching tests. Best for verifying UI elements using template matching and image recognition.",
            "example": "# Vision tests use template configuration rather than code",
        },
    }

    context_parts = [
        "## Behavior to Verify",
        f"{behavior_description}",
        "",
    ]

    if test_type and test_type in test_type_info:
        info = test_type_info[test_type]
        context_parts.extend(
            [
                f"## Requested Test Type: {info['name']}",
                info["description"],
                "",
                "### Example",
                f"```\n{info['example']}\n```",
            ]
        )
    else:
        context_parts.extend(
            [
                "## Available Test Types",
                "",
            ]
        )
        for tt, info in test_type_info.items():
            context_parts.extend(
                [
                    f"### {info['name']} (`{tt}`)",
                    info["description"],
                    "",
                ]
            )

    if existing_tests:
        context_parts.extend(
            [
                "## Existing Tests (for reference)",
                f"Total: {len(existing_tests)} tests",
                "",
            ]
        )
        for test in existing_tests[:5]:
            context_parts.append(
                f"- **{test.get('name', 'Unknown')}** ({test.get('test_type', 'unknown')}): "
                f"{test.get('description', 'No description')[:80]}"
            )

    prompt_text = f"""Create a verification test for the described behavior:

{chr(10).join(context_parts)}

## Test Creation Guidelines

1. **Choose the right test type:**
   - Use `playwright_cdp` for browser/DOM verification
   - Use `python_script` for data/API/file verification
   - Use `qontinui_vision` for visual pattern matching

2. **Write focused assertions:**
   - Test ONE behavior per test
   - Make assertions specific and actionable
   - Include meaningful error messages

3. **Consider edge cases:**
   - What if the element is loading?
   - What if data is empty?
   - What if the previous step failed?

4. **Use the create_test tool** with:
   - Clear name describing what's verified
   - Detailed description of success criteria
   - Appropriate category (visual, dom, data, etc.)
   - Test code implementing the verification

Please generate the complete test definition that can be created using the `create_test` tool."""

    return types.GetPromptResult(
        description=f"Create verification test for: {behavior_description[:50]}...",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_analyze_automation_run_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for analyzing automation run results.

    Provides comprehensive context about an automation run for debugging.
    """
    arguments = arguments or {}
    run_id = arguments.get("run_id", "")
    focus_on_failures = arguments.get("focus_on_failures", "true").lower() == "true"

    if not run_id:
        # Get the most recent run if no ID provided
        runs_response = await client.get_automation_runs(limit=1)
        if runs_response.success and runs_response.data:
            runs = runs_response.data.get("runs", [])
            if runs:
                run_id = runs[0].get("id", "")

    if not run_id:
        return types.GetPromptResult(
            description="Analyze automation run (no runs found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: No automation runs found. Please provide a run_id or run an automation first.",
                    ),
                )
            ],
        )

    # Fetch run details
    run_response = await client.get_automation_run(run_id)
    if not run_response.success:
        return types.GetPromptResult(
            description=f"Analyze automation run: {run_id} (not found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Error: Automation run not found: {run_id}. Error: {run_response.error}",
                    ),
                )
            ],
        )

    run_data = run_response.data or {}

    context_parts = [
        "## Automation Run Summary",
        f"- **Run ID:** {run_id}",
        f"- **Workflow:** {run_data.get('workflow_name', 'Unknown')}",
        f"- **Status:** {run_data.get('status', 'Unknown')}",
        f"- **Started:** {run_data.get('started_at', 'N/A')}",
        f"- **Duration:** {run_data.get('duration_ms', 0)}ms",
        "",
    ]

    # Actions summary
    actions_summary = run_data.get("actions_summary", [])
    if actions_summary:
        context_parts.extend(
            [
                "## Actions Executed",
            ]
        )
        for action in actions_summary[:20]:
            status = "[OK]" if action.get("success", False) else "[FAIL]"
            context_parts.append(
                f"  {status} {action.get('name', 'Unknown')} ({action.get('duration_ms', 0)}ms)"
            )
        context_parts.append("")

    # States visited
    states = run_data.get("states_visited", [])
    if states:
        context_parts.extend(
            [
                "## States Visited",
                f"  {' -> '.join(states[:20])}",
                "",
            ]
        )

    # Template matches
    template_matches = run_data.get("template_matches", [])
    if template_matches:
        context_parts.append("## Template Matches")
        for match in template_matches[:10]:
            confidence = match.get("confidence", 0)
            status = (
                "[MATCH]"
                if confidence > 0.8
                else "[WEAK]" if confidence > 0.5 else "[FAIL]"
            )
            context_parts.append(
                f"  {status} {match.get('template_name', 'Unknown')} - {confidence:.1%}"
            )
        context_parts.append("")

    # Anomalies/failures
    anomalies = run_data.get("anomalies", [])
    if anomalies or focus_on_failures:
        context_parts.append("## Issues/Anomalies")
        if anomalies:
            for anomaly in anomalies[:10]:
                context_parts.append(
                    f"  - {anomaly.get('type', 'Unknown')}: {anomaly.get('message', 'No details')}"
                )
        else:
            context_parts.append("  No anomalies recorded.")
        context_parts.append("")

    prompt_text = f"""Analyze this automation run and provide insights:

{chr(10).join(context_parts)}

## Analysis Tasks

1. **Overall Assessment:**
   - Did the automation achieve its goal?
   - What was the success rate of actions?

2. **Identify Issues:**
   - Which actions failed and why?
   - Were there timing or synchronization issues?
   - Did template matching work reliably?

3. **State Transitions:**
   - Did the workflow follow the expected path?
   - Were there unexpected states or loops?

4. **Recommendations:**
   - What improvements could make this more reliable?
   - Are there missing error handlers?
   - Should any thresholds be adjusted?"""

    return types.GetPromptResult(
        description=f"Analyze automation run: {run_id}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_debug_image_recognition_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for debugging image recognition issues.

    Helps diagnose template matching failures in Qontinui vision.
    """
    arguments = arguments or {}
    template_name = arguments.get("template_name", "")
    last_n_attempts = int(arguments.get("last_n_attempts", "10"))

    # Read image recognition logs
    logs_response = await client.read_runner_logs(
        log_type="image-recognition", limit=last_n_attempts * 2
    )
    recognition_logs = []
    if logs_response.success and logs_response.data:
        recognition_logs = logs_response.data.get("entries", [])

    # Filter by template name if provided
    if template_name and recognition_logs:
        recognition_logs = [
            log
            for log in recognition_logs
            if template_name.lower() in str(log.get("data", {})).lower()
        ]

    # Get screenshots
    screenshots_response = await client.list_screenshots()
    screenshots = []
    if screenshots_response.success and screenshots_response.data:
        screenshots = screenshots_response.data.get("screenshots", [])

    context_parts = []

    if template_name:
        context_parts.extend(
            [
                f"## Template: {template_name}",
                "",
            ]
        )

    context_parts.extend(
        [
            "## Recent Image Recognition Attempts",
        ]
    )

    if recognition_logs:
        for log in recognition_logs[:last_n_attempts]:
            data = log.get("data", {})
            template = data.get("template_name", "unknown")
            result = data.get("result", "unknown")
            confidence = data.get("confidence", 0)
            location = data.get("location", "N/A")
            timestamp = log.get("timestamp", "")

            status = "[MATCH]" if result == "found" else "[FAIL]"
            context_parts.append(f"  {status} {template} @ {timestamp}")
            context_parts.append(
                f"      Confidence: {confidence:.2%}, Location: {location}"
            )
    else:
        context_parts.append("  No recognition attempts found in logs.")

    context_parts.extend(
        [
            "",
            "## Available Screenshots",
            _format_screenshots(screenshots[:10]),
        ]
    )

    prompt_text = f"""Debug image recognition issues:

{chr(10).join(context_parts)}

## Debugging Steps

1. **Check template quality:**
   - Is the template image clear and distinctive?
   - Does it have sufficient contrast with background?
   - Is it the right size (not too small or too large)?

2. **Verify screen capture:**
   - Is the target application visible on screen?
   - Is there anything overlapping the target area?
   - Is the resolution/scaling correct?

3. **Analyze confidence scores:**
   - Scores > 0.8 are reliable matches
   - Scores 0.5-0.8 may need threshold adjustment
   - Scores < 0.5 indicate template/target mismatch

4. **Common issues:**
   - Theme/color changes in application
   - Dynamic content in template area
   - Resolution or DPI scaling differences
   - Window focus or overlay issues

5. **Recommendations:**
   - Consider more stable template regions
   - Adjust confidence thresholds
   - Use multiple fallback templates
   - Add pre-match state verification"""

    return types.GetPromptResult(
        description=f"Debug image recognition{': ' + template_name if template_name else ''}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_analyze_verification_failure_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for analyzing verification failures.

    Helps diagnose why a verification criterion failed.
    """
    arguments = arguments or {}
    task_id = arguments.get("task_id", "")
    criterion_id = arguments.get("criterion_id", "")

    if not task_id:
        return types.GetPromptResult(
            description="Analyze verification failure (missing task_id)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: task_id argument is required for analyze_verification_failure prompt.",
                    ),
                )
            ],
        )

    # Fetch task run details
    task_response = await client.get_task_run(task_id)
    if not task_response.success:
        return types.GetPromptResult(
            description=f"Analyze verification failure: {task_id} (not found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Error: Task run not found: {task_id}. Error: {task_response.error}",
                    ),
                )
            ],
        )

    task_data = task_response.data or {}

    # Get test results for this task
    test_results_response = await client.list_test_results(task_run_id=task_id)
    test_results = []
    if test_results_response.success and test_results_response.data:
        test_results = test_results_response.data.get("results", [])

    # Get events for this task
    events_response = await client.get_task_run_events(task_run_id=task_id, limit=50)
    events = []
    if events_response.success and events_response.data:
        events = events_response.data.get("events", [])

    context_parts = [
        "## Task Information",
        f"- **Task ID:** {task_id}",
        f"- **Task Name:** {task_data.get('task_name', 'Unknown')}",
        f"- **Status:** {task_data.get('status', 'Unknown')}",
        "",
    ]

    if criterion_id:
        context_parts.extend(
            [
                "## Failing Criterion",
                f"- **Criterion ID:** {criterion_id}",
                "",
            ]
        )

    # Add failed test results
    failed_tests = [r for r in test_results if r.get("status") in ["failed", "error"]]
    if failed_tests:
        context_parts.append("## Failed Tests")
        for test in failed_tests[:5]:
            context_parts.extend(
                [
                    f"### {test.get('test_name', 'Unknown Test')}",
                    f"- Status: {test.get('status')}",
                    f"- Duration: {test.get('duration_ms', 0)}ms",
                    f"- Error: {test.get('error_message', 'No error message')[:300]}",
                    "",
                ]
            )

    # Add recent error events
    error_events = [
        e
        for e in events
        if e.get("event_type") == "error" or "error" in str(e.get("data", {})).lower()
    ]
    if error_events:
        context_parts.append("## Error Events")
        for event in error_events[:5]:
            context_parts.append(
                f"  - {event.get('timestamp', '')}: {event.get('data', {})}"
            )
        context_parts.append("")

    prompt_text = f"""Analyze why verification failed for this task:

{chr(10).join(context_parts)}

## Analysis Tasks

1. **Identify the Root Cause:**
   - What specifically failed?
   - Was it a test failure, timeout, or system error?
   - Are there preconditions that weren't met?

2. **Categorize the Failure:**
   - Test bug (selector, timing, assertion)
   - Application bug (real regression)
   - Environment issue (data, services, configuration)
   - Verification plan issue (unrealistic criterion)

3. **Recommend Fixes:**
   - What code changes are needed?
   - Should the verification plan be updated?
   - Are there missing preconditions to add?

4. **Suggest Prevention:**
   - How can this failure be prevented in the future?
   - Should additional tests be added?"""

    return types.GetPromptResult(
        description=f"Analyze verification failure: {task_id}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_create_verification_plan_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for creating verification plans.

    Helps generate verification plans for features.
    """
    arguments = arguments or {}
    feature_description = arguments.get("feature_description", "")
    strategy = arguments.get("strategy", "")

    if not feature_description:
        return types.GetPromptResult(
            description="Create verification plan (missing feature_description)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: feature_description argument is required for create_verification_plan prompt.",
                    ),
                )
            ],
        )

    # Get existing tests for reference
    tests_response = await client.list_tests()
    existing_tests = []
    if tests_response.success and tests_response.data:
        existing_tests = tests_response.data.get("tests", [])

    # Get recent verification history
    # Note: This assumes there's a verification history endpoint
    context_parts = [
        "## Feature to Verify",
        f"{feature_description}",
        "",
    ]

    if strategy:
        strategy_descriptions = {
            "exhaustive": "Test every possible scenario and edge case",
            "smoke": "Quick tests of critical paths only",
            "targeted": "Focus on areas most likely to have issues",
        }
        context_parts.extend(
            [
                f"## Verification Strategy: {strategy}",
                strategy_descriptions.get(strategy, strategy),
                "",
            ]
        )

    if existing_tests:
        context_parts.extend(
            [
                "## Existing Tests (for reference)",
                f"Total tests: {len(existing_tests)}",
                "",
            ]
        )
        # Group by type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for test in existing_tests:
            t = test.get("test_type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(test)

        for test_type, tests in by_type.items():
            context_parts.append(f"**{test_type}:** {len(tests)} tests")

    prompt_text = f"""Create a verification plan for the described feature:

{chr(10).join(context_parts)}

## Verification Plan Guidelines

1. **Define Success Criteria:**
   - What must be true for this feature to be verified?
   - What are the critical user journeys?
   - What edge cases need coverage?

2. **Design Test Suite:**
   - Which test types are appropriate? (playwright_cdp, python_script, qontinui_vision)
   - What preconditions are needed?
   - What assertions validate each criterion?

3. **Plan Structure:**
   ```yaml
   verification_plan:
     name: "Feature Name Verification"
     criteria:
       - id: criterion_1
         description: "What to verify"
         test_type: playwright_cdp
         priority: high
         assertions:
           - "Expected outcome 1"
           - "Expected outcome 2"
   ```

4. **Consider Dependencies:**
   - What must be set up before verification?
   - Are there external services needed?
   - What test data is required?

Please generate a complete verification plan that can be executed."""

    return types.GetPromptResult(
        description=f"Create verification plan for: {feature_description[:50]}...",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


async def build_summarize_task_progress_prompt(
    client: "QontinuiClient", arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Build prompt for summarizing task run progress.

    Provides a comprehensive overview of a task run's status and progress.
    """
    arguments = arguments or {}
    task_run_id = arguments.get("task_run_id", "")

    if not task_run_id:
        # Get the most recent task run
        runs_response = await client.get_task_runs()
        if runs_response.success and runs_response.data:
            runs = runs_response.data.get("task_runs", [])
            if runs:
                task_run_id = runs[0].get("id", "")

    if not task_run_id:
        return types.GetPromptResult(
            description="Summarize task progress (no tasks found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Error: No task runs found. Please provide a task_run_id or start a task first.",
                    ),
                )
            ],
        )

    # Fetch task run details
    run_response = await client.get_task_run(task_run_id)
    if not run_response.success:
        return types.GetPromptResult(
            description=f"Summarize task progress: {task_run_id} (not found)",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Error: Task run not found: {task_run_id}. Error: {run_response.error}",
                    ),
                )
            ],
        )

    task_data = run_response.data or {}

    # Fetch related events
    events_response = await client.get_task_run_events(
        task_run_id=task_run_id, limit=50
    )
    events = []
    if events_response.success and events_response.data:
        events = events_response.data.get("events", [])

    # Fetch test results
    test_results_response = await client.list_test_results(task_run_id=task_run_id)
    test_results = []
    if test_results_response.success and test_results_response.data:
        test_results = test_results_response.data.get("results", [])

    context_parts = [
        "## Task Run Summary",
        f"- **Task ID:** {task_run_id}",
        f"- **Task Name:** {task_data.get('task_name', 'Unknown')}",
        f"- **Status:** {task_data.get('status', 'Unknown')}",
        f"- **Started:** {task_data.get('started_at', 'N/A')}",
        f"- **Completed:** {task_data.get('completed_at', 'N/A')}",
        "",
    ]

    # Execution steps
    execution_steps = task_data.get("execution_steps_json", [])
    if execution_steps:
        context_parts.append("## Execution Steps")
        for step in execution_steps[:20]:
            status = "[DONE]" if step.get("completed") else "[...]"
            context_parts.append(f"  {status} {step.get('name', 'Unknown step')}")
        context_parts.append("")

    # Test results
    if test_results:
        passed = sum(1 for r in test_results if r.get("status") == "passed")
        failed = sum(1 for r in test_results if r.get("status") == "failed")
        context_parts.extend(
            [
                "## Test Results",
                f"  Passed: {passed}, Failed: {failed}, Total: {len(test_results)}",
                "",
            ]
        )

    # Recent events
    if events:
        context_parts.extend(
            [
                "## Recent Events",
                _format_events(events[:15]),
                "",
            ]
        )

    # Output log excerpt
    output_log = task_data.get("output_log", "")
    if output_log:
        # Get last 1000 chars
        excerpt = output_log[-1000:] if len(output_log) > 1000 else output_log
        context_parts.extend(
            [
                "## Output Log (excerpt)",
                f"```\n{excerpt}\n```",
            ]
        )

    prompt_text = f"""Summarize the progress of this task:

{chr(10).join(context_parts)}

## Summary Tasks

1. **Current Status:**
   - Is the task running, completed, or failed?
   - How far through the execution is it?

2. **Key Accomplishments:**
   - What has been completed successfully?
   - What tests have passed?

3. **Issues Found:**
   - What has failed or encountered errors?
   - Are there any blocking issues?

4. **Next Steps:**
   - What remains to be done?
   - Are there any recommendations for addressing issues?"""

    return types.GetPromptResult(
        description=f"Task progress: {task_data.get('task_name', task_run_id)}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )
