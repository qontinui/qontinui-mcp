# Workflow Expectations Schema Documentation

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Schema Reference](#schema-reference)
  - [WorkflowExpectations](#workflowexpectations)
  - [Global Expectations](#global-expectations)
  - [Success Criteria](#success-criteria)
  - [Checkpoints](#checkpoints)
  - [Action-Level Expectations](#action-level-expectations)
- [Complete Examples](#complete-examples)
- [Common Use Cases](#common-use-cases)
- [Best Practices](#best-practices)

---

## Overview

The Qontinui Expectations System provides a flexible way to define success conditions, validation checkpoints, and failure handling for workflow automation. It allows you to:

- **Define success criteria** - Specify when a workflow is considered successful beyond simple pass/fail
- **Create validation checkpoints** - Capture screenshots and validate screen state using OCR assertions
- **Configure action behavior** - Control retry logic, timeouts, and failure handling per action
- **Set global constraints** - Apply workflow-wide timing, error handling, and matching thresholds

### Key Concepts

1. **Success Criteria** - Define what constitutes a successful workflow execution
2. **Checkpoints** - Named validation points with OCR assertions and Claude reviews
3. **OCR Assertions** - Text-based validations on screenshots
4. **Global Expectations** - Workflow-wide settings and constraints
5. **Action Expectations** - Per-action configuration for retry, timeout, and checkpoint capture

---

## Quick Start

Here's a minimal example of a workflow with expectations:

```json
{
  "workflow": {
    "id": "login-workflow",
    "name": "User Login",
    "description": "Log into application",
    "steps": [
      {
        "id": "step1",
        "name": "Find username field",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "username_field", "path": "username.png"}]}
          }
        }
      }
    ]
  },
  "expectations": {
    "global": {
      "max_total_duration_ms": 30000,
      "screenshot_on_failure": true
    },
    "success_criteria": [
      {
        "type": "all_actions_pass",
        "description": "All login steps must complete successfully"
      }
    ],
    "checkpoints": {
      "login_complete": {
        "description": "Verify user is logged in",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Welcome",
            "description": "Welcome message should appear"
          }
        ]
      }
    }
  }
}
```

---

## Schema Reference

### WorkflowExpectations

The root expectations object for a workflow.

```typescript
interface WorkflowExpectations {
  global?: GlobalExpectations;
  success_criteria?: SuccessCriterion[];
  checkpoints?: Record<string, CheckpointDefinition>;
}
```

**Fields:**

- `global` - Global workflow-level settings
- `success_criteria` - Array of success criteria (workflow passes if ANY criteria passes)
- `checkpoints` - Named checkpoints with validation rules

---

### Global Expectations

Global settings that apply to the entire workflow execution.

```typescript
interface GlobalExpectations {
  max_workflow_duration_ms?: number;
  max_action_duration_ms?: number;
  screenshot_on_failure?: boolean;
  fail_fast?: boolean;
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_workflow_duration_ms` | number | none | Maximum allowed workflow execution time (ms) |
| `max_action_duration_ms` | number | none | Default timeout for all actions (ms) |
| `screenshot_on_failure` | boolean | `true` | Capture screenshot when workflow fails |
| `fail_fast` | boolean | `false` | Stop execution immediately on first failure |

**Example:**

```json
{
  "global": {
    "max_workflow_duration_ms": 60000,
    "max_action_duration_ms": 5000,
    "screenshot_on_failure": true,
    "fail_fast": false
  }
}
```

**Use Cases:**

- Set aggressive timeouts for fast-fail testing
- Enable screenshot capture for debugging
- Control whether workflow continues after non-critical failures

---

### Success Criteria

Define when a workflow is considered successful. Multiple criteria can be specified; the workflow passes if **ANY** criterion is satisfied.

#### 1. All Actions Pass

The default behavior - all actions must succeed.

```typescript
interface AllActionsPassCriteria {
  type: "all_actions_pass";
  description?: string;
}
```

**Example:**

```json
{
  "type": "all_actions_pass",
  "description": "Every action in the workflow must complete successfully"
}
```

**Use Cases:**
- Standard automation where every step is critical
- Strict validation scenarios

---

#### 2. Minimum Matches

Workflow succeeds if at least N pattern matches are found across all FIND actions.

```typescript
interface MinMatchesCriteria {
  type: "min_matches";
  min_matches: number;
  description?: string;
}
```

**Example:**

```json
{
  "type": "min_matches",
  "min_matches": 3,
  "description": "Must find at least 3 product images on the page"
}
```

**Use Cases:**
- Validating search results contain enough items
- Checking that multiple UI elements are visible
- Data scraping workflows where some matches are acceptable

---

#### 3. Maximum Failures

Allow up to N action failures while still considering workflow successful.

```typescript
interface MaxFailuresCriteria {
  type: "max_failures";
  max_failures: number;
  description?: string;
}
```

**Example:**

```json
{
  "type": "max_failures",
  "max_failures": 2,
  "description": "Workflow succeeds even if up to 2 non-critical actions fail"
}
```

**Use Cases:**
- Workflows with optional steps
- Resilient automation that tolerates minor failures
- Best-effort operations

---

#### 4. Checkpoint Passed

Workflow succeeds if a specific named checkpoint passes validation.

```typescript
interface CheckpointPassedCriteria {
  type: "checkpoint_passed";
  checkpoint_name: string;
  description?: string;
}
```

**Example:**

```json
{
  "type": "checkpoint_passed",
  "checkpoint_name": "payment_success",
  "description": "Workflow succeeds if payment confirmation appears"
}
```

**Use Cases:**
- Goal-oriented automation (reached final state)
- Workflows where intermediate failures are acceptable
- End-to-end testing focused on final outcome

---

#### 5. Required States

Workflow succeeds if all specified states are visited during execution.

```typescript
interface RequiredStatesCriteria {
  type: "required_states";
  required_states: string[];
  description?: string;
}
```

**Example:**

```json
{
  "type": "required_states",
  "required_states": ["home", "search_results", "product_detail", "cart"],
  "description": "Must visit all major pages in shopping flow"
}
```

**Use Cases:**
- State machine validation
- User journey testing
- Coverage testing for multi-page flows

---

#### 6. Custom Condition

Advanced: Evaluate a custom Python expression against execution state.

```typescript
interface CustomCriteria {
  type: "custom";
  custom_condition: string;
  description?: string;
}
```

**Example:**

```json
{
  "type": "custom",
  "custom_condition": "len(matches) >= 5 and failures == 0",
  "description": "At least 5 matches found with no failures"
}
```

**Available Variables in Custom Condition:**

- `matches` - List of all Match objects from FIND actions
- `failures` - Number of failed actions
- `states` - List of visited state names
- `duration_ms` - Total execution time

**Use Cases:**
- Complex validation logic
- Combining multiple conditions
- Domain-specific success criteria

---

### Checkpoints

Named validation points in the workflow. Checkpoints capture screenshots and validate screen state using OCR assertions and optional Claude AI review.

```typescript
interface CheckpointDefinition {
  description?: string;
  screenshot_required?: boolean;
  ocr_assertions?: OCRAssertion[];
  claude_review?: string[];
  max_wait_ms?: number;
  retry_interval_ms?: number;
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | none | Human-readable explanation |
| `screenshot_required` | boolean | `true` | Whether to capture screenshot |
| `ocr_assertions` | OCRAssertion[] | `[]` | Array of text-based validations |
| `claude_review` | string[] | `[]` | Natural language instructions for Claude to review screenshot |
| `max_wait_ms` | number | `5000` | Maximum time to wait for assertions to pass |
| `retry_interval_ms` | number | `500` | Time between assertion retry attempts |

---

#### OCR Assertions

Text-based validation on screenshots captured at checkpoints.

##### Text Present

Check if text exists anywhere on screen.

```typescript
interface TextPresentAssertion {
  type: "text_present";
  text: string;
  case_sensitive?: boolean;
}
```

**Example:**

```json
{
  "type": "text_present",
  "text": "Login successful",
  "case_sensitive": false
}
```

---

##### Text Absent

Check if text does NOT exist on screen.

```typescript
interface TextAbsentAssertion {
  type: "text_absent";
  text: string;
  case_sensitive?: boolean;
}
```

**Example:**

```json
{
  "type": "text_absent",
  "text": "Error",
  "case_sensitive": false
}
```

**Use Cases:**
- Verify error messages don't appear
- Confirm loading states have cleared
- Validate unwanted UI elements are hidden

---

##### No Duplicate Matches

Verify text pattern appears at most once.

```typescript
interface NoDuplicateMatchesAssertion {
  type: "no_duplicate_matches";
  text: string;
  case_sensitive?: boolean;
}
```

**Example:**

```json
{
  "type": "no_duplicate_matches",
  "text": "Order #12345",
  "case_sensitive": false
}
```

**Use Cases:**
- Prevent duplicate UI elements
- Validate unique identifiers
- Check for rendering bugs

---

##### Text Count

Check exact or bounded count of text occurrences.

```typescript
interface TextCountAssertion {
  type: "text_count";
  text: string;
  expected_count?: number;  // Exact count
  case_sensitive?: boolean;
  // Optional: use region to limit search area
  region?: Region;
}
```

**Example - Exact Count:**

```json
{
  "type": "text_count",
  "text": "Item",
  "expected_count": 5
}
```

**Use Cases:**
- Validate list item counts
- Check repeated UI elements
- Count search results

---

##### Text in Region

Verify text appears in a specific screen region.

```typescript
interface TextInRegionAssertion {
  type: "text_in_region";
  text: string;
  region: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  case_sensitive?: boolean;
}
```

**Example:**

```json
{
  "type": "text_in_region",
  "text": "$99.99",
  "region": {
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 50
  },
  "case_sensitive": false
}
```

**Use Cases:**
- Validate text in specific UI areas (header, footer, sidebar)
- Check table cells contain expected values
- Verify modal dialog content

---

#### Claude Review

Natural language instructions for Claude AI to analyze checkpoint screenshots.

**Example:**

```json
{
  "checkpoints": {
    "payment_complete": {
      "screenshot_required": true,
      "claude_review": [
        "Verify the payment confirmation modal is displayed",
        "Check that the order number is visible and follows format ORD-XXXXX",
        "Ensure the success message says 'Payment processed successfully'",
        "Confirm the 'Continue Shopping' button is present"
      ]
    }
  }
}
```

**Use Cases:**
- Complex visual validation beyond text matching
- Layout and design verification
- Semantic understanding of UI state
- Accessibility checks

---

### Action-Level Expectations

Configure expectations for individual workflow actions.

```typescript
interface ActionExpectations {
  is_terminal_on_failure?: boolean;
  capture_checkpoint_on_failure?: boolean;
  capture_checkpoint_after?: boolean;
  checkpoint_name?: string;
  max_retries?: number;
  retry_delay_ms?: number;
  max_duration_ms?: number;
  expected_state_after?: string;
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_terminal_on_failure` | boolean | `true` | Stop workflow if this action fails |
| `capture_checkpoint_on_failure` | boolean | `false` | Capture checkpoint on failure |
| `capture_checkpoint_after` | boolean | `false` | Capture checkpoint after success |
| `checkpoint_name` | string | auto-generated | Name for captured checkpoint |
| `max_retries` | number | `0` | Number of retry attempts |
| `retry_delay_ms` | number | `1000` | Delay between retries |
| `max_duration_ms` | number | none | Timeout for this action |
| `expected_state_after` | string | none | State that should be active after action |

**Example:**

```json
{
  "id": "step3",
  "name": "Click submit button",
  "action": {
    "type": "CLICK",
    "options": {
      "target": {"images": [{"name": "submit_btn", "path": "submit.png"}]}
    }
  },
  "expectations": {
    "is_terminal_on_failure": true,
    "capture_checkpoint_after": true,
    "checkpoint_name": "form_submitted",
    "max_retries": 3,
    "retry_delay_ms": 2000,
    "max_duration_ms": 10000,
    "expected_state_after": "confirmation_page"
  }
}
```

---

## Complete Examples

### Example 1: E-commerce Checkout Flow

```json
{
  "workflow": {
    "id": "checkout-flow",
    "name": "Complete Purchase",
    "description": "Add item to cart and complete checkout",
    "steps": [
      {
        "id": "step1",
        "name": "Find product",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "product", "path": "product.png"}]}
          }
        },
        "expectations": {
          "is_terminal_on_failure": true,
          "max_duration_ms": 5000
        }
      },
      {
        "id": "step2",
        "name": "Click add to cart",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "add_to_cart", "path": "add_cart.png"}]}
          }
        },
        "expectations": {
          "capture_checkpoint_after": true,
          "checkpoint_name": "item_added",
          "max_retries": 2,
          "retry_delay_ms": 1000
        }
      },
      {
        "id": "step3",
        "name": "Navigate to cart",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "cart_icon", "path": "cart.png"}]}
          }
        }
      },
      {
        "id": "step4",
        "name": "Click checkout",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "checkout_btn", "path": "checkout.png"}]}
          }
        },
        "expectations": {
          "capture_checkpoint_after": true,
          "checkpoint_name": "checkout_started"
        }
      },
      {
        "id": "step5",
        "name": "Confirm purchase",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "confirm_btn", "path": "confirm.png"}]}
          }
        },
        "expectations": {
          "capture_checkpoint_after": true,
          "checkpoint_name": "purchase_complete",
          "expected_state_after": "order_confirmation"
        }
      }
    ]
  },
  "expectations": {
    "global": {
      "max_workflow_duration_ms": 120000,
      "max_action_duration_ms": 10000,
      "screenshot_on_failure": true,
      "fail_fast": false
    },
    "success_criteria": [
      {
        "type": "checkpoint_passed",
        "checkpoint_name": "purchase_complete",
        "description": "Workflow succeeds if final purchase confirmation appears"
      }
    ],
    "checkpoints": {
      "item_added": {
        "description": "Verify item was added to cart",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Added to cart",
            "case_sensitive": false
          },
          {
            "type": "text_absent",
            "text": "Out of stock",
            "case_sensitive": false
          }
        ],
        "max_wait_ms": 3000,
        "retry_interval_ms": 500
      },
      "checkout_started": {
        "description": "Verify checkout page loaded",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Checkout",
            "case_sensitive": false
          },
          {
            "type": "text_present",
            "text": "Order Summary",
            "case_sensitive": false
          }
        ]
      },
      "purchase_complete": {
        "description": "Verify order was placed successfully",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Order confirmed",
            "case_sensitive": false
          },
          {
            "type": "text_present",
            "text": "Thank you",
            "case_sensitive": false
          },
          {
            "type": "no_duplicate_matches",
            "text": "Order #"
          }
        ],
        "claude_review": [
          "Verify the order confirmation page is displayed",
          "Check that an order number is visible",
          "Ensure there are no error messages",
          "Confirm the page shows order details and estimated delivery"
        ],
        "max_wait_ms": 10000,
        "retry_interval_ms": 1000
      }
    }
  }
}
```

---

### Example 2: Form Submission with Validation

```json
{
  "workflow": {
    "id": "contact-form",
    "name": "Submit Contact Form",
    "description": "Fill and submit contact form with validation",
    "steps": [
      {
        "id": "step1",
        "name": "Find name field",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "name_field", "path": "name.png"}]}
          }
        }
      },
      {
        "id": "step2",
        "name": "Type name",
        "action": {
          "type": "TYPE",
          "options": {
            "text": "John Doe"
          }
        },
        "expectations": {
          "is_terminal_on_failure": false,
          "max_retries": 1
        }
      },
      {
        "id": "step3",
        "name": "Find email field",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "email_field", "path": "email.png"}]}
          }
        }
      },
      {
        "id": "step4",
        "name": "Type email",
        "action": {
          "type": "TYPE",
          "options": {
            "text": "john@example.com"
          }
        }
      },
      {
        "id": "step5",
        "name": "Submit form",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "submit", "path": "submit.png"}]}
          }
        },
        "expectations": {
          "capture_checkpoint_after": true,
          "checkpoint_name": "form_submitted"
        }
      }
    ]
  },
  "expectations": {
    "global": {
      "max_workflow_duration_ms": 30000,
      "screenshot_on_failure": true,
      "fail_fast": false
    },
    "success_criteria": [
      {
        "type": "max_failures",
        "max_failures": 1,
        "description": "Allow one non-critical failure (e.g., typing)"
      }
    ],
    "checkpoints": {
      "form_submitted": {
        "description": "Verify form submission success",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Thank you",
            "case_sensitive": false
          },
          {
            "type": "text_present",
            "text": "message has been sent",
            "case_sensitive": false
          },
          {
            "type": "text_absent",
            "text": "error",
            "case_sensitive": false
          },
          {
            "type": "text_absent",
            "text": "invalid",
            "case_sensitive": false
          }
        ],
        "max_wait_ms": 5000,
        "retry_interval_ms": 500
      }
    }
  }
}
```

---

### Example 3: Data Extraction with Minimum Matches

```json
{
  "workflow": {
    "id": "scrape-products",
    "name": "Extract Product Data",
    "description": "Find and extract product information from search results",
    "steps": [
      {
        "id": "step1",
        "name": "Find search field",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "search", "path": "search.png"}]}
          }
        }
      },
      {
        "id": "step2",
        "name": "Enter search term",
        "action": {
          "type": "TYPE",
          "options": {
            "text": "laptop"
          }
        }
      },
      {
        "id": "step3",
        "name": "Submit search",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "search_btn", "path": "search_btn.png"}]}
          }
        },
        "expectations": {
          "capture_checkpoint_after": true,
          "checkpoint_name": "search_results"
        }
      },
      {
        "id": "step4",
        "name": "Find product items",
        "action": {
          "type": "FIND",
          "options": {
            "target": {"images": [{"name": "product_card", "path": "product.png"}]},
            "find_all": true
          }
        }
      }
    ]
  },
  "expectations": {
    "global": {
      "max_workflow_duration_ms": 60000,
      "screenshot_on_failure": true
    },
    "success_criteria": [
      {
        "type": "min_matches",
        "min_matches": 10,
        "description": "Must find at least 10 product items"
      }
    ],
    "checkpoints": {
      "search_results": {
        "description": "Verify search results page loaded",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "results for",
            "case_sensitive": false
          },
          {
            "type": "text_count",
            "text": "laptop",
            "expected_count": 10
          }
        ],
        "claude_review": [
          "Verify the search results page is displayed",
          "Check that product cards are visible",
          "Ensure there are no 'no results found' messages"
        ],
        "max_wait_ms": 10000,
        "retry_interval_ms": 1000
      }
    }
  }
}
```

---

### Example 4: Multi-State Journey Validation

```json
{
  "workflow": {
    "id": "user-journey",
    "name": "Complete User Onboarding",
    "description": "Navigate through all onboarding steps",
    "steps": [
      {
        "id": "step1",
        "name": "Welcome screen",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "get_started", "path": "start.png"}]}
          }
        },
        "expectations": {
          "expected_state_after": "profile_setup"
        }
      },
      {
        "id": "step2",
        "name": "Profile setup",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "next", "path": "next.png"}]}
          }
        },
        "expectations": {
          "expected_state_after": "preferences"
        }
      },
      {
        "id": "step3",
        "name": "Set preferences",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "continue", "path": "continue.png"}]}
          }
        },
        "expectations": {
          "expected_state_after": "tutorial"
        }
      },
      {
        "id": "step4",
        "name": "Complete tutorial",
        "action": {
          "type": "CLICK",
          "options": {
            "target": {"images": [{"name": "finish", "path": "finish.png"}]}
          }
        },
        "expectations": {
          "expected_state_after": "dashboard",
          "capture_checkpoint_after": true,
          "checkpoint_name": "onboarding_complete"
        }
      }
    ]
  },
  "expectations": {
    "global": {
      "max_workflow_duration_ms": 90000,
      "screenshot_on_failure": true,
      "fail_fast": true
    },
    "success_criteria": [
      {
        "type": "required_states",
        "required_states": ["profile_setup", "preferences", "tutorial", "dashboard"],
        "description": "Must visit all onboarding states"
      }
    ],
    "checkpoints": {
      "onboarding_complete": {
        "description": "Verify onboarding completed and dashboard visible",
        "screenshot_required": true,
        "ocr_assertions": [
          {
            "type": "text_present",
            "text": "Welcome to your dashboard",
            "case_sensitive": false
          },
          {
            "type": "text_absent",
            "text": "Complete your profile",
            "case_sensitive": false
          }
        ],
        "claude_review": [
          "Verify the dashboard is fully loaded",
          "Check that all onboarding prompts are gone",
          "Ensure user navigation elements are present"
        ]
      }
    }
  }
}
```

---

## Common Use Cases

### 1. Strict Validation (All Actions Must Pass)

```json
{
  "success_criteria": [
    {
      "type": "all_actions_pass",
      "description": "Every step is critical"
    }
  ],
  "global": {
    "fail_fast": true,
    "screenshot_on_failure": true
  }
}
```

**When to use:**
- Critical business processes (payment, data deletion)
- Compliance testing
- Security-sensitive operations

---

### 2. Resilient Automation (Allow Some Failures)

```json
{
  "success_criteria": [
    {
      "type": "max_failures",
      "max_failures": 3,
      "description": "Allow up to 3 optional steps to fail"
    }
  ],
  "global": {
    "fail_fast": false
  }
}
```

**When to use:**
- Workflows with optional steps
- Best-effort data collection
- UI testing where some elements may be conditionally visible

---

### 3. Goal-Oriented Automation

```json
{
  "success_criteria": [
    {
      "type": "checkpoint_passed",
      "checkpoint_name": "final_state",
      "description": "Only care about reaching the final goal"
    }
  ],
  "checkpoints": {
    "final_state": {
      "screenshot_required": true,
      "ocr_assertions": [
        {
          "type": "text_present",
          "text": "Success"
        }
      ]
    }
  }
}
```

**When to use:**
- End-to-end testing focused on outcomes
- User journey validation
- Navigation flows where intermediate steps may vary

---

### 4. Data Collection (Minimum Matches Required)

```json
{
  "success_criteria": [
    {
      "type": "min_matches",
      "min_matches": 20,
      "description": "Must collect at least 20 data points"
    }
  ]
}
```

**When to use:**
- Web scraping
- Search result validation
- Inventory checking

---

### 5. Retry with Checkpoints

```json
{
  "steps": [
    {
      "id": "critical_action",
      "action": {"type": "CLICK"},
      "expectations": {
        "max_retries": 5,
        "retry_delay_ms": 2000,
        "capture_checkpoint_on_failure": true,
        "checkpoint_name": "retry_failure"
      }
    }
  ],
  "checkpoints": {
    "retry_failure": {
      "description": "Capture state when retries are exhausted",
      "screenshot_required": true,
      "claude_review": [
        "Analyze why the action keeps failing",
        "Check for UI blocking elements",
        "Look for error messages"
      ]
    }
  }
}
```

**When to use:**
- Flaky UI elements
- Network-dependent actions
- Async UI updates

---

### 6. Regional OCR Validation

```json
{
  "checkpoints": {
    "price_check": {
      "ocr_assertions": [
        {
          "type": "text_in_region",
          "text": "$",
          "region": {
            "x": 500,
            "y": 300,
            "width": 200,
            "height": 50
          }
        }
      ]
    }
  }
}
```

**When to use:**
- Validating specific UI sections
- Table cell validation
- Modal dialog content checking

---

### 7. Count-Based Validation

```json
{
  "checkpoints": {
    "cart_items": {
      "ocr_assertions": [
        {
          "type": "text_count",
          "text": "Item",
          "expected_count": 3
        }
      ]
    }
  }
}
```

**When to use:**
- Shopping cart validation
- List verification
- Repeated element checking

---

## Best Practices

### 1. Checkpoint Naming

Use clear, descriptive names that indicate what is being validated:

**Good:**
- `login_success`
- `payment_complete`
- `search_results_loaded`

**Bad:**
- `checkpoint1`
- `test`
- `check`

---

### 2. OCR Assertion Design

- **Be specific:** Use unique text that clearly indicates the state
- **Use multiple assertions:** Combine positive (text_present) and negative (text_absent) checks
- **Consider case sensitivity:** Most UI text should use `case_sensitive: false`
- **Use regions for precision:** When text appears in multiple places, use `text_in_region`

**Example:**

```json
{
  "ocr_assertions": [
    {
      "type": "text_present",
      "text": "Order confirmed",
      "case_sensitive": false
    },
    {
      "type": "text_absent",
      "text": "Error",
      "case_sensitive": false
    },
    {
      "type": "no_duplicate_matches",
      "text": "Order #"
    }
  ]
}
```

---

### 3. Success Criteria Selection

Choose the right criteria for your use case:

| Use Case | Criteria Type | Reason |
|----------|---------------|--------|
| Critical process | `all_actions_pass` | No failures tolerated |
| Optional steps | `max_failures` | Some steps can fail |
| Final state matters | `checkpoint_passed` | Path doesn't matter |
| Data collection | `min_matches` | Need minimum results |
| Journey validation | `required_states` | Must visit all states |

---

### 4. Timeout Configuration

Set appropriate timeouts at different levels:

```json
{
  "global": {
    "max_workflow_duration_ms": 120000,  // Overall workflow timeout
    "max_action_duration_ms": 10000      // Default action timeout
  },
  "steps": [
    {
      "id": "slow_action",
      "expectations": {
        "max_duration_ms": 30000  // Override for specific action
      }
    }
  ],
  "checkpoints": {
    "async_validation": {
      "max_wait_ms": 15000,      // Time to wait for assertions
      "retry_interval_ms": 1000   // Check every second
    }
  }
}
```

---

### 5. Retry Strategy

Configure retries for flaky actions:

```json
{
  "expectations": {
    "max_retries": 3,
    "retry_delay_ms": 2000,
    "capture_checkpoint_on_failure": true
  }
}
```

**Guidelines:**
- Use retries for network-dependent actions
- Increase `retry_delay_ms` for async operations
- Capture checkpoints to debug retry failures
- Don't retry actions with side effects (payments, submissions)

---

### 6. Screenshot Strategy

Control when screenshots are captured:

```json
{
  "global": {
    "screenshot_on_failure": true  // Always capture on workflow failure
  },
  "steps": [
    {
      "id": "important_step",
      "expectations": {
        "capture_checkpoint_after": true,  // Capture after success
        "capture_checkpoint_on_failure": true  // And on failure
      }
    }
  ]
}
```

**Guidelines:**
- Enable `screenshot_on_failure` globally for debugging
- Use `capture_checkpoint_after` for important state transitions
- Use named checkpoints for organized screenshot management

---

### 7. Claude Review Usage

Use Claude review for complex visual validation:

```json
{
  "claude_review": [
    "Verify the modal dialog is centered on screen",
    "Check that all form fields are properly aligned",
    "Ensure the submit button is not disabled",
    "Look for any visual glitches or rendering issues"
  ]
}
```

**When to use Claude review:**
- Layout validation
- Design consistency checks
- Complex UI state verification
- Accessibility validation

**When to use OCR instead:**
- Simple text presence/absence
- Known exact text values
- Performance-critical validation

---

### 8. Fail Fast vs. Fail Slow

**Fail Fast:**
```json
{
  "global": {
    "fail_fast": true
  },
  "success_criteria": [
    {
      "type": "all_actions_pass"
    }
  ]
}
```

Use when:
- Time is critical
- Early failures invalidate later steps
- Testing critical paths

**Fail Slow:**
```json
{
  "global": {
    "fail_fast": false
  },
  "success_criteria": [
    {
      "type": "max_failures",
      "max_failures": 5
    }
  ]
}
```

Use when:
- Want to collect multiple failure points
- Testing optional features
- Debugging workflow issues

---

### 9. Combining Multiple Criteria

Use multiple success criteria for OR logic:

```json
{
  "success_criteria": [
    {
      "type": "checkpoint_passed",
      "checkpoint_name": "success_path",
      "description": "Primary success condition"
    },
    {
      "type": "checkpoint_passed",
      "checkpoint_name": "alternate_path",
      "description": "Alternate success condition"
    }
  ]
}
```

The workflow passes if **ANY** criterion is satisfied.

---

### 10. Error Message Validation

Always check for absence of error messages:

```json
{
  "ocr_assertions": [
    {
      "type": "text_absent",
      "text": "error",
      "case_sensitive": false
    },
    {
      "type": "text_absent",
      "text": "failed",
      "case_sensitive": false
    },
    {
      "type": "text_absent",
      "text": "invalid",
      "case_sensitive": false
    }
  ]
}
```

---

## Validation Workflow

When a checkpoint is evaluated:

1. **Screenshot Capture** (if `screenshot_required: true`)
2. **OCR Extraction** - Extract text from screenshot
3. **OCR Assertions** - Evaluate all assertions
4. **Retry Logic** - If assertions fail, retry up to `max_wait_ms` with `retry_interval_ms` intervals
5. **Claude Review** (if specified) - Send screenshot and instructions to Claude
6. **Result Aggregation** - Combine all validation results

---

## Troubleshooting

### Common Issues

**Issue:** Checkpoint assertions always fail
- **Solution:** Check OCR text extraction quality, adjust `case_sensitive`, use broader text patterns

**Issue:** Workflow times out
- **Solution:** Increase `max_workflow_duration_ms` or individual action `max_duration_ms`

**Issue:** Too many false positives
- **Solution:** Add more specific OCR assertions, use `text_in_region` for precision

**Issue:** Claude review inconsistent
- **Solution:** Make instructions more specific, add context about what to look for

**Issue:** Retries exhausted
- **Solution:** Increase `max_retries`, increase `retry_delay_ms`, check if action has side effects

---

## Schema Versioning

Current schema version: **1.0.0**

The expectations schema follows semantic versioning. Breaking changes will increment the major version.

---

## Related Documentation

- [Workflow Schema Reference](./workflow-schema.md)
- [Action Types Reference](./action-types.md)
- [MCP Tool Reference](./mcp-tools.md)
- [State Machine Documentation](./state-machine.md)

---

## Support

For questions or issues with the expectations schema:
- GitHub Issues: https://github.com/qontinui/qontinui-mcp/issues
- Documentation: https://qontinui.github.io/multistate/
