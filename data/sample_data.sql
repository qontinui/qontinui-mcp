-- ============================================================================
-- qontinui-mcp Sample Data
-- Example nodes, templates, and usage statistics for testing and development
-- ============================================================================

-- ============================================================================
-- SAMPLE NODES
-- ============================================================================

-- ACTION NODES

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_click_001',
  'CLICK',
  'action',
  'Click Element',
  'Clicks on a UI element identified by image or text matching',
  json('{"target": {"type": "string", "required": true, "description": "Image path or text to match"}, "offsetX": {"type": "number", "default": 0}, "offsetY": {"type": "number", "default": 0}, "clickType": {"type": "enum", "values": ["left", "right", "double"], "default": "left"}}'),
  json('[{"title": "Click Button", "code": "CLICK(\"submit_button.png\")"}, {"title": "Right Click Menu", "code": "CLICK(\"menu_icon.png\", clickType=\"right\")"}]'),
  '# CLICK Node\n\nClicks on UI elements using image or text recognition.\n\n## Parameters\n- **target**: Image path or text to find\n- **offsetX/Y**: Pixel offset from match center\n- **clickType**: left, right, or double click\n\n## Use Cases\n- Button clicks\n- Menu navigation\n- Form submission',
  'cursor-click',
  '#3B82F6',
  json('["click", "mouse", "interaction", "ui", "button"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_type_002',
  'TYPE',
  'action',
  'Type Text',
  'Types text into the currently focused input field or at a specific location',
  json('{"text": {"type": "string", "required": true, "description": "Text to type"}, "target": {"type": "string", "required": false, "description": "Optional image/text to click before typing"}, "pressEnter": {"type": "boolean", "default": false}, "clearFirst": {"type": "boolean", "default": false}}'),
  json('[{"title": "Type in Search Box", "code": "TYPE(\"search query\", target=\"search_box.png\")"}, {"title": "Type and Submit", "code": "TYPE(\"username\", pressEnter=true)"}]'),
  '# TYPE Node\n\nTypes text into input fields.\n\n## Parameters\n- **text**: Text to type\n- **target**: Optional element to click first\n- **pressEnter**: Press Enter after typing\n- **clearFirst**: Clear field before typing\n\n## Use Cases\n- Form filling\n- Search queries\n- Text input',
  'keyboard',
  '#10B981',
  json('["type", "keyboard", "input", "text", "form"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_wait_003',
  'WAIT',
  'action',
  'Wait',
  'Waits for a condition to be met or for a specified duration',
  json('{"duration": {"type": "number", "description": "Seconds to wait"}, "target": {"type": "string", "description": "Image/text to wait for"}, "timeout": {"type": "number", "default": 30}, "condition": {"type": "enum", "values": ["appear", "disappear"], "default": "appear"}}'),
  json('[{"title": "Wait 2 Seconds", "code": "WAIT(duration=2)"}, {"title": "Wait for Element", "code": "WAIT(target=\"loading.png\", condition=\"disappear\")"}]'),
  '# WAIT Node\n\nWaits for conditions or time delays.\n\n## Parameters\n- **duration**: Fixed time to wait (seconds)\n- **target**: Element to wait for\n- **condition**: Wait for appear/disappear\n- **timeout**: Maximum wait time\n\n## Use Cases\n- Loading screens\n- Async operations\n- Animation delays',
  'clock',
  '#F59E0B',
  json('["wait", "delay", "timeout", "synchronization"]')
);

-- CONTROL FLOW NODES

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_if_004',
  'IF',
  'control-flow',
  'If Condition',
  'Conditional branching based on element presence or variable values',
  json('{"condition": {"type": "string", "required": true, "description": "Condition to evaluate"}, "target": {"type": "string", "description": "Image/text to check for"}, "operator": {"type": "enum", "values": ["exists", "not_exists", "equals", "greater_than", "less_than"], "default": "exists"}}'),
  json('[{"title": "If Element Exists", "code": "IF(target=\"popup.png\", operator=\"exists\") { CLICK(\"close.png\") }"}, {"title": "If Variable Equals", "code": "IF(condition=\"count == 10\") { ... }"}]'),
  '# IF Node\n\nConditional branching logic.\n\n## Parameters\n- **condition**: Expression to evaluate\n- **target**: Optional element to check\n- **operator**: Comparison operator\n\n## Use Cases\n- Error handling\n- Adaptive workflows\n- State-based logic',
  'git-branch',
  '#8B5CF6',
  json('["if", "condition", "branch", "logic", "control-flow"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_loop_005',
  'LOOP',
  'control-flow',
  'Loop',
  'Repeats actions for a specified count or while a condition is true',
  json('{"count": {"type": "number", "description": "Number of iterations"}, "condition": {"type": "string", "description": "Condition to continue looping"}, "maxIterations": {"type": "number", "default": 100}, "breakOn": {"type": "string", "description": "Element that stops loop"}}'),
  json('[{"title": "Loop 10 Times", "code": "LOOP(count=10) { CLICK(\"next.png\") }"}, {"title": "Loop Until Done", "code": "LOOP(breakOn=\"done.png\") { ... }"}]'),
  '# LOOP Node\n\nRepeat actions multiple times.\n\n## Parameters\n- **count**: Fixed iteration count\n- **condition**: Loop while true\n- **maxIterations**: Safety limit\n- **breakOn**: Element to break loop\n\n## Use Cases\n- Batch processing\n- Data extraction\n- Pagination',
  'repeat',
  '#EC4899',
  json('["loop", "repeat", "iterate", "while", "control-flow"]')
);

-- DATA OPERATIONS NODES

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_extract_006',
  'EXTRACT',
  'data-operations',
  'Extract Data',
  'Extracts text or data from screen region and stores in variable',
  json('{"region": {"type": "string", "required": true, "description": "Screen region to extract from"}, "variable": {"type": "string", "required": true, "description": "Variable name to store result"}, "pattern": {"type": "string", "description": "Regex pattern to match"}, "ocrEngine": {"type": "enum", "values": ["tesseract", "easyocr"], "default": "tesseract"}}'),
  json('[{"title": "Extract Text", "code": "EXTRACT(region=\"price_area.png\", variable=\"price\")"}, {"title": "Extract with Pattern", "code": "EXTRACT(region=\"invoice.png\", variable=\"invoice_num\", pattern=\"INV-[0-9]+\")"}]'),
  '# EXTRACT Node\n\nExtracts text data from screen.\n\n## Parameters\n- **region**: Area to extract from\n- **variable**: Storage variable name\n- **pattern**: Optional regex filter\n- **ocrEngine**: OCR engine to use\n\n## Use Cases\n- Data scraping\n- Form reading\n- Invoice processing',
  'scissors',
  '#06B6D4',
  json('["extract", "ocr", "data", "scrape", "text"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_set_007',
  'SET',
  'data-operations',
  'Set Variable',
  'Sets a variable to a specific value for use in the workflow',
  json('{"variable": {"type": "string", "required": true, "description": "Variable name"}, "value": {"type": "any", "required": true, "description": "Value to set"}, "type": {"type": "enum", "values": ["string", "number", "boolean", "json"], "default": "string"}}'),
  json('[{"title": "Set Counter", "code": "SET(variable=\"counter\", value=0, type=\"number\")"}, {"title": "Set Config", "code": "SET(variable=\"config\", value={\"retry\": 3}, type=\"json\")"}]'),
  '# SET Node\n\nSets workflow variables.\n\n## Parameters\n- **variable**: Variable name\n- **value**: Value to assign\n- **type**: Data type\n\n## Use Cases\n- Configuration\n- Counters\n- State management',
  'variable',
  '#14B8A6',
  json('["set", "variable", "assign", "data", "state"]')
);

-- TRIGGER NODES

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_schedule_008',
  'SCHEDULE',
  'trigger',
  'Schedule Trigger',
  'Triggers workflow execution on a schedule (cron-like)',
  json('{"cron": {"type": "string", "required": true, "description": "Cron expression"}, "timezone": {"type": "string", "default": "UTC"}, "enabled": {"type": "boolean", "default": true}}'),
  json('[{"title": "Daily at 9 AM", "code": "SCHEDULE(cron=\"0 9 * * *\")"}, {"title": "Every 15 Minutes", "code": "SCHEDULE(cron=\"*/15 * * * *\")"}]'),
  '# SCHEDULE Node\n\nTriggers workflows on schedule.\n\n## Parameters\n- **cron**: Cron expression\n- **timezone**: Timezone for schedule\n- **enabled**: Enable/disable trigger\n\n## Use Cases\n- Automated reports\n- Periodic checks\n- Batch jobs',
  'calendar-clock',
  '#EF4444',
  json('["schedule", "cron", "trigger", "automation", "timer"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_watch_009',
  'WATCH',
  'trigger',
  'Watch Element',
  'Triggers workflow when a specific element appears on screen',
  json('{"target": {"type": "string", "required": true, "description": "Image/text to watch for"}, "interval": {"type": "number", "default": 1, "description": "Check interval in seconds"}, "region": {"type": "string", "description": "Screen region to watch"}}'),
  json('[{"title": "Watch for Error", "code": "WATCH(target=\"error_dialog.png\")"}, {"title": "Watch Notification", "code": "WATCH(target=\"new_message.png\", interval=5)"}]'),
  '# WATCH Node\n\nTriggers on element appearance.\n\n## Parameters\n- **target**: Element to watch for\n- **interval**: Check frequency\n- **region**: Optional screen region\n\n## Use Cases\n- Error handling\n- Notifications\n- State monitoring',
  'eye',
  '#F97316',
  json('["watch", "monitor", "trigger", "observe", "event"]')
);

INSERT INTO nodes (id, type, category, name, description, properties, examples, documentation, icon, color, tags)
VALUES (
  'node_screenshot_010',
  'SCREENSHOT',
  'action',
  'Take Screenshot',
  'Captures a screenshot of the entire screen or a specific region',
  json('{"filepath": {"type": "string", "required": true, "description": "Path to save screenshot"}, "region": {"type": "string", "description": "Optional region to capture"}, "format": {"type": "enum", "values": ["png", "jpg"], "default": "png"}}'),
  json('[{"title": "Full Screenshot", "code": "SCREENSHOT(filepath=\"screen.png\")"}, {"title": "Region Screenshot", "code": "SCREENSHOT(filepath=\"region.png\", region=\"dialog.png\")"}]'),
  '# SCREENSHOT Node\n\nCaptures screen images.\n\n## Parameters\n- **filepath**: Save path\n- **region**: Optional capture area\n- **format**: Image format\n\n## Use Cases\n- Documentation\n- Error evidence\n- Monitoring',
  'camera',
  '#6366F1',
  json('["screenshot", "capture", "image", "evidence"]')
);

-- ============================================================================
-- SAMPLE TEMPLATES
-- ============================================================================

INSERT INTO templates (id, name, description, category, workflow, format, tags, usage_count)
VALUES (
  'template_login_001',
  'Login Flow',
  'Standard login workflow with username and password',
  'gui-automation',
  json('[{"type": "CLICK", "params": {"target": "username_field.png"}}, {"type": "TYPE", "params": {"text": "{username}", "clearFirst": true}}, {"type": "CLICK", "params": {"target": "password_field.png"}}, {"type": "TYPE", "params": {"text": "{password}", "clearFirst": true}}, {"type": "CLICK", "params": {"target": "login_button.png"}}, {"type": "WAIT", "params": {"target": "dashboard.png", "timeout": 10}}]'),
  'sequential',
  json('["login", "authentication", "credentials"]'),
  45
);

INSERT INTO templates (id, name, description, category, workflow, format, tags, usage_count)
VALUES (
  'template_data_extract_002',
  'Extract Table Data',
  'Extracts data from a table with pagination',
  'data-processing',
  json('[{"type": "SET", "params": {"variable": "page", "value": 1, "type": "number"}}, {"type": "LOOP", "params": {"breakOn": "next_disabled.png", "maxIterations": 100}, "children": [{"type": "EXTRACT", "params": {"region": "table.png", "variable": "page_data"}}, {"type": "CLICK", "params": {"target": "next_page.png"}}, {"type": "WAIT", "params": {"duration": 1}}]}]'),
  'sequential',
  json('["extract", "table", "pagination", "data"]'),
  23
);

INSERT INTO templates (id, name, description, category, workflow, format, tags, usage_count)
VALUES (
  'template_error_handler_003',
  'Error Handler',
  'Checks for error dialogs and handles them',
  'control-flow',
  json('[{"type": "IF", "params": {"target": "error_dialog.png", "operator": "exists"}, "children": [{"type": "SCREENSHOT", "params": {"filepath": "error_{timestamp}.png"}}, {"type": "CLICK", "params": {"target": "ok_button.png"}}, {"type": "WAIT", "params": {"duration": 1}}]}]'),
  'sequential',
  json('["error", "handler", "dialog", "recovery"]'),
  67
);

INSERT INTO templates (id, name, description, category, workflow, format, tags, usage_count)
VALUES (
  'template_batch_process_004',
  'Batch File Processor',
  'Processes multiple files in a folder sequentially',
  'data-processing',
  json('[{"type": "SET", "params": {"variable": "file_count", "value": 0, "type": "number"}}, {"type": "LOOP", "params": {"breakOn": "no_more_files.png"}, "children": [{"type": "CLICK", "params": {"target": "file_item.png"}}, {"type": "CLICK", "params": {"target": "process_button.png"}}, {"type": "WAIT", "params": {"target": "process_complete.png", "timeout": 60}}, {"type": "SET", "params": {"variable": "file_count", "value": "{file_count + 1}", "type": "number"}}]}]'),
  'sequential',
  json('["batch", "files", "processing", "loop"]'),
  12
);

INSERT INTO templates (id, name, description, category, workflow, format, tags, usage_count)
VALUES (
  'template_form_fill_005',
  'Multi-Page Form Filler',
  'Fills out a multi-page form with validation',
  'gui-automation',
  json('[{"type": "CLICK", "params": {"target": "name_field.png"}}, {"type": "TYPE", "params": {"text": "{name}"}}, {"type": "CLICK", "params": {"target": "email_field.png"}}, {"type": "TYPE", "params": {"text": "{email}"}}, {"type": "CLICK", "params": {"target": "next_button.png"}}, {"type": "WAIT", "params": {"target": "page2.png"}}, {"type": "CLICK", "params": {"target": "address_field.png"}}, {"type": "TYPE", "params": {"text": "{address}"}}, {"type": "CLICK", "params": {"target": "submit_button.png"}}, {"type": "IF", "params": {"target": "success.png", "operator": "exists"}, "children": [{"type": "SCREENSHOT", "params": {"filepath": "success.png"}}]}]'),
  'sequential',
  json('["form", "input", "multi-page", "validation"]'),
  34
);

-- ============================================================================
-- SAMPLE NODE USAGE STATISTICS
-- ============================================================================

INSERT INTO node_usage (node_type, usage_count, last_used) VALUES
  ('CLICK', 1523, datetime('now', '-2 hours')),
  ('TYPE', 892, datetime('now', '-1 hour')),
  ('WAIT', 756, datetime('now', '-3 hours')),
  ('IF', 234, datetime('now', '-5 hours')),
  ('LOOP', 189, datetime('now', '-1 day')),
  ('EXTRACT', 145, datetime('now', '-2 days')),
  ('SET', 98, datetime('now', '-4 hours')),
  ('SCREENSHOT', 67, datetime('now', '-6 hours')),
  ('SCHEDULE', 23, datetime('now', '-3 days')),
  ('WATCH', 12, datetime('now', '-1 week'));

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify nodes inserted
SELECT 'Nodes Count:' as label, COUNT(*) as value FROM nodes
UNION ALL
SELECT 'Templates Count:', COUNT(*) FROM templates
UNION ALL
SELECT 'Usage Records:', COUNT(*) FROM node_usage;

-- Verify FTS5 index populated
SELECT 'FTS5 Records:' as label, COUNT(*) as value FROM nodes_fts;

-- Show sample search results
SELECT 'Sample Search Results:' as label, '' as value
UNION ALL
SELECT '  ' || type || ' - ' || name, '' FROM nodes_fts WHERE nodes_fts MATCH 'click' LIMIT 3;
