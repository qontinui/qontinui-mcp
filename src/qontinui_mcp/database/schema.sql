-- SQLite schema for Qontinui MCP server
-- Stores node metadata and enables FTS5 full-text search

-- Create nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    action_type TEXT NOT NULL,
    parameters TEXT NOT NULL, -- JSON array of NodeParameter
    examples TEXT, -- JSON array of example strings
    tags TEXT, -- JSON array of tags
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT, -- JSON array
    complexity TEXT CHECK(complexity IN ('simple', 'medium', 'complex')),
    template TEXT NOT NULL, -- JSON workflow object
    use_cases TEXT, -- JSON array
    customization_points TEXT, -- JSON array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create FTS5 virtual table for node search
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    name,
    category,
    description,
    action_type,
    tags,
    examples,
    content='nodes',
    content_rowid='rowid'
);

-- Create FTS5 virtual table for workflow search
CREATE VIRTUAL TABLE IF NOT EXISTS workflows_fts USING fts5(
    id UNINDEXED,
    name,
    category,
    description,
    tags,
    use_cases,
    content='workflows',
    content_rowid='rowid'
);

-- Triggers to keep FTS5 tables in sync with main tables

-- Node insert trigger
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, name, category, description, action_type, tags, examples)
    VALUES (new.rowid, new.id, new.name, new.category, new.description, new.action_type, new.tags, new.examples);
END;

-- Node update trigger
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    UPDATE nodes_fts SET
        name = new.name,
        category = new.category,
        description = new.description,
        action_type = new.action_type,
        tags = new.tags,
        examples = new.examples
    WHERE rowid = new.rowid;
END;

-- Node delete trigger
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    DELETE FROM nodes_fts WHERE rowid = old.rowid;
END;

-- Workflow insert trigger
CREATE TRIGGER IF NOT EXISTS workflows_ai AFTER INSERT ON workflows BEGIN
    INSERT INTO workflows_fts(rowid, id, name, category, description, tags, use_cases)
    VALUES (new.rowid, new.id, new.name, new.category, new.description, new.tags, new.use_cases);
END;

-- Workflow update trigger
CREATE TRIGGER IF NOT EXISTS workflows_au AFTER UPDATE ON workflows BEGIN
    UPDATE workflows_fts SET
        name = new.name,
        category = new.category,
        description = new.description,
        tags = new.tags,
        use_cases = new.use_cases
    WHERE rowid = new.rowid;
END;

-- Workflow delete trigger
CREATE TRIGGER IF NOT EXISTS workflows_ad AFTER DELETE ON workflows BEGIN
    DELETE FROM workflows_fts WHERE rowid = old.rowid;
END;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category);
CREATE INDEX IF NOT EXISTS idx_nodes_action_type ON nodes(action_type);
CREATE INDEX IF NOT EXISTS idx_workflows_category ON workflows(category);
CREATE INDEX IF NOT EXISTS idx_workflows_complexity ON workflows(complexity);
