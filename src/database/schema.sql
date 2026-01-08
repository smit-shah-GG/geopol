-- Schema for GDELT event storage with deduplication support
-- Designed for temporal knowledge graph construction

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- GDELT identifiers
    gdelt_id TEXT UNIQUE,

    -- Temporal fields
    event_date TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    time_window TEXT NOT NULL,  -- Hour-based window for deduplication

    -- Actor fields
    actor1_code TEXT,
    actor2_code TEXT,

    -- Event classification
    event_code TEXT,
    quad_class INTEGER,  -- 1=Verbal Cooperation, 2=Material Cooperation, 3=Verbal Conflict, 4=Material Conflict

    -- Event metrics
    goldstein_scale REAL,  -- Conflict/cooperation scale (-10 to +10)
    num_mentions INTEGER,   -- Number of mentions (for GDELT100 filtering)
    num_sources INTEGER,    -- Number of unique sources
    tone REAL,              -- Average tone of coverage

    -- Source information
    url TEXT,
    title TEXT,
    domain TEXT,

    -- Deduplication fields
    content_hash TEXT NOT NULL,  -- Hash of actor1+actor2+event_code+location

    -- Store raw JSON for future processing
    raw_json TEXT,

    -- Indexes for efficient queries
    CHECK (quad_class IN (1, 2, 3, 4) OR quad_class IS NULL)
);

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_event_date ON events(event_date);

-- Index for QuadClass filtering (conflicts and cooperation)
CREATE INDEX IF NOT EXISTS idx_quad_class ON events(quad_class);

-- Composite index for deduplication
CREATE UNIQUE INDEX IF NOT EXISTS idx_deduplication ON events(content_hash, time_window);

-- Index for actor-based queries
CREATE INDEX IF NOT EXISTS idx_actor1 ON events(actor1_code);
CREATE INDEX IF NOT EXISTS idx_actor2 ON events(actor2_code);

-- Index for high-confidence events (GDELT100)
CREATE INDEX IF NOT EXISTS idx_num_mentions ON events(num_mentions);

-- Index for tone-based filtering
CREATE INDEX IF NOT EXISTS idx_tone ON events(tone);

-- View for high-confidence conflict events
CREATE VIEW IF NOT EXISTS conflict_events AS
SELECT * FROM events
WHERE quad_class = 4  -- Material Conflict
  AND num_mentions >= 100  -- GDELT100 threshold
ORDER BY event_date DESC;

-- View for diplomatic events
CREATE VIEW IF NOT EXISTS diplomatic_events AS
SELECT * FROM events
WHERE quad_class = 1  -- Verbal Cooperation
  AND tone BETWEEN -2 AND 2  -- Neutral tone range
ORDER BY event_date DESC;

-- Statistics table for monitoring
CREATE TABLE IF NOT EXISTS ingestion_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_date TEXT NOT NULL DEFAULT (datetime('now')),
    events_fetched INTEGER,
    events_deduplicated INTEGER,
    events_inserted INTEGER,
    processing_time_seconds REAL
);