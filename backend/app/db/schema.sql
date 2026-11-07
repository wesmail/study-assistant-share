-- Study-assistant storage: the tables plus two virtual indexes. The sqlite-vec extension must be loaded before this
-- file runs; app.db.connection.connect() does that.
--
-- Every statement is IF NOT EXISTS, so applying the file is idempotent.

-- One row per library file. The row outlives re-ingestion, because notes and
-- annotations point at it: a changed file updates sha256 in place.
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,      -- relative to the library folder
    sha256      TEXT NOT NULL,
    title       TEXT,                      -- never the file name: app/ingest/metadata.py
    author      TEXT,
    year        INTEGER,
    publisher   TEXT,
    kind        TEXT NOT NULL CHECK (kind IN ('pdf', 'pptx')),
    pages       INTEGER,
    status      TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'parsing', 'chunking', 'indexing',
                                  'indexed', 'failed', 'missing')),
    collection  TEXT,                      -- 'test': the evaluation books, which a library setting can hide
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

-- Every emitted Markdown block with its position: the citation anchor.
-- bbox is JSON [x0, y0, x1, y1] in page points, y measured from the page top,
-- matching PyMuPDF and PDF.js. Multi-item blocks keep extra regions in
-- bboxes_json (a formula split across a page break has two).
CREATE TABLE IF NOT EXISTS blocks (
    id           INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    "order"      INTEGER NOT NULL,
    kind         TEXT NOT NULL,            -- heading, paragraph, list_item,
                                           -- equation, code, table, figure, ...
    page         INTEGER,                  -- 1-based PDF page
    bbox         TEXT,
    bboxes_json  TEXT,
    text         TEXT NOT NULL,
    docling_ref  TEXT,
    UNIQUE (doc_id, "order")
);
CREATE INDEX IF NOT EXISTS idx_blocks_doc_page ON blocks(doc_id, page);

-- Retrieval units, including summary-tree nodes (level 1 = a section, chapter or part, 2 = the whole document).
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('text', 'figure', 'summary', 'note')),
    level         INTEGER NOT NULL DEFAULT 0,
    seq           INTEGER NOT NULL,        -- reading order within the document
    heading_path  TEXT NOT NULL DEFAULT '',
    text          TEXT NOT NULL,           -- heading-path prefix + body
    token_count   INTEGER NOT NULL,
    block_ids     TEXT NOT NULL DEFAULT '[]',  -- JSON list of blocks.id
    page_start    INTEGER,
    page_end      INTEGER,
    meta          TEXT                     -- JSON, e.g. image path for figures
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, kind, seq);

-- Keyword index. External content: FTS5 stores the inverted index only and
-- reads text from `chunks`, so the triggers below must keep the two in step.
-- porter stems English; German stems oddly but consistently, so it still
-- matches itself. remove_diacritics folds accents and umlauts (Lorenz/Lorénz).
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'porter unicode61 remove_diacritics 2'
);

-- Vector index: bge-m3, 1024 floats, exact cosine search.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[1024] distance_metric=cosine
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
    DELETE FROM chunk_vec WHERE chunk_id = old.id;
END;

-- Changed text invalidates the embedding, so the vector goes too; the index
-- stage re-embeds any chunk that has no vector.
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunk_fts (rowid, text) VALUES (new.id, new.text);
    DELETE FROM chunk_vec WHERE chunk_id = old.id;
END;

CREATE TABLE IF NOT EXISTS figures (
    id           INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page         INTEGER,
    image_path   TEXT,
    caption      TEXT,
    description  TEXT,
    image_sha    TEXT
);
CREATE INDEX IF NOT EXISTS idx_figures_doc ON figures(doc_id, page);
CREATE INDEX IF NOT EXISTS idx_figures_sha ON figures(image_sha);

-- Figure descriptions, cached by image hash so that re-ingesting a book, or the same
-- image in two places, costs nothing. A new describer or prompt version is a new row.
CREATE TABLE IF NOT EXISTS figure_descriptions (
    image_sha       TEXT NOT NULL,
    describer       TEXT NOT NULL,             -- provider:model
    prompt_version  INTEGER NOT NULL,
    text            TEXT NOT NULL,
    cost_usd        REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (image_sha, describer, prompt_version)
);

-- Summary-tree texts, cached by a hash of everything the summarizer was given (model,
-- prompt and input), so a rebuild with unchanged text costs nothing.
CREATE TABLE IF NOT EXISTS summary_cache (
    input_sha       TEXT PRIMARY KEY,
    summarizer      TEXT NOT NULL,             -- provider:model
    prompt_version  INTEGER NOT NULL,
    text            TEXT NOT NULL,
    seconds         REAL NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    doc_id          INTEGER                    -- the document it was written for; removing it removes the row
);

-- The reader's library settings, such as hiding the test collection. Values are JSON.
CREATE TABLE IF NOT EXISTS app_settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    bbox        TEXT NOT NULL,             -- JSON [x0, y0, x1, y1]: the union of rects
    rects       TEXT,                      -- JSON list of boxes, one per line, PDF points from the top left
    quote       TEXT NOT NULL,
    prefix      TEXT,                      -- the words just before and after the quote, to re-anchor it
    suffix      TEXT,
    anchor_sha  TEXT,                      -- the file the anchor was made on (app/anchors.py)
    anchor_status TEXT,                    -- NULL or 'ok', 'moved', 'unanchored' after the file changed
    batch       TEXT,                      -- AI suggestions made together (app/suggest.py)
    instruction TEXT,                      -- what the reader asked for, for AI suggestions
    kind        TEXT NOT NULL,
    color       TEXT,
    body        TEXT,
    origin      TEXT NOT NULL CHECK (origin IN ('user', 'ai')),
    status      TEXT NOT NULL CHECK (status IN ('suggested', 'accepted')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_annotations_doc ON annotations(doc_id, page);

CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    doc_title   TEXT,                      -- the book's title when the note was written: it outlives the book
    page        INTEGER,
    bbox        TEXT,                      -- JSON [x0, y0, x1, y1]: the union of rects
    rects       TEXT,                      -- JSON list of boxes, one per line, as for annotations
    quote       TEXT,
    prefix      TEXT,
    suffix      TEXT,
    anchor_sha  TEXT,
    anchor_status TEXT,
    body_md     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY,
    title       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    model            TEXT,
    hosted           INTEGER NOT NULL DEFAULT 0,
    citations        TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    option           TEXT,                 -- the answer option it was asked with
    details          TEXT                  -- JSON about the turn, for the feedback list (app/feedback.py)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

-- The reader's rating of one answer (roadmap item 2), with a copy of the turn taken when it
-- was rated, so the review list doesn't depend on the conversation. One per question:
-- rating again changes it. `candidate` marks a thumbs-down as a gold-question candidate,
-- written to data/gold_candidates.yaml (app/feedback.py).
CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY,
    question_id  INTEGER UNIQUE REFERENCES messages(id) ON DELETE SET NULL,
    answer_id    INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    rating       TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    note         TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    candidate    INTEGER NOT NULL DEFAULT 0,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL DEFAULT '',
    error        TEXT,
    option       TEXT,
    model        TEXT,
    hosted       INTEGER NOT NULL DEFAULT 0,
    details      TEXT NOT NULL DEFAULT '{}',   -- sources, web pages, served_by: as in messages.details
    asked_at     TEXT NOT NULL,
    answered_at  TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_list ON feedback(status, rating, id);

-- One row per call to the hosted model, for the audit and the bill. No prompt or
-- answer text, and no key.
CREATE TABLE IF NOT EXISTS hosted_calls (
    id                  INTEGER PRIMARY KEY,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    purpose             TEXT NOT NULL,             -- chat, or eval:<run name>
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,          -- thinking included
    cache_read_tokens   INTEGER NOT NULL,
    cache_write_tokens  INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    seconds             REAL NOT NULL,
    stop_reason         TEXT,                      -- empty if it failed or was interrupted
    error               TEXT,
    turn                TEXT                       -- the question it was for, so its cost adds up
);
CREATE INDEX IF NOT EXISTS idx_hosted_calls_turn ON hosted_calls(turn);

-- Resumable ingestion state: one row per stage of one file version. A changed
-- file gets a new sha256 and so a new set of rows; the UNIQUE key is what makes
-- a repeated scan harmless.
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sha256       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    stage_order  INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'running', 'done', 'failed')),
    progress     INTEGER NOT NULL DEFAULT 0,
    total        INTEGER,
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    waiting      TEXT,                      -- 'approval': a pending job that runs only once the reader agrees
    detail       TEXT,                      -- JSON: what it waits on (figures, estimated cost), or what it did
    approved_usd REAL,                      -- the estimate the reader agreed to
    UNIQUE (doc_id, sha256, stage)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, doc_id, stage_order);

-- The notes' own search index (roadmap item 4, step 3): kept apart from chunks, whose rows
-- go with their document, since a note outlives its book. note_fts holds the note, its
-- quote, tags and book title (rowid = note id); note_vec its bge-m3 vector, which
-- notes.index_notes fills for any note without one, before notes are searched.
CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
    text,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS note_vec USING vec0(
    note_id   INTEGER PRIMARY KEY,
    embedding FLOAT[1024] distance_metric=cosine
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO note_fts (rowid, text) VALUES (new.id, new.body_md || ' ' || coalesce(new.quote, '') || ' '
        || new.tags || ' ' || coalesce(new.doc_title, ''));
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    DELETE FROM note_fts WHERE rowid = old.id;
    DELETE FROM note_vec WHERE note_id = old.id;
END;

-- Changed words: the keywords are rewritten and the vector goes, to be made again.
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE OF body_md, quote, tags ON notes BEGIN
    DELETE FROM note_fts WHERE rowid = old.id;
    INSERT INTO note_fts (rowid, text) VALUES (new.id, new.body_md || ' ' || coalesce(new.quote, '') || ' '
        || new.tags || ' ' || coalesce(new.doc_title, ''));
    DELETE FROM note_vec WHERE note_id = old.id;
END;
