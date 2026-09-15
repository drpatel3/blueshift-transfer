"""
Database layer for the mineral processing pipeline.

SQLite-backed storage with Pydantic validation. Replaces JSON files
as the system of record while maintaining parallel JSON writes during transition.

Database: data/pipeline.db
"""

import hashlib
import json
import sqlite3
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

import config


# ---------------------------------------------------------------------------
# Pydantic models — validate data at pipeline boundaries
# ---------------------------------------------------------------------------

class Stage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    order: int


class Unit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    label: str
    type: str
    stage_id: str
    cells: int = 1


class Connection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    parent_id: str
    child_id: str


class FlowsheetData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stages: list[Stage] = Field(default_factory=list)
    units: list[Unit] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image_id: str
    document_id: str
    image_type: str  # "flowsheet" or "other"
    page_number: Optional[int] = None
    file_path: Optional[str] = None
    caption: Optional[str] = None
    text_before: Optional[str] = None
    text_after: Optional[str] = None
    embedding: Optional[list[float]] = None


class TableRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    table_id: str
    document_id: str
    page_number: int
    table_index: int = 0
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    caption: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    text_before: Optional[str] = None
    text_after: Optional[str] = None


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    file_path        TEXT,
    file_size_bytes  INTEGER,
    document_version INTEGER DEFAULT 1,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    document_id      TEXT REFERENCES documents(document_id),
    started_at       TEXT DEFAULT (datetime('now')),
    finished_at      TEXT,
    status           TEXT DEFAULT 'running',
    steps_completed  TEXT DEFAULT '[]',
    error_summary    TEXT
);

CREATE TABLE IF NOT EXISTS images (
    image_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id),
    run_id           TEXT REFERENCES pipeline_runs(run_id),
    image_type       TEXT NOT NULL,
    page_number      INTEGER,
    file_path        TEXT,
    caption          TEXT,
    text_before      TEXT,
    text_after       TEXT,
    embedding        TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS flowsheets (
    image_id         TEXT PRIMARY KEY REFERENCES images(image_id),
    run_id           TEXT REFERENCES pipeline_runs(run_id),
    stages_json      TEXT NOT NULL DEFAULT '[]',
    units_json       TEXT NOT NULL DEFAULT '[]',
    connections_json TEXT NOT NULL DEFAULT '[]',
    llm_model        TEXT,
    llm_tokens       INTEGER,
    llm_cost_usd     REAL,
    error            TEXT,
    raw_response     TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extracted_tables (
    table_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id),
    run_id           TEXT REFERENCES pipeline_runs(run_id),
    page_number      INTEGER NOT NULL,
    table_index      INTEGER DEFAULT 0,
    section_number   TEXT,
    section_title    TEXT,
    caption          TEXT,
    headers_json     TEXT,
    rows_json        TEXT NOT NULL,
    text_before      TEXT,
    text_after       TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS normalization_dict (
    normalized_id    TEXT NOT NULL,
    original_id      TEXT NOT NULL,
    first_seen_run   TEXT REFERENCES pipeline_runs(run_id),
    PRIMARY KEY (normalized_id, original_id)
);

CREATE TABLE IF NOT EXISTS step_errors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    step_name        TEXT NOT NULL,
    image_id         TEXT,
    page_number      INTEGER,
    error_type       TEXT NOT NULL,
    error_message    TEXT NOT NULL,
    traceback        TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

def _default_db_path() -> str:
    return config.DB_PATH


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Return a connection with row_factory set to sqlite3.Row."""
    if db_path is None:
        db_path = _default_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = None) -> None:
    """Create all tables if they don't exist. Creates data/ directory if needed."""
    if db_path is None:
        db_path = _default_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------

def compute_document_id(file_path: str) -> str:
    """SHA-256 hash of file contents. Deterministic — same file always returns same ID."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_document(conn: sqlite3.Connection, document_id: str,
                    filename: str, file_path: str = None,
                    file_size: int = None) -> int:
    """Insert or update a document. Returns the document_version."""
    row = conn.execute(
        "SELECT document_version FROM documents WHERE document_id = ?",
        (document_id,)
    ).fetchone()

    if row is None:
        conn.execute(
            """INSERT INTO documents (document_id, filename, file_path, file_size_bytes, document_version)
               VALUES (?, ?, ?, ?, 1)""",
            (document_id, filename, file_path, file_size)
        )
        conn.commit()
        return 1
    else:
        new_version = row["document_version"] + 1
        conn.execute(
            """UPDATE documents
               SET document_version = ?, file_path = ?, file_size_bytes = ?,
                   updated_at = datetime('now')
               WHERE document_id = ?""",
            (new_version, file_path, file_size, document_id)
        )
        conn.commit()
        return new_version


# ---------------------------------------------------------------------------
# Pipeline runs
# ---------------------------------------------------------------------------

def start_run(conn: sqlite3.Connection, document_id: str = None) -> str:
    """Create a new pipeline run. Returns the run_id (UUID4)."""
    run_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO pipeline_runs (run_id, document_id, status)
           VALUES (?, ?, 'running')""",
        (run_id, document_id)
    )
    conn.commit()
    return run_id


def complete_run(conn: sqlite3.Connection, run_id: str,
                 status: str = "completed") -> None:
    """Mark a pipeline run as finished."""
    conn.execute(
        """UPDATE pipeline_runs
           SET status = ?, finished_at = datetime('now')
           WHERE run_id = ?""",
        (status, run_id)
    )
    conn.commit()


def mark_step_completed(conn: sqlite3.Connection, run_id: str,
                        step_name: str) -> None:
    """Append a step name to the steps_completed JSON array."""
    row = conn.execute(
        "SELECT steps_completed FROM pipeline_runs WHERE run_id = ?",
        (run_id,)
    ).fetchone()
    if row is None:
        return
    steps = json.loads(row["steps_completed"])
    if step_name not in steps:
        steps.append(step_name)
    conn.execute(
        "UPDATE pipeline_runs SET steps_completed = ? WHERE run_id = ?",
        (json.dumps(steps), run_id)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def log_step_error(conn: sqlite3.Connection, run_id: str, step_name: str,
                   error: Exception, image_id: str = None,
                   page_number: int = None) -> None:
    """Log a step-level error to the database."""
    conn.execute(
        """INSERT INTO step_errors (run_id, step_name, image_id, page_number,
                                    error_type, error_message, traceback)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, step_name, image_id, page_number,
         type(error).__name__, str(error), traceback.format_exc())
    )
    conn.commit()


def get_errors_for_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Return all errors logged for a pipeline run."""
    rows = conn.execute(
        """SELECT step_name, image_id, page_number, error_type, error_message,
                  traceback, created_at
           FROM step_errors WHERE run_id = ? ORDER BY id""",
        (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Image storage
# ---------------------------------------------------------------------------

def save_image_record(conn: sqlite3.Connection, image: ImageRecord,
                      run_id: str = None) -> None:
    """Save an image record (flowsheet or other). Idempotent via INSERT OR REPLACE."""
    embedding_json = json.dumps(image.embedding) if image.embedding else None
    conn.execute(
        """INSERT OR REPLACE INTO images
           (image_id, document_id, run_id, image_type, page_number,
            file_path, caption, text_before, text_after, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (image.image_id, image.document_id, run_id, image.image_type,
         image.page_number, image.file_path, image.caption,
         image.text_before, image.text_after, embedding_json)
    )
    conn.commit()


def get_images_for_document(conn: sqlite3.Connection, document_id: str,
                            image_type: str = None) -> list[dict]:
    """Retrieve images for a document, optionally filtered by type."""
    if image_type:
        rows = conn.execute(
            "SELECT * FROM images WHERE document_id = ? AND image_type = ?",
            (document_id, image_type)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM images WHERE document_id = ?",
            (document_id,)
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("embedding"):
            d["embedding"] = json.loads(d["embedding"])
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Flowsheet storage
# ---------------------------------------------------------------------------

def save_flowsheet(conn: sqlite3.Connection, image_id: str, run_id: str,
                   data: FlowsheetData = None, llm_model: str = None,
                   llm_tokens: int = None, llm_cost: float = None,
                   error: str = None, raw_response: str = None) -> None:
    """Save flowsheet extraction results. Idempotent via INSERT OR REPLACE."""
    if data:
        stages_json = json.dumps([s.model_dump() for s in data.stages])
        units_json = json.dumps([u.model_dump() for u in data.units])
        connections_json = json.dumps([c.model_dump() for c in data.connections])
    else:
        stages_json = "[]"
        units_json = "[]"
        connections_json = "[]"

    conn.execute(
        """INSERT OR REPLACE INTO flowsheets
           (image_id, run_id, stages_json, units_json, connections_json,
            llm_model, llm_tokens, llm_cost_usd, error, raw_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (image_id, run_id, stages_json, units_json, connections_json,
         llm_model, llm_tokens, llm_cost, error, raw_response)
    )
    conn.commit()


def get_flowsheet(conn: sqlite3.Connection, image_id: str) -> Optional[dict]:
    """Retrieve a flowsheet by image_id. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM flowsheets WHERE image_id = ?", (image_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["stages"] = json.loads(d.pop("stages_json"))
    d["units"] = json.loads(d.pop("units_json"))
    d["connections"] = json.loads(d.pop("connections_json"))
    return d


def get_all_flowsheets(conn: sqlite3.Connection) -> dict:
    """Retrieve all flowsheets joined with image metadata.
    Returns dict keyed by image_id, matching results.json structure."""
    rows = conn.execute(
        """SELECT f.image_id, f.stages_json, f.units_json, f.connections_json,
                  f.error, f.raw_response, f.llm_tokens, f.llm_cost_usd,
                  i.image_type, i.page_number, i.caption, i.text_before, i.text_after
           FROM flowsheets f
           JOIN images i ON f.image_id = i.image_id"""
    ).fetchall()
    result = {}
    for r in rows:
        d = dict(r)
        entry = {"type": d["image_type"]}
        if d.get("error"):
            entry["error"] = d["error"]
            entry["raw"] = d.get("raw_response", "")
        else:
            entry["stages"] = json.loads(d["stages_json"])
            entry["units"] = json.loads(d["units_json"])
            entry["connections"] = json.loads(d["connections_json"])
        entry["text_before"] = d.get("text_before", "")
        entry["text_after"] = d.get("text_after", "")
        entry["caption"] = d.get("caption", "")
        entry["page"] = d.get("page_number")
        result[d["image_id"]] = entry
    return result


# ---------------------------------------------------------------------------
# Table storage
# ---------------------------------------------------------------------------

def save_table_record(conn: sqlite3.Connection, table: TableRecord,
                      run_id: str = None) -> None:
    """Save an extracted table record. Idempotent via INSERT OR REPLACE."""
    conn.execute(
        """INSERT OR REPLACE INTO extracted_tables
           (table_id, document_id, run_id, page_number, table_index,
            section_number, section_title, caption, headers_json, rows_json,
            text_before, text_after)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (table.table_id, table.document_id, run_id, table.page_number,
         table.table_index, table.section_number, table.section_title,
         table.caption, json.dumps(table.headers), json.dumps(table.rows),
         table.text_before, table.text_after)
    )
    conn.commit()


def get_tables_for_document(conn: sqlite3.Connection, document_id: str,
                            section_number: str = None) -> list[dict]:
    """Retrieve tables for a document, optionally filtered by section."""
    if section_number:
        rows = conn.execute(
            """SELECT * FROM extracted_tables
               WHERE document_id = ? AND section_number LIKE ?
               ORDER BY page_number, table_index""",
            (document_id, f"{section_number}%")
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM extracted_tables
               WHERE document_id = ? ORDER BY page_number, table_index""",
            (document_id,)
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["headers"] = json.loads(d.pop("headers_json"))
        d["rows"] = json.loads(d.pop("rows_json"))
        results.append(d)
    return results


def import_s3_tables(conn: sqlite3.Connection, result_data: dict,
                     run_id: str = None) -> int:
    """
    Import tables from an S3 per-PDF result JSON into SQLite.

    Upserts the document record and saves all table records.
    Returns count of tables imported.
    """
    doc_id = result_data["document_id"]
    filename = result_data.get("filename", "unknown.pdf")
    upsert_document(conn, doc_id, filename)

    count = 0
    for idx, t in enumerate(result_data.get("tables", [])):
        record = TableRecord(
            table_id=t.get("table_id", f"{doc_id}_table_{idx}"),
            document_id=doc_id,
            page_number=t["page"],
            table_index=idx,
            section_number=t.get("section_number"),
            section_title=t.get("section_title"),
            caption=t.get("caption"),
            headers=t.get("headers", []),
            rows=t.get("rows", []),
            text_before=t.get("text_before"),
            text_after=t.get("text_after"),
        )
        save_table_record(conn, record, run_id)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Normalization dictionary
# ---------------------------------------------------------------------------

def save_normalization_entries(conn: sqlite3.Connection, run_id: str,
                               normalized_groups: dict) -> None:
    """Save normalization dictionary entries. Merges with existing."""
    for normalized_id, original_ids in normalized_groups.items():
        for original_id in original_ids:
            conn.execute(
                """INSERT OR IGNORE INTO normalization_dict
                   (normalized_id, original_id, first_seen_run)
                   VALUES (?, ?, ?)""",
                (normalized_id, original_id, run_id)
            )
    conn.commit()


def load_normalization_dict(conn: sqlite3.Connection) -> dict:
    """Load the full normalization dictionary. Returns {normalized_id: [original_ids]}."""
    rows = conn.execute(
        "SELECT normalized_id, original_id FROM normalization_dict ORDER BY normalized_id"
    ).fetchall()
    groups = {}
    for r in rows:
        groups.setdefault(r["normalized_id"], []).append(r["original_id"])
    return groups


# ---------------------------------------------------------------------------
# JSON import helpers (migration)
# ---------------------------------------------------------------------------

def import_results_json(conn: sqlite3.Connection, results_path: str,
                        document_id: str, run_id: str) -> int:
    """Import entries from a results.json file into the database. Returns count."""
    with open(results_path, "r") as f:
        data = json.load(f)

    count = 0
    for image_id, entry in data.items():
        image_type = entry.get("type", "flowsheet")
        image = ImageRecord(
            image_id=image_id,
            document_id=document_id,
            image_type=image_type,
            page_number=entry.get("page"),
            caption=entry.get("caption"),
            text_before=entry.get("text_before"),
            text_after=entry.get("text_after"),
        )
        save_image_record(conn, image, run_id)

        if image_type == "flowsheet" and "error" not in entry:
            stages = entry.get("stages", [])
            units = entry.get("units", [])
            connections = entry.get("connections", [])
            flowsheet_data = FlowsheetData(
                stages=[Stage(**s) for s in stages],
                units=[Unit(**u) for u in units],
                connections=[Connection(**c) for c in connections],
            )
            save_flowsheet(conn, image_id, run_id, data=flowsheet_data)
        elif image_type == "flowsheet" and "error" in entry:
            save_flowsheet(conn, image_id, run_id,
                           error=entry["error"],
                           raw_response=entry.get("raw"))
        count += 1
    return count


def import_processed_pdfs(conn: sqlite3.Connection,
                          processed_path: str) -> int:
    """Import processed_pdfs.json entries as documents (with placeholder IDs)."""
    with open(processed_path, "r") as f:
        stems = json.load(f)

    count = 0
    for stem in stems:
        # Use stem as placeholder document_id since we don't have the original file
        placeholder_id = hashlib.sha256(stem.encode()).hexdigest()
        row = conn.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (placeholder_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                """INSERT INTO documents (document_id, filename, document_version)
                   VALUES (?, ?, 1)""",
                (placeholder_id, f"{stem}.pdf")
            )
            count += 1
    conn.commit()
    return count


def import_normalization_json(conn: sqlite3.Connection, norm_path: str,
                              run_id: str) -> int:
    """Import normalization dictionary from JSON file."""
    with open(norm_path, "r") as f:
        groups = json.load(f)
    save_normalization_entries(conn, run_id, groups)
    return sum(len(v) for v in groups.values())


# ---------------------------------------------------------------------------
# JSON export helpers (transition validation)
# ---------------------------------------------------------------------------

def export_results_json(conn: sqlite3.Connection, output_path: str) -> None:
    """Export database flowsheets to results.json format for comparison."""
    data = get_all_flowsheets(conn)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_normalization_json(conn: sqlite3.Connection,
                              output_path: str) -> None:
    """Export normalization dictionary to JSON format."""
    groups = load_normalization_dict(conn)
    with open(output_path, "w") as f:
        json.dump(groups, f, indent=2)
