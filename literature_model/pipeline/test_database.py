"""Unit tests for database.py — all use in-memory SQLite for speed."""

import json
import sqlite3
import tempfile
import pytest
from pathlib import Path

from database import (
    init_db, get_connection,
    compute_document_id, upsert_document,
    start_run, complete_run, mark_step_completed,
    log_step_error, get_errors_for_run,
    save_image_record, get_images_for_document,
    save_flowsheet, get_flowsheet, get_all_flowsheets,
    save_table_record, get_tables_for_document, import_s3_tables,
    save_normalization_entries, load_normalization_dict,
    import_results_json, import_processed_pdfs, import_normalization_json,
    export_results_json, export_normalization_json,
    Stage, Unit, Connection, FlowsheetData, ImageRecord, TableRecord,
    _SCHEMA_SQL,
)


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema initialized."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(_SCHEMA_SQL)
    return c


@pytest.fixture
def sample_document(conn):
    """Insert a sample document and return its ID."""
    doc_id = "abc123"
    conn.execute(
        "INSERT INTO documents (document_id, filename) VALUES (?, ?)",
        (doc_id, "test.pdf")
    )
    conn.commit()
    return doc_id


@pytest.fixture
def sample_run(conn, sample_document):
    """Insert a sample pipeline run and return its ID."""
    return start_run(conn, sample_document)


# ---------------------------------------------------------------------------
# TestInitDb
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_all_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = get_connection(db_path)
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "documents" in tables
        assert "pipeline_runs" in tables
        assert "images" in tables
        assert "flowsheets" in tables
        assert "normalization_dict" in tables
        assert "step_errors" in tables

    def test_idempotent_creation(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        init_db(db_path)  # Second call should not error
        conn = get_connection(db_path)
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "documents" in tables

    def test_creates_data_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "nested" / "test.db")
        init_db(db_path)
        assert Path(db_path).exists()


# ---------------------------------------------------------------------------
# TestDocumentIdentity
# ---------------------------------------------------------------------------

class TestDocumentIdentity:
    def test_compute_document_id_deterministic(self, tmp_path):
        f = tmp_path / "sample.pdf"
        f.write_bytes(b"hello world pdf content")
        id1 = compute_document_id(str(f))
        id2 = compute_document_id(str(f))
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex

    def test_compute_document_id_different_files(self, tmp_path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_document_id(str(f1)) != compute_document_id(str(f2))

    def test_upsert_document_creates_new(self, conn):
        version = upsert_document(conn, "doc1", "test.pdf", "/path/test.pdf", 1024)
        assert version == 1
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = 'doc1'"
        ).fetchone()
        assert row["filename"] == "test.pdf"
        assert row["file_size_bytes"] == 1024

    def test_upsert_document_increments_version(self, conn):
        upsert_document(conn, "doc1", "test.pdf")
        v2 = upsert_document(conn, "doc1", "test.pdf", "/new/path", 2048)
        assert v2 == 2
        v3 = upsert_document(conn, "doc1", "test.pdf")
        assert v3 == 3


# ---------------------------------------------------------------------------
# TestPipelineRuns
# ---------------------------------------------------------------------------

class TestPipelineRuns:
    def test_start_run_returns_uuid(self, conn, sample_document):
        run_id = start_run(conn, sample_document)
        assert len(run_id) == 36  # UUID format
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "running"
        assert row["document_id"] == sample_document

    def test_complete_run_sets_status(self, conn, sample_document):
        run_id = start_run(conn, sample_document)
        complete_run(conn, run_id, "completed")
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "completed"
        assert row["finished_at"] is not None

    def test_mark_step_completed_appends(self, conn, sample_document):
        run_id = start_run(conn, sample_document)
        mark_step_completed(conn, run_id, "pdf_extraction")
        mark_step_completed(conn, run_id, "llm_extraction")
        row = conn.execute(
            "SELECT steps_completed FROM pipeline_runs WHERE run_id = ?",
            (run_id,)
        ).fetchone()
        steps = json.loads(row["steps_completed"])
        assert steps == ["pdf_extraction", "llm_extraction"]

    def test_mark_step_completed_no_duplicates(self, conn, sample_document):
        run_id = start_run(conn, sample_document)
        mark_step_completed(conn, run_id, "pdf_extraction")
        mark_step_completed(conn, run_id, "pdf_extraction")
        row = conn.execute(
            "SELECT steps_completed FROM pipeline_runs WHERE run_id = ?",
            (run_id,)
        ).fetchone()
        steps = json.loads(row["steps_completed"])
        assert steps == ["pdf_extraction"]


# ---------------------------------------------------------------------------
# TestErrorLogging
# ---------------------------------------------------------------------------

class TestErrorLogging:
    def test_log_and_retrieve_step_error(self, conn, sample_run):
        error = ValueError("bad data")
        log_step_error(conn, sample_run, "llm_extraction", error,
                       image_id="test_pfs_1", page_number=42)
        errors = get_errors_for_run(conn, sample_run)
        assert len(errors) == 1
        assert errors[0]["step_name"] == "llm_extraction"
        assert errors[0]["error_type"] == "ValueError"
        assert errors[0]["error_message"] == "bad data"
        assert errors[0]["image_id"] == "test_pfs_1"
        assert errors[0]["page_number"] == 42

    def test_log_error_with_traceback(self, conn, sample_run):
        try:
            raise RuntimeError("test traceback")
        except RuntimeError as e:
            log_step_error(conn, sample_run, "pdf_extraction", e)
        errors = get_errors_for_run(conn, sample_run)
        assert "RuntimeError" in errors[0]["traceback"]

    def test_multiple_errors_per_run(self, conn, sample_run):
        log_step_error(conn, sample_run, "step1", ValueError("err1"))
        log_step_error(conn, sample_run, "step2", TypeError("err2"))
        errors = get_errors_for_run(conn, sample_run)
        assert len(errors) == 2

    def test_no_errors_returns_empty_list(self, conn, sample_run):
        errors = get_errors_for_run(conn, sample_run)
        assert errors == []


# ---------------------------------------------------------------------------
# TestImageStorage
# ---------------------------------------------------------------------------

class TestImageStorage:
    def test_save_and_retrieve_flowsheet_image(self, conn, sample_document, sample_run):
        image = ImageRecord(
            image_id="test_pfs_213",
            document_id=sample_document,
            image_type="flowsheet",
            page_number=213,
            file_path="/path/to/test_pfs_213.png",
            caption="Figure 1: Process Flow",
            text_before="casino project",
            text_after="august revision",
        )
        save_image_record(conn, image, sample_run)
        results = get_images_for_document(conn, sample_document, "flowsheet")
        assert len(results) == 1
        assert results[0]["image_id"] == "test_pfs_213"
        assert results[0]["page_number"] == 213

    def test_save_and_retrieve_other_image_with_embedding(self, conn, sample_document, sample_run):
        embedding = [0.1, 0.2, 0.3] * 170 + [0.4, 0.5]  # 512-dim
        image = ImageRecord(
            image_id="test_page_42",
            document_id=sample_document,
            image_type="other",
            page_number=42,
            embedding=embedding,
        )
        save_image_record(conn, image, sample_run)
        results = get_images_for_document(conn, sample_document, "other")
        assert len(results) == 1
        assert results[0]["embedding"] == embedding

    def test_filter_by_image_type(self, conn, sample_document, sample_run):
        for img_type, img_id in [("flowsheet", "pfs_1"), ("other", "other_1")]:
            save_image_record(conn, ImageRecord(
                image_id=img_id, document_id=sample_document, image_type=img_type
            ), sample_run)
        assert len(get_images_for_document(conn, sample_document, "flowsheet")) == 1
        assert len(get_images_for_document(conn, sample_document, "other")) == 1
        assert len(get_images_for_document(conn, sample_document)) == 2

    def test_idempotent_image_save(self, conn, sample_document, sample_run):
        image = ImageRecord(
            image_id="test_pfs_1", document_id=sample_document, image_type="flowsheet"
        )
        save_image_record(conn, image, sample_run)
        save_image_record(conn, image, sample_run)  # Should not duplicate
        results = get_images_for_document(conn, sample_document)
        assert len(results) == 1

    def test_pydantic_rejects_missing_required_fields(self):
        with pytest.raises(Exception):
            ImageRecord(image_id="x")  # Missing document_id and image_type


# ---------------------------------------------------------------------------
# TestFlowsheetStorage
# ---------------------------------------------------------------------------

class TestFlowsheetStorage:
    def test_save_and_retrieve_flowsheet_data(self, conn, sample_document, sample_run):
        # Must create image record first (foreign key)
        save_image_record(conn, ImageRecord(
            image_id="pfs_1", document_id=sample_document, image_type="flowsheet"
        ), sample_run)

        data = FlowsheetData(
            stages=[Stage(id="crushing", name="Crushing", order=1)],
            units=[Unit(id="crusher_1", label="JAW CRUSHER", type="crusher",
                        stage_id="crushing", cells=1)],
            connections=[Connection(parent_id="input", child_id="crusher_1")],
        )
        save_flowsheet(conn, "pfs_1", sample_run, data=data,
                       llm_model="gpt-5.2", llm_tokens=500, llm_cost=0.05)

        result = get_flowsheet(conn, "pfs_1")
        assert result is not None
        assert len(result["stages"]) == 1
        assert result["stages"][0]["id"] == "crushing"
        assert len(result["units"]) == 1
        assert result["units"][0]["type"] == "crusher"
        assert len(result["connections"]) == 1
        assert result["llm_model"] == "gpt-5.2"

    def test_save_flowsheet_with_error(self, conn, sample_document, sample_run):
        save_image_record(conn, ImageRecord(
            image_id="pfs_2", document_id=sample_document, image_type="flowsheet"
        ), sample_run)
        save_flowsheet(conn, "pfs_2", sample_run,
                       error="JSON parse error", raw_response="invalid{json")
        result = get_flowsheet(conn, "pfs_2")
        assert result["error"] == "JSON parse error"
        assert result["raw_response"] == "invalid{json"

    def test_idempotent_flowsheet_save(self, conn, sample_document, sample_run):
        save_image_record(conn, ImageRecord(
            image_id="pfs_3", document_id=sample_document, image_type="flowsheet"
        ), sample_run)
        data = FlowsheetData(stages=[Stage(id="mill", name="Milling", order=1)])
        save_flowsheet(conn, "pfs_3", sample_run, data=data)
        save_flowsheet(conn, "pfs_3", sample_run, data=data)  # No duplicate
        row = conn.execute("SELECT COUNT(*) as c FROM flowsheets").fetchone()
        assert row["c"] == 1

    def test_get_flowsheet_not_found(self, conn):
        assert get_flowsheet(conn, "nonexistent") is None

    def test_pydantic_validates_stage_structure(self):
        with pytest.raises(Exception):
            Stage(id="x", name="X")  # Missing 'order'

    def test_pydantic_validates_connection_structure(self):
        with pytest.raises(Exception):
            Connection(parent_id="a")  # Missing 'child_id'

    def test_pydantic_ignores_extra_fields(self):
        stage = Stage(id="x", name="X", order=1, extra_field="ignored")
        assert not hasattr(stage, "extra_field")

    def test_get_all_flowsheets(self, conn, sample_document, sample_run):
        # Create two flowsheet images
        for img_id in ["pfs_1", "pfs_2"]:
            save_image_record(conn, ImageRecord(
                image_id=img_id, document_id=sample_document,
                image_type="flowsheet", page_number=100
            ), sample_run)
        data1 = FlowsheetData(stages=[Stage(id="crush", name="Crushing", order=1)])
        data2 = FlowsheetData(stages=[Stage(id="mill", name="Milling", order=2)])
        save_flowsheet(conn, "pfs_1", sample_run, data=data1)
        save_flowsheet(conn, "pfs_2", sample_run, data=data2)

        all_fs = get_all_flowsheets(conn)
        assert len(all_fs) == 2
        assert "pfs_1" in all_fs
        assert "pfs_2" in all_fs
        assert all_fs["pfs_1"]["type"] == "flowsheet"
        assert len(all_fs["pfs_1"]["stages"]) == 1


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_save_and_load_normalization_dict(self, conn, sample_run):
        groups = {
            "crusher": ["primary_crushing", "jaw_crusher"],
            "mill": ["ball_mill", "grinding"],
        }
        save_normalization_entries(conn, sample_run, groups)
        loaded = load_normalization_dict(conn)
        assert set(loaded["crusher"]) == {"primary_crushing", "jaw_crusher"}
        assert set(loaded["mill"]) == {"ball_mill", "grinding"}

    def test_merge_with_existing_entries(self, conn, sample_run):
        save_normalization_entries(conn, sample_run,
                                   {"crusher": ["jaw_crusher"]})
        save_normalization_entries(conn, sample_run,
                                   {"crusher": ["jaw_crusher", "cone_crusher"]})
        loaded = load_normalization_dict(conn)
        assert set(loaded["crusher"]) == {"jaw_crusher", "cone_crusher"}

    def test_empty_groups(self, conn, sample_run):
        save_normalization_entries(conn, sample_run, {})
        loaded = load_normalization_dict(conn)
        assert loaded == {}


# ---------------------------------------------------------------------------
# TestJsonImport
# ---------------------------------------------------------------------------

class TestJsonImport:
    def test_import_results_json(self, conn, sample_document, sample_run, tmp_path):
        results = {
            "test_pfs_1": {
                "type": "flowsheet",
                "stages": [{"id": "crush", "name": "Crushing", "order": 1}],
                "units": [{"id": "c1", "label": "Crusher", "type": "crusher",
                           "stage_id": "crush", "cells": 1}],
                "connections": [{"parent_id": "input", "child_id": "c1"}],
                "text_before": "context",
                "caption": "Figure 1",
                "page": 100,
            },
            "test_pfs_2": {
                "type": "flowsheet",
                "error": "parse error",
                "raw": "bad json",
                "page": 200,
            },
        }
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))

        count = import_results_json(conn, str(f), sample_document, sample_run)
        assert count == 2

        # Check flowsheet with data
        fs = get_flowsheet(conn, "test_pfs_1")
        assert len(fs["stages"]) == 1

        # Check flowsheet with error
        fs_err = get_flowsheet(conn, "test_pfs_2")
        assert fs_err["error"] == "parse error"

    def test_import_processed_pdfs(self, conn, tmp_path):
        f = tmp_path / "processed_pdfs.json"
        f.write_text(json.dumps(["test_casino", "test_mining"]))
        count = import_processed_pdfs(conn, str(f))
        assert count == 2
        rows = conn.execute("SELECT * FROM documents").fetchall()
        filenames = [r["filename"] for r in rows]
        assert "test_casino.pdf" in filenames
        assert "test_mining.pdf" in filenames

    def test_import_normalization_json(self, conn, sample_run, tmp_path):
        groups = {"crusher": ["jaw", "cone"], "mill": ["ball", "sag"]}
        f = tmp_path / "norm.json"
        f.write_text(json.dumps(groups))
        count = import_normalization_json(conn, str(f), sample_run)
        assert count == 4
        loaded = load_normalization_dict(conn)
        assert len(loaded["crusher"]) == 2

    def test_import_results_with_from_to_connections(self, conn, sample_document,
                                                      sample_run, tmp_path):
        """Test importing results that use old from_type/to_type connection format."""
        results = {
            "test_pfs_old": {
                "type": "flowsheet",
                "stages": [{"id": "crush", "name": "Crushing", "order": 1}],
                "units": [],
                "connections": [{"parent_id": "input", "child_id": "crusher"}],
                "page": 50,
            }
        }
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        count = import_results_json(conn, str(f), sample_document, sample_run)
        assert count == 1


# ---------------------------------------------------------------------------
# TestJsonExport
# ---------------------------------------------------------------------------

class TestImportS3Tables:
    def test_import_tables(self, conn, sample_run):
        result_data = {
            "document_id": "hash_123",
            "filename": "test_report.pdf",
            "tables": [
                {
                    "table_id": "hash_123_table_50_0",
                    "page": 50,
                    "section_number": "13",
                    "section_title": "Mineral Processing and Metallurgical Testing",
                    "caption": "Table 13-1: Flotation Results",
                    "headers": ["Stage", "Time", "Recovery"],
                    "rows": [["Primary", "8 min", "85%"]],
                    "text_before": "Testing was conducted...",
                    "text_after": "Results indicate...",
                },
                {
                    "table_id": "hash_123_table_55_0",
                    "page": 55,
                    "section_number": "13",
                    "section_title": "Mineral Processing and Metallurgical Testing",
                    "caption": "Table 13-2: Grade Results",
                    "headers": ["Sample", "Cu%"],
                    "rows": [["A", "2.3"], ["B", "1.8"]],
                    "text_before": "",
                    "text_after": "",
                },
            ],
        }
        count = import_s3_tables(conn, result_data, sample_run)
        assert count == 2

        # Verify document was created
        doc = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", ("hash_123",)
        ).fetchone()
        assert doc is not None
        assert doc["filename"] == "test_report.pdf"

        # Verify tables are retrievable
        tables = get_tables_for_document(conn, "hash_123")
        assert len(tables) == 2
        assert tables[0]["table_id"] == "hash_123_table_50_0"
        assert tables[0]["headers"] == ["Stage", "Time", "Recovery"]
        assert tables[0]["rows"] == [["Primary", "8 min", "85%"]]

    def test_import_no_tables(self, conn, sample_run):
        result_data = {
            "document_id": "hash_empty",
            "filename": "empty.pdf",
            "tables": [],
        }
        count = import_s3_tables(conn, result_data, sample_run)
        assert count == 0

    def test_import_filters_by_section(self, conn, sample_run):
        result_data = {
            "document_id": "hash_multi",
            "filename": "multi.pdf",
            "tables": [
                {"table_id": "t1", "page": 10, "section_number": "7",
                 "headers": [], "rows": [[]], "text_before": "", "text_after": ""},
                {"table_id": "t2", "page": 50, "section_number": "13",
                 "headers": [], "rows": [[]], "text_before": "", "text_after": ""},
            ],
        }
        import_s3_tables(conn, result_data, sample_run)
        sec13 = get_tables_for_document(conn, "hash_multi", section_number="13")
        assert len(sec13) == 1
        assert sec13[0]["table_id"] == "t2"


class TestJsonExport:
    def test_export_results_json(self, conn, sample_document, sample_run, tmp_path):
        save_image_record(conn, ImageRecord(
            image_id="pfs_1", document_id=sample_document,
            image_type="flowsheet", page_number=100,
            caption="Fig 1", text_before="before", text_after="after",
        ), sample_run)
        data = FlowsheetData(
            stages=[Stage(id="crush", name="Crushing", order=1)],
            units=[],
            connections=[],
        )
        save_flowsheet(conn, "pfs_1", sample_run, data=data)

        out = tmp_path / "exported.json"
        export_results_json(conn, str(out))
        exported = json.loads(out.read_text())
        assert "pfs_1" in exported
        assert exported["pfs_1"]["type"] == "flowsheet"
        assert exported["pfs_1"]["page"] == 100

    def test_export_normalization_json(self, conn, sample_run, tmp_path):
        save_normalization_entries(conn, sample_run,
                                   {"crusher": ["jaw", "cone"]})
        out = tmp_path / "norm_export.json"
        export_normalization_json(conn, str(out))
        exported = json.loads(out.read_text())
        assert set(exported["crusher"]) == {"jaw", "cone"}
