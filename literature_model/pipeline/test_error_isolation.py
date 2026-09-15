"""
Error isolation tests — verify that individual page/image/PDF failures
do not crash the pipeline. Uses mocks; no real PDFs, LLM calls, or CLIP.
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from database import _SCHEMA_SQL, init_db, get_connection, start_run, get_errors_for_run
from main import run_process_pdfs, run_test_refinement, get_processed_pdfs


@pytest.fixture
def db_conn(tmp_path):
    """In-memory SQLite connection with schema initialized + a sample document."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO documents (document_id, filename) VALUES (?, ?)",
        ("test_doc_id", "test.pdf")
    )
    conn.commit()
    return conn


@pytest.fixture
def run_id(db_conn):
    return start_run(db_conn, "test_doc_id")


# ---------------------------------------------------------------------------
# PDF extraction isolation
# ---------------------------------------------------------------------------

class TestPdfExtractionIsolation:
    @patch('pdf_extraction.extract_all_figures')
    def test_single_pdf_error_does_not_stop_pipeline(self, mock_extract,
                                                      tmp_path, monkeypatch):
        """If extract_all_figures raises for one PDF, others should still process."""
        monkeypatch.chdir(tmp_path)

        # First call raises, second succeeds
        mock_extract.side_effect = [
            RuntimeError("corrupted PDF"),
            [{"type": "flowsheet", "image_path": "good.png"}],
        ]

        # Simulate the per-PDF loop from main.py __main__
        results = []
        errors = []
        for pdf_name in ["bad.pdf", "good.pdf"]:
            try:
                data = run_process_pdfs(pdf_name)
                results.extend(data)
            except Exception as e:
                errors.append(str(e))
                continue

        assert len(errors) == 1
        assert "corrupted PDF" in errors[0]
        assert len(results) == 1
        assert results[0]["type"] == "flowsheet"


# ---------------------------------------------------------------------------
# LLM extraction isolation
# ---------------------------------------------------------------------------

class TestLlmExtractionIsolation:
    @patch('flowsheet_extraction.extract_with_terms')
    def test_single_flowsheet_error_continues_to_next(self, mock_extract,
                                                       tmp_path,
                                                       db_conn, run_id):
        """If extract_with_terms raises for one flowsheet, the next should process."""
        results_path = str(tmp_path / "results.json")

        # First call raises, second returns success
        mock_extract.side_effect = [
            RuntimeError("API timeout"),
            {
                "data": {"stages": [], "units": [], "connections": []},
                "total_tokens": 100,
                "total_cost": 0.01,
                "costs_by_model": {"gpt-5.2": 0.01},
            },
        ]

        extracted_data = [
            {"type": "flowsheet", "image_path": "bad_pfs_1.png", "page": 10},
            {"type": "flowsheet", "image_path": "good_pfs_2.png", "page": 20},
        ]

        run_test_refinement(extracted_data, conn=db_conn, run_id=run_id,
                            document_id="test_doc_id", results_path=results_path)

        # Second flowsheet should still be processed
        with open(results_path) as f:
            results = json.load(f)
        assert "good_pfs_2" in results
        assert "stages" in results["good_pfs_2"]

    @patch('flowsheet_extraction.extract_with_terms')
    def test_api_timeout_recorded_as_error(self, mock_extract,
                                            tmp_path,
                                            db_conn, run_id):
        """API timeout should be logged to step_errors table."""
        results_path = str(tmp_path / "results.json")

        mock_extract.side_effect = TimeoutError("Connection timed out")

        extracted_data = [
            {"type": "flowsheet", "image_path": "timeout_pfs_1.png", "page": 5},
        ]

        run_test_refinement(extracted_data, conn=db_conn, run_id=run_id,
                            document_id="test_doc_id", results_path=results_path)

        errors = get_errors_for_run(db_conn, run_id)
        assert len(errors) == 1
        assert errors[0]["error_type"] == "TimeoutError"
        assert errors[0]["step_name"] == "llm_extraction"
        assert errors[0]["image_id"] == "timeout_pfs_1"

    @patch('flowsheet_extraction.extract_with_terms')
    def test_invalid_json_from_llm_stored_as_error(self, mock_extract,
                                                    tmp_path):
        """LLM returning invalid JSON should be stored as an error entry (not crash)."""
        results_path = str(tmp_path / "results.json")

        # extract_with_terms already catches JSONDecodeError and returns error dict
        mock_extract.return_value = {
            "error": "JSONDecodeError: Expecting value",
            "raw": "{invalid json",
            "total_tokens": 50,
            "total_cost": 0.005,
            "costs_by_model": {"gpt-5.2": 0.005},
        }

        extracted_data = [
            {"type": "flowsheet", "image_path": "badjson_pfs_1.png"},
        ]

        run_test_refinement(extracted_data, results_path=results_path)

        with open(results_path) as f:
            results = json.load(f)
        assert "badjson_pfs_1" in results
        assert "error" in results["badjson_pfs_1"]

    @patch('flowsheet_extraction.extract_with_terms')
    def test_multiple_errors_all_logged(self, mock_extract,
                                         tmp_path,
                                         db_conn, run_id):
        """Multiple flowsheet failures should each be logged."""
        results_path = str(tmp_path / "results.json")

        mock_extract.side_effect = [
            ValueError("bad image 1"),
            TypeError("bad image 2"),
        ]

        extracted_data = [
            {"type": "flowsheet", "image_path": "fail1_pfs_1.png", "page": 1},
            {"type": "flowsheet", "image_path": "fail2_pfs_2.png", "page": 2},
        ]

        run_test_refinement(extracted_data, conn=db_conn, run_id=run_id,
                            document_id="test_doc_id", results_path=results_path)

        errors = get_errors_for_run(db_conn, run_id)
        assert len(errors) == 2
        assert errors[0]["error_type"] == "ValueError"
        assert errors[1]["error_type"] == "TypeError"


# ---------------------------------------------------------------------------
# Pipeline completion despite errors
# ---------------------------------------------------------------------------

class TestPipelineCompletionDespiteErrors:
    @patch('flowsheet_extraction.extract_with_terms')
    def test_pipeline_produces_results_despite_partial_failures(
            self, mock_extract, tmp_path, db_conn, run_id):
        """Pipeline should write results.json even when some flowsheets fail."""
        results_path = str(tmp_path / "results.json")

        mock_extract.side_effect = [
            RuntimeError("fail"),
            {
                "data": {"stages": [{"id": "mill", "name": "Milling", "order": 1}],
                         "units": [], "connections": []},
                "total_tokens": 200,
                "total_cost": 0.02,
                "costs_by_model": {"gpt-5.2": 0.02},
            },
        ]

        extracted_data = [
            {"type": "flowsheet", "image_path": "fail_pfs_1.png", "page": 10},
            {"type": "flowsheet", "image_path": "success_pfs_2.png", "page": 20},
        ]

        run_test_refinement(extracted_data, conn=db_conn, run_id=run_id,
                            document_id="test_doc_id", results_path=results_path)

        # Results file should exist with the successful entry
        with open(results_path) as f:
            results = json.load(f)
        assert "success_pfs_2" in results
        assert results["success_pfs_2"]["stages"][0]["id"] == "mill"

        # Errors should be logged
        errors = get_errors_for_run(db_conn, run_id)
        assert len(errors) == 1

    @patch('flowsheet_extraction.extract_with_terms')
    def test_errors_logged_to_database_with_context(self, mock_extract,
                                                     tmp_path,
                                                     db_conn, run_id):
        """Errors should include page number and image ID for debugging."""
        results_path = str(tmp_path / "results.json")

        mock_extract.side_effect = RuntimeError("GPU OOM")

        extracted_data = [
            {"type": "flowsheet", "image_path": "large_pfs_500.png", "page": 500},
        ]

        run_test_refinement(extracted_data, conn=db_conn, run_id=run_id,
                            document_id="test_doc_id", results_path=results_path)

        errors = get_errors_for_run(db_conn, run_id)
        assert len(errors) == 1
        assert errors[0]["image_id"] == "large_pfs_500"
        assert errors[0]["page_number"] == 500
        assert "GPU OOM" in errors[0]["error_message"]
        assert errors[0]["traceback"] is not None
