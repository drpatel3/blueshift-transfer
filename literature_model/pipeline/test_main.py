import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from main import get_processed_pdfs, run_process_pdfs, run_test_refinement


class TestGetProcessedPdfs:
    """Tests for get_processed_pdfs function."""

    def test_returns_empty_set_when_file_missing(self, tmp_path):
        """Should return empty set when results.json doesn't exist."""
        result = get_processed_pdfs(str(tmp_path / "nonexistent.json"))
        assert result == set()

    def test_returns_empty_set_when_no_entries(self, tmp_path):
        """Should return empty set when results.json is empty."""
        results_file = tmp_path / "results.json"
        results_file.write_text("{}")

        result = get_processed_pdfs(str(results_file))
        assert result == set()

    def test_extracts_pdf_stem_from_pfs_key(self, tmp_path):
        """Should extract PDF stem from flowsheet keys like 'test_casino_pfs_213'."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps({
            "test_casino_pfs_213": {"type": "flowsheet"},
            "test_casino_pfs_215": {"type": "flowsheet"},
        }))

        result = get_processed_pdfs(str(results_file))
        assert "test_casino" in result

    def test_extracts_pdf_stem_from_other_key(self, tmp_path):
        """Should extract PDF stem from 'other' image keys like 'test_casino_123'."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps({
            "test_casino_123": {"type": "other"},
        }))

        result = get_processed_pdfs(str(results_file))
        assert "test_casino" in result

    def test_handles_multiple_pdfs(self, tmp_path):
        """Should correctly identify multiple processed PDFs."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps({
            "test_casino_pfs_1": {"type": "flowsheet"},
            "test_mining_pfs_5": {"type": "flowsheet"},
            "test_refinery_42": {"type": "other"},
        }))

        result = get_processed_pdfs(str(results_file))
        assert "test_casino" in result
        assert "test_mining" in result
        assert "test_refinery" in result


class TestRunProcessPdfs:
    """Tests for run_process_pdfs function."""

    @patch('pdf_extraction.extract_all_figures')
    def test_returns_extracted_data(self, mock_extract):
        """Should return data from extract_all_figures."""
        mock_extract.return_value = [
            {"type": "flowsheet", "image_path": "img1.png"},
            {"type": "other", "image_path": "img2.png"},
        ]

        result = run_process_pdfs("test.pdf")

        assert len(result) == 2
        mock_extract.assert_called_once_with(pdf_location="test.pdf")

    @patch('pdf_extraction.extract_all_figures')
    def test_handles_empty_extraction(self, mock_extract):
        """Should handle PDFs with no images."""
        mock_extract.return_value = []

        result = run_process_pdfs("empty.pdf")

        assert result == []


class TestRunTestRefinement:
    """Tests for run_test_refinement function."""

    @patch('flowsheet_extraction.extract_with_terms')
    def test_only_processes_flowsheets(self, mock_extract, tmp_path):
        """Should only run LLM extraction on flowsheet type images."""
        results_path = str(tmp_path / "results.json")

        mock_extract.return_value = {
            "data": {"stages": []},
            "total_tokens": 100,
            "total_cost": 0.01,
            "costs_by_model": {"gpt-4": 0.01}
        }

        extracted_data = [
            {"type": "flowsheet", "image_path": "flow1.png"},
            {"type": "other", "image_path": "other1.png"},
            {"type": "other", "image_path": "other2.png"},
        ]

        run_test_refinement(extracted_data, results_path=results_path)

        # Should only be called once for the single flowsheet
        assert mock_extract.call_count == 1

    @patch('flowsheet_extraction.extract_with_terms')
    def test_other_images_skipped_by_llm(self, mock_extract, tmp_path):
        """Should skip LLM extraction for 'other' type images."""
        results_path = str(tmp_path / "results.json")

        extracted_data = [
            {"type": "other", "image_path": "other1.png", "caption": "Figure 1"},
        ]

        run_test_refinement(extracted_data, results_path=results_path)

        # LLM should not be called for 'other' type
        mock_extract.assert_not_called()

    @patch('flowsheet_extraction.extract_with_terms')
    def test_handles_extraction_error(self, mock_extract, tmp_path):
        """Should handle and store errors from LLM extraction."""
        results_path = str(tmp_path / "results.json")

        mock_extract.return_value = {
            "error": "API timeout",
            "raw": "partial response"
        }

        extracted_data = [
            {"type": "flowsheet", "image_path": "flow1.png"},
        ]

        run_test_refinement(extracted_data, results_path=results_path)

        with open(results_path) as f:
            results = json.load(f)

        assert "flow1" in results
        assert results["flow1"]["error"] == "API timeout"
