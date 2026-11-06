from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import react_workbench


class ReactWorkbenchTests(unittest.TestCase):
    def test_analysis_keeps_session_state_without_returning_credentials(self) -> None:
        credentials = {
            "api_key": "test-key",
            "endpoint": "https://example.test",
            "deployment": "test-deployment",
            "api_version": "test-version",
        }
        chunking = {
            "technique": "heading_aware",
            "chunk_size": 2600,
            "overlap": 250,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        document = {
            "pdf_name": "sample.pdf",
            "document_id": "doc-1",
            "markdown": "Body",
            "page_markdowns": [
                {"page_no": 2, "markdown": "Page body"},
            ],
        }
        object_record = {
            "object_id": "doc-1:page-2:table-1",
            "doc_name": "sample.pdf",
            "page_no": 2,
            "object_type": "table",
            "object_no": "table 1",
            "parts": [],
        }

        def fake_process(paths, output_dir, mode, text_documents=None):
            text_documents.append(document)
            return [], [object_record]

        def fake_chunk(text, pdf_name, document_id, **options):
            return [{
                "chunk_id": document_id,
                "pdf_name": pdf_name,
                "source_text": text,
                "page_no": None,
            }]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(react_workbench, "_output_root", Path(temp_dir)),
                patch.dict(react_workbench._sessions, {}, clear=True),
                patch.object(react_workbench.custom_logic, "configure_azure_openai") as configure,
                patch.object(
                    react_workbench.custom_logic,
                    "process_pdf_files",
                    side_effect=fake_process,
                ),
                patch.object(
                    react_workbench.custom_logic,
                    "chunk_document_text",
                    side_effect=fake_chunk,
                ),
            ):
                result = react_workbench.analyze_pdfs(
                    [("sample.pdf", b"%PDF-1.4 test")],
                    "crop",
                    chunking,
                    credentials,
                )
                configure.assert_called_once_with(
                    "test-key",
                    "https://example.test",
                    "test-deployment",
                    "test-version",
                )

        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["text_chunks"][0]["page_no"], 2)
        self.assertEqual(result["text_chunks"][0]["pages"], [2])
        self.assertNotIn("api_key", result)

    def test_chunking_reports_extraction_errors_and_page_metadata(self) -> None:
        options = {
            "technique": "heading_aware",
            "chunk_size": 2600,
            "overlap": 250,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        documents = [
            {"pdf_name": "bad.pdf", "error": "conversion failed"},
            {
                "pdf_name": "good.pdf",
                "document_id": "doc-2",
                "page_markdowns": [{"page_no": 4, "markdown": "Some prose"}],
            },
        ]
        with patch.object(
            react_workbench.custom_logic,
            "chunk_document_text",
            return_value=[{"chunk_id": "doc-2:text:1"}],
        ) as chunk_document:
            chunks, errors = react_workbench._chunk_documents(documents, options)

        self.assertEqual(errors, [{"pdf_name": "bad.pdf", "error": "conversion failed"}])
        self.assertEqual(chunks[0]["page_no"], 4)
        self.assertEqual(chunks[0]["pages"], [4])
        self.assertEqual(chunk_document.call_args.args[2], "doc-2:page-4")


if __name__ == "__main__":
    unittest.main()
