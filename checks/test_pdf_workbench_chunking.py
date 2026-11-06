from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pdf_workbench.custom_logic import (
    EXCEL_COLUMNS,
    build_excel_bytes,
    build_qa_records,
    chunk_document_text,
)


class PdfWorkbenchChunkingTests(unittest.TestCase):
    markdown = """# Main heading

## Details

Original capitalization must survive token chunking. The next sentence adds
enough information to require a second chunk. A final sentence completes the
test document.

| table | cells |
| --- | --- |
| hidden | content |

![hidden image](image.png)
"""

    def test_heading_aware_remains_the_default_and_omits_tables_and_images(self) -> None:
        chunks = chunk_document_text(self.markdown, "sample.pdf", "doc-1")

        self.assertTrue(chunks)
        self.assertIn("Main heading > Details", chunks[0]["section"])
        self.assertIn("Section: Main heading > Details", chunks[0]["source_text"])
        self.assertNotIn("hidden", " ".join(chunk["source_text"] for chunk in chunks))
        self.assertNotIn("image.png", " ".join(chunk["source_text"] for chunk in chunks))

    def test_chonkie_chunkers_preserve_source_case_and_metadata(self) -> None:
        for technique in ("recursive", "sentence", "token"):
            with self.subTest(technique=technique):
                chunks = chunk_document_text(
                    self.markdown,
                    "sample.pdf",
                    "doc-1",
                    chunk_size=12,
                    overlap=2,
                    technique=technique,
                    min_sentences_per_chunk=2,
                )

                self.assertTrue(chunks)
                self.assertEqual(chunks[0]["section"], "Main heading > Details")
                self.assertIn("Original", " ".join(chunk["source_text"] for chunk in chunks))
                self.assertEqual(
                    [chunk["chunk_id"] for chunk in chunks],
                    [f"doc-1:text:{index}" for index in range(1, len(chunks) + 1)],
                )

    def test_semantic_chunker_receives_threshold_and_preserves_source_text(self) -> None:
        section_text = "Original capitalization stays intact."
        fake_chunker = SimpleNamespace(
            chunk=lambda text: [
                SimpleNamespace(start_index=0, end_index=len(text), text=text.lower())
            ]
        )
        with (
            patch(
                "pdf_workbench.custom_logic._get_chonkie_semantic_embedding_model",
                return_value=object(),
            ),
            patch("chonkie.SemanticChunker", return_value=fake_chunker) as semantic_chunker,
        ):
            chunks = chunk_document_text(
                "# Section\n\n" + section_text,
                "sample.pdf",
                "doc-1",
                chunk_size=512,
                technique="semantic",
                semantic_threshold=0.65,
            )

        self.assertEqual(chunks[0]["source_text"], f"Section: Section\n\n{section_text}")
        semantic_chunker.assert_called_once()
        self.assertEqual(semantic_chunker.call_args.kwargs["threshold"], 0.65)

    def test_rejects_invalid_chunking_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document_text(
                self.markdown,
                "sample.pdf",
                "doc-1",
                chunk_size=10,
                overlap=10,
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            chunk_document_text(
                self.markdown,
                "sample.pdf",
                "doc-1",
                technique="semantic",
                semantic_threshold=1.1,
            )
        with self.assertRaisesRegex(ValueError, "exclusive"):
            chunk_document_text(
                self.markdown,
                "sample.pdf",
                "doc-1",
                technique="semantic",
                semantic_threshold=1.0,
            )

    def test_text_qa_export_has_visible_chunk_as_final_excel_column(self) -> None:
        source_chunk = "Section: Details\n\nThe source passage for this question."
        text_qa = [{
            "pdf_name": "sample.pdf",
            "section": "Details",
            "chunk_id": "doc-1:text:1",
            "source_text": source_chunk,
            "chunk_technique": "recursive",
            "page_no": 3,
            "pages": [3],
            "question": "What does the passage say?",
            "expected_answer": "It describes the source passage.",
            "category": "specific_fact",
        }]

        rows = build_qa_records([], text_qa=text_qa)
        workbook_bytes = build_excel_bytes([], text_qa=text_qa)
        row = rows[0]

        self.assertEqual(EXCEL_COLUMNS[-1], "chunk")
        self.assertEqual(row["chunk"], source_chunk)
        self.assertEqual(row["page_no"], 3)
        self.assertEqual(row["object_type"], "text")
        self.assertEqual(row["analysis_mode"], "text chunking: recursive")
        self.assertEqual(row["source_pages"], "[3]")

        from io import BytesIO
        from openpyxl import load_workbook

        worksheet = load_workbook(BytesIO(workbook_bytes)).active
        headers = {
            cell.value: cell.column
            for cell in worksheet[1]
        }
        self.assertEqual(worksheet.cell(row=1, column=worksheet.max_column).value, "chunk")
        self.assertEqual(worksheet.cell(row=2, column=worksheet.max_column).value, source_chunk)
        self.assertFalse(worksheet.column_dimensions[worksheet.cell(row=1, column=worksheet.max_column).column_letter].hidden)
        self.assertEqual(worksheet.cell(row=2, column=headers["page_no"]).value, 3)
        self.assertEqual(worksheet.cell(row=2, column=headers["object_type"]).value, "text")
        self.assertEqual(
            worksheet.cell(row=2, column=headers["analysis_mode"]).value,
            "text chunking: recursive",
        )


if __name__ == "__main__":
    unittest.main()
