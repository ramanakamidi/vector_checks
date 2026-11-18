from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pdf_workbench import custom_logic
from pdf_workbench.custom_logic import (
    EXCEL_COLUMNS,
    build_excel_bytes,
    build_qa_records,
    chunk_document_text,
    extract_objects_from_pdf,
    generate_qa_for_text_chunk,
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

    @staticmethod
    def _sentence_chunker_modules() -> dict[str, types.ModuleType]:
        class OffsetTokenizer:
            def encode(self, text, add_special_tokens=True):
                token_ids = list(range(len(text)))
                return [101, *token_ids, 102] if add_special_tokens else token_ids

            def __call__(
                self,
                text,
                *,
                add_special_tokens=True,
                return_offsets_mapping=False,
            ):
                token_ids = self.encode(text, add_special_tokens=add_special_tokens)
                offsets = (
                    [(0, 0), *[(index, index + 1) for index in range(len(text))], (0, 0)]
                    if add_special_tokens
                    else [(index, index + 1) for index in range(len(text))]
                )
                return {"input_ids": token_ids, "offset_mapping": offsets}

        class FakeChunker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def chunk(self, text):
                return [SimpleNamespace(start_index=0, end_index=len(text), text=text)]

        class FakeTokenChunker(FakeChunker):
            def chunk(self, text):
                token_count = len(self.kwargs["tokenizer"].encode(text))
                step = self.kwargs["chunk_size"] - self.kwargs["chunk_overlap"]
                results = []
                for token_start in range(0, token_count, step):
                    token_end = min(
                        token_start + self.kwargs["chunk_size"],
                        token_count,
                    )
                    results.append(SimpleNamespace(
                        start_index=len(text) + 1,
                        end_index=len(text) + 10,
                        text=text.lower(),
                    ))
                    if token_end == token_count:
                        break
                return results

        chonkie = types.ModuleType("chonkie")
        for name in ("RecursiveChunker", "SentenceChunker", "SemanticChunker"):
            setattr(chonkie, name, FakeChunker)
        chonkie.TokenChunker = FakeTokenChunker
        embeddings = types.ModuleType("services.local_embeddings")
        embeddings.get_project_tokenizer = OffsetTokenizer
        return {
            "chonkie": chonkie,
            "services.local_embeddings": embeddings,
        }

    def test_sentence_is_the_default_and_omits_tables_and_images(self) -> None:
        with patch.dict(sys.modules, self._sentence_chunker_modules()):
            chunks = chunk_document_text(self.markdown, "sample.pdf", "doc-1")

        self.assertTrue(chunks)
        self.assertIn("Main heading > Details", chunks[0]["section"])
        self.assertNotIn("Section:", chunks[0]["source_text"])
        self.assertNotIn("hidden", " ".join(chunk["source_text"] for chunk in chunks))
        self.assertNotIn("image.png", " ".join(chunk["source_text"] for chunk in chunks))

    def test_skips_caption_only_chunks_but_keeps_captions_with_prose(self) -> None:
        for technique in ("recursive", "sentence", "token", "semantic"):
            with self.subTest(technique=technique):
                with (
                    patch.dict(sys.modules, self._sentence_chunker_modules()),
                    patch.object(
                        custom_logic,
                        "_get_chonkie_semantic_embedding_model",
                        return_value=object(),
                    ),
                ):
                    caption_only = chunk_document_text(
                        "# Figures\n\nFigure 1: RAG pipeline architecture <!-- image -->",
                        "sample.pdf",
                        "doc-1",
                        technique=technique,
                    )
                    chunks_with_prose = chunk_document_text(
                        "# Figures\n\n"
                        "The architecture connects retrieval with generation.\n\n"
                        "Figure 1: RAG pipeline architecture <!-- image -->",
                        "sample.pdf",
                        "doc-1",
                        technique=technique,
                    )

                self.assertEqual(caption_only, [])
                self.assertEqual(len(chunks_with_prose), 1)
                self.assertIn(
                    "The architecture connects retrieval with generation.",
                    chunks_with_prose[0]["source_text"],
                )
                self.assertIn(
                    "Figure 1: RAG pipeline architecture",
                    chunks_with_prose[0]["source_text"],
                )

    def test_recursive_chunking_combines_short_heading_sections(self) -> None:
        markdown = (
            "# The State of Technology in 2025\n\n"
            "A Technical Overview of AI, Cloud, Semiconductors and Cybersecurity\n\n"
            "# 1. Introduction\n\n"
            "The technology industry is undergoing major transformations. "
            "AI assistants are now widely used.\n\n"
            "# 2. Artificial Intelligence and Vector Databases\n\n"
            "Modern AI applications rely on embeddings and vector databases."
        )
        embeddings = types.ModuleType("services.local_embeddings")
        embeddings.get_project_tokenizer = lambda: "character"

        with patch.dict(sys.modules, {"services.local_embeddings": embeddings}):
            chunks = chunk_document_text(
                markdown,
                "sample.pdf",
                "doc-1",
                technique="recursive",
                chunk_size=512,
            )

        self.assertEqual(len(chunks), 1)
        self.assertIn("A Technical Overview", chunks[0]["source_text"])
        self.assertIn("The technology industry", chunks[0]["source_text"])
        self.assertIn("Modern AI applications", chunks[0]["source_text"])

    def test_token_chunking_uses_source_offsets_not_decoded_chunk_offsets(self) -> None:
        source_text = "Café and résumé preserve their original casing and accents. " * 4
        with patch.dict(sys.modules, self._sentence_chunker_modules()):
            chunks = chunk_document_text(
                "# Unicode\n\n" + source_text,
                "sample.pdf",
                "doc-1",
                technique="token",
                chunk_size=64,
                overlap=10,
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0]["source_text"].startswith("Café"))
        self.assertTrue(
            all(chunk["source_text"] in source_text for chunk in chunks)
        )

    def test_all_chunkers_cross_heading_sections_and_retain_section_context(self) -> None:
        markdown = (
            "# Introduction\n\nFirst sentence.\n\n"
            "# Details\n\nSecond sentence.\n\n"
            "# Conclusion\n\nThird sentence."
        )
        for technique in ("recursive", "sentence", "token", "semantic"):
            with self.subTest(technique=technique):
                with (
                    patch.dict(sys.modules, self._sentence_chunker_modules()),
                    patch.object(
                        custom_logic,
                        "_get_chonkie_semantic_embedding_model",
                        return_value=object(),
                    ),
                ):
                    chunks = chunk_document_text(
                        markdown,
                        "sample.pdf",
                        "doc-1",
                        technique=technique,
                        min_sentences_per_chunk=3,
                    )

            self.assertEqual(len(chunks), 1)
            self.assertEqual(
                chunks[0]["section"],
                "Introduction | Details | Conclusion",
            )
            self.assertNotIn("Section:", chunks[0]["source_text"])
            self.assertIn("First sentence.", chunks[0]["source_text"])
            self.assertIn("Second sentence.", chunks[0]["source_text"])
            self.assertIn("Third sentence.", chunks[0]["source_text"])

    def test_chonkie_chunkers_preserve_source_case_and_metadata(self) -> None:
        for technique in ("recursive", "sentence", "token"):
            with self.subTest(technique=technique), patch.dict(
                sys.modules,
                self._sentence_chunker_modules(),
            ):
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
                SimpleNamespace(start_index=0, end_index=len(text), text=text)
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

        self.assertEqual(chunks[0]["source_text"], section_text)
        semantic_chunker.assert_called_once()
        self.assertEqual(semantic_chunker.call_args.kwargs["threshold"], 0.65)
        self.assertEqual(semantic_chunker.call_args.kwargs["min_sentences_per_chunk"], 1)

    def test_text_qa_generation_receives_section_as_separate_context(self) -> None:
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"pairs":[]}'))],
        ))
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )
        with (
            patch.object(custom_logic, "client", mock_client),
            patch.object(custom_logic, "MODEL_NAME", "test-deployment"),
            patch.object(
                custom_logic,
                "_parse_qa_with_category",
                return_value={"pairs": []},
            ),
        ):
            result = generate_qa_for_text_chunk(
                {"source_text": "The text content.", "section": "Introduction"},
                num_qa=1,
            )

        self.assertEqual(result, [])
        user_message = create.call_args.kwargs["messages"][1]["content"]
        evidence = json.loads(user_message.split(":\n", 1)[1])
        self.assertEqual(evidence, {
            "document_text": "The text content.",
            "document_section": "Introduction",
        })

    def test_rejects_invalid_chunking_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document_text(
                self.markdown,
                "sample.pdf",
                "doc-1",
                chunk_size=10,
                overlap=10,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            chunk_document_text(
                self.markdown,
                "sample.pdf",
                "doc-1",
                technique="heading_aware",
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
        source_chunk = (
            "Section: Details\n\n"
            + "The full source passage for this question must remain visible in Excel. " * 24
        )
        question = "What does this source passage explain?"
        text_qa = [{
            "pdf_name": "sample.pdf",
            "section": "Details",
            "chunk_id": "doc-1:text:1",
            "source_text": source_chunk,
            "chunk_technique": "recursive",
            "page_no": 3,
            "pages": [3],
            "question": question,
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
        self.assertTrue(
            {
                "source_type",
                "source_id",
                "source_section",
                "source_text",
                "source_pages",
            }.isdisjoint(headers)
        )
        self.assertEqual(worksheet.cell(row=1, column=worksheet.max_column).value, "chunk")
        self.assertEqual(worksheet.cell(row=2, column=worksheet.max_column).value, source_chunk)
        self.assertFalse(worksheet.column_dimensions[worksheet.cell(row=1, column=worksheet.max_column).column_letter].hidden)
        self.assertEqual(worksheet.cell(row=2, column=headers["page_no"]).value, 3)
        self.assertEqual(worksheet.cell(row=2, column=headers["object_type"]).value, "text")
        self.assertEqual(
            worksheet.cell(row=2, column=headers["analysis_mode"]).value,
            "text chunking: recursive",
        )
        self.assertEqual(worksheet.cell(row=2, column=headers["question"]).value, question)
        chunk_cell = worksheet.cell(row=2, column=headers["chunk"])
        self.assertEqual(chunk_cell.value, source_chunk)
        self.assertTrue(chunk_cell.alignment.wrap_text)
        self.assertEqual(chunk_cell.alignment.vertical, "top")
        self.assertGreater(worksheet.row_dimensions[2].height, 15)
        self.assertEqual(
            worksheet.column_dimensions[chunk_cell.column_letter].width,
            100,
        )

    def test_table_markdown_is_extracted_and_added_to_each_table_question(self) -> None:
        markdown = "| Year | Revenue |\n| --- | --- |\n| 2025 | $12M |"
        table = SimpleNamespace(
            prov=[SimpleNamespace(page_no=1)],
            captions=[],
            export_to_markdown=Mock(return_value=markdown),
        )
        document = SimpleNamespace(tables=[table], pictures=[])
        converter = SimpleNamespace(
            convert=Mock(return_value=SimpleNamespace(document=document))
        )
        crop = {
            "page_no": 1,
            "top": 0,
            "path": "table.png",
            "width": 200,
            "height": 100,
            "position": "top",
            "caption_in_crop": False,
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(custom_logic, "get_converter", return_value=converter),
            patch.object(custom_logic, "crop_object_to_image", return_value=crop),
        ):
            records = extract_objects_from_pdf("sample.pdf", temp_dir)

        self.assertEqual(records[0]["table_markdown"], markdown)
        table.export_to_markdown.assert_called_once_with(doc=document)

        object_record = {
            "object_id": "doc-1:page-1:table-1",
            "doc_name": "sample.pdf",
            "page_no": 1,
            "pages": [1],
            "object_type": "table",
            "object_no": "table 1",
            "table_markdown": markdown,
        }
        qa_by_object = {
            object_record["object_id"]: [
                {"question": "What was revenue in 2025?", "expected_answer": "$12M"},
                {"question": "Which year is listed?", "expected_answer": "2025"},
            ],
        }
        qa_rows = build_qa_records([object_record], qa_by_object)

        self.assertEqual([row["chunk"] for row in qa_rows], [markdown, markdown])
        self.assertEqual([row["source_text"] for row in qa_rows], [markdown, markdown])

    def test_merged_table_keeps_markdown_from_all_pages(self) -> None:
        records = [
            {
                "object_type": "table",
                "doc_name": "sample.pdf",
                "_doc_idx": 0,
                "_top": 10,
                "page_no": 1,
                "pages": [1],
                "parts": [{"page_no": 1, "image_path": "page1.png"}],
                "table_markdown": "| Year | Revenue |\n| --- | --- |\n| 2024 | $10M |",
                "docling_title": "Revenue",
                "characteristics": {"continues_on_next_page": True},
            },
            {
                "object_type": "table",
                "doc_name": "sample.pdf",
                "_doc_idx": 0,
                "_top": 10,
                "page_no": 2,
                "pages": [2],
                "parts": [{"page_no": 2, "image_path": "page2.png"}],
                "table_markdown": "| Year | Revenue |\n| --- | --- |\n| 2025 | $12M |",
                "docling_title": None,
                "characteristics": {"continues_from_previous_page": True},
            },
        ]

        with patch.object(
            custom_logic,
            "analyze_object",
            return_value={"classification": "table", "characteristics": {}},
        ):
            merged = custom_logic.merge_continued_tables(records)

        self.assertEqual(len(merged), 1)
        self.assertIn("2024", merged[0]["table_markdown"])
        self.assertIn("2025", merged[0]["table_markdown"])
        self.assertEqual(merged[0]["pages"], [1, 2])


if __name__ == "__main__":
    unittest.main()
