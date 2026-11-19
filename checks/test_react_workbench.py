from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import react_workbench


class ReactWorkbenchTests(unittest.TestCase):
    def test_qa_generation_job_reports_progress_and_completion(self) -> None:
        progress_snapshots = []

        def fake_generate_qa(_session_id, **kwargs):
            callback = kwargs["progress_callback"]
            callback(1, 2, "Generating single-chunk Q&A: 1 of 2 chunks complete")
            callback(2, 2, "Generating single-chunk Q&A: 2 of 2 chunks complete")
            callback(2, 2, "Saving generated Q&A to this session...", "finalizing")
            progress_snapshots.append(
                react_workbench.get_qa_generation_progress(job["job_id"])
            )
            return {"qa_records": [{}, {}, {}, {}]}

        with (
            patch.dict(react_workbench._qa_generation_progress, {}, clear=True),
            patch.object(react_workbench, "generate_qa", side_effect=fake_generate_qa),
        ):
            job = react_workbench.create_qa_generation_job()
            react_workbench.run_qa_generation_job(job["job_id"], "session", {})
            progress = react_workbench.get_qa_generation_progress(job["job_id"])

        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["progress"], 100)
        self.assertEqual(progress["completed"], 2)
        self.assertEqual(progress["total"], 2)
        self.assertEqual(progress["message"], "Generated 4 question-and-answer pairs.")
        self.assertEqual(progress_snapshots[0]["status"], "finalizing")
        self.assertEqual(progress_snapshots[0]["progress"], 95)

    def test_text_generation_can_combine_single_and_multi_hop_modes(self) -> None:
        state = {
            "objects": [],
            "text_chunks": [
                {"chunk_id": f"doc:{index}", "pdf_name": "sample.pdf", "source_text": f"Chunk {index}"}
                for index in range(1, 4)
            ],
            "text_enabled": True,
            "qa_by_object": {},
            "text_qa": [],
        }
        single_pairs = [{"question": "Single-chunk question"}]
        multi_hop_pairs = [{"question": "Multi-hop question"}]
        credentials = {
            "api_key": "test-key",
            "endpoint": "https://example.test",
            "deployment": "test-deployment",
            "api_version": "test-version",
        }
        progress_updates = []

        def generate_single(chunks, _question_count, progress_callback):
            progress_callback(len(chunks), len(chunks))
            return single_pairs

        def generate_multi_hop(chunks, _chunk_count, _question_count, progress_callback):
            progress_callback(1, 1)
            return multi_hop_pairs

        with (
            patch.object(react_workbench, "_get_state", return_value=state),
            patch.object(react_workbench, "_save_session"),
            patch.object(react_workbench, "_public_session", return_value={}),
            patch.object(react_workbench.custom_logic, "configure_azure_openai"),
            patch.object(
                react_workbench.custom_logic,
                "generate_qa_for_text_chunks",
                side_effect=generate_single,
            ) as generate_single,
            patch.object(
                react_workbench.custom_logic,
                "generate_qa_for_adjacent_text_chunks",
                side_effect=generate_multi_hop,
            ) as generate_multi_hop,
        ):
            react_workbench.generate_qa(
                "session",
                [],
                False,
                True,
                5,
                2,
                credentials,
                generate_single_chunk=True,
                generate_multi_hop=True,
                multi_hop_chunk_count=3,
                multi_hop_question_count=1,
                progress_callback=lambda completed, total, message, status="running": progress_updates.append(
                    (completed, total, message, status)
                ),
            )

        generate_single.assert_called_once()
        generate_multi_hop.assert_called_once()
        self.assertEqual(state["text_qa"], single_pairs + multi_hop_pairs)
        self.assertEqual(progress_updates[-1], (
            4,
            4,
            "Preparing generated Q&A results...",
            "finalizing",
        ))
        self.assertIn("running", {update[3] for update in progress_updates})

    def test_saved_session_restores_object_images_after_memory_is_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            session_id = "a" * 32
            session_dir = output_root / session_id
            object_dir = session_dir / "objects"
            object_dir.mkdir(parents=True)
            image_path = object_dir / "sample_page2_table1.png"
            image_path.write_bytes(b"png")
            state = {
                "session_dir": str(session_dir),
                "objects": [{
                    "object_id": "doc-1:page-2:table-1",
                    "doc_name": "sample.pdf",
                    "page_no": 2,
                    "pages": [2],
                    "object_type": "table",
                    "object_no": "table 1",
                    "position": "top",
                    "docling_title": "Sample table",
                    "characteristics": {"content_summary": "A test table."},
                    "parts": [{"image_path": str(image_path)}],
                }],
                "documents": [],
                "text_chunks": [],
                "text_errors": [],
                "chunking": {},
                "mode": "crop",
                "qa_by_object": {},
                "text_qa": [],
            }

            with (
                patch.object(react_workbench, "_output_root", output_root),
                patch.dict(react_workbench._sessions, {}, clear=True),
                patch.object(react_workbench.custom_logic, "build_qa_records", return_value=[]),
            ):
                react_workbench._save_session(session_id, state)
                restored = react_workbench.get_session(session_id)
                restored_image = react_workbench.get_object_image(
                    session_id,
                    "doc-1:page-2:table-1",
                    0,
                )

        self.assertEqual(restored["objects"][0]["image_urls"], [
            f"/api/workbench/{session_id}/objects/doc-1:page-2:table-1/image?part=0",
        ])
        self.assertEqual(restored_image, image_path.resolve())

    def test_analysis_keeps_session_state_without_returning_credentials(self) -> None:
        credentials = {
            "api_key": "test-key",
            "endpoint": "https://example.test",
            "deployment": "test-deployment",
            "api_version": "test-version",
        }
        chunking = {
            "technique": "sentence",
            "chunk_size": 512,
            "overlap": 0,
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

        processed_paths = []

        def fake_process(
            paths,
            output_dir,
            mode,
            progress_callback=None,
            text_documents=None,
            object_types=None,
        ):
            processed_paths.extend(paths)
            self.assertEqual(object_types, {"table"})
            if progress_callback:
                progress_callback("Analyzing sample.pdf page 2", 0.5)
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
            analysis_job = react_workbench.create_analysis_job()
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
                    ["table", "text"],
                    analysis_job["job_id"],
                )
                configure.assert_called_once_with(
                    "test-key",
                    "https://example.test",
                    "test-deployment",
                    "test-version",
                )

        self.assertEqual(len(result["objects"]), 1)
        self.assertTrue(result["text_enabled"])
        self.assertEqual(result["text_chunks"][0]["page_no"], 2)
        self.assertEqual(result["text_chunks"][0]["pages"], [2])
        self.assertNotIn("api_key", result)
        self.assertEqual(processed_paths[0].name, "sample.pdf")
        self.assertEqual(
            react_workbench.get_analysis_progress(analysis_job["job_id"])["status"],
            "complete",
        )
        self.assertEqual(
            react_workbench.get_analysis_progress(analysis_job["job_id"])["progress"],
            100,
        )

    def test_analysis_returns_only_selected_object_types(self) -> None:
        credentials = {
            "api_key": "test-key",
            "endpoint": "https://example.test",
            "deployment": "test-deployment",
            "api_version": "test-version",
        }
        chunking = {
            "technique": "sentence",
            "chunk_size": 512,
            "overlap": 0,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        records = [
            {
                "object_id": "doc-1:page-1:image-1",
                "doc_name": "sample.pdf",
                "page_no": 1,
                "object_type": "image",
                "object_no": "image 1",
                "parts": [],
            },
            {
                "object_id": "doc-1:page-1:table-1",
                "doc_name": "sample.pdf",
                "page_no": 1,
                "object_type": "table",
                "object_no": "table 1",
                "parts": [],
            },
            {
                "object_id": "doc-1:page-1:graph-1",
                "doc_name": "sample.pdf",
                "page_no": 1,
                "object_type": "graph",
                "object_no": "graph 1",
                "parts": [],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(react_workbench, "_output_root", Path(temp_dir)),
                patch.dict(react_workbench._sessions, {}, clear=True),
                patch.object(react_workbench.custom_logic, "configure_azure_openai"),
                patch.object(
                    react_workbench.custom_logic,
                    "process_pdf_files",
                    return_value=([], records),
                ) as process_files,
                patch.object(react_workbench, "_chunk_documents", return_value=([], [])) as chunk_documents,
            ):
                result = react_workbench.analyze_pdfs(
                    [("sample.pdf", b"%PDF-1.4 test")],
                    "crop",
                    chunking,
                    credentials,
                    ["image", "graph"],
                )

        self.assertEqual(
            [item["object_type"] for item in result["objects"]],
            ["image", "graph"],
        )
        self.assertEqual(process_files.call_args.kwargs["object_types"], {"image", "graph"})
        self.assertIsNone(process_files.call_args.kwargs["text_documents"])
        self.assertFalse(result["text_enabled"])
        self.assertEqual(result["text_chunks"], [])
        chunk_documents.assert_not_called()

    def test_text_only_analysis_extracts_and_chunks_text_without_object_detection(self) -> None:
        credentials = {
            "api_key": "test-key",
            "endpoint": "https://example.test",
            "deployment": "test-deployment",
            "api_version": "test-version",
        }
        chunking = {
            "technique": "sentence",
            "chunk_size": 512,
            "overlap": 0,
            "min_sentences_per_chunk": 1,
            "semantic_threshold": 0.8,
        }
        document = {
            "pdf_name": "sample.pdf",
            "document_id": "doc-1",
            "page_markdowns": [{"page_no": 1, "markdown": "Text body"}],
        }

        def fake_process(paths, output_dir, mode, text_documents=None, object_types=None):
            self.assertEqual(object_types, set())
            self.assertIsNotNone(text_documents)
            text_documents.append(document)
            return [], []

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(react_workbench, "_output_root", Path(temp_dir)),
                patch.dict(react_workbench._sessions, {}, clear=True),
                patch.object(react_workbench.custom_logic, "configure_azure_openai"),
                patch.object(
                    react_workbench.custom_logic,
                    "process_pdf_files",
                    side_effect=fake_process,
                ) as process_files,
                patch.object(
                    react_workbench,
                    "_chunk_documents",
                    return_value=([{"chunk_id": "doc-1:text:1"}], []),
                ) as chunk_documents,
            ):
                result = react_workbench.analyze_pdfs(
                    [("sample.pdf", b"%PDF-1.4 test")],
                    "crop",
                    chunking,
                    credentials,
                    ["text"],
                )

        self.assertEqual(process_files.call_args.kwargs["object_types"], set())
        self.assertTrue(result["text_enabled"])
        self.assertEqual(result["text_chunks"], [{"chunk_id": "doc-1:text:1"}])
        chunk_documents.assert_called_once_with([document], chunking)

    def test_chunking_reports_extraction_errors_and_page_metadata(self) -> None:
        options = {
            "technique": "sentence",
            "chunk_size": 512,
            "overlap": 0,
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
