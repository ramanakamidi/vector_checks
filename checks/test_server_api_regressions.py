import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    import server


class ServerApiRegressionTests(unittest.TestCase):
    def test_pdf_ingestion_returns_before_background_processing_finishes(self) -> None:
        class FakeUpload:
            filename = "sample.pdf"

            async def read(self) -> bytes:
                return b"%PDF sample"

            async def close(self) -> None:
                pass

        outcome = {
            "status": "ingested",
            "pdf_name": "sample.pdf",
            "chunks_ingested": 2,
            "image_records_ingested": 1,
            "text_records_ingested": 1,
            "table_records_ingested": 0,
            "collection_name": "project_collection",
        }
        background_tasks = server.BackgroundTasks()
        with (
            patch.object(server._pm, "get_project_dir", return_value="C:/projects/project"),
            patch.dict(server.PDF_INGESTION_JOBS, {}, clear=True),
            patch.object(server, "ingest_pdf", return_value=outcome) as ingest_pdf,
        ):
            response = asyncio.run(
                server.ingest_project_pdfs(
                    "project",
                    background_tasks,
                    [FakeUpload()],
                    "job-1",
                    "{}",
                )
            )

            self.assertEqual(response, {"status": "accepted", "job_id": "job-1"})
            ingest_pdf.assert_not_called()
            self.assertEqual(
                server.PDF_INGESTION_JOBS["job-1"]["status"],
                "running",
            )

            asyncio.run(background_tasks())
            progress = asyncio.run(
                server.get_pdf_ingestion_progress("project", "job-1")
            )

        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["result"]["chunks_ingested"], 2)
        ingest_pdf.assert_called_once()

    def test_project_vector_records_returns_full_text_for_requested_page(self) -> None:
        collection = Mock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "ids": ["chunk-2"],
            "documents": ["Full text for the second page."],
            "metadatas": [{"pdf_name": "guide.pdf", "page_number": 4}],
        }
        project_dir = "C:/projects/project-1"
        with (
            patch.object(server._pm, "get_project_dir", return_value=project_dir),
            patch.object(server, "open_project_collection", return_value=collection) as open_collection,
        ):
            result = asyncio.run(
                server.get_project_vector_records("project-1", page=2, page_size=1)
            )

        self.assertEqual(result, {
            "records": [{
                "id": "chunk-2",
                "text": "Full text for the second page.",
                "metadata": {"pdf_name": "guide.pdf", "page_number": 4},
            }],
            "total": 3,
            "page": 2,
            "page_size": 1,
        })
        open_collection.assert_called_once_with(
            "project-1",
            create=True,
            projects_dir=Path(project_dir).parent,
        )
        collection.get.assert_called_once_with(
            limit=1,
            offset=1,
            include=["documents", "metadatas"],
        )

    def test_collections_lists_active_faiss_collection_without_client(self) -> None:
        connector = SimpleNamespace(collection_name="sample_faiss")
        with patch.dict(
            server._state,
            {"connector": connector, "db_type": "faiss"},
        ):
            result = asyncio.run(server.list_collections())

        self.assertEqual(result, {"collections": ["sample_faiss"]})

    def test_random_projection_is_labeled_and_deterministic(self) -> None:
        connector = SimpleNamespace(
            scan=lambda: iter(
                [
                    SimpleNamespace(id="one", vector=[1.0, 0.0, 0.0], metadata={}),
                    SimpleNamespace(id="two", vector=[0.0, 1.0, 0.0], metadata={}),
                    SimpleNamespace(id="three", vector=[0.0, 0.0, 1.0], metadata={}),
                ]
            )
        )
        with patch.dict(server._state, {"connector": connector}):
            first = asyncio.run(server.get_embedding_data(10, "random"))
            second = asyncio.run(server.get_embedding_data(10, "random"))
            with patch.object(
                server,
                "_pca_project",
                return_value=[[1, 2], [3, 4], [5, 6]],
            ):
                pca = asyncio.run(server.get_embedding_data(10, "pca"))

        self.assertEqual(first["method"], "RANDOM")
        self.assertEqual(first["points"], second["points"])
        self.assertEqual(len(first["points"]), 3)
        self.assertEqual(pca["method"], "PCA")
        self.assertEqual(len(pca["points"]), 3)

    def test_insufficient_random_projection_keeps_requested_method(self) -> None:
        connector = SimpleNamespace(
            scan=lambda: iter(
                [SimpleNamespace(id="one", vector=[1.0, 0.0], metadata={})]
            )
        )
        with patch.dict(server._state, {"connector": connector}):
            result = asyncio.run(server.get_embedding_data(10, "random"))

        self.assertEqual(result["method"], "RANDOM")
        self.assertEqual(result["points"], [])
        self.assertIn("Not enough vectors", result["message"])
