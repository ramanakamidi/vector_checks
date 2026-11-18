from __future__ import annotations

import os
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from connectors import faiss_connector
from connectors.faiss_connector import FaissConnector
from services.project_chroma import open_project_collection
from services.project_manager import ProjectManager
from services.ingestion import IngestionService


class FaissConnectorTests(unittest.TestCase):
    def test_upsert_query_update_delete_and_persist(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            collection = FaissConnector("docs", temp_dir)
            collection.connect()
            collection.upsert(
                ids=["first", "second"],
                embeddings=[[1.0, 0.0], [0.0, 1.0]],
                documents=["First document", "Second document"],
                metadatas=[{"kind": "text"}, {"kind": "table"}],
            )
            self.assertTrue((collection.persist_directory / "index.faiss").is_file())

            result = collection.query(
                query_embeddings=np.asarray([[0.9, 0.1]], dtype=np.float32),
                n_results=2,
                include=["documents", "metadatas", "distances"],
            )
            self.assertEqual(result["ids"][0][0], "first")
            self.assertEqual(result["documents"][0][0], "First document")
            self.assertEqual(result["metadatas"][0][1]["kind"], "table")
            self.assertAlmostEqual(result["distances"][0][0], 1 - 0.9938837, places=5)

            collection.update(
                ids=["first"],
                metadatas=[{"kind": "updated"}],
            )
            collection.delete(where={"kind": "table"})

            reopened = FaissConnector("docs", temp_dir)
            with patch(
                "connectors.faiss_connector.faiss.read_index",
                wraps=faiss_connector.faiss.read_index,
            ) as read_index:
                reopened.connect()
            read_index.assert_called_once()
            self.assertEqual(reopened.count(), 1)
            self.assertEqual(reopened.indexed_count(), 1)
            self.assertEqual(
                reopened.get(include=["metadatas"])["metadatas"],
                [{"kind": "updated"}],
            )
            self.assertEqual(
                reopened.fetch("first").metadata["document"],
                "First document",
            )

    def test_project_dispatch_uses_configured_faiss_store(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ProjectManager(temp_dir)
            project = manager.create("FAISS project", db_type="faiss")
            collection = open_project_collection(
                project["id"],
                create=True,
                projects_dir=manager.base_dir,
            )
            self.assertIsInstance(collection, FaissConnector)
            self.assertEqual(collection.name, f"{project['id']}_knowledge_base")
            self.assertTrue(collection.persist_directory.is_dir())
            with self.assertRaisesRegex(ValueError, "fixed when the project is created"):
                manager.update_settings(
                    project["id"],
                    {"vector_store_config": {"db_type": "chroma"}},
                )

    def test_project_creation_keeps_chroma_as_the_default(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ProjectManager(temp_dir)
            project = manager.create("Chroma project")
            vector_config = project["settings"]["vector_store_config"]
            self.assertEqual(vector_config["db_type"], "chroma")
            self.assertEqual(
                vector_config["persist_directory"],
                os.path.join(manager.base_dir, project["id"], "chroma_data"),
            )
            collection = Mock()
            collection.name = vector_config["collection_name"]
            client = Mock()
            client.list_collections.return_value = []
            client.get_or_create_collection.return_value = collection
            with patch(
                "services.project_chroma.chromadb.PersistentClient",
                return_value=client,
            ):
                opened = open_project_collection(
                    project["id"],
                    create=True,
                    projects_dir=temp_dir,
                )
            self.assertIs(opened, collection)
            client.get_or_create_collection.assert_called_once()

    def test_json_ingestion_uses_selected_faiss_store(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ProjectManager(temp_dir)
            project = manager.create("FAISS ingestion", db_type="faiss")
            service = IngestionService(project["id"], projects_dir=temp_dir)
            embeddings = ModuleType("services.local_embeddings")
            embeddings.embed_texts = Mock(return_value=[[1.0, 0.0]])
            with patch.dict("sys.modules", {"services.local_embeddings": embeddings}):
                result = service.ingest_json(
                    [{"id": "record-1", "question": "Q", "answer": "A"}]
                )

            self.assertEqual(result["status"], "success")
            collection = open_project_collection(
                project["id"],
                projects_dir=temp_dir,
            )
            self.assertIsInstance(collection, FaissConnector)
            self.assertEqual(collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
