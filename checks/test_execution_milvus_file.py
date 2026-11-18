import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.execution import ExecutionService
from services.project_manager import ProjectManager


class MilvusLiteExecutionPathTests(unittest.TestCase):
    def test_existing_database_directory_does_not_skip_vector_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ProjectManager(base_dir=temp_dir)
            project = manager.create("Local Milvus", db_type="milvus_lite")
            vector_config = project["settings"]["vector_store_config"]
            Path(vector_config["persist_directory"]).mkdir()

            collection = Mock()
            collection.name = vector_config["collection_name"]
            collection.count.return_value = 1
            collection.get.return_value = {
                "ids": ["record-1"],
                "embeddings": [[0.1, 0.2, 0.3]],
                "metadatas": [{}],
                "documents": ["test"],
            }
            service = ExecutionService(project["id"], projects_dir=manager.base_dir)

            with patch(
                "services.project_chroma.open_project_collection",
                return_value=collection,
            ):
                result = service._run_vector_checks(collection.name, [])

            self.assertEqual(result["status"], "completed")
