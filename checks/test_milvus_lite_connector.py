from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from connectors.milvus_lite_connector import (
    MilvusLiteConnector,
    close_database,
)
from services.project_manager import ProjectManager
from services.project_chroma import open_project_collection


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict] = []

    def add_field(self, **field: object) -> None:
        self.fields.append(field)


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **index: object) -> None:
        self.indexes.append(index)


class FakeMilvusClient:
    instances: list["FakeMilvusClient"] = []

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.rows: dict[str, dict] = {}
        self.collection_created = False
        self.collection_state = 3
        self.schema: FakeSchema | None = None
        self.indexes: FakeIndexParams | None = None
        self.load_calls = 0
        self.search_calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def create_schema(self, **_kwargs: object) -> FakeSchema:
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self) -> FakeIndexParams:
        self.indexes = FakeIndexParams()
        return self.indexes

    def create_collection(self, **_kwargs: object) -> None:
        self.collection_created = True

    def has_collection(self, _collection_name: str) -> bool:
        return self.collection_created

    def get_load_state(self, _collection_name: str) -> dict[str, int]:
        return {"state": self.collection_state}

    def load_collection(self, _collection_name: str) -> None:
        self.collection_state = 3
        self.load_calls += 1

    def release_collection(self, _collection_name: str) -> None:
        self.collection_state = 1

    def upsert(self, *, data: list[dict], **_kwargs: object) -> None:
        for row in data:
            self.rows[row["id"]] = row

    def get_collection_stats(self, _collection_name: str) -> dict[str, str]:
        return {"row_count": str(len(self.rows))}

    def query(self, *, filter: str, **_kwargs: object) -> list[dict]:
        if self.collection_state != 3:
            raise RuntimeError("Collection released; call load() before query")
        if filter.startswith("id in "):
            import json

            requested = set(json.loads(filter.removeprefix("id in ")))
            return [row for key, row in self.rows.items() if key in requested]
        if filter.startswith('metadata['):
            key, value = filter.removeprefix("metadata[").split("] == ", 1)
            import json

            metadata_key = json.loads(key)
            expected = json.loads(value)
            return [
                row for row in self.rows.values()
                if row["metadata"].get(metadata_key) == expected
            ]
        return list(self.rows.values())

    def search(self, **kwargs: object) -> list[list[dict]]:
        if self.collection_state != 3:
            raise RuntimeError("Collection released; call load() before search")
        self.search_calls.append(kwargs)
        return [[
            {
                "id": record_id,
                "distance": 0.9,
                "entity": {
                    "embedding": row["embedding"],
                    "document": row["document"],
                    "metadata": row["metadata"],
                },
            }
            for record_id, row in self.rows.items()
        ]]

    def delete(self, *, ids: list[str] | None = None, **_kwargs: object) -> None:
        for record_id in ids or []:
            self.rows.pop(record_id, None)

    def close(self) -> None:
        self.closed = True


def fake_pymilvus_module() -> ModuleType:
    module = ModuleType("pymilvus")
    module.__path__ = []
    module.DataType = SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR", JSON="JSON")
    module.MilvusClient = FakeMilvusClient
    return module


def fake_pymilvus_modules() -> dict[str, ModuleType]:
    module = fake_pymilvus_module()
    client_package = ModuleType("pymilvus.client")
    client_package.__path__ = []
    types_module = ModuleType("pymilvus.client.types")
    types_module.LoadState = SimpleNamespace(Loaded=3, NotLoad=1)
    return {
        "pymilvus": module,
        "pymilvus.client": client_package,
        "pymilvus.client.types": types_module,
    }


class MilvusLiteConnectorTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeMilvusClient.instances.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self._close_test_clients)
        self.pymilvus_patch = patch.dict(
            sys.modules,
            fake_pymilvus_modules(),
        )
        self.pymilvus_patch.start()
        self.addCleanup(self.pymilvus_patch.stop)
        custom_logic = ModuleType("pdf_workbench.custom_logic")
        custom_logic.ensure_torch_available = lambda: None
        self.custom_logic_patch = patch.dict(
            sys.modules,
            {"pdf_workbench.custom_logic": custom_logic},
        )
        self.custom_logic_patch.start()
        self.addCleanup(self.custom_logic_patch.stop)

    def _close_test_clients(self) -> None:
        for client in FakeMilvusClient.instances:
            close_database(client.uri)

    def test_upsert_retrieval_and_search_use_local_milvus_schema(self) -> None:
        database_path = Path(self.temp_dir.name) / "project.db"
        connector = MilvusLiteConnector("docs", str(database_path))
        connector.upsert(
            ids=["record-1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["A local Milvus record."],
            metadatas=[{"category": "policy"}],
        )

        self.assertEqual(connector.count(), 1)
        client = FakeMilvusClient.instances[0]
        self.assertEqual(client.uri, str(database_path.resolve()))
        self.assertEqual(
            next(
                field["dim"] for field in client.schema.fields
                if field["datatype"] == "FLOAT_VECTOR"
            ),
            3,
        )
        client.release_collection("docs")
        self.assertEqual(
            connector.get(
                ids=["record-1"],
                include=["documents", "metadatas", "embeddings"],
            ),
            {
                "ids": ["record-1"],
                "documents": ["A local Milvus record."],
                "metadatas": [{"category": "policy"}],
                "embeddings": [[0.1, 0.2, 0.3]],
            },
        )
        self.assertEqual(client.load_calls, 1)
        client.release_collection("docs")
        result = connector.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            include=["distances", "documents", "embeddings"],
        )
        self.assertEqual(result["ids"], [["record-1"]])
        self.assertAlmostEqual(result["distances"][0][0], 0.1)
        self.assertEqual(client.load_calls, 2)
        self.assertEqual(result["embeddings"], [[[0.1, 0.2, 0.3]]])
        self.assertEqual(
            client.search_calls[-1]["output_fields"],
            ["document", "metadata", "embedding"],
        )

    @unittest.skipUnless(os.name == "nt", "The DLL ordering issue is Windows-specific.")
    def test_torch_runtime_is_initialized_before_milvus_client(self) -> None:
        events = []
        custom_logic = sys.modules["pdf_workbench.custom_logic"]
        custom_logic.ensure_torch_available = lambda: events.append("torch")

        class RecordingMilvusClient(FakeMilvusClient):
            def __init__(self, uri: str) -> None:
                events.append("milvus")
                super().__init__(uri)

        sys.modules["pymilvus"].MilvusClient = RecordingMilvusClient
        connector = MilvusLiteConnector(
            "docs",
            str(Path(self.temp_dir.name) / "order-test.db"),
        )
        connector.connect()

        self.assertEqual(events, ["torch", "milvus"])

    def test_project_creation_and_collection_dispatch_select_milvus_lite(self) -> None:
        manager = ProjectManager(base_dir=self.temp_dir.name)
        project = manager.create("Local Milvus", db_type="milvus_lite")
        vector_config = project["settings"]["vector_store_config"]

        self.assertEqual(vector_config["db_type"], "milvus_lite")
        self.assertEqual(
            Path(vector_config["persist_directory"]),
            Path(self.temp_dir.name) / project["id"] / "milvus_lite.db",
        )
        with self.assertRaises(FileNotFoundError):
            open_project_collection(
                project["id"],
                projects_dir=self.temp_dir.name,
            )
        collection = open_project_collection(
            project["id"],
            create=True,
            projects_dir=self.temp_dir.name,
        )
        self.assertIsInstance(collection, MilvusLiteConnector)
        self.assertEqual(collection.name, vector_config["collection_name"])
        collection.upsert(
            ids=["record-1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["test"],
            metadatas=[{}],
        )
        database_path = Path(vector_config["persist_directory"])
        database_path.mkdir()
        reopened = open_project_collection(
            project["id"],
            projects_dir=self.temp_dir.name,
        )
        self.assertEqual(reopened.count(), 1)
        self.assertFalse(database_path.is_file())
        self.assertTrue(database_path.is_dir())
        with self.assertRaisesRegex(RuntimeError, "Restart the backend"):
            manager.delete(project["id"])
        self.assertTrue(
            (Path(self.temp_dir.name) / project["id"] / "config.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
