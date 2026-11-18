from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from connectors.base import VectorRecord
from checks.negative_queries import check_negative_queries


class NegativeQueryTests(unittest.TestCase):
    def test_uses_connector_interface_for_cosine_search_and_nearest_text(self) -> None:
        class ConnectorWithoutCollection:
            def __init__(self) -> None:
                self.search_calls: list[tuple[list[float], int]] = []
                self.fetch_calls: list[str] = []

            def knn_search(self, vector: list[float], k: int) -> list[VectorRecord]:
                self.search_calls.append((vector, k))
                return [VectorRecord("record-1", [6.0, 8.0], {})]

            def fetch(self, record_id: str) -> VectorRecord:
                self.fetch_calls.append(record_id)
                return VectorRecord(
                    record_id,
                    [6.0, 8.0],
                    {"document": "Nearest matching text"},
                )

        connector = ConnectorWithoutCollection()
        with patch(
            "services.local_embeddings.embed_text_query",
            return_value=[0.6, 0.8],
        ) as embed_texts:
            result = check_negative_queries(
                connector,
                ["  unrelated question  "],
                maximum_similarity=0.9,
            )

        embed_texts.assert_called_once_with("unrelated question")
        self.assertEqual(connector.search_calls, [([0.6, 0.8], 1)])
        self.assertEqual(connector.fetch_calls, ["record-1"])
        self.assertFalse(hasattr(connector, "collection"))
        self.assertFalse(result["passed"])
        query_result = result["detail"]["query_results"][0]
        self.assertEqual(query_result["cosine_similarity"], 1.0)
        self.assertEqual(
            query_result["nearest_chunk"],
            {"id": "record-1", "text": "Nearest matching text"},
        )

    def test_rejects_chroma_collections_that_do_not_use_cosine(self) -> None:
        connector = SimpleNamespace(
            collection=SimpleNamespace(metadata={"hnsw:space": "l2"}),
        )

        with self.assertRaisesRegex(ValueError, "requires a Chroma collection"):
            check_negative_queries(connector, ["off-topic question"])

    def test_rejects_nearest_vectors_with_mismatched_dimensions(self) -> None:
        class ConnectorWithoutCollection:
            def knn_search(self, _vector: list[float], k: int) -> list[VectorRecord]:
                return [VectorRecord("record-1", [1.0], {})]

        with (
            patch(
                "services.local_embeddings.embed_text_query",
                return_value=[1.0, 0.0],
            ),
            self.assertRaisesRegex(ValueError, "different dimension"),
        ):
            check_negative_queries(
                ConnectorWithoutCollection(),
                ["off-topic question"],
            )


if __name__ == "__main__":
    unittest.main()
