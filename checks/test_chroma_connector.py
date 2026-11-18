from __future__ import annotations

import unittest
from unittest.mock import Mock

from connectors.chroma_connector import ChromaConnector


class ChromaConnectorTests(unittest.TestCase):
    def test_knn_search_returns_document_text_with_each_record(self) -> None:
        collection = Mock()
        collection.query.return_value = {
            "ids": [["chunk-1"]],
            "embeddings": [[[0.1, 0.2]]],
            "metadatas": [[{"pdf_name": "guide.pdf"}]],
            "documents": [["The guide describes the process."]],
        }
        connector = ChromaConnector("documents")
        connector.collection = collection

        records = connector.knn_search([0.3, 0.4], k=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "chunk-1")
        self.assertEqual(
            records[0].metadata,
            {
                "pdf_name": "guide.pdf",
                "document": "The guide describes the process.",
            },
        )
        self.assertEqual(
            collection.query.call_args.kwargs["include"],
            ["embeddings", "metadatas", "documents"],
        )


if __name__ == "__main__":
    unittest.main()
