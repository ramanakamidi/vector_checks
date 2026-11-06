"""
Chroma connector.

Requires: pip install chromadb
Works with both the in-process (persistent) client and the HTTP client.
"""

from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


class ChromaConnector(VectorDBConnector):
    def __init__(self, collection_name: str,
                 persist_directory: Optional[str] = None,
                 host: Optional[str] = None, port: int = 8000):
        """
        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Path for persistent local storage (uses PersistentClient).
            host: If provided, connects via HTTP to a Chroma server instead.
            port: Port for the HTTP client (default 8000).
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.client = None
        self.collection = None

    def connect(self) -> None:
        import chromadb
        if self.host:
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        elif self.persist_directory:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()  # ephemeral in-memory
        self.collection = self.client.get_collection(self.collection_name)

    def _safe_embedding(self, embeddings, index: int) -> List[float]:
        """Safely extract an embedding at the given index.
        Chroma may return embeddings as a numpy array or a list,
        so we can't use simple truthiness checks."""
        if embeddings is None:
            return []
        try:
            vec = embeddings[index]
            return list(vec) if vec is not None else []
        except (IndexError, TypeError):
            return []

    def _safe_metadata(self, metadatas, index: int) -> Dict:
        """Safely extract metadata at the given index."""
        if metadatas is None:
            return {}
        try:
            meta = metadatas[index]
            return dict(meta) if meta is not None else {}
        except (IndexError, TypeError):
            return {}

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        result = self.collection.get(
            ids=[record_id], include=["embeddings", "metadatas", "documents"]
        )
        if not result["ids"]:
            return None
        meta = self._safe_metadata(result["metadatas"], 0)
        # Include document text in metadata if Chroma stored it
        docs = result.get("documents")
        if docs and len(docs) > 0 and docs[0]:
            meta["document"] = docs[0]
        return VectorRecord(
            id=result["ids"][0],
            vector=self._safe_embedding(result["embeddings"], 0),
            metadata=meta,
        )

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        result = self.collection.query(
            query_embeddings=[vector], n_results=k,
            include=["embeddings", "metadatas"],
        )
        records = []
        ids_list = result["ids"][0] if result["ids"] else []
        embeddings = result["embeddings"][0] if result["embeddings"] is not None else None
        metadatas = result["metadatas"][0] if result["metadatas"] is not None else None
        for i, rid in enumerate(ids_list):
            records.append(VectorRecord(
                id=rid,
                vector=self._safe_embedding(embeddings, i),
                metadata=self._safe_metadata(metadatas, i),
            ))
        return records

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Page through all records using Chroma's get() with offset/limit."""
        total = self.collection.count()
        offset = 0
        while offset < total:
            result = self.collection.get(
                include=["embeddings", "metadatas"],
                limit=batch_size, offset=offset,
            )
            if not result["ids"]:
                break
            for i, rid in enumerate(result["ids"]):
                yield VectorRecord(
                    id=rid,
                    vector=self._safe_embedding(result["embeddings"], i),
                    metadata=self._safe_metadata(result["metadatas"], i),
                )
            offset += len(result["ids"])

    def count(self) -> int:
        return self.collection.count()

    def indexed_count(self) -> int:
        """Count stored records that actually have a non-empty embedding.

        Chroma does not expose a portable count of its internal ANN index
        through the collection API. Searchability is checked separately by
        querying every vector-bearing record against the index.
        """
        return sum(
            1
            for record in self.scan(batch_size=500)
            if record.vector
        )

    def filter_metadata(self, field: str, value: Any, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Use Chroma's native where filter for server-side metadata filtering."""
        total = self.collection.count()
        offset = 0
        while offset < total:
            result = self.collection.get(
                where={field: value},
                include=["embeddings", "metadatas"],
                limit=batch_size, offset=offset,
            )
            if not result["ids"]:
                break
            for i, rid in enumerate(result["ids"]):
                yield VectorRecord(
                    id=rid,
                    vector=self._safe_embedding(result["embeddings"], i),
                    metadata=self._safe_metadata(result["metadatas"], i),
                )
            offset += len(result["ids"])
