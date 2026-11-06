"""
Pinecone connector.

Requires: pip install pinecone-client>=3.0
Assumes a Pinecone index already exists with the correct dimension.
"""

from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


class PineconeConnector(VectorDBConnector):
    def __init__(self, api_key: str, index_name: str,
                 environment: Optional[str] = None, namespace: str = ""):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.namespace = namespace
        self.index = None

    def connect(self) -> None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=self.api_key)
        self.index = pc.Index(self.index_name)

    def _match_to_record(self, match: Any) -> VectorRecord:
        mid = getattr(match, "id", None) or match.get("id", "") if isinstance(match, dict) else getattr(match, "id", "")
        values = getattr(match, "values", None) or (match.get("values", []) if isinstance(match, dict) else [])
        metadata = getattr(match, "metadata", None) or (match.get("metadata", {}) if isinstance(match, dict) else {})
        return VectorRecord(
            id=str(mid),
            vector=list(values) if values else [],
            metadata=dict(metadata) if metadata else {},
        )

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        result = self.index.fetch(ids=[record_id], namespace=self.namespace)
        vectors = getattr(result, "vectors", None) or result.get("vectors", {})
        if record_id not in vectors:
            return None
        v = vectors[record_id]
        vec_values = getattr(v, "values", None) or (v.get("values", []) if isinstance(v, dict) else [])
        vec_metadata = getattr(v, "metadata", None) or (v.get("metadata", {}) if isinstance(v, dict) else {})
        return VectorRecord(id=record_id, vector=list(vec_values),
                            metadata=dict(vec_metadata) if vec_metadata else {})

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        result = self.index.query(
            vector=vector, top_k=k, include_values=True,
            include_metadata=True, namespace=self.namespace,
        )
        matches = getattr(result, "matches", None) or result.get("matches", [])
        return [self._match_to_record(m) for m in matches]

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Page through all vectors using Pinecone's list + fetch pattern.
        Pinecone doesn't have a native scroll/scan — we list IDs in pages,
        then fetch each page's vectors in bulk."""
        paginator = self.index.list_paginated(
            namespace=self.namespace, limit=batch_size
        )
        while True:
            ids = [v.id for v in paginator.vectors]
            if not ids:
                break
            fetched = self.index.fetch(ids=ids, namespace=self.namespace)
            for vid, vdata in fetched.get("vectors", {}).items():
                yield VectorRecord(
                    id=vid,
                    vector=vdata.get("values", []),
                    metadata=vdata.get("metadata", {}),
                )
            if paginator.pagination and paginator.pagination.next:
                paginator = self.index.list_paginated(
                    namespace=self.namespace, limit=batch_size,
                    pagination_token=paginator.pagination.next,
                )
            else:
                break

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or (stats.get("namespaces", {}) if isinstance(stats, dict) else {})
        ns_stats = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else getattr(namespaces, self.namespace, {})
        vector_count = getattr(ns_stats, "vector_count", None) or (ns_stats.get("vector_count", 0) if isinstance(ns_stats, dict) else 0)
        return int(vector_count)

    def indexed_count(self) -> int:
        # Pinecone doesn't distinguish stored vs. indexed — all stored
        # vectors are indexed. Return the same as count().
        return self.count()
