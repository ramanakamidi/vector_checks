"""
Elasticsearch connector.

Requires: pip install elasticsearch>=8.0
Assumes an index with a dense_vector field (default name: "embedding")
and a knn-enabled mapping. See Elastic's docs on the dense_vector field
type if the mapping needs adjusting for your index.
"""

from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


class ElasticConnector(VectorDBConnector):
    def __init__(self, hosts: List[str], index: str, vector_field: str = "embedding",
                 api_key: Optional[str] = None, cloud_id: Optional[str] = None):
        self.hosts = hosts
        self.index = index
        self.vector_field = vector_field
        self.api_key = api_key
        self.cloud_id = cloud_id
        self.client = None

    def connect(self) -> None:
        from elasticsearch import Elasticsearch  # imported here so the package
        # doesn't require elasticsearch installed unless this connector is used
        kwargs: Dict[str, Any] = {}
        if self.cloud_id:
            kwargs["cloud_id"] = self.cloud_id
        else:
            kwargs["hosts"] = self.hosts
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self.client = Elasticsearch(**kwargs)

    def _hit_to_record(self, hit: Dict[str, Any]) -> VectorRecord:
        source = dict(hit.get("_source", {}))
        vector = source.pop(self.vector_field, [])
        return VectorRecord(id=hit["_id"], vector=vector, metadata=source)

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        try:
            result = self.client.get(index=self.index, id=record_id)
        except Exception:
            return None
        return self._hit_to_record(result)

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        result = self.client.search(
            index=self.index,
            knn={
                "field": self.vector_field,
                "query_vector": vector,
                "k": k,
                "num_candidates": max(k * 10, 50),
            },
            size=k,
        )
        return [self._hit_to_record(h) for h in result["hits"]["hits"]]

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        from elasticsearch.helpers import scan as es_scan
        for hit in es_scan(self.client, index=self.index, size=batch_size,
                            query={"query": {"match_all": {}}}):
            yield self._hit_to_record(hit)

    def count(self) -> int:
        result = self.client.count(index=self.index)
        return result["count"]

    def indexed_count(self) -> int:
        # Elastic doesn't separately expose a "vectors actually in the HNSW
        # graph" count via a simple API; the closest practical proxy is the
        # stats API's doc count for the index's primary shards, which should
        # equal count() once the index is not mid-refresh. A persistent gap
        # here after a refresh usually means indexing failures are occurring.
        stats = self.client.indices.stats(index=self.index)
        return stats["indices"][self.index]["primaries"]["docs"]["count"]

    def filter_metadata(self, field: str, value: Any, batch_size: int = 500) -> Iterator[VectorRecord]:
        from elasticsearch.helpers import scan as es_scan
        query = {"query": {"term": {field: value}}}
        for hit in es_scan(self.client, index=self.index, size=batch_size, query=query):
            yield self._hit_to_record(hit)
