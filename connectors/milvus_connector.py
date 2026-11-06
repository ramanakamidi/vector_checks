"""
Milvus connector.

Requires: pip install pymilvus>=2.3
Assumes a Milvus collection already exists with a vector field and
an auto-id or explicit id field.
"""

from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


class MilvusConnector(VectorDBConnector):
    def __init__(self, host: str, port: int, collection_name: str,
                 vector_field: str = "embedding", id_field: str = "id",
                 token: Optional[str] = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.id_field = id_field
        self.token = token
        self.collection = None

    def connect(self) -> None:
        from pymilvus import connections, Collection
        conn_kwargs = {"host": self.host, "port": self.port}
        if self.token:
            conn_kwargs["token"] = self.token
        connections.connect("default", **conn_kwargs)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _hit_to_record(self, hit: Dict[str, Any]) -> VectorRecord:
        metadata = {k: v for k, v in hit.items()
                    if k not in (self.id_field, self.vector_field, "distance", "score")}
        return VectorRecord(
            id=str(hit.get(self.id_field, "")),
            vector=hit.get(self.vector_field, []),
            metadata=metadata,
        )

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        expr = f'{self.id_field} == "{record_id}"'
        # Get all field names for output
        field_names = [f.name for f in self.collection.schema.fields]
        results = self.collection.query(
            expr=expr, output_fields=field_names,
        )
        if not results:
            return None
        return self._hit_to_record(results[0])

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        field_names = [f.name for f in self.collection.schema.fields
                       if f.name != self.vector_field]
        results = self.collection.search(
            data=[vector],
            anns_field=self.vector_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=k,
            output_fields=field_names,
        )
        records = []
        for hits in results:
            for hit in hits:
                entity = {}
                # Extract fields from the hit entity
                for fname in field_names:
                    try:
                        entity[fname] = hit.entity.get(fname)
                    except Exception:
                        pass
                entity[self.id_field] = str(hit.id)
                records.append(self._hit_to_record(entity))
        return records

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Page through all records using query with offset pagination."""
        field_names = [f.name for f in self.collection.schema.fields]
        offset = 0
        while True:
            results = self.collection.query(
                expr=f"{self.id_field} >= 0", output_fields=field_names,
                offset=offset, limit=batch_size,
            )
            if not results:
                break
            for row in results:
                yield self._hit_to_record(row)
            if len(results) < batch_size:
                break
            offset += batch_size

    def count(self) -> int:
        return self.collection.num_entities

    def indexed_count(self) -> int:
        # Milvus reports num_entities as the count of entities that have
        # been inserted AND flushed. After flush, all are indexed.
        # A gap between num_entities before and after flush() indicates
        # un-indexed data.
        from pymilvus import utility
        stats = utility.get_query_segment_info(self.collection_name)
        indexed = sum(seg.num_rows for seg in stats if seg.state.name == "Sealed")
        growing = sum(seg.num_rows for seg in stats if seg.state.name == "Growing")
        return indexed + growing
