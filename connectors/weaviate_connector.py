"""
Weaviate connector.

Requires: pip install weaviate-client>=4.0
Assumes a Weaviate class (collection) already exists.
"""

from typing import Any, Dict, Iterator, List, Optional

from .base import VectorDBConnector, VectorRecord


class WeaviateConnector(VectorDBConnector):
    def __init__(self, url: str, class_name: str,
                 api_key: Optional[str] = None,
                 additional_headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.class_name = class_name
        self.api_key = api_key
        self.additional_headers = additional_headers or {}
        self.client = None

    def connect(self) -> None:
        import weaviate
        auth = None
        if self.api_key:
            auth = weaviate.auth.AuthApiKey(api_key=self.api_key)
        self.client = weaviate.connect_to_custom(
            http_host=self.url,
            auth_credentials=auth,
            additional_headers=self.additional_headers,
        )

    def _object_to_record(self, obj: Any) -> VectorRecord:
        """Convert a Weaviate object to a VectorRecord."""
        props = obj.properties if hasattr(obj, "properties") else {}
        vector = obj.vector.get("default", []) if hasattr(obj, "vector") and obj.vector else []
        obj_id = str(obj.uuid) if hasattr(obj, "uuid") else str(obj.id)
        return VectorRecord(id=obj_id, vector=list(vector), metadata=dict(props))

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        import weaviate.classes.query as wq
        collection = self.client.collections.get(self.class_name)
        try:
            obj = collection.query.fetch_object_by_id(
                record_id, include_vector=True
            )
            if obj is None:
                return None
            return self._object_to_record(obj)
        except Exception:
            return None

    def knn_search(self, vector: List[float], k: int = 10) -> List[VectorRecord]:
        collection = self.client.collections.get(self.class_name)
        result = collection.query.near_vector(
            near_vector=vector, limit=k, include_vector=True,
        )
        return [self._object_to_record(o) for o in result.objects]

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        """Iterate all objects using Weaviate's cursor-based iteration."""
        collection = self.client.collections.get(self.class_name)
        for obj in collection.iterator(include_vector=True):
            yield self._object_to_record(obj)

    def count(self) -> int:
        collection = self.client.collections.get(self.class_name)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count

    def indexed_count(self) -> int:
        # Weaviate indexes all objects automatically; stored == indexed.
        return self.count()

    def filter_metadata(self, field: str, value: Any, batch_size: int = 500) -> Iterator[VectorRecord]:
        import weaviate.classes.query as wq
        collection = self.client.collections.get(self.class_name)
        result = collection.query.fetch_objects(
            filters=wq.Filter.by_property(field).equal(value),
            include_vector=True, limit=10000,
        )
        for obj in result.objects:
            yield self._object_to_record(obj)
