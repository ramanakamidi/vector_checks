"""Project-local Milvus Lite connector with a Chroma-compatible collection API."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from connectors.base import VectorDBConnector, VectorRecord


_CLIENTS: dict[str, tuple[Any, Any, Any]] = {}
_CLIENTS_LOCK = threading.RLock()


def close_database(database_path: str | Path) -> None:
    """Release this connector's client reference."""
    key = str(Path(database_path).resolve())
    with _CLIENTS_LOCK:
        cached = _CLIENTS.pop(key, None)
    if cached is not None:
        cached[0].close()


def is_database_open(database_path: str | Path) -> bool:
    """Whether this process currently owns a client for the database."""
    key = str(Path(database_path).resolve())
    with _CLIENTS_LOCK:
        return key in _CLIENTS


class MilvusLiteConnector(VectorDBConnector):
    """Persist vectors in a local Milvus Lite database path."""

    def __init__(self, collection_name: str, persist_directory: str):
        self.collection_name = collection_name
        self.name = collection_name
        self.database_path = Path(persist_directory)
        self.client: Any = None
        self._data_type: Any = None
        self._load_state: Any = None
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        if os.name == "nt":
            # Torch must initialize before Milvus Lite's native DLLs on Windows.
            from pdf_workbench.custom_logic import ensure_torch_available

            ensure_torch_available()
        try:
            from pymilvus import DataType, MilvusClient
            from pymilvus.client.types import LoadState
        except ImportError as exc:
            raise ImportError(
                "Milvus Lite requires pymilvus and milvus-lite. "
                "Install them with: python -m pip install "
                "\"pymilvus[milvus-lite]\""
            ) from exc

        self.database_path = self.database_path.resolve()
        key = str(self.database_path)
        with _CLIENTS_LOCK:
            cached = _CLIENTS.get(key)
            if cached is None:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                client = MilvusClient(uri=str(self.database_path))
                cached = (client, DataType, LoadState)
                _CLIENTS[key] = cached
        self.client, self._data_type, self._load_state = cached
        self._connected = True

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def _ensure_collection_loaded(self) -> None:
        state = self.client.get_load_state(self.collection_name).get("state")
        if state != self._load_state.Loaded:
            self.client.load_collection(self.collection_name)

    def _has_collection(self) -> bool:
        self._ensure_connected()
        return bool(self.client.has_collection(self.collection_name))

    def _create_collection(self, dimension: int) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=self._data_type.VARCHAR,
            is_primary=True,
            max_length=512,
        )
        schema.add_field(
            field_name="embedding",
            datatype=self._data_type.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            field_name="document",
            datatype=self._data_type.VARCHAR,
            max_length=65535,
        )
        schema.add_field(field_name="metadata", datatype=self._data_type.JSON)

        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
        )

    @staticmethod
    def _empty(include: list[str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"ids": []}
        for field in include or []:
            if field in {"documents", "metadatas", "embeddings", "distances"}:
                result[field] = []
        return result

    @staticmethod
    def _filter_expression(where: dict[str, Any] | None) -> str:
        if not where:
            return ""
        expressions = []
        for key, value in where.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(
                    f"Unsupported Milvus Lite metadata filter value for {key!r}."
                )
            expressions.append(
                f"metadata[{json.dumps(str(key))}] == "
                f"{json.dumps(value, ensure_ascii=False)}"
            )
        return " and ".join(expressions)

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> VectorRecord:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        document = row.get("document")
        if isinstance(document, str):
            metadata["document"] = document
        vector = row.get("embedding") or []
        return VectorRecord(
            id=str(row.get("id", "")),
            vector=[float(value) for value in vector],
            metadata=metadata,
        )

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: Any,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ensure_connected()
        if embeddings is None or len(ids) != len(embeddings):
            raise ValueError("Each record must have exactly one embedding.")
        if documents is not None and len(ids) != len(documents):
            raise ValueError("Document count must match ID count.")
        if metadatas is not None and len(ids) != len(metadatas):
            raise ValueError("Metadata count must match ID count.")
        if any(not isinstance(record_id, str) or not record_id for record_id in ids):
            raise ValueError("Record IDs must be non-empty strings.")
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be unique within an upsert batch.")

        vectors = [[float(value) for value in vector] for vector in embeddings]
        dimensions = {len(vector) for vector in vectors}
        if not vectors or any(not vector for vector in vectors) or len(dimensions) != 1:
            raise ValueError(
                "Embeddings must be non-empty vectors of one consistent dimension."
            )
        if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise ValueError("Embeddings must contain only finite numbers.")

        if not self._has_collection():
            self._create_collection(len(vectors[0]))

        rows = []
        for index, record_id in enumerate(ids):
            rows.append({
                "id": record_id,
                "embedding": vectors[index],
                "document": documents[index] if documents is not None else "",
                "metadata": metadatas[index] if metadatas is not None else {},
            })
        self.client.upsert(collection_name=self.collection_name, data=rows)

    def update(
        self,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: Any = None,
    ) -> None:
        values = [
            value for value in (documents, metadatas, embeddings)
            if value is not None
        ]
        if any(len(value) != len(ids) for value in values):
            raise ValueError("Updated values must match the ID count.")
        records = self.get(
            ids=ids,
            include=["embeddings", "documents", "metadatas"],
        )
        rows = {
            record_id: {
                "id": record_id,
                "embedding": records["embeddings"][index],
                "document": records["documents"][index] or "",
                "metadata": records["metadatas"][index] or {},
            }
            for index, record_id in enumerate(records["ids"])
        }
        for index, record_id in enumerate(ids):
            row = rows.get(record_id)
            if row is None:
                continue
            if documents is not None:
                row["document"] = documents[index]
            if metadatas is not None:
                row["metadata"] = metadatas[index]
            if embeddings is not None:
                row["embedding"] = [float(value) for value in embeddings[index]]
        if rows:
            self.client.upsert(
                collection_name=self.collection_name,
                data=list(rows.values()),
            )

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        if not self._has_collection():
            return
        if ids is None and where:
            self.client.delete(
                collection_name=self.collection_name,
                filter=self._filter_expression(where),
            )
        elif ids and where:
            self.client.delete(
                collection_name=self.collection_name,
                filter=(
                    f"id in {json.dumps(ids, ensure_ascii=False)} and "
                    f"({self._filter_expression(where)})"
                ),
            )
        elif ids:
            self.client.delete(collection_name=self.collection_name, ids=ids)

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        include = include or []
        if not self._has_collection():
            return self._empty(include)
        if ids is not None and not ids:
            return self._empty(include)
        expressions = []
        if ids is not None:
            expressions.append(f"id in {json.dumps(ids, ensure_ascii=False)}")
        if where:
            expressions.append(self._filter_expression(where))
        filter_expression = " and ".join(expressions)
        total = self.count()
        query_limit = min(limit if limit is not None else total, 16384)
        if query_limit <= 0 or offset >= total:
            return self._empty(include)
        self._ensure_collection_loaded()
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expression,
            output_fields=["id", "embedding", "document", "metadata"],
            limit=query_limit,
            offset=offset,
        )
        by_id = {str(row["id"]): row for row in rows}
        ordered_ids = (
            [record_id for record_id in ids if record_id in by_id]
            if ids is not None
            else [str(row["id"]) for row in rows]
        )
        selected = [by_id[record_id] for record_id in ordered_ids]
        result: dict[str, Any] = {"ids": ordered_ids}
        if "documents" in include:
            result["documents"] = [row.get("document") for row in selected]
        if "metadatas" in include:
            result["metadatas"] = [row.get("metadata") or {} for row in selected]
        if "embeddings" in include:
            result["embeddings"] = [row.get("embedding") or [] for row in selected]
        return result

    def query(
        self,
        *,
        query_embeddings: Any,
        n_results: int = 10,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        include = include or []
        if query_embeddings is None or len(query_embeddings) == 0:
            raise ValueError("At least one query embedding is required.")
        if not self._has_collection() or self.count() == 0 or n_results < 1:
            return {
                "ids": [[] for _ in query_embeddings],
                **{
                    field: [[] for _ in query_embeddings]
                    for field in include
                    if field in {"documents", "metadatas", "embeddings", "distances"}
                },
            }

        self._ensure_collection_loaded()
        output_fields = ["document", "metadata"]
        if "embeddings" in include:
            output_fields.append("embedding")
        hits_by_query = self.client.search(
            collection_name=self.collection_name,
            data=[[float(value) for value in vector] for vector in query_embeddings],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {}},
            limit=min(n_results, self.count()),
            output_fields=output_fields,
        )
        result: dict[str, Any] = {"ids": []}
        for field in ("documents", "metadatas", "embeddings", "distances"):
            if field in include:
                result[field] = []
        for hits in hits_by_query:
            result["ids"].append([str(hit["id"]) for hit in hits])
            entities = [hit.get("entity") or {} for hit in hits]
            if "documents" in include:
                result["documents"].append([entity.get("document") for entity in entities])
            if "metadatas" in include:
                result["metadatas"].append([
                    entity.get("metadata") or {} for entity in entities
                ])
            if "embeddings" in include:
                result["embeddings"].append([
                    entity.get("embedding") or [] for entity in entities
                ])
            if "distances" in include:
                result["distances"].append([
                    1.0 - float(hit.get("distance", 0.0)) for hit in hits
                ])
        return result

    def count(self) -> int:
        self._ensure_connected()
        if not self._has_collection():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def fetch(self, record_id: str) -> VectorRecord | None:
        result = self.get(
            ids=[record_id],
            include=["embeddings", "metadatas", "documents"],
        )
        if not result["ids"]:
            return None
        return self._row_to_record({
            "id": result["ids"][0],
            "embedding": result["embeddings"][0],
            "metadata": result["metadatas"][0],
            "document": result["documents"][0],
        })

    def knn_search(self, vector: list[float], k: int = 10) -> list[VectorRecord]:
        result = self.query(
            query_embeddings=[vector],
            n_results=k,
            include=["embeddings", "metadatas", "documents"],
        )
        return [
            self._row_to_record({
                "id": record_id,
                "embedding": result["embeddings"][0][index],
                "metadata": result["metadatas"][0][index],
                "document": result["documents"][0][index],
            })
            for index, record_id in enumerate(result["ids"][0])
        ]

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        total = self.count()
        for offset in range(0, total, batch_size):
            result = self.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=["embeddings", "metadatas", "documents"],
            )
            for index, record_id in enumerate(result["ids"]):
                yield self._row_to_record({
                    "id": record_id,
                    "embedding": result["embeddings"][index],
                    "metadata": result["metadatas"][index],
                    "document": result["documents"][index],
                })

    def indexed_count(self) -> int:
        return self.count()

    def filter_metadata(
        self,
        field: str,
        value: Any,
        batch_size: int = 500,
    ) -> Iterator[VectorRecord]:
        offset = 0
        while True:
            records = self.get(
                where={field: value},
                limit=batch_size,
                offset=offset,
                include=["embeddings", "metadatas", "documents"],
            )
            if not records["ids"]:
                break
            for index, record_id in enumerate(records["ids"]):
                yield self._row_to_record({
                    "id": record_id,
                    "embedding": records["embeddings"][index],
                    "metadata": records["metadatas"][index],
                    "document": records["documents"][index],
                })
            offset += len(records["ids"])
