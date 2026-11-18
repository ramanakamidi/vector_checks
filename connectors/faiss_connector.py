"""Persistent, project-local FAISS connector with a Chroma-like collection API."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

import faiss
import numpy as np

from connectors.base import VectorDBConnector, VectorRecord


class FaissConnector(VectorDBConnector):
    """FAISS IndexFlatIP store with cosine-normalized vectors and JSON metadata."""

    def __init__(self, collection_name: str, persist_directory: str):
        self.collection_name = collection_name
        self.name = collection_name
        self.persist_directory = Path(persist_directory)
        self._data_path = self.persist_directory / "records.json"
        self._index_path = self.persist_directory / "index.faiss"
        self._index_signature_path = self.persist_directory / "index.sha256"
        self._records: dict[str, dict[str, Any]] = {}
        self._index: Any = None
        self._dimension: int | None = None
        self._connected = False

    def connect(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._records = {}
        if self._data_path.exists():
            try:
                data = json.loads(self._data_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("FAISS records file has an invalid format.")
                if data.get("collection_name") != self.collection_name:
                    raise ValueError("FAISS data belongs to a different collection.")
                records = data.get("records")
                if not isinstance(records, list):
                    raise ValueError("FAISS records file has an invalid format.")
                loaded_records: dict[str, dict[str, Any]] = {}
                internal_ids: set[int] = set()
                for item in records:
                    if not isinstance(item, dict):
                        raise ValueError("FAISS records file has an invalid record.")
                    record_id = str(item.get("id", ""))
                    internal_id = int(item["internal_id"])
                    metadata = item.get("metadata") or {}
                    if (
                        not record_id
                        or record_id in loaded_records
                        or internal_id < 0
                        or internal_id in internal_ids
                        or not isinstance(metadata, dict)
                    ):
                        raise ValueError("FAISS records file has invalid record values.")
                    loaded_records[record_id] = {
                        "internal_id": internal_id,
                        "document": item.get("document"),
                        "metadata": metadata,
                        "embedding": item["embedding"],
                    }
                    internal_ids.add(internal_id)
                self._records = loaded_records
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                AttributeError,
            ) as exc:
                raise ValueError(f"Could not read FAISS data: {exc}") from exc
        signature = self._record_signature()
        if self._index_path.is_file() and self._index_signature_path.is_file():
            saved_signature = self._index_signature_path.read_text(
                encoding="ascii"
            ).strip()
            if saved_signature == signature:
                try:
                    index = faiss.read_index(str(self._index_path))
                    indexed_ids = sorted(
                        int(value) for value in faiss.vector_to_array(index.id_map)
                    )
                    expected_ids = sorted(
                        record["internal_id"] for record in self._records.values()
                    )
                    expected_dimension = (
                        len(next(iter(self._records.values()))["embedding"])
                        if self._records
                        else None
                    )
                    if (
                        index.ntotal == len(self._records)
                        and indexed_ids == expected_ids
                        and (expected_dimension is None or index.d == expected_dimension)
                    ):
                        self._index = index
                        self._dimension = expected_dimension
                        self._connected = True
                        return
                except (RuntimeError, AttributeError, ValueError):
                    pass
        self._rebuild_index()
        self._write_index()
        self._connected = True

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    @staticmethod
    def _normalize_vector(vector: Any) -> np.ndarray:
        try:
            values = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embeddings must be finite numeric vectors.") from exc
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Embeddings must be non-empty finite numeric vectors.")
        norm = float(np.linalg.norm(values))
        if norm == 0:
            raise ValueError("Embeddings must not be zero vectors.")
        return values / norm

    def _rebuild_index(self) -> None:
        self._index = None
        self._dimension = None
        if not self._records:
            return

        ordered_records = sorted(
            self._records.values(), key=lambda record: record["internal_id"]
        )
        vectors = [self._normalize_vector(record["embedding"]) for record in ordered_records]
        dimensions = {vector.size for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("All vectors in a FAISS collection must have the same dimension.")
        self._dimension = int(vectors[0].size)
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
        self._index.add_with_ids(
            np.vstack(vectors).astype(np.float32, copy=False),
            np.asarray(
                [record["internal_id"] for record in ordered_records],
                dtype=np.int64,
            ),
        )

    def _record_signature(self) -> str:
        vectors = [
            [record["internal_id"], record["embedding"]]
            for record in sorted(
                self._records.values(),
                key=lambda record: record["internal_id"],
            )
        ]
        encoded = json.dumps(vectors, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _write_index(self) -> None:
        if self._index is None:
            for path in (self._index_path, self._index_signature_path):
                if path.exists():
                    path.unlink()
            return

        descriptor, temp_index_path = tempfile.mkstemp(
            dir=self.persist_directory,
            suffix=".faiss.tmp",
        )
        os.close(descriptor)
        temp_signature_path: str | None = None
        try:
            faiss.write_index(self._index, temp_index_path)
            os.replace(temp_index_path, self._index_path)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="ascii",
                dir=self.persist_directory,
                suffix=".tmp",
                delete=False,
            ) as signature_file:
                temp_signature_path = signature_file.name
                signature_file.write(self._record_signature())
                signature_file.flush()
                os.fsync(signature_file.fileno())
            os.replace(temp_signature_path, self._index_signature_path)
        except Exception:
            if os.path.exists(temp_index_path):
                os.unlink(temp_index_path)
            if temp_signature_path and os.path.exists(temp_signature_path):
                os.unlink(temp_signature_path)
            raise

    def _save(self) -> None:
        records = [
            {"id": record_id, **record}
            for record_id, record in sorted(
                self._records.items(),
                key=lambda item: item[1]["internal_id"],
            )
        ]
        payload = {"collection_name": self.collection_name, "records": records}
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.persist_directory,
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                json.dump(payload, temp_file, ensure_ascii=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self._data_path)
            self._write_index()
        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        return not where or all(metadata.get(key) == value for key, value in where.items())

    def _filtered_ids(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        requested = set(ids) if ids is not None else None
        return [
            record_id
            for record_id, record in sorted(
                self._records.items(),
                key=lambda item: item[1]["internal_id"],
            )
            if (requested is None or record_id in requested)
            and self._metadata_matches(record["metadata"], where)
        ]

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
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be unique within an upsert batch.")
        if any(not isinstance(record_id, str) or not record_id for record_id in ids):
            raise ValueError("Record IDs must be non-empty strings.")
        if documents is not None and len(ids) != len(documents):
            raise ValueError("Document count must match ID count.")
        if metadatas is not None and len(ids) != len(metadatas):
            raise ValueError("Metadata count must match ID count.")

        normalized_vectors = [self._normalize_vector(vector) for vector in embeddings]
        dimensions = {vector.size for vector in normalized_vectors}
        if self._dimension is not None:
            dimensions.add(self._dimension)
        if len(dimensions) != 1:
            raise ValueError("All vectors in a FAISS collection must have the same dimension.")

        next_internal_id = max(
            (record["internal_id"] for record in self._records.values()),
            default=-1,
        ) + 1
        for index, record_id in enumerate(ids):
            vector = normalized_vectors[index]
            existing = self._records.get(str(record_id))
            self._records[str(record_id)] = {
                "internal_id": (
                    existing["internal_id"] if existing else next_internal_id
                ),
                "document": documents[index] if documents is not None else (
                    existing.get("document") if existing else None
                ),
                "metadata": metadatas[index] if metadatas is not None else (
                    existing.get("metadata", {}) if existing else {}
                ),
                "embedding": vector.tolist(),
            }
            if existing is None:
                next_internal_id += 1
        self._rebuild_index()
        self._save()

    def update(
        self,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: Any = None,
    ) -> None:
        self._ensure_connected()
        values = [value for value in (documents, metadatas, embeddings) if value is not None]
        if any(len(value) != len(ids) for value in values):
            raise ValueError("Updated values must match the ID count.")
        if len(set(ids)) != len(ids):
            raise ValueError("Record IDs must be unique within an update batch.")
        if any(not isinstance(record_id, str) or not record_id for record_id in ids):
            raise ValueError("Record IDs must be non-empty strings.")
        normalized_updates = (
            [self._normalize_vector(vector) for vector in embeddings]
            if embeddings is not None
            else None
        )
        if normalized_updates is not None:
            target_vectors = {
                str(record_id): normalized_updates[index]
                for index, record_id in enumerate(ids)
                if str(record_id) in self._records
            }
            dimensions = {
                target_vectors.get(record_id, self._normalize_vector(record["embedding"])).size
                for record_id, record in self._records.items()
            }
            if len(dimensions) > 1:
                raise ValueError(
                    "All vectors in a FAISS collection must have the same dimension."
                )
        for index, record_id in enumerate(ids):
            record = self._records.get(str(record_id))
            if record is None:
                continue
            if documents is not None:
                record["document"] = documents[index]
            if metadatas is not None:
                record["metadata"] = metadatas[index]
            if normalized_updates is not None:
                record["embedding"] = normalized_updates[index].tolist()
        self._rebuild_index()
        self._save()

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_connected()
        for record_id in self._filtered_ids(ids, where):
            self._records.pop(record_id, None)
        self._rebuild_index()
        self._save()

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
        selected_ids = self._filtered_ids(ids, where)[offset:]
        if limit is not None:
            selected_ids = selected_ids[:limit]
        result: dict[str, Any] = {"ids": selected_ids}
        include = include or []
        if "documents" in include:
            result["documents"] = [
                self._records[record_id]["document"] for record_id in selected_ids
            ]
        if "metadatas" in include:
            result["metadatas"] = [
                self._records[record_id]["metadata"] for record_id in selected_ids
            ]
        if "embeddings" in include:
            result["embeddings"] = [
                self._records[record_id]["embedding"] for record_id in selected_ids
            ]
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
        if self._index is None or n_results < 1:
            neighbors: list[list[tuple[str, float]]] = [[] for _ in query_embeddings]
        else:
            queries = np.vstack(
                [self._normalize_vector(vector) for vector in query_embeddings]
            ).astype(np.float32, copy=False)
            if queries.shape[1] != self._dimension:
                raise ValueError(
                    f"Query embedding dimension {queries.shape[1]} does not match "
                    f"collection dimension {self._dimension}."
                )
            scores, labels = self._index.search(
                queries, min(n_results, self._index.ntotal)
            )
            internal_to_record = {
                record["internal_id"]: record_id
                for record_id, record in self._records.items()
            }
            neighbors = [
                [
                    (internal_to_record[int(label)], float(score))
                    for score, label in zip(row_scores, row_labels)
                    if int(label) in internal_to_record
                ]
                for row_scores, row_labels in zip(scores, labels)
            ]

        result: dict[str, Any] = {
            "ids": [[record_id for record_id, _ in row] for row in neighbors]
        }
        if "documents" in include:
            result["documents"] = [
                [self._records[record_id]["document"] for record_id, _ in row]
                for row in neighbors
            ]
        if "metadatas" in include:
            result["metadatas"] = [
                [self._records[record_id]["metadata"] for record_id, _ in row]
                for row in neighbors
            ]
        if "embeddings" in include:
            result["embeddings"] = [
                [self._records[record_id]["embedding"] for record_id, _ in row]
                for row in neighbors
            ]
        if "distances" in include:
            result["distances"] = [
                [1.0 - score for _, score in row] for row in neighbors
            ]
        return result

    def fetch(self, record_id: str) -> Optional[VectorRecord]:
        result = self.get(ids=[record_id], include=["embeddings", "metadatas", "documents"])
        if not result["ids"]:
            return None
        metadata = dict(result["metadatas"][0] or {})
        if result["documents"][0]:
            metadata["document"] = result["documents"][0]
        return VectorRecord(result["ids"][0], result["embeddings"][0], metadata)

    def knn_search(self, vector: list[float], k: int = 10) -> list[VectorRecord]:
        result = self.query(
            query_embeddings=[vector],
            n_results=k,
            include=["embeddings", "metadatas", "documents"],
        )
        records = []
        for index, record_id in enumerate(result["ids"][0]):
            metadata = dict(result["metadatas"][0][index] or {})
            if result["documents"][0][index]:
                metadata["document"] = result["documents"][0][index]
            records.append(
                VectorRecord(
                    record_id,
                    result["embeddings"][0][index],
                    metadata,
                )
            )
        return records

    def scan(self, batch_size: int = 500) -> Iterator[VectorRecord]:
        self._ensure_connected()
        ids = self._filtered_ids()
        for offset in range(0, len(ids), batch_size):
            data = self.get(
                ids=ids[offset : offset + batch_size],
                include=["embeddings", "metadatas"],
            )
            for index, record_id in enumerate(data["ids"]):
                yield VectorRecord(
                    record_id,
                    data["embeddings"][index],
                    data["metadatas"][index],
                )

    def count(self) -> int:
        self._ensure_connected()
        return len(self._records)

    def indexed_count(self) -> int:
        self._ensure_connected()
        return int(self._index.ntotal) if self._index is not None else 0

    def filter_metadata(
        self, field: str, value: Any, batch_size: int = 500
    ) -> Iterator[VectorRecord]:
        self._ensure_connected()
        ids = self._filtered_ids(where={field: value})
        for offset in range(0, len(ids), batch_size):
            data = self.get(
                ids=ids[offset : offset + batch_size],
                include=["embeddings", "metadatas"],
            )
            for index, record_id in enumerate(data["ids"]):
                yield VectorRecord(
                    record_id,
                    data["embeddings"][index],
                    data["metadatas"][index],
                )
