"""Build a UMAP and HDBSCAN view of a project's persisted vectors."""

from __future__ import annotations

import threading
from typing import Any


_VISUALIZATION_COMPUTE_LOCK = threading.Lock()


def build_project_visualization(
    project_id: str,
    sample_size: int = 300,
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    """Project and cluster a bounded sample of a project's Chroma records."""
    if not 3 <= sample_size <= 10000:
        raise ValueError("sample_size must be between 3 and 10000.")
    if not 2 <= min_cluster_size <= 100:
        raise ValueError("min_cluster_size must be between 2 and 100.")

    from services.project_chroma import open_project_collection

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Vector visualization requires NumPy. "
            "Install the project's Python requirements."
        ) from exc

    collection = open_project_collection(project_id)
    total_records = collection.count()
    data = collection.get(
        limit=sample_size,
        include=["embeddings", "documents", "metadatas"],
    )
    ids = data.get("ids") or []
    embeddings = data.get("embeddings")
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    valid_ids: list[str] = []
    vectors: list[np.ndarray] = []
    texts: list[str] = []
    record_metadatas: list[dict[str, Any]] = []
    dimension: int | None = None
    skipped_records = 0

    for index, record_id in enumerate(ids):
        vector = None if embeddings is None or index >= len(embeddings) else embeddings[index]
        if vector is None:
            skipped_records += 1
            continue

        try:
            vector_array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            skipped_records += 1
            continue
        if (
            vector_array.ndim != 1
            or vector_array.size == 0
            or not np.isfinite(vector_array).all()
            or (dimension is not None and vector_array.size != dimension)
        ):
            skipped_records += 1
            continue

        dimension = vector_array.size
        valid_ids.append(str(record_id))
        vectors.append(vector_array)
        text = documents[index] if index < len(documents) else None
        texts.append(text if isinstance(text, str) else "")
        metadata = metadatas[index] if index < len(metadatas) else None
        record_metadatas.append(dict(metadata) if isinstance(metadata, dict) else {})

    if len(vectors) < 3:
        return {
            "project_id": project_id,
            "collection_name": collection.name,
            "total_records": total_records,
            "sampled_records": len(ids),
            "projected_records": 0,
            "skipped_records": skipped_records,
            "projection": "UMAP",
            "message": "At least 3 valid vectors are required for UMAP visualization.",
            "points": [],
        }

    try:
        import umap
        from sklearn.cluster import HDBSCAN
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires scikit-learn and 'umap-learn'. "
            "Install the project's Python requirements."
        ) from exc

    vector_matrix = np.vstack(vectors)
    with _VISUALIZATION_COMPUTE_LOCK:
        with threadpool_limits(limits=1):
            coordinates = umap.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=min(15, len(vectors) - 1),
                init="random",
                n_epochs=100,
                n_jobs=1,
                random_state=42,
            ).fit_transform(vector_matrix)

            effective_min_cluster_size = min(min_cluster_size, len(vectors))
            clusters = HDBSCAN(
                min_cluster_size=max(2, effective_min_cluster_size),
                min_samples=1,
                metric="euclidean",
                copy=True,
            ).fit_predict(coordinates)

    points = [
        {
            "id": record_id,
            "x": round(float(coordinates[index, 0]), 6),
            "y": round(float(coordinates[index, 1]), 6),
            "cluster": int(clusters[index]),
            "text": texts[index],
            "metadata": record_metadatas[index],
        }
        for index, record_id in enumerate(valid_ids)
    ]

    return {
        "project_id": project_id,
        "collection_name": collection.name,
        "total_records": total_records,
        "sampled_records": len(ids),
        "projected_records": len(points),
        "skipped_records": skipped_records,
        "projection": "UMAP",
        "clustering": "HDBSCAN",
        "clustering_space": "umap_2d",
        "min_cluster_size": max(2, effective_min_cluster_size),
        "points": points,
    }
