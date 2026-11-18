"""Calibrate cosine similarity against user-supplied off-topic queries."""

from __future__ import annotations

import math
from typing import Any

from connectors.base import VectorDBConnector


MAX_NEGATIVE_QUERIES = 100
MAX_QUERY_LENGTH = 2000
DEFAULT_MAX_SIMILARITY = 0.32


def check_negative_queries(
    connector: VectorDBConnector,
    queries: list[str],
    maximum_similarity: float = DEFAULT_MAX_SIMILARITY,
) -> dict[str, Any]:
    """Measure each off-topic query's best cosine match."""
    if not 0.0 <= maximum_similarity <= 1.0:
        raise ValueError("maximum_similarity must be between 0 and 1.")
    if not queries:
        raise ValueError("Enter at least one off-topic query.")
    if len(queries) > MAX_NEGATIVE_QUERIES:
        raise ValueError(
            f"Enter no more than {MAX_NEGATIVE_QUERIES} off-topic queries."
        )

    unique_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Off-topic queries cannot be empty.")
        normalized_query = query.strip()
        if len(normalized_query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Each off-topic query must be at most {MAX_QUERY_LENGTH} characters."
            )
        dedupe_key = " ".join(normalized_query.casefold().split())
        if dedupe_key not in seen_queries:
            seen_queries.add(dedupe_key)
            unique_queries.append(normalized_query)

    collection = getattr(connector, "collection", None)
    if collection is not None:
        collection_metadata = getattr(collection, "metadata", None) or {}
        metric = collection_metadata.get("hnsw:space", "l2")
        if metric != "cosine":
            raise ValueError(
                "Negative-query similarity requires a Chroma collection using "
                f"cosine distance; this collection uses {metric!r}."
            )

    from services.local_embeddings import embed_text_query

    query_vectors = [embed_text_query(query) for query in unique_queries]
    if len(query_vectors) != len(unique_queries):
        raise ValueError("The embedding model did not return a vector per query.")
    validated_query_vectors: list[list[float]] = []
    for query, vector in zip(unique_queries, query_vectors):
        try:
            vector_values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Could not create a valid embedding for query {query!r}."
            ) from exc
        if not vector_values:
            raise ValueError(f"Could not create an embedding for query {query!r}.")
        if not all(math.isfinite(value) for value in vector_values):
            raise ValueError(f"Query {query!r} produced a non-finite embedding.")
        if not any(value != 0.0 for value in vector_values):
            raise ValueError(f"Query {query!r} produced a zero embedding.")
        validated_query_vectors.append(vector_values)

    query_results: list[dict[str, Any]] = []
    for index, query in enumerate(unique_queries):
        records = connector.knn_search(validated_query_vectors[index], k=1)
        if not records:
            raise ValueError(
                f"The vector store returned no nearest result for query {query!r}."
            )

        nearest = records[0]
        try:
            nearest_vector = [float(value) for value in nearest.vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"The nearest vector for query {query!r} is invalid."
            ) from exc
        if len(nearest_vector) != len(validated_query_vectors[index]):
            raise ValueError(
                f"The nearest vector for query {query!r} has a different dimension."
            )
        if not nearest_vector or not all(math.isfinite(value) for value in nearest_vector):
            raise ValueError(
                f"The nearest vector for query {query!r} is empty or non-finite."
            )

        query_vector = validated_query_vectors[index]
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        nearest_norm = math.sqrt(sum(value * value for value in nearest_vector))
        if query_norm == 0.0 or nearest_norm == 0.0:
            raise ValueError(
                f"Could not calculate cosine similarity for query {query!r}."
            )
        similarity = sum(
            query_value * nearest_value
            for query_value, nearest_value in zip(query_vector, nearest_vector)
        ) / (query_norm * nearest_norm)
        if not math.isfinite(similarity):
            raise ValueError(
                f"The vector store returned a non-finite similarity for query {query!r}."
            )
        similarity = max(-1.0, min(1.0, similarity))
        query_result = {
            "query": query,
            "cosine_similarity": round(similarity, 6),
            "above_threshold": similarity > maximum_similarity,
        }
        if query_result["above_threshold"]:
            metadata = nearest.metadata or {}
            document = metadata.get("document")
            if not isinstance(document, str) or not document:
                record = connector.fetch(nearest.id)
                document = (
                    (record.metadata or {}).get("document")
                    if record is not None
                    else None
                )
            if isinstance(document, str) and document:
                query_result["nearest_chunk"] = {
                    "id": str(nearest.id),
                    "text": document,
                }
        query_results.append(query_result)

    maximum_negative_similarity = max(
        item["cosine_similarity"] for item in query_results
    )
    above_threshold_count = sum(
        item["above_threshold"] for item in query_results
    )

    return {
        "check": "negative_query_similarity",
        "passed": above_threshold_count == 0,
        "detail": {
            "queries_tested": len(query_results),
            "maximum_similarity_threshold": maximum_similarity,
            "above_threshold_count": above_threshold_count,
            "maximum_negative_similarity": maximum_negative_similarity,
            "query_results": query_results,
        },
    }
