"""Calibrate cosine similarity against user-supplied off-topic queries."""

from __future__ import annotations

import math
from typing import Any

from connectors.chroma_connector import ChromaConnector


MAX_NEGATIVE_QUERIES = 100
MAX_QUERY_LENGTH = 2000
DEFAULT_MAX_SIMILARITY = 0.32


def check_negative_queries(
    connector: ChromaConnector,
    queries: list[str],
    maximum_similarity: float = DEFAULT_MAX_SIMILARITY,
) -> dict[str, Any]:
    """Measure each off-topic query's best cosine match with one batch search."""
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

    collection = connector.collection
    collection_metadata = getattr(collection, "metadata", None) or {}
    metric = collection_metadata.get("hnsw:space", "l2")
    if metric != "cosine":
        raise ValueError(
            "Negative-query similarity requires a Chroma collection using "
            f"cosine distance; this collection uses {metric!r}."
        )

    from services.local_embeddings import embed_texts

    query_vectors = embed_texts(unique_queries)
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

    results = collection.query(
        query_embeddings=validated_query_vectors,
        n_results=1,
        include=["documents", "distances"],
    )
    ids_by_query = results.get("ids") or []
    distances_by_query = results.get("distances") or []
    documents_by_query = results.get("documents") or []

    query_results: list[dict[str, Any]] = []
    for index, query in enumerate(unique_queries):
        ids = ids_by_query[index] if index < len(ids_by_query) else []
        distances = distances_by_query[index] if index < len(distances_by_query) else []
        if not ids or not distances or distances[0] is None:
            raise ValueError(
                f"Chroma returned no nearest result for query {query!r}."
            )

        similarity = 1.0 - float(distances[0])
        if not math.isfinite(similarity):
            raise ValueError(
                f"Chroma returned a non-finite similarity for query {query!r}."
            )
        similarity = max(-1.0, min(1.0, similarity))
        query_result = {
            "query": query,
            "cosine_similarity": round(similarity, 6),
            "above_threshold": similarity > maximum_similarity,
        }
        if query_result["above_threshold"]:
            documents = (
                documents_by_query[index]
                if index < len(documents_by_query)
                else []
            )
            document = documents[0] if documents else None
            if isinstance(document, str) and document:
                query_result["nearest_chunk"] = {
                    "id": str(ids[0]),
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
