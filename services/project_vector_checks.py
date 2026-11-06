"""Run selectable vector-quality checks against one project's Chroma store."""

from __future__ import annotations

from typing import Any

from checks import (
    duplicate_detection,
    index_health,
    metadata_completeness,
    negative_queries,
    neighbor_consistency,
    synthetic_question_retrieval,
    vector_integrity,
)
from connectors.chroma_connector import ChromaConnector


CHECK_CATEGORIES = {
    "vector_integrity",
    "neighbor_consistency",
    "duplicate_detection",
    "metadata_completeness",
    "index_health",
    "negative_query_similarity",
    "synthetic_question_retrieval",
}

DEFAULT_REQUIRED_FIELDS = duplicate_detection.DEFAULT_METADATA_FIELDS


def _connect_project(project_id: str, persist_directory: str) -> ChromaConnector:
    connector = ChromaConnector(
        collection_name=f"{project_id}_knowledge_base",
        persist_directory=persist_directory,
    )
    try:
        connector.connect()
    except Exception as exc:
        import chromadb

        if isinstance(exc, chromadb.errors.NotFoundError):
            raise ValueError(
                "This project has no Chroma collection yet. Upload a PDF in App Under Test Config first."
            ) from exc
        raise
    if connector.count() == 0:
        raise ValueError(
            "The project's Chroma collection is empty. Upload a PDF in App Under Test Config first."
        )
    return connector


def run_project_checks(
    project_id: str,
    persist_directory: str,
    selected_checks: list[str],
    required_fields: list[str] | None = None,
    duplicate_min_similarity: float = duplicate_detection.DEFAULT_MIN_SIMILARITY,
    duplicate_max_similarity: float = duplicate_detection.DEFAULT_MAX_SIMILARITY,
    negative_query_texts: list[str] | None = None,
    negative_query_max_similarity: float = negative_queries.DEFAULT_MAX_SIMILARITY,
    synthetic_question_record_id: str | None = None,
    synthetic_question_texts: list[str] | None = None,
    synthetic_question_top_k: int = 1,
) -> dict[str, Any]:
    unknown_checks = set(selected_checks) - CHECK_CATEGORIES
    if unknown_checks:
        raise ValueError(f"Unknown check category/categories: {', '.join(sorted(unknown_checks))}")
    if not selected_checks:
        raise ValueError("Select at least one vector-check category.")
    if duplicate_min_similarity > duplicate_max_similarity:
        raise ValueError(
            "The minimum cosine similarity cannot exceed the maximum."
        )
    if (
        not 0.0 <= negative_query_max_similarity <= 1.0
    ):
        raise ValueError(
            "The maximum negative-query cosine similarity must be between 0 and 1."
        )

    connector = _connect_project(project_id, persist_directory)
    results: list[dict[str, Any]] = []

    if "vector_integrity" in selected_checks:
        results.extend(vector_integrity.run_all(connector))

    if "neighbor_consistency" in selected_checks:
        pairs = neighbor_consistency.generate_adjacent_chunk_pairs(connector)
        neighbor_result = neighbor_consistency.check_known_similar_pairs(
            connector,
            pairs,
        )
        neighbor_result["passed"] = None
        if isinstance(neighbor_result.get("detail"), dict):
            detail = neighbor_result["detail"]
            detail.pop("pairs_failed", None)
            detail["pairs_not_mutual_top_k"] = sum(
                item.get("status") in {"one_way", "neither"}
                for item in detail.get("pair_results", [])
            )
            ranking_misses = {
                tuple(item.get("pair", ()))
                for item in detail.get("pair_results", [])
                if item.get("status") in {"one_way", "neither"}
            }
            detail["pairs_outside_top_k"] = [
                failure
                for failure in detail.pop("failures", [])
                if tuple(failure.get("pair", ())) in ranking_misses
            ]
            detail["unavailable_pairs"] = detail.pop("unavailable_pairs", [])
            detail.update({
                "pair_source": "automatic adjacent chunks from the same PDF/document",
                "pairs_sampled": len(pairs),
                "methodology_note": (
                    "This is an informational diagnostic, not a pass/fail quality "
                    "score. Candidate pairs are consecutive chunks from the same "
                    "source, filtered to the same section and page when available, "
                    "excluding reference sections and incompatible content types. "
                    "The table shows every sampled pair, including pairs found in "
                    "both directions. Metadata and overlap still cannot prove "
                    "semantic similarity."
                ),
            })
        else:
            neighbor_result["detail"] = {
                "message": neighbor_result["detail"],
                "pair_source": "automatic adjacent chunks from the same PDF/document",
                "pairs_sampled": 0,
                "methodology_note": (
                    "No qualifying adjacent chunks were available. Check that the "
                    "collection includes source IDs, chunk indexes, and usable "
                    "section/page metadata."
                ),
            }
        results.append(neighbor_result)

    if "duplicate_detection" in selected_checks:
        results.append(
            duplicate_detection.find_near_duplicates(
                connector,
                minimum_similarity=duplicate_min_similarity,
                maximum_similarity=duplicate_max_similarity,
                metadata_fields=required_fields or DEFAULT_REQUIRED_FIELDS,
            )
        )

    if "metadata_completeness" in selected_checks:
        results.append(
            metadata_completeness.check_required_fields(
                connector,
                required_fields or DEFAULT_REQUIRED_FIELDS,
            )
        )

    if "index_health" in selected_checks:
        results.append(index_health.check_searchability(connector))

    if "negative_query_similarity" in selected_checks:
        results.append(
            negative_queries.check_negative_queries(
                connector,
                negative_query_texts or [],
                maximum_similarity=negative_query_max_similarity,
            )
        )

    if "synthetic_question_retrieval" in selected_checks:
        results.append(
            synthetic_question_retrieval.check_synthetic_question_retrieval(
                connector,
                synthetic_question_record_id or "",
                synthetic_question_texts or [],
                synthetic_question_top_k,
            )
        )

    summary = {
        "passed": sum(result.get("passed") is True for result in results),
        "failed": sum(result.get("passed") is False for result in results),
        "inconclusive": sum(result.get("passed") is None for result in results),
        "total": len(results),
    }
    return {"results": results, "summary": summary}
