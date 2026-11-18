"""Run selectable vector-quality checks against a project's vector store."""

from __future__ import annotations

from typing import Any

from checks import (
    duplicate_detection,
    metadata_completeness,
    negative_queries,
    neighbor_consistency,
    vector_integrity,
)
from connectors.chroma_connector import ChromaConnector


CHECK_CATEGORIES = {
    "vector_integrity",
    "neighbor_consistency",
    "duplicate_detection",
    "metadata_completeness",
    "negative_query_similarity",
}

DEFAULT_REQUIRED_FIELDS = duplicate_detection.DEFAULT_METADATA_FIELDS
QUESTION_RECORDS_FOR_INTEGRITY_CHECKS = 3


def _connect_project(project_id: str, persist_directory: str) -> Any:
    import json
    from pathlib import Path

    project_dir = Path(persist_directory).parent
    try:
        config = json.loads(
            (project_dir / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read project configuration: {exc}") from exc
    vector_config = (
        (config.get("settings") or {}).get("vector_store_config") or {}
    )
    db_type = vector_config.get("db_type", "chroma")
    collection_name = vector_config.get("collection_name") or (
        f"{project_id}_knowledge_base"
    )
    if db_type == "faiss":
        from connectors.faiss_connector import FaissConnector

        connector = FaissConnector(
            collection_name=collection_name,
            persist_directory=vector_config.get("persist_directory") or persist_directory,
        )
    elif db_type == "milvus_lite":
        from connectors.milvus_lite_connector import MilvusLiteConnector

        connector = MilvusLiteConnector(
            collection_name=collection_name,
            persist_directory=vector_config.get("persist_directory") or persist_directory,
        )
    elif db_type == "chroma":
        from services.project_chroma import open_project_collection

        try:
            collection = open_project_collection(
                project_id,
                projects_dir=project_dir.parent,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "This project has no Chroma collection yet. Upload a PDF in "
                "App Under Test Config first."
            ) from exc
        connector = ChromaConnector(
            collection_name=collection.name,
            persist_directory=vector_config.get("persist_directory") or persist_directory,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {db_type!r}.")
    try:
        connector.connect()
    except Exception as exc:
        if db_type == "chroma":
            import chromadb
            if isinstance(exc, chromadb.errors.NotFoundError):
                raise ValueError(
                    f"This project has no {db_type.upper()} collection yet. "
                    "Upload a PDF in App Under Test Config first."
                ) from exc
        raise
    if connector.count() == 0:
        raise ValueError(
            f"The project's {db_type.upper()} collection is empty. "
            "Upload a PDF in App Under Test Config first."
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
    workbook_questions: list[str] | None = None,
    question_retrieval_top_k: int = 5,
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
    question_retrievals = None
    if workbook_questions and set(selected_checks).intersection(
        {"neighbor_consistency", "duplicate_detection"}
    ):
        _, question_retrievals = (
            neighbor_consistency.retrieve_record_pairs_for_questions(
                connector,
                workbook_questions,
                question_retrieval_top_k,
            )
        )
        for question_index, retrieval in enumerate(question_retrievals):
            retrieval["question_index"] = question_index

    question_quality_retrievals = None
    if workbook_questions and set(selected_checks).intersection(
        {"vector_integrity", "metadata_completeness"}
    ):
        if (
            question_retrievals is not None
            and question_retrieval_top_k == QUESTION_RECORDS_FOR_INTEGRITY_CHECKS
        ):
            question_quality_retrievals = question_retrievals
        else:
            _, question_quality_retrievals = (
                neighbor_consistency.retrieve_record_pairs_for_questions(
                    connector,
                    workbook_questions,
                    QUESTION_RECORDS_FOR_INTEGRITY_CHECKS,
                )
            )
        for question_index, retrieval in enumerate(question_quality_retrievals):
            retrieval["question_index"] = question_index

    if "vector_integrity" in selected_checks:
        if question_quality_retrievals is not None:
            results.append(
                vector_integrity.check_retrieved_vectors_per_question(
                    connector,
                    question_quality_retrievals,
                )
            )
        else:
            results.extend(vector_integrity.run_all(connector))

    if "neighbor_consistency" in selected_checks:
        if question_retrievals is not None:
            retrievals_by_question = question_retrievals
            question_pair_results = []
            for retrieval in retrievals_by_question:
                pairs = [
                    tuple(candidate["ids"])
                    for candidate in retrieval["candidate_pairs"]
                ]
                question_result = neighbor_consistency.check_known_similar_pairs(
                    connector,
                    pairs,
                    k=question_retrieval_top_k,
                )
                retrieval["check_result"] = question_result
                question_pair_results.append(question_result)

            pair_results = []
            pairs_outside_top_k = []
            unavailable_pairs = []
            pairs_checked = 0
            for retrieval, question_result in zip(
                retrievals_by_question,
                question_pair_results,
            ):
                question = retrieval["question"]
                detail = question_result.get("detail")
                if not isinstance(detail, dict):
                    continue
                pairs_checked += detail.get("pairs_checked", 0)
                pair_results.extend(
                    {
                        **item,
                        "question_index": retrieval["question_index"],
                        "question": question,
                    }
                    for item in detail.get("pair_results", [])
                )
                pairs_outside_top_k.extend(
                    {
                        **item,
                        "question_index": retrieval["question_index"],
                        "question": question,
                    }
                    for item in detail.get("failures", [])
                )
                unavailable_pairs.extend(
                    {
                        **item,
                        "question_index": retrieval["question_index"],
                        "question": question,
                    }
                    for item in detail.get("unavailable_pairs", [])
                )

            neighbor_result = {
                "check": "known_similar_pairs",
                "passed": None,
                "detail": {
                    "pair_source": (
                        "all distinct retrieved-record pairs, evaluated per workbook question"
                    ),
                    "question_count": len(retrievals_by_question),
                    "question_retrieval_top_k": question_retrieval_top_k,
                    "k": question_retrieval_top_k,
                    "pairs_sampled": sum(
                        len(retrieval["candidate_pairs"])
                        for retrieval in retrievals_by_question
                    ),
                    "pairs_checked": pairs_checked,
                    "pairs_not_mutual_top_k": sum(
                        item.get("status") in {"one_way", "neither"}
                        for item in pair_results
                    ),
                    "pairs_outside_top_k": pairs_outside_top_k,
                    "unavailable_pairs": unavailable_pairs,
                    "pair_results": pair_results,
                    "retrievals_by_question": retrievals_by_question,
                    "methodology_note": (
                        "For each workbook question, retrieve its configured top-k "
                        "records, form every distinct pair among those records, "
                        "then check each pair's reciprocal top-k ranking. "
                        "This is an informational diagnostic, not pass/fail ground "
                        "truth; retrieval proximity alone cannot prove semantic similarity."
                    ),
                },
            }
            pair_source = (
                "all distinct retrieved-record pairs, evaluated per workbook question"
            )
        else:
            pairs = neighbor_consistency.generate_adjacent_chunk_pairs(connector)
            retrievals_by_question = []
            pair_source = "automatic adjacent chunks from the same PDF/document"
            neighbor_result = neighbor_consistency.check_known_similar_pairs(
                connector,
                pairs,
            )
        if not workbook_questions and isinstance(neighbor_result.get("detail"), dict):
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
                "pair_source": pair_source,
                "pairs_sampled": len(pairs),
                "retrievals_by_question": retrievals_by_question,
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
        elif not workbook_questions:
            neighbor_result["detail"] = {
                "message": neighbor_result["detail"],
                "pair_source": pair_source,
                "pairs_sampled": 0,
                "retrievals_by_question": retrievals_by_question,
                "methodology_note": (
                    "No qualifying adjacent chunks were available. Check that the "
                    "collection includes source IDs, chunk indexes, and usable "
                    "section/page metadata."
                ),
            }
        results.append(neighbor_result)

    if "duplicate_detection" in selected_checks:
        if question_retrievals is not None:
            duplicate_result = duplicate_detection.find_near_duplicates_for_questions(
                connector,
                question_retrievals,
                minimum_similarity=duplicate_min_similarity,
                maximum_similarity=duplicate_max_similarity,
                metadata_fields=required_fields or DEFAULT_REQUIRED_FIELDS,
            )
        else:
            duplicate_result = duplicate_detection.find_near_duplicates(
                connector,
                minimum_similarity=duplicate_min_similarity,
                maximum_similarity=duplicate_max_similarity,
                metadata_fields=required_fields or DEFAULT_REQUIRED_FIELDS,
            )
        results.append(duplicate_result)

    if "metadata_completeness" in selected_checks:
        if question_quality_retrievals is not None:
            metadata_result = metadata_completeness.check_retrieved_fields_per_question(
                connector,
                required_fields or DEFAULT_REQUIRED_FIELDS,
                question_quality_retrievals,
            )
        else:
            metadata_result = metadata_completeness.check_required_fields(
                connector,
                required_fields or DEFAULT_REQUIRED_FIELDS,
            )
        results.append(metadata_result)

    if "negative_query_similarity" in selected_checks:
        results.append(
            negative_queries.check_negative_queries(
                connector,
                negative_query_texts or [],
                maximum_similarity=negative_query_max_similarity,
            )
        )

    summary = {
        "passed": sum(result.get("passed") is True for result in results),
        "failed": sum(result.get("passed") is False for result in results),
        "inconclusive": sum(result.get("passed") is None for result in results),
        "total": len(results),
    }
    return {"results": results, "summary": summary}
