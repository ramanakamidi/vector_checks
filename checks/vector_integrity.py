"""
Category 1: vector dimension consistency.

Checks that every stored vector has the same dimensionality, which is
required for meaningful nearest-neighbor comparisons.
"""

import math
from typing import Dict, List

from connectors.base import VectorDBConnector


def check_dimension_consistency(
    connector: VectorDBConnector,
    batch_size: int = 500,
    field_mapper=None,
) -> Dict:
    """Verify every vector in the collection has the same dimensionality."""
    dimensions_seen: Dict[int, int] = {}
    sample_ids: Dict[int, List[str]] = {}
    sample_records: Dict[int, List[Dict]] = {}
    total_scanned = 0

    for record in connector.scan(batch_size=batch_size):
        total_scanned += 1
        dimension = len(record.vector) if record.vector else 0
        dimensions_seen[dimension] = dimensions_seen.get(dimension, 0) + 1
        if dimension not in sample_ids:
            sample_ids[dimension] = []
            sample_records[dimension] = []
        if len(sample_ids[dimension]) < 5:
            sample_ids[dimension].append(record.id)
            metadata = record.metadata or {}
            if field_mapper:
                metadata = field_mapper.extract_display_metadata(metadata)
            sample_records[dimension].append({
                "id": record.id,
                "metadata": metadata,
            })

    if len(dimensions_seen) <= 1:
        return {
            "check": "dimension_consistency",
            "passed": True,
            "detail": {
                "total_scanned": total_scanned,
                "uniform_dimension": (
                    next(iter(dimensions_seen)) if dimensions_seen else 0
                ),
            },
        }

    return {
        "check": "dimension_consistency",
        "passed": False,
        "detail": {
            "total_scanned": total_scanned,
            "dimensions_found": dimensions_seen,
            "sample_ids_per_dimension": sample_ids,
            "sample_records_per_dimension": sample_records,
            "note": (
                "Multiple dimensions found — likely mixed embedding models "
                "or truncated vectors."
            ),
        },
    }


def run_all(
    connector: VectorDBConnector,
    batch_size: int = 500,
    field_mapper=None,
) -> List[Dict]:
    """Run the active vector-integrity check."""
    return [
        check_dimension_consistency(
            connector,
            batch_size=batch_size,
            field_mapper=field_mapper,
        )
    ]


def check_retrieved_vectors_per_question(
    connector: VectorDBConnector,
    retrievals_by_question: List[Dict],
) -> Dict:
    """Check dimensions and basic vector validity for each question's retrieved records."""
    question_results = []

    for question_index, retrieval in enumerate(retrievals_by_question):
        record_ids = list(dict.fromkeys(retrieval.get("retrieved_ids", [])))
        dimensions_seen: Dict[int, int] = {}
        record_results = []
        unavailable_records = []
        issues = []

        for record_id in record_ids:
            record = connector.fetch(record_id)
            if record is None:
                unavailable_records.append(str(record_id))
                continue

            vector = record.vector or []
            dimension = len(vector)
            dimensions_seen[dimension] = dimensions_seen.get(dimension, 0) + 1
            record_issues = []
            if not vector:
                record_issues.append("Vector is empty.")
            else:
                try:
                    vector_values = [float(value) for value in vector]
                except (TypeError, ValueError, OverflowError):
                    record_issues.append("Vector contains a non-numeric value.")
                else:
                    if not all(math.isfinite(value) for value in vector_values):
                        record_issues.append("Vector contains a non-finite value.")
                    elif not any(value != 0.0 for value in vector_values):
                        record_issues.append("Vector has zero magnitude.")

            record_result = {
                "id": str(record_id),
                "dimension": dimension,
                "status": "FAIL" if record_issues else "PASS",
                "issues": record_issues,
            }
            record_results.append(record_result)
            if record_issues:
                issues.extend(
                    {"id": str(record_id), "reason": reason}
                    for reason in record_issues
                )

        if len(dimensions_seen) > 1:
            issues.append({
                "reason": "Retrieved records have inconsistent vector dimensions."
            })

        passed = (
            False if issues
            else None if unavailable_records or not record_ids
            else True
        )
        question_results.append({
            "question_index": retrieval.get("question_index", question_index),
            "question": retrieval.get("question", ""),
            "retrieved_sources": retrieval.get("retrieved_sources", []),
            "records_requested": len(record_ids),
            "records_checked": len(record_results),
            "dimensions_found": dimensions_seen,
            "record_results": record_results,
            "unavailable_records": unavailable_records,
            "issues": issues,
            "passed": passed,
            "status": "PASS" if passed is True else "FAIL" if passed is False else "INFO",
        })

    overall_passed = (
        False if any(item["passed"] is False for item in question_results)
        else None if any(item["passed"] is None for item in question_results)
        else True
    )
    return {
        "check": "dimension_consistency_by_question",
        "passed": overall_passed,
        "detail": {
            "records_per_question": 3,
            "question_count": len(question_results),
            "question_results": question_results,
        },
    }
