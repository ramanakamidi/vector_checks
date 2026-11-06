"""Index Health check: verify every stored record is searchable by kNN."""

from typing import Dict, List

from connectors.base import VectorDBConnector


def check_searchability(
    connector: VectorDBConnector,
    k: int = 1,
) -> Dict:
    """Query every record's vector and verify the same record is returned.

    Records without vectors are counted as failures because they cannot be
    queried, rather than silently omitted from the check.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    failures: List[Dict] = []
    records_checked = 0
    vectors_queried = 0

    for record in connector.scan(batch_size=500):
        records_checked += 1
        if not record.vector:
            failures.append({
                "id": record.id,
                "expected": "non-empty vector searchable in kNN",
                "got_ids": [],
                "reason": "Record has no vector; search could not be run.",
            })
            continue

        vectors_queried += 1
        neighbors = connector.knn_search(record.vector, k=k)
        neighbor_ids = {neighbor.id for neighbor in neighbors}
        if record.id not in neighbor_ids:
            failures.append({
                "id": record.id,
                "expected": f"self in top-{k} results",
                "got_ids": [neighbor.id for neighbor in neighbors],
                "reason": "Self was absent from the returned nearest-neighbor results.",
            })

    if records_checked == 0:
        return {
            "check": "searchability_spot_check",
            "passed": None,
            "detail": {
                "records_checked": 0,
                "vectors_queried": 0,
                "failures": 0,
                "failure_details": [],
                "note": (
                    "The collection contained no records to test. "
                    "This full scan performs one nearest-neighbor query per "
                    "record with a vector."
                ),
            },
        }

    return {
        "check": "searchability_spot_check",
        "passed": len(failures) == 0,
        "detail": {
            "records_checked": records_checked,
            "vectors_queried": vectors_queried,
            "failures": len(failures),
            "failure_details": failures[:20],
            "failure_details_truncated": len(failures) > 20,
            "note": (
                "Every stored record was checked; all queried vectors found "
                "themselves in kNN results. This performs one nearest-neighbor "
                "query per record with a vector."
                if not failures
                else (
                    f"{len(failures)} records were not successfully searchable. "
                    "This full scan performs one nearest-neighbor query per "
                    "record with a vector."
                )
            ),
        },
    }
