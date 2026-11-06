"""
Category 1: vector dimension consistency.

Checks that every stored vector has the same dimensionality, which is
required for meaningful nearest-neighbor comparisons.
"""

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
