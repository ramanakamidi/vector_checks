"""
Category 4: Metadata Completeness checks.

Scans every record and checks whether required metadata fields are
present and non-empty. This is pure counting — no vector math involved.

Common required fields might include:
  - doc_id (which source document this chunk came from)
  - artifact_type (what kind of document it is)
  - chunk_index (position within the source document)
  - source_url or file_path (provenance)

The specific required fields depend on your schema — pass them in
as a list.
"""

from typing import Any, Dict, List

from connectors.base import VectorDBConnector


def check_required_fields(connector: VectorDBConnector,
                           required_fields: List[str],
                           batch_size: int = 500,
                           field_mapper=None) -> Dict:
    """Scan all records and report which required metadata fields are
    missing or empty.

    Args:
        required_fields: List of metadata field names that every record
                         should have.
        field_mapper: Optional FieldMapper instance. When provided, each
            sample record includes display metadata alongside the ID.

    Returns:
        A result dict with per-field missing counts and sample IDs.
    """
    field_missing_counts: Dict[str, int] = {f: 0 for f in required_fields}
    field_empty_counts: Dict[str, int] = {f: 0 for f in required_fields}
    sample_missing_ids: Dict[str, List[str]] = {f: [] for f in required_fields}
    sample_missing_records: Dict[str, List[Dict]] = {f: [] for f in required_fields}
    total_scanned = 0

    for record in connector.scan(batch_size=batch_size):
        total_scanned += 1
        for fld in required_fields:
            is_missing = fld not in record.metadata
            is_empty = (not is_missing) and _is_empty(record.metadata[fld])
            if is_missing:
                field_missing_counts[fld] += 1
            elif is_empty:
                field_empty_counts[fld] += 1
            else:
                continue  # field is present and non-empty

            if len(sample_missing_ids[fld]) < 10:
                sample_missing_ids[fld].append(record.id)
                meta = (field_mapper.extract_display_metadata(record.metadata or {})
                        if field_mapper else {})
                sample_missing_records[fld].append({
                    "id": record.id, "metadata": meta
                })

    total_issues = sum(field_missing_counts.values()) + sum(field_empty_counts.values())

    field_details = {}
    for fld in required_fields:
        missing = field_missing_counts[fld]
        empty = field_empty_counts[fld]
        field_details[fld] = {
            "missing": missing,
            "empty": empty,
            "total_incomplete": missing + empty,
            "completeness_pct": round(
                (1 - (missing + empty) / max(total_scanned, 1)) * 100, 2
            ),
            # Backward-compatible: plain ID list
            "sample_ids": sample_missing_ids[fld],
            # Enriched: ID + metadata dicts
            "sample_records": sample_missing_records[fld],
        }

    return {
        "check": "metadata_completeness",
        "passed": total_issues == 0,
        "detail": {
            "total_scanned": total_scanned,
            "required_fields": required_fields,
            "total_issues": total_issues,
            "per_field": field_details,
        },
    }


def check_metadata_value_distribution(connector: VectorDBConnector,
                                       field: str,
                                       batch_size: int = 500) -> Dict:
    """Report the distribution of values for a single metadata field.
    Useful for spotting unexpected values, skewed distributions, or
    fields that are technically present but always the same value."""
    value_counts: Dict[Any, int] = {}
    missing_count = 0
    total_scanned = 0

    for record in connector.scan(batch_size=batch_size):
        total_scanned += 1
        if field not in record.metadata:
            missing_count += 1
            continue
        val = record.metadata[field]
        val_key = str(val) if not isinstance(val, (str, int, float, bool)) else val
        value_counts[val_key] = value_counts.get(val_key, 0) + 1

    # Sort by count descending, cap at top 50
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "check": "metadata_value_distribution",
        "passed": None,  # informational — no pass/fail
        "detail": {
            "field": field,
            "total_scanned": total_scanned,
            "missing_count": missing_count,
            "unique_values": len(value_counts),
            "top_values": dict(sorted_values[:50]),
        },
    }


def _is_empty(value: Any) -> bool:
    """Check if a metadata value is effectively empty."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False
