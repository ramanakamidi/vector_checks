"""
Category 2: Neighbor Consistency checks.

These checks validate whether likely-related chunks appear near each other
in kNN results. Automatic candidates come from consecutive chunks in the same
source and section (or page when sections are unavailable), with matching
content types when available. They remain metadata-based heuristics, not
ground-truth labels.
"""

import random
from typing import Dict, List, Tuple

from connectors.base import VectorDBConnector, VectorRecord


DEFAULT_NEIGHBOR_K = 10


def generate_adjacent_chunk_pairs(
    connector: VectorDBConnector,
    max_pairs: int = 25,
) -> List[Tuple[str, str]]:
    """Generate likely-similar pairs from consecutive chunks of one source.

    This is a metadata-based heuristic, not ground-truth labeling. It only
    creates pairs when source identity and consecutive chunk indexes exist.
    """
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative.")

    chunks_by_source: Dict[str, Dict[int, VectorRecord]] = {}
    source_fields = ("pdf_sha256", "doc_id", "document_id")

    for record in connector.scan(batch_size=500):
        if not record.vector:
            continue

        source = next(
            (
                (field, str(record.metadata[field]))
                for field in source_fields
                if record.metadata.get(field) not in (None, "")
            ),
            None,
        )
        raw_index = record.metadata.get("chunk_index")
        if source is None or isinstance(raw_index, bool):
            continue
        try:
            chunk_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if str(chunk_index) != str(raw_index).strip():
            continue

        source_key = f"{source[0]}:{source[1]}"
        compact_record = VectorRecord(
            id=record.id,
            vector=[],
            metadata=dict(record.metadata or {}),
        )
        chunks_by_source.setdefault(source_key, {}).setdefault(
            chunk_index,
            compact_record,
        )

    candidate_pairs = []
    for source_key in sorted(chunks_by_source):
        chunks = chunks_by_source[source_key]
        for chunk_index in sorted(chunks):
            record = chunks[chunk_index]
            next_record = chunks.get(chunk_index + 1)
            if next_record is not None and _same_chunk_context(record, next_record):
                candidate_pairs.append((record.id, next_record.id))

    if len(candidate_pairs) > max_pairs:
        return random.Random(0).sample(candidate_pairs, max_pairs)
    return candidate_pairs


def _same_chunk_context(record_a: VectorRecord, record_b: VectorRecord) -> bool:
    """Avoid treating section or content-type boundaries as positive examples."""
    metadata_a = record_a.metadata or {}
    metadata_b = record_b.metadata or {}

    type_a = metadata_a.get("chunk_type") or metadata_a.get("object_type")
    type_b = metadata_b.get("chunk_type") or metadata_b.get("object_type")
    if type_a and type_b and str(type_a).casefold() != str(type_b).casefold():
        return False

    section_a = str(metadata_a.get("section_path") or "").strip()
    section_b = str(metadata_b.get("section_path") or "").strip()
    if section_a or section_b:
        if not section_a or not section_b or section_a.casefold() != section_b.casefold():
            return False
        section_name = section_a.split(">")[-1].strip().casefold()
        if section_name in {
            "reference", "references", "bibliography", "works cited", "literature cited"
        }:
            return False

    page_a = metadata_a.get("page_number")
    page_b = metadata_b.get("page_number")
    known_page_a = page_a not in (None, "", -1, "-1")
    known_page_b = page_b not in (None, "", -1, "-1")
    if known_page_a != known_page_b:
        return False
    if known_page_a and known_page_b:
        try:
            if int(page_a) != int(page_b):
                return False
        except (TypeError, ValueError):
            if str(page_a) != str(page_b):
                return False

    return True


def _chunk_details(record: VectorRecord) -> Tuple[Dict, str]:
    metadata = dict(record.metadata or {})
    chunk = metadata.pop("document", "")
    return metadata, chunk if isinstance(chunk, str) else ""


def _pair_display_metadata(metadata: Dict) -> Dict[str, str]:
    display_fields = (
        ("chunk_index", "Chunk"),
        ("page_number", "Page"),
        ("section_path", "Section"),
        ("chunk_type", "Type"),
        ("pdf_name", "Source"),
    )
    return {
        label: str(metadata[field])
        for field, label in display_fields
        if metadata.get(field) not in (None, "")
    }


def _boundary_overlap(text_a: str, text_b: str) -> str:
    """Return the largest exact suffix of A that is also a prefix of B."""
    for length in range(min(len(text_a), len(text_b)), 0, -1):
        if text_a[-length:] == text_b[:length]:
            return text_b[:length]
    return ""


def check_known_similar_pairs(connector: VectorDBConnector,
                               similar_pairs: List[Tuple[str, str]],
                               k: int = DEFAULT_NEIGHBOR_K,
                               field_mapper=None) -> Dict:
    """For each pair of IDs that should be semantically similar, check
    whether each appears in the other's top-k nearest neighbors.

    Args:
        similar_pairs: List of (id_a, id_b) tuples where a and b should
                       be close in embedding space.
        k: How many neighbors to check. Defaults to 10 based on observed
            reciprocal ranks in the current project collection.
        field_mapper: Optional FieldMapper instance. When provided, each
            failure record includes display metadata for both records.

    Returns:
        A result dict with pass/fail and details on which pairs failed.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not similar_pairs:
        return {
            "check": "known_similar_pairs",
            "passed": None,
            "detail": "No suitable adjacent chunk pairs were available to check.",
        }

    collection_count = connector.count()
    if collection_count <= k + 1:
        return {
            "check": "known_similar_pairs",
            "passed": None,
            "detail": {
                "pairs_checked": 0,
                "pairs_sampled": len(similar_pairs),
                "pairs_not_mutual_top_k": 0,
                "k": k,
                "collection_count": collection_count,
                "diagnostic_status": "inconclusive",
                "message": (
                    f"The collection has {collection_count} records, so each query "
                    f"can return at most {max(collection_count - 1, 0)} non-self "
                    f"neighbors. That does not exceed k={k}; this top-k check "
                    "cannot distinguish close pairs from other records."
                ),
                "pair_results": [],
                "pairs_outside_top_k": [],
            },
        }

    failures = []
    unavailable = []
    checked = 0
    pair_results = []

    for id_a, id_b in similar_pairs:
        record_a = connector.fetch(id_a)
        record_b = connector.fetch(id_b)

        if record_a is None or record_b is None:
            missing_id = id_a if record_a is None else id_b
            unavailable_entry = {
                "pair": (id_a, id_b),
                "reason": f"Could not fetch {missing_id} — record not found",
            }
            failures.append(unavailable_entry)
            unavailable.append(unavailable_entry)
            pair_results.append({
                "pair": (id_a, id_b),
                "status": "unavailable",
                "reason": unavailable_entry["reason"],
                "overlap_chars": 0,
            })
            continue

        if not record_a.vector or not record_b.vector:
            unavailable_entry = {
                "pair": (id_a, id_b),
                "reason": "One or both records have no vector to search.",
            }
            failures.append(unavailable_entry)
            unavailable.append(unavailable_entry)
            pair_results.append({
                "pair": (id_a, id_b),
                "status": "unavailable",
                "reason": unavailable_entry["reason"],
                "overlap_chars": 0,
            })
            continue

        checked += 1

        # Include self plus one result beyond k so near-threshold misses can
        # show rank k+1 instead of being reported only as absent.
        query_k = min(k + 2, collection_count)
        neighbors_of_a = connector.knn_search(record_a.vector, k=query_k)
        neighbors_of_b = connector.knn_search(record_b.vector, k=query_k)

        ranks_from_a = {
            neighbor.id: rank
            for rank, neighbor in enumerate(
                (r for r in neighbors_of_a if r.id != id_a), start=1
            )
        }
        ranks_from_b = {
            neighbor.id: rank
            for rank, neighbor in enumerate(
                (r for r in neighbors_of_b if r.id != id_b), start=1
            )
        }

        a_rank = ranks_from_a.get(id_b)
        b_rank = ranks_from_b.get(id_a)
        a_finds_b = a_rank is not None and a_rank <= k
        b_finds_a = b_rank is not None and b_rank <= k
        metadata_a, chunk_a = _chunk_details(record_a)
        metadata_b, chunk_b = _chunk_details(record_b)
        overlap_text = _boundary_overlap(chunk_a, chunk_b)
        pair_status = (
            "mutual" if a_finds_b and b_finds_a
            else "one_way" if a_finds_b or b_finds_a
            else "neither"
        )
        pair_results.append({
            "pair": (id_a, id_b),
            "rank_b_from_a": a_rank,
            "rank_a_from_b": b_rank,
            "a_finds_b": a_finds_b,
            "b_finds_a": b_finds_a,
            "status": pair_status,
            "overlap_chars": len(overlap_text),
            "metadata_a": _pair_display_metadata(metadata_a),
            "metadata_b": _pair_display_metadata(metadata_b),
            "chunk_a": chunk_a,
            "chunk_b": chunk_b,
        })

        if not a_finds_b or not b_finds_a:
            failure_entry = {
                "pair": (id_a, id_b),
                "a_finds_b": a_finds_b,
                "b_finds_a": b_finds_a,
                "rank_b_from_a": a_rank,
                "rank_a_from_b": b_rank,
                "reason": (
                    "One or both chunks were not found in the other chunk's "
                    f"top-{k} non-self neighbors."
                ),
                "metadata_a": metadata_a,
                "metadata_b": metadata_b,
                "chunk_a": chunk_a,
                "chunk_b": chunk_b,
                "overlap_chars": len(overlap_text),
                "overlap_text": overlap_text,
            }
            if field_mapper:
                failure_entry["metadata_a"] = field_mapper.extract_display_metadata(
                    metadata_a)
                failure_entry["metadata_b"] = field_mapper.extract_display_metadata(
                    metadata_b)
            failures.append(failure_entry)

    return {
        "check": "known_similar_pairs",
        "passed": len(failures) == 0 and len(unavailable) == 0,
        "detail": {
            "pairs_checked": checked,
            "pairs_failed": len(failures),
            "pairs_unavailable": len(unavailable),
            "pairs_sampled": len(similar_pairs),
            "k": k,
            "collection_count": collection_count,
            "neighbor_definition": "Top-k results excluding the query record itself.",
            "pair_results": pair_results,
            "failures": failures[:20],  # cap output
            "unavailable_pairs": unavailable[:20],
        },
    }


def check_known_different_pairs(connector: VectorDBConnector,
                                 different_pairs: List[Tuple[str, str]],
                                 k: int = 10,
                                 field_mapper=None) -> Dict:
    """For each pair of IDs that should be semantically different, check
    that they do NOT appear in each other's top-k nearest neighbors.

    Args:
        different_pairs: List of (id_a, id_b) tuples where a and b should
                         be far apart in embedding space.
        k: How many neighbors to check (lower k = more strict).
        field_mapper: Optional FieldMapper instance. When provided, each
            failure record includes display metadata for both records.

    Returns:
        A result dict with pass/fail and details on which pairs failed.
    """
    if not different_pairs:
        return {
            "check": "known_different_pairs",
            "passed": None,
            "detail": "No different pairs provided — skipping. Supply pairs to enable this check.",
        }

    failures = []
    checked = 0

    for id_a, id_b in different_pairs:
        record_a = connector.fetch(id_a)
        record_b = connector.fetch(id_b)

        if record_a is None or record_b is None:
            missing_id = id_a if record_a is None else id_b
            failures.append({
                "pair": (id_a, id_b),
                "reason": f"Could not fetch {missing_id} — record not found",
            })
            continue

        checked += 1

        neighbors_of_a = connector.knn_search(record_a.vector, k=k)
        neighbor_ids_of_a = {r.id for r in neighbors_of_a}

        neighbors_of_b = connector.knn_search(record_b.vector, k=k)
        neighbor_ids_of_b = {r.id for r in neighbors_of_b}

        a_finds_b = id_b in neighbor_ids_of_a
        b_finds_a = id_a in neighbor_ids_of_b

        if a_finds_b or b_finds_a:
            failure_entry = {
                "pair": (id_a, id_b),
                "a_finds_b": a_finds_b,
                "b_finds_a": b_finds_a,
                "reason": "Records that should be far apart appeared in each other's top-k",
            }
            if field_mapper:
                failure_entry["metadata_a"] = field_mapper.extract_display_metadata(
                    record_a.metadata or {})
                failure_entry["metadata_b"] = field_mapper.extract_display_metadata(
                    record_b.metadata or {})
            failures.append(failure_entry)

    return {
        "check": "known_different_pairs",
        "passed": len(failures) == 0,
        "detail": {
            "pairs_checked": checked,
            "pairs_failed": len(failures),
            "k": k,
            "failures": failures[:20],
        },
    }
