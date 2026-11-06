"""
Category 3: Duplicate Detection.

Uses the database's own kNN search to find high-similarity vector candidates.
Pairs are reported when their cosine similarity is within an inclusive range
(0.65–0.98 by default). This can identify:
  - Exact duplicates (same content indexed twice)
  - Near-duplicates (trivially different content, e.g. whitespace changes,
    that produce almost-identical embeddings)

This is a sampling-based check — on large collections, scanning every
vector and running a kNN query for each would be prohibitively expensive.
The sample_size parameter controls how many vectors to spot-check.
"""

import math
import re
from typing import Dict, List

from connectors.base import VectorDBConnector

MIN_SIMILARITY_THRESHOLD = 0.65
MAX_SIMILARITY_THRESHOLD = 0.98
DEFAULT_MIN_SIMILARITY = 0.65
DEFAULT_MAX_SIMILARITY = 0.98
DEFAULT_METADATA_FIELDS = [
    "pdf_name",
    "pdf_sha256",
    "page_number",
    "chunk_index",
]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity, returning zero for unusable vector pairs."""
    if not a or len(a) != len(b):
        return 0.0
    if not all(math.isfinite(float(value)) for value in a + b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, float(dot / (norm_a * norm_b))))


def _text_token_overlap(text_a: str, text_b: str) -> float | None:
    """Return Jaccard overlap of normalized word tokens, if both texts exist."""
    tokens_a = set(re.findall(r"\w+", text_a.casefold()))
    tokens_b = set(re.findall(r"\w+", text_b.casefold()))
    if not tokens_a or not tokens_b:
        return None
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 4)


def find_near_duplicates(connector: VectorDBConnector,
                          minimum_similarity: float = DEFAULT_MIN_SIMILARITY,
                          maximum_similarity: float = DEFAULT_MAX_SIMILARITY,
                          k: int = 5,
                          sample_size: int = 500,
                          field_mapper=None,
                          metadata_fields: List[str] | None = None) -> Dict:
    """Find high-cosine nearest-neighbor candidates for manual duplicate review.

    Args:
        minimum_similarity: Inclusive lower bound for cosine similarity, 0.65–0.98.
        maximum_similarity: Inclusive upper bound for cosine similarity, 0.65–0.98.
        k: Number of non-self nearest neighbors to inspect per sampled vector.
        sample_size: How many vectors to sample and check.
        field_mapper: Optional FieldMapper instance. When provided, each
            duplicate pair includes display metadata for both records.
        metadata_fields: Metadata keys to include with each candidate. Defaults
            to the required PDF identity and chunk-location fields.

    Returns:
        A result dict listing all discovered duplicate clusters.
    """
    if not MIN_SIMILARITY_THRESHOLD <= minimum_similarity <= MAX_SIMILARITY_THRESHOLD:
        raise ValueError(
            "minimum_similarity must be between "
            f"{MIN_SIMILARITY_THRESHOLD:.2f} and {MAX_SIMILARITY_THRESHOLD:.2f}."
        )
    if not MIN_SIMILARITY_THRESHOLD <= maximum_similarity <= MAX_SIMILARITY_THRESHOLD:
        raise ValueError(
            "maximum_similarity must be between "
            f"{MIN_SIMILARITY_THRESHOLD:.2f} and {MAX_SIMILARITY_THRESHOLD:.2f}."
        )
    if minimum_similarity > maximum_similarity:
        raise ValueError("minimum_similarity cannot exceed maximum_similarity.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    display_metadata_fields = (
        metadata_fields
        if metadata_fields is not None
        else DEFAULT_METADATA_FIELDS
    )

    similarity_candidates: List[Dict] = []
    seen_pairs = set()
    scanned = 0

    for record in connector.scan(batch_size=500):
        if not record.vector:
            continue
        scanned += 1
        if scanned > sample_size:
            break

        neighbors = connector.knn_search(record.vector, k=k + 1)  # +1 because self is returned

        for neighbor in neighbors:
            if neighbor.id == record.id:
                continue  # skip self-match

            pair_key = tuple(sorted([record.id, neighbor.id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = _cosine_similarity(record.vector, neighbor.vector)
            if minimum_similarity <= sim <= maximum_similarity:
                pair_entry = {
                    "id_a": record.id,
                    "id_b": neighbor.id,
                    "cosine_similarity": round(sim, 6),
                }

                record_with_text = connector.fetch(record.id)
                neighbor_with_text = connector.fetch(neighbor.id)
                chunk_a = (
                    (record_with_text.metadata or {}).get("document")
                    if record_with_text else None
                )
                chunk_b = (
                    (neighbor_with_text.metadata or {}).get("document")
                    if neighbor_with_text else None
                )
                if isinstance(chunk_a, str):
                    pair_entry["chunk_a"] = chunk_a
                if isinstance(chunk_b, str):
                    pair_entry["chunk_b"] = chunk_b
                pair_entry["text_token_overlap"] = (
                    _text_token_overlap(chunk_a, chunk_b)
                    if isinstance(chunk_a, str) and isinstance(chunk_b, str)
                    else None
                )

                metadata_a = (
                    record_with_text.metadata if record_with_text else record.metadata
                ) or {}
                metadata_b = (
                    neighbor_with_text.metadata if neighbor_with_text else neighbor.metadata
                ) or {}
                metadata_a = {
                    key: metadata_a[key]
                    for key in display_metadata_fields
                    if key in metadata_a
                }
                metadata_b = {
                    key: metadata_b[key]
                    for key in display_metadata_fields
                    if key in metadata_b
                }
                if field_mapper:
                    metadata_a = field_mapper.extract_display_metadata(metadata_a)
                    metadata_b = field_mapper.extract_display_metadata(metadata_b)
                pair_entry["metadata_a"] = metadata_a
                pair_entry["metadata_b"] = metadata_b
                similarity_candidates.append(pair_entry)

    similarity_candidates.sort(
        key=lambda pair: pair["cosine_similarity"],
        reverse=True,
    )
    candidates_found = len(similarity_candidates)
    return {
        "check": "near_duplicate_detection",
        "passed": None,
        "detail": {
            "vectors_sampled": min(scanned, sample_size),
            "minimum_similarity": minimum_similarity,
            "maximum_similarity": maximum_similarity,
            "neighbors_per_vector": k,
            "candidate_pairs_found": candidates_found,
            "candidate_pairs_reported": min(candidates_found, 50),
            "similarity_candidates": similarity_candidates[:50],
            "similarity_candidates_truncated": candidates_found > 50,
            # Retain legacy keys for consumers that expect duplicate-pair fields.
            "duplicates_found": candidates_found,
            "duplicate_pairs": similarity_candidates[:50],
            "duplicate_pairs_truncated": candidates_found > 50,
            "note": (
                f"Found {candidates_found} neighbor candidate pair(s) with cosine "
                f"similarity between {minimum_similarity:.2f} and "
                f"{maximum_similarity:.2f}. "
                "Cosine similarity measures embedding similarity, not duplicate "
                "text. Review text token overlap and content; this diagnostic "
                "does not automatically declare pairs to be duplicates. It checks "
                f"up to {k} nearest neighbors for each of "
                f"{min(scanned, sample_size)} sampled vectors."
            ),
        },
    }
