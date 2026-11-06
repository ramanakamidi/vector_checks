"""
Auto-Pair Generation from Metadata.

Scans the vector database, groups records by metadata fields, and
automatically generates similar/different pairs + drift batches.
Exports everything to an Excel file so developers can:
  1. See what was generated and why
  2. Optionally feed it through an LLM for review (Phase 2)
  3. Run the final neighbor consistency checks (Phase 3)

This eliminates manual curation — pairs are derived from metadata
that already exists in the database (doc_id, object_type, etc.).

Usage::

    from checks.auto_pairs import generate_pairs_from_metadata

    result = generate_pairs_from_metadata(
        connector=my_connector,
        group_field="doc_id",           # same doc_id → similar
        contrast_field="object_type",   # different object_type → different
        time_field="ingestion_date",    # for drift batches
        output_path="auto_pairs.xlsx",
    )
"""

import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from connectors.base import VectorDBConnector, VectorRecord

# openpyxl is only needed for Excel export — imported lazily
# so the module can be imported even without it installed.


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _sample_pairs_within_group(
    ids: List[str],
    max_pairs: int = 3,
) -> List[Tuple[str, str]]:
    """Pick random pairs from within a single group."""
    if len(ids) < 2:
        return []
    pairs = []
    available = list(ids)
    random.shuffle(available)
    for i in range(0, min(len(available) - 1, max_pairs * 2), 2):
        if i + 1 < len(available):
            pairs.append((available[i], available[i + 1]))
        if len(pairs) >= max_pairs:
            break
    return pairs


def _sample_pairs_across_groups(
    groups: Dict[Any, List[str]],
    max_pairs: int = 10,
) -> List[Tuple[str, str, Any, Any]]:
    """Pick random pairs from different groups.
    Returns (id_a, id_b, group_a, group_b) tuples.
    """
    group_keys = [k for k, v in groups.items() if len(v) > 0]
    if len(group_keys) < 2:
        return []
    pairs = []
    for _ in range(max_pairs * 3):  # oversample then trim
        g1, g2 = random.sample(group_keys, 2)
        a = random.choice(groups[g1])
        b = random.choice(groups[g2])
        pairs.append((a, b, g1, g2))
        if len(pairs) >= max_pairs:
            break
    return pairs[:max_pairs]


def generate_pairs_from_metadata(
    connector: VectorDBConnector,
    group_field: str = "doc_id",
    contrast_field: Optional[str] = "object_type",
    time_field: Optional[str] = "ingestion_date",
    similar_pairs_per_group: int = 3,
    different_pairs_total: int = 20,
    drift_sample_size: int = 500,
    metadata_fields_to_capture: Optional[List[str]] = None,
    output_path: str = "auto_pairs.xlsx",
    seed: int = 42,
) -> Dict:
    """Scan the database, group by metadata, and auto-generate pairs.

    This function:
      1. Scans all records once
      2. Groups them by ``group_field`` (e.g., doc_id)
      3. Generates similar pairs (same group) and different pairs (cross-group)
      4. Computes actual cosine similarity for each pair
      5. Optionally groups by ``time_field`` for drift batches
      6. Exports everything to an Excel file with columns:
         - id_a, id_b, group_a, group_b, expected_relationship,
           actual_cosine_similarity, metadata_a, metadata_b

    Args:
        connector: Any VectorDBConnector implementation.
        group_field: Metadata field to group by for similar pairs.
            Records sharing the same value are expected to be similar.
        contrast_field: Metadata field for generating different pairs.
            Records with different values are expected to be far apart.
            If None, uses group_field for cross-group different pairs.
        time_field: Metadata field for drift batches (e.g., ingestion_date).
            If None, drift batches are not generated.
        similar_pairs_per_group: Max similar pairs to generate per group.
        different_pairs_total: Total different pairs to generate.
        drift_sample_size: Max vectors per drift batch.
        metadata_fields_to_capture: Extra metadata fields to include in
            the Excel output. If None, captures all available fields.
        output_path: Where to save the Excel file.
        seed: Random seed for reproducibility.

    Returns:
        A result dict with:
          - similar_pairs: list of (id_a, id_b) tuples
          - different_pairs: list of (id_a, id_b) tuples
          - drift_reference / drift_current: vector lists (if time_field set)
          - output_path: path to the generated Excel file
          - statistics: group counts, pair counts, etc.
    """
    random.seed(seed)

    # ── Phase 1: Scan and group ───────────────────────────────────
    records_by_id: Dict[str, VectorRecord] = {}
    groups: Dict[Any, List[str]] = {}           # group_field → [ids]
    contrast_groups: Dict[Any, List[str]] = {}  # contrast_field → [ids]
    time_groups: Dict[Any, List[str]] = {}      # time_field → [ids]

    total_scanned = 0
    for record in connector.scan(batch_size=500):
        total_scanned += 1
        records_by_id[record.id] = record

        # Group by primary field
        gval = record.metadata.get(group_field)
        if gval is not None:
            groups.setdefault(str(gval), []).append(record.id)

        # Group by contrast field
        if contrast_field:
            cval = record.metadata.get(contrast_field)
            if cval is not None:
                contrast_groups.setdefault(str(cval), []).append(record.id)

        # Group by time field
        if time_field:
            tval = record.metadata.get(time_field)
            if tval is not None:
                time_groups.setdefault(str(tval), []).append(record.id)

    # ── Phase 2: Generate similar pairs ───────────────────────────
    similar_pairs_raw: List[Dict] = []
    for group_val, ids in groups.items():
        pairs = _sample_pairs_within_group(ids, similar_pairs_per_group)
        for id_a, id_b in pairs:
            rec_a = records_by_id[id_a]
            rec_b = records_by_id[id_b]
            cos_sim = 0.0
            if rec_a.vector and rec_b.vector:
                cos_sim = _cosine_similarity(rec_a.vector, rec_b.vector)
            similar_pairs_raw.append({
                "id_a": id_a,
                "id_b": id_b,
                "group_field": group_field,
                "group_value_a": group_val,
                "group_value_b": group_val,
                "expected_relationship": "SIMILAR",
                "reason": f"Same {group_field} = '{group_val}'",
                "actual_cosine_similarity": round(cos_sim, 6),
                "metadata_a": _extract_metadata(rec_a, metadata_fields_to_capture),
                "metadata_b": _extract_metadata(rec_b, metadata_fields_to_capture),
            })

    # ── Phase 3: Generate different pairs ─────────────────────────
    different_pairs_raw: List[Dict] = []
    source_groups = contrast_groups if contrast_field and contrast_groups else groups
    source_field = contrast_field if contrast_field and contrast_groups else group_field

    cross_pairs = _sample_pairs_across_groups(source_groups, different_pairs_total)
    for id_a, id_b, g_a, g_b in cross_pairs:
        rec_a = records_by_id[id_a]
        rec_b = records_by_id[id_b]
        cos_sim = 0.0
        if rec_a.vector and rec_b.vector:
            cos_sim = _cosine_similarity(rec_a.vector, rec_b.vector)
        different_pairs_raw.append({
            "id_a": id_a,
            "id_b": id_b,
            "group_field": source_field,
            "group_value_a": g_a,
            "group_value_b": g_b,
            "expected_relationship": "DIFFERENT",
            "reason": f"Different {source_field}: '{g_a}' vs '{g_b}'",
            "actual_cosine_similarity": round(cos_sim, 6),
            "metadata_a": _extract_metadata(rec_a, metadata_fields_to_capture),
            "metadata_b": _extract_metadata(rec_b, metadata_fields_to_capture),
        })

    # ── Phase 4: Drift batches ────────────────────────────────────
    drift_reference = []
    drift_current = []
    drift_info = {}
    if time_field and len(time_groups) >= 2:
        sorted_times = sorted(time_groups.keys())
        ref_time = sorted_times[0]
        cur_time = sorted_times[-1]
        ref_ids = time_groups[ref_time][:drift_sample_size]
        cur_ids = time_groups[cur_time][:drift_sample_size]
        drift_reference = [
            records_by_id[rid].vector for rid in ref_ids
            if records_by_id[rid].vector
        ]
        drift_current = [
            records_by_id[rid].vector for rid in cur_ids
            if records_by_id[rid].vector
        ]
        drift_info = {
            "reference_time": ref_time,
            "current_time": cur_time,
            "reference_count": len(drift_reference),
            "current_count": len(drift_current),
            "all_time_periods": sorted_times,
        }

    # ── Phase 5: Export to Excel ──────────────────────────────────
    _export_to_excel(
        similar_pairs_raw,
        different_pairs_raw,
        drift_info,
        output_path,
    )

    # ── Build result ──────────────────────────────────────────────
    similar_tuples = [(p["id_a"], p["id_b"]) for p in similar_pairs_raw]
    different_tuples = [(p["id_a"], p["id_b"]) for p in different_pairs_raw]

    return {
        "similar_pairs": similar_tuples,
        "different_pairs": different_tuples,
        "similar_pairs_detail": similar_pairs_raw,
        "different_pairs_detail": different_pairs_raw,
        "drift_reference": drift_reference,
        "drift_current": drift_current,
        "drift_info": drift_info,
        "output_path": output_path,
        "statistics": {
            "total_scanned": total_scanned,
            "groups_found": len(groups),
            "group_field": group_field,
            "contrast_groups_found": len(contrast_groups) if contrast_field else 0,
            "contrast_field": contrast_field,
            "time_periods_found": len(time_groups) if time_field else 0,
            "time_field": time_field,
            "similar_pairs_generated": len(similar_tuples),
            "different_pairs_generated": len(different_tuples),
            "drift_reference_size": len(drift_reference),
            "drift_current_size": len(drift_current),
        },
    }


def _extract_metadata(
    record: VectorRecord,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract metadata fields from a record for Excel export."""
    if fields is None:
        return dict(record.metadata)
    return {f: record.metadata.get(f) for f in fields}


def _export_to_excel(
    similar_pairs: List[Dict],
    different_pairs: List[Dict],
    drift_info: Dict,
    output_path: str,
) -> None:
    """Export auto-generated pairs to an Excel workbook with 3 sheets.

    Sheet 1 — "Similar Pairs": pairs expected to be close
    Sheet 2 — "Different Pairs": pairs expected to be far apart
    Sheet 3 — "Drift Info": time-based batch information
    Sheet 4 — "Instructions": how to use this file

    Each pair row includes:
      - id_a, id_b
      - group_field, group_value_a, group_value_b
      - expected_relationship (SIMILAR / DIFFERENT)
      - reason (why this pair was generated)
      - actual_cosine_similarity (measured from the DB)
      - llm_verdict (empty — for Phase 2 LLM review)
      - llm_reason (empty — for Phase 2 LLM review)
      - final_expected (empty — for Phase 2 or manual override)
      - metadata columns for both records
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        # Fallback: export as CSV if openpyxl not installed
        _export_to_csv_fallback(similar_pairs, different_pairs, drift_info, output_path)
        return

    wb = Workbook()

    # ── Styles ────────────────────────────────────────────────────
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    similar_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    different_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )

    def style_header(ws, num_cols):
        for col in range(1, num_cols + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
            cell.border = thin_border

    # ── Collect all metadata keys ─────────────────────────────────
    all_meta_keys = set()
    for p in similar_pairs + different_pairs:
        all_meta_keys.update(p.get("metadata_a", {}).keys())
        all_meta_keys.update(p.get("metadata_b", {}).keys())
    meta_keys = sorted(all_meta_keys)

    # ── Sheet 1: Similar Pairs ────────────────────────────────────
    ws_sim = wb.active
    ws_sim.title = "Similar Pairs"

    base_headers = [
        "ID A", "ID B", "Group Field", "Group Value A", "Group Value B",
        "Expected Relationship", "Reason", "Actual Cosine Similarity",
        "LLM Verdict", "LLM Reason", "Final Expected",
    ]
    meta_headers_a = [f"A: {k}" for k in meta_keys]
    meta_headers_b = [f"B: {k}" for k in meta_keys]
    all_headers = base_headers + meta_headers_a + meta_headers_b

    ws_sim.append(all_headers)
    style_header(ws_sim, len(all_headers))

    for p in similar_pairs:
        meta_a = p.get("metadata_a", {})
        meta_b = p.get("metadata_b", {})
        row = [
            p["id_a"], p["id_b"], p["group_field"],
            str(p["group_value_a"]), str(p["group_value_b"]),
            p["expected_relationship"], p["reason"],
            p["actual_cosine_similarity"],
            "",  # LLM Verdict (Phase 2)
            "",  # LLM Reason (Phase 2)
            "",  # Final Expected (Phase 2 or manual)
        ]
        row += [str(meta_a.get(k, "")) for k in meta_keys]
        row += [str(meta_b.get(k, "")) for k in meta_keys]
        ws_sim.append(row)

    # Color-code the expected relationship column
    for row_idx in range(2, len(similar_pairs) + 2):
        ws_sim.cell(row=row_idx, column=6).fill = similar_fill

    # Auto-width
    for col in ws_sim.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws_sim.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    # ── Sheet 2: Different Pairs ──────────────────────────────────
    ws_diff = wb.create_sheet("Different Pairs")
    ws_diff.append(all_headers)
    style_header(ws_diff, len(all_headers))

    for p in different_pairs:
        meta_a = p.get("metadata_a", {})
        meta_b = p.get("metadata_b", {})
        row = [
            p["id_a"], p["id_b"], p["group_field"],
            str(p["group_value_a"]), str(p["group_value_b"]),
            p["expected_relationship"], p["reason"],
            p["actual_cosine_similarity"],
            "",  # LLM Verdict
            "",  # LLM Reason
            "",  # Final Expected
        ]
        row += [str(meta_a.get(k, "")) for k in meta_keys]
        row += [str(meta_b.get(k, "")) for k in meta_keys]
        ws_diff.append(row)

    for row_idx in range(2, len(different_pairs) + 2):
        ws_diff.cell(row=row_idx, column=6).fill = different_fill

    for col in ws_diff.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws_diff.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    # ── Sheet 3: Drift Info ───────────────────────────────────────
    ws_drift = wb.create_sheet("Drift Info")
    drift_headers = ["Property", "Value"]
    ws_drift.append(drift_headers)
    style_header(ws_drift, 2)

    if drift_info:
        for key, val in drift_info.items():
            ws_drift.append([str(key), str(val)])
    else:
        ws_drift.append(["Status", "No time field provided or insufficient time periods"])

    for col in ws_drift.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws_drift.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)

    # ── Sheet 4: Instructions ─────────────────────────────────────
    ws_inst = wb.create_sheet("Instructions")
    instructions = [
        ["Vector Database Auto-Pair Generation Report"],
        [""],
        ["HOW THIS FILE WAS GENERATED"],
        ["This file was auto-generated by scanning the vector database and grouping"],
        ["records by metadata fields. No manual curation was needed."],
        [""],
        ["SHEET DESCRIPTIONS"],
        ["Similar Pairs — Records that share the same metadata group value."],
        ["  These are EXPECTED to be close in embedding space (high cosine similarity)."],
        ["Different Pairs — Records from different metadata groups."],
        ["  These are EXPECTED to be far apart (low cosine similarity)."],
        ["Drift Info — Time-based batch information for drift detection."],
        [""],
        ["COLUMN DESCRIPTIONS"],
        ["ID A / ID B — Record identifiers in the vector database."],
        ["Group Field — The metadata field used for grouping."],
        ["Group Value A / B — The actual metadata values for each record."],
        ["Expected Relationship — SIMILAR or DIFFERENT (auto-determined from metadata)."],
        ["Reason — Why this pair was generated (human-readable explanation)."],
        ["Actual Cosine Similarity — Measured cosine similarity between the two vectors."],
        ["LLM Verdict — (Phase 2) LLM's assessment: CLOSE / NOT_CLOSE / UNCERTAIN."],
        ["LLM Reason — (Phase 2) LLM's reasoning for its verdict."],
        ["Final Expected — (Phase 2/Manual) The final expected relationship after review."],
        ["A: / B: columns — Metadata fields for each record in the pair."],
        [""],
        ["THREE-PHASE WORKFLOW"],
        ["Phase 1 (this file): Auto-generate pairs from metadata. DONE."],
        ["Phase 2 (optional): Feed this file to an LLM for review."],
        ["  The LLM reads the metadata and fills in LLM Verdict + LLM Reason."],
        ["  It may override Expected Relationship based on semantic understanding."],
        ["Phase 3: Run the reviewed file through pair_runner.py to execute"],
        ["  the actual neighbor consistency checks against the vector database."],
        [""],
        [f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"],
    ]
    for row in instructions:
        ws_inst.append(row)

    # Style the title
    ws_inst.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws_inst.column_dimensions['A'].width = 80

    wb.save(output_path)


def _export_to_csv_fallback(
    similar_pairs: List[Dict],
    different_pairs: List[Dict],
    drift_info: Dict,
    output_path: str,
) -> None:
    """Fallback CSV export if openpyxl is not installed."""
    import csv

    csv_path = output_path.replace(".xlsx", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id_a", "id_b", "group_field", "group_value_a", "group_value_b",
            "expected_relationship", "reason", "actual_cosine_similarity",
            "llm_verdict", "llm_reason", "final_expected",
        ])
        for p in similar_pairs + different_pairs:
            writer.writerow([
                p["id_a"], p["id_b"], p["group_field"],
                p["group_value_a"], p["group_value_b"],
                p["expected_relationship"], p["reason"],
                p["actual_cosine_similarity"],
                "", "", "",
            ])
