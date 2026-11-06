"""
HTML Report Generator.

Takes check results (list of dicts from the check modules) and produces a
self-contained, interactive HTML report with:
  - Failure tracebacks with root-cause analysis and remediation steps
  - Drilldown tables for null/zero/NaN vectors, duplicates, missing metadata
  - Help text tooltips on every check and chart
  - Interactive status filters (All / Pass / Fail / Info) and live search
  - Database-agnostic design — works with any vector DB connector

Usage::

    from report.html_report import generate_report

    html = generate_report(
        check_results=results,
        db_label="Chroma / my_collection",
        charts={"embedding_visualization": b64_png_string},
        hero_chart_b64=main_umap_b64,
    )
    with open("report.html", "w") as f:
        f.write(html)
"""

import json
import html as html_module
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from report.templates import (
    COLORS,
    REPORT_TEMPLATE,
    SUMMARY_TEMPLATE,
    HERO_CHART_TEMPLATE,
    CATEGORY_TEMPLATE,
    CHECK_CARD_TEMPLATE,
    CHART_EMBED_TEMPLATE,
    TRACEBACK_TEMPLATE,
    TRACEBACK_ROW_TEMPLATE,
    REMEDIATION_TEMPLATE,
    REMEDIATION_STEP_TEMPLATE,
    DETAIL_JSON_TEMPLATE,
    DETAIL_TABLE_TEMPLATE,
    DETAIL_ROW_TEMPLATE,
    STAT_ROW_TEMPLATE,
    STAT_ITEM_TEMPLATE,
    DRILLDOWN_TEMPLATE,
    DRILLDOWN_HEADER_CELL_TEMPLATE,
    DRILLDOWN_ROW_TEMPLATE,
    DRILLDOWN_CELL_TEMPLATE,
    SAMPLE_IDS_TEMPLATE,
    SAMPLE_ID_TEMPLATE,
    HELP_TEXTS,
    REMEDIATION_ADVICE,
)

# ─── Check-to-category mapping ───────────────────────────────────────
CHECK_CATEGORIES = {
    "dimension_consistency": "Vector Integrity",
    "magnitude_outliers": "Vector Integrity",
    "near_duplicate_detection": "Similarity Candidates",
    "metadata_completeness": "Metadata Completeness",
    "metadata_value_distribution": "Metadata Completeness",
    "known_similar_pairs": "Neighbor Consistency",
    "known_different_pairs": "Neighbor Consistency",
    "negative_query_similarity": "Similarity Calibration",
    "synthetic_question_retrieval": "Similarity Calibration",
    "embedding_drift": "Drift Detection",
    "drift_baseline": "Drift Detection",
    "searchability_spot_check": "Index Health",
    "empty_vector_audit": "Index Health",
    "embedding_visualization": "Visualization",
    "drift_visualization": "Visualization",
}

CATEGORY_ORDER = [
    "Vector Integrity",
    "Similarity Candidates",
    "Metadata Completeness",
    "Neighbor Consistency",
    "Similarity Calibration",
    "Drift Detection",
    "Index Health",
    "Visualization",
]

CHECK_DISPLAY_NAMES = {
    "dimension_consistency": "Dimension Consistency",
    "magnitude_outliers": "Magnitude Outliers",
    "near_duplicate_detection": "Duplicate Similarity Candidates",
    "metadata_completeness": "Metadata Completeness",
    "metadata_value_distribution": "Metadata Value Distribution",
    "known_similar_pairs": "Adjacent-Chunk Neighbor Diagnostic",
    "known_different_pairs": "Known Different Pairs",
    "negative_query_similarity": "Negative-Query Similarity Calibration",
    "synthetic_question_retrieval": "Synthetic Question Retrieval",
    "embedding_drift": "Embedding Drift",
    "drift_baseline": "Drift Baseline",
    "searchability_spot_check": "Full-Collection Searchability",
    "empty_vector_audit": "Empty Vector Audit",
    "embedding_visualization": "Embedding Space Visualization",
    "drift_visualization": "Drift Comparison Visualization",
}


# ─── Utility helpers ─────────────────────────────────────────────────

def _esc(value: Any) -> str:
    """HTML-escape a value for safe embedding."""
    return html_module.escape(str(value))


def _status_class(passed: Optional[bool]) -> str:
    if passed is True:
        return "pass"
    elif passed is False:
        return "fail"
    return "info"


def _status_label(passed: Optional[bool]) -> str:
    if passed is True:
        return "PASS"
    elif passed is False:
        return "FAIL"
    return "INFO"


def _fmt(value: Any) -> str:
    """Format a number for display."""
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)


# ─── Drilldown ID generator ──────────────────────────────────────────

_drilldown_counter = 0


def _next_dd_id() -> str:
    global _drilldown_counter
    _drilldown_counter += 1
    return f"dd{_drilldown_counter}"


# ─── Metadata-aware drilldown helpers ─────────────────────────────────

def _records_to_drilldown_columns_and_rows(
    records: List[Dict], extra_columns: Optional[List[str]] = None,
    extra_values_fn=None,
) -> tuple:
    """Convert enriched record dicts (with id + metadata) into columns/rows
    suitable for ``_build_drilldown``.

    If the first record has non-empty metadata keys, those become extra
    columns after Record ID.  Falls back to a plain [Record ID] table
    when metadata is absent.

    Args:
        records: List of ``{"id": ..., "metadata": {...}}`` dicts.
        extra_columns: Optional additional fixed columns to append
            (e.g. ["Fault Type"]).
        extra_values_fn: Optional callable ``(record) -> list[str]``
            returning values for *extra_columns*.

    Returns:
        (columns, rows) tuple ready for ``_build_drilldown``.
    """
    if not records:
        return (["Record ID"], [])

    # Determine metadata columns from the first record that has them
    meta_keys = []
    for rec in records:
        mk = list((rec.get("metadata") or {}).keys())
        if mk:
            meta_keys = mk
            break

    columns = ["Record ID"] + meta_keys + (extra_columns or [])
    rows = []
    for rec in records:
        rid = str(rec["id"]) if isinstance(rec, dict) else str(rec)
        meta = rec.get("metadata", {}) if isinstance(rec, dict) else {}
        row = [rid] + [str(meta.get(k, "")) for k in meta_keys]
        if extra_values_fn:
            row += extra_values_fn(rec)
        rows.append(row)
    return (columns, rows)


# ─── Component builders ──────────────────────────────────────────────

def _build_drilldown(label: str, columns: List[str], rows: List[List[str]]) -> str:
    """Build a collapsible, sortable, searchable drilldown table."""
    dd_id = _next_dd_id()
    hdr = ""
    for i, col in enumerate(columns):
        hdr += DRILLDOWN_HEADER_CELL_TEMPLATE.format(
            table_id=dd_id, col_idx=i, col_name=_esc(col))
    body = ""
    for row in rows:
        cells = ""
        for j, v in enumerate(row):
            cells += DRILLDOWN_CELL_TEMPLATE.format(
                cell_class="id-cell" if j == 0 else "", value=_esc(v))
        body += DRILLDOWN_ROW_TEMPLATE.format(cells=cells)
    return DRILLDOWN_TEMPLATE.format(
        drilldown_id=dd_id, toggle_label=_esc(label),
        item_count=len(rows), header_cells=hdr, body_rows=body)


def _build_stats(items: List[Dict[str, str]]) -> str:
    """Build an inline stat row."""
    html_parts = ""
    for it in items:
        html_parts += STAT_ITEM_TEMPLATE.format(
            label=_esc(it["label"]), value=_esc(it["value"]),
            color=it.get("color", COLORS["text"]))
    return STAT_ROW_TEMPLATE.format(items_html=html_parts)


def _build_traceback(check_name: str, detail: Dict) -> str:
    """Build a failure traceback panel with root-cause analysis."""
    rows = ""

    if check_name == "dimension_consistency":
        dims = detail.get("dimensions_found", {})
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Multiple vector dimensions detected in the same index")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Dimensions Found",
            value=", ".join(f"{d}d ({c} vectors)" for d, c in dims.items()))
        for dim, ids in detail.get("sample_ids_per_dimension", {}).items():
            if ids:
                rows += TRACEBACK_ROW_TEMPLATE.format(
                    label=f"Sample IDs ({dim}d)",
                    value=", ".join(str(i) for i in ids[:5]))

    elif check_name == "zero_nan_vectors":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Embedding pipeline produced invalid vectors")
        for ft, key in [("Zero", "zero_vectors"), ("NaN", "nan_vectors"), ("Inf", "inf_vectors")]:
            cnt = detail.get(key, 0)
            if cnt > 0:
                rows += TRACEBACK_ROW_TEMPLATE.format(
                    label=f"{ft} Vectors", value=f"{cnt} found")

    elif check_name == "magnitude_outliers":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Vector magnitudes outside expected distribution")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Outlier Count", value=str(detail.get("outlier_count", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Mean Norm", value=_fmt(detail.get("mean_norm", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Std Dev", value=_fmt(detail.get("std_norm", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Z-Threshold", value=str(detail.get("z_threshold", 3.0)))

    elif check_name == "near_duplicate_detection":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Finding",
            value="High-cosine neighbor candidates require content review; cosine similarity alone does not establish duplicate text.")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Candidate Pairs",
            value=str(detail.get("candidate_pairs_found", detail.get("duplicates_found", 0))))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Cosine Similarity Range",
            value=(
                f"{detail.get('minimum_similarity', 0.65)}–"
                f"{detail.get('maximum_similarity', 0.98)}"
            ))

    elif check_name == "metadata_completeness":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Required metadata fields missing or empty on some records")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Total Issues", value=str(detail.get("total_issues", 0)))
        for field, info in detail.get("per_field", {}).items():
            ti = info.get("total_incomplete", 0)
            if ti > 0:
                rows += TRACEBACK_ROW_TEMPLATE.format(
                    label=f"Field '{field}'",
                    value=f"{ti} incomplete ({info.get('completeness_pct', 0)}% complete)")

    elif check_name in ("known_similar_pairs", "known_different_pairs"):
        cause = ("Known-similar records not found in each other's nearest neighbors"
                 if check_name == "known_similar_pairs"
                 else "Known-different records appeared in each other's nearest neighbors")
        rows += TRACEBACK_ROW_TEMPLATE.format(label="Root Cause", value=cause)
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Pairs Failed",
            value=f"{detail.get('pairs_failed', 0)} of {detail.get('pairs_checked', 0)}")

    elif check_name == "negative_query_similarity":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Queries Above Maximum",
            value=(
                f"{detail.get('above_threshold_count', 0)} of "
                f"{detail.get('queries_tested', 0)}"
            ))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Maximum Negative Similarity",
            value=_fmt(detail.get("maximum_negative_similarity", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Configured Maximum Similarity",
            value=_fmt(detail.get("maximum_similarity_threshold", 0.32)))

    elif check_name == "embedding_drift":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Embedding distribution has shifted from the reference baseline")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Fréchet Distance", value=_fmt(detail.get("frechet_distance", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Centroid Cosine Similarity",
            value=_fmt(detail.get("centroid_cosine_similarity", 0)))
        if detail.get("alert_threshold"):
            rows += TRACEBACK_ROW_TEMPLATE.format(
                label="Alert Threshold", value=_fmt(detail["alert_threshold"]))

    elif check_name == "stored_vs_indexed_count":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Mismatch between stored and indexed record counts")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Stored Count", value=str(detail.get("stored_count", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Indexed Count", value=str(detail.get("indexed_count", 0)))
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Gap", value=str(detail.get("gap", 0)))

    elif check_name == "searchability_spot_check":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause",
            value="Some vectors cannot find themselves via kNN search — index may be corrupt")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Failures",
            value=f"{detail.get('failures', 0)} of {detail.get('records_checked', 0)} records checked")

    elif check_name == "empty_vector_audit":
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Root Cause", value="Records exist with no vector data")
        rows += TRACEBACK_ROW_TEMPLATE.format(
            label="Empty Vectors", value=str(detail.get("empty_vector_count", 0)))

    else:
        note = detail.get("note", "Check failed — see details below.")
        rows += TRACEBACK_ROW_TEMPLATE.format(label="Details", value=str(note))

    # Remediation
    advice = REMEDIATION_ADVICE.get(check_name, [])
    if advice:
        steps = "".join(REMEDIATION_STEP_TEMPLATE.format(step=_esc(s)) for s in advice)
        rem_html = REMEDIATION_TEMPLATE.format(steps_html=steps)
    else:
        rem_html = ""

    return TRACEBACK_TEMPLATE.format(traceback_rows=rows, remediation_html=rem_html)


# ─── Per-check body builders ─────────────────────────────────────────

def _body_dimension_consistency(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    stats = [{"label": "Total Scanned", "value": str(d.get("total_scanned", 0))}]
    if p:
        stats.append({"label": "Uniform Dimension",
                       "value": str(d.get("uniform_dimension", "?")),
                       "color": COLORS["pass"]})
    else:
        for dim, cnt in d.get("dimensions_found", {}).items():
            stats.append({"label": f"{dim}d Vectors", "value": str(cnt),
                           "color": COLORS["fail"]})
    body += _build_stats(stats)
    if not p and isinstance(d, dict):
        body += _build_traceback("dimension_consistency", d)
        # Prefer enriched records; fall back to plain IDs
        records_map = d.get("sample_records_per_dimension", {})
        ids_map = d.get("sample_ids_per_dimension", {})
        if records_map:
            all_records = []
            for dim, recs in records_map.items():
                for rec in recs:
                    rec_copy = dict(rec)
                    rec_copy.setdefault("_dim", str(dim))
                    all_records.append(rec_copy)
            if all_records:
                cols, rows = _records_to_drilldown_columns_and_rows(
                    all_records,
                    extra_columns=["Dimension"],
                    extra_values_fn=lambda rec: [str(rec.get("_dim", ""))],
                )
                body += _build_drilldown("View Affected Records", cols, rows)
        elif ids_map:
            rows = []
            for dim, ids in ids_map.items():
                for rid in ids:
                    rows.append([str(rid), str(dim)])
            if rows:
                body += _build_drilldown("View Affected Record IDs",
                                         ["Record ID", "Dimension"], rows)
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Dimension distribution",
            help_text="Bar chart showing how many vectors exist at each dimensionality. "
                      "A healthy database shows a single bar.")
    return body


def _body_zero_nan(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    c_pass, c_fail = COLORS["pass"], COLORS["fail"]
    stats = [
        {"label": "Total Scanned", "value": str(d.get("total_scanned", 0))},
        {"label": "Zero Vectors", "value": str(d.get("zero_vectors", 0)),
         "color": c_pass if d.get("zero_vectors", 0) == 0 else c_fail},
        {"label": "NaN Vectors", "value": str(d.get("nan_vectors", 0)),
         "color": c_pass if d.get("nan_vectors", 0) == 0 else c_fail},
        {"label": "Inf Vectors", "value": str(d.get("inf_vectors", 0)),
         "color": c_pass if d.get("inf_vectors", 0) == 0 else c_fail},
    ]
    body += _build_stats(stats)
    if not p and isinstance(d, dict):
        body += _build_traceback("zero_nan_vectors", d)
        for ft, rec_key, id_key in [
            ("Zero", "sample_zero_records", "sample_zero_ids"),
            ("NaN", "sample_nan_records", "sample_nan_ids"),
            ("Inf", "sample_inf_records", "sample_inf_ids"),
        ]:
            records = d.get(rec_key, [])
            ids = d.get(id_key, [])
            if records and isinstance(records[0], dict) and records[0].get("metadata"):
                cols, rows = _records_to_drilldown_columns_and_rows(
                    records,
                    extra_columns=["Fault Type"],
                    extra_values_fn=lambda _rec, _ft=ft: [_ft],
                )
                body += _build_drilldown(f"View {ft} Vector Records", cols, rows)
            elif ids:
                body += _build_drilldown(
                    f"View {ft} Vector IDs",
                    ["Record ID", "Fault Type"],
                    [[str(rid), ft] for rid in ids])
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Zero/NaN vectors highlighted",
            help_text="Embedding space plot with faulty vectors highlighted in red. "
                      "Zero vectors collapse to the origin. NaN/Inf vectors are shown at edges.")
    return body


def _body_magnitude_outliers(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    oc = d.get("outlier_count", 0)
    stats = [
        {"label": "Total Scanned", "value": str(d.get("total_scanned", 0))},
        {"label": "Mean Norm", "value": _fmt(d.get("mean_norm", 0))},
        {"label": "Std Dev", "value": _fmt(d.get("std_norm", 0))},
        {"label": "Outliers", "value": str(oc),
         "color": COLORS["pass"] if oc == 0 else COLORS["fail"]},
    ]
    if d.get("outlier_percentage") is not None:
        stats.append({"label": "Outlier %", "value": f"{d['outlier_percentage']}%",
                       "color": COLORS["pass"] if d["outlier_percentage"] == 0 else COLORS["fail"]})
    body += _build_stats(stats)
    if not p and isinstance(d, dict):
        body += _build_traceback("magnitude_outliers", d)
        outliers = d.get("sample_outliers", [])
        if outliers:
            # Check if outliers have enriched metadata
            has_meta = outliers[0].get("metadata") and len(outliers[0]["metadata"]) > 0
            if has_meta:
                meta_keys = list(outliers[0]["metadata"].keys())
                columns = ["Record ID"] + meta_keys + ["L2 Norm", "Z-Score"]
                rows = []
                for o in outliers:
                    meta = o.get("metadata", {})
                    row = [str(o["id"])]
                    row += [str(meta.get(k, "")) for k in meta_keys]
                    row += [_fmt(o["norm"]), _fmt(o["z_score"])]
                    rows.append(row)
                body += _build_drilldown("View Outlier Records", columns, rows)
            else:
                body += _build_drilldown(
                    "View Outlier Records",
                    ["Record ID", "L2 Norm", "Z-Score"],
                    [[str(o["id"]), _fmt(o["norm"]), _fmt(o["z_score"])] for o in outliers])
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Norm distribution histogram",
            help_text="Histogram of L2 norms (vector magnitudes). Red dashed lines show the "
                      "outlier threshold (mean ± z × std). Points beyond these lines are flagged.")
    return body


def _body_duplicates(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    body = ""
    dc = d.get("duplicates_found", 0)
    stats = [
        {"label": "Vectors Sampled", "value": str(d.get("vectors_sampled", 0))},
        {
            "label": "Cosine Similarity Range",
            "value": (
                f"{d.get('minimum_similarity', 0.65)}–"
                f"{d.get('maximum_similarity', 0.98)}"
            ),
        },
        {"label": "Candidate Pairs", "value": str(
            d.get("candidate_pairs_found", dc)
        )},
        {"label": "Neighbors per vector", "value": str(
            d.get("neighbors_per_vector", 5)
        )},
    ]
    body += _build_stats(stats)
    if d:
        body += (
            '<p style="font-size:0.82rem;color:#64748b;margin-top:0.5rem;">'
            "These are high-cosine neighbor candidates, not confirmed duplicates. "
            "Cosine measures embedding similarity; compare text token overlap and "
            "chunk content before calling a pair duplicate."
            "</p>"
        )
    if isinstance(d, dict):
        pairs = d.get("similarity_candidates", d.get("duplicate_pairs", []))
        if pairs:
            body += (
                '<details class="duplicate-candidates">'
                '<summary class="duplicate-candidates-toggle">'
                f'Show candidate pairs ({d.get("candidate_pairs_found", dc)})'
                '</summary>'
                '<div class="duplicate-candidates-content">'
                '<p style="font-size:0.82rem;color:#64748b;margin:0.5rem 0 0.6rem;">'
                'Sorted by cosine similarity, highest first. Open a pair to view '
                'its metadata and chunk text.'
                '</p>'
            )
            for index, pair in enumerate(pairs, start=1):
                body += _duplicate_pair_card(index, pair)
            if d.get(
                "similarity_candidates_truncated",
                d.get("duplicate_pairs_truncated"),
            ):
                body += (
                    '<p style="font-size:0.82rem;color:#64748b;margin-top:0.5rem;">'
                    'Showing the first 50 pairs; the total count includes every '
                    'candidate pair found in the scan.</p>'
                )
            body += '</div></details>'
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Similarity candidate visualization",
            help_text="Embedding-space plot connecting high-cosine neighbor candidates. "
                      "These candidates are not confirmed duplicate text.")
    return body


def _duplicate_pair_card(index: int, pair: Dict) -> str:
    similarity = pair.get("cosine_similarity", 0)
    overlap = pair.get("text_token_overlap")
    record_a = pair.get("id_a", "?")
    record_b = pair.get("id_b", "?")
    evidence = []
    for label, key in (("Record A content", "chunk_a"), ("Record B content", "chunk_b")):
        content = pair.get(key)
        if content:
            evidence.append(
                '<details class="duplicate-evidence">'
                f'<summary>{_esc(label)}</summary>'
                f'<div class="neighbor-chunk-text">{_esc(content)}</div>'
                '</details>'
            )
    metadata_html = (
        '<div class="neighbor-metadata">'
        f'{_neighbor_metadata_html("Record A metadata", pair.get("metadata_a", {}))}'
        f'{_neighbor_metadata_html("Record B metadata", pair.get("metadata_b", {}))}'
        '</div>'
    )
    return (
        '<details class="duplicate-pair">'
        '<summary class="duplicate-pair-toggle">'
        f'<strong>Pair {index}</strong>'
        f'<span class="duplicate-pair-ids">{_esc(record_a)} &harr; {_esc(record_b)}</span>'
        f'<span class="duplicate-similarity">Cosine {_fmt(similarity)}</span>'
        '</summary>'
        '<div class="duplicate-pair-content">'
        '<p style="font-size:0.8rem;color:#475569;margin-top:0.35rem;">'
        f'Text token overlap: {_esc(_fmt(overlap) if overlap is not None else "Unavailable")}'
        '</p>'
        f'{metadata_html}'
        f'{"".join(evidence)}'
        '</div></details>'
    )


def _body_metadata_completeness(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    stats = [
        {"label": "Total Scanned", "value": str(d.get("total_scanned", 0))},
        {"label": "Total Issues", "value": str(d.get("total_issues", 0)),
         "color": COLORS["pass"] if d.get("total_issues", 0) == 0 else COLORS["fail"]},
    ]
    body += _build_stats(stats)
    per_field = d.get("per_field", {})
    if per_field:
        rows = [
            [
                str(field),
                str(max(d.get("total_scanned", 0) - info.get("total_incomplete", 0), 0)),
                str(info.get("missing", 0)),
                str(info.get("empty", 0)),
                f'{info.get("completeness_pct", 0):g}%',
            ]
            for field, info in per_field.items()
        ]
        body += _build_drilldown(
            "Metadata field counts",
            ["Metadata field", "Populated", "Missing", "Empty", "Completeness"],
            rows,
        )
    if not p and isinstance(d, dict):
        body += _build_traceback("metadata_completeness", d)
        for fld, info in d.get("per_field", {}).items():
            ti = info.get("total_incomplete", 0)
            if ti > 0:
                sample_records = info.get("sample_records", [])
                sids = info.get("sample_ids", [])
                if sample_records and isinstance(sample_records[0], dict) and sample_records[0].get("metadata"):
                    cols, rows = _records_to_drilldown_columns_and_rows(
                        sample_records,
                        extra_columns=["Missing Field", "Field Completeness"],
                        extra_values_fn=lambda _rec, _f=fld, _i=info: [
                            _f, f"{_i.get('completeness_pct', 0)}%"
                        ],
                    )
                    body += _build_drilldown(
                        f"View Records Missing '{fld}' ({ti} records)", cols, rows)
                elif sids:
                    body += _build_drilldown(
                        f"View IDs Missing '{fld}' ({ti} records)",
                        ["Record ID", "Missing Field", "Field Completeness"],
                        [[str(rid), fld, f"{info.get('completeness_pct', 0)}%"]
                         for rid in sids])
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Metadata completeness chart",
            help_text="Horizontal bar chart showing completeness percentage for each required "
                      "metadata field. 100% means every record has that field populated.")
    return body


def _body_metadata_distribution(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    body = ""
    stats = [
        {"label": "Field", "value": str(d.get("field", "?")), "color": COLORS["info"]},
        {"label": "Total Scanned", "value": str(d.get("total_scanned", 0))},
        {"label": "Missing", "value": str(d.get("missing_count", 0))},
        {"label": "Unique Values", "value": str(d.get("unique_values", 0))},
    ]
    body += _build_stats(stats)
    tv = d.get("top_values", {})
    if tv:
        body += _build_drilldown("View Value Distribution", ["Value", "Count"],
                                 [[str(v), str(c)] for v, c in tv.items()])
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Metadata value distribution",
            help_text="Distribution of values for this metadata field. Highly skewed "
                      "distributions or unexpected values may indicate data quality issues.")
    return body


def _body_neighbor(r: Dict, check_name: str, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    missed_pairs = d.get("pairs_outside_top_k", d.get("failures", []))
    missed_count = d.get("pairs_not_mutual_top_k", d.get("pairs_failed", 0))
    body = ""
    if isinstance(d, str):
        body += f'<p style="color:{COLORS["text_light"]};font-size:0.85rem;">{_esc(d)}</p>'
        return body
    stats = [
        {
            "label": "Pairs checked",
            "value": (
                f'{d.get("pairs_checked", 0)} / '
                f'{d.get("pairs_sampled", d.get("pairs_checked", 0))}'
            ),
        },
        {"label": "Not mutual", "value": str(missed_count),
         "color": (
             COLORS["warn"] if missed_count and p is None
             else COLORS["fail"] if missed_count and p is False
             else COLORS["text"]
         )},
        {"label": "Top-k", "value": str(d.get("k", "?"))},
    ]
    body += _build_stats(stats)
    if p is None:
        note = d.get("methodology_note") or d.get("message") or d.get("note")
        if note:
            body += (
                f'<p style="color:{COLORS["warn"]};font-size:0.85rem;">'
                f'{_esc(note)}</p>'
            )
    pair_results = d.get("pair_results", [])
    if pair_results:
        failure_by_pair = {
            tuple(failure.get("pair", ())): failure
            for failure in missed_pairs
        }
        body += (
            '<details class="drilldown neighbor-pairs">'
            f'<summary class="drilldown-toggle">All sampled pairs ({len(pair_results)})</summary>'
            '<div class="drilldown-content open">'
        )
        for index, item in enumerate(pair_results, start=1):
            pair = item.get("pair", ("?", "?"))
            pair_key = tuple(pair)
            status = item.get("status", "unknown").replace("_", " ").title()
            failure = failure_by_pair.get(pair_key)
            body += _neighbor_pair_card(
                index,
                item,
                d.get("k", 10),
                failure=failure,
            )
        body += '</div></details>'
    if missed_pairs:
        body += (
            '<details class="drilldown neighbor-pairs">'
            f'<summary class="drilldown-toggle">Outside top-{_esc(d.get("k", 10))} '
            f'({len(missed_pairs)})</summary>'
            '<div class="drilldown-content open">'
        )
        pair_result_by_pair = {
            tuple(item.get("pair", ())): item
            for item in pair_results
        }
        for index, failure in enumerate(missed_pairs, start=1):
            pair = tuple(failure.get("pair", ()))
            item = pair_result_by_pair.get(pair, {
                "pair": failure.get("pair", ("?", "?")),
                "status": "unavailable",
                "rank_b_from_a": failure.get("rank_b_from_a"),
                "rank_a_from_b": failure.get("rank_a_from_b"),
                "metadata_a": failure.get("metadata_a", {}),
                "metadata_b": failure.get("metadata_b", {}),
                "chunk_a": failure.get("chunk_a", ""),
                "chunk_b": failure.get("chunk_b", ""),
            })
            body += _neighbor_pair_card(index, item, d.get("k", 10), failure)
        body += '</div></details>'
    if p is False:
        body += _build_traceback(check_name, d)
    unavailable = d.get("unavailable_pairs", [])
    if unavailable:
        body += _build_drilldown(
            "View Unavailable Pairs",
            ["Chunk A ID", "Chunk B ID", "Reason"],
            [
                [
                    str(item.get("pair", ("?", "?"))[0]),
                    str(item.get("pair", ("?", "?"))[1]),
                    item.get("reason", "Unavailable"),
                ]
                for item in unavailable
            ],
        )
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Neighbor consistency visualization",
            help_text="Embedding space plot with known pairs highlighted. Lines connect "
                      "pairs that should be close (green) or far apart (red).")
    return body


def _neighbor_rank_label(rank: Optional[int], k: int) -> str:
    if rank is not None and rank <= k:
        return str(rank)
    return f"Outside top-{k}"


def _neighbor_metadata_html(label: str, metadata: Dict) -> str:
    if not metadata:
        return (
            '<div class="neighbor-metadata-side">'
            f'<strong>{_esc(label)}</strong><span>Not available</span></div>'
        )
    fields = "".join(
        f'<span><strong>{_esc(key)}:</strong> {_esc(value)}</span>'
        for key, value in metadata.items()
    )
    return (
        '<div class="neighbor-metadata-side">'
        f'<strong>{_esc(label)}</strong>{fields}</div>'
    )


def _neighbor_pair_card(
    index: int,
    item: Dict,
    k: int,
    failure: Optional[Dict] = None,
) -> str:
    pair = item.get("pair", ("?", "?"))
    status = item.get("status", "unknown").replace("_", " ").title()
    chunk_a = item.get("chunk_a") or (failure or {}).get("chunk_a")
    chunk_b = item.get("chunk_b") or (failure or {}).get("chunk_b")
    return (
        '<section class="neighbor-pair">'
        f'<strong>Pair {index}</strong>'
        '<div class="neighbor-pair-id">'
        f'{_esc(pair[0])} &harr; {_esc(pair[1])}</div>'
        '<div class="neighbor-ranks">'
        f'A to B: {_esc(_neighbor_rank_label(item.get("rank_b_from_a"), k))}'
        ' &nbsp;·&nbsp; '
        f'B to A: {_esc(_neighbor_rank_label(item.get("rank_a_from_b"), k))}'
        f' &nbsp;·&nbsp; Result: {_esc(status)}</div>'
        '<div class="neighbor-metadata">'
        f'{_neighbor_metadata_html("Chunk A metadata", item.get("metadata_a", {}))}'
        f'{_neighbor_metadata_html("Chunk B metadata", item.get("metadata_b", {}))}'
        '</div>'
        '<div class="neighbor-metadata">'
        f'{_neighbor_metadata_html("Chunk A content", {"Content": chunk_a or "Text unavailable."})}'
        f'{_neighbor_metadata_html("Chunk B content", {"Content": chunk_b or "Text unavailable."})}'
        '</div>'
        '</section>'
    )


def _body_drift(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    if isinstance(d, str):
        body += f'<p style="color:{COLORS["text_light"]};font-size:0.85rem;">{_esc(d)}</p>'
        return body
    stats = [
        {"label": "Fréchet Distance", "value": _fmt(d.get("frechet_distance", 0)),
         "color": COLORS["fail"] if p is False else COLORS["text"]},
        {"label": "Centroid Cosine Sim", "value": _fmt(d.get("centroid_cosine_similarity", 0))},
        {"label": "Reference Size", "value": str(d.get("reference_sample_size", 0))},
        {"label": "Current Size", "value": str(d.get("current_sample_size", 0))},
    ]
    if d.get("baseline_distance") is not None:
        stats.append({"label": "Baseline", "value": _fmt(d["baseline_distance"])})
    if d.get("alert_threshold") is not None:
        stats.append({"label": "Alert Threshold", "value": _fmt(d["alert_threshold"]),
                       "color": COLORS["warn"]})
    body += _build_stats(stats)
    if p is False:
        body += _build_traceback("embedding_drift", d)
    # Top shifted dimensions drilldown
    top_dims = d.get("top_shifted_dimensions", [])
    if top_dims:
        body += _build_drilldown(
            "View Top Shifted Dimensions",
            ["Dimension", "Shift", "Ref Mean", "Cur Mean"],
            [[str(td["dimension"]), _fmt(td["shift"]),
              _fmt(td["ref_mean"]), _fmt(td["cur_mean"])] for td in top_dims])
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Drift comparison",
            help_text="Side-by-side 2D projection of reference (blue) and current (coral) "
                      "vectors. If the two clouds overlap well, there is little drift.")
    return body


def _body_stored_vs_indexed(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    stats = [
        {"label": "Stored Count", "value": str(d.get("stored_count", 0))},
        {"label": "Indexed Count", "value": str(d.get("indexed_count", 0))},
        {"label": "Gap", "value": str(d.get("gap", 0)),
         "color": COLORS["pass"] if d.get("gap", 0) == 0 else COLORS["fail"]},
    ]
    body += _build_stats(stats)
    if not p:
        body += _build_traceback("stored_vs_indexed_count", d)
    note = d.get("note", "")
    if note:
        body += f'<p style="font-size:0.82rem;color:{COLORS["text"]};margin-top:0.5rem;">{_esc(note)}</p>'
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Stored vs indexed count",
            help_text="Gauge showing the ratio of indexed to stored records. "
                      "100% means all records are searchable.")
    return body


def _body_searchability(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    stats = [
        {"label": "Records Checked", "value": str(d.get("records_checked", 0))},
        {"label": "Vectors Queried", "value": str(d.get("vectors_queried", 0))},
        {"label": "Failures", "value": str(d.get("failures", 0)),
         "color": COLORS["pass"] if d.get("failures", 0) == 0 else COLORS["fail"]},
    ]
    body += _build_stats(stats)
    if p is False:
        body += _build_traceback("searchability_spot_check", d)
        fd = d.get("failure_details", [])
        if fd:
            body += _build_drilldown(
                "View Failed Searchability Records",
                ["Record ID", "Reason", "Expected", "Got IDs"],
                [[str(f["id"]), f.get("reason", ""),
                  f.get("expected", "?"),
                  ", ".join(str(i) for i in f.get("got_ids", []))] for f in fd])
            if d.get("failure_details_truncated"):
                body += (
                    '<p style="font-size:0.82rem;color:'
                    f'{COLORS["text"]};margin-top:0.5rem;">'
                    "Showing the first 20 failure details; the failure count "
                    "includes every record."
                    "</p>"
                )
    note = d.get("note", "")
    if note:
        body += f'<p style="font-size:0.82rem;color:{COLORS["text"]};margin-top:0.5rem;">{_esc(note)}</p>'
    return body


def _body_empty_vectors(r: Dict, chart: Optional[str]) -> str:
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    ec = d.get("empty_vector_count", 0)
    stats = [
        {"label": "Total Scanned", "value": str(d.get("total_scanned", 0))},
        {"label": "Empty Vectors", "value": str(ec),
         "color": COLORS["pass"] if ec == 0 else COLORS["fail"]},
    ]
    body += _build_stats(stats)
    if not p:
        body += _build_traceback("empty_vector_audit", d)
        sample_records = d.get("sample_records", [])
        sids = d.get("sample_ids", [])
        if sample_records and isinstance(sample_records[0], dict) and sample_records[0].get("metadata"):
            cols, rows = _records_to_drilldown_columns_and_rows(sample_records)
            body += _build_drilldown("View Empty Vector Records", cols, rows)
        elif sids:
            body += _build_drilldown(
                "View Empty Vector IDs",
                ["Record ID"],
                [[str(rid)] for rid in sids])
    note = d.get("note", "")
    if note:
        body += f'<p style="font-size:0.82rem;color:{COLORS["text"]};margin-top:0.5rem;">{_esc(note)}</p>'
    return body


def _body_generic(r: Dict, chart: Optional[str]) -> str:
    """Fallback body builder for unknown check types."""
    d = r.get("detail", {})
    p = r.get("passed")
    body = ""
    if isinstance(d, str):
        body += f'<p style="font-size:0.85rem;">{_esc(d)}</p>'
    elif isinstance(d, dict):
        note = d.get("note", "")
        if note:
            body += f'<p style="font-size:0.82rem;margin-bottom:0.5rem;">{_esc(note)}</p>'
        if p is False:
            body += _build_traceback(r.get("check", "unknown"), d)
        # Render remaining detail as JSON
        body += DETAIL_JSON_TEMPLATE.format(
            json_content=_esc(json.dumps(d, indent=2, default=str)))
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart, alt_text="Check visualization",
            help_text="Visualization for this check result.")
    return body


def _body_negative_queries(r: Dict, chart: Optional[str]) -> str:
    """Show every tested query, its score, and matching text for threshold failures."""
    detail = r.get("detail", {})
    body = _build_stats([
        {"label": "Questions Tested", "value": str(detail.get("queries_tested", 0))},
        {
            "label": "Above Maximum",
            "value": str(detail.get("above_threshold_count", 0)),
        },
        {
            "label": "Configured Maximum",
            "value": _fmt(detail.get("maximum_similarity_threshold", 0.32)),
        },
    ])
    for item in detail.get("query_results", []):
        chunk = item.get("nearest_chunk")
        above_threshold = bool(item.get("above_threshold"))
        status = "Above maximum" if above_threshold else "Below maximum"
        chunk_html = ""
        if above_threshold and chunk:
            chunk_html = (
                f'<p><strong>Top matching chunk ID:</strong> '
                f'{_esc(chunk.get("id", ""))}</p>'
                '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;">'
                f'{_esc(chunk.get("text", ""))}</pre>'
            )
        body += (
            '<details style="margin:0.6rem 0;padding:0.6rem;border:1px solid '
            '#e2e8f0;border-radius:6px;">'
            f'<summary>{_esc(item.get("query", ""))} — '
            f'{_fmt(item.get("cosine_similarity", 0))} '
            f'({status})</summary>'
            f'<p><strong>Status:</strong> {status}</p>'
            f'<p><strong>Cosine similarity:</strong> '
            f'{_fmt(item.get("cosine_similarity", 0))}</p>'
            f'{chunk_html}</details>'
        )
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart,
            alt_text="Negative-query similarity",
            help_text="Similarity scores for the submitted off-topic questions.",
        )
    return body


def _body_synthetic_question_retrieval(
    r: Dict,
    chart: Optional[str],
) -> str:
    detail = r.get("detail", {})
    body = _build_stats([
        {
            "label": "Source Chunk",
            "value": str(detail.get("source_record_id", "")),
        },
        {"label": "Top-k", "value": str(detail.get("top_k", 0))},
        {"label": "Passed", "value": str(detail.get("passed_questions", 0))},
        {"label": "Failed", "value": str(detail.get("failed_questions", 0))},
    ])
    source_text = detail.get("source_chunk_text")
    if isinstance(source_text, str) and source_text:
        body += (
            '<details style="margin:0.6rem 0;padding:0.6rem;border:1px solid '
            '#e2e8f0;border-radius:6px;">'
            '<summary>Selected source chunk text</summary>'
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;">'
            f'{_esc(source_text)}</pre></details>'
        )
    rows = []
    for item in detail.get("question_results", []):
        rank = item.get("source_rank")
        rows.append([
            str(item.get("question", "")),
            f"Rank {rank}" if rank is not None else "Not in top-k",
            ", ".join(str(value) for value in item.get("retrieved_ids", [])),
        ])
    if rows:
        body += _build_drilldown(
            "Generated questions and retrieval results",
            ["Question", "Source chunk", "Retrieved record IDs"],
            rows,
        )
    if chart:
        body += CHART_EMBED_TEMPLATE.format(
            image_b64=chart,
            alt_text="Synthetic question retrieval",
            help_text="Whether generated questions retrieve their source chunk in top-k.",
        )
    return body


# ─── Body builder dispatch ───────────────────────────────────────────

_BODY_BUILDERS = {
    "dimension_consistency": lambda r, c: _body_dimension_consistency(r, c),
    "zero_nan_vectors": lambda r, c: _body_zero_nan(r, c),
    "magnitude_outliers": lambda r, c: _body_magnitude_outliers(r, c),
    "near_duplicate_detection": lambda r, c: _body_duplicates(r, c),
    "metadata_completeness": lambda r, c: _body_metadata_completeness(r, c),
    "metadata_value_distribution": lambda r, c: _body_metadata_distribution(r, c),
    "known_similar_pairs": lambda r, c: _body_neighbor(r, "known_similar_pairs", c),
    "known_different_pairs": lambda r, c: _body_neighbor(r, "known_different_pairs", c),
    "negative_query_similarity": _body_negative_queries,
    "synthetic_question_retrieval": _body_synthetic_question_retrieval,
    "embedding_drift": lambda r, c: _body_drift(r, c),
    "drift_baseline": lambda r, c: _body_generic(r, c),
    "stored_vs_indexed_count": lambda r, c: _body_stored_vs_indexed(r, c),
    "searchability_spot_check": lambda r, c: _body_searchability(r, c),
    "empty_vector_audit": lambda r, c: _body_empty_vectors(r, c),
    "embedding_visualization": lambda r, c: _body_generic(r, c),
    "drift_visualization": lambda r, c: _body_generic(r, c),
}


# ─── Main report generator ──────────────────────────────────────────

def generate_report(
    check_results: List[Dict],
    db_label: str = "Vector Database",
    charts: Optional[Dict[str, str]] = None,
    hero_chart_b64: Optional[str] = None,
    hero_chart_title: str = "Embedding Space (2D Projection)",
    timestamp: Optional[str] = None,
) -> str:
    """Generate a self-contained interactive HTML report.

    Args:
        check_results: List of result dicts from the check modules.
            Each dict must have ``check`` (str), ``passed`` (bool|None),
            and ``detail`` (dict|str) keys.
        db_label: Human-readable label for the database being tested.
            This is database-agnostic — pass any string like
            ``"Chroma / my_collection"`` or ``"Pinecone / prod-index"``.
        charts: Optional dict mapping check names to base64-encoded PNG
            strings for per-check visualizations.
        hero_chart_b64: Optional base64-encoded PNG for the main hero
            chart (e.g., UMAP embedding space plot).
        hero_chart_title: Title for the hero chart.
        timestamp: Optional ISO timestamp string. Defaults to now (UTC).

    Returns:
        A complete HTML string ready to write to a file.
    """
    global _drilldown_counter
    _drilldown_counter = 0  # reset for each report

    if charts is None:
        charts = {}
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Count statuses ────────────────────────────────────────────
    passed_count = sum(1 for r in check_results if r.get("passed") is True)
    failed_count = sum(1 for r in check_results if r.get("passed") is False)
    info_count = sum(1 for r in check_results if r.get("passed") is None)
    total_count = len(check_results)

    # ── Summary HTML ──────────────────────────────────────────────
    summary_html = SUMMARY_TEMPLATE.format(
        passed=passed_count,
        failed=failed_count,
        inconclusive=info_count,
        total=total_count,
        bg_dark=COLORS["bg_dark"],
    )

    # ── Hero chart HTML ───────────────────────────────────────────
    hero_html = ""
    if hero_chart_b64:
        hero_html = HERO_CHART_TEMPLATE.format(
            title=_esc(hero_chart_title),
            image_b64=hero_chart_b64,
            help_text=HELP_TEXTS.get("embedding_visualization",
                                     "2D projection of the embedding space."),
        )

    # ── Group checks by category ──────────────────────────────────
    categorized: Dict[str, List[Dict]] = {}
    for r in check_results:
        check_name = r.get("check", "unknown")
        cat = CHECK_CATEGORIES.get(check_name, "Other")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(r)

    # ── Build check cards ─────────────────────────────────────────
    checks_html = ""
    for cat_name in CATEGORY_ORDER:
        if cat_name not in categorized:
            continue
        cards_html = ""
        for r in categorized[cat_name]:
            check_name = r.get("check", "unknown")
            passed = r.get("passed")
            sc = _status_class(passed)
            sl = _status_label(passed)
            display_name = CHECK_DISPLAY_NAMES.get(check_name, check_name)
            help_text = HELP_TEXTS.get(check_name, "No description available.")

            # Build body using the appropriate builder
            builder = _BODY_BUILDERS.get(check_name, lambda r2, c2: _body_generic(r2, c2))
            chart_b64 = charts.get(check_name)
            body_html = builder(r, chart_b64)

            cards_html += CHECK_CARD_TEMPLATE.format(
                status_class=sc,
                status_label=sl,
                check_name=_esc(display_name),
                help_text=help_text,
                body_html=body_html,
            )

        checks_html += CATEGORY_TEMPLATE.format(
            category_name=_esc(cat_name),
            cards_html=cards_html,
        )

    # Handle any uncategorized checks
    for cat_name, results in categorized.items():
        if cat_name in CATEGORY_ORDER:
            continue
        cards_html = ""
        for r in results:
            check_name = r.get("check", "unknown")
            passed = r.get("passed")
            sc = _status_class(passed)
            sl = _status_label(passed)
            display_name = CHECK_DISPLAY_NAMES.get(check_name, check_name)
            help_text = HELP_TEXTS.get(check_name, "No description available.")
            chart_b64 = charts.get(check_name)
            body_html = _body_generic(r, chart_b64)
            cards_html += CHECK_CARD_TEMPLATE.format(
                status_class=sc, status_label=sl,
                check_name=_esc(display_name), help_text=help_text,
                body_html=body_html)
        checks_html += CATEGORY_TEMPLATE.format(
            category_name=_esc(cat_name), cards_html=cards_html)

    # ── Subtitle ──────────────────────────────────────────────────
    subtitle = f"Generated {timestamp} &middot; {total_count} checks run"

    # ── Assemble final HTML ───────────────────────────────────────
    return REPORT_TEMPLATE.format(
        report_subtitle=subtitle,
        db_label=_esc(db_label),
        summary_html=summary_html,
        hero_chart_html=hero_html,
        checks_html=checks_html,
        timestamp=timestamp,
        total_checks=total_count,
        passed=passed_count,
        failed=failed_count,
        inconclusive=info_count,
        # Colors
        bg_page=COLORS["bg_page"],
        bg_dark=COLORS["bg_dark"],
        bg_card=COLORS["bg_card"],
        text=COLORS["text"],
        text_light=COLORS["text_light"],
        border=COLORS["border"],
        pass_color=COLORS["pass"],
        fail_color=COLORS["fail"],
        info_color=COLORS["info"],
        warn_color=COLORS["warn"],
    )
