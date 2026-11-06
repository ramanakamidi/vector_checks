"""
Pair Runner (Phase 3).

Reads the reviewed Excel file (from Phase 1 or Phase 2) and runs the
actual neighbor consistency checks against the vector database.

This is the final step in the three-phase workflow:
  Phase 1: auto_pairs.py generates pairs from metadata → Excel
  Phase 2: llm_review.py (optional) reviews pairs with an LLM → Excel
  Phase 3: pair_runner.py reads Excel → runs checks → produces results

The runner reads the "Final Expected" column (set by LLM or manually)
to determine which pairs should be similar vs. different. If "Final
Expected" is empty, it falls back to "Expected Relationship" (the
auto-generated value from Phase 1).

Usage::

    from checks.pair_runner import run_pairs_from_excel

    results = run_pairs_from_excel(
        connector=my_connector,
        input_path="reviewed_pairs.xlsx",
        output_path="final_results.xlsx",
    )
    # results contains check dicts ready for the HTML report
"""

from typing import Any, Dict, List, Optional, Tuple

from connectors.base import VectorDBConnector


def run_pairs_from_excel(
    connector: VectorDBConnector,
    input_path: str = "auto_pairs.xlsx",
    output_path: Optional[str] = None,
    k_similar: int = 20,
    k_different: int = 10,
) -> Dict:
    """Read pairs from Excel and run neighbor consistency checks.

    Args:
        connector: Any VectorDBConnector implementation.
        input_path: Path to the Excel file (from Phase 1 or 2).
        output_path: Where to save the results Excel. If None,
            generates a name based on input_path.
        k_similar: Number of neighbors to check for similar pairs.
        k_different: Number of neighbors to check for different pairs.

    Returns:
        A dict containing:
          - similar_check: result dict from check_known_similar_pairs
          - different_check: result dict from check_known_different_pairs
          - detailed_results: per-pair results with pass/fail
          - output_path: path to the results Excel file
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        return _run_from_csv_fallback(connector, input_path, k_similar, k_different)

    if output_path is None:
        output_path = input_path.replace(".xlsx", "_results.xlsx")

    wb = load_workbook(input_path)

    # ── Read similar pairs ────────────────────────────────────────
    similar_pairs: List[Tuple[str, str]] = []
    similar_details: List[Dict] = []

    if "Similar Pairs" in wb.sheetnames:
        ws = wb["Similar Pairs"]
        headers = [cell.value for cell in ws[1]]
        col_map = {h: i for i, h in enumerate(headers) if h}

        for row_idx in range(2, ws.max_row + 1):
            id_a = ws.cell(row=row_idx, column=col_map.get("ID A", 0) + 1).value
            id_b = ws.cell(row=row_idx, column=col_map.get("ID B", 0) + 1).value

            if not id_a or not id_b:
                continue

            # Use Final Expected if available, else Expected Relationship
            final = ws.cell(row=row_idx, column=col_map.get("Final Expected", 10) + 1).value
            expected = ws.cell(row=row_idx, column=col_map.get("Expected Relationship", 5) + 1).value
            effective = (final or expected or "SIMILAR").upper()

            llm_verdict = ws.cell(row=row_idx, column=col_map.get("LLM Verdict", 8) + 1).value or ""
            llm_reason = ws.cell(row=row_idx, column=col_map.get("LLM Reason", 9) + 1).value or ""
            reason = ws.cell(row=row_idx, column=col_map.get("Reason", 6) + 1).value or ""
            cosine = ws.cell(row=row_idx, column=col_map.get("Actual Cosine Similarity", 7) + 1).value or 0

            if effective == "SIMILAR":
                similar_pairs.append((str(id_a), str(id_b)))

            similar_details.append({
                "id_a": str(id_a),
                "id_b": str(id_b),
                "expected": expected or "",
                "final_expected": effective,
                "llm_verdict": llm_verdict,
                "llm_reason": llm_reason,
                "auto_reason": reason,
                "cosine_similarity": cosine,
                "sheet": "Similar Pairs",
            })

    # ── Read different pairs ──────────────────────────────────────
    different_pairs: List[Tuple[str, str]] = []
    different_details: List[Dict] = []

    if "Different Pairs" in wb.sheetnames:
        ws = wb["Different Pairs"]
        headers = [cell.value for cell in ws[1]]
        col_map = {h: i for i, h in enumerate(headers) if h}

        for row_idx in range(2, ws.max_row + 1):
            id_a = ws.cell(row=row_idx, column=col_map.get("ID A", 0) + 1).value
            id_b = ws.cell(row=row_idx, column=col_map.get("ID B", 0) + 1).value

            if not id_a or not id_b:
                continue

            final = ws.cell(row=row_idx, column=col_map.get("Final Expected", 10) + 1).value
            expected = ws.cell(row=row_idx, column=col_map.get("Expected Relationship", 5) + 1).value
            effective = (final or expected or "DIFFERENT").upper()

            llm_verdict = ws.cell(row=row_idx, column=col_map.get("LLM Verdict", 8) + 1).value or ""
            llm_reason = ws.cell(row=row_idx, column=col_map.get("LLM Reason", 9) + 1).value or ""
            reason = ws.cell(row=row_idx, column=col_map.get("Reason", 6) + 1).value or ""
            cosine = ws.cell(row=row_idx, column=col_map.get("Actual Cosine Similarity", 7) + 1).value or 0

            if effective == "DIFFERENT":
                different_pairs.append((str(id_a), str(id_b)))

            different_details.append({
                "id_a": str(id_a),
                "id_b": str(id_b),
                "expected": expected or "",
                "final_expected": effective,
                "llm_verdict": llm_verdict,
                "llm_reason": llm_reason,
                "auto_reason": reason,
                "cosine_similarity": cosine,
                "sheet": "Different Pairs",
            })

    # ── Run the actual checks ─────────────────────────────────────
    from checks.neighbor_consistency import (
        check_known_similar_pairs,
        check_known_different_pairs,
    )

    similar_result = check_known_similar_pairs(
        connector, similar_pairs, k=k_similar
    )
    different_result = check_known_different_pairs(
        connector, different_pairs, k=k_different
    )

    # ── Merge check results with pair details ─────────────────────
    # Build a lookup of failures by pair
    sim_failures = set()
    if isinstance(similar_result.get("detail"), dict):
        for f in similar_result["detail"].get("failures", []):
            pair = f.get("pair", ())
            if len(pair) == 2:
                sim_failures.add((str(pair[0]), str(pair[1])))

    diff_failures = set()
    if isinstance(different_result.get("detail"), dict):
        for f in different_result["detail"].get("failures", []):
            pair = f.get("pair", ())
            if len(pair) == 2:
                diff_failures.add((str(pair[0]), str(pair[1])))

    # Mark each detail with pass/fail
    for d in similar_details:
        pair_key = (d["id_a"], d["id_b"])
        pair_key_rev = (d["id_b"], d["id_a"])
        if d["final_expected"] == "SIMILAR":
            d["check_result"] = "FAIL" if (pair_key in sim_failures or pair_key_rev in sim_failures) else "PASS"
        else:
            d["check_result"] = "SKIPPED"

    for d in different_details:
        pair_key = (d["id_a"], d["id_b"])
        pair_key_rev = (d["id_b"], d["id_a"])
        if d["final_expected"] == "DIFFERENT":
            d["check_result"] = "FAIL" if (pair_key in diff_failures or pair_key_rev in diff_failures) else "PASS"
        else:
            d["check_result"] = "SKIPPED"

    all_details = similar_details + different_details

    # ── Export results to Excel ───────────────────────────────────
    _export_results_excel(all_details, similar_result, different_result, output_path)

    return {
        "similar_check": similar_result,
        "different_check": different_result,
        "detailed_results": all_details,
        "output_path": output_path,
        "summary": {
            "similar_pairs_tested": len(similar_pairs),
            "different_pairs_tested": len(different_pairs),
            "similar_passed": sum(1 for d in similar_details if d.get("check_result") == "PASS"),
            "similar_failed": sum(1 for d in similar_details if d.get("check_result") == "FAIL"),
            "different_passed": sum(1 for d in different_details if d.get("check_result") == "PASS"),
            "different_failed": sum(1 for d in different_details if d.get("check_result") == "FAIL"),
            "skipped": sum(1 for d in all_details if d.get("check_result") == "SKIPPED"),
        },
    }


def _export_results_excel(
    all_details: List[Dict],
    similar_result: Dict,
    different_result: Dict,
    output_path: str,
) -> None:
    """Export the final results to an Excel workbook."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        return

    wb = Workbook()

    # Styles
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    pass_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fail_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    skip_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )

    result_fills = {"PASS": pass_fill, "FAIL": fail_fill, "SKIPPED": skip_fill}

    def style_header(ws, num_cols):
        for col in range(1, num_cols + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
            cell.border = thin_border

    # ── Sheet 1: Detailed Results ─────────────────────────────────
    ws = wb.active
    ws.title = "Pair Check Results"

    headers = [
        "ID A", "ID B", "Source Sheet",
        "Auto Expected", "LLM Verdict", "LLM Reason",
        "Final Expected", "Auto Reason",
        "Cosine Similarity", "Check Result",
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    for d in all_details:
        row = [
            d["id_a"], d["id_b"], d.get("sheet", ""),
            d.get("expected", ""), d.get("llm_verdict", ""),
            d.get("llm_reason", ""), d.get("final_expected", ""),
            d.get("auto_reason", ""),
            d.get("cosine_similarity", ""),
            d.get("check_result", ""),
        ]
        ws.append(row)

    # Color-code the Check Result column
    result_col = len(headers)  # last column
    for row_idx in range(2, len(all_details) + 2):
        cell = ws.cell(row=row_idx, column=result_col)
        fill = result_fills.get(str(cell.value), skip_fill)
        cell.fill = fill

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 45)

    # ── Sheet 2: Summary ──────────────────────────────────────────
    ws_sum = wb.create_sheet("Summary")
    summary_headers = ["Metric", "Value"]
    ws_sum.append(summary_headers)
    style_header(ws_sum, 2)

    # Similar pairs summary
    sim_detail = similar_result.get("detail", {})
    diff_detail = different_result.get("detail", {})

    summary_rows = [
        ["SIMILAR PAIRS CHECK", ""],
        ["Overall Result", "PASS" if similar_result.get("passed") else "FAIL" if similar_result.get("passed") is False else "INFO"],
        ["Pairs Checked", sim_detail.get("pairs_checked", 0) if isinstance(sim_detail, dict) else 0],
        ["Pairs Failed", sim_detail.get("pairs_failed", 0) if isinstance(sim_detail, dict) else 0],
        ["k (neighbors checked)", sim_detail.get("k", "?") if isinstance(sim_detail, dict) else "?"],
        ["", ""],
        ["DIFFERENT PAIRS CHECK", ""],
        ["Overall Result", "PASS" if different_result.get("passed") else "FAIL" if different_result.get("passed") is False else "INFO"],
        ["Pairs Checked", diff_detail.get("pairs_checked", 0) if isinstance(diff_detail, dict) else 0],
        ["Pairs Failed", diff_detail.get("pairs_failed", 0) if isinstance(diff_detail, dict) else 0],
        ["k (neighbors checked)", diff_detail.get("k", "?") if isinstance(diff_detail, dict) else "?"],
        ["", ""],
        ["WORKFLOW", ""],
        ["Phase 1", "Auto-generated pairs from metadata"],
        ["Phase 2", "LLM review (if LLM Verdict column is filled)"],
        ["Phase 3", "Neighbor consistency checks (this file)"],
    ]

    for row in summary_rows:
        ws_sum.append(row)

    # Color-code pass/fail rows
    for row_idx in range(2, ws_sum.max_row + 1):
        val = str(ws_sum.cell(row=row_idx, column=2).value)
        if val == "PASS":
            ws_sum.cell(row=row_idx, column=2).fill = pass_fill
        elif val == "FAIL":
            ws_sum.cell(row=row_idx, column=2).fill = fail_fill

    ws_sum.column_dimensions['A'].width = 30
    ws_sum.column_dimensions['B'].width = 50

    wb.save(output_path)


def _run_from_csv_fallback(
    connector: VectorDBConnector,
    input_path: str,
    k_similar: int,
    k_different: int,
) -> Dict:
    """Fallback for when openpyxl is not installed — read CSV instead."""
    import csv

    csv_path = input_path.replace(".xlsx", ".csv")
    similar_pairs = []
    different_pairs = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                final = row.get("final_expected") or row.get("expected_relationship", "")
                id_a = row.get("id_a", "")
                id_b = row.get("id_b", "")
                if not id_a or not id_b:
                    continue
                if final.upper() == "SIMILAR":
                    similar_pairs.append((id_a, id_b))
                elif final.upper() == "DIFFERENT":
                    different_pairs.append((id_a, id_b))
    except FileNotFoundError:
        return {"error": f"File not found: {csv_path}", "similar_check": {}, "different_check": {}}

    from checks.neighbor_consistency import (
        check_known_similar_pairs,
        check_known_different_pairs,
    )

    return {
        "similar_check": check_known_similar_pairs(connector, similar_pairs, k=k_similar),
        "different_check": check_known_different_pairs(connector, different_pairs, k=k_different),
        "detailed_results": [],
        "output_path": csv_path.replace(".csv", "_results.csv"),
        "summary": {
            "similar_pairs_tested": len(similar_pairs),
            "different_pairs_tested": len(different_pairs),
        },
    }
