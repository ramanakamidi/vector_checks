"""
LLM-Assisted Pair Review (Phase 2).

Reads the auto-generated pairs Excel file from Phase 1, sends each pair's
metadata to an LLM for semantic review, and writes back the LLM's verdict
and reasoning into the Excel file.

The LLM acts as a "second opinion" — it reads the metadata (doc_name,
page_number, object_type, etc.) and decides whether two records SHOULD
be semantically close or not, providing a reason.

This module is LLM-agnostic: it accepts a callable ``llm_fn`` that takes
a prompt string and returns a response string. You can plug in OpenAI,
Anthropic, a local model, or any other LLM.

Usage::

    from checks.llm_review import review_pairs_with_llm

    # Define your LLM function (any callable that takes str → str)
    def my_llm(prompt: str) -> str:
        import openai
        resp = openai.ChatCompletion.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    review_pairs_with_llm(
        input_path="auto_pairs.xlsx",
        output_path="reviewed_pairs.xlsx",
        llm_fn=my_llm,
    )
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

# openpyxl imported lazily


def _build_review_prompt(pair: Dict) -> str:
    """Build a structured prompt for the LLM to review a single pair.

    The prompt gives the LLM:
      - The two record IDs
      - Their metadata (doc_name, page_number, object_type, etc.)
      - The auto-generated expected relationship and reason
      - The actual cosine similarity measured from the database
      - Clear instructions on what to output
    """
    meta_a_str = "\n".join(f"    {k}: {v}" for k, v in pair.get("metadata_a", {}).items() if v)
    meta_b_str = "\n".join(f"    {k}: {v}" for k, v in pair.get("metadata_b", {}).items() if v)

    return f"""You are reviewing vector database record pairs to determine if they should be semantically similar or different.

RECORD A (ID: {pair['id_a']}):
  Metadata:
{meta_a_str}

RECORD B (ID: {pair['id_b']}):
  Metadata:
{meta_b_str}

AUTO-GENERATED ASSESSMENT:
  Expected Relationship: {pair['expected_relationship']}
  Reason: {pair['reason']}
  Actual Cosine Similarity: {pair['actual_cosine_similarity']}

TASK:
Based on the metadata above, determine whether these two records SHOULD be semantically close in embedding space.

Consider:
- Do they come from the same document or topic?
- Are they about the same subject matter?
- Would a user searching for one expect to find the other?
- Does the actual cosine similarity match what you'd expect?

Respond in EXACTLY this JSON format (no other text):
{{
  "verdict": "CLOSE" or "NOT_CLOSE" or "UNCERTAIN",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "reason": "Your one-sentence explanation"
}}"""


def _parse_llm_response(response: str) -> Dict[str, str]:
    """Parse the LLM's JSON response, handling common formatting issues."""
    # Try to extract JSON from the response
    # LLMs sometimes wrap JSON in markdown code blocks
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "verdict": str(parsed.get("verdict", "UNCERTAIN")).upper(),
                "confidence": str(parsed.get("confidence", "LOW")).upper(),
                "reason": str(parsed.get("reason", "Could not parse LLM response")),
            }
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract verdict from plain text
    response_upper = response.upper()
    if "CLOSE" in response_upper and "NOT_CLOSE" not in response_upper:
        verdict = "CLOSE"
    elif "NOT_CLOSE" in response_upper or "NOT CLOSE" in response_upper:
        verdict = "NOT_CLOSE"
    else:
        verdict = "UNCERTAIN"

    return {
        "verdict": verdict,
        "confidence": "LOW",
        "reason": response[:200] if response else "No response from LLM",
    }


def review_pairs_with_llm(
    input_path: str = "auto_pairs.xlsx",
    output_path: Optional[str] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    sheets_to_review: Optional[List[str]] = None,
    batch_size: int = 10,
    skip_already_reviewed: bool = True,
) -> Dict:
    """Read the auto-generated pairs Excel, review each with an LLM,
    and write back the verdicts.

    Args:
        input_path: Path to the Excel file from Phase 1.
        output_path: Where to save the reviewed file. If None, overwrites
            the input file.
        llm_fn: A callable that takes a prompt string and returns a
            response string. If None, a placeholder function is used
            that marks all pairs as UNCERTAIN (useful for testing the
            workflow without an actual LLM).
        sheets_to_review: Which sheets to review. Default: both
            "Similar Pairs" and "Different Pairs".
        batch_size: How many pairs to review before saving progress.
            Useful for long-running reviews — if interrupted, already-
            reviewed pairs are preserved.
        skip_already_reviewed: If True, skip pairs that already have
            an LLM Verdict filled in (for resuming interrupted reviews).

    Returns:
        A summary dict with counts of reviewed pairs and verdicts.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import PatternFill
    except ImportError:
        return {
            "error": "openpyxl is required for LLM review. Install with: pip install openpyxl",
            "reviewed": 0,
        }

    if output_path is None:
        output_path = input_path

    if llm_fn is None:
        llm_fn = _placeholder_llm

    if sheets_to_review is None:
        sheets_to_review = ["Similar Pairs", "Different Pairs"]

    wb = load_workbook(input_path)

    # ── Styles for verdicts ───────────────────────────────────────
    close_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    not_close_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    uncertain_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")

    verdict_fills = {
        "CLOSE": close_fill,
        "NOT_CLOSE": not_close_fill,
        "UNCERTAIN": uncertain_fill,
    }

    total_reviewed = 0
    total_skipped = 0
    verdict_counts = {"CLOSE": 0, "NOT_CLOSE": 0, "UNCERTAIN": 0}
    overrides = 0  # cases where LLM disagrees with auto-generated expectation

    for sheet_name in sheets_to_review:
        if sheet_name not in wb.sheetnames:
            continue

        ws = wb[sheet_name]
        headers = [cell.value for cell in ws[1]]

        # Find column indices
        col_map = {h: i for i, h in enumerate(headers) if h}
        id_a_col = col_map.get("ID A", 0)
        id_b_col = col_map.get("ID B", 1)
        group_field_col = col_map.get("Group Field", 2)
        group_val_a_col = col_map.get("Group Value A", 3)
        group_val_b_col = col_map.get("Group Value B", 4)
        expected_col = col_map.get("Expected Relationship", 5)
        reason_col = col_map.get("Reason", 6)
        cosine_col = col_map.get("Actual Cosine Similarity", 7)
        verdict_col = col_map.get("LLM Verdict", 8)
        llm_reason_col = col_map.get("LLM Reason", 9)
        final_col = col_map.get("Final Expected", 10)

        # Find metadata columns
        meta_a_cols = {h: i for i, h in enumerate(headers) if h and h.startswith("A: ")}
        meta_b_cols = {h: i for i, h in enumerate(headers) if h and h.startswith("B: ")}

        reviewed_in_batch = 0

        for row_idx in range(2, ws.max_row + 1):
            # Skip if already reviewed
            existing_verdict = ws.cell(row=row_idx, column=verdict_col + 1).value
            if skip_already_reviewed and existing_verdict:
                total_skipped += 1
                continue

            # Build pair dict for the prompt
            pair = {
                "id_a": ws.cell(row=row_idx, column=id_a_col + 1).value or "",
                "id_b": ws.cell(row=row_idx, column=id_b_col + 1).value or "",
                "group_field": ws.cell(row=row_idx, column=group_field_col + 1).value or "",
                "group_value_a": ws.cell(row=row_idx, column=group_val_a_col + 1).value or "",
                "group_value_b": ws.cell(row=row_idx, column=group_val_b_col + 1).value or "",
                "expected_relationship": ws.cell(row=row_idx, column=expected_col + 1).value or "",
                "reason": ws.cell(row=row_idx, column=reason_col + 1).value or "",
                "actual_cosine_similarity": ws.cell(row=row_idx, column=cosine_col + 1).value or 0,
                "metadata_a": {},
                "metadata_b": {},
            }

            # Extract metadata from columns
            for header, col_idx in meta_a_cols.items():
                field_name = header[3:]  # strip "A: " prefix
                val = ws.cell(row=row_idx, column=col_idx + 1).value
                if val:
                    pair["metadata_a"][field_name] = val

            for header, col_idx in meta_b_cols.items():
                field_name = header[3:]  # strip "B: " prefix
                val = ws.cell(row=row_idx, column=col_idx + 1).value
                if val:
                    pair["metadata_b"][field_name] = val

            # Call LLM
            prompt = _build_review_prompt(pair)
            try:
                response = llm_fn(prompt)
                result = _parse_llm_response(response)
            except Exception as e:
                result = {
                    "verdict": "UNCERTAIN",
                    "confidence": "LOW",
                    "reason": f"LLM error: {str(e)[:100]}",
                }

            # Write results back to Excel
            ws.cell(row=row_idx, column=verdict_col + 1).value = result["verdict"]
            ws.cell(row=row_idx, column=llm_reason_col + 1).value = result["reason"]

            # Color-code the verdict cell
            fill = verdict_fills.get(result["verdict"], uncertain_fill)
            ws.cell(row=row_idx, column=verdict_col + 1).fill = fill

            # Set Final Expected based on LLM verdict
            expected = pair["expected_relationship"]
            llm_verdict = result["verdict"]

            if llm_verdict == "CLOSE":
                final = "SIMILAR"
            elif llm_verdict == "NOT_CLOSE":
                final = "DIFFERENT"
            else:
                final = expected  # keep original if uncertain

            ws.cell(row=row_idx, column=final_col + 1).value = final

            # Track overrides
            if (expected == "SIMILAR" and llm_verdict == "NOT_CLOSE") or \
               (expected == "DIFFERENT" and llm_verdict == "CLOSE"):
                overrides += 1

            verdict_counts[result["verdict"]] = verdict_counts.get(result["verdict"], 0) + 1
            total_reviewed += 1
            reviewed_in_batch += 1

            # Save progress periodically
            if reviewed_in_batch >= batch_size:
                wb.save(output_path)
                reviewed_in_batch = 0

    # Final save
    wb.save(output_path)

    return {
        "output_path": output_path,
        "total_reviewed": total_reviewed,
        "total_skipped": total_skipped,
        "verdict_counts": verdict_counts,
        "overrides": overrides,
        "note": (
            f"LLM reviewed {total_reviewed} pairs. "
            f"{overrides} overrides (LLM disagreed with auto-generated expectation). "
            f"Skipped {total_skipped} already-reviewed pairs."
        ),
    }


def _placeholder_llm(prompt: str) -> str:
    """Placeholder LLM function for testing the workflow.
    Returns UNCERTAIN for all pairs.
    """
    return json.dumps({
        "verdict": "UNCERTAIN",
        "confidence": "LOW",
        "reason": "Placeholder — no real LLM configured. Replace llm_fn with your LLM.",
    })


def create_openai_llm(model: str = "gpt-4", temperature: float = 0.1) -> Callable[[str], str]:
    """Create an LLM function using OpenAI's API.
    
    Reads the API key from the OPENAI_API_KEY environment variable.
    Requires: pip install openai
    
    Args:
        model: OpenAI model name (default: gpt-4)
        temperature: Sampling temperature (default: 0.1 for consistency)
    
    Returns:
        A callable that takes a prompt string and returns a response string.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: set OPENAI_API_KEY=sk-your-key-here"
        )
    
    def llm_fn(prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
    
    return llm_fn
