"""Evaluate Q&A rows in an uploaded Excel workbook."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Callable
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


PASS_SCORE_THRESHOLD = 0.6
REQUIRED_COLUMNS = {
    "question",
    "expected_answer",
    "actual_answer",
    "status",
    "reason",
    "score",
}


class WorkbookEvaluationServiceError(RuntimeError):
    """A shared AUT or judge service failure that prevents row evaluation."""


def evaluate_workbook(
    workbook_bytes: bytes,
    project_id: str,
    query_fn: Callable[[str, str], dict[str, Any]],
    judge_fn: Callable[[str, str, str], dict[str, Any]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Run each question through the AUT, judge its answer, and return a completed workbook."""
    try:
        workbook = load_workbook(BytesIO(workbook_bytes))
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError("Upload a valid .xlsx workbook.") from exc
    worksheet = workbook.active
    header_columns: dict[str, int] = {}
    for column in range(1, worksheet.max_column + 1):
        value = worksheet.cell(row=1, column=column).value
        if isinstance(value, str) and value.strip():
            header = value.strip().lower()
            if header in header_columns:
                raise ValueError(f"Workbook contains a duplicate '{header}' column.")
            header_columns[header] = column

    missing_columns = sorted(REQUIRED_COLUMNS - header_columns.keys())
    if missing_columns:
        raise ValueError(
            "Workbook is missing required column(s): " + ", ".join(missing_columns)
        )

    question_column = header_columns["question"]
    expected_column = header_columns["expected_answer"]
    actual_column = header_columns["actual_answer"]
    status_column = header_columns["status"]
    reason_column = header_columns["reason"]
    score_column = header_columns["score"]
    rows = [
        row
        for row in range(2, worksheet.max_row + 1)
        if str(worksheet.cell(row=row, column=question_column).value or "").strip()
    ]
    summary: dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "processed": 0,
        "error_details": [],
    }

    for index, row in enumerate(rows, start=1):
        question = str(worksheet.cell(row=row, column=question_column).value).strip()
        expected_answer = worksheet.cell(row=row, column=expected_column).value
        worksheet.cell(row=row, column=actual_column).value = None
        worksheet.cell(row=row, column=status_column).value = None
        worksheet.cell(row=row, column=reason_column).value = None
        worksheet.cell(row=row, column=score_column).value = None
        try:
            if expected_answer is None or not str(expected_answer).strip():
                raise ValueError("Expected answer is empty.")

            response = query_fn(question, project_id)
            metadata = response.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("answer_generated_by_llm") is not True:
                raise ValueError(
                    "The running Mock AUT is outdated and did not confirm LLM answer "
                    "generation. Click 'Start Mock AUT' to restart it with the current "
                    "code, then run the workbook again."
                )
            if metadata.get("num_contexts_retrieved", 0) < 1:
                raise ValueError("Mock AUT did not retrieve any document chunks.")
            actual_answer = response.get("answer")
            if not isinstance(actual_answer, str) or not actual_answer.strip():
                raise ValueError("Mock AUT returned an empty answer.")
            worksheet.cell(row=row, column=actual_column).value = actual_answer.strip()

            judgment = judge_fn(question, str(expected_answer), actual_answer.strip())
            score = judgment.get("score")
            reason = judgment.get("reason")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("LLM judge returned a non-numeric score.")
            if not 0 <= score <= 1:
                raise ValueError("LLM judge score must be between 0 and 1.")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("LLM judge returned an empty reason.")

            status = "PASS" if score >= PASS_SCORE_THRESHOLD else "FAIL"
            worksheet.cell(row=row, column=status_column).value = status
            worksheet.cell(row=row, column=reason_column).value = reason.strip()
            worksheet.cell(row=row, column=score_column).value = float(score)
            summary["passed" if status == "PASS" else "failed"] += 1
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and status_code >= 500:
                raise WorkbookEvaluationServiceError(
                    f"Workbook evaluation stopped at row {row}: {exc}"
                ) from exc
            error_message = f"{type(exc).__name__}: {exc}"
            worksheet.cell(row=row, column=status_column).value = "ERROR"
            worksheet.cell(row=row, column=reason_column).value = error_message
            worksheet.cell(row=row, column=score_column).value = None
            summary["errors"] += 1
            summary["error_details"].append(
                {"row": row, "question": question, "error": error_message}
            )

        summary["processed"] += 1
        if progress_callback:
            progress_callback(index, len(rows))

    output = BytesIO()
    workbook.save(output)
    return output.getvalue(), summary
