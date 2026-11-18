from __future__ import annotations

from io import BytesIO
import unittest

from openpyxl import Workbook, load_workbook

from services.workbook_evaluation import (
    WorkbookEvaluationServiceError,
    evaluate_workbook,
)


class WorkbookEvaluationTests(unittest.TestCase):
    def _workbook_bytes(self) -> bytes:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(
            [
                "question",
                "expected_answer",
                "actual_answer",
                "status",
                "reason",
                "score",
            ]
        )
        worksheet.append(["Question one", "Expected one", None, None, None, None])
        worksheet.append(["Question two", "Expected two", None, None, None, None])
        output = BytesIO()
        workbook.save(output)
        return output.getvalue()

    def test_upstream_aut_failure_stops_instead_of_marking_all_rows_error(self) -> None:
        class ApiError(RuntimeError):
            status_code = 502

        calls = 0

        def query(_question: str, _project_id: str):
            nonlocal calls
            calls += 1
            raise ApiError("Mock AUT query failed")

        with self.assertRaisesRegex(
            WorkbookEvaluationServiceError,
            "stopped at row 2: Mock AUT query failed",
        ):
            evaluate_workbook(
                self._workbook_bytes(),
                "project",
                query,
                lambda *_args: self.fail("Judge must not run after AUT failure"),
            )

        self.assertEqual(calls, 1)

    def test_workbook_header_matching_is_case_insensitive(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(
            [
                "Question",
                "Expected_Answer",
                "Actual_Answer",
                "Status",
                "Reason",
                "Score",
            ]
        )
        worksheet.append(["Question one", "Expected one", None, None, None, None])
        source = BytesIO()
        workbook.save(source)

        completed, summary = evaluate_workbook(
            source.getvalue(),
            "project",
            lambda *_args: {
                "answer": "Actual one",
                "metadata": {
                    "answer_generated_by_llm": True,
                    "num_contexts_retrieved": 1,
                },
            },
            lambda *_args: {"score": 1, "reason": "Correct"},
        )

        result = load_workbook(BytesIO(completed)).active
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(result.cell(row=2, column=3).value, "Actual one")
        self.assertEqual(result.cell(row=2, column=4).value, "PASS")


if __name__ == "__main__":
    unittest.main()
