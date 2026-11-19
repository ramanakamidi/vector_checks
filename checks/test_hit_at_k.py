from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from openpyxl import Workbook

from checks.hit_at_k import evaluate_hit_at_k, read_hit_at_k_rows
from connectors.base import VectorRecord
from services.project_vector_checks import run_project_checks


class HitAtKTests(unittest.TestCase):
    def test_evaluates_exact_text_and_image_matches_at_their_ranks(self) -> None:
        connector = Mock()
        connector.count.return_value = 80
        text_records = [
            VectorRecord(f"text-{index}", [], {"document": f"chunk {index}"})
            for index in range(1, 51)
        ]
        text_records[2].metadata["document"] = " the   expected\nchunk "
        image_records = [
            VectorRecord(f"image-{index}", [], {"image_path": f"C:\\docs\\{index}.png"})
            for index in range(1, 51)
        ]
        image_records[49].metadata["image_path"] = "C:\\docs\\target.PNG"
        table_records = [
            VectorRecord(f"table-{index}", [], {"document": f"table {index}"})
            for index in range(1, 51)
        ]
        table_records[9].metadata["document"] = "| item | value |"
        connector.knn_search.side_effect = [
            text_records,
            image_records,
            table_records,
            text_records,
        ]
        rows = [
            {
                "question": "text question",
                "object_type": "text",
                "chunk": "The expected chunk",
            },
            {
                "question": "image question",
                "object_type": "graph",
                "object_image_paths": '["folder/target.png"]',
            },
            {
                "question": "table question",
                "object_type": "table",
                "chunk": "| item | value |",
            },
            {
                "question": "missing question",
                "object_type": "text",
                "chunk": "not stored",
            },
        ]

        with patch(
            "services.local_embeddings.embed_text_query",
            return_value=[0.1, 0.2],
        ):
            result = evaluate_hit_at_k(connector, rows)

        self.assertEqual(result["detail"]["top_k"], 50)
        self.assertEqual(
            [call.kwargs["k"] for call in connector.knn_search.call_args_list],
            [50, 50, 50, 50],
        )
        question_results = result["detail"]["question_results"]
        self.assertEqual(
            question_results[0],
            {
                "row": None,
                "question": "text question",
                "type": "text",
                "rank": 3,
                "hit@1": 0,
                "hit@3": 1,
                "hit@5": 1,
                "hit@10": 1,
            },
        )
        self.assertEqual(question_results[1]["rank"], 50)
        self.assertEqual(question_results[1]["hit@10"], 0)
        self.assertEqual(question_results[2]["rank"], 10)
        self.assertEqual(question_results[2]["hit@10"], 1)
        self.assertEqual(question_results[3]["rank"], "not found")
        self.assertEqual(question_results[3]["hit@10"], 0)

    def test_reads_expected_columns_and_keeps_empty_chunks_as_unmatched_targets(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(["question", "object_type", "chunk", "object_image_paths"])
        worksheet.append(["Question?", "text", None, "[]"])
        output = BytesIO()
        workbook.save(output)

        rows = read_hit_at_k_rows(output.getvalue())

        self.assertEqual(rows[0]["question"], "Question?")
        self.assertIsNone(rows[0]["chunk"])
        self.assertEqual(rows[0]["object_image_paths"], "[]")

    def test_hit_at_k_requires_workbook_target_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a workbook"):
            run_project_checks(
                "project-1",
                "projects/project-1/chroma_data",
                ["hit_at_k"],
            )

    def test_hit_at_k_is_run_as_a_project_check(self) -> None:
        connector = Mock()
        expected_result = {
            "check": "hit_at_k",
            "passed": None,
            "detail": {"question_results": []},
        }
        rows = [{"question": "Question?", "object_type": "text", "chunk": "chunk"}]

        with (
            patch(
                "services.project_vector_checks._connect_project",
                return_value=connector,
            ),
            patch(
                "services.project_vector_checks.hit_at_k.evaluate_hit_at_k",
                return_value=expected_result,
            ) as evaluate,
        ):
            report = run_project_checks(
                "project-1",
                "projects/project-1/chroma_data",
                ["hit_at_k"],
                hit_at_k_rows=rows,
            )

        evaluate.assert_called_once_with(connector, rows)
        self.assertEqual(report["results"], [expected_result])


if __name__ == "__main__":
    unittest.main()
