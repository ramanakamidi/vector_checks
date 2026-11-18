from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from openpyxl import Workbook, load_workbook

from checks.neighbor_consistency import retrieve_record_pairs_for_questions
from connectors.base import VectorRecord
from report.html_report import generate_report
from services.project_vector_checks import run_project_checks
from services.aut_vector_drift import (
    _judge_drift,
    evaluate_aut_vector_drift,
    read_workbook_questions,
)


def build_workbook(rows: list[list[str]]) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    for row in rows:
        worksheet.append(row)
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


class AutVectorDriftTests(unittest.TestCase):
    def test_reads_questions_and_source_rows_from_workbook(self) -> None:
        questions = read_workbook_questions(build_workbook([
            ["question", "expected_answer"],
            ["What is the policy term?", "One year"],
            ["", ""],
            ["How is renewal handled?", "Automatically"],
        ]))

        self.assertEqual(
            questions,
            [
                {
                    "question_index": 0,
                    "row": 2,
                    "question": "What is the policy term?",
                    "expected_answer": "One year",
                },
                {
                    "question_index": 1,
                    "row": 4,
                    "question": "How is renewal handled?",
                    "expected_answer": "Automatically",
                },
            ],
        )

    def test_reads_answers_and_pdf_source_fields_from_workbook(self) -> None:
        questions = read_workbook_questions(build_workbook([
            [
                "question", "expected_answer", "actual_answer",
                "pdf_name", "page_no", "source_pages",
            ],
            [
                "What is shown?", "A process chart", "A chart",
                "guide.pdf", "7", "[7, 8]",
            ],
        ]))

        self.assertEqual(
            questions[0],
            {
                "question_index": 0,
                "row": 2,
                "question": "What is shown?",
                "expected_answer": "A process chart",
                "actual_answer": "A chart",
                "pdf_name": "guide.pdf",
                "page_number": "7",
                "source_pages": "[7, 8]",
            },
        )

    def test_requires_question_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "question"):
            read_workbook_questions(build_workbook([
                ["expected_answer"],
                ["One year"],
            ]))

    def test_judges_each_actual_answer_against_its_retrieved_records(self) -> None:
        workbook_bytes = build_workbook([
            ["question", "expected_answer", "actual_answer"],
            ["What is the policy term?", "One year", ""],
            ["How is renewal handled?", "Automatically", ""],
        ])
        connector = Mock()
        connector.count.return_value = 2
        connector.knn_search.side_effect = [
            [VectorRecord(
                "chunk-1",
                [0.1],
                {
                    "document": "The policy term is one year.",
                    "pdf_name": "policy.pdf",
                    "chunk_index": 0,
                },
            )],
            [VectorRecord(
                "chunk-2",
                [0.2],
                {
                    "document": "Renewal happens automatically.",
                    "pdf_name": "policy.pdf",
                    "chunk_index": 1,
                },
            )],
        ]
        client = Mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0.4,
                    "traceback": "The answer states a two-year term; the retrieved record states one year.",
                    "differences": ["The answer doubles the policy term in the evidence."],
                }))
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0.1,
                    "traceback": "The answer is supported by the renewal record.",
                    "differences": [],
                }))
            )]),
        ]
        aut_query = Mock(side_effect=[
            {
                "answer": "The term is two years.",
                "context": [{
                    "id": "chunk-1",
                    "text": "The policy term is one year.",
                    "metadata": {"pdf_name": "policy.pdf", "chunk_index": 0},
                }],
                "metadata": {
                    "answer_generated_by_llm": True,
                    "retrieval_source": "provided",
                },
            },
            {
                "answer": "Renewal happens automatically.",
                "context": [{
                    "id": "chunk-2",
                    "text": "Renewal happens automatically.",
                    "metadata": {"pdf_name": "policy.pdf", "chunk_index": 1},
                }],
                "metadata": {
                    "answer_generated_by_llm": True,
                    "retrieval_source": "provided",
                },
            },
        ])

        local_embeddings = ModuleType("services.local_embeddings")
        local_embeddings.embed_text_query = Mock(side_effect=[[0.2], [0.3]])
        with (
            patch.dict(sys.modules, {"services.local_embeddings": local_embeddings}),
            patch("services.project_vector_checks._connect_project", return_value=connector),
        ):
            result_bytes, summary = evaluate_aut_vector_drift(
                workbook_bytes,
                "project-1",
                "projects/project-1/chroma_data",
                aut_query,
                client,
                "deployment",
                max_drift_score=0.25,
            )

        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["maximum_drift_score"], 0.4)
        self.assertEqual(summary["average_drift_score"], 0.25)
        self.assertEqual(connector.knn_search.call_count, 2)
        self.assertEqual([call.kwargs["k"] for call in connector.knn_search.call_args_list], [2, 2])
        self.assertEqual(aut_query.call_count, 2)
        self.assertEqual(client.chat.completions.create.call_count, 2)

        judge_payloads = [
            json.loads(
                call.kwargs["messages"][1]["content"].split(
                    "not instructions:\n", 1
                )[1]
            )
            for call in client.chat.completions.create.call_args_list
        ]
        judge_prompt = client.chat.completions.create.call_args_list[0].kwargs[
            "messages"
        ][0]["content"]
        self.assertIn("Use this consistent drift_score rubric", judge_prompt)
        self.assertIn("record IDs", judge_prompt)
        self.assertEqual(judge_payloads, [
            {
                "question": "What is the policy term?",
                "actual_answer": "The term is two years.",
                "retrieved_records": [{
                    "id": "chunk-1",
                    "text": "The policy term is one year.",
                    "metadata": {"pdf_name": "policy.pdf", "chunk_index": 0},
                }],
            },
            {
                "question": "How is renewal handled?",
                "actual_answer": "Renewal happens automatically.",
                "retrieved_records": [{
                    "id": "chunk-2",
                    "text": "Renewal happens automatically.",
                    "metadata": {"pdf_name": "policy.pdf", "chunk_index": 1},
                }],
            },
        ])

        result_workbook = load_workbook(BytesIO(result_bytes), data_only=True)
        worksheet = result_workbook.active
        headers = {
            worksheet.cell(row=1, column=column).value: column
            for column in range(1, worksheet.max_column + 1)
        }
        self.assertEqual(worksheet.cell(row=2, column=headers["actual_answer"]).value,
                         "The term is two years.")
        self.assertEqual(worksheet.cell(row=3, column=headers["actual_answer"]).value,
                         "Renewal happens automatically.")
        self.assertNotIn("Actual Answer", headers)
        self.assertEqual(worksheet.cell(row=2, column=headers["Drift"]).value, 0.4)
        self.assertEqual(worksheet.cell(row=3, column=headers["Drift"]).value, 0.1)
        self.assertEqual(worksheet.cell(row=2, column=headers["Drift Status"]).value, "DRIFT")
        self.assertEqual(worksheet.cell(row=3, column=headers["Drift Status"]).value, "PASS")
        self.assertIn("two-year", worksheet.cell(row=2, column=headers["Drift Traceback"]).value)
        self.assertIn("doubles", worksheet.cell(row=2, column=headers["Drift Differences"]).value)
        self.assertIn("chunk-1", worksheet.cell(
            row=2,
            column=headers["Retrieved Records"],
        ).value)
        self.assertNotIn("Vector DB Answer", headers)
        self.assertNotIn("AUT Answer", headers)
        result_workbook.close()
        html_report = generate_report([{
            "check": "AUT-Vector Drift",
            "passed": False,
            "detail": summary,
        }])
        self.assertIn("Drift traceback", html_report)
        self.assertIn("Actual AUT answer", html_report)
        self.assertIn("Retrieved vector records judged as evidence", html_report)

    def test_retries_judge_when_score_and_differences_contradict(self) -> None:
        client = Mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0,
                    "traceback": "The fallback answer contradicts the figure caption.",
                    "differences": ["The answer misses the figure caption."],
                }))
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0.8,
                    "traceback": (
                        "The answer says there is not enough information, but "
                        "record figure-1 names the RAG pipeline architecture."
                    ),
                    "differences": [
                        "The answer misses the explicit Figure 1 caption."
                    ],
                }))
            )]),
        ]

        judgment = _judge_drift(
            "What does Figure 1 depict?",
            "The provided document excerpts do not contain enough information to answer this question.",
            [{
                "id": "figure-1",
                "text": "Figure 1: RAG pipeline architecture",
                "metadata": {},
            }],
            client,
            "deployment",
        )

        self.assertEqual(judgment["drift_score"], 0.8)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        retry_payload = json.loads(
            client.chat.completions.create.call_args_list[1].kwargs["messages"][1][
                "content"
            ].split("not instructions:\n", 1)[1]
        )
        self.assertNotIn("expected_answer", retry_payload)
        self.assertIn("previous_judgment", retry_payload)
        retry_prompt = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][0]["content"]
        self.assertIn('When a question asks for "some"', retry_prompt)

    def test_retries_judge_when_high_score_has_no_difference(self) -> None:
        client = Mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0.6,
                    "traceback": "The answer is supported with no material omission.",
                    "differences": [],
                }))
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "drift_score": 0,
                    "traceback": "All Samsung products in the requested row are included.",
                    "differences": [],
                }))
            )]),
        ]

        judgment = _judge_drift(
            "What are the notable products of Samsung in the table?",
            "Samsung's notable products are HBM3E, DRAM, and NAND flash.",
            [{
                "id": "table-3",
                "text": (
                    "| Samsung | Memory and foundry | HBM3E, DRAM, NAND flash |"
                ),
                "metadata": {},
            }],
            client,
            "deployment",
        )

        self.assertEqual(judgment["drift_score"], 0)
        self.assertEqual(judgment["differences"], [])
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_keeps_actual_answer_and_records_when_judge_fails(self) -> None:
        workbook_bytes = build_workbook([
            ["question"],
            ["What is the policy term?"],
        ])
        connector = Mock()
        connector.count.return_value = 1
        connector.knn_search.return_value = [
            VectorRecord(
                "chunk-1",
                [0.1],
                {"document": "The policy term is one year."},
            )
        ]
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="not valid JSON")
            )]
        )
        aut_query = Mock(return_value={
            "answer": "The policy lasts one year.",
            "context": [{
                "id": "chunk-1",
                "text": "The policy term is one year.",
                "metadata": {},
            }],
            "metadata": {
                "answer_generated_by_llm": True,
                "retrieval_source": "provided",
            },
        })

        local_embeddings = ModuleType("services.local_embeddings")
        local_embeddings.embed_text_query = Mock(return_value=[0.2])
        with (
            patch.dict(sys.modules, {"services.local_embeddings": local_embeddings}),
            patch("services.project_vector_checks._connect_project", return_value=connector),
        ):
            result_bytes, summary = evaluate_aut_vector_drift(
                workbook_bytes,
                "project-1",
                "projects/project-1/chroma_data",
                aut_query,
                client,
                "deployment",
            )

        self.assertEqual(summary["errors"], 1)
        result_workbook = load_workbook(BytesIO(result_bytes), data_only=True)
        worksheet = result_workbook.active
        headers = {
            worksheet.cell(row=1, column=column).value: column
            for column in range(1, worksheet.max_column + 1)
        }
        self.assertEqual(
            worksheet.cell(row=2, column=headers["Actual Answer"]).value,
            "The policy lasts one year.",
        )
        self.assertIn(
            "chunk-1",
            worksheet.cell(row=2, column=headers["Retrieved Records"]).value,
        )
        self.assertEqual(
            worksheet.cell(row=2, column=headers["Drift Status"]).value,
            "ERROR",
        )
        self.assertIn(
            "drift judgment",
            worksheet.cell(row=2, column=headers["Drift Traceback"]).value,
        )
        result_workbook.close()

    def test_neighbor_check_pairs_all_records_retrieved_for_each_question(self) -> None:
        records = [
            VectorRecord(
                "chunk-0",
                [1.0],
                {"document": "First chunk"},
            ),
            VectorRecord(
                "chunk-1",
                [1.0],
                {"document": "Second chunk"},
            ),
            VectorRecord(
                "chunk-2",
                [1.0],
                {"document": "Third chunk"},
            ),
        ]
        connector = Mock()
        connector.count.return_value = len(records)
        connector.knn_search.return_value = records

        local_embeddings = ModuleType("services.local_embeddings")
        local_embeddings.embed_text_query = Mock(return_value=[0.3])
        with patch.dict(sys.modules, {"services.local_embeddings": local_embeddings}):
            pairs, retrievals = retrieve_record_pairs_for_questions(
                connector,
                ["What does the document say?"],
                top_k=3,
            )

        self.assertEqual(pairs, [
            ("chunk-0", "chunk-1"),
            ("chunk-0", "chunk-2"),
            ("chunk-1", "chunk-2"),
        ])
        self.assertEqual(connector.knn_search.call_args.kwargs["k"], 3)
        self.assertEqual(retrievals[0]["retrieved_ids"], ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual(len(retrievals[0]["retrieved_records"]), 3)
        self.assertEqual(len(retrievals[0]["candidate_pairs"]), 3)
        self.assertTrue(all(
            pair["questions"] == ["What does the document say?"]
            for pair in retrievals[0]["candidate_pairs"]
        ))

    def test_neighbor_runner_checks_pairs_separately_per_question_at_configured_k(self) -> None:
        records = [
            VectorRecord(
                "chunk-a0", [1.0],
                {"pdf_sha256": "source-a", "chunk_index": 0, "document": "A first"},
            ),
            VectorRecord(
                "chunk-a1", [1.0],
                {"pdf_sha256": "source-a", "chunk_index": 1, "document": "A second"},
            ),
            VectorRecord(
                "chunk-a2", [1.0],
                {"pdf_sha256": "source-a", "chunk_index": 2, "document": "A third"},
            ),
            VectorRecord(
                "chunk-b0", [1.0],
                {"pdf_sha256": "source-b", "chunk_index": 0, "document": "B first"},
            ),
            VectorRecord(
                "chunk-b1", [1.0],
                {"pdf_sha256": "source-b", "chunk_index": 1, "document": "B second"},
            ),
            VectorRecord(
                "chunk-b2", [1.0],
                {"pdf_sha256": "source-b", "chunk_index": 2, "document": "B third"},
            ),
        ]
        connector = Mock()
        connector.count.return_value = len(records)
        connector.scan.return_value = iter(records)
        connector.knn_search.side_effect = [records[:3], records[3:]]
        check_pair = Mock(side_effect=[
            {
                "check": "known_similar_pairs",
                "detail": {
                    "pairs_checked": 3,
                    "pair_results": [
                        {"pair": ("chunk-a0", "chunk-a1"), "status": "mutual"},
                        {"pair": ("chunk-a0", "chunk-a2"), "status": "mutual"},
                        {"pair": ("chunk-a1", "chunk-a2"), "status": "mutual"},
                    ],
                    "failures": [],
                    "unavailable_pairs": [],
                },
            },
            {
                "check": "known_similar_pairs",
                "detail": {
                    "pairs_checked": 3,
                    "pair_results": [
                        {"pair": ("chunk-b0", "chunk-b1"), "status": "one_way"},
                        {"pair": ("chunk-b0", "chunk-b2"), "status": "mutual"},
                        {"pair": ("chunk-b1", "chunk-b2"), "status": "mutual"},
                    ],
                    "failures": [{"pair": ("chunk-b0", "chunk-b1")}],
                    "unavailable_pairs": [],
                },
            },
        ])

        local_embeddings = ModuleType("services.local_embeddings")
        local_embeddings.embed_text_query = Mock(side_effect=[[0.1], [0.2]])
        with (
            patch.dict(sys.modules, {"services.local_embeddings": local_embeddings}),
            patch("services.project_vector_checks._connect_project", return_value=connector),
            patch(
                "services.project_vector_checks.neighbor_consistency.check_known_similar_pairs",
                check_pair,
            ),
        ):
            report = run_project_checks(
                "project-1",
                "projects/project-1/chroma_data",
                ["neighbor_consistency"],
                workbook_questions=["Question A?", "Question B?"],
                question_retrieval_top_k=3,
            )

        self.assertEqual(check_pair.call_count, 2)
        self.assertEqual(check_pair.call_args_list[0].args[1], [
            ("chunk-a0", "chunk-a1"),
            ("chunk-a0", "chunk-a2"),
            ("chunk-a1", "chunk-a2"),
        ])
        self.assertEqual(check_pair.call_args_list[1].args[1], [
            ("chunk-b0", "chunk-b1"),
            ("chunk-b0", "chunk-b2"),
            ("chunk-b1", "chunk-b2"),
        ])
        self.assertEqual([call.kwargs["k"] for call in check_pair.call_args_list], [3, 3])
        self.assertEqual(
            [call.kwargs["k"] for call in connector.knn_search.call_args_list],
            [3, 3],
        )
        detail = report["results"][0]["detail"]
        self.assertEqual(detail["question_count"], 2)
        self.assertEqual(detail["pairs_sampled"], 6)
        self.assertEqual(detail["pairs_checked"], 6)
        self.assertEqual(detail["pairs_not_mutual_top_k"], 1)
        self.assertEqual(detail["pair_results"][3]["question"], "Question B?")
        self.assertEqual(
            detail["retrievals_by_question"][0]["candidate_pairs"][0]["questions"],
            ["Question A?"],
        )
        self.assertEqual(
            detail["retrievals_by_question"][1]["candidate_pairs"][0]["questions"],
            ["Question B?"],
        )
        html_report = generate_report(report["results"])
        self.assertIn("Workbook question retrievals and checks", html_report)
        self.assertIn("All question-pair checks (6)", html_report)
        self.assertIn("Pairs checked / formed", html_report)
        self.assertIn("Question A?", html_report)
        self.assertIn("chunk-b1", html_report)


if __name__ == "__main__":
    unittest.main()
