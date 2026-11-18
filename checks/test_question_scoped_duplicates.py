from __future__ import annotations

import unittest
import math
from unittest.mock import Mock, patch

from checks.duplicate_detection import find_near_duplicates_for_questions
from checks.metadata_completeness import check_retrieved_fields_per_question
from checks.vector_integrity import check_retrieved_vectors_per_question
from connectors.base import VectorRecord
from services.project_vector_checks import run_project_checks


class QuestionScopedDuplicateTests(unittest.TestCase):
    def test_index_health_is_no_longer_a_selectable_project_check(self) -> None:
        with patch("services.project_vector_checks._connect_project") as connect:
            with self.assertRaisesRegex(ValueError, "index_health"):
                run_project_checks(
                    "project-1",
                    "projects/project-1/chroma_data",
                    ["index_health"],
                )

        connect.assert_not_called()

    def test_reports_similarity_candidates_per_question(self) -> None:
        records = {
            "a": VectorRecord(
                "a",
                [1.0, 0.0],
                {"document": "shared policy wording", "pdf_name": "policy.pdf"},
            ),
            "b": VectorRecord(
                "b",
                [0.8, 0.6],
                {"document": "shared policy wording", "pdf_name": "policy.pdf"},
            ),
            "c": VectorRecord(
                "c",
                [0.0, 1.0],
                {"document": "unrelated text", "pdf_name": "other.pdf"},
            ),
        }
        connector = Mock()
        connector.fetch.side_effect = records.get

        result = find_near_duplicates_for_questions(
            connector,
            [
                {"question": "Question one?", "retrieved_ids": ["a", "b"]},
                {"question": "Question two?", "retrieved_ids": ["b", "c"]},
            ],
            minimum_similarity=0.75,
            maximum_similarity=0.9,
        )

        detail = result["detail"]
        self.assertIsNone(result["passed"])
        self.assertEqual(detail["candidate_pairs_found"], 1)
        self.assertEqual(detail["question_results"][0]["status"], "CANDIDATES")
        self.assertEqual(detail["question_results"][0]["question_index"], 0)
        self.assertEqual(
            detail["question_results"][0]["candidate_pairs"][0]["cosine_similarity"],
            0.8,
        )
        self.assertEqual(detail["question_results"][1]["status"], "CLEAR")
        self.assertEqual(detail["question_results"][1]["candidate_pairs_found"], 0)

    def test_neighbor_and_duplicate_checks_share_question_retrievals(self) -> None:
        records = {
            "a": VectorRecord("a", [1.0, 0.0], {"document": "alpha"}),
            "b": VectorRecord("b", [0.8, 0.6], {"document": "alpha"}),
        }
        connector = Mock()
        connector.fetch.side_effect = records.get
        retrievals = [{
            "question": "Question one?",
            "retrieved_ids": ["a", "b"],
            "retrieved_records": [],
            "candidate_pairs": [{"ids": ["a", "b"], "questions": ["Question one?"]}],
        }]

        with (
            patch("services.project_vector_checks._connect_project", return_value=connector),
            patch(
                "services.project_vector_checks.neighbor_consistency.retrieve_record_pairs_for_questions",
                return_value=([], retrievals),
            ) as retrieve,
            patch(
                "services.project_vector_checks.neighbor_consistency.check_known_similar_pairs",
                return_value={
                    "check": "known_similar_pairs",
                    "passed": True,
                    "detail": {
                        "pairs_checked": 1,
                        "pair_results": [{"pair": ("a", "b"), "status": "mutual"}],
                        "failures": [],
                        "unavailable_pairs": [],
                    },
                },
            ),
        ):
            report = run_project_checks(
                "project-1",
                "projects/project-1/chroma_data",
                ["neighbor_consistency", "duplicate_detection"],
                duplicate_min_similarity=0.75,
                duplicate_max_similarity=0.9,
                workbook_questions=["Question one?"],
                question_retrieval_top_k=2,
            )

        retrieve.assert_called_once()
        neighbor_result, duplicate_result = report["results"]
        self.assertEqual(
            neighbor_result["detail"]["retrievals_by_question"][0]["question_index"],
            0,
        )
        duplicate_question = duplicate_result["detail"]["question_results"][0]
        self.assertEqual(duplicate_question["question_index"], 0)
        self.assertEqual(duplicate_question["candidate_pairs_found"], 1)

    def test_incompatible_vectors_make_question_result_incomplete(self) -> None:
        connector = Mock()
        connector.fetch.side_effect = {
            "a": VectorRecord("a", [1.0, 0.0], {}),
            "b": VectorRecord("b", [1.0, 0.0, 0.0], {}),
        }.get

        result = find_near_duplicates_for_questions(
            connector,
            [{"question": "Question?", "retrieved_ids": ["a", "b"]}],
        )

        question_result = result["detail"]["question_results"][0]
        self.assertEqual(question_result["status"], "INCOMPLETE")
        self.assertEqual(question_result["pairs_compared"], 0)
        self.assertIn("different vector dimensions", question_result["unavailable_pairs"][0]["reason"])

    def test_vector_integrity_checks_retrieved_dimensions_and_vector_values(self) -> None:
        connector = Mock()
        connector.fetch.side_effect = {
            "valid": VectorRecord("valid", [1.0, 0.0], {}),
            "zero": VectorRecord("zero", [0.0, 0.0], {}),
            "non-finite": VectorRecord("non-finite", [math.inf, 1.0], {}),
        }.get

        result = check_retrieved_vectors_per_question(
            connector,
            [{
                "question_index": 0,
                "question": "Question?",
                "retrieved_ids": ["valid", "zero", "non-finite"],
            }],
        )

        question_result = result["detail"]["question_results"][0]
        self.assertEqual(result["check"], "dimension_consistency_by_question")
        self.assertEqual(result["passed"], False)
        self.assertEqual(question_result["records_checked"], 3)
        self.assertEqual(question_result["dimensions_found"], {2: 3})
        self.assertEqual(question_result["status"], "FAIL")
        self.assertEqual(len(question_result["issues"]), 2)

    def test_metadata_completeness_reports_missing_and_empty_fields_per_question(self) -> None:
        connector = Mock()
        connector.fetch.side_effect = {
            "missing": VectorRecord("missing", [1.0], {}),
            "empty": VectorRecord("empty", [1.0], {"pdf_name": "  "}),
            "complete": VectorRecord(
                "complete",
                [1.0],
                {"pdf_name": "guide.pdf", "page_number": 2},
            ),
        }.get

        result = check_retrieved_fields_per_question(
            connector,
            ["pdf_name", "page_number"],
            [{
                "question_index": 0,
                "question": "Question?",
                "retrieved_ids": ["missing", "empty", "complete"],
            }],
        )

        question_result = result["detail"]["question_results"][0]
        self.assertEqual(result["check"], "metadata_completeness_by_question")
        self.assertEqual(result["passed"], False)
        self.assertEqual(question_result["per_field"]["pdf_name"]["missing"], 1)
        self.assertEqual(question_result["per_field"]["pdf_name"]["empty"], 1)
        self.assertEqual(question_result["per_field"]["page_number"]["missing"], 2)
        self.assertEqual(question_result["total_issues"], 4)

    def test_workbook_integrity_checks_retrieve_three_records(self) -> None:
        connector = Mock()
        retrievals = [{
            "question": "Question?",
            "retrieved_ids": ["a", "b", "c"],
        }]
        vector_result = {"check": "dimension_consistency_by_question", "passed": True}
        metadata_result = {"check": "metadata_completeness_by_question", "passed": True}

        with (
            patch("services.project_vector_checks._connect_project", return_value=connector),
            patch(
                "services.project_vector_checks.neighbor_consistency.retrieve_record_pairs_for_questions",
                return_value=([], retrievals),
            ) as retrieve,
            patch(
                "services.project_vector_checks.vector_integrity.check_retrieved_vectors_per_question",
                return_value=vector_result,
            ) as vector_check,
            patch(
                "services.project_vector_checks.metadata_completeness.check_retrieved_fields_per_question",
                return_value=metadata_result,
            ) as metadata_check,
            patch("services.project_vector_checks.vector_integrity.run_all") as global_vector_check,
            patch("services.project_vector_checks.metadata_completeness.check_required_fields") as global_metadata_check,
        ):
            report = run_project_checks(
                "project-1",
                "projects/project-1/chroma_data",
                ["vector_integrity", "metadata_completeness"],
                workbook_questions=["Question?"],
            )

        retrieve.assert_called_once_with(connector, ["Question?"], 3)
        vector_check.assert_called_once_with(connector, retrievals)
        metadata_check.assert_called_once()
        global_vector_check.assert_not_called()
        global_metadata_check.assert_not_called()
        self.assertEqual(report["results"], [vector_result, metadata_result])


if __name__ == "__main__":
    unittest.main()
