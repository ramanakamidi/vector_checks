from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from checks.off_topic_questions import generate_off_topic_questions


class OffTopicQuestionSuggestionTests(unittest.TestCase):
    def test_suggests_requested_questions_for_each_unique_topic(self) -> None:
        responses = [
            {"questions": ["How does auto insurance work?", "What affects a premium?"]},
            {"questions": ["How does compound interest work?", "What is a savings account?"]},
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **request: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=json.dumps(responses.pop(0)),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        )

        with patch("openai.AzureOpenAI", return_value=client):
            suggestions = generate_off_topic_questions(
                ["Insurance", "insurance", "Personal finance"],
                2,
                "test-key",
                "https://example.test",
                "test-deployment",
                "test-version",
            )

        self.assertEqual(
            suggestions,
            [
                {
                    "topic": "Insurance",
                    "questions": [
                        "How does auto insurance work?",
                        "What affects a premium?",
                    ],
                },
                {
                    "topic": "Personal finance",
                    "questions": [
                        "How does compound interest work?",
                        "What is a savings account?",
                    ],
                },
            ],
        )

    def test_rejects_empty_topics(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 10 topics"):
            generate_off_topic_questions(
                [],
                2,
                "test-key",
                "https://example.test",
                "test-deployment",
                "test-version",
            )

    def test_rejects_question_counts_outside_supported_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 10 questions"):
            generate_off_topic_questions(
                ["Insurance"],
                11,
                "test-key",
                "https://example.test",
                "test-deployment",
                "test-version",
            )


if __name__ == "__main__":
    unittest.main()
