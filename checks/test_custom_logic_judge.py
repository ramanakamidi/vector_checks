from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pdf_workbench import custom_logic


class JudgeAnswerTests(unittest.TestCase):
    def test_judge_can_use_request_scoped_client(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"score": 1, "reason": "The answer is complete."}'
                    )
                )
            ]
        )

        with (
            patch.object(custom_logic, "client", None),
            patch.object(custom_logic, "MODEL_NAME", None),
        ):
            result = custom_logic.judge_answer(
                "Question",
                "Expected",
                "Actual",
                client_override=client,
                model_name="deployment",
            )

        self.assertEqual(result, {"score": 1.0, "reason": "The answer is complete."})
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "deployment",
        )


if __name__ == "__main__":
    unittest.main()
