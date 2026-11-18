from __future__ import annotations

import json
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from mock_aut import MockAUTEngine, QueryRequest, engine, query_endpoint


class MockAutApiTests(unittest.IsolatedAsyncioTestCase):
    def test_engine_prompt_uses_question_and_full_retrieved_record(self) -> None:
        question = "What is the policy term?"
        context = {
            "id": "chunk-1",
            "text": "The policy term is one year.",
            "distance": 0.1,
            "metadata": {
                "pdf_name": "policy.pdf",
                "chunk_index": 3,
                "section_path": "Coverage/Term",
            },
        }
        azure_responses = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"The policy term is two years."}')
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=(
                    '{"answer":"The candidate answer is correct. '
                    'The policy term is one year."}'
                ))
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"The policy term is one year."}')
            )]),
        ]
        azure_client = Mock()
        openai_module = ModuleType("openai")
        openai_module.AzureOpenAI = Mock(return_value=azure_client)
        query_engine = MockAUTEngine()

        with (
            patch.dict(sys.modules, {"openai": openai_module}),
            patch.dict(os.environ, {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://example.test",
                "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
                "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
            }),
            patch.object(
                query_engine,
                "_create_answer_completion",
                side_effect=azure_responses,
            ) as answer_completion,
            patch.object(query_engine, "_get_project_collection") as get_collection,
        ):
            result = query_engine.query(
                question,
                "project-1",
                top_k=1,
                retrieved_contexts=[context],
            )

        get_collection.assert_not_called()
        self.assertEqual(answer_completion.call_count, 3)
        initial_messages = answer_completion.call_args_list[0].args[2]
        self.assertIn(
            "Every factual claim must be directly supported",
            initial_messages[0]["content"],
        )
        self.assertIn("untrusted document data", initial_messages[0]["content"])
        prompt_payload = json.loads(
            initial_messages[1]["content"].split("retrieved records:\n", 1)[1]
        )
        self.assertEqual(prompt_payload, {
            "question": question,
            "retrieved_records": [{
                "id": "chunk-1",
                "text": "The policy term is one year.",
                "metadata": {
                    "pdf_name": "policy.pdf",
                    "chunk_index": 3,
                    "section_path": "Coverage/Term",
                },
            }],
        })
        verification_messages = answer_completion.call_args_list[1].args[2]
        self.assertIn(
            "candidate answer may be wrong",
            verification_messages[0]["content"],
        )
        verification_payload = json.loads(
            verification_messages[1]["content"].split("JSON input:\n", 1)[1]
        )
        self.assertEqual(verification_payload, {
            "question": question,
            "candidate_answer": "The policy term is two years.",
            "retrieved_records": [{
                "id": "chunk-1",
                "text": "The policy term is one year.",
                "metadata": {
                    "pdf_name": "policy.pdf",
                    "chunk_index": 3,
                    "section_path": "Coverage/Term",
                },
            }],
        })
        retry_messages = answer_completion.call_args_list[2].args[2]
        self.assertIn(
            "response describes answer evaluation",
            retry_messages[-1]["content"],
        )
        self.assertEqual(result["answer"], "The policy term is one year.")
        self.assertTrue(result["metadata"]["answer_generated_by_llm"])
        self.assertEqual(
            result["metadata"]["answer_prompt_version"],
            "direct-answer-v9",
        )
        self.assertEqual(result["metadata"]["retrieval_source"], "provided")

    def test_uses_matching_figure_caption_when_generation_returns_fallback(self) -> None:
        question = "What does Figure 1 depict?"
        context = {
            "id": "figure-1",
            "text": "Section: Artificial Intelligence\nFigure 1: RAG pipeline architecture",
            "distance": 0.1,
            "metadata": {"pdf_name": "overview.pdf"},
        }
        azure_responses = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=(
                    '{"answer":"The provided document excerpts do not contain '
                    'enough information to answer this question."}'
                ))
            )]),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=(
                    '{"answer":"RAG pipeline architecture"}'
                ))
            )]),
        ]
        azure_client = Mock()
        openai_module = ModuleType("openai")
        openai_module.AzureOpenAI = Mock(return_value=azure_client)
        query_engine = MockAUTEngine()

        with (
            patch.dict(sys.modules, {"openai": openai_module}),
            patch.dict(os.environ, {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://example.test",
                "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
                "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
            }),
            patch.object(
                query_engine,
                "_create_answer_completion",
                side_effect=azure_responses,
            ) as answer_completion,
            patch.object(query_engine, "_get_project_collection"),
        ):
            result = query_engine.query(
                question,
                "project-1",
                top_k=1,
                retrieved_contexts=[context],
            )

        self.assertEqual(result["answer"], "RAG pipeline architecture")
        self.assertEqual(answer_completion.call_count, 2)
        verification_payload = json.loads(
            answer_completion.call_args_list[1].args[2][1]["content"].split(
                "JSON input:\n",
                1,
            )[1]
        )
        self.assertEqual(
            verification_payload["candidate_answer"],
            "RAG pipeline architecture",
        )

    async def test_query_endpoint_passes_supplied_vector_context_to_engine(self) -> None:
        context = {
            "id": "chunk-1",
            "text": "The policy term is one year.",
            "metadata": {"pdf_name": "policy.pdf"},
        }
        response = {
            "answer": "One year.",
            "context": [context],
            "confidence": 1.0,
            "sources": ["policy.pdf"],
            "metadata": {"retrieval_source": "provided"},
        }
        with (
            patch("mock_aut.os.path.isdir", return_value=True),
            patch(
                "mock_aut.run_in_threadpool",
                new=AsyncMock(return_value=response),
            ) as run_in_threadpool,
        ):
            result = await query_endpoint(QueryRequest(
                question="What is the policy term?",
                project_id="project",
                top_k=1,
                retrieved_contexts=[context],
            ))

        self.assertEqual(result["context"], [context])
        self.assertEqual(
            run_in_threadpool.await_args.args,
            (
                engine.query,
                "What is the policy term?",
                "project",
                1,
                None,
                [{
                    **context,
                    "distance": None,
                }],
            ),
        )

    async def test_query_errors_return_actionable_upstream_details(self) -> None:
        failure = RuntimeError("The provider request failed.")
        with (
            patch("mock_aut.os.path.isdir", return_value=True),
            patch("mock_aut.run_in_threadpool", new=AsyncMock(side_effect=failure)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await query_endpoint(
                    QueryRequest(question="Question", project_id="project")
                )

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("RuntimeError", raised.exception.detail)
        self.assertIn("The provider request failed.", raised.exception.detail)
        self.assertIn("query_endpoint", raised.exception.detail)

    async def test_query_error_details_redact_azure_api_key(self) -> None:
        secret = "local-test-secret"
        failure = RuntimeError(f"Rejected key {secret}")
        with (
            patch("mock_aut.os.path.isdir", return_value=True),
            patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": secret}),
            patch("mock_aut.run_in_threadpool", new=AsyncMock(side_effect=failure)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await query_endpoint(
                    QueryRequest(question="Question", project_id="project")
                )

        self.assertIn("[REDACTED]", raised.exception.detail)
        self.assertNotIn(secret, raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
