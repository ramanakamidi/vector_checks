"""Generate off-topic questions for negative-query checks."""

from __future__ import annotations

import json
from typing import Any


def generate_off_topic_questions(
    topics: list[str],
    count_per_topic: int,
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str,
) -> list[dict[str, Any]]:
    if not topics or len(topics) > 10:
        raise ValueError("Enter between 1 and 10 topics.")
    if isinstance(count_per_topic, bool) or not 1 <= count_per_topic <= 10:
        raise ValueError("Generate between 1 and 10 questions per topic.")

    normalized_topics: list[str] = []
    seen_topics: set[str] = set()
    for topic in topics:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topics cannot be empty.")
        topic = topic.strip()
        if len(topic) > 120:
            raise ValueError("Each topic must be at most 120 characters.")
        key = " ".join(topic.casefold().split())
        if key not in seen_topics:
            seen_topics.add(key)
            normalized_topics.append(topic)

    if len(normalized_topics) > 10:
        raise ValueError("Enter no more than 10 unique topics.")
    missing_settings = [
        name
        for name, value in (
            ("Azure OpenAI API key", api_key),
            ("Azure OpenAI endpoint", endpoint),
            ("Azure OpenAI deployment", deployment),
            ("Azure OpenAI API version", api_version),
        )
        if not isinstance(value, str) or not value.strip()
    ]
    if missing_settings:
        raise ValueError(
            "Configure " + ", ".join(missing_settings) + " before suggesting questions."
        )

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    results: list[dict[str, Any]] = []
    for topic in normalized_topics:
        request: dict[str, Any] = {
            "model": deployment,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate natural questions about the user's topic. "
                        "These questions will be tested as off-topic against an "
                        "unrelated document collection. Do not mention or assume "
                        "anything about the document collection. Return exactly "
                        "the requested number of distinct questions and no answers. "
                        "Return JSON with one property, questions, containing an "
                        "array of strings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate exactly {count_per_topic} questions about this "
                        f"JSON-encoded topic: {json.dumps(topic, ensure_ascii=False)}"
                    ),
                },
            ],
            "temperature": 0.7,
            "max_tokens": min(2000, count_per_topic * 120),
            "response_format": {"type": "json_object"},
        }
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message:
                raise
            request.pop("response_format")
            response = client.chat.completions.create(**request)

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Azure OpenAI returned no questions for topic {topic!r}.")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Azure OpenAI returned invalid question JSON for topic {topic!r}."
            ) from exc
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list) or len(questions) != count_per_topic:
            raise RuntimeError(
                f"Azure OpenAI must return exactly {count_per_topic} questions for topic {topic!r}."
            )

        normalized_questions: list[str] = []
        seen_questions: set[str] = set()
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                raise RuntimeError(
                    f"Azure OpenAI returned an empty or invalid question for topic {topic!r}."
                )
            question = question.strip()
            key = " ".join(question.casefold().split())
            if key in seen_questions:
                raise RuntimeError(
                    f"Azure OpenAI returned duplicate questions for topic {topic!r}."
                )
            seen_questions.add(key)
            normalized_questions.append(question)
        results.append({"topic": topic, "questions": normalized_questions})
    return results
