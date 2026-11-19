"""Per-question Hit@k evaluation for workbook-grounded retrieval."""

from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any
from zipfile import BadZipFile

from connectors.base import VectorDBConnector, VectorRecord
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


TOP_K = 50
HIT_KS = (1, 3, 5, 10)
VISUAL_TYPES = {"image", "graph"}


def read_hit_at_k_rows(workbook_bytes: bytes) -> list[dict[str, Any]]:
    """Read the fields needed to identify each question's expected record."""
    try:
        workbook = load_workbook(
            BytesIO(workbook_bytes),
            read_only=True,
            data_only=True,
        )
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError("Upload a valid .xlsx workbook.") from exc

    try:
        worksheet = workbook.active
        columns: dict[str, int] = {}
        for column in range(1, worksheet.max_column + 1):
            value = worksheet.cell(row=1, column=column).value
            if isinstance(value, str) and value.strip():
                header = value.strip().casefold()
                if header in columns:
                    raise ValueError(
                        f"Workbook contains a duplicate '{header}' column."
                    )
                columns[header] = column

        required_columns = {
            "question",
            "object_type",
            "chunk",
            "object_image_paths",
        }
        missing_columns = sorted(required_columns - columns.keys())
        if missing_columns:
            raise ValueError(
                "Hit@k requires workbook column(s): "
                + ", ".join(missing_columns)
            )

        rows = []
        for row in range(2, worksheet.max_row + 1):
            question = worksheet.cell(row=row, column=columns["question"]).value
            if question is None or not str(question).strip():
                continue
            rows.append({
                "row": row,
                "question": str(question).strip(),
                "object_type": str(
                    worksheet.cell(row=row, column=columns["object_type"]).value
                    or ""
                ).strip(),
                "chunk": worksheet.cell(row=row, column=columns["chunk"]).value,
                "object_image_paths": worksheet.cell(
                    row=row,
                    column=columns["object_image_paths"],
                ).value,
            })
    finally:
        workbook.close()

    if not rows:
        raise ValueError("Workbook contains no non-empty questions.")
    return rows


def evaluate_hit_at_k(
    connector: VectorDBConnector,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retrieve the top 50 records for each question and report its target rank."""
    from services.local_embeddings import embed_text_query

    record_count = connector.count()
    if record_count < 1:
        raise ValueError("The project's vector collection is empty.")

    question_results = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError("Hit@k workbook rows must contain a question.")

        embedding = embed_text_query(question)
        vector = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )
        retrieved = connector.knn_search(vector, k=min(TOP_K, record_count))
        rank = _find_rank(retrieved, row)
        question_result: dict[str, Any] = {
            "row": row.get("row"),
            "question": question,
            "type": str(row.get("object_type") or ""),
            "rank": rank if rank is not None else "not found",
        }
        question_result.update({
            f"hit@{k}": int(rank is not None and rank <= k)
            for k in HIT_KS
        })
        question_results.append(question_result)

    return {
        "check": "hit_at_k",
        "passed": None,
        "detail": {
            "top_k": TOP_K,
            "question_count": len(question_results),
            "question_results": question_results,
            "methodology_note": (
                "Each workbook question is embedded with the project's query "
                "embedding model, then matched against the top 50 mixed text, "
                "table, image, and graph records. Text and table chunks use "
                "normalized exact text matching; images and graphs use the "
                "expected image filename. No averages are calculated."
            ),
        },
    }


def _find_rank(
    records: list[VectorRecord],
    row: dict[str, Any],
) -> int | None:
    object_type = str(row.get("object_type") or "").strip().casefold()
    if object_type in VISUAL_TYPES:
        expected_names = _image_filenames(row.get("object_image_paths"))
        if not expected_names:
            return None
        for rank, record in enumerate(records, start=1):
            image_path = (record.metadata or {}).get("image_path")
            if _image_filename(image_path) in expected_names:
                return rank
        return None

    expected_chunk = _clean_text(row.get("chunk"))
    if not expected_chunk:
        return None
    for rank, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        actual_chunk = metadata.get("document") or metadata.get("text")
        if _clean_text(actual_chunk) == expected_chunk:
            return rank
    return None


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _image_filenames(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = re.findall(
                r"""[^\\/"']+\.(?:png|jpe?g)""",
                value,
                flags=re.IGNORECASE,
            )
    else:
        parsed = value
    paths = parsed if isinstance(parsed, list) else [parsed]
    return {
        filename
        for filename in (_image_filename(path) for path in paths)
        if filename
    }


def _image_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.split(r"[\\/]", value.strip())[-1].casefold()
