"""Match PDF Workbench Q&A rows to selected project vector chunks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


_STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "before",
    "being", "between", "but", "can", "could", "does", "during", "each",
    "for", "from", "had", "has", "have", "into", "its", "may", "more",
    "most", "not", "our", "over", "same", "should", "than", "that", "the",
    "their", "there", "these", "they", "this", "those", "through", "under",
    "was", "were", "what", "when", "where", "which", "while", "with", "would",
}


def load_workbench_qa_workbook(content: bytes) -> list[dict[str, Any]]:
    """Read table and text Q&A rows from a PDF Workbench Excel workbook."""
    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError("Upload a valid PDF Workbench .xlsx workbook.") from exc

    worksheet = workbook.active
    header_columns: dict[str, int] = {}
    for column, value in enumerate(
        next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), ()),
        start=1,
    ):
        if isinstance(value, str) and value.strip():
            header = value.strip().casefold()
            if header in header_columns:
                raise ValueError(f"Workbook contains a duplicate {header!r} column.")
            header_columns[header] = column - 1

    required = {"question", "expected_answer"}
    missing = required - header_columns.keys()
    if missing:
        raise ValueError(
            "Workbook is missing required column(s): " + ", ".join(sorted(missing))
        )

    rows = []
    for values in worksheet.iter_rows(min_row=2, values_only=True):
        question = str(values[header_columns["question"]] or "").strip()
        answer = str(values[header_columns["expected_answer"]] or "").strip()
        if not question or not answer:
            continue

        object_type = _cell_text(values, header_columns.get("object_type")).casefold()
        source_type = _cell_text(values, header_columns.get("source_type")).casefold()
        if source_type == "object":
            source_type = "table" if object_type == "table" else ""
        elif not source_type:
            source_type = "table" if object_type == "table" else "text" if not object_type else ""
        if source_type not in {"table", "text"}:
            continue

        rows.append({
            "pdf_name": _cell_text(values, header_columns.get("pdf_name")),
            "page_no": _page_number(
                _cell_value(values, header_columns.get("page_no"))
            ),
            "source_pages": _parse_pages(
                _cell_value(values, header_columns.get("source_pages"))
            ),
            "object_type": object_type,
            "object_id": _cell_text(values, header_columns.get("object_id")),
            "source_type": source_type,
            "source_id": _cell_text(values, header_columns.get("source_id")),
            "source_section": _cell_text(values, header_columns.get("source_section")),
            "source_text": _cell_text(values, header_columns.get("source_text")),
            "category": _cell_text(values, header_columns.get("category")),
            "characteristics": _cell_text(
                values,
                header_columns.get("characteristics"),
            ),
            "description": _cell_text(values, header_columns.get("description")),
            "question": question,
            "expected_answer": answer,
        })
    workbook.close()
    return rows


def match_chunk_qa(
    point: dict[str, Any],
    qa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only workbook Q&A rows associated with a selected text/table point."""
    metadata = point.get("metadata")
    if not isinstance(metadata, dict):
        return []

    chunk_type = str(
        metadata.get("chunk_type") or metadata.get("object_type") or ""
    ).strip().casefold()
    if chunk_type not in {"text", "table"}:
        return []

    point_pdf = _filename(
        metadata.get("pdf_name") or metadata.get("source_pdf")
    )
    if not point_pdf:
        return []
    same_pdf = [
        row for row in qa_rows
        if row["source_type"] == chunk_type
        and _filename(row.get("pdf_name")) == point_pdf
    ]
    if not same_pdf:
        return []

    if chunk_type == "table":
        page = _page_number(metadata.get("page_number"))
        if page is None or page < 1:
            return []
        on_page = [
            row for row in same_pdf
            if page in (row.get("source_pages") or [])
            or page == row.get("page_no")
        ]
        if not on_page:
            return []
        groups = _group_rows(on_page, "object_id", "page_no")
        if len(groups) == 1:
            return next(iter(groups.values()))

        point_text = str(point.get("text") or "")
        scored_groups = [
            (
                _content_similarity(
                    point_text,
                    " ".join(
                        f"{row.get('characteristics', '')} "
                        f"{row.get('description', '')} "
                        f"{row.get('question', '')} "
                        f"{row.get('expected_answer', '')}"
                        for row in group
                    ),
                ),
                group,
            )
            for group in groups.values()
        ]
        scored_groups.sort(key=lambda item: item[0], reverse=True)
        if not scored_groups or scored_groups[0][0] < 0.2:
            return []
        if (
            len(scored_groups) > 1
            and scored_groups[0][0] - scored_groups[1][0] < 0.05
        ):
            return []
        return scored_groups[0][1]

    point_text = str(point.get("text") or "")
    point_section = _normalized(metadata.get("section_path"))
    groups = _group_rows(same_pdf, "source_id", "source_section")
    scored_groups: list[tuple[float, list[dict[str, Any]]]] = []
    for group in groups.values():
        source_text = next(
            (str(row.get("source_text") or "") for row in group if row.get("source_text")),
            "",
        )
        if not source_text:
            continue
        score = _content_similarity(point_text, source_text)
        row_section = _normalized(group[0].get("source_section"))
        if point_section and row_section and point_section == row_section:
            score += 0.05
        scored_groups.append((score, group))

    scored_groups.sort(key=lambda item: item[0], reverse=True)
    if not scored_groups or scored_groups[0][0] < 0.55:
        return []
    if (
        len(scored_groups) > 1
        and scored_groups[0][0] - scored_groups[1][0] < 0.03
    ):
        return []
    return scored_groups[0][1]


def _cell_value(values: tuple[Any, ...], index: int | None) -> Any:
    return values[index] if index is not None and index < len(values) else None


def _cell_text(values: tuple[Any, ...], index: int | None) -> str:
    value = _cell_value(values, index)
    return str(value).strip() if value is not None else ""


def _page_number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        page = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return page if page > 0 else None


def _parse_pages(value: Any) -> list[int]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        pages = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(pages, list):
        return []
    return [page for item in pages if (page := _page_number(item)) is not None]


def _filename(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).name.casefold()


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", str(value or "").casefold())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _content_similarity(left: str, right: str) -> float:
    normalized_left = _normalized(left)
    normalized_right = _normalized(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    denominator = min(len(left_tokens), len(right_tokens))
    if denominator == 0:
        return 0.0
    return len(left_tokens & right_tokens) / denominator


def _group_rows(
    rows: list[dict[str, Any]],
    primary_key: str,
    fallback_key: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get(primary_key) or "").strip()
        if not group_id:
            group_id = (
                f"{row.get('pdf_name', '')}|{row.get(fallback_key, '')}|"
                f"{row.get('page_no', '')}|{row.get('source_text', '')[:80]}"
            )
        groups[group_id].append(row)
    return dict(groups)
