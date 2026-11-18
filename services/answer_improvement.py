"""Review expected answers against retrieved project Chroma context."""

from __future__ import annotations

import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from services.project_chroma import open_project_collection as _open_collection
from services.local_embeddings import embed_text_query
from services.paths import PROJECT_ROOT


QUESTION_COLUMN = "question"
EXPECTED_ANSWER_COLUMN = "expected_answer"
OBJECT_TYPE_COLUMN = "object_type"
OBJECT_IMAGE_PATHS_COLUMN = "object_image_paths"
VISUAL_OBJECT_TYPES = {"graph", "chart", "image", "figure", "picture", "photo"}
IMPROVEMENT_COLUMN = "Improvement Required?"
REASON_COLUMN = "Improvement Reason"
IMPROVED_ANSWER_COLUMN = "Improved Expected Answer"
SCAN_BATCH_SIZE = 500
MAX_RETRIEVAL_CHUNKS = 8


def _retrieve_context(
    collection: Any,
    query_vector: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    count = collection.count()
    if count < 1:
        return []
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0] or []
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    contexts = []
    for index, record_id in enumerate(ids):
        text = documents[index] if index < len(documents) else None
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = metadatas[index] if index < len(metadatas) else None
        distance = distances[index] if index < len(distances) else None
        contexts.append({
            "id": record_id,
            "text": text.strip(),
            "metadata": metadata or {},
            "distance": distance,
        })
    return contexts


def _object_image_paths(
    worksheet: Any,
    row: int,
    object_type_column: int | None,
    image_paths_column: int | None,
) -> list[str]:
    if object_type_column is None:
        return []
    object_type = str(
        worksheet.cell(row=row, column=object_type_column).value or ""
    ).strip().casefold()
    if object_type not in VISUAL_OBJECT_TYPES:
        return []
    if image_paths_column is None:
        raise ValueError(
            "Visual object workbook is missing the object_image_paths column. "
            "Download a fresh Q&A workbook from PDF Workbench."
        )

    raw_paths = worksheet.cell(row=row, column=image_paths_column).value
    try:
        relative_paths = json.loads(raw_paths) if isinstance(raw_paths, str) else raw_paths
    except json.JSONDecodeError as exc:
        raise ValueError("object_image_paths is not valid JSON.") from exc
    if (
        not isinstance(relative_paths, list)
        or not relative_paths
        or any(not isinstance(path, str) or not path.strip() for path in relative_paths)
    ):
        raise ValueError(
            "No cropped object image path was included. Download a fresh Q&A "
            "workbook from PDF Workbench."
        )

    workspace_root = PROJECT_ROOT.resolve()
    resolved_paths = []
    for relative_path in relative_paths:
        image_path = Path(relative_path)
        if image_path.is_absolute():
            raise ValueError("object_image_paths must contain workspace-relative paths.")
        resolved = (workspace_root / image_path).resolve()
        if not resolved.is_relative_to(workspace_root):
            raise ValueError("An object image path points outside the workspace.")
        if not resolved.is_file():
            raise FileNotFoundError(f"Cropped object image not found: {relative_path}")
        resolved_paths.append(str(resolved))
    return resolved_paths


def resolve_session_object_images(
    objects: list[dict[str, Any]],
    qa_by_object: dict[str, list[dict[str, Any]]],
    question: str,
    expected_answer: str,
) -> list[str]:
    """Resolve visual crops for a legacy three-argument reviewer callback."""
    visual_types = {"graph", "chart", "image", "figure", "picture", "photo"}
    matching_paths: set[tuple[str, ...]] = set()
    matched_visual_question = False
    for record in objects:
        if str(record.get("object_type", "")).casefold() not in visual_types:
            continue
        object_id = record.get("object_id")
        for pair in qa_by_object.get(object_id, []):
            if (
                str(pair.get("question", "")).strip() != question.strip()
                or str(pair.get("expected_answer", "")).strip()
                != expected_answer.strip()
            ):
                continue
            matched_visual_question = True
            paths = [
                str(Path(part["image_path"]).resolve())
                for part in (record.get("parts") or [])
                if part.get("image_path")
            ]
            if not paths and record.get("image_path"):
                paths = [str(Path(record["image_path"]).resolve())]
            if not paths:
                raise RuntimeError(
                    "The matching PDF Workbench object has no saved crop. "
                    "Regenerate the object analysis before reviewing this workbook."
                )
            matching_paths.add(tuple(paths))

    if not matched_visual_question:
        raise RuntimeError(
            "The workbook service did not provide image paths, and no matching "
            "PDF Workbench crop was found. Generate a fresh workbook from "
            "PDF Workbench and upload it again."
        )
    if len(matching_paths) != 1:
        raise RuntimeError(
            "Multiple PDF Workbench crops match this question and answer. "
            "Generate a fresh workbook from PDF Workbench to preserve "
            "row-specific image paths."
        )
    return list(next(iter(matching_paths)))


def review_workbook(
    workbook_bytes: bytes,
    project_id: str,
    top_k: int,
    review_fn: Callable[
        [str, str, list[dict[str, Any]], list[str]],
        dict[str, Any],
    ],
    progress_callback: Callable[[int, int], None] | None = None,
    aut_drift_review_fn: Callable[
        [str, str, str, str | None, str | None, list[dict[str, Any]]],
        dict[str, Any],
    ] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Review each workbook answer using retrieved project context and an LLM."""
    if top_k < 1:
        raise ValueError("The number of retrieved chunks must be at least 1.")
    initial_top_k = min(top_k, MAX_RETRIEVAL_CHUNKS)
    try:
        workbook = load_workbook(BytesIO(workbook_bytes))
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError("Upload a valid .xlsx workbook.") from exc

    worksheet = workbook.active
    header_columns: dict[str, int] = {}
    for column in range(1, worksheet.max_column + 1):
        value = worksheet.cell(row=1, column=column).value
        if isinstance(value, str) and value.strip():
            header = value.strip().casefold()
            if header in header_columns:
                raise ValueError(f"Workbook contains a duplicate {header!r} column.")
            header_columns[header] = column

    question_column = header_columns.get(QUESTION_COLUMN)
    answer_column = header_columns.get(EXPECTED_ANSWER_COLUMN)
    actual_answer_column = (
        header_columns.get("actual_answer")
        or header_columns.get("actual answer")
    )
    drift_traceback_column = header_columns.get("drift traceback")
    drift_differences_column = header_columns.get("drift differences")
    object_type_column = header_columns.get(OBJECT_TYPE_COLUMN)
    image_paths_column = header_columns.get(OBJECT_IMAGE_PATHS_COLUMN)
    missing_columns = [
        name
        for name, column in (
            (QUESTION_COLUMN, question_column),
            (EXPECTED_ANSWER_COLUMN, answer_column),
        )
        if column is None
    ]
    if missing_columns:
        raise ValueError(
            "Workbook is missing required column(s): " + ", ".join(missing_columns)
        )

    rows = [
        row
        for row in range(2, worksheet.max_row + 1)
        if str(worksheet.cell(row=row, column=question_column).value or "").strip()
    ]
    if not rows:
        raise ValueError("The workbook contains no question rows to review.")
    def output_column(name: str) -> int:
        existing = header_columns.get(name.casefold())
        if existing is not None:
            return existing
        column = worksheet.max_column + 1
        worksheet.cell(row=1, column=column).value = name
        header_columns[name.casefold()] = column
        return column

    improvement_column = output_column(IMPROVEMENT_COLUMN)
    reason_column = output_column(REASON_COLUMN)
    improved_answer_column = output_column(IMPROVED_ANSWER_COLUMN)
    collection = None
    summary: dict[str, Any] = {
        "processed": 0,
        "improvement_required": 0,
        "no_improvement": 0,
        "unable_to_verify": 0,
        "errors": 0,
        "error_details": [],
    }

    for index, row in enumerate(rows, start=1):
        question = str(worksheet.cell(row=row, column=question_column).value).strip()
        expected = worksheet.cell(row=row, column=answer_column).value
        try:
            if expected is None or not str(expected).strip():
                raise ValueError("Expected answer is empty.")
            image_paths = _object_image_paths(
                worksheet,
                row,
                object_type_column,
                image_paths_column,
            )
            expected_text = str(expected).strip()
            is_visual = bool(image_paths)
            contexts: list[dict[str, Any]] = []
            if is_visual:
                review = review_fn(question, expected_text, contexts, image_paths)
            else:
                if aut_drift_review_fn is None:
                    raise ValueError(
                        "Text and table reviews require the grounded answer reviewer."
                    )
                if collection is None:
                    collection = _open_collection(project_id)
                query_vector = embed_text_query(question)
                record_count = collection.count()
                current_k = min(initial_top_k, record_count)
                while True:
                    contexts = _retrieve_context(collection, query_vector, current_k)
                    actual_value = (
                        worksheet.cell(row=row, column=actual_answer_column).value
                        if actual_answer_column is not None
                        else None
                    )
                    traceback_value = (
                        worksheet.cell(row=row, column=drift_traceback_column).value
                        if drift_traceback_column is not None
                        else None
                    )
                    differences_value = (
                        worksheet.cell(row=row, column=drift_differences_column).value
                        if drift_differences_column is not None
                        else None
                    )
                    review = aut_drift_review_fn(
                        question,
                        expected_text,
                        str(actual_value).strip()
                        if actual_value is not None
                        else "",
                        str(traceback_value).strip()
                        if traceback_value is not None
                        and str(traceback_value).strip()
                        else None,
                        str(differences_value).strip()
                        if differences_value is not None
                        and str(differences_value).strip()
                        else None,
                        contexts,
                    )
                    evidence_sufficient = review.get("evidence_sufficient")
                    if not isinstance(evidence_sufficient, bool):
                        raise ValueError(
                            "LLM review must return a boolean evidence_sufficient."
                        )
                    if evidence_sufficient:
                        break
                    if current_k >= min(record_count, MAX_RETRIEVAL_CHUNKS):
                        review = {
                            "evidence_sufficient": False,
                            "improvement_required": False,
                            "reason": (
                                "The required information was not found in the "
                                f"retrieved records after checking up to {current_k} "
                                "chunk(s). The answer could not be verified or "
                                "improved from the available evidence."
                            ),
                            "improved_expected_answer": "",
                        }
                        break
                    current_k = min(
                        current_k * 2,
                        MAX_RETRIEVAL_CHUNKS,
                        record_count,
                    )

            evidence_sufficient = review.get("evidence_sufficient")
            if not isinstance(evidence_sufficient, bool):
                raise ValueError(
                    "LLM review must return a boolean evidence_sufficient."
                )
            if not evidence_sufficient:
                worksheet.cell(row=row, column=improvement_column).value = (
                    "Unable to verify"
                )
                worksheet.cell(row=row, column=reason_column).value = (
                    str(review.get("reason") or "").strip()
                    or (
                        "The required information could not be verified from "
                        "the supplied cropped image(s)."
                        if is_visual
                        else "The required information was not found in the "
                        "retrieved records."
                    )
                )
                worksheet.cell(row=row, column=improved_answer_column).value = None
                summary["unable_to_verify"] += 1
                summary["processed"] += 1
                if progress_callback:
                    progress_callback(index, len(rows))
                continue

            required = review.get("improvement_required")
            reason = review.get("reason")
            improved_answer = review.get("improved_expected_answer")
            if not isinstance(required, bool):
                raise ValueError("LLM review must return a boolean improvement_required.")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("LLM review returned an empty reason.")
            if required and (
                not isinstance(improved_answer, str) or not improved_answer.strip()
            ):
                raise ValueError(
                    "LLM review requires an improved expected answer but returned none."
                )

            worksheet.cell(row=row, column=improvement_column).value = (
                "Yes" if required else "No"
            )
            worksheet.cell(row=row, column=reason_column).value = reason.strip()
            worksheet.cell(row=row, column=improved_answer_column).value = (
                improved_answer.strip() if required else None
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            worksheet.cell(row=row, column=improvement_column).value = "ERROR"
            worksheet.cell(row=row, column=reason_column).value = error_message
            worksheet.cell(row=row, column=improved_answer_column).value = None
            summary["errors"] += 1
            summary["error_details"].append(
                {"row": row, "question": question, "error": error_message}
            )
        else:
            summary["improvement_required" if required else "no_improvement"] += 1

        summary["processed"] += 1
        if progress_callback:
            progress_callback(index, len(rows))

    output = BytesIO()
    workbook.save(output)
    return output.getvalue(), summary
