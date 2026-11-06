"""Session-backed operations used by the React PDF Workbench."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any

from pdf_workbench import custom_logic
from services.paths import PROJECT_ROOT


_lock = threading.RLock()
_sessions: dict[str, dict[str, Any]] = {}
_output_root = PROJECT_ROOT / "streamlit_output" / "react_sessions"
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_MAX_PDF_COUNT = 10


def _chunk_documents(
    documents: list[dict[str, Any]],
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    chunks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for document in documents:
        if document.get("error"):
            errors.append({
                "pdf_name": str(document.get("pdf_name", "")),
                "error": str(document["error"]),
            })
            continue
        pages = document.get("page_markdowns") or []
        if pages:
            for page in pages:
                page_number = int(page["page_no"])
                page_chunks = custom_logic.chunk_document_text(
                    page.get("markdown", ""),
                    document["pdf_name"],
                    f"{document['document_id']}:page-{page_number}",
                    **options,
                )
                for chunk in page_chunks:
                    chunk["page_no"] = page_number
                    chunk["pages"] = [page_number]
                chunks.extend(page_chunks)
        else:
            chunks.extend(
                custom_logic.chunk_document_text(
                    document.get("markdown", ""),
                    document["pdf_name"],
                    document["document_id"],
                    **options,
                )
            )
    return chunks, errors


def _image_url(session_id: str, object_id: str, part: int) -> str:
    return (
        f"/api/workbench/{session_id}/objects/"
        f"{object_id}/image?part={part}"
    )


def _public_object(session_id: str, record: dict[str, Any]) -> dict[str, Any]:
    parts = record.get("parts") or []
    return {
        "object_id": record.get("object_id"),
        "pdf_name": record.get("doc_name"),
        "page_no": record.get("page_no"),
        "pages": record.get("pages") or [record.get("page_no")],
        "object_type": record.get("object_type"),
        "object_no": record.get("object_no"),
        "position": record.get("position"),
        "title": record.get("docling_title"),
        "characteristics": record.get("characteristics") or {},
        "image_urls": [
            _image_url(session_id, record["object_id"], index)
            for index in range(len(parts))
        ],
    }


def _public_session(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "objects": [
            _public_object(session_id, record)
            for record in state["objects"]
        ],
        "text_chunks": state["text_chunks"],
        "text_errors": state["text_errors"],
        "chunking": state["chunking"],
        "qa_records": custom_logic.build_qa_records(
            state["objects"], state["qa_by_object"], state["text_qa"]
        ),
    }


def analyze_pdfs(
    uploads: list[tuple[str, bytes]],
    mode: str,
    chunking: dict[str, Any],
    credentials: dict[str, str],
) -> dict[str, Any]:
    if not uploads:
        raise ValueError("Upload at least one PDF.")
    if len(uploads) > _MAX_PDF_COUNT:
        raise ValueError(f"Upload no more than {_MAX_PDF_COUNT} PDFs at once.")
    if sum(len(content) for _, content in uploads) > _MAX_UPLOAD_BYTES:
        raise ValueError("The combined PDF upload must not exceed 100 MB.")
    if mode not in {"crop", "page", "page_context"}:
        raise ValueError("Choose a supported vision context mode.")
    if any(not name.casefold().endswith(".pdf") for name, _ in uploads):
        raise ValueError("Only PDF files are supported.")

    session_id = uuid.uuid4().hex
    session_dir = _output_root / session_id
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=False)
    pdf_paths: list[Path] = []
    for index, (filename, content) in enumerate(uploads):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        if not safe_name.casefold().endswith(".pdf"):
            safe_name += ".pdf"
        destination = upload_dir / f"{index + 1}_{safe_name}"
        destination.write_bytes(content)
        pdf_paths.append(destination)

    documents: list[dict[str, Any]] = []
    with _lock:
        custom_logic.configure_azure_openai(
            credentials["api_key"],
            credentials["endpoint"],
            credentials["deployment"],
            credentials["api_version"],
        )
        _, objects = custom_logic.process_pdf_files(
            pdf_paths,
            str(session_dir / "objects"),
            mode,
            text_documents=documents,
        )
        text_chunks, text_errors = _chunk_documents(documents, chunking)
        state = {
            "session_dir": session_dir,
            "objects": objects,
            "documents": documents,
            "text_chunks": text_chunks,
            "text_errors": text_errors,
            "chunking": dict(chunking),
            "mode": mode,
            "qa_by_object": {},
            "text_qa": [],
        }
        _sessions[session_id] = state
        return _public_session(session_id, state)


def get_session(session_id: str) -> dict[str, Any]:
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        return _public_session(session_id, state)


def update_chunking(
    session_id: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        chunks, errors = _chunk_documents(state["documents"], options)
        state["text_chunks"] = chunks
        state["text_errors"] = errors
        state["text_qa"] = []
        state["chunking"] = dict(options)
        return _public_session(session_id, state)


def generate_qa(
    session_id: str,
    object_ids: list[str],
    generate_objects: bool,
    generate_text: bool,
    object_question_count: int,
    text_question_count: int,
    credentials: dict[str, str],
) -> dict[str, Any]:
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        if not generate_objects and not generate_text:
            raise ValueError("Choose object Q&A, text Q&A, or both.")
        objects_by_id = {
            record["object_id"]: record for record in state["objects"]
        }
        unknown_ids = set(object_ids) - objects_by_id.keys()
        if unknown_ids:
            raise ValueError("One or more selected objects are not in this session.")
        if generate_objects and not object_ids:
            raise ValueError("Select at least one object for object Q&A.")
        if generate_text and not state["text_chunks"]:
            raise ValueError("There are no text chunks to generate Q&A from.")

        custom_logic.configure_azure_openai(
            credentials["api_key"],
            credentials["endpoint"],
            credentials["deployment"],
            credentials["api_version"],
        )
        if generate_objects:
            for object_id in object_ids:
                generated = custom_logic.generate_qa_for_record(
                    objects_by_id[object_id], object_question_count
                )
                state["qa_by_object"][object_id] = generated.get("pairs", [])
        if generate_text:
            state["text_qa"] = custom_logic.generate_qa_for_text_chunks(
                state["text_chunks"], text_question_count
            )
        return _public_session(session_id, state)


def get_object_image(
    session_id: str,
    object_id: str,
    part_index: int,
) -> Path:
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        record = next(
            (item for item in state["objects"] if item.get("object_id") == object_id),
            None,
        )
        if record is None:
            raise KeyError(f"Object '{object_id}' was not found.")
        parts = record.get("parts") or []
        if not 0 <= part_index < len(parts):
            raise ValueError("Object image part is out of range.")
        image_path = Path(parts[part_index]["image_path"]).resolve()
        session_root = Path(state["session_dir"]).resolve()
        if not image_path.is_relative_to(session_root) or not image_path.is_file():
            raise FileNotFoundError("The requested object crop is unavailable.")
        return image_path


def build_workbook(session_id: str) -> bytes:
    with _lock:
        state = _sessions.get(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        if not state["qa_by_object"] and not state["text_qa"]:
            raise ValueError("Generate Q&A before downloading the workbook.")
        return custom_logic.build_excel_bytes(
            state["objects"], state["qa_by_object"], state["text_qa"]
        )


def review_workbook(
    workbook_bytes: bytes,
    project_id: str,
    top_k: int,
    credentials: dict[str, str],
) -> tuple[bytes, dict[str, Any]]:
    from pdf_workbench.answer_review import review_expected_answer
    from services.answer_improvement import review_workbook as review

    with _lock:
        custom_logic.configure_azure_openai(
            credentials["api_key"],
            credentials["endpoint"],
            credentials["deployment"],
            credentials["api_version"],
        )

        def review_answer(
            question: str,
            expected_answer: str,
            contexts: list[dict[str, Any]],
            image_paths: list[str],
        ) -> dict[str, Any]:
            return review_expected_answer(
                question,
                expected_answer,
                contexts,
                image_paths=image_paths,
                azure_client=custom_logic.client,
                deployment=custom_logic.MODEL_NAME,
            )

        return review(
            workbook_bytes,
            project_id,
            top_k,
            review_answer,
        )


def workbench_output_path(session_id: str) -> Path:
    return _output_root / session_id

