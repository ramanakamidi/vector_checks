"""Session-backed operations used by the React PDF Workbench."""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from pdf_workbench import custom_logic
from services.paths import PROJECT_ROOT


_lock = threading.RLock()
_sessions: dict[str, dict[str, Any]] = {}
_analysis_progress_lock = threading.Lock()
_analysis_progress: dict[str, dict[str, Any]] = {}
_qa_generation_progress_lock = threading.Lock()
_qa_generation_progress: dict[str, dict[str, Any]] = {}
_logger = logging.getLogger(__name__)
_output_root = PROJECT_ROOT / "ui" / "workbench_output" / "react_sessions"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_PDF_COUNT = 10


def create_analysis_job() -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    progress = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "message": "Preparing to analyze PDFs...",
    }
    with _analysis_progress_lock:
        _analysis_progress[job_id] = progress
    return dict(progress)


def get_analysis_progress(job_id: str) -> dict[str, Any]:
    with _analysis_progress_lock:
        progress = _analysis_progress.get(job_id)
        if progress is None:
            raise KeyError(f"Workbench analysis job '{job_id}' was not found.")
        return dict(progress)


def _update_analysis_progress(
    job_id: str | None,
    progress: int,
    message: str,
    status: str = "running",
) -> None:
    if job_id is None:
        return
    with _analysis_progress_lock:
        if job_id in _analysis_progress:
            _analysis_progress[job_id].update({
                "status": status,
                "progress": progress,
                "message": message,
            })


def fail_analysis_job(job_id: str, message: str) -> None:
    _update_analysis_progress(job_id, 0, message, "failed")


def create_qa_generation_job() -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    progress = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "completed": 0,
        "total": 0,
        "message": "Preparing Q&A generation...",
    }
    with _qa_generation_progress_lock:
        _qa_generation_progress[job_id] = progress
    return dict(progress)


def get_qa_generation_progress(job_id: str) -> dict[str, Any]:
    with _qa_generation_progress_lock:
        progress = _qa_generation_progress.get(job_id)
        if progress is None:
            raise KeyError(f"Q&A generation job '{job_id}' was not found.")
        return dict(progress)


def _update_qa_generation_progress(
    job_id: str,
    completed: int,
    total: int,
    message: str,
    status: str = "running",
) -> None:
    if status == "complete":
        progress = 100
    elif status == "finalizing":
        progress = 95
    else:
        progress = min(90, round(completed * 90 / total)) if total else 0
    with _qa_generation_progress_lock:
        if job_id in _qa_generation_progress:
            _qa_generation_progress[job_id].update({
                "status": status,
                "progress": progress,
                "completed": completed,
                "total": total,
                "message": message,
            })


def run_qa_generation_job(
    job_id: str,
    session_id: str,
    config: dict[str, Any],
) -> None:
    try:
        result = generate_qa(
            session_id,
            **config,
            progress_callback=lambda completed, total, message, status="running": (
                _update_qa_generation_progress(
                    job_id,
                    completed,
                    total,
                    message,
                    status,
                )
            ),
        )
    except Exception as exc:
        _logger.exception("Q&A generation job %s failed", job_id)
        current = get_qa_generation_progress(job_id)
        _update_qa_generation_progress(
            job_id,
            current["completed"],
            current["total"],
            str(exc),
            "failed",
        )
    else:
        total_pairs = len(result["qa_records"])
        current = get_qa_generation_progress(job_id)
        _update_qa_generation_progress(
            job_id,
            current["completed"],
            current["total"],
            f"Generated {total_pairs} question-and-answer pairs.",
            "complete",
        )


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
        "text_enabled": state.get(
            "text_enabled",
            bool(state.get("documents") or state.get("text_chunks")),
        ),
        "text_chunks": state["text_chunks"],
        "text_errors": state["text_errors"],
        "chunking": state["chunking"],
        "qa_records": custom_logic.build_qa_records(
            state["objects"], state["qa_by_object"], state["text_qa"]
        ),
    }


def _save_session(session_id: str, state: dict[str, Any]) -> None:
    session_dir = Path(state["session_dir"])
    manifest_path = session_dir / "session.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest:
        json.dump(state, manifest, ensure_ascii=False)
    temporary_path.replace(manifest_path)


def _get_state(session_id: str) -> dict[str, Any] | None:
    state = _sessions.get(session_id)
    if state is not None:
        return state
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        return None

    session_dir = (_output_root / session_id).resolve()
    if not session_dir.is_relative_to(_output_root.resolve()):
        return None
    manifest_path = session_dir / "session.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest:
            state = json.load(manifest)
    except FileNotFoundError:
        return None
    if not isinstance(state, dict):
        raise ValueError(f"Workbench session '{session_id}' has invalid saved data.")
    state["session_dir"] = session_dir
    _sessions[session_id] = state
    return state


def analyze_pdfs(
    uploads: list[tuple[str, bytes]],
    mode: str,
    chunking: dict[str, Any],
    credentials: dict[str, str],
    object_types: list[str] | None = None,
    progress_job_id: str | None = None,
) -> dict[str, Any]:
    if not uploads:
        raise ValueError("Upload at least one PDF.")
    if len(uploads) > MAX_PDF_COUNT:
        raise ValueError(f"Upload no more than {MAX_PDF_COUNT} PDFs at once.")
    if sum(len(content) for _, content in uploads) > MAX_UPLOAD_BYTES:
        raise ValueError("The combined PDF upload must not exceed 100 MB.")
    if mode not in {"crop", "page", "page_context"}:
        raise ValueError("Choose a supported vision context mode.")
    selected_types = set(
        {"table", "image", "graph"} if object_types is None else object_types
    )
    supported_types = {"table", "image", "graph", "text"}
    unsupported_types = selected_types - supported_types
    if unsupported_types:
        raise ValueError(f"Unsupported object type(s): {', '.join(sorted(unsupported_types))}.")
    if not selected_types:
        raise ValueError("Select at least one object type to analyze.")
    if any(not name.casefold().endswith(".pdf") for name, _ in uploads):
        raise ValueError("Only PDF files are supported.")

    session_id = uuid.uuid4().hex
    session_dir = _output_root / session_id
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=False)
    pdf_paths: list[Path] = []
    _update_analysis_progress(progress_job_id, 4, "Saving uploaded PDFs...")
    for index, (filename, content) in enumerate(uploads):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        if not safe_name.casefold().endswith(".pdf"):
            safe_name += ".pdf"
        file_dir = upload_dir / str(index + 1)
        file_dir.mkdir()
        destination = file_dir / safe_name
        destination.write_bytes(content)
        pdf_paths.append(destination)
    _update_analysis_progress(progress_job_id, 8, "Starting PDF analysis...")

    text_enabled = "text" in selected_types
    object_types = selected_types & {"table", "image", "graph"}
    documents: list[dict[str, Any]] = []
    with _lock:
        custom_logic.configure_azure_openai(
            credentials["api_key"],
            credentials["endpoint"],
            credentials["deployment"],
            credentials["api_version"],
        )
        process_options: dict[str, Any] = {}
        if progress_job_id is not None:
            process_options["progress_callback"] = lambda message, fraction: (
                _update_analysis_progress(
                    progress_job_id,
                    8 + round(fraction * 78),
                    message,
                )
            )
        _, objects = custom_logic.process_pdf_files(
            pdf_paths,
            str(session_dir / "objects"),
            mode,
            text_documents=documents if text_enabled else None,
            object_types=object_types,
            **process_options,
        )
        _update_analysis_progress(
            progress_job_id,
            89,
            "Preparing extracted text and detected objects...",
        )
        objects = [
            record
            for record in objects
            if str(record.get("object_type", "")).casefold() in selected_types
        ]
        if text_enabled:
            text_chunks, text_errors = _chunk_documents(documents, chunking)
        else:
            text_chunks, text_errors = [], []
        _update_analysis_progress(
            progress_job_id,
            96,
            "Saving analysis results...",
        )
        state = {
            "session_dir": str(session_dir),
            "objects": objects,
            "documents": documents,
            "text_enabled": text_enabled,
            "text_chunks": text_chunks,
            "text_errors": text_errors,
            "chunking": dict(chunking),
            "mode": mode,
            "qa_by_object": {},
            "text_qa": [],
        }
        _sessions[session_id] = state
        _save_session(session_id, state)
        _update_analysis_progress(
            progress_job_id,
            100,
            f"Analysis complete: {len(objects)} objects and {len(text_chunks)} text chunks.",
            "complete",
        )
        return _public_session(session_id, state)


def get_session(session_id: str) -> dict[str, Any]:
    with _lock:
        state = _get_state(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        return _public_session(session_id, state)


def update_chunking(
    session_id: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    with _lock:
        state = _get_state(session_id)
        if state is None:
            raise KeyError(f"Workbench session '{session_id}' was not found.")
        if not state.get("text_enabled", bool(state.get("documents") or state.get("text_chunks"))):
            raise ValueError("Text was not selected for this analysis. Select Text and analyze the PDFs first.")
        chunks, errors = _chunk_documents(state["documents"], options)
        state["text_chunks"] = chunks
        state["text_errors"] = errors
        state["text_qa"] = []
        state["chunking"] = dict(options)
        _save_session(session_id, state)
        return _public_session(session_id, state)


def generate_qa(
    session_id: str,
    object_ids: list[str],
    generate_objects: bool,
    generate_text: bool,
    object_question_count: int,
    text_question_count: int,
    credentials: dict[str, str],
    generate_single_chunk: bool = True,
    generate_multi_hop: bool = False,
    multi_hop_chunk_count: int = 2,
    multi_hop_question_count: int = 2,
    progress_callback=None,
) -> dict[str, Any]:
    with _lock:
        state = _get_state(session_id)
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
        if generate_text and not (generate_single_chunk or generate_multi_hop):
            raise ValueError("Select single-chunk Q&A, multi-hop Q&A, or both.")
        if generate_text and not state.get(
            "text_enabled",
            bool(state.get("documents") or state.get("text_chunks")),
        ):
            raise ValueError("Text was not selected for this analysis.")
        if generate_text and generate_single_chunk and not 1 <= text_question_count <= 3:
            raise ValueError("Text Q&A generation supports 1 to 3 questions per chunk.")
        if generate_text and generate_multi_hop:
            if not 1 <= multi_hop_question_count <= 3:
                raise ValueError("Multi-hop Q&A generation supports 1 to 3 questions per group.")
            if not 2 <= multi_hop_chunk_count <= 10:
                raise ValueError("Multi-hop Q&A requires 2 to 10 adjacent chunks per group.")

        text_chunks = state["text_chunks"] if generate_text else []
        single_chunk_count = len(text_chunks) if generate_single_chunk else 0
        multi_hop_groups = (
            custom_logic.get_adjacent_text_chunk_groups(
                text_chunks,
                multi_hop_chunk_count,
            )
            if generate_multi_hop and generate_text
            else []
        )
        if generate_multi_hop and generate_text and not multi_hop_groups:
            raise ValueError(
                f"No groups of {multi_hop_chunk_count} adjacent text chunks were found within one PDF."
            )
        object_count = len(object_ids) if generate_objects else 0
        total_steps = object_count + single_chunk_count + len(multi_hop_groups)

        def report_progress(
            completed: int,
            message: str,
            status: str = "running",
        ) -> None:
            if progress_callback:
                progress_callback(completed, total_steps, message, status)

        report_progress(0, "Starting Q&A generation...")

        custom_logic.configure_azure_openai(
            credentials["api_key"],
            credentials["endpoint"],
            credentials["deployment"],
            credentials["api_version"],
        )
        if generate_objects:
            for index, object_id in enumerate(object_ids, start=1):
                report_progress(
                    index - 1,
                    f"Generating object Q&A: {index} of {object_count}",
                )
                generated = custom_logic.generate_qa_for_record(
                    objects_by_id[object_id], object_question_count
                )
                state["qa_by_object"][object_id] = generated.get("pairs", [])
                report_progress(
                    index,
                    f"Completed object Q&A: {index} of {object_count}",
                )
        completed_before_text = object_count
        if generate_text:
            single_chunk_qa = []
            multi_hop_qa = []
            if generate_multi_hop:
                report_progress(
                    completed_before_text,
                    f"Preparing multi-hop Q&A: 0 of {len(multi_hop_groups)} groups",
                )
                multi_hop_qa = custom_logic.generate_qa_for_adjacent_text_chunks(
                    state["text_chunks"],
                    multi_hop_chunk_count,
                    multi_hop_question_count,
                    progress_callback=lambda current, total: report_progress(
                        completed_before_text + current,
                        f"Generating multi-hop Q&A: {current} of {total} groups complete",
                    ),
                )
            if generate_single_chunk:
                single_chunk_offset = completed_before_text + len(multi_hop_groups)
                report_progress(
                    single_chunk_offset,
                    f"Preparing single-chunk Q&A: 0 of {single_chunk_count} chunks",
                )
                single_chunk_qa = custom_logic.generate_qa_for_text_chunks(
                    state["text_chunks"],
                    text_question_count,
                    progress_callback=lambda current, total: report_progress(
                        single_chunk_offset + current,
                        f"Generating single-chunk Q&A: {current} of {total} chunks complete",
                    ),
                )
            state["text_qa"] = single_chunk_qa + multi_hop_qa
        report_progress(
            total_steps,
            "Saving generated Q&A to this session...",
            "finalizing",
        )
        _save_session(session_id, state)
        report_progress(
            total_steps,
            "Preparing generated Q&A results...",
            "finalizing",
        )
        return _public_session(session_id, state)


def get_object_image(
    session_id: str,
    object_id: str,
    part_index: int,
) -> Path:
    with _lock:
        state = _get_state(session_id)
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
        state = _get_state(session_id)
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

        def review_aut_drift_answer(
            question: str,
            expected_answer: str,
            actual_answer: str,
            drift_traceback: str | None,
            drift_differences: str | None,
            contexts: list[dict[str, Any]],
        ) -> dict[str, Any]:
            return review_expected_answer(
                question,
                expected_answer,
                contexts,
                aut_actual_answer=actual_answer,
                drift_traceback=drift_traceback,
                drift_differences=drift_differences,
                azure_client=custom_logic.client,
                deployment=custom_logic.MODEL_NAME,
            )

        return review(
            workbook_bytes,
            project_id,
            top_k,
            review_answer,
            aut_drift_review_fn=review_aut_drift_answer,
        )


def workbench_output_path(session_id: str) -> Path:
    return _output_root / session_id
