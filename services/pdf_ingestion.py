"""Extract PDF content with Docling and persist embedded chunks in Chroma."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from services.paths import PROJECTS_DIR


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TABLE_ROW_OVERLAP = 1
EMBEDDING_TOKEN_BUDGET = 240
EMBEDDING_BATCH_SIZE = 128
INGESTION_VERSION = 4


def _preferred_chunk_end(text: str, start: int, limit: int) -> int:
    window = text[start:limit]
    boundaries = [
        start + match.end()
        for match in re.finditer(r"(?<=[.!?])\s+|\n+", window)
    ]
    if boundaries:
        return boundaries[-1]
    boundary = text.rfind(" ", start, limit)
    return boundary if boundary > start else limit


def _chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    token_budget: int = EMBEDDING_TOKEN_BUDGET,
) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if (
        not text
        or size < 1
        or overlap < 0
        or overlap >= size
        or token_budget < 1
    ):
        return []

    from services.local_embeddings import count_embedding_tokens

    def safe_end(start: int, limit: int) -> int:
        low = start + 1
        high = limit
        best = start
        while low <= high:
            middle = (low + high) // 2
            if count_embedding_tokens(text[start:middle]) <= token_budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == start:
            return min(start + 1, limit)
        sentence_boundary = _preferred_chunk_end(text, start, best)
        return sentence_boundary if sentence_boundary > start else best

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if (
            end < len(text)
            or count_embedding_tokens(text[start:end]) > token_budget
        ):
            end = safe_end(start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        if next_start < end:
            next_boundary = text.find(" ", next_start, end)
            if next_boundary >= 0:
                next_start = next_boundary + 1
        start = next_start
    return chunks


def _chunk_table(text: str, size: int = CHUNK_SIZE) -> list[str]:
    from services.local_embeddings import count_embedding_tokens

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    separator_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "|" in line and "-" in line and re.fullmatch(r"[\s|:-]+", line)
        ),
        None,
    )
    if separator_index is None or separator_index == 0:
        return _chunk_text(text, size=size)

    header_text = "\n".join(lines[: separator_index + 1])
    body_rows = lines[separator_index + 1 :]
    if not body_rows:
        return [header_text]

    available = size - len(header_text) - 1
    header_token_count = count_embedding_tokens(header_text + "\n")
    if available < 2 or header_token_count >= EMBEDDING_TOKEN_BUDGET:
        return _chunk_text(text, size=size)
    row_token_budget = max(
        1,
        EMBEDDING_TOKEN_BUDGET - header_token_count - 4,
    )

    chunks: list[str] = []
    body: list[str] = []
    body_length = 0
    for row in body_rows:
        row_chunks = (
            [row]
            if len(row) <= available
            and count_embedding_tokens(row) <= row_token_budget
            else _chunk_text(
                row,
                size=available,
                overlap=min(CHUNK_OVERLAP, available - 1),
                token_budget=row_token_budget,
            )
        )
        for row_chunk in row_chunks:
            row_length = len(row_chunk) + (1 if body else 0)
            candidate_body = "\n".join([*body, row_chunk])
            candidate = f"{header_text}\n{candidate_body}"
            if body and (
                body_length + row_length > available
                or count_embedding_tokens(candidate) > EMBEDDING_TOKEN_BUDGET
            ):
                chunks.append(f"{header_text}\n" + "\n".join(body))
                body = body[-TABLE_ROW_OVERLAP:]
                body_length = sum(len(item) for item in body) + max(
                    0, len(body) - 1
                )
                row_length = len(row_chunk)
                if body:
                    row_length += 1
            body.append(row_chunk)
            body_length += row_length
    if body:
        chunks.append(f"{header_text}\n" + "\n".join(body))
    return chunks


def _docling_segments(document: Any) -> list[tuple[int | None, str, str]]:
    from docling.datamodel.document import PictureItem, TableItem

    segments: list[tuple[int | None, str, str]] = []
    section_stack: list[tuple[int, str]] = []
    for item, _level in document.iterate_items(traverse_pictures=True):
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document)
        else:
            text = getattr(item, "text", "") or ""

        if isinstance(item, PictureItem) and not text.strip():
            from pdf_workbench.custom_logic import get_object_caption

            text = get_object_caption(document, item) or ""

        text = text.strip()
        if not text:
            continue

        label = getattr(getattr(item, "label", None), "value", None)
        element_type = str(label or "text")
        if element_type in {"title", "section_header"}:
            numbered_heading = re.match(r"^(\d+(?:\.\d+)*)\s+", text)
            level = (
                len(numbered_heading.group(1).split("."))
                if numbered_heading
                else 1
            )
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, text))
        elif section_stack:
            text = "Section: " + " > ".join(
                title for _, title in section_stack
            ) + "\n" + text

        provenance = getattr(item, "prov", None) or []
        page_number = getattr(provenance[0], "page_no", None) if provenance else None
        segments.append((page_number, element_type, text))
    return segments


def _structure_chunks(
    segments: list[tuple[int | None, str, str]],
) -> list[tuple[int | None, str, str, str]]:
    """Group narrative elements by page/section while keeping each table distinct."""
    chunks: list[tuple[int | None, str, str, str]] = []
    buffered_text: list[str] = []
    buffered_page: int | None = None
    buffered_section = ""

    def flush_text() -> None:
        if not buffered_text:
            return
        text = "\n".join(buffered_text)
        if buffered_section:
            text = f"Section: {buffered_section}\n{text}"
        chunks.extend(
            (buffered_page, "text", chunk, buffered_section)
            for chunk in _chunk_text(text)
        )
        buffered_text.clear()

    for page_number, element_type, text in segments:
        section_match = re.match(r"^Section:\s*(.+)\n", text)
        section = section_match.group(1).strip() if section_match else ""
        content = text[section_match.end():] if section_match else text

        if element_type.lower() == "table":
            flush_text()
            chunks.extend(
                (page_number, "table", chunk, section)
                for chunk in _chunk_table(text)
            )
            continue

        if element_type in {"title", "section_header"}:
            flush_text()
            section = section or content.strip()
            content = ""

        if buffered_text and (
            buffered_page != page_number or buffered_section != section
        ):
            flush_text()

        if not buffered_text:
            buffered_page = page_number
            buffered_section = section
        if content.strip():
            buffered_text.append(content.strip())

    flush_text()
    return chunks


def ingest_pdf(project_id: str, filename: str, content: bytes) -> dict[str, Any]:
    """Parse, chunk, embed, and store one PDF; skip an identical PDF by hash."""
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF"):
        raise ValueError(f"{filename!r} is not a valid PDF upload.")

    project_dir = PROJECTS_DIR / project_id
    if not (project_dir / "config.json").is_file():
        raise FileNotFoundError(f"Project {project_id!r} not found.")

    pdf_hash = hashlib.sha256(content).hexdigest()

    import chromadb

    chroma_path = project_dir / "chroma_data"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_name = f"{project_id}_knowledge_base"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    existing_ids: list[str] = []
    existing_complete = True
    existing_current_version = True
    offset = 0
    while True:
        existing = collection.get(
            where={"pdf_sha256": pdf_hash},
            limit=500,
            offset=offset,
            include=["metadatas"],
        )
        batch_ids = existing.get("ids") or []
        if not batch_ids:
            break
        existing_ids.extend(batch_ids)
        existing_metadata = existing.get("metadatas") or []
        existing_complete = existing_complete and all(
            isinstance(metadata, dict)
            and metadata.get("ingestion_status") == "complete"
            for metadata in existing_metadata
        ) and len(existing_metadata) == len(batch_ids)
        existing_current_version = existing_current_version and all(
            isinstance(metadata, dict)
            and metadata.get("ingestion_version") == INGESTION_VERSION
            for metadata in existing_metadata
        ) and len(existing_metadata) == len(batch_ids)
        offset += len(batch_ids)

    if existing_ids and existing_complete and existing_current_version:
        return {
            "status": "already_ingested",
            "pdf_name": Path(filename).name,
            "pdf_sha256": pdf_hash,
            "chunks_ingested": 0,
            "collection_name": collection_name,
        }
    if existing_ids:
        collection.delete(ids=existing_ids)

    source_dir = project_dir / "source_pdfs"
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        Path(filename).name,
    ).strip(" .")
    if not safe_name:
        raise ValueError("The uploaded PDF filename is invalid.")
    stored_path = source_dir / f"{pdf_hash[:12]}_{safe_name}"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", dir=source_dir, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        from pdf_workbench.custom_logic import get_converter

        conversion = get_converter().convert(temp_path)
        conversion_status = getattr(conversion.status, "value", conversion.status)
        if str(conversion_status).lower() == "failure":
            error_details = "; ".join(
                str(getattr(error, "error_message", error))
                for error in getattr(conversion, "errors", [])
            )
            raise ValueError(
                f"Docling could not convert {safe_name!r}"
                + (f": {error_details}" if error_details else ".")
            )
        document = conversion.document
        segments = _docling_segments(document)
        document_name = str(getattr(document, "name", "") or safe_name)
        origin = getattr(document, "origin", None)
        source_mime_type = str(getattr(origin, "mimetype", "") or "application/pdf")
        source_binary_hash = str(getattr(origin, "binary_hash", "") or pdf_hash)
        page_count = len(getattr(document, "pages", {}) or {})
        chunks = _structure_chunks(segments)
        if not chunks:
            raise ValueError(
                f"Docling extracted no indexable text from {safe_name!r}."
            )

        from services.local_embeddings import embed_texts

        uploaded_at = datetime.now(timezone.utc).isoformat()
        all_ids: list[str] = []
        complete_metadata: list[dict[str, Any]] = []
        for offset in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[offset : offset + EMBEDDING_BATCH_SIZE]
            documents = [chunk for _, _, chunk, _ in batch]
            embeddings = embed_texts(documents)
            ids = [
                f"{pdf_hash[:24]}_{index:06d}"
                for index in range(offset, offset + len(batch))
            ]
            metadatas = [
                {
                    "pdf_name": safe_name,
                    "source_pdf": safe_name,
                    "pdf_sha256": pdf_hash,
                    "doc_id": pdf_hash,
                    "docling_document_name": document_name,
                    "source_mime_type": source_mime_type,
                    "source_binary_hash": source_binary_hash,
                    "page_count": page_count,
                    "chunk_index": offset + index,
                    "page_number": page_number if page_number is not None else -1,
                    "object_type": element_type,
                    "chunk_type": element_type,
                    "section_path": section_path,
                    "uploaded_at": uploaded_at,
                    "ingestion_status": "in_progress",
                    "ingestion_version": INGESTION_VERSION,
                }
                for index, (page_number, element_type, _chunk, section_path) in enumerate(batch)
            ]
            all_ids.extend(ids)
            complete_metadata.extend(
                {**metadata, "ingestion_status": "complete"}
                for metadata in metadatas
            )
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=[
                    [float(value) for value in embedding]
                    for embedding in embeddings
                ],
                metadatas=metadatas,
            )

        os.replace(temp_path, stored_path)
        temp_path = None
        for offset in range(0, len(all_ids), 5000):
            collection.update(
                ids=all_ids[offset : offset + 5000],
                metadatas=complete_metadata[offset : offset + 5000],
            )
        return {
            "status": "ingested",
            "pdf_name": safe_name,
            "pdf_sha256": pdf_hash,
            "chunks_ingested": len(chunks),
            "collection_name": collection_name,
            "extraction_status": str(conversion_status),
            "page_count": page_count,
            "elements_extracted": len(segments),
            "tables_extracted": sum(
                element_type.lower() == "table"
                for _, element_type, _ in segments
            ),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
