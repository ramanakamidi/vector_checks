"""Extract and embed PDF figures, prose, and tables in project vector stores."""

from __future__ import annotations

import hashlib
import json
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
CHUNK_TOKEN_BUDGET = 240
MAX_CHUNK_CONTENT_TOKENS = CHUNK_TOKEN_BUDGET - 2
EMBEDDING_BATCH_SIZE = (
    1
    if os.getenv("PROJECT_EMBEDDING_DEVICE", "cpu").lower() == "cpu"
    else 4
)
INGESTION_VERSION = 10
MAX_INGESTION_CHUNK_SIZE = 8000
DEFAULT_CHUNKING = {
    "technique": "sentence",
    "chunk_size": 512,
    "overlap": 0,
    "min_sentences_per_chunk": 1,
    "semantic_threshold": 0.8,
}


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
    token_budget: int = CHUNK_TOKEN_BUDGET,
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


def _chunk_prose(
    text: str,
    *,
    technique: str,
    size: int,
    overlap: int,
    min_sentences_per_chunk: int,
    semantic_threshold: float,
) -> list[str]:
    supported_techniques = {"recursive", "sentence", "token", "semantic"}
    if technique not in supported_techniques:
        raise ValueError(f"Unsupported text chunking technique: {technique}.")
    if size < 1:
        raise ValueError("Text chunk size must be greater than zero.")
    effective_size = min(size, MAX_CHUNK_CONTENT_TOKENS)
    if technique in {"recursive", "sentence", "token"} and not 0 <= overlap < effective_size:
        raise ValueError(
            "Text chunking requires chunk_size > overlap >= 0 after applying "
            f"the {MAX_CHUNK_CONTENT_TOKENS}-token chunk content limit."
        )
    if min_sentences_per_chunk < 1:
        raise ValueError("Minimum sentences per chunk must be at least one.")
    if not 0 < semantic_threshold < 1:
        raise ValueError("Semantic similarity threshold must be between 0 and 1.")

    try:
        from chonkie import RecursiveChunker, SentenceChunker, SemanticChunker, TokenChunker
    except ImportError as exc:
        raise RuntimeError(
            "Chonkie is required for the selected text chunking technique. "
            'Install it with: pip install "chonkie[semantic]"'
        ) from exc

    from services.local_embeddings import get_project_tokenizer

    tokenizer = get_project_tokenizer()
    if technique == "semantic":
        from pdf_workbench.custom_logic import _get_chonkie_semantic_embedding_model

        chunker = SemanticChunker(
            embedding_model=_get_chonkie_semantic_embedding_model(),
            chunk_size=effective_size,
            threshold=semantic_threshold,
            min_sentences_per_chunk=min_sentences_per_chunk,
        )
    else:
        if technique == "recursive":
            chunker = RecursiveChunker(tokenizer=tokenizer, chunk_size=effective_size)
        elif technique == "sentence":
            chunker = SentenceChunker(
                tokenizer=tokenizer,
                chunk_size=effective_size,
                chunk_overlap=overlap,
                min_sentences_per_chunk=min_sentences_per_chunk,
            )
        else:
            chunker = TokenChunker(
                tokenizer=tokenizer,
                chunk_size=effective_size,
                chunk_overlap=overlap,
            )

    chunks: list[str] = []
    for result in chunker.chunk(text):
        chunk = str(result.text).strip()
        if not chunk:
            continue
        if _count_tokens(chunk) <= CHUNK_TOKEN_BUDGET:
            chunks.append(chunk)
        else:
            fallback_overlap = (
                overlap
                if technique in {"sentence", "token"}
                else 0
            )
            bounded_chunks = TokenChunker(
                tokenizer=tokenizer,
                chunk_size=effective_size,
                chunk_overlap=min(fallback_overlap, effective_size - 1),
            ).chunk(chunk)
            for bounded_chunk in bounded_chunks:
                bounded_text = str(bounded_chunk.text).strip()
                if not bounded_text:
                    continue
                if _count_tokens(bounded_text) > CHUNK_TOKEN_BUDGET:
                    raise RuntimeError(
                        "Chonkie produced a chunk above the configured "
                        f"{CHUNK_TOKEN_BUDGET}-token chunk budget after splitting."
                    )
                chunks.append(bounded_text)
    return chunks


def _count_tokens(text: str) -> int:
    from services.local_embeddings import count_embedding_tokens

    return count_embedding_tokens(text)


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
    if available < 2 or header_token_count >= CHUNK_TOKEN_BUDGET:
        return _chunk_text(text, size=size)
    row_token_budget = max(
        1,
        CHUNK_TOKEN_BUDGET - header_token_count - 4,
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
                or count_embedding_tokens(candidate) > CHUNK_TOKEN_BUDGET
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


def _docling_image_records(
    document: Any,
    project_dir: Path,
    pdf_hash: str,
) -> list[tuple[int | None, Path]]:
    """Save Docling picture crops and return image record tuples."""
    from docling.datamodel.document import PictureItem

    objects_dir = project_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int | None, Path]] = []
    for item, _level in document.iterate_items(traverse_pictures=True):
        if not isinstance(item, PictureItem):
            continue

        page_number: int | None = None
        provenance = getattr(item, "prov", None) or []
        if provenance:
            page_number = getattr(provenance[0], "page_no", None)

        try:
            image = item.get_image(document)
        except (AttributeError, TypeError, ValueError):
            image = getattr(getattr(item, "image", None), "pil_image", None)
        if image is None:
            continue

        if hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            continue

        index = len(records)
        page_label = page_number if page_number is not None else "unknown"
        image_path = (
            objects_dir
            / f"{pdf_hash[:12]}_page_{page_label}_image_{index:04d}.png"
        )
        image.save(image_path, format="PNG")

        records.append((page_number, image_path))
    return records


def _docling_text_records(
    document: Any,
    pdf_name: str,
    document_id: str,
    chunking: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract prose and table chunks with the PDF Workbench chunkers."""
    from pdf_workbench.custom_logic import (
        chunk_document_text,
        chunk_text_by_technique,
    )

    settings = {**DEFAULT_CHUNKING, **(chunking or {})}
    records: list[dict[str, Any]] = []
    for page_number in sorted(getattr(document, "pages", {}) or {}):
        markdown = document.export_to_markdown(page_no=page_number)
        if not isinstance(markdown, str):
            raise TypeError("Docling page Markdown export did not return text.")
        for chunk in chunk_document_text(
            markdown,
            pdf_name,
            f"{document_id}:page-{page_number}",
            **settings,
        ):
            records.append({
                "modality": "text",
                "object_type": "text",
                "page_number": page_number,
                "source_pages": json.dumps([page_number]),
                "chunk_id": chunk["chunk_id"],
                "section_path": chunk.get("section", ""),
                "text": chunk["source_text"],
            })

    for table_index, table in enumerate(
        getattr(document, "tables", None) or [],
        start=1,
    ):
        markdown = table.export_to_markdown(doc=document)
        if not isinstance(markdown, str):
            raise TypeError("Docling table Markdown export did not return text.")
        if not markdown.strip():
            continue
        pages = list(dict.fromkeys(
            int(provenance.page_no)
            for provenance in (getattr(table, "prov", None) or [])
            if getattr(provenance, "page_no", None) is not None
        ))
        page_number = pages[0] if pages else -1
        for chunk_index, (_start, _end, text) in enumerate(
            chunk_text_by_technique(markdown, **settings),
            start=1,
        ):
            records.append({
                "modality": "text",
                "object_type": "table",
                "page_number": page_number,
                "source_pages": json.dumps(pages),
                "chunk_id": f"{document_id}:table-{table_index}:{chunk_index}",
                "section_path": "",
                "text": text,
            })
    return records


def _structure_chunks(
    segments: list[tuple[int | None, str, str]],
    chunking: dict[str, Any] | None = None,
) -> list[tuple[int | None, str, str, str]]:
    """Group narrative elements by page/section while keeping each table distinct."""
    settings = {**DEFAULT_CHUNKING, **(chunking or {})}
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
            for chunk in _chunk_prose(
                text,
                technique=settings["technique"],
                size=settings["chunk_size"],
                overlap=settings["overlap"],
                min_sentences_per_chunk=settings["min_sentences_per_chunk"],
                semantic_threshold=settings["semantic_threshold"],
            )
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


def ingest_pdf(
    project_id: str,
    filename: str,
    content: bytes,
    progress_callback: Any = None,
    chunking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and embed PDF figures, prose, and tables."""
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF"):
        raise ValueError(f"{filename!r} is not a valid PDF upload.")

    project_dir = PROJECTS_DIR / project_id
    if not (project_dir / "config.json").is_file():
        raise FileNotFoundError(f"Project {project_id!r} not found.")

    settings = {**DEFAULT_CHUNKING, **(chunking or {})}
    if settings["technique"] not in {"recursive", "sentence", "token", "semantic"}:
        raise ValueError(f"Unsupported text chunking technique: {settings['technique']}.")
    if settings["chunk_size"] < 64:
        raise ValueError("PDF ingestion chunk size must be at least 64 tokens.")
    if settings["chunk_size"] > MAX_INGESTION_CHUNK_SIZE:
        raise ValueError(
            f"PDF ingestion chunk size cannot exceed {MAX_INGESTION_CHUNK_SIZE} "
            "tokens because of the embedding model's input limit."
        )
    if (
        settings["technique"] in {"sentence", "token"}
        and not 0 <= settings["overlap"] <= 4096
    ):
        raise ValueError("PDF ingestion overlap must be between 0 and 4096 tokens.")
    if (
        settings["technique"] in {"sentence", "token"}
        and settings["overlap"] >= settings["chunk_size"]
    ):
        raise ValueError("PDF ingestion overlap must be smaller than chunk size.")
    if not 1 <= settings["min_sentences_per_chunk"] <= 20:
        raise ValueError("Minimum sentences per chunk must be between 1 and 20.")
    if not 0.05 <= settings["semantic_threshold"] <= 0.95:
        raise ValueError("Semantic threshold must be between 0.05 and 0.95.")
    chunking_signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()

    pdf_hash = hashlib.sha256(content).hexdigest()
    if progress_callback:
        progress_callback(
            5,
            "validating",
            f"Validated {filename}; preparing document parsing",
        )

    from services.project_chroma import open_project_collection

    collection = open_project_collection(project_id, create=True)
    collection_name = collection.name
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
            and metadata.get("chunking_signature") == chunking_signature
            for metadata in existing_metadata
        ) and len(existing_metadata) == len(batch_ids)
        offset += len(batch_ids)

    if (
        existing_ids
        and existing_complete
        and existing_current_version
    ):
        if progress_callback:
            progress_callback(
                100,
                "complete",
                f"{filename} is already ingested with the current PDF pipeline",
            )
        return {
            "status": "already_ingested",
            "pdf_name": Path(filename).name,
            "pdf_sha256": pdf_hash,
            "chunks_ingested": 0,
            "image_records_ingested": 0,
            "text_records_ingested": 0,
            "table_records_ingested": 0,
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
        if progress_callback:
            progress_callback(
                10,
                "parsing",
                f"Parsing {filename} with Docling",
            )
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
        image_records = _docling_image_records(document, project_dir, pdf_hash)
        document_name = str(getattr(document, "name", "") or safe_name)
        text_records = _docling_text_records(
            document,
            document_name,
            pdf_hash,
            settings,
        )
        records = [
            {
                "modality": "image",
                "object_type": "image",
                "page_number": page_number,
                "source_pages": json.dumps(
                    [page_number] if page_number is not None else []
                ),
                "image_path": str(image_path),
            }
            for page_number, image_path in image_records
        ]
        records.extend(text_records)
        if progress_callback:
            progress_callback(
                40,
                "extracting",
                (
                    f"Extracted {len(image_records)} figure images and "
                    f"{len(text_records)} text/table chunks from {filename}"
                ),
            )
        origin = getattr(document, "origin", None)
        source_mime_type = str(getattr(origin, "mimetype", "") or "application/pdf")
        source_binary_hash = str(getattr(origin, "binary_hash", "") or pdf_hash)
        page_count = len(getattr(document, "pages", {}) or {})
        if not records:
            raise ValueError(
                f"Docling detected no figures, prose, or tables in {safe_name!r}; "
                "no vector records were created."
            )
        if progress_callback:
            progress_callback(
                45,
                "extracting",
                f"Prepared {len(records)} image and text records",
            )

        from services.local_embeddings import (
            DOCUMENT_INSTRUCTION,
            embed_records,
        )

        uploaded_at = datetime.now(timezone.utc).isoformat()
        all_ids: list[str] = []
        complete_metadata: list[dict[str, Any]] = []
        for offset in range(0, len(records), EMBEDDING_BATCH_SIZE):
            batch = records[offset : offset + EMBEDDING_BATCH_SIZE]
            batch_number = offset // EMBEDDING_BATCH_SIZE + 1
            total_batches = (
                len(records) + EMBEDDING_BATCH_SIZE - 1
            ) // EMBEDDING_BATCH_SIZE
            if progress_callback:
                percentage = 45 + int(50 * (offset / len(records)))
                progress_callback(
                    percentage,
                    "loading_model" if offset == 0 else "embedding",
                    (
                        "Loading the local Qwen3-VL model"
                        if offset == 0
                        else (
                            f"Preparing embedding record {batch_number}/{total_batches}"
                            if EMBEDDING_BATCH_SIZE == 1
                            else f"Preparing embedding batch {batch_number}/{total_batches}"
                        )
                    ),
                    batch_current=batch_number,
                    batch_total=total_batches,
                    records_current=offset,
                    records_total=len(records),
                )
            embedding_inputs: list[dict[str, Any]] = []
            for chunk in batch:
                embedding_input: dict[str, Any] = {
                    "instruction": DOCUMENT_INSTRUCTION,
                }
                if chunk["modality"] == "image":
                    embedding_input["image"] = chunk["image_path"]
                else:
                    embedding_input["text"] = chunk["text"]
                embedding_inputs.append(embedding_input)

            def report_embedding_started() -> None:
                if progress_callback:
                    progress_callback(
                        percentage,
                        "embedding",
                        (
                            f"Generating Qwen embedding for record "
                            f"{batch_number}/{total_batches}"
                            if EMBEDDING_BATCH_SIZE == 1
                            else f"Generating Qwen embeddings for batch {batch_number}/{total_batches}"
                        ),
                        batch_current=batch_number,
                        batch_total=total_batches,
                        records_current=offset,
                        records_total=len(records),
                    )

            embeddings = embed_records(
                embedding_inputs,
                model_loaded_callback=(
                    report_embedding_started if offset == 0 else None
                ),
            )
            ids = [
                f"{pdf_hash[:24]}_{index:06d}"
                for index in range(offset, offset + len(batch))
            ]
            metadatas = []
            for index, chunk in enumerate(batch):
                metadata = {
                    "pdf_name": safe_name,
                    "source_pdf": safe_name,
                    "pdf_sha256": pdf_hash,
                    "doc_id": pdf_hash,
                    "docling_document_name": document_name,
                    "source_mime_type": source_mime_type,
                    "source_binary_hash": source_binary_hash,
                    "page_count": page_count,
                    "chunk_index": offset + index,
                    "page_number": (
                        chunk["page_number"]
                        if chunk["page_number"] is not None
                        else -1
                    ),
                    "source_pages": chunk["source_pages"],
                    "object_type": chunk["object_type"],
                    "chunk_type": chunk["object_type"],
                    "uploaded_at": uploaded_at,
                    "ingestion_status": "in_progress",
                    "ingestion_version": INGESTION_VERSION,
                    "chunking_signature": chunking_signature,
                    "chunking_technique": settings["technique"],
                    "chunk_size": settings["chunk_size"],
                    "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
                    "embedding_dimension": 2048,
                    "modality": chunk["modality"],
                }
                if chunk.get("image_path"):
                    metadata["image_path"] = chunk["image_path"]
                if chunk.get("chunk_id"):
                    metadata["chunk_id"] = chunk["chunk_id"]
                if chunk.get("section_path"):
                    metadata["section_path"] = chunk["section_path"]
                metadatas.append(metadata)
            all_ids.extend(ids)
            complete_metadata.extend(
                {**metadata, "ingestion_status": "complete"}
                for metadata in metadatas
            )
            for modality in ("image", "text"):
                selected = [
                    index
                    for index, chunk in enumerate(batch)
                    if chunk["modality"] == modality
                ]
                if not selected:
                    continue
                upsert: dict[str, Any] = {
                    "ids": [ids[index] for index in selected],
                    "embeddings": [
                        [float(value) for value in embeddings[index]]
                        for index in selected
                    ],
                    "metadatas": [metadatas[index] for index in selected],
                }
                if modality == "text":
                    upsert["documents"] = [
                        batch[index]["text"] for index in selected
                    ]
                collection.upsert(**upsert)
            if progress_callback:
                completed = offset + len(batch)
                percentage = 45 + int(50 * (completed / len(records)))
                progress_callback(
                    percentage,
                    "embedding",
                    (
                        f"Embedded and stored record {batch_number}/{total_batches}"
                        if EMBEDDING_BATCH_SIZE == 1
                        else f"Embedded and stored batch {batch_number}/{total_batches}"
                    ),
                    batch_current=batch_number,
                    batch_total=total_batches,
                    records_current=completed,
                    records_total=len(records),
                )

        os.replace(temp_path, stored_path)
        temp_path = None
        for offset in range(0, len(all_ids), 5000):
            collection.update(
                ids=all_ids[offset : offset + 5000],
                metadatas=complete_metadata[offset : offset + 5000],
            )
        if progress_callback:
            progress_callback(
                98,
                "storing",
                f"Finalized {len(all_ids)} image, text, and table records for {safe_name}",
                records_current=len(all_ids),
                records_total=len(all_ids),
            )
        table_record_count = sum(
            record["object_type"] == "table" for record in text_records
        )
        text_record_count = len(text_records) - table_record_count
        return {
            "status": "ingested",
            "pdf_name": safe_name,
            "pdf_sha256": pdf_hash,
            "chunks_ingested": len(records),
            "image_records_ingested": len(image_records),
            "text_records_ingested": text_record_count,
            "table_records_ingested": table_record_count,
            "collection_name": collection_name,
            "extraction_status": str(conversion_status),
            "page_count": page_count,
            "images_extracted": len(image_records),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
