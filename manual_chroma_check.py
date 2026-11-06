"""Manually inspect a project's persistent Chroma collection and query results.

Edit the settings below, then run this file. It is read-only: it scans the
collection and runs a vector query, then prints the results as inspection
evidence.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb

from services.local_embeddings import embed_texts
from services.paths import PROJECTS_DIR


# Edit these settings before running the script.
PROJECT_ID = "proj_1477c01d6e79"
COLLECTION_NAME: str | None = None  # None selects the project's configured collection.
K = 5
QUESTION = "RAG has been increasingly applied to structured knowledge extraction and domain-specific reasoning tasks."  # Use "" to skip question retrieval.
REQUIRED_FIELDS = (
    "pdf_name",
    "pdf_sha256",
    "page_number",
    "chunk_index",
)
SCAN_BATCH_SIZE = 500
TEXT_PREVIEW_CHARS = 2000


def _project_configs() -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    if not PROJECTS_DIR.is_dir():
        return projects

    for config_path in sorted(PROJECTS_DIR.glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable project config {config_path}: {exc}")
            continue
        project_id = str(config.get("id") or config_path.parent.name)
        projects.append({
            "id": project_id,
            "name": str(config.get("name") or project_id),
            "directory": config_path.parent,
            "settings": config.get("settings") or {},
        })
    return projects


def _select_project(projects: list[dict[str, Any]]) -> dict[str, Any]:
    for project in projects:
        if project["id"] == PROJECT_ID:
            return project
    raise ValueError(f"Project {PROJECT_ID!r} was not found under {PROJECTS_DIR}.")


def _collection_names(client: Any) -> list[str]:
    return [
        str(getattr(collection, "name", collection))
        for collection in client.list_collections()
    ]


def _select_collection(
    project: dict[str, Any],
    names: list[str],
) -> str:
    project_id = project["id"]
    config_name = (
        (project["settings"].get("vector_store_config") or {}).get("collection_name")
    )
    preferred = f"{project_id}_knowledge_base"
    if COLLECTION_NAME:
        if COLLECTION_NAME not in names:
            raise ValueError(
                f"Collection {COLLECTION_NAME!r} was not found. "
                f"Available collections: {', '.join(names) or '(none)'}."
            )
        return COLLECTION_NAME

    for name in (config_name, preferred):
        if name and name in names:
            return name
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError("No Chroma collections exist in this project's persistent store.")
    raise ValueError(
        "Could not choose a collection automatically. Set COLLECTION_NAME at "
        f"the top of this file. Available collections: {', '.join(names)}."
    )


def _is_empty_metadata_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _scan_collection(
    collection: Any,
    required_fields: list[str],
) -> dict[str, Any]:
    total = collection.count()
    offset = 0
    scanned = 0
    vectors_with_data = 0
    records_without_vectors: list[str] = []
    empty_documents: list[str] = []
    dimensions: Counter[int] = Counter()
    metadata_key_counts: Counter[str] = Counter()
    required_field_issues = {
        field: {"missing": 0, "empty": 0}
        for field in required_fields
    }

    while offset < total:
        batch = collection.get(
            limit=SCAN_BATCH_SIZE,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        embeddings = batch.get("embeddings")
        metadatas = batch.get("metadatas")
        documents = batch.get("documents")

        for index, record_id in enumerate(ids):
            scanned += 1
            vector = embeddings[index] if embeddings is not None else None
            if vector is None or len(vector) == 0:
                records_without_vectors.append(str(record_id))
            else:
                vectors_with_data += 1
                dimensions[len(vector)] += 1

            metadata = (
                metadatas[index]
                if metadatas is not None and index < len(metadatas)
                else None
            ) or {}
            for key in metadata:
                metadata_key_counts[str(key)] += 1
            for field, counts in required_field_issues.items():
                if field not in metadata:
                    counts["missing"] += 1
                elif _is_empty_metadata_value(metadata[field]):
                    counts["empty"] += 1

            document = (
                documents[index]
                if documents is not None and index < len(documents)
                else None
            )
            if not isinstance(document, str) or not document.strip():
                empty_documents.append(str(record_id))

        offset += len(ids)

    return {
        "collection_count": total,
        "records_scanned": scanned,
        "vectors_with_data": vectors_with_data,
        "records_without_vectors": records_without_vectors,
        "dimension_counts": dimensions,
        "metadata_key_counts": metadata_key_counts,
        "required_field_issues": required_field_issues,
        "empty_documents": empty_documents,
    }


def _print_scan_summary(summary: dict[str, Any]) -> None:
    print("\n=== Collection inspection ===")
    print(f"Collection records:       {summary['collection_count']}")
    print(f"Records scanned:          {summary['records_scanned']}")
    print(f"Records with vectors:     {summary['vectors_with_data']}")
    print(f"Records without vectors:  {len(summary['records_without_vectors'])}")
    print(f"Records without text:     {len(summary['empty_documents'])}")
    if summary["dimension_counts"]:
        dims = ", ".join(
            f"{dimension}D: {count}"
            for dimension, count in sorted(summary["dimension_counts"].items())
        )
        print(f"Vector dimensions:        {dims}")
    else:
        print("Vector dimensions:        no vectors found")

    if summary["records_without_vectors"]:
        print("IDs without vectors (first 20):")
        for record_id in summary["records_without_vectors"][:20]:
            print(f"  - {record_id}")
    if summary["empty_documents"]:
        print("IDs without stored text (first 20):")
        for record_id in summary["empty_documents"][:20]:
            print(f"  - {record_id}")

    print("\nMetadata field coverage:")
    if summary["metadata_key_counts"]:
        for field, count in sorted(summary["metadata_key_counts"].items()):
            print(f"  {field}: {count}/{summary['records_scanned']} records")
    else:
        print("  No metadata found.")

    print("\nRequired-field inspection:")
    if not summary["required_field_issues"]:
        print("  No required fields were selected.")
    for field, counts in summary["required_field_issues"].items():
        print(
            f"  {field}: missing={counts['missing']}, empty={counts['empty']}"
        )


def _run_question(collection: Any, question: str, requested_k: int) -> None:
    count = collection.count()
    if count == 0:
        print("\n=== Question retrieval ===")
        print("No records are available to query.")
        return

    actual_k = min(requested_k, count)
    if actual_k < requested_k:
        print(
            f"\nRequested k={requested_k}, but the collection has only "
            f"{count} record(s); querying k={actual_k}."
        )

    query_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = (result.get("ids") or [[]])[0] or []
    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []

    print("\n=== Question retrieval ===")
    print(f"Question: {question}")
    print(f"Requested k: {requested_k}")
    print(f"Results returned: {len(ids)}")
    print("Chroma distance is printed as returned; smaller means closer.")

    if not ids:
        print("Chroma returned no matching records.")
        return

    for index, record_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        document = documents[index] if index < len(documents) else None
        distance = distances[index] if index < len(distances) else None
        print(f"\n--- Result {index + 1} ---")
        print(f"ID: {record_id}")
        print(f"Distance: {distance if distance is not None else 'unavailable'}")
        if metadata:
            print("Metadata:")
            print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        else:
            print("Metadata: unavailable")
        text = document if isinstance(document, str) else ""
        if text:
            preview = text[:TEXT_PREVIEW_CHARS]
            suffix = (
                f"\n[Preview truncated; {len(text)} total characters]"
                if len(text) > TEXT_PREVIEW_CHARS
                else ""
            )
            print(f"Retrieved chunk:\n{preview}{suffix}")
        else:
            print("Retrieved chunk: no document text stored for this record.")


def main() -> int:
    if K < 1:
        raise ValueError("K must be at least 1.")
    projects = _project_configs()
    project = _select_project(projects)
    persist_directory = project["directory"] / "chroma_data"
    if not persist_directory.is_dir():
        raise FileNotFoundError(
            f"Persistent Chroma directory does not exist: {persist_directory}"
        )

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection_name = _select_collection(
        project,
        _collection_names(client),
    )
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    print("Read-only manual Chroma inspection")
    print(f"Project: {project['name']} ({project['id']})")
    print(f"Persistent directory: {persist_directory}")
    print(f"Collection: {collection_name}")
    print(f"Collection metadata: {collection.metadata or {}}")

    summary = _scan_collection(collection, list(REQUIRED_FIELDS))
    _print_scan_summary(summary)

    if QUESTION.strip():
        _run_question(collection, QUESTION, K)
    else:
        print("\nQuestion retrieval skipped (no question provided).")

    print("\nInspection complete. No collection records were changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
