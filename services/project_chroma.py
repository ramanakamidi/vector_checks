"""Helpers for opening a project's configured persistent vector collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from services.paths import PROJECTS_DIR


def _project_directory(
    project_id: str,
    projects_dir: str | Path = PROJECTS_DIR,
) -> Path:
    projects_dir = Path(projects_dir).resolve()
    project_dir = (projects_dir / project_id).resolve()
    if project_dir.parent != projects_dir:
        raise FileNotFoundError(f"Project {project_id!r} was not found.")
    if not (project_dir / "config.json").is_file():
        raise FileNotFoundError(f"Project {project_id!r} was not found.")
    return project_dir


def default_storage_location(project_dir: str | Path, db_type: str) -> Path:
    """Return the default project storage location for a vector backend."""
    project_dir = Path(project_dir)
    if db_type == "faiss":
        return project_dir / "faiss_data"
    if db_type == "milvus_lite":
        return project_dir / "milvus_lite.db"
    return project_dir / "chroma_data"


def collection_name(
    client: Any,
    project_id: str,
    projects_dir: str | Path = PROJECTS_DIR,
) -> str:
    """Resolve the collection belonging to a project."""
    names = {
        str(getattr(collection, "name", collection))
        for collection in client.list_collections()
    }
    preferred_name = f"{project_id}_knowledge_base"
    if preferred_name in names:
        return preferred_name

    config_path = _project_directory(project_id, projects_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read project configuration: {exc}") from exc

    configured_name = (
        (config.get("settings") or {})
        .get("vector_store_config", {})
        .get("collection_name")
    )
    if configured_name in names:
        return configured_name
    if len(names) == 1:
        return next(iter(names))
    if not names:
        raise ValueError(
            f"No Chroma collections were found for project {project_id!r}."
        )
    raise ValueError(
        "Could not select the project's Chroma collection automatically. "
        f"Available collections: {', '.join(sorted(names))}."
    )


def open_project_collection(
    project_id: str,
    *,
    create: bool = False,
    projects_dir: str | Path = PROJECTS_DIR,
) -> Any:
    """Open a project's configured vector collection."""
    project_dir = _project_directory(project_id, projects_dir)
    try:
        config = json.loads(
            (project_dir / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read project configuration: {exc}") from exc

    vector_config = (
        (config.get("settings") or {}).get("vector_store_config") or {}
    )
    db_type = vector_config.get("db_type", "chroma")
    name = vector_config.get("collection_name") or f"{project_id}_knowledge_base"
    configured_directory = vector_config.get("persist_directory")

    if db_type == "faiss":
        from connectors.faiss_connector import FaissConnector

        persist_directory = Path(configured_directory) if configured_directory else (
            default_storage_location(project_dir, db_type)
        )
        connector = FaissConnector(name, str(persist_directory))
        connector.connect()
        if not create and connector.count() == 0:
            raise ValueError(f"FAISS collection {name!r} contains no records.")
        return connector
    if db_type == "milvus_lite":
        from connectors.milvus_lite_connector import MilvusLiteConnector

        database_path = Path(configured_directory) if configured_directory else (
            default_storage_location(project_dir, db_type)
        )
        if not database_path.exists() and not create:
            raise FileNotFoundError(
                f"Project {project_id!r} has no Milvus Lite database path."
            )
        connector = MilvusLiteConnector(name, str(database_path))
        connector.connect()
        if not create and connector.count() == 0:
            raise ValueError(f"Milvus Lite collection {name!r} contains no records.")
        return connector
    if db_type != "chroma":
        raise ValueError(f"Unsupported vector store type: {db_type!r}.")

    persist_directory = Path(configured_directory) if configured_directory else (
        default_storage_location(project_dir, db_type)
    )
    if not persist_directory.is_dir() and not create:
        raise FileNotFoundError(
            f"Project {project_id!r} has no persistent Chroma directory."
        )
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    if create:
        existing_names = {
            str(getattr(item, "name", item))
            for item in client.list_collections()
        }
        preferred_name = f"{project_id}_knowledge_base"
        if preferred_name in existing_names:
            name = preferred_name
        elif name not in existing_names and len(existing_names) == 1:
            name = next(iter(existing_names))
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    name = collection_name(client, project_id, projects_dir)
    collection = client.get_collection(name=name, embedding_function=None)
    if collection.count() == 0:
        raise ValueError(f"Chroma collection {name!r} contains no records.")
    return collection
