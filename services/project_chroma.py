"""Helpers for opening a project's persistent Chroma collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from services.paths import PROJECTS_DIR


def _project_directory(project_id: str) -> Path:
    projects_dir = PROJECTS_DIR.resolve()
    project_dir = (PROJECTS_DIR / project_id).resolve()
    if project_dir.parent != projects_dir:
        raise FileNotFoundError(f"Project {project_id!r} was not found.")
    if not (project_dir / "config.json").is_file():
        raise FileNotFoundError(f"Project {project_id!r} was not found.")
    return project_dir


def collection_name(client: Any, project_id: str) -> str:
    """Resolve the collection belonging to a project."""
    names = {
        str(getattr(collection, "name", collection))
        for collection in client.list_collections()
    }
    preferred_name = f"{project_id}_knowledge_base"
    if preferred_name in names:
        return preferred_name

    config_path = _project_directory(project_id) / "config.json"
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


def open_project_collection(project_id: str) -> Any:
    """Open and return a non-empty project's persistent Chroma collection."""
    project_dir = _project_directory(project_id)
    persist_directory = project_dir / "chroma_data"
    if not persist_directory.is_dir():
        raise FileNotFoundError(
            f"Project {project_id!r} has no persistent Chroma directory."
        )

    client = chromadb.PersistentClient(path=str(persist_directory))
    name = collection_name(client, project_id)
    collection = client.get_collection(name=name, embedding_function=None)
    if collection.count() == 0:
        raise ValueError(f"Chroma collection {name!r} contains no records.")
    return collection
