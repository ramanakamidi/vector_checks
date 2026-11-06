"""
Project Manager — CRUD operations for project-based workspaces.

Each project is stored as a directory under ``projects/`` with a
``config.json`` file containing all settings and metadata.

Usage::

    from services.project_manager import ProjectManager

    pm = ProjectManager()
    project = pm.create("Insurance RAG Test", "Testing insurance docs")
    pm.update_settings(project["id"], {"llm_config": {"provider": "openai"}})
    all_projects = pm.list_all()
"""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.paths import PROJECTS_DIR


DEFAULT_SETTINGS: Dict[str, Any] = {
    "files_directory": "",
    "extraction_mode": "docling",
    "llm_config": {
        "provider": "openai",
        "api_key": "",
        "model": "gpt-4",
    },
    "aut_config": {
        "url": "http://localhost:8000/mock-aut",
        "method": "POST",
        "endpoint": "/ask",
        "headers": {},
        "request_template": {"question": "{{query}}"},
        "response_answer_path": "answer",
        "response_context_path": "context",
    },
    "vector_store_config": {
        "db_type": "chroma",
        "collection_name": "",
        "persist_directory": "",
    },
    "field_mapper": {
        "doc_id_field": "doc_id",
        "doc_name_field": "doc_name",
        "text_field": "text",
        "page_field": "page_number",
        "chunk_index_field": "chunk_index",
        "object_type_field": "object_type",
        "category_field": None,
        "source_url_field": "source_url",
        "file_path_field": "file_path",
        "ingestion_date_field": "ingestion_date",
        "batch_id_field": "batch_id",
        "display_fields": [
            "doc_id", "doc_name", "page_number", "object_type", "text"
        ],
        "required_fields": ["doc_id", "object_type"],
        "similar_group_field": "doc_id",
        "different_group_field": "object_type",
        "drift_time_field": "ingestion_date",
    },
}


class ProjectManager:
    """Manages project lifecycle with JSON-file persistence."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.fspath(PROJECTS_DIR)
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _project_dir(self, project_id: str) -> str:
        return os.path.join(self.base_dir, project_id)

    def _config_path(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "config.json")

    def _read_config(self, project_id: str) -> Dict[str, Any]:
        path = self._config_path(project_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_config(self, project_id: str, config: Dict[str, Any]) -> None:
        path = self._config_path(project_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_id() -> str:
        return f"proj_{uuid.uuid4().hex[:12]}"

    def list_all(self) -> List[Dict[str, Any]]:
        """Return summary info for every project."""
        projects = []
        if not os.path.isdir(self.base_dir):
            return projects
        for name in sorted(os.listdir(self.base_dir)):
            config_path = os.path.join(self.base_dir, name, "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    projects.append({
                        "id": cfg.get("id", name),
                        "name": cfg.get("name", name),
                        "description": cfg.get("description", ""),
                        "created_at": cfg.get("created_at", ""),
                        "updated_at": cfg.get("updated_at", ""),
                        "status": cfg.get("status", "created"),
                    })
                except (json.JSONDecodeError, OSError):
                    continue
        return projects

    def create(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new project with default settings."""
        project_id = self._generate_id()
        project_dir = self._project_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)

        now = self._now_iso()
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        settings["vector_store_config"]["collection_name"] = f"{project_id}_chunks"
        settings["vector_store_config"]["persist_directory"] = os.path.join(
            project_dir, "chroma_data"
        )

        config: Dict[str, Any] = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "status": "created",
            "settings": settings,
        }
        self._write_config(project_id, config)
        return config

    def get(self, project_id: str) -> Dict[str, Any]:
        """Get full project config."""
        return self._read_config(project_id)

    def update(self, project_id: str, name: Optional[str] = None,
               description: Optional[str] = None) -> Dict[str, Any]:
        """Update project name and/or description."""
        config = self._read_config(project_id)
        if name is not None:
            config["name"] = name
        if description is not None:
            config["description"] = description
        config["updated_at"] = self._now_iso()
        self._write_config(project_id, config)
        return config

    def delete(self, project_id: str) -> None:
        """Delete a project and all its data."""
        project_dir = self._project_dir(project_id)
        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        shutil.rmtree(project_dir)

    def update_settings(self, project_id: str,
                        settings_update: Dict[str, Any]) -> Dict[str, Any]:
        """Merge partial settings update into existing config."""
        config = self._read_config(project_id)
        existing = config.get("settings", {})
        for key, value in settings_update.items():
            if isinstance(value, dict) and isinstance(existing.get(key), dict):
                existing[key].update(value)
            else:
                existing[key] = value
        config["settings"] = existing
        config["updated_at"] = self._now_iso()
        if self._is_configured(existing):
            config["status"] = "configured"
        self._write_config(project_id, config)
        return config

    def get_settings(self, project_id: str) -> Dict[str, Any]:
        """Get just the settings portion of a project config."""
        config = self._read_config(project_id)
        return config.get("settings", {})

    def update_status(self, project_id: str, status: str) -> None:
        """Update the project status."""
        config = self._read_config(project_id)
        config["status"] = status
        config["updated_at"] = self._now_iso()
        self._write_config(project_id, config)

    def get_project_dir(self, project_id: str) -> str:
        """Return the filesystem path for a project directory."""
        d = self._project_dir(project_id)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        return d

    @staticmethod
    def _is_configured(settings: Dict[str, Any]) -> bool:
        vs = settings.get("vector_store_config", {})
        return bool(vs.get("collection_name") and vs.get("persist_directory"))

    def save_json(self, project_id: str, filename: str, data: Any) -> str:
        """Save arbitrary JSON data to a project file."""
        project_dir = self._project_dir(project_id)
        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        path = os.path.join(project_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def load_json(self, project_id: str, filename: str) -> Any:
        """Load JSON data from a project file."""
        path = os.path.join(self._project_dir(project_id), filename)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_result(self, project_id: str, execution_id: str,
                    data: Dict[str, Any]) -> str:
        """Save execution results to the results/ subdirectory."""
        results_dir = os.path.join(self._project_dir(project_id), "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{execution_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def list_results(self, project_id: str) -> List[str]:
        """List execution IDs for a project."""
        results_dir = os.path.join(self._project_dir(project_id), "results")
        if not os.path.isdir(results_dir):
            return []
        return [
            f.replace(".json", "")
            for f in sorted(os.listdir(results_dir))
            if f.endswith(".json")
        ]

    def load_result(self, project_id: str,
                    execution_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific execution result."""
        path = os.path.join(
            self._project_dir(project_id), "results", f"{execution_id}.json"
        )
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
