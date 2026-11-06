"""
Ingestion Service — handles JSON Q&A data ingestion into project-specific
Chroma vector stores.

Each project gets its own PersistentClient at
``projects/{project_id}/chroma_data/`` with a collection named
``{project_id}_knowledge_base``.

Usage::

    from services.ingestion import IngestionService

    svc = IngestionService("proj_abc123")
    result = svc.ingest_json([{"question": "...", "answer": "...", "context": "..."}])
    stats  = svc.get_stats()
    page   = svc.get_records(page=1, page_size=50)
    svc.clear_data()
"""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.paths import PROJECTS_DIR


class IngestionService:
    """Manages ingestion of Q&A JSON data into a project-specific Chroma store."""

    def __init__(self, project_id: str, projects_dir: Optional[str] = None):
        if projects_dir is None:
            projects_dir = os.fspath(PROJECTS_DIR)
        self.project_id = project_id
        self.project_dir = os.path.join(projects_dir, project_id)
        self.chroma_path = os.path.join(self.project_dir, "chroma_data")
        self.json_path = os.path.join(self.project_dir, "ingested_data.json")
        self.collection_name = f"{project_id}_knowledge_base"

        if not os.path.isdir(self.project_dir):
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")

    def _get_chroma_collection(self):
        """Create or connect to the project-specific Chroma collection."""
        import chromadb

        os.makedirs(self.chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        return client, collection

    def ingest_json(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and ingest Q&A records into Chroma and save a JSON backup.

        Parameters
        ----------
        records : list[dict]
            Each dict should have ``question``, ``answer``, ``context``,
            and optionally ``id`` and ``metadata``.

        Returns
        -------
        dict
            ``{"status": "success", "records_ingested": N,
              "collection_name": "...", "errors": [...]}``
        """
        errors: List[str] = []
        valid_records: List[Dict[str, Any]] = []

        # ── Validate ──────────────────────────────────────────────
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                errors.append(f"Record {idx}: not a JSON object")
                continue

            question = rec.get("question", "")
            answer = rec.get("answer", "")
            context = rec.get("context", "")

            if not question and not answer and not context:
                errors.append(f"Record {idx}: must have at least one of question, answer, or context")
                continue

            record_id = rec.get("id") or str(uuid.uuid4())
            metadata = rec.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            valid_records.append({
                "id": str(record_id),
                "question": str(question),
                "answer": str(answer),
                "context": str(context),
                "metadata": metadata,
            })

        if not valid_records:
            return {
                "status": "error",
                "records_ingested": 0,
                "collection_name": self.collection_name,
                "errors": errors or ["No valid records found in the input"],
            }

        # ── Ingest into Chroma ────────────────────────────────────
        _client, collection = self._get_chroma_collection()

        # Chroma batch-add (max 5461 per batch due to Chroma limits)
        batch_size = 5000
        for i in range(0, len(valid_records), batch_size):
            batch = valid_records[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for rec in batch:
                ids.append(rec["id"])
                # The context field is the document text that gets embedded
                doc_text = rec["context"] if rec["context"] else f"{rec['question']} {rec['answer']}"
                documents.append(doc_text)

                # Build metadata — Chroma metadata values must be str, int, float, or bool
                meta: Dict[str, Any] = {}
                meta["question"] = rec["question"]
                meta["answer"] = rec["answer"]

                # Flatten user-provided metadata
                for k, v in rec["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif v is None:
                        meta[k] = ""
                    else:
                        meta[k] = str(v)

                metadatas.append(meta)

            from services.local_embeddings import embed_texts

            embeddings = embed_texts(documents)
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # ── Save JSON backup ──────────────────────────────────────
        # Merge with existing data if any
        existing = self._load_json_backup()
        existing_map = {r["id"]: r for r in existing}
        for rec in valid_records:
            existing_map[rec["id"]] = rec
        all_records = list(existing_map.values())

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, default=str)

        # ── Update project config with ingestion stats ────────────
        self._update_project_config(len(all_records))

        return {
            "status": "success",
            "records_ingested": len(valid_records),
            "collection_name": self.collection_name,
            "errors": errors,
        }

    def get_records(
        self, page: int = 1, page_size: int = 50, search: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return paginated records from the saved JSON backup.

        Parameters
        ----------
        page : int
            1-based page number.
        page_size : int
            Records per page.
        search : str, optional
            Case-insensitive text search across question, answer, context.

        Returns
        -------
        dict
            ``{"records": [...], "total": N, "page": P, "page_size": S}``
        """
        records = self._load_json_backup()

        # Apply search filter
        if search:
            search_lower = search.lower()
            records = [
                r
                for r in records
                if search_lower in r.get("question", "").lower()
                or search_lower in r.get("answer", "").lower()
                or search_lower in r.get("context", "").lower()
                or search_lower in str(r.get("metadata", {})).lower()
            ]

        total = len(records)
        start = (page - 1) * page_size
        end = start + page_size
        page_records = records[start:end]

        return {
            "records": page_records,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return ingestion statistics.

        Returns
        -------
        dict
            ``{"total_records": N, "collections": [...],
              "metadata_fields": [...], "ingested_at": "...",
              "chroma_path": "..."}``
        """
        records = self._load_json_backup()
        total = len(records)

        # Collect all unique metadata field names
        meta_fields: set = set()
        for rec in records:
            meta = rec.get("metadata", {})
            if isinstance(meta, dict):
                meta_fields.update(meta.keys())

        # Get ingested_at from project config
        ingested_at = None
        config_path = os.path.join(self.project_dir, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                ingested_at = cfg.get("settings", {}).get("ingestion_stats", {}).get("ingested_at")
            except (json.JSONDecodeError, OSError):
                pass

        # Check if Chroma collection exists
        collections: List[str] = []
        if os.path.isdir(self.chroma_path):
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.chroma_path)
                raw = client.list_collections()
                for c in raw:
                    collections.append(c.name if hasattr(c, "name") else str(c))
            except Exception:
                if total > 0:
                    collections = [self.collection_name]

        return {
            "total_records": total,
            "collections": collections,
            "metadata_fields": sorted(meta_fields),
            "ingested_at": ingested_at,
            "chroma_path": self.chroma_path,
        }

    def clear_data(self) -> Dict[str, Any]:
        """Delete the Chroma collection and JSON backup file.

        Returns
        -------
        dict
            ``{"status": "success", "message": "All ingested data cleared"}``
        """
        # Delete Chroma collection
        if os.path.isdir(self.chroma_path):
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.chroma_path)
                try:
                    client.delete_collection(self.collection_name)
                except Exception:
                    pass
            except Exception:
                pass
            # Remove the entire chroma_data directory
            try:
                shutil.rmtree(self.chroma_path)
            except Exception:
                pass

        # Delete JSON backup
        if os.path.isfile(self.json_path):
            os.remove(self.json_path)

        # Update project config
        self._update_project_config(0, clear=True)

        return {
            "status": "success",
            "message": "All ingested data cleared",
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _load_json_backup(self) -> List[Dict[str, Any]]:
        """Load the JSON backup file, returning [] if it doesn't exist."""
        if not os.path.isfile(self.json_path):
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _update_project_config(self, record_count: int, clear: bool = False) -> None:
        """Update the project config.json with ingestion stats."""
        config_path = os.path.join(self.project_dir, "config.json")
        if not os.path.isfile(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        settings = cfg.setdefault("settings", {})

        if clear:
            settings.pop("ingestion_stats", None)
            cfg["status"] = "configured"
        else:
            settings["ingestion_stats"] = {
                "record_count": record_count,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "collection_name": self.collection_name,
            }
            cfg["status"] = "ingested"

        cfg["updated_at"] = datetime.now(timezone.utc).isoformat()

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)
