"""
Data Generation Service — generates test Q&A datasets from ingested data.

Each project can have multiple generated datasets stored as JSON files at
``projects/{project_id}/datasets/{dataset_id}.json``.

Usage::

    from services.data_generation import DataGenerationService

    svc = DataGenerationService("proj_abc123")
    result = svc.generate_dataset({
        "num_pairs_per_chunk": 3,
        "total_limit": 100,
        "query_conditions": {},
        "include_context": True,
        "difficulty_levels": ["easy", "medium", "hard"],
        "generation_mode": "from_ingested",
    })
    datasets = svc.list_datasets()
    dataset  = svc.get_dataset("dataset_id")
    svc.delete_dataset("dataset_id")
"""

import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.paths import PROJECTS_DIR


class DataGenerationService:
    """Generates test Q&A datasets from ingested knowledge base data."""

    def __init__(self, project_id: str, projects_dir: Optional[str] = None):
        if projects_dir is None:
            projects_dir = os.fspath(PROJECTS_DIR)
        self.project_id = project_id
        self.project_dir = os.path.join(projects_dir, project_id)
        self.datasets_dir = os.path.join(self.project_dir, "datasets")
        self.json_path = os.path.join(self.project_dir, "ingested_data.json")

        if not os.path.isdir(self.project_dir):
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")

    # ── Public API ────────────────────────────────────────────────────

    def generate_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a test dataset from ingested data.

        Parameters
        ----------
        config : dict
            Configuration with keys:
            - num_pairs_per_chunk (int): Q&A pairs per chunk (default 3)
            - total_limit (int): Max total pairs, 0 = unlimited (default 100)
            - query_conditions (dict): Metadata filter conditions
            - include_context (bool): Include context in output (default True)
            - difficulty_levels (list[str]): Mix of difficulties
            - generation_mode (str): "from_ingested" or "llm_augmented"

        Returns
        -------
        dict
            Generated dataset with dataset_id, pairs, metadata, etc.
        """
        num_pairs = config.get("num_pairs_per_chunk", 3)
        total_limit = config.get("total_limit", 100)
        query_conditions = config.get("query_conditions", {})
        include_context = config.get("include_context", True)
        difficulty_levels = config.get("difficulty_levels", ["easy", "medium", "hard"])
        generation_mode = config.get("generation_mode", "from_ingested")

        # Load ingested data
        records = self._load_ingested_data()
        if not records:
            return {
                "status": "error",
                "message": "No ingested data found. Please ingest data first.",
                "dataset_id": None,
                "total_pairs": 0,
                "pairs": [],
            }

        # Apply query conditions (filter by metadata)
        filtered = self._apply_filters(records, query_conditions)
        if not filtered:
            return {
                "status": "error",
                "message": "No records match the specified query conditions.",
                "dataset_id": None,
                "total_pairs": 0,
                "pairs": [],
            }

        # Generate pairs based on mode
        if generation_mode == "llm_augmented":
            # Check if API key is configured
            api_key = self._get_api_key()
            if not api_key:
                # Fall back to from_ingested mode
                generation_mode = "from_ingested"

        pairs: List[Dict[str, Any]] = []

        if generation_mode == "from_ingested":
            pairs = self._generate_from_ingested(
                filtered, num_pairs, difficulty_levels, include_context
            )
        else:
            # LLM augmented — for now, fall back to from_ingested
            # (LLM integration can be added later)
            pairs = self._generate_from_ingested(
                filtered, num_pairs, difficulty_levels, include_context
            )

        # Apply total limit
        if total_limit > 0 and len(pairs) > total_limit:
            pairs = pairs[:total_limit]

        # Build dataset
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        dataset = {
            "dataset_id": dataset_id,
            "generated_at": now,
            "total_pairs": len(pairs),
            "config_used": {
                "num_pairs_per_chunk": num_pairs,
                "total_limit": total_limit,
                "query_conditions": query_conditions,
                "include_context": include_context,
                "difficulty_levels": difficulty_levels,
                "generation_mode": generation_mode,
            },
            "pairs": pairs,
        }

        # Save dataset
        self._save_dataset(dataset_id, dataset)

        # Update project status
        self._update_project_status()

        return dataset

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all generated datasets for this project.

        Returns
        -------
        list[dict]
            Summary info for each dataset (without the full pairs array).
        """
        if not os.path.isdir(self.datasets_dir):
            return []

        datasets = []
        for filename in sorted(os.listdir(self.datasets_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.datasets_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets.append({
                    "dataset_id": data.get("dataset_id", filename.replace(".json", "")),
                    "generated_at": data.get("generated_at", ""),
                    "total_pairs": data.get("total_pairs", 0),
                    "config_used": data.get("config_used", {}),
                })
            except (json.JSONDecodeError, OSError):
                continue

        return datasets

    def get_dataset(
        self, dataset_id: str, page: int = 1, page_size: int = 50
    ) -> Optional[Dict[str, Any]]:
        """Get a specific dataset by ID with optional pagination.

        Parameters
        ----------
        dataset_id : str
            The dataset identifier.
        page : int
            1-based page number for pairs pagination.
        page_size : int
            Number of pairs per page.

        Returns
        -------
        dict or None
            The dataset with paginated pairs, or None if not found.
        """
        filepath = os.path.join(self.datasets_dir, f"{dataset_id}.json")
        if not os.path.isfile(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Paginate pairs
        all_pairs = data.get("pairs", [])
        total = len(all_pairs)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_pairs = all_pairs[start:end]

        return {
            "dataset_id": data.get("dataset_id", dataset_id),
            "generated_at": data.get("generated_at", ""),
            "total_pairs": total,
            "config_used": data.get("config_used", {}),
            "pairs": paginated_pairs,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a specific dataset.

        Parameters
        ----------
        dataset_id : str
            The dataset identifier.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        filepath = os.path.join(self.datasets_dir, f"{dataset_id}.json")
        if not os.path.isfile(filepath):
            return False
        os.remove(filepath)
        return True

    # ── Private helpers ───────────────────────────────────────────────

    def _load_ingested_data(self) -> List[Dict[str, Any]]:
        """Load the ingested data JSON backup."""
        if not os.path.isfile(self.json_path):
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _apply_filters(
        self, records: List[Dict[str, Any]], conditions: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Filter records by metadata conditions."""
        if not conditions:
            return records

        filtered = []
        for rec in records:
            meta = rec.get("metadata", {})
            match = True
            for key, value in conditions.items():
                if not value:  # Skip empty condition values
                    continue
                # Check in metadata first, then in top-level record fields
                rec_value = meta.get(key, rec.get(key, ""))
                if str(rec_value).lower() != str(value).lower():
                    match = False
                    break
            if match:
                filtered.append(rec)

        return filtered

    def _generate_from_ingested(
        self,
        records: List[Dict[str, Any]],
        num_pairs_per_chunk: int,
        difficulty_levels: List[str],
        include_context: bool,
    ) -> List[Dict[str, Any]]:
        """Generate test pairs from ingested data without LLM.

        For each record, creates variations:
        - Original Q&A pair (easy)
        - Paraphrased question with simple word reordering (medium)
        - Negative test with wrong answer paired with question (hard)
        """
        pairs: List[Dict[str, Any]] = []
        pair_counter = 0

        if not difficulty_levels:
            difficulty_levels = ["easy", "medium", "hard"]

        for rec in records:
            question = rec.get("question", "")
            answer = rec.get("answer", "")
            context = rec.get("context", "")
            record_id = rec.get("id", "")
            metadata = rec.get("metadata", {})

            if not question and not answer:
                continue

            generated_for_chunk = 0

            # 1. Original Q&A pair (easy)
            if generated_for_chunk < num_pairs_per_chunk and "easy" in difficulty_levels:
                pair_counter += 1
                pairs.append({
                    "id": f"pair_{pair_counter:04d}",
                    "question": question,
                    "expected_answer": answer,
                    "context": context if include_context else "",
                    "source_record_id": record_id,
                    "difficulty": "easy",
                    "type": "original",
                    "metadata": metadata,
                })
                generated_for_chunk += 1

            # 2. Paraphrased question (medium)
            if generated_for_chunk < num_pairs_per_chunk and "medium" in difficulty_levels:
                paraphrased = self._paraphrase_question(question)
                pair_counter += 1
                pairs.append({
                    "id": f"pair_{pair_counter:04d}",
                    "question": paraphrased,
                    "expected_answer": answer,
                    "context": context if include_context else "",
                    "source_record_id": record_id,
                    "difficulty": "medium",
                    "type": "paraphrased",
                    "metadata": metadata,
                })
                generated_for_chunk += 1

            # 3. Negative test (hard) — wrong answer paired with question
            if generated_for_chunk < num_pairs_per_chunk and "hard" in difficulty_levels:
                wrong_answer = self._get_wrong_answer(records, rec)
                pair_counter += 1
                pairs.append({
                    "id": f"pair_{pair_counter:04d}",
                    "question": question,
                    "expected_answer": answer,
                    "context": context if include_context else "",
                    "source_record_id": record_id,
                    "difficulty": "hard",
                    "type": "negative",
                    "metadata": {
                        **metadata,
                        "negative_answer": wrong_answer,
                        "is_negative_test": True,
                    },
                })
                generated_for_chunk += 1

            # 4. Additional variations if more pairs needed
            while generated_for_chunk < num_pairs_per_chunk:
                # Create additional paraphrased variants
                difficulty = random.choice(difficulty_levels)
                variant_q = self._create_variant_question(question, generated_for_chunk)
                pair_counter += 1
                pairs.append({
                    "id": f"pair_{pair_counter:04d}",
                    "question": variant_q,
                    "expected_answer": answer,
                    "context": context if include_context else "",
                    "source_record_id": record_id,
                    "difficulty": difficulty,
                    "type": "paraphrased",
                    "metadata": metadata,
                })
                generated_for_chunk += 1

        return pairs

    def _paraphrase_question(self, question: str) -> str:
        """Simple paraphrasing by reordering and rephrasing.

        This is a basic transformation without LLM. It applies simple
        heuristics to create a slightly different version of the question.
        """
        if not question:
            return question

        # Strategy 1: Convert question format
        q = question.strip()

        # Remove trailing question mark for processing
        has_qmark = q.endswith("?")
        if has_qmark:
            q = q[:-1].strip()

        words = q.split()
        if len(words) < 3:
            return question

        # Strategy: Rearrange common question patterns
        lower = q.lower()

        if lower.startswith("what is"):
            result = f"Can you explain {q[8:].strip()}"
        elif lower.startswith("what are"):
            result = f"Could you describe {q[9:].strip()}"
        elif lower.startswith("how does"):
            result = f"In what way does {q[9:].strip()}"
        elif lower.startswith("how do"):
            result = f"In what way do {q[7:].strip()}"
        elif lower.startswith("why is"):
            result = f"What is the reason {q[6:].strip()} is"
        elif lower.startswith("why do"):
            result = f"For what reason do {q[7:].strip()}"
        elif lower.startswith("when"):
            result = f"At what point {q[5:].strip()}"
        elif lower.startswith("where"):
            result = f"In what location {q[6:].strip()}"
        elif lower.startswith("who"):
            result = f"Which person or entity {q[4:].strip()}"
        elif lower.startswith("can you"):
            result = f"Is it possible to {q[8:].strip()}"
        elif lower.startswith("does"):
            result = f"Is it true that {q[5:].strip()}"
        elif lower.startswith("is"):
            result = f"Would you say that {q[3:].strip()}"
        else:
            # Default: prepend "Regarding" and restructure
            result = f"Regarding the following: {q}"

        return result + ("?" if has_qmark else "")

    def _create_variant_question(self, question: str, variant_idx: int) -> str:
        """Create additional question variants."""
        if not question:
            return question

        prefixes = [
            "Please explain: ",
            "Could you clarify: ",
            "Tell me about: ",
            "I'd like to know: ",
            "Help me understand: ",
        ]
        idx = variant_idx % len(prefixes)
        # Strip question mark, add prefix, re-add question mark
        q = question.strip()
        if q.endswith("?"):
            q = q[:-1].strip()
        return f"{prefixes[idx]}{q}?"

    def _get_wrong_answer(
        self, all_records: List[Dict[str, Any]], current_record: Dict[str, Any]
    ) -> str:
        """Get a wrong answer from a different record for negative testing."""
        current_answer = current_record.get("answer", "")
        other_answers = [
            r.get("answer", "")
            for r in all_records
            if r.get("answer", "") and r.get("answer", "") != current_answer
        ]
        if other_answers:
            return random.choice(other_answers)
        return "This information is not available in the knowledge base."

    def _save_dataset(self, dataset_id: str, dataset: Dict[str, Any]) -> None:
        """Save a dataset to the datasets directory."""
        os.makedirs(self.datasets_dir, exist_ok=True)
        filepath = os.path.join(self.datasets_dir, f"{dataset_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, default=str)

    def _get_api_key(self) -> str:
        """Get the LLM API key from project config."""
        config_path = os.path.join(self.project_dir, "config.json")
        if not os.path.isfile(config_path):
            return ""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("settings", {}).get("llm_config", {}).get("api_key", "")
        except (json.JSONDecodeError, OSError):
            return ""

    def _update_project_status(self) -> None:
        """Update the project config status to 'generated'."""
        config_path = os.path.join(self.project_dir, "config.json")
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["status"] = "generated"
            cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, default=str)
        except (json.JSONDecodeError, OSError):
            pass

    def get_metadata_fields(self) -> List[str]:
        """Get all unique metadata field names from ingested data.

        Returns
        -------
        list[str]
            Sorted list of unique metadata field names.
        """
        records = self._load_ingested_data()
        fields: set = set()
        for rec in records:
            meta = rec.get("metadata", {})
            if isinstance(meta, dict):
                fields.update(meta.keys())
        return sorted(fields)
