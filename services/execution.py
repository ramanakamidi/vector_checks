"""
Execution Service — runs test executions against the AUT and vector store.

Each execution:
1. Loads a dataset from ``projects/{project_id}/datasets/{dataset_id}.json``
2. Sends each question to the AUT endpoint
3. Compares AUT responses with expected answers
4. Optionally runs vector store health checks
5. Saves results to ``projects/{project_id}/executions/{execution_id}.json``

Usage::

    from services.execution import ExecutionService

    svc = ExecutionService("proj_abc123")
    result = svc.execute({
        "dataset_id": "ds_xyz",
        "aut_endpoint": "http://localhost:8001/api/query",
        "aut_project_id": "proj_abc123",
        "run_vector_checks": True,
        "checks_to_run": ["completeness", "consistency", "statistical", "neighbor"],
    })
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from services.paths import PROJECTS_DIR


class ExecutionService:
    """Runs test executions against the AUT and vector store."""

    def __init__(self, project_id: str, projects_dir: Optional[str] = None):
        if projects_dir is None:
            projects_dir = os.fspath(PROJECTS_DIR)
        self.project_id = project_id
        self.project_dir = os.path.join(projects_dir, project_id)
        self.executions_dir = os.path.join(self.project_dir, "executions")
        self.datasets_dir = os.path.join(self.project_dir, "datasets")

        if not os.path.isdir(self.project_dir):
            raise FileNotFoundError(
                f"Project directory not found: {self.project_dir}"
            )

        os.makedirs(self.executions_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────

    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a full test execution.

        Parameters
        ----------
        config : dict
            Configuration with keys:
            - dataset_id (str): ID of the dataset to test with
            - aut_endpoint (str): URL of the AUT query endpoint
            - aut_project_id (str): project_id to pass to the AUT
            - run_vector_checks (bool): whether to run vector store checks
            - checks_to_run (list[str]): which checks to run
            - collection_name (str, optional): Chroma collection name

        Returns
        -------
        dict
            Full execution results with summary, per-question results,
            and optional vector check results.
        """
        dataset_id = config.get("dataset_id", "")
        aut_endpoint = config.get("aut_endpoint", "http://localhost:8001/api/query")
        aut_project_id = config.get("aut_project_id", self.project_id)
        run_vector_checks = config.get("run_vector_checks", False)
        checks_to_run = config.get("checks_to_run", [])
        collection_name = config.get("collection_name", "")

        # 1. Load the dataset
        dataset = self._load_dataset(dataset_id)
        if dataset is None:
            return {
                "status": "error",
                "message": f"Dataset '{dataset_id}' not found.",
                "execution_id": None,
            }

        pairs = dataset.get("pairs", [])
        if not pairs:
            return {
                "status": "error",
                "message": "Dataset has no test pairs.",
                "execution_id": None,
            }

        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # 2. Execute each question against the AUT
        results: List[Dict[str, Any]] = []
        correct_count = 0
        total_confidence = 0.0
        total_context_relevance = 0.0
        total_response_time = 0.0

        for pair in pairs:
            question = pair.get("question", "")
            expected_answer = pair.get("expected_answer", "")
            pair_id = pair.get("id", "")
            metadata = pair.get("metadata", {})

            # Query the AUT
            q_start = time.time()
            aut_response = self._query_aut(question, aut_endpoint, aut_project_id)
            q_elapsed_ms = (time.time() - q_start) * 1000

            aut_answer = aut_response.get("answer", "")
            aut_confidence = aut_response.get("confidence", 0.0)
            aut_context = aut_response.get("context", [])
            aut_sources = aut_response.get("sources", [])

            # Compare answers
            keyword_overlap = self._compute_keyword_overlap(
                expected_answer, aut_answer
            )
            context_relevance = self._compute_context_relevance(
                question, aut_context
            )
            is_correct = keyword_overlap >= 0.5

            if is_correct:
                correct_count += 1
            total_confidence += aut_confidence
            total_context_relevance += context_relevance
            total_response_time += q_elapsed_ms

            results.append({
                "pair_id": pair_id,
                "question": question,
                "expected_answer": expected_answer,
                "aut_answer": aut_answer,
                "aut_confidence": round(aut_confidence, 4),
                "aut_context": aut_context,
                "aut_sources": aut_sources,
                "is_correct": is_correct,
                "keyword_overlap": round(keyword_overlap, 4),
                "context_relevance": round(context_relevance, 4),
                "response_time_ms": round(q_elapsed_ms, 1),
                "metadata": metadata,
            })

        # 3. Run vector store checks if requested
        vector_checks: Dict[str, Any] = {}
        if run_vector_checks and checks_to_run:
            vector_checks = self._run_vector_checks(
                collection_name or f"{self.project_id}_knowledge_base",
                checks_to_run,
            )

        # 4. Compute aggregate metrics
        total_questions = len(results)
        accuracy = correct_count / total_questions if total_questions > 0 else 0.0
        avg_confidence = (
            total_confidence / total_questions if total_questions > 0 else 0.0
        )
        avg_context_relevance = (
            total_context_relevance / total_questions if total_questions > 0 else 0.0
        )
        avg_response_time = (
            total_response_time / total_questions if total_questions > 0 else 0.0
        )

        duration_seconds = round(time.time() - start_time, 2)
        now = datetime.now(timezone.utc).isoformat()

        execution_data: Dict[str, Any] = {
            "execution_id": execution_id,
            "executed_at": now,
            "duration_seconds": duration_seconds,
            "dataset_id": dataset_id,
            "status": "completed",
            "summary": {
                "total_questions": total_questions,
                "correct_answers": correct_count,
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(avg_confidence, 4),
                "avg_context_relevance": round(avg_context_relevance, 4),
                "avg_response_time_ms": round(avg_response_time, 1),
                "aut_endpoint": aut_endpoint,
                "vector_store_health": vector_checks.get("summary", {}),
            },
            "results": results,
            "vector_checks": vector_checks,
        }

        # 5. Save execution results
        self._save_execution(execution_id, execution_data)

        # 6. Update project status
        self._update_project_status()

        return execution_data

    def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions for this project (summaries only)."""
        if not os.path.isdir(self.executions_dir):
            return []

        executions = []
        for filename in sorted(os.listdir(self.executions_dir), reverse=True):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.executions_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                executions.append({
                    "execution_id": data.get("execution_id", filename.replace(".json", "")),
                    "executed_at": data.get("executed_at", ""),
                    "duration_seconds": data.get("duration_seconds", 0),
                    "dataset_id": data.get("dataset_id", ""),
                    "status": data.get("status", "completed"),
                    "summary": data.get("summary", {}),
                })
            except (json.JSONDecodeError, OSError):
                continue

        return executions

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific execution result."""
        filepath = os.path.join(self.executions_dir, f"{execution_id}.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def delete_execution(self, execution_id: str) -> bool:
        """Delete an execution."""
        filepath = os.path.join(self.executions_dir, f"{execution_id}.json")
        if not os.path.isfile(filepath):
            return False
        os.remove(filepath)
        return True

    # ── Private helpers ───────────────────────────────────────────────

    def _load_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Load a dataset from the datasets directory."""
        filepath = os.path.join(self.datasets_dir, f"{dataset_id}.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _query_aut(self, question: str, endpoint: str, project_id: str) -> dict:
        """Send a question to the AUT and return the response."""
        try:
            resp = requests.post(
                endpoint,
                json={
                    "question": question,
                    "project_id": project_id,
                    "top_k": 3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {
                "answer": f"ERROR: {str(e)}",
                "confidence": 0,
                "context": [],
                "sources": [],
            }

    def _compute_keyword_overlap(self, expected: str, actual: str) -> float:
        """Simple keyword overlap between expected and actual answers."""
        if not expected:
            return 0.0
        # Remove common stop words for better comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when",
            "where", "how", "what", "which", "who", "whom", "this",
            "that", "these", "those", "it", "its",
        }
        exp_words = set(expected.lower().split()) - stop_words
        act_words = set(actual.lower().split()) - stop_words
        if not exp_words:
            return 1.0 if not act_words else 0.0
        return len(exp_words & act_words) / len(exp_words)

    def _compute_context_relevance(self, question: str, contexts: list) -> float:
        """Simple relevance score based on keyword overlap with question."""
        if not contexts:
            return 0.0
        q_words = set(question.lower().split())
        if not q_words:
            return 0.0
        scores = []
        for ctx in contexts:
            text = ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
            c_words = set(text.lower().split())
            scores.append(len(q_words & c_words) / len(q_words))
        return sum(scores) / len(scores) if scores else 0.0

    def _run_vector_checks(
        self, collection_name: str, checks_to_run: List[str]
    ) -> Dict[str, Any]:
        """Run vector store health checks using the project's Chroma store."""
        chroma_path = os.path.join(self.project_dir, "chroma_data")
        if not os.path.isdir(chroma_path):
            return {
                "status": "skipped",
                "message": "No Chroma data directory found.",
                "summary": {},
                "checks": [],
            }

        try:
            import chromadb

            client = chromadb.PersistentClient(path=chroma_path)
            try:
                collection = client.get_collection(collection_name)
            except Exception:
                # Try listing collections and use the first one
                collections = client.list_collections()
                if not collections:
                    return {
                        "status": "skipped",
                        "message": "No collections found in Chroma store.",
                        "summary": {},
                        "checks": [],
                    }
                collection = collections[0] if hasattr(collections[0], 'count') else client.get_collection(
                    collections[0].name if hasattr(collections[0], 'name') else str(collections[0])
                )

            count = collection.count()
            if count == 0:
                return {
                    "status": "skipped",
                    "message": "Collection is empty.",
                    "summary": {},
                    "checks": [],
                }

            # Run simplified checks directly on the collection
            check_results = []
            passed = 0
            failed = 0

            # Fetch a sample of records for checks
            sample_size = min(count, 500)
            sample = collection.get(
                limit=sample_size,
                include=["embeddings", "metadatas", "documents"],
            )

            ids = sample.get("ids", []) or []
            raw_embeddings = sample.get("embeddings", None)
            raw_metadatas = sample.get("metadatas", None)
            raw_documents = sample.get("documents", None)

            # Convert numpy arrays to plain Python lists to avoid
            # "truth value of an array is ambiguous" errors.
            # Chroma may return numpy.ndarray for embeddings.
            def _to_list(obj):
                """Safely convert numpy arrays or None to Python lists."""
                if obj is None:
                    return []
                try:
                    return list(obj)
                except TypeError:
                    return []

            embeddings = []
            for emb in _to_list(raw_embeddings):
                try:
                    embeddings.append([float(v) for v in emb])
                except (TypeError, ValueError):
                    embeddings.append(None)

            documents = _to_list(raw_documents)
            metadatas = _to_list(raw_metadatas)

            # ── Completeness check ────────────────────────────────
            if "completeness" in checks_to_run:
                missing_docs = sum(
                    1 for d in documents
                    if d is None or (isinstance(d, str) and not d.strip())
                )
                missing_meta = sum(
                    1 for m in metadatas
                    if m is None or (isinstance(m, dict) and len(m) == 0)
                )
                completeness_pass = missing_docs == 0 and missing_meta == 0
                if completeness_pass:
                    passed += 1
                else:
                    failed += 1
                check_results.append({
                    "check": "completeness",
                    "passed": completeness_pass,
                    "detail": {
                        "total_records": len(ids),
                        "missing_documents": missing_docs,
                        "missing_metadata": missing_meta,
                        "completeness_rate": round(
                            1 - (missing_docs + missing_meta) / max(len(ids) * 2, 1),
                            4,
                        ),
                    },
                })

            # ── Consistency check (dimension consistency) ─────────
            if "consistency" in checks_to_run and embeddings:
                dims = set()
                for emb in embeddings:
                    if emb is not None:
                        dims.add(len(emb))
                consistency_pass = len(dims) <= 1
                if consistency_pass:
                    passed += 1
                else:
                    failed += 1
                check_results.append({
                    "check": "consistency",
                    "passed": consistency_pass,
                    "detail": {
                        "total_vectors": len(embeddings),
                        "unique_dimensions": list(dims),
                        "is_consistent": consistency_pass,
                    },
                })

            # ── Statistical check (magnitude outliers) ────────────
            if "statistical" in checks_to_run and embeddings:
                import math

                norms = []
                for emb in embeddings:
                    if emb is not None:
                        norm = math.sqrt(sum(v * v for v in emb))
                        norms.append(norm)

                if norms:
                    mean_norm = sum(norms) / len(norms)
                    variance = sum((n - mean_norm) ** 2 for n in norms) / max(
                        len(norms) - 1, 1
                    )
                    std_norm = math.sqrt(variance)
                    z_threshold = 3.0
                    outliers = [
                        n
                        for n in norms
                        if std_norm > 0
                        and abs(n - mean_norm) / std_norm > z_threshold
                    ]
                    statistical_pass = len(outliers) == 0
                else:
                    mean_norm = 0.0
                    std_norm = 0.0
                    outliers = []
                    statistical_pass = True

                if statistical_pass:
                    passed += 1
                else:
                    failed += 1
                check_results.append({
                    "check": "statistical",
                    "passed": statistical_pass,
                    "detail": {
                        "total_vectors": len(norms),
                        "mean_norm": round(mean_norm, 4),
                        "std_norm": round(std_norm, 4),
                        "outlier_count": len(outliers),
                        "z_threshold": 3.0,
                    },
                })

            # ── Neighbor check (zero/NaN vectors) ─────────────────
            if "neighbor" in checks_to_run and embeddings:
                zero_count = 0
                nan_count = 0
                for emb in embeddings:
                    if emb is not None:
                        if all(v == 0 for v in emb):
                            zero_count += 1
                        if any(
                            v != v or v == float("inf") or v == float("-inf")
                            for v in emb
                        ):
                            nan_count += 1
                neighbor_pass = zero_count == 0 and nan_count == 0
                if neighbor_pass:
                    passed += 1
                else:
                    failed += 1
                check_results.append({
                    "check": "neighbor",
                    "passed": neighbor_pass,
                    "detail": {
                        "total_vectors": len(embeddings),
                        "zero_vectors": zero_count,
                        "nan_inf_vectors": nan_count,
                    },
                })

            return {
                "status": "completed",
                "summary": {
                    "total_checks": len(check_results),
                    "passed": passed,
                    "failed": failed,
                },
                "checks": check_results,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "summary": {},
                "checks": [],
            }

    def _save_execution(
        self, execution_id: str, data: Dict[str, Any]
    ) -> None:
        """Save execution results to the executions directory."""
        os.makedirs(self.executions_dir, exist_ok=True)
        filepath = os.path.join(self.executions_dir, f"{execution_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _update_project_status(self) -> None:
        """Update the project config status to 'executed'."""
        config_path = os.path.join(self.project_dir, "config.json")
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["status"] = "executed"
            cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, default=str)
        except (json.JSONDecodeError, OSError):
            pass
