"""Typed convenience methods for the repository's FastAPI service."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from checks.negative_queries import DEFAULT_MAX_SIMILARITY


class ApiError(RuntimeError):
    """An HTTP or transport error returned by the backend API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    """HTTP client for the FastAPI endpoints under ``/api``."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        files: Any = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.request(
                method.upper(),
                url,
                params=params,
                json=json_body,
                files=files,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach backend at {url}: {exc}") from exc

        if not response.ok:
            try:
                body = response.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                detail = response.text
            raise ApiError(f"API {response.status_code}: {detail}", response.status_code)

        if response.status_code == 204 or not response.content:
            return None
        if "json" in response.headers.get("content-type", "").lower():
            return response.json()
        return response.content

    @staticmethod
    def _id(value: str) -> str:
        return quote(str(value), safe="")

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def api_health(self) -> dict[str, Any]:
        return self.request("GET", "/api/health")

    def connect(self, db_type: str = "chroma", params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", "/api/connect", json_body={"db_type": db_type, "params": params or {}})

    def collections(self) -> dict[str, Any]:
        return self.request("GET", "/api/collections")

    def get_field_mapper(self) -> dict[str, Any]:
        return self.request("GET", "/api/field-mapper")

    def save_field_mapper(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/field-mapper", json_body=config)

    def run_checks(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", "/api/run-checks", json_body=config or {})

    def results(self) -> dict[str, Any]:
        return self.request("GET", "/api/results")

    def record(self, record_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/record/{self._id(record_id)}")

    def search_records(self, field: str, value: Any, limit: int = 50) -> dict[str, Any]:
        return self.request("POST", "/api/records/search", json_body={"field": field, "value": value, "limit": limit})

    def embedding_data(self, sample_size: int = 2000, method: str = "pca") -> dict[str, Any]:
        return self.request("GET", "/api/embedding-data", params={"sample_size": sample_size, "method": method})

    def auto_pairs(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", "/api/auto-pairs", json_body=config or {})

    def llm_review(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/llm-review", json_body=config)

    def export_html(self) -> bytes:
        return self.request("GET", "/api/export/html")

    def export_json(self) -> Any:
        return self.request("GET", "/api/export/json")

    def list_projects(self) -> dict[str, Any]:
        return self.request("GET", "/api/projects")

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        return self.request("POST", "/api/projects", json_body={"name": name, "description": description})

    def project(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}")

    def update_project(self, project_id: str, data: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/api/projects/{self._id(project_id)}", json_body=data)

    def delete_project(self, project_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/api/projects/{self._id(project_id)}")

    def project_settings(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/settings")

    def save_project_settings(self, project_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/api/projects/{self._id(project_id)}/settings", json_body=settings)

    def ingest_file(self, project_id: str, filename: str, content: bytes, content_type: str = "application/json") -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/projects/{self._id(project_id)}/ingest",
            files={"file": (filename, content, content_type)},
        )

    def ingest_json(self, project_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/projects/{self._id(project_id)}/ingest-json",
            json_body={"source": "upload", "data": records},
        )

    def ingest_pdfs(self, project_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/projects/{self._id(project_id)}/ingest-pdfs",
            files=[
                ("files", (filename, content, "application/pdf"))
                for filename, content in files
            ],
        )

    def ingested_data(self, project_id: str, page: int = 1, page_size: int = 50, search: str = "") -> dict[str, Any]:
        return self.request(
            "GET", f"/api/projects/{self._id(project_id)}/ingested-data",
            params={"page": page, "page_size": page_size, "search": search},
        )

    def project_vector_records(
        self,
        project_id: str,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/api/projects/{self._id(project_id)}/vector-records",
        )

    def clear_ingested_data(self, project_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/api/projects/{self._id(project_id)}/ingested-data")

    def ingestion_stats(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/ingestion-stats")

    def project_visualization_data(
        self,
        project_id: str,
        sample_size: int = 300,
        min_cluster_size: int = 2,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/api/projects/{self._id(project_id)}/visualization-data",
            params={
                "sample_size": sample_size,
                "min_cluster_size": min_cluster_size,
            },
        )

    def generate_dataset(self, project_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/api/projects/{self._id(project_id)}/generate-dataset", json_body=config)

    def datasets(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/datasets")

    def dataset(self, project_id: str, dataset_id: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return self.request(
            "GET", f"/api/projects/{self._id(project_id)}/datasets/{self._id(dataset_id)}",
            params={"page": page, "page_size": page_size},
        )

    def delete_dataset(self, project_id: str, dataset_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/api/projects/{self._id(project_id)}/datasets/{self._id(dataset_id)}")

    def metadata_fields(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/metadata-fields")

    def start_mock_aut(
        self,
        project_id: str | None = None,
        port: int = 8001,
        *,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/api/mock-aut/start",
            json_body={
                "project_id": project_id,
                "port": port,
                "api_key": api_key,
                "endpoint": endpoint,
                "deployment": deployment,
                "api_version": api_version,
            },
        )

    def stop_mock_aut(self) -> dict[str, Any]:
        return self.request("POST", "/api/mock-aut/stop")

    def mock_aut_status(self) -> dict[str, Any]:
        return self.request("GET", "/api/mock-aut/status")

    def query_mock_aut(
        self,
        question: str,
        project_id: str,
        top_k: int = 5,
        retrieved_contexts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "project_id": project_id,
            "top_k": top_k,
        }
        if retrieved_contexts is not None:
            payload["retrieved_contexts"] = retrieved_contexts
        return self.request(
            "POST",
            "/api/query",
            json_body=payload,
        )

    def execute(self, project_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/api/projects/{self._id(project_id)}/execute", json_body=config)

    def run_project_vector_checks(
        self,
        project_id: str,
        selected_checks: list[str],
        required_fields: list[str] | None = None,
        duplicate_min_similarity: float = 0.65,
        duplicate_max_similarity: float = 0.98,
        negative_query_texts: list[str] | None = None,
        negative_query_max_similarity: float = DEFAULT_MAX_SIMILARITY,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/projects/{self._id(project_id)}/run-vector-checks",
            json_body={
                "selected_checks": selected_checks,
                "required_fields": required_fields
                or ["pdf_name", "pdf_sha256", "page_number", "chunk_index"],
                "duplicate_min_similarity": duplicate_min_similarity,
                "duplicate_max_similarity": duplicate_max_similarity,
                "negative_query_texts": negative_query_texts or [],
                "negative_query_max_similarity": negative_query_max_similarity,
            },
        )

    def executions(self, project_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/executions")

    def execution(self, project_id: str, execution_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/executions/{self._id(execution_id)}")

    def delete_execution(self, project_id: str, execution_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/api/projects/{self._id(project_id)}/executions/{self._id(execution_id)}")

    def execution_report(self, project_id: str, execution_id: str) -> bytes:
        return self.request("GET", f"/api/projects/{self._id(project_id)}/executions/{self._id(execution_id)}/report")