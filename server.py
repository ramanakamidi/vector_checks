"""
FastAPI backend for the Vector Database Testing Framework.

Wraps the existing check/report framework as a REST API consumed by
the React frontend.  Start with::

    cd files_extracted
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

import os
import base64

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import huggingface_hub.file_download
import psutil

huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False

import sys
import math
import json
import traceback
import subprocess
import uuid
import socket
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

# Ensure the project root is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr
from starlette.concurrency import run_in_threadpool
from typing import Any, Dict, List, Literal, Optional

from connectors.base import VectorDBConnector, VectorRecord
from connectors.chroma_connector import ChromaConnector
from config import FieldMapper, ReportConfig
from run_checks import run_full_report
from checks.negative_queries import DEFAULT_MAX_SIMILARITY
from report.html_report import generate_report
from services.pdf_ingestion import ingest_pdf
from services.project_vector_checks import CHECK_CATEGORIES, run_project_checks
from services.project_visualization import build_project_visualization
from services.project_chroma import open_project_collection
from services import react_workbench

# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Vector DB Testing Framework",
    description="REST API for running vector-database health checks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level state ────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "connector": None,
    "field_mapper": FieldMapper(),
    "results": None,
    "fault_annotations": {},
}

# ── Pydantic request / response models ───────────────────────────────


class ConnectRequest(BaseModel):
    db_type: str = "chroma"
    params: Dict[str, Any] = {}


class FieldMapperRequest(BaseModel):
    doc_id_field: str = "doc_id"
    doc_name_field: Optional[str] = "doc_name"
    text_field: Optional[str] = "text"
    page_field: Optional[str] = "page_number"
    chunk_index_field: Optional[str] = "chunk_index"
    object_type_field: Optional[str] = "object_type"
    category_field: Optional[str] = None
    source_url_field: Optional[str] = "source_url"
    file_path_field: Optional[str] = "file_path"
    ingestion_date_field: Optional[str] = "ingestion_date"
    batch_id_field: Optional[str] = "batch_id"
    required_fields: List[str] = ["doc_id", "object_type"]
    display_fields: List[str] = [
        "doc_id", "doc_name", "page_number", "object_type", "text"
    ]
    similar_group_field: str = "doc_id"
    different_group_field: str = "object_type"
    drift_time_field: Optional[str] = "ingestion_date"


class RunChecksRequest(BaseModel):
    required_fields: List[str] = ["doc_id"]
    batch_size: int = 500
    sample_size: int = 500


class SearchRequest(BaseModel):
    field: str
    value: Any
    limit: int = 50


class WorkbenchQARequest(BaseModel):
    object_ids: List[str] = Field(default_factory=list)
    generate_objects: bool = True
    generate_text: bool = True
    object_question_count: int = Field(default=5, ge=1, le=20)
    text_question_count: int = Field(default=2, ge=1, le=3)
    api_key: SecretStr
    endpoint: str
    deployment: str
    api_version: str


class WorkbenchChunkingRequest(BaseModel):
    technique: Literal["heading_aware", "recursive", "sentence", "token", "semantic"]
    chunk_size: int = Field(default=512, ge=64, le=12000)
    overlap: int = Field(default=0, ge=0, le=4096)
    min_sentences_per_chunk: int = Field(default=1, ge=1, le=20)
    semantic_threshold: float = Field(default=0.8, ge=0.05, le=0.95)


class AutoPairsRequest(BaseModel):
    group_field: str = "doc_id"
    contrast_field: Optional[str] = "object_type"
    output_path: str = "auto_pairs.xlsx"
    similar_pairs_per_group: int = 3
    different_pairs_total: int = 20


# ── Helper utilities ──────────────────────────────────────────────────


def _require_connector() -> VectorDBConnector:
    """Return the active connector or raise 400."""
    conn = _state.get("connector")
    if conn is None:
        raise HTTPException(status_code=400, detail="Not connected to any database. Call POST /api/connect first.")
    return conn


def _vector_norm(vec: List[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(v * v for v in vec)) if vec else 0.0


def _build_fault_annotations(results: Dict) -> Dict[str, str]:
    """Walk check results and map record IDs → fault type strings."""
    annotations: Dict[str, str] = {}
    if not results:
        return annotations
    for r in results.get("results", []):
        check = r.get("check", "")
        detail = r.get("detail", {})
        if not isinstance(detail, dict):
            continue

        # Zero / NaN / Inf vectors
        if check == "zero_nan_vectors":
            for rid in detail.get("sample_zero_ids", []):
                annotations[str(rid)] = "zero_vector"
            for rec in detail.get("sample_zero_records", []):
                if isinstance(rec, dict):
                    annotations[str(rec.get("id", ""))] = "zero_vector"
            for rid in detail.get("sample_nan_ids", []):
                annotations[str(rid)] = "nan_vector"
            for rec in detail.get("sample_nan_records", []):
                if isinstance(rec, dict):
                    annotations[str(rec.get("id", ""))] = "nan_vector"
            for rid in detail.get("sample_inf_ids", []):
                annotations[str(rid)] = "inf_vector"
            for rec in detail.get("sample_inf_records", []):
                if isinstance(rec, dict):
                    annotations[str(rec.get("id", ""))] = "inf_vector"

        # High-cosine neighbor candidates are not confirmed duplicates.
        elif check == "near_duplicate_detection":
            for p in detail.get(
                "similarity_candidates",
                detail.get("duplicate_pairs", []),
            ):
                if isinstance(p, dict):
                    annotations[str(p.get("id_a", ""))] = "similarity_candidate"
                    annotations[str(p.get("id_b", ""))] = "similarity_candidate"

        # Empty vectors
        elif check == "empty_vector_audit":
            for rid in detail.get("sample_ids", []):
                annotations[str(rid)] = "empty_vector"
            for rec in detail.get("sample_records", []):
                if isinstance(rec, dict):
                    annotations[str(rec.get("id", ""))] = "empty_vector"

        # Metadata completeness
        elif check == "metadata_completeness":
            for fld_info in detail.get("per_field", {}).values():
                for rid in fld_info.get("sample_ids", []):
                    annotations.setdefault(str(rid), "missing_metadata")
                for rec in fld_info.get("sample_records", []):
                    if isinstance(rec, dict):
                        annotations.setdefault(str(rec.get("id", "")), "missing_metadata")

    return annotations


def _build_fault_details(record_id: str, vec: List[float], results: Dict) -> List[Dict[str, str]]:
    """Build a list of expected-vs-actual fault detail dicts for a record.

    Scans through check results to find every check that flagged this
    specific record and produces human-readable expected/actual strings.
    """
    fault_details: List[Dict[str, str]] = []
    if not results:
        return fault_details

    rid = str(record_id)
    vector_dim = len(vec) if vec else 0

    for r in results.get("results", []):
        check_name = r.get("check", "")
        detail = r.get("detail", {})
        if not isinstance(detail, dict):
            continue

        # ── Zero / NaN / Inf vectors ──────────────────────────────
        if check_name == "zero_nan_vectors":
            # Collect all zero IDs from both plain-id and record lists
            zero_id_set = set(str(x) for x in detail.get("sample_zero_ids", []))
            for rec in detail.get("sample_zero_records", []):
                if isinstance(rec, dict):
                    zero_id_set.add(str(rec.get("id", "")))

            if rid in zero_id_set:
                fault_details.append({
                    "check": "Zero Vector",
                    "expected": "Non-zero vector (valid embedding from model)",
                    "actual": f"All-zero vector ({vector_dim} dimensions, all 0.0)",
                    "severity": "error",
                })

            nan_id_set = set(str(x) for x in detail.get("sample_nan_ids", []))
            for rec in detail.get("sample_nan_records", []):
                if isinstance(rec, dict):
                    nan_id_set.add(str(rec.get("id", "")))

            if rid in nan_id_set:
                fault_details.append({
                    "check": "NaN Vector",
                    "expected": "Finite numeric values in all dimensions",
                    "actual": "Vector contains NaN (Not a Number) values",
                    "severity": "error",
                })

            inf_id_set = set(str(x) for x in detail.get("sample_inf_ids", []))
            for rec in detail.get("sample_inf_records", []):
                if isinstance(rec, dict):
                    inf_id_set.add(str(rec.get("id", "")))

            if rid in inf_id_set:
                fault_details.append({
                    "check": "Inf Vector",
                    "expected": "Finite numeric values in all dimensions",
                    "actual": "Vector contains Infinity values",
                    "severity": "error",
                })

        # ── Dimension consistency ─────────────────────────────────
        elif check_name == "dimension_consistency":
            if not detail.get("dimensions_found"):
                continue  # passed — single dimension

            # Find the majority dimension
            dims_found = detail.get("dimensions_found", {})
            if dims_found:
                expected_dim = max(dims_found, key=lambda d: dims_found[d])
            else:
                continue

            # Check sample_ids_per_dimension for this record
            sample_ids_per_dim = detail.get("sample_ids_per_dimension", {})
            sample_recs_per_dim = detail.get("sample_records_per_dimension", {})

            for dim_str, id_list in sample_ids_per_dim.items():
                actual_dim = int(dim_str) if isinstance(dim_str, str) and dim_str.isdigit() else dim_str
                if actual_dim == expected_dim:
                    continue  # skip majority dimension
                if rid in [str(x) for x in id_list]:
                    fault_details.append({
                        "check": "Dimension Mismatch",
                        "expected": f"Vector dimension = {expected_dim} (majority dimension)",
                        "actual": f"Vector dimension = {actual_dim}",
                        "severity": "error",
                    })
                    break

            # Also check sample_records_per_dimension
            if not any(fd["check"] == "Dimension Mismatch" for fd in fault_details):
                for dim_str, rec_list in sample_recs_per_dim.items():
                    actual_dim = int(dim_str) if isinstance(dim_str, str) and dim_str.isdigit() else dim_str
                    if actual_dim == expected_dim:
                        continue
                    for rec in rec_list:
                        if isinstance(rec, dict) and str(rec.get("id", "")) == rid:
                            fault_details.append({
                                "check": "Dimension Mismatch",
                                "expected": f"Vector dimension = {expected_dim} (majority dimension)",
                                "actual": f"Vector dimension = {actual_dim}",
                                "severity": "error",
                            })
                            break

        # ── Empty vectors ─────────────────────────────────────────
        elif check_name == "empty_vector_audit":
            empty_id_set = set(str(x) for x in detail.get("sample_ids", []))
            for rec in detail.get("sample_records", []):
                if isinstance(rec, dict):
                    empty_id_set.add(str(rec.get("id", "")))

            if rid in empty_id_set:
                fault_details.append({
                    "check": "Empty Vector",
                    "expected": "Non-empty vector with valid embeddings",
                    "actual": "Empty vector (no data)",
                    "severity": "error",
                })

        # ── Metadata completeness ─────────────────────────────────
        elif check_name == "metadata_completeness":
            per_field = detail.get("per_field", {})
            for field_name, fld_info in per_field.items():
                if not isinstance(fld_info, dict):
                    continue
                field_ids = set(str(x) for x in fld_info.get("sample_ids", []))
                for rec in fld_info.get("sample_records", []):
                    if isinstance(rec, dict):
                        field_ids.add(str(rec.get("id", "")))

                if rid in field_ids:
                    fault_details.append({
                        "check": "Missing Metadata",
                        "expected": f"Required field '{field_name}' present and non-empty",
                        "actual": f"Field '{field_name}' is missing or empty",
                        "severity": "warning",
                    })

        # ── High-cosine neighbor candidates ────────────────────────
        elif check_name == "near_duplicate_detection":
            minimum_similarity = detail.get("minimum_similarity", 0.65)
            maximum_similarity = detail.get("maximum_similarity", 0.98)
            for p in detail.get(
                "similarity_candidates",
                detail.get("duplicate_pairs", []),
            ):
                if not isinstance(p, dict):
                    continue
                id_a = str(p.get("id_a", ""))
                id_b = str(p.get("id_b", ""))
                sim = p.get("cosine_similarity", 0.0)

                if rid == id_a:
                    fault_details.append({
                        "check": "Similarity Candidate",
                        "expected": (
                            "Review text; cosine similarity range is "
                            f"{minimum_similarity:.2f}–{maximum_similarity:.2f}"
                        ),
                        "actual": f"Cosine similarity = {sim:.4f} with record '{id_b}'",
                        "severity": "info",
                    })
                elif rid == id_b:
                    fault_details.append({
                        "check": "Similarity Candidate",
                        "expected": (
                            "Review text; cosine similarity range is "
                            f"{minimum_similarity:.2f}–{maximum_similarity:.2f}"
                        ),
                        "actual": f"Cosine similarity = {sim:.4f} with record '{id_a}'",
                        "severity": "info",
                    })

    return fault_details


# ── API Endpoints ─────────────────────────────────────────────────────


@app.post("/api/connect")
async def connect_db(req: ConnectRequest):
    """Connect to a vector database."""
    try:
        db_type = req.db_type.lower()
        params = req.params

        if db_type == "chroma":
            from connectors.chroma_connector import ChromaConnector
            connector = ChromaConnector(
                collection_name=params.get("collection_name", "default"),
                persist_directory=params.get("persist_directory"),
                host=params.get("host"),
                port=params.get("port", 8000),
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported db_type '{db_type}'. Currently supported: chroma",
            )

        connector.connect()
        _state["connector"] = connector
        _state["results"] = None
        _state["fault_annotations"] = {}

        # Gather collection list if possible
        collections: List[str] = []
        try:
            if hasattr(connector, "client") and connector.client is not None:
                raw = connector.client.list_collections()
                # chromadb >=0.4 returns Collection objects; older returns strings
                for c in raw:
                    collections.append(c.name if hasattr(c, "name") else str(c))
        except Exception:
            collections = [params.get("collection_name", "default")]

        return {
            "status": "connected",
            "record_count": connector.count(),
            "collections": collections,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/collections")
async def list_collections():
    """List available collections (requires prior connect)."""
    try:
        conn = _require_connector()
        collections: List[str] = []
        if hasattr(conn, "client") and conn.client is not None:
            raw = conn.client.list_collections()
            for c in raw:
                collections.append(c.name if hasattr(c, "name") else str(c))
        return {"collections": collections}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/field-mapper")
async def get_field_mapper():
    """Return the current FieldMapper configuration."""
    try:
        fm = _state["field_mapper"]
        return fm.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/field-mapper")
async def save_field_mapper(req: FieldMapperRequest):
    """Save a new FieldMapper configuration."""
    try:
        data = req.model_dump()
        _state["field_mapper"] = FieldMapper.from_dict(data)
        return {"status": "saved"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run-checks")
async def run_checks(req: Optional[RunChecksRequest] = Body(default=None)):
    """Run all health checks against the connected database."""
    try:
        conn = _require_connector()
        fm = _state["field_mapper"]

        if req is None:
            req = RunChecksRequest()

        report = run_full_report(
            connector=conn,
            required_metadata_fields=req.required_fields,
            batch_size=req.batch_size,
            sample_size=req.sample_size,
            field_mapper=fm,
        )

        _state["results"] = report
        _state["fault_annotations"] = _build_fault_annotations(report)

        return report
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/results")
async def get_results():
    """Return the latest check results (if any)."""
    try:
        results = _state.get("results")
        if results is None:
            raise HTTPException(status_code=404, detail="No check results available. Run POST /api/run-checks first.")
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/record/{record_id}")
async def get_record(record_id: str):
    """Fetch a single record by ID with full metadata and vector info."""
    try:
        conn = _require_connector()
        record = conn.fetch(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

        vec = record.vector or []
        meta = dict(record.metadata) if record.metadata else {}

        # Include document text in metadata if the connector stored it separately
        doc_text = getattr(record, "document", None)
        if doc_text and "document" not in meta:
            meta["document"] = doc_text

        # Determine which checks flagged this record
        fault_types = []
        results = _state.get("results")
        if results and "results" in results:
            for check_result in results["results"]:
                detail = check_result.get("detail", {})
                failed_records = detail.get("failed_records", detail.get("records", detail.get("items", [])))
                if isinstance(failed_records, list):
                    for fr in failed_records:
                        fr_id = str(fr.get("id", fr.get("record_id", fr.get("doc_id", ""))))
                        if fr_id == record_id:
                            fault_types.append(check_result.get("check", "unknown"))
                            break

        # Build detailed expected-vs-actual fault information
        fault_details = _build_fault_details(record_id, vec, results) if results else []

        return {
            "id": record.id,
            "vector_preview": vec[:10],
            "vector_dim": len(vec),
            "vector_norm": round(_vector_norm(vec), 6),
            "metadata": meta,
            "fault_types": fault_types,
            "fault_details": fault_details,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/records/search")
async def search_records(req: SearchRequest):
    """Search records by metadata field value."""
    try:
        conn = _require_connector()
        fm = _state["field_mapper"]
        records: List[Dict[str, Any]] = []
        count = 0
        for rec in conn.filter_metadata(req.field, req.value):
            records.append({
                "id": rec.id,
                "metadata": fm.extract_display_metadata(rec.metadata),
                "vector_dim": len(rec.vector) if rec.vector else 0,
            })
            count += 1
            if count >= req.limit:
                break
        return {"records": records, "total_returned": len(records)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/embedding-data")
async def get_embedding_data(
    sample_size: int = Query(default=2000, ge=1, le=50000),
    method: str = Query(default="pca", pattern="^(pca|random)$"),
):
    """Return 2D-projected embedding data for the Plotly scatter plot.

    Samples vectors from the DB, projects them to 2D via PCA, and
    annotates each point with fault type if check results exist.
    """
    try:
        conn = _require_connector()
        fm = _state["field_mapper"]
        annotations = _state.get("fault_annotations", {})

        # Collect vectors (up to sample_size)
        ids: List[str] = []
        vectors: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        count = 0
        for rec in conn.scan():
            if rec.vector and len(rec.vector) > 0:
                ids.append(rec.id)
                vectors.append(rec.vector)
                metadatas.append(fm.extract_display_metadata(rec.metadata))
            count += 1
            if count >= sample_size:
                break

        if len(vectors) < 2:
            return {"points": [], "method": method.upper(), "message": "Not enough vectors to project."}

        # PCA projection to 2D
        points_2d = _pca_project(vectors)

        # Build response points
        points = []
        for i, rid in enumerate(ids):
            fault = annotations.get(str(rid), None)
            points.append({
                "id": rid,
                "x": round(points_2d[i][0], 6),
                "y": round(points_2d[i][1], 6),
                "metadata": metadatas[i],
                "fault_type": fault,
            })

        return {"points": points, "method": "PCA"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _pca_project(vectors: List[List[float]]) -> List[List[float]]:
    """Project vectors to 2D using a simple PCA implementation.

    Uses numpy if available, otherwise falls back to a pure-Python
    power-iteration approach (slower but dependency-free).
    """
    try:
        import numpy as np
        mat = np.array(vectors, dtype=np.float64)
        # Center
        mean = mat.mean(axis=0)
        centered = mat - mean
        # Covariance (use smaller dimension for efficiency)
        n, d = centered.shape
        if n < d:
            # Compute n×n gram matrix, get eigenvectors, project back
            gram = centered @ centered.T
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            # Take top 2 eigenvectors (sorted descending)
            idx = np.argsort(eigenvalues)[::-1][:2]
            top_vecs = eigenvectors[:, idx]
            # Project back to d-dimensional space
            components = (centered.T @ top_vecs).T
            # Normalise components
            for i in range(components.shape[0]):
                norm = np.linalg.norm(components[i])
                if norm > 0:
                    components[i] /= norm
            projected = centered @ components.T
        else:
            cov = (centered.T @ centered) / max(n - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1][:2]
            components = eigenvectors[:, idx].T
            projected = centered @ components.T

        return projected.tolist()
    except ImportError:
        # Pure-Python fallback
        return _pca_project_pure(vectors)


def _pca_project_pure(vectors: List[List[float]]) -> List[List[float]]:
    """Pure-Python PCA fallback using power iteration for top-2 components."""
    import random

    n = len(vectors)
    d = len(vectors[0])

    # Center the data
    mean = [0.0] * d
    for v in vectors:
        for j in range(d):
            mean[j] += v[j]
    for j in range(d):
        mean[j] /= n

    centered = []
    for v in vectors:
        centered.append([v[j] - mean[j] for j in range(d)])

    def dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def mat_vec(mat: List[List[float]], vec: List[float]) -> List[float]:
        return [dot(row, vec) for row in mat]

    def vec_norm(vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def normalize(vec: List[float]) -> List[float]:
        nm = vec_norm(vec)
        if nm == 0:
            return vec
        return [x / nm for x in vec]

    # Compute X^T X (d×d) — only feasible for moderate d
    # For very high d, just return random projection
    if d > 1000:
        random.seed(42)
        result = []
        for v in centered:
            result.append([random.gauss(0, 1), random.gauss(0, 1)])
        return result

    xtx = [[0.0] * d for _ in range(d)]
    for v in centered:
        for i in range(d):
            for j in range(i, d):
                val = v[i] * v[j]
                xtx[i][j] += val
                if i != j:
                    xtx[j][i] += val

    components = []
    for comp_idx in range(2):
        # Power iteration
        random.seed(42 + comp_idx)
        vec = normalize([random.gauss(0, 1) for _ in range(d)])
        for _ in range(100):
            new_vec = mat_vec(xtx, vec)
            # Deflate against previous components
            for prev in components:
                proj = dot(new_vec, prev)
                new_vec = [new_vec[j] - proj * prev[j] for j in range(d)]
            vec = normalize(new_vec)
        components.append(vec)

    # Project
    result = []
    for v in centered:
        result.append([dot(v, components[0]), dot(v, components[1])])
    return result


@app.post("/api/auto-pairs")
async def generate_auto_pairs(req: Optional[AutoPairsRequest] = Body(default=None)):
    """Generate auto pairs from metadata grouping."""
    try:
        conn = _require_connector()

        if req is None:
            req = AutoPairsRequest()

        from checks.auto_pairs import generate_pairs_from_metadata

        result = generate_pairs_from_metadata(
            connector=conn,
            group_field=req.group_field,
            contrast_field=req.contrast_field,
            output_path=req.output_path,
            similar_pairs_per_group=req.similar_pairs_per_group,
            different_pairs_total=req.different_pairs_total,
        )

        return {
            "statistics": result.get("statistics", result),
            "output_path": req.output_path,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/llm-review")
async def llm_review(request: dict):
    """Run LLM review on auto-generated pairs."""
    input_path = request.get("input_path", "auto_pairs.xlsx")
    output_path = request.get("output_path", "reviewed_pairs.xlsx")
    model = request.get("model", "gpt-4")

    from checks.llm_review import review_pairs_with_llm, create_openai_llm

    try:
        llm_fn = create_openai_llm(model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = review_pairs_with_llm(
        input_path=input_path,
        output_path=output_path,
        llm_fn=llm_fn,
    )
    return result


@app.get("/api/export/html")
async def export_html():
    """Download the HTML report for the latest check results."""
    try:
        results = _state.get("results")
        if results is None:
            raise HTTPException(status_code=404, detail="No check results available. Run checks first.")

        conn = _state.get("connector")
        db_label = "Vector Database"
        if conn is not None and hasattr(conn, "collection_name"):
            db_label = f"Chroma / {conn.collection_name}"

        html_content = generate_report(
            check_results=results.get("results", []),
            db_label=db_label,
        )

        return HTMLResponse(
            content=html_content,
            headers={
                "Content-Disposition": "attachment; filename=vector_db_report.html",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/export/json")
async def export_json():
    """Download the JSON results for the latest check results."""
    try:
        results = _state.get("results")
        if results is None:
            raise HTTPException(status_code=404, detail="No check results available. Run checks first.")

        return JSONResponse(
            content=results,
            headers={
                "Content-Disposition": "attachment; filename=vector_db_results.json",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Health check ──────────────────────────────────────────────────────


@app.get("/health")
@app.get("/api/health")
async def health():
    """Simple health check endpoint."""
    connected = _state.get("connector") is not None
    has_results = _state.get("results") is not None
    return {
        "status": "ok",
        "connected": connected,
        "has_results": has_results,
    }


# ══════════════════════════════════════════════════════════════════════
# ── PROJECT-BASED APIs (Phase 1 — Foundation) ─────────────────────────
# ══════════════════════════════════════════════════════════════════════

from services.project_manager import ProjectManager
from services.ingestion import IngestionService
from services.data_generation import DataGenerationService
from services.execution import ExecutionService
from services.report_generator import ReportGenerator

_pm = ProjectManager()


# ── Pydantic models for project APIs ─────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class LLMConfigModel(BaseModel):
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4"


class AUTConfigModel(BaseModel):
    url: str = "http://localhost:8000/mock-aut"
    method: str = "POST"
    endpoint: str = "/ask"
    headers: Dict[str, str] = {}
    request_template: Dict[str, Any] = {"question": "{{query}}"}
    response_answer_path: str = "answer"
    response_context_path: str = "context"


class VectorStoreConfigModel(BaseModel):
    db_type: str = "chroma"
    collection_name: str = ""
    persist_directory: str = ""


class ProjectSettingsModel(BaseModel):
    files_directory: Optional[str] = None
    extraction_mode: Optional[str] = None
    llm_config: Optional[LLMConfigModel] = None
    aut_config: Optional[AUTConfigModel] = None
    vector_store_config: Optional[VectorStoreConfigModel] = None
    field_mapper: Optional[Dict[str, Any]] = None


# ── Project CRUD endpoints ────────────────────────────────────────────


@app.get("/api/projects")
async def list_projects():
    """List all projects."""
    try:
        return {"projects": _pm.list_all()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    """Create a new project."""
    try:
        project = _pm.create(req.name, req.description)
        return project
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get full project details."""
    try:
        config = _pm.get(project_id)
        # Mask the API key in the response
        settings = config.get("settings", {})
        llm = settings.get("llm_config", {})
        if llm.get("api_key"):
            llm_copy = dict(llm)
            llm_copy["api_key_set"] = True
            llm_copy["api_key"] = "***"
            settings_copy = dict(settings)
            settings_copy["llm_config"] = llm_copy
            config_copy = dict(config)
            config_copy["settings"] = settings_copy
            return config_copy
        return config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest):
    """Update project name/description."""
    try:
        config = _pm.update(project_id, name=req.name, description=req.description)
        return {"status": "updated", "id": config["id"]}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its data."""
    try:
        _pm.delete(project_id)
        return {"status": "deleted", "id": project_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Project Settings endpoints ────────────────────────────────────────


@app.get("/api/projects/{project_id}/settings")
async def get_project_settings(project_id: str):
    """Get current settings for a project."""
    try:
        settings = _pm.get_settings(project_id)
        # Mask API key
        llm = settings.get("llm_config", {})
        if llm.get("api_key"):
            llm["api_key_set"] = True
            llm["api_key"] = "***"
        return settings
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/projects/{project_id}/settings")
async def save_project_settings(project_id: str, req: ProjectSettingsModel):
    """Save settings for a project (partial update)."""
    try:
        update_data = {}
        if req.files_directory is not None:
            update_data["files_directory"] = req.files_directory
        if req.extraction_mode is not None:
            update_data["extraction_mode"] = req.extraction_mode
        if req.llm_config is not None:
            llm_dict = req.llm_config.model_dump()
            # If api_key is "***", don't overwrite the stored key
            if llm_dict.get("api_key") == "***":
                existing = _pm.get_settings(project_id)
                llm_dict["api_key"] = existing.get("llm_config", {}).get("api_key", "")
            update_data["llm_config"] = llm_dict
        if req.aut_config is not None:
            update_data["aut_config"] = req.aut_config.model_dump()
        if req.vector_store_config is not None:
            update_data["vector_store_config"] = req.vector_store_config.model_dump()
        if req.field_mapper is not None:
            update_data["field_mapper"] = req.field_mapper

        _pm.update_settings(project_id, update_data)
        return {"status": "saved", "project_id": project_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════
# ── INGESTION APIs (Phase 3 — Document Ingestion) ─────────────────────
# ══════════════════════════════════════════════════════════════════════


class IngestBodyRequest(BaseModel):
    """Accept JSON body with records directly."""
    source: Optional[str] = "upload"
    data: List[Dict[str, Any]] = []
    file_path: Optional[str] = None


class ProjectVectorChecksRequest(BaseModel):
    selected_checks: List[str]
    required_fields: List[str] = Field(
        default_factory=lambda: ["pdf_name", "pdf_sha256", "page_number", "chunk_index"]
    )
    duplicate_min_similarity: float = Field(default=0.65, ge=0.65, le=0.98)
    duplicate_max_similarity: float = Field(default=0.98, ge=0.65, le=0.98)
    negative_query_texts: List[str] = Field(
        default_factory=list,
        max_length=100,
    )
    negative_query_max_similarity: float = Field(
        default=DEFAULT_MAX_SIMILARITY,
        ge=0.0,
        le=1.0,
    )
    synthetic_question_record_id: Optional[str] = None
    synthetic_question_texts: List[str] = Field(default_factory=list, max_length=3)
    synthetic_question_top_k: int = Field(default=1, ge=1, le=100)


@app.get("/api/projects/{project_id}/vector-records")
async def get_project_vector_records(project_id: str):
    """Return all text records from the project's Chroma collection."""
    try:
        collection = open_project_collection(project_id)
        total = collection.count()
        records = []
        batch_size = 500
        for offset in range(0, total, batch_size):
            data = collection.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            for index, record_id in enumerate(data.get("ids") or []):
                records.append({
                    "id": str(record_id),
                    "text": (
                        documents[index]
                        if index < len(documents) and documents[index] is not None
                        else ""
                    ),
                    "metadata": (
                        metadatas[index] or {}
                        if index < len(metadatas)
                        else {}
                    ),
                })
        return {
            "records": records,
            "total": len(records),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/ingest")
async def ingest_data(
    project_id: str,
    file: Optional[UploadFile] = File(None),
    body: Optional[str] = None,
):
    """Ingest Q&A data into a project-specific Chroma vector store.

    Accepts either:
    - A JSON file upload (multipart form)
    - A JSON body with the data directly (application/json)
    """
    try:
        svc = IngestionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    records: List[Dict[str, Any]] = []

    try:
        if file is not None:
            # File upload path
            content = await file.read()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict) and "data" in parsed:
                records = parsed["data"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must be an array of records or an object with a 'data' array",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No file uploaded. Send a JSON file as multipart form data.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse input: {exc}")

    try:
        result = await run_in_threadpool(svc.ingest_json, records)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/ingest-json")
async def ingest_json_body(project_id: str, req: IngestBodyRequest):
    """Ingest Q&A data from a JSON body (alternative to file upload)."""
    try:
        svc = IngestionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    records = req.data
    if not records and req.file_path:
        # Try loading from a file path on the server
        try:
            file_path = req.file_path
            if not os.path.isabs(file_path):
                file_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), file_path
                )
            with open(file_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict) and "data" in parsed:
                records = parsed["data"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="File JSON must be an array or object with 'data' array",
                )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in file: {e}")

    if not records:
        raise HTTPException(status_code=400, detail="No records provided")

    try:
        result = await run_in_threadpool(svc.ingest_json, records)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/ingested-data")
async def get_ingested_data(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    search: str = Query(default=""),
):
    """Return paginated ingested data."""
    try:
        svc = IngestionService(project_id)
        return svc.get_records(page=page, page_size=page_size, search=search or None)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/projects/{project_id}/ingested-data")
async def delete_ingested_data(project_id: str):
    """Clear all ingested data (Chroma collection + JSON file)."""
    try:
        svc = IngestionService(project_id)
        return svc.clear_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/ingestion-stats")
async def get_ingestion_stats(project_id: str):
    """Return statistics about the ingested data."""
    try:
        svc = IngestionService(project_id)
        return svc.get_stats()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════
# ── DATA GENERATION APIs (Phase 4 — Data Generation) ──────────────────
# ══════════════════════════════════════════════════════════════════════


class DatasetConfigRequest(BaseModel):
    """Request body for dataset generation."""
    num_pairs_per_chunk: int = 3
    total_limit: int = 100
    query_conditions: Dict[str, str] = {}
    include_context: bool = True
    difficulty_levels: List[str] = ["easy", "medium", "hard"]
    generation_mode: str = "from_ingested"


@app.post("/api/projects/{project_id}/generate-dataset")
async def generate_dataset(project_id: str, req: DatasetConfigRequest):
    """Generate a new test dataset from ingested data."""
    try:
        svc = DataGenerationService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        config = req.model_dump()
        result = svc.generate_dataset(config)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Generation failed"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/datasets")
async def list_datasets(project_id: str):
    """List all generated datasets for a project."""
    try:
        svc = DataGenerationService(project_id)
        datasets = svc.list_datasets()
        return {"datasets": datasets}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/datasets/{dataset_id}")
async def get_dataset(
    project_id: str,
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    """Get a specific dataset with paginated pairs."""
    try:
        svc = DataGenerationService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        dataset = svc.get_dataset(dataset_id, page=page, page_size=page_size)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        return dataset
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/projects/{project_id}/datasets/{dataset_id}")
async def delete_dataset(project_id: str, dataset_id: str):
    """Delete a specific dataset."""
    try:
        svc = DataGenerationService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        deleted = svc.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        return {"status": "deleted", "dataset_id": dataset_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/metadata-fields")
async def get_metadata_fields(project_id: str):
    """Get available metadata fields from ingested data (for query condition dropdowns)."""
    try:
        svc = DataGenerationService(project_id)
        fields = svc.get_metadata_fields()
        return {"fields": fields}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/visualization-data")
async def get_project_visualization_data(
    project_id: str,
    sample_size: int = Query(default=300, ge=3, le=10000),
    min_cluster_size: int = Query(default=2, ge=2, le=100),
):
    """Return UMAP coordinates, HDBSCAN labels, and record details for a project."""
    try:
        return await run_in_threadpool(
            build_project_visualization,
            project_id,
            sample_size,
            min_cluster_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/visualization-data/with-qa")
async def get_project_visualization_with_qa(
    project_id: str,
    file: UploadFile = File(...),
    sample_size: int = Form(300, ge=3, le=10000),
    min_cluster_size: int = Form(2, ge=2, le=100),
):
    """Return the project UMAP points with matching PDF Workbench Q&A rows."""
    from services.visual_qa import load_workbench_qa_workbook, match_chunk_qa

    try:
        workbook_bytes = await file.read()
        qa_rows = load_workbench_qa_workbook(workbook_bytes)
        result = await run_in_threadpool(
            build_project_visualization,
            project_id,
            sample_size,
            min_cluster_size,
        )
        for point in result.get("points", []):
            point["qa"] = match_chunk_qa(point, qa_rows)
        result["qa_matched_points"] = sum(
            bool(point.get("qa")) for point in result.get("points", [])
        )
        result["qa_rows_loaded"] = len(qa_rows)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/workbench/analyze")
async def analyze_workbench_pdfs(
    files: List[UploadFile] = File(...),
    mode: str = Form("crop"),
    technique: str = Form("heading_aware"),
    chunk_size: int = Form(2600),
    overlap: int = Form(250),
    min_sentences_per_chunk: int = Form(1),
    semantic_threshold: float = Form(0.8),
    api_key: str = Form(...),
    endpoint: str = Form(...),
    deployment: str = Form(...),
    api_version: str = Form(...),
):
    """Analyze PDFs and keep the interactive workbench data for this API process."""
    uploads = []
    try:
        for uploaded in files:
            uploads.append((
                os.path.basename(uploaded.filename or "upload.pdf"),
                await uploaded.read(),
            ))
        chunking = WorkbenchChunkingRequest(
            technique=technique,
            chunk_size=chunk_size,
            overlap=overlap,
            min_sentences_per_chunk=min_sentences_per_chunk,
            semantic_threshold=semantic_threshold,
        ).model_dump()
        credentials = {
            "api_key": api_key,
            "endpoint": endpoint,
            "deployment": deployment,
            "api_version": api_version,
        }
        return await run_in_threadpool(
            react_workbench.analyze_pdfs,
            uploads,
            mode,
            chunking,
            credentials,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"PDF analysis or text chunking failed: {exc}",
        ) from exc
    finally:
        for uploaded in files:
            await uploaded.close()


@app.get("/api/workbench/{session_id}")
async def get_workbench_session(session_id: str):
    try:
        return react_workbench.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/workbench/{session_id}/chunking")
async def update_workbench_chunking(
    session_id: str,
    req: WorkbenchChunkingRequest,
):
    try:
        return await run_in_threadpool(
            react_workbench.update_chunking,
            session_id,
            req.model_dump(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/workbench/{session_id}/qa")
async def generate_workbench_qa(
    session_id: str,
    req: WorkbenchQARequest,
):
    credentials = {
        "api_key": req.api_key.get_secret_value(),
        "endpoint": req.endpoint,
        "deployment": req.deployment,
        "api_version": req.api_version,
    }
    try:
        return await run_in_threadpool(
            react_workbench.generate_qa,
            session_id,
            req.object_ids,
            req.generate_objects,
            req.generate_text,
            req.object_question_count,
            req.text_question_count,
            credentials,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Q&A generation failed: {exc}") from exc


@app.get("/api/workbench/{session_id}/workbook")
async def download_workbench_workbook(session_id: str):
    try:
        workbook = await run_in_threadpool(
            react_workbench.build_workbook, session_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "filename": "object_qa_validation.xlsx",
        "content_base64": base64.b64encode(workbook).decode("ascii"),
    }


@app.get("/api/workbench/{session_id}/objects/{object_id}/image")
async def get_workbench_object_image(
    session_id: str,
    object_id: str,
    part: int = Query(default=0, ge=0),
):
    try:
        image_path = react_workbench.get_object_image(
            session_id, object_id, part
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image_path, media_type="image/png")


@app.post("/api/answer-improvement/review")
async def review_expected_answers(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    top_k: int = Form(2, ge=1, le=8),
    api_key: str = Form(...),
    endpoint: str = Form(...),
    deployment: str = Form(...),
    api_version: str = Form(...),
):
    filename = os.path.basename(file.filename or "answers.xlsx")
    if not filename.casefold().endswith(".xlsx"):
        await file.close()
        raise HTTPException(status_code=422, detail="Upload a valid .xlsx workbook.")
    try:
        workbook_bytes = await file.read()
        credentials = {
            "api_key": api_key,
            "endpoint": endpoint,
            "deployment": deployment,
            "api_version": api_version,
        }
        reviewed, summary = await run_in_threadpool(
            react_workbench.review_workbook,
            workbook_bytes,
            project_id,
            top_k,
            credentials,
        )
        return {
            "filename": f"{Path(filename).stem}_answer_reviewed.xlsx",
            "content_base64": base64.b64encode(reviewed).decode("ascii"),
            "summary": summary,
        }
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Answer review failed: {exc}") from exc
    finally:
        await file.close()


@app.post("/api/projects/{project_id}/ingest-pdfs")
async def ingest_project_pdfs(
    project_id: str,
    files: List[UploadFile] = File(...),
):
    """Extract uploaded PDFs with Docling and persist their chunks in Chroma."""
    try:
        _pm.get_project_dir(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    outcomes = []
    errors = []
    for uploaded in files:
        filename = os.path.basename(uploaded.filename or "upload.pdf")
        content = await uploaded.read()
        try:
            outcome = await run_in_threadpool(ingest_pdf, project_id, filename, content)
            outcomes.append(outcome)
        except (ValueError, FileNotFoundError) as exc:
            errors.append({"pdf_name": filename, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            errors.append({
                "pdf_name": filename,
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            await uploaded.close()

    if errors and not outcomes:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return {
        "status": "partial" if errors else "success",
        "files": outcomes,
        "errors": errors,
        "chunks_ingested": sum(item["chunks_ingested"] for item in outcomes),
    }


@app.post("/api/projects/{project_id}/run-vector-checks")
async def run_project_vector_checks(
    project_id: str,
    req: ProjectVectorChecksRequest,
):
    """Run selected checks against the project's persistent Chroma collection."""
    try:
        project_dir = _pm.get_project_dir(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    unknown_checks = set(req.selected_checks) - CHECK_CATEGORIES
    if unknown_checks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown check category/categories: {', '.join(sorted(unknown_checks))}",
        )
    try:
        check_report = await run_in_threadpool(
            run_project_checks,
            project_id,
            os.path.join(project_dir, "chroma_data"),
            req.selected_checks,
            req.required_fields,
            req.duplicate_min_similarity,
            req.duplicate_max_similarity,
            req.negative_query_texts,
            req.negative_query_max_similarity,
            req.synthetic_question_record_id,
            req.synthetic_question_texts,
            req.synthetic_question_top_k,
        )
        html_report = generate_report(
            check_report["results"],
            db_label=f"Chroma / {project_id}_knowledge_base",
        )
        report_id = f"vector_checks_{uuid.uuid4().hex[:12]}"
        _pm.save_result(project_id, report_id, check_report)
        return {
            **check_report,
            "project_id": project_id,
            "report_id": report_id,
            "html_report": html_report,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════
# ── MOCK AUT MANAGEMENT APIs (Phase 5 — Mock AUT) ─────────────────────
# ══════════════════════════════════════════════════════════════════════

# Module-level state for the mock AUT subprocess
_mock_aut_state: Dict[str, Any] = {
    "process": None,
    "port": None,
    "project_id": None,
    "source_mtime_ns": None,
}

MOCK_AUT_PROMPT_VERSION = "direct-answer-v3"


def _mock_aut_server_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _is_managed_mock_aut_process(process: psutil.Process) -> bool:
    try:
        expected_python = os.path.normcase(os.path.realpath(sys.executable))
        process_python = os.path.normcase(os.path.realpath(process.exe()))
        process_cwd = os.path.normcase(os.path.realpath(process.cwd()))
        expected_cwd = os.path.normcase(os.path.realpath(_mock_aut_server_directory()))
        command = process.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False
    return (
        process_python == expected_python
        and process_cwd == expected_cwd
        and "uvicorn" in command
        and "mock_aut:app" in command
    )


def _find_managed_mock_aut_processes() -> list[psutil.Process]:
    matches = []
    for candidate in psutil.process_iter(["pid"]):
        try:
            process = psutil.Process(candidate.info["pid"])
            if _is_managed_mock_aut_process(process):
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return matches


def _mock_aut_port(process: psutil.Process) -> int | None:
    try:
        command = process.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return None
    try:
        return int(command[command.index("--port") + 1])
    except (ValueError, IndexError):
        return None


def _probe_mock_aut(port: int) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/health",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _terminate_mock_aut_process(proc: subprocess.Popen | psutil.Process) -> None:
    try:
        managed_process = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    if not _is_managed_mock_aut_process(managed_process):
        raise RuntimeError(
            f"Refusing to stop PID {proc.pid}: it is not this workspace's Mock AUT."
        )
    managed_process.terminate()
    try:
        managed_process.wait(timeout=5)
    except psutil.TimeoutExpired:
        managed_process.kill()
        managed_process.wait(timeout=5)


class MockAUTStartRequest(BaseModel):
    project_id: Optional[str] = None
    port: int = 8001
    api_key: SecretStr
    endpoint: str
    deployment: str
    api_version: str


@app.post("/api/mock-aut/start")
async def start_mock_aut(req: MockAUTStartRequest):
    """Start the mock AUT server on the specified port as a subprocess."""
    try:
        mock_aut_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mock_aut.py"
        )
        if not os.path.isfile(mock_aut_path):
            raise HTTPException(
                status_code=500,
                detail="mock_aut.py not found in the server directory",
            )
        source_mtime_ns = os.stat(mock_aut_path).st_mtime_ns

        credentials = {
            "AZURE_OPENAI_API_KEY": req.api_key.get_secret_value().strip(),
            "AZURE_OPENAI_ENDPOINT": req.endpoint.strip(),
            "AZURE_OPENAI_DEPLOYMENT": req.deployment.strip(),
            "AZURE_OPENAI_API_VERSION": req.api_version.strip(),
        }
        missing_env = [name for name, value in credentials.items() if not value]
        if missing_env:
            raise HTTPException(
                status_code=400,
                detail="Enter all Azure OpenAI credentials before starting the Mock AUT: "
                + ", ".join(missing_env),
            )

        # A backend reload can lose its Popen handle while leaving the AUT child alive.
        existing_processes = _find_managed_mock_aut_processes()
        restarted = bool(existing_processes)
        for existing_process in existing_processes:
            _terminate_mock_aut_process(existing_process)
        _mock_aut_state.update(
            process=None,
            port=None,
            project_id=None,
            source_mtime_ns=None,
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if listener.connect_ex(("127.0.0.1", req.port)) == 0:
                running_service = _probe_mock_aut(req.port)
                service_name = (
                    running_service.get("service")
                    if running_service
                    else "another service"
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Port {req.port} is already in use by {service_name}. "
                        "Stop that process or choose another Mock AUT port."
                    ),
                )

        # Start the Mock AUT as a subprocess
        port = req.port
        server_dir = os.path.dirname(os.path.abspath(__file__))
        mock_aut_env = os.environ.copy()
        mock_aut_env.update(credentials)
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "mock_aut:app",
                "--host", "0.0.0.0",
                "--port", str(port),
            ],
            cwd=server_dir,
            env=mock_aut_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            ),
        )

        _mock_aut_state["process"] = proc
        _mock_aut_state["port"] = port
        _mock_aut_state["project_id"] = req.project_id
        _mock_aut_state["source_mtime_ns"] = source_mtime_ns

        return {
            "status": "running",
            "url": f"http://localhost:{port}",
            "pid": proc.pid,
            "project_id": req.project_id,
            "prompt_version": MOCK_AUT_PROMPT_VERSION,
            "restarted_stale_process": restarted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/mock-aut/stop")
async def stop_mock_aut():
    """Stop the mock AUT server subprocess."""
    try:
        processes = _find_managed_mock_aut_processes()
        for process in processes:
            _terminate_mock_aut_process(process)
        _mock_aut_state.update(
            process=None,
            port=None,
            project_id=None,
            source_mtime_ns=None,
        )
        return {"status": "stopped", "stopped_processes": len(processes)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/mock-aut/status")
async def mock_aut_status():
    """Check whether the mock AUT server is running."""
    try:
        processes = _find_managed_mock_aut_processes()
        if processes:
            process = processes[0]
            port = _mock_aut_port(process)
            if port is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not determine the Mock AUT port for PID {process.pid}.",
                )
            health = _probe_mock_aut(port)
            prompt_version = (
                health.get("answer_prompt_version")
                if health and health.get("service") == "mock-aut"
                else None
            )
            _mock_aut_state.update(
                process=process,
                port=port,
                project_id=_mock_aut_state.get("project_id"),
            )
            return {
                "running": True,
                "url": f"http://localhost:{port}",
                "pid": process.pid,
                "project_id": _mock_aut_state.get("project_id"),
                "prompt_version": prompt_version,
                "outdated": prompt_version != MOCK_AUT_PROMPT_VERSION,
            }
        _mock_aut_state.update(
            process=None,
            port=None,
            project_id=None,
            source_mtime_ns=None,
        )
        return {
            "running": False,
            "url": None,
            "pid": None,
            "project_id": None,
            "prompt_version": None,
            "outdated": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════
# ── EXECUTION APIs (Phase 6 — Execute Stage) ──────────────────────────
# ══════════════════════════════════════════════════════════════════════

_report_gen = ReportGenerator()


class ExecutionConfigRequest(BaseModel):
    """Request body for starting an execution."""
    dataset_id: str
    aut_endpoint: str = "http://localhost:8001/api/query"
    aut_project_id: Optional[str] = None
    run_vector_checks: bool = False
    checks_to_run: List[str] = []
    collection_name: Optional[str] = None


@app.post("/api/projects/{project_id}/execute")
async def execute_project_endpoint(project_id: str, req: ExecutionConfigRequest):
    """Start a test execution against the AUT."""
    try:
        svc = ExecutionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        config = req.model_dump()
        if not config.get("aut_project_id"):
            config["aut_project_id"] = project_id
        result = await run_in_threadpool(svc.execute, config)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Execution failed"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/executions")
async def list_executions_endpoint(project_id: str):
    """List all executions for a project."""
    try:
        svc = ExecutionService(project_id)
        executions = svc.list_executions()
        return {"executions": executions}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/executions/{execution_id}")
async def get_execution_endpoint(project_id: str, execution_id: str):
    """Get a specific execution result."""
    try:
        svc = ExecutionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        execution = svc.get_execution(execution_id)
        if execution is None:
            raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found.")
        return execution
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/projects/{project_id}/executions/{execution_id}")
async def delete_execution_endpoint(project_id: str, execution_id: str):
    """Delete an execution."""
    try:
        svc = ExecutionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        deleted = svc.delete_execution(execution_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found.")
        return {"status": "deleted", "execution_id": execution_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/executions/{execution_id}/report")
async def get_execution_report(project_id: str, execution_id: str):
    """Generate and download an HTML report for an execution."""
    try:
        svc = ExecutionService(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        execution = svc.get_execution(execution_id)
        if execution is None:
            raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found.")

        # Get project settings for the report
        project_config = _pm.get(project_id)
        project_settings = {
            "name": project_config.get("name", "RAG Testing Project"),
            **project_config.get("settings", {}),
        }

        html_content = _report_gen.generate(execution, project_settings)

        return HTMLResponse(
            content=html_content,
            headers={
                "Content-Disposition": f"attachment; filename=rag_report_{execution_id}.html",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
