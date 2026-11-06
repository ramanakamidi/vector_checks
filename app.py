"""Streamlit dashboard for the PDF pipeline and vector-quality API."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import huggingface_hub.file_download

huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False

from openai import OpenAIError

from pdf_workbench import custom_logic
from pdf_workbench.answer_review import review_expected_answer
import streamlit as st

from clients.api_client import ApiClient, ApiError
from checks.negative_queries import DEFAULT_MAX_SIMILARITY
from checks.synthetic_question_retrieval import generate_synthetic_questions
from services.answer_improvement import review_workbook, resolve_session_object_images
from services.visual_qa import load_workbench_qa_workbook, match_chunk_qa
from services.workbook_evaluation import evaluate_workbook


API_ENDPOINTS = [
    ("GET", "/health", "Health probe"),
    ("GET", "/api/health", "Health details"),
    ("POST", "/api/connect", "Connect to vector database"),
    ("GET", "/api/collections", "List vector collections"),
    ("GET", "/api/field-mapper", "Read field mapper"),
    ("POST", "/api/field-mapper", "Save field mapper"),
    ("POST", "/api/run-checks", "Run vector checks"),
    ("GET", "/api/results", "Read latest check results"),
    ("GET", "/api/record/{record_id}", "Read record details"),
    ("POST", "/api/records/search", "Search vector records"),
    ("GET", "/api/embedding-data", "Read embedding projection"),
    ("POST", "/api/auto-pairs", "Generate validation pairs"),
    ("POST", "/api/llm-review", "Review validation pairs with an LLM"),
    ("GET", "/api/export/html", "Download latest HTML report"),
    ("GET", "/api/export/json", "Download latest JSON report"),
    ("GET", "/api/projects", "List projects"),
    ("POST", "/api/projects", "Create project"),
    ("GET", "/api/projects/{project_id}", "Read project"),
    ("PUT", "/api/projects/{project_id}", "Update project"),
    ("DELETE", "/api/projects/{project_id}", "Delete project"),
    ("GET", "/api/projects/{project_id}/settings", "Read project settings"),
    ("PUT", "/api/projects/{project_id}/settings", "Save project settings"),
    ("POST", "/api/projects/{project_id}/ingest", "Upload JSON data file"),
    ("POST", "/api/projects/{project_id}/ingest-json", "Ingest JSON records"),
    ("GET", "/api/projects/{project_id}/ingested-data", "Read ingested records"),
    ("DELETE", "/api/projects/{project_id}/ingested-data", "Clear ingested records"),
    ("GET", "/api/projects/{project_id}/ingestion-stats", "Read ingestion statistics"),
    ("POST", "/api/projects/{project_id}/generate-dataset", "Generate test dataset"),
    ("GET", "/api/projects/{project_id}/datasets", "List datasets"),
    ("GET", "/api/projects/{project_id}/datasets/{dataset_id}", "Read dataset"),
    ("DELETE", "/api/projects/{project_id}/datasets/{dataset_id}", "Delete dataset"),
    ("GET", "/api/projects/{project_id}/metadata-fields", "Read metadata fields"),
    ("POST", "/api/projects/{project_id}/run-vector-checks", "Run selected project checks"),
    ("GET", "/api/projects/{project_id}/vector-records", "List project Chroma text records"),
    ("GET", "/api/projects/{project_id}/visualization-data", "Visualize project embeddings"),
    ("POST", "/api/mock-aut/start", "Start mock AUT"),
    ("POST", "/api/mock-aut/stop", "Stop mock AUT"),
    ("GET", "/api/mock-aut/status", "Read mock AUT status"),
    ("POST", "/api/projects/{project_id}/execute", "Execute test dataset"),
    ("GET", "/api/projects/{project_id}/executions", "List executions"),
    ("GET", "/api/projects/{project_id}/executions/{execution_id}", "Read execution"),
    ("DELETE", "/api/projects/{project_id}/executions/{execution_id}", "Delete execution"),
    ("GET", "/api/projects/{project_id}/executions/{execution_id}/report", "Download execution report"),
]

st.set_page_config(page_title="Vector Quality Studio", page_icon="V", layout="wide")
st.markdown(
    """
    <style>
      :root { --ink: #202a2a; --muted: #647271; --accent: #c65a36; --line: #dce3df; }
      .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
      h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
      h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; }
      [data-testid="stMetric"] { background: #f3f6f3; border-left: 3px solid var(--accent); padding: .8rem 1rem; }
      div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
      div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent); }
      .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
      .muted { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


def state_defaults() -> None:
    if st.session_state.get("api_base_url") == "http://127.0.0.1:8001":
        st.session_state.api_base_url = "http://127.0.0.1:8000"
    defaults = {
        "api_base_url": "http://127.0.0.1:8000",
        "pipeline_session_id": uuid.uuid4().hex[:10],
        "pipeline_objects": [],
        "pipeline_raw_objects": [],
        "pipeline_qa": {},
        "pipeline_text_documents": [],
        "pipeline_text_chunks": [],
        "pipeline_text_qa": [],
        "pipeline_text_chunking": None,
        "pipeline_selected_object_ids": [],
        "pipeline_object_selection_generation": 0,
        "projects": [],
        "selected_project_id": "",
        "project_settings": {},
        "vector_connection": None,
        "legacy_results": None,
        "mock_aut": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def parse_json_input(value: str, label: str, default: Any) -> Any:
    if not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc


def api_result(action, *, success_message: str | None = None):
    try:
        result = action()
    except ApiError as exc:
        st.error(str(exc))
        return None
    if success_message:
        st.success(success_message)
    return result


def show_json(value: Any) -> None:
    if isinstance(value, bytes):
        st.download_button("Download response", value, file_name="api-response.bin")
    elif value is not None:
        st.json(value)


def get_project_options() -> dict[str, str]:
    return {
        str(project.get("id")): f"{project.get('name', 'Untitled')} · {project.get('id')}"
        for project in st.session_state.projects
        if project.get("id")
    }


def current_project_id() -> str:
    return st.session_state.get("selected_project_id", "")


def refresh_projects(client: ApiClient) -> dict[str, Any] | None:
    response = api_result(client.list_projects)
    if response is not None:
        st.session_state.projects = response.get("projects", [])
    return response


def render_sidebar() -> ApiClient:
    with st.sidebar:
        st.markdown('<p class="eyebrow">Connection</p>', unsafe_allow_html=True)
        st.text_input("Backend URL", key="api_base_url")
        timeout = st.number_input("Request timeout (seconds)", min_value=5, max_value=600, value=300, step=5)
        client = ApiClient(st.session_state.api_base_url, timeout=float(timeout))
        if st.button("Check backend", width="stretch"):
            with st.spinner("Checking backend…"):
                result = api_result(client.health)
            if result:
                st.success(f"Backend: {result.get('status', 'available')}")
        st.divider()
        st.markdown('<p class="eyebrow">PDF Workbench · Azure OpenAI</p>', unsafe_allow_html=True)
        st.text_input("Endpoint", key="azure_endpoint", placeholder="https://your-resource.openai.azure.com")
        st.text_input("Deployment name", key="azure_deployment", value=st.session_state.get("azure_deployment", "gpt-4.1-nano"))
        st.text_input("API version", key="azure_api_version", value=st.session_state.get("azure_api_version", "2024-12-01-preview"))
        st.text_input("API key", type="password", key="azure_api_key")
        st.caption("These credentials are used by the PDF Workbench, Mock AUT, and answer judge.")
        st.caption("Credentials are kept in this Streamlit session and are not saved in the project.")
        st.divider()
        st.caption("Generated files are saved under `streamlit_output/` in this workspace.")
    return client


def render_pdf_workbench(client: ApiClient) -> None:
    st.subheader("PDF object analysis")
    st.write(
        "Extract tables, charts, and images, then optionally generate grounded "
        "Q&A from both detected objects and document prose."
    )
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploads")
    mode_label = st.radio(
        "Vision context",
        ["Object crop", "Crop + full page", "Crop + neighboring pages"],
        horizontal=True,
        help="Neighboring-page mode checks table continuation and merges confirmed table parts.",
    )
    mode = {"Object crop": "crop", "Crop + full page": "page", "Crop + neighboring pages": "page_context"}[mode_label]
    st.markdown("#### Text chunking")
    chunking_label = st.selectbox(
        "Technique",
        [
            "Heading-aware (current)",
            "Recursive",
            "Sentence",
            "Token",
            "Semantic",
        ],
        key="pdf_text_chunking_technique",
        help="Choose how prose extracted from the PDFs is divided before text Q&A generation.",
    )
    chunking_technique = {
        "Heading-aware (current)": "heading_aware",
        "Recursive": "recursive",
        "Sentence": "sentence",
        "Token": "token",
        "Semantic": "semantic",
    }[chunking_label]
    if chunking_technique == "heading_aware":
        chunk_size = st.number_input(
            "Maximum chunk size (characters)",
            min_value=128,
            max_value=12000,
            value=2600,
            step=100,
            key="pdf_text_chunk_size_heading",
        )
        overlap = st.number_input(
            "Character overlap",
            min_value=0,
            max_value=int(chunk_size) - 1,
            value=min(250, int(chunk_size) - 1),
            step=25,
            key="pdf_text_overlap_heading",
        )
    else:
        chunk_size = st.number_input(
            "Maximum chunk body size (tokens)",
            min_value=64,
            max_value=4096,
            value=512,
            step=32,
            key=f"pdf_text_chunk_size_{chunking_technique}",
        )
        overlap = 0
        if chunking_technique in {"sentence", "token"}:
            overlap = st.number_input(
                "Token overlap",
                min_value=0,
                max_value=int(chunk_size) - 1,
                value=min(50, int(chunk_size) - 1),
                step=5,
                key=f"pdf_text_overlap_{chunking_technique}",
            )
    min_sentences_per_chunk = 1
    semantic_threshold = 0.8
    if chunking_technique == "sentence":
        min_sentences_per_chunk = st.number_input(
            "Minimum sentences per chunk",
            min_value=1,
            max_value=20,
            value=1,
            key="pdf_text_min_sentences",
            help="SentenceChunker treats this as a minimum, while the token size remains the upper bound.",
        )
    elif chunking_technique == "semantic":
        semantic_threshold = st.slider(
            "Semantic similarity threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Lower values group more sentences together; the first semantic run downloads Chonkie's default embedding model.",
        )
    chunking_options = {
        "technique": chunking_technique,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "min_sentences_per_chunk": int(min_sentences_per_chunk),
        "semantic_threshold": float(semantic_threshold),
    }
    if chunking_technique != "heading_aware":
        if chunking_technique == "semantic":
            st.caption(
                "Semantic chunking uses Chonkie's minishlab/potion-base-32M "
                "embedding model and its tokenizer. Section headings are added "
                "as context after chunking and are not included in the token limit."
            )
        else:
            st.caption(
                "Chonkie measures tokens with this project's "
                "all-MiniLM-L6-v2 tokenizer. Section headings are added as "
                "context after chunking and are not included in the token limit."
            )

    def make_text_chunks(
        documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        chunks: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for document in documents:
            if document.get("error"):
                errors.append({
                    "PDF": str(document.get("pdf_name", "")),
                    "Error": str(document["error"]),
                })
                continue
            page_markdowns = document.get("page_markdowns") or []
            if page_markdowns:
                for page in page_markdowns:
                    page_no = int(page["page_no"])
                    page_chunks = custom_logic.chunk_document_text(
                        page.get("markdown", ""),
                        document["pdf_name"],
                        f"{document['document_id']}:page-{page_no}",
                        chunk_size=chunking_options["chunk_size"],
                        overlap=chunking_options["overlap"],
                        technique=chunking_options["technique"],
                        min_sentences_per_chunk=chunking_options["min_sentences_per_chunk"],
                        semantic_threshold=chunking_options["semantic_threshold"],
                    )
                    for chunk in page_chunks:
                        chunk["page_no"] = page_no
                        chunk["pages"] = [page_no]
                    chunks.extend(page_chunks)
            else:
                chunks.extend(
                    custom_logic.chunk_document_text(
                        document.get("markdown", ""),
                        document["pdf_name"],
                        document["document_id"],
                        chunk_size=chunking_options["chunk_size"],
                        overlap=chunking_options["overlap"],
                        technique=chunking_options["technique"],
                        min_sentences_per_chunk=chunking_options["min_sentences_per_chunk"],
                        semantic_threshold=chunking_options["semantic_threshold"],
                    )
                )
        return chunks, errors

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_analysis = st.button("Analyze PDFs", type="primary", disabled=not uploads, width="stretch")
    with info_col:
        st.caption("Docling performs local PDF extraction; Azure OpenAI analyzes the selected image context.")

    if run_analysis:
        try:
            custom_logic.configure_azure_openai(
                st.session_state.get("azure_api_key", ""),
                st.session_state.get("azure_endpoint", ""),
                st.session_state.get("azure_deployment", ""),
                st.session_state.get("azure_api_version", ""),
            )
            session_dir = Path(__file__).parent / "streamlit_output" / st.session_state.pipeline_session_id
            upload_dir = session_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_paths = []
            for uploaded in uploads:
                name = Path(uploaded.name).name
                destination = upload_dir / name
                destination.write_bytes(uploaded.getvalue())
                pdf_paths.append(destination)
            bar = st.progress(0.0, text="Preparing PDF analysis")

            def update_progress(message: str, fraction: float) -> None:
                bar.progress(fraction, text=message)

            text_documents: list[dict[str, Any]] = []
            with st.spinner("Extracting PDF text and analyzing objects…"):
                raw_objects, objects = custom_logic.process_pdf_files(
                    pdf_paths,
                    str(session_dir / "objects"),
                    mode,
                    update_progress,
                    text_documents=text_documents,
                )
            st.session_state.pipeline_raw_objects = raw_objects
            st.session_state.pipeline_objects = objects
            st.session_state.pipeline_qa = {}
            st.session_state.pipeline_selected_object_ids = []
            st.session_state.pipeline_object_selection_generation += 1
            st.session_state.filtered_object_ids = [
                item["object_id"] for item in objects
            ]
            st.session_state.pipeline_text_documents = text_documents
            st.session_state.pipeline_text_chunks = []
            st.session_state.pipeline_text_qa = []
            st.session_state.pipeline_text_errors = []
            st.session_state.pipeline_output_dir = str(session_dir)
            text_chunks, text_errors = make_text_chunks(text_documents)
            st.session_state.pipeline_text_chunks = text_chunks
            st.session_state.pipeline_text_errors = text_errors
            st.session_state.pipeline_text_chunking = chunking_options.copy()
            st.success(
                f"Analysis complete: {len(objects)} useful object(s) and "
                f"{len(text_chunks)} prose chunk(s) from {len(uploads)} PDF(s)."
            )
        except Exception as exc:
            st.error(f"PDF analysis or text chunking failed: {exc}")

    extracted_documents = st.session_state.get("pipeline_text_documents", [])
    chunking_settings_pending = False
    if extracted_documents:
        prior_chunking = st.session_state.get("pipeline_text_chunking")
        chunking_settings_pending = prior_chunking != chunking_options
        if st.button(
            "Apply chunking settings to extracted text",
            disabled=not any(not item.get("error") for item in extracted_documents),
            width="stretch",
        ):
            try:
                with st.spinner("Re-chunking extracted PDF text…"):
                    text_chunks, text_errors = make_text_chunks(extracted_documents)
                st.session_state.pipeline_text_chunks = text_chunks
                st.session_state.pipeline_text_qa = []
                st.session_state.pipeline_text_errors = text_errors
                st.session_state.pipeline_text_chunking = chunking_options.copy()
                chunking_settings_pending = False
                st.success(f"Created {len(text_chunks)} text chunk(s).")
            except Exception as exc:
                st.error(f"Could not apply text chunking settings: {exc}")
        if chunking_settings_pending:
            st.info(
                "Chunking settings changed. Apply them to the extracted text "
                "below; the PDFs will not need to be analyzed again."
            )

    objects = st.session_state.pipeline_objects
    text_chunks = st.session_state.get("pipeline_text_chunks", [])
    text_errors = st.session_state.get("pipeline_text_errors", [])
    if text_errors:
        st.warning(
            "Text extraction failed for some PDFs. Object Q&A is still available "
            "for successfully analyzed objects."
        )
        st.dataframe(text_errors, width="stretch", hide_index=True)
    if not objects and not text_chunks:
        st.info("Upload one or more PDFs and run analysis to populate this workspace.")
        return

    if text_chunks:
        with st.expander("Text chunk preview", expanded=False):
            st.caption(
                f"Showing up to 3 of {len(text_chunks)} chunks. "
                "Changing the technique clears generated text Q&A when applied."
            )
            for index, chunk in enumerate(text_chunks[:3], start=1):
                section = chunk.get("section") or "No heading"
                st.caption(
                    f"{index}. {chunk.get('pdf_name', '')} · {section}"
                )
                st.text(str(chunk.get("source_text", ""))[:2000])

    st.divider()
    table_col, chart_col, image_col, text_col = st.columns(4)
    table_col.metric("Tables", sum(item.get("object_type") == "table" for item in objects))
    chart_col.metric("Charts", sum(item.get("object_type") == "graph" for item in objects))
    image_col.metric("Images", sum(item.get("object_type") == "image" for item in objects))
    text_col.metric("Text chunks", len(text_chunks))

    with st.expander("Filter objects", expanded=True):
        query_col, filter_col = st.columns([4, 1])
        with query_col:
            query = st.text_input("Natural-language filter", placeholder='e.g. tables with more than 5 rows and page 3')
        with filter_col:
            st.write("")
            apply_clicked = st.button("Apply filter", width="stretch")
        if apply_clicked:
            parsed = custom_logic.parse_query(query)
            if parsed is None:
                st.error("Could not understand that filter. Try a type, page, row/column count, or title condition.")
                st.caption(custom_logic.FILTER_HELP_TEXT)
                st.session_state.filtered_object_ids = [item["object_id"] for item in objects]
            else:
                matched = custom_logic.apply_filter(objects, parsed)
                st.session_state.filtered_object_ids = [item["object_id"] for item in matched]
                st.caption("Parsed as: " + custom_logic.describe_filter(parsed))
                st.success(f"{len(matched)} object(s) match.")

    filtered_ids = st.session_state.get("filtered_object_ids", [item["object_id"] for item in objects])
    objects_by_id = {item["object_id"]: item for item in objects}
    visible_objects = [objects_by_id[item_id] for item_id in filtered_ids if item_id in objects_by_id]
    selected_object_ids = set(
        st.session_state.get("pipeline_selected_object_ids", [])
    )
    object_rows = []
    for item in visible_objects:
        characteristics = item.get("characteristics", {})
        object_rows.append({
            "Select": item["object_id"] in selected_object_ids,
            "_object_id": item["object_id"],
            "Object": item.get("object_no"), "PDF": item.get("doc_name"),
            "Page(s)": ", ".join(map(str, item.get("pages") or [item.get("page_no")])),
            "Type": item.get("object_type"), "Caption": item.get("docling_title"),
            "Summary": characteristics.get("content_summary") or characteristics.get("description"),
            "Confidence": characteristics.get("confidence"),
        })
    if object_rows:
        visible_object_ids = {row["_object_id"] for row in object_rows}
        selection_signature = hashlib.sha1(
            "\0".join(row["_object_id"] for row in object_rows).encode("utf-8")
        ).hexdigest()[:12]
        edited_object_rows = st.data_editor(
            object_rows,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key=(
                f"pdf_object_selection_"
                f"{st.session_state.pipeline_object_selection_generation}_"
                f"{selection_signature}"
            ),
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select for Q&A",
                    help="Check the objects to include in object Q&A generation.",
                    default=False,
                ),
                "_object_id": None,
            },
            disabled=[
                column
                for column in object_rows[0]
                if column not in {"Select"}
            ],
        )
        selected_visible_ids = {
            row["_object_id"]
            for row in edited_object_rows
            if row.get("Select")
        }
        selected_object_ids.difference_update(visible_object_ids)
        selected_object_ids.update(selected_visible_ids)
        st.session_state.pipeline_selected_object_ids = list(selected_object_ids)
    else:
        st.info("No objects are currently shown to select.")
    selected_ids = [
        item["object_id"]
        for item in objects
        if item["object_id"] in selected_object_ids
    ]
    st.caption(
        f"{len(selected_ids)} object(s) selected. Check **Select for Q&A** "
        "in the objects table to change the selection."
    )
    labels = {
        item["object_id"]: (
            f"{item.get('doc_name')} · p.{item.get('page_no')} · "
            f"{item.get('object_no')} · {item.get('object_type')}"
        )
        for item in objects
    }
    st.markdown("#### Generate Q&A")
    st.caption(
        "Object actions use the selected objects. Text actions use all extracted prose chunks."
    )
    count_col, text_count_col = st.columns(2)
    qa_count = count_col.number_input(
        "Questions per selected object",
        min_value=1,
        max_value=20,
        value=5,
    )
    text_qa_count = text_count_col.number_input(
        "Questions per text chunk",
        min_value=1,
        max_value=3,
        value=2,
    )
    object_button, combined_button, text_button = st.columns(3)
    generate_objects_clicked = object_button.button(
        "Generate Q&A for objects",
        type="primary",
        disabled=not selected_ids,
        width="stretch",
    )
    generate_combined_clicked = combined_button.button(
        "Generate Q&A for objects and text",
        disabled=not selected_ids or not text_chunks or chunking_settings_pending,
        width="stretch",
        help=(
            "Apply changed text chunking settings first."
            if chunking_settings_pending
            else "Generates Q&A for the selected objects and all extracted prose chunks."
        ),
    )
    generate_text_clicked = text_button.button(
        "Generate Q&A for text only",
        disabled=not text_chunks or chunking_settings_pending,
        width="stretch",
        help=(
            "Apply changed text chunking settings first."
            if chunking_settings_pending
            else "Generates questions from prose chunks; tables and figures are not included."
        ),
    )
    generate_objects = generate_objects_clicked or generate_combined_clicked
    generate_text = generate_text_clicked or generate_combined_clicked
    if generate_objects or generate_text:
        try:
            custom_logic.configure_azure_openai(
                st.session_state.get("azure_api_key", ""),
                st.session_state.get("azure_endpoint", ""),
                st.session_state.get("azure_deployment", ""),
                st.session_state.get("azure_api_version", ""),
            )
        except Exception as exc:
            st.error(f"Could not configure Azure OpenAI for Q&A generation: {exc}")
        else:
            if generate_objects:
                try:
                    progress = st.progress(0.0, text="Generating object questions")
                    for index, object_id in enumerate(selected_ids):
                        with st.spinner(f"Generating Q&A for {labels[object_id]}…"):
                            generated = custom_logic.generate_qa_for_record(
                                objects_by_id[object_id],
                                int(qa_count),
                            )
                        st.session_state.pipeline_qa[object_id] = generated.get("pairs", [])
                        progress.progress(
                            (index + 1) / len(selected_ids),
                            text=f"Completed {index + 1} of {len(selected_ids)} objects",
                        )
                    st.success("Object Q&A generation finished.")
                except Exception as exc:
                    st.error(f"Object Q&A generation failed: {exc}")

            if generate_text:
                try:
                    progress = st.progress(0.0, text="Generating Q&A from document text")

                    def update_text_progress(current: int, total: int) -> None:
                        progress.progress(
                            current / total if total else 1.0,
                            text=f"Processed text chunk {current} of {total}",
                        )

                    with st.spinner(f"Generating text Q&A from {len(text_chunks)} chunk(s)…"):
                        st.session_state.pipeline_text_qa = (
                            custom_logic.generate_qa_for_text_chunks(
                                text_chunks,
                                int(text_qa_count),
                                progress_callback=update_text_progress,
                            )
                        )
                    st.success(
                        f"Generated {len(st.session_state.pipeline_text_qa)} "
                        "text-based question(s)."
                    )
                except Exception as exc:
                    st.error(f"Text Q&A generation failed: {exc}")

    qa_by_object = st.session_state.pipeline_qa
    text_qa = st.session_state.get("pipeline_text_qa", [])
    if qa_by_object or text_qa:
        all_qa = custom_logic.build_qa_records(objects, qa_by_object, text_qa)
        st.subheader("Generated Q&A")
        display_columns = [
            "source_type", "pdf_name", "source_section", "object_type",
            "category", "question", "expected_answer",
        ]
        st.dataframe(
            [{key: row.get(key) for key in display_columns} for row in all_qa],
            width="stretch",
            hide_index=True,
        )
        json_bytes = json.dumps(
            custom_logic.build_characteristics_payload(objects, qa_by_object),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        excel_bytes = custom_logic.build_excel_bytes(objects, qa_by_object, text_qa)
        download_col, json_col, ingest_col = st.columns(3)
        download_col.download_button(
            "Download Excel",
            excel_bytes,
            file_name="object_qa_validation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        json_col.download_button(
            "Download characteristics JSON", json_bytes,
            file_name="characteristics.json", mime="application/json",
        )
        try:
            output_dir = Path(st.session_state.pipeline_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            excel_path = output_dir / "object_qa_validation.xlsx"
            json_path = output_dir / "characteristics.json"
            excel_path.write_bytes(excel_bytes)
            json_path.write_bytes(json_bytes)
            st.caption(f"Saved Excel and JSON locally to `{output_dir}`.")
        except OSError as exc:
            st.error(f"Could not save PDF Workbench outputs locally: {exc}")
        project_id = current_project_id()
        if project_id and ingest_col.button("Send Q&A to selected project", type="primary"):
            ingest_records = []
            for qa_record in all_qa:
                source_id = qa_record.get("source_id") or qa_record.get("object_id")
                ingest_records.append({
                    "id": (
                        f"{source_id}-qa-"
                        f"{len(ingest_records) + 1}"
                    ),
                    "question": qa_record["question"],
                    "answer": qa_record["expected_answer"],
                    "context": (
                        qa_record.get("source_text")
                        or qa_record.get("characteristics")
                        or ""
                    ),
                    "metadata": {
                        "source_pdf": qa_record.get("pdf_name"),
                        "page_no": qa_record.get("page_no"),
                        "object_type": qa_record.get("object_type"),
                        "source_type": qa_record.get("source_type"),
                        "source_section": qa_record.get("source_section"),
                        "source_id": source_id,
                        "category": qa_record.get("category"),
                    },
                })
            with st.spinner("Ingesting generated Q&A into the project…"):
                result = api_result(lambda: client.ingest_json(project_id, ingest_records))
            if result:
                st.success(f"Ingested {result.get('records_ingested', len(ingest_records))} Q&A record(s).")


def render_app_under_test_config(client: ApiClient) -> None:
    st.subheader("App Under Test Config")
    st.write("Create or load a project, ingest PDFs with Docling into persistent Chroma, and start the Mock AUT.")

    create_col, project_col = st.columns([1, 1.5])
    with create_col:
        with st.form("create_project_form"):
            name = st.text_input("New project name")
            description = st.text_area("Description", height=90)
            create_clicked = st.form_submit_button("Create project", type="primary")
        if create_clicked:
            if not name.strip():
                st.error("Enter a project name.")
            else:
                with st.spinner("Creating project…"):
                    project = api_result(lambda: client.create_project(name.strip(), description))
                if project:
                    st.session_state.selected_project_id = str(project.get("id", ""))
                    refresh_projects(client)
                    st.success("Project created.")

    with project_col:
        if st.button("Load / refresh projects"):
            with st.spinner("Loading projects…"):
                refresh_projects(client)
        projects = st.session_state.projects
        if projects:
            options = get_project_options()
            project_ids = list(options)
            current = current_project_id()
            selected = st.selectbox(
                "Active project",
                project_ids,
                index=project_ids.index(current) if current in project_ids else 0,
                format_func=lambda value: options.get(value, value),
            )
            st.session_state.selected_project_id = selected
        else:
            st.info("Load projects or create a new project to continue.")

    project_id = current_project_id()
    if not project_id:
        return

    st.markdown(f"**Active project:** `{project_id}`")
    st.divider()
    st.markdown("#### PDF ingestion")
    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="aut_config_pdf_uploads",
    )
    if st.button("Parse and ingest PDFs", type="primary", disabled=not uploads):
        uploaded_files = [(Path(item.name).name, item.getvalue()) for item in uploads]
        with st.spinner("Docling is extracting PDF content; embeddings are being saved to Chroma…"):
            result = api_result(lambda: client.ingest_pdfs(project_id, uploaded_files))
        if result:
            st.session_state.last_pdf_ingestion = result
            st.success(
                f"Processed {len(result.get('files', []))} file(s); "
                f"{result.get('chunks_ingested', 0)} new chunk(s) added."
            )
            if result.get("errors"):
                st.warning("Some files could not be ingested.")
                st.json(result["errors"])
            st.json(result.get("files", []))
    elif st.session_state.get("last_pdf_ingestion"):
        st.json(st.session_state.last_pdf_ingestion.get("files", []))

    st.divider()
    st.markdown("#### Mock AUT service")
    st.caption(
        "The Mock AUT retrieves chunks from this project's Chroma collection and "
        "uses the Azure OpenAI credentials configured in the sidebar. If you change "
        "those credentials, stop and restart the Mock AUT."
    )
    status_col, port_col, start_col, stop_col = st.columns([2, 1, 1, 1])
    port = port_col.number_input("Mock AUT port", min_value=1024, max_value=65535, value=8001)
    if status_col.button("Check AUT status"):
        with st.spinner("Checking Mock AUT…"):
            status = api_result(client.mock_aut_status)
        if status:
            st.session_state.mock_aut = status
    if start_col.button("Start Mock AUT", type="primary"):
        with st.spinner("Starting Mock AUT…"):
            status = api_result(
                lambda: client.start_mock_aut(
                    project_id,
                    int(port),
                    api_key=st.session_state.get("azure_api_key", ""),
                    endpoint=st.session_state.get("azure_endpoint", ""),
                    deployment=st.session_state.get("azure_deployment", ""),
                    api_version=st.session_state.get("azure_api_version", ""),
                ),
                success_message="Mock AUT started.",
            )
        if status:
            st.session_state.mock_aut = status
    if stop_col.button("Stop Mock AUT"):
        with st.spinner("Stopping Mock AUT…"):
            status = api_result(client.stop_mock_aut, success_message="Mock AUT stopped.")
        if status:
            st.session_state.mock_aut = status
    mock_aut_state = st.session_state.get("mock_aut") or {}
    if st.session_state.get("mock_aut"):
        st.json(st.session_state.mock_aut)
    if mock_aut_state.get("outdated"):
        st.warning(
            "The running Mock AUT is outdated. Click **Start Mock AUT** to replace "
            "the stale process before evaluating the workbook."
        )

    mock_aut_running = (
        (
            mock_aut_state.get("running") is True
            or mock_aut_state.get("status") in {"running", "already_running"}
        )
        and not mock_aut_state.get("outdated", False)
    )
    mock_aut_url = mock_aut_state.get("url")
    st.divider()
    st.markdown("#### Run and judge an Excel Q&A workbook")
    st.caption(
        "For each row, the Mock AUT answers the question from this project's vector "
        "store. The sidebar-configured Azure deployment then judges that answer "
        "against the expected answer. Scores range from 0 to 1; scores of 0.6 or "
        "higher are marked PASS."
    )
    workbook_upload = st.file_uploader(
        "Upload the Q&A validation Excel workbook",
        type=["xlsx"],
        key="qa_validation_workbook",
        disabled=not mock_aut_running,
    )
    evaluate_clicked = st.button(
        "Run AUT and judge workbook",
        type="primary",
        disabled=not mock_aut_running or workbook_upload is None,
    )
    if not mock_aut_running:
        st.info("Start the Mock AUT before uploading and evaluating a workbook.")
    if evaluate_clicked and workbook_upload is not None and mock_aut_url:
        credentials = {
            "api_key": st.session_state.get("azure_api_key", ""),
            "endpoint": st.session_state.get("azure_endpoint", ""),
            "deployment": st.session_state.get("azure_deployment", ""),
            "api_version": st.session_state.get("azure_api_version", ""),
        }
        try:
            custom_logic.configure_azure_openai(
                credentials["api_key"],
                credentials["endpoint"],
                credentials["deployment"],
                credentials["api_version"],
            )
            aut_client = ApiClient(mock_aut_url, timeout=client.timeout)
            progress = st.progress(0.0, text="Preparing workbook")

            def update_workbook_progress(current: int, total: int) -> None:
                progress.progress(
                    current / total if total else 1.0,
                    text=f"Evaluated {current} of {total} question(s)",
                )

            with st.spinner("Running each question through the AUT and LLM judge…"):
                workbook_bytes, summary = evaluate_workbook(
                    workbook_upload.getvalue(),
                    project_id,
                    lambda question, active_project: aut_client.query_mock_aut(
                        question,
                        active_project,
                    ),
                    custom_logic.judge_answer,
                    update_workbook_progress,
                )
            output_dir = (
                Path(__file__).parent
                / "streamlit_output"
                / st.session_state.pipeline_session_id
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{Path(workbook_upload.name).stem}_evaluated.xlsx"
            output_path = output_dir / output_filename
            output_path.write_bytes(workbook_bytes)
            st.session_state.workbook_evaluation = {
                "bytes": workbook_bytes,
                "filename": output_filename,
                "path": str(output_path),
                "summary": summary,
                "project_id": project_id,
            }
        except (ValueError, RuntimeError, OSError, ApiError) as exc:
            st.error(f"Workbook evaluation failed: {exc}")

    evaluation = st.session_state.get("workbook_evaluation")
    if evaluation and evaluation["project_id"] == project_id:
        summary = evaluation["summary"]
        st.success(
            f"Workbook evaluation complete: {summary['passed']} passed, "
            f"{summary['failed']} failed, {summary['errors']} error(s)."
        )
        st.caption(f"Completed workbook saved locally to `{evaluation['path']}`.")
        if summary.get("error_details"):
            st.warning("Some rows could not be answered or judged:")
            st.dataframe(summary["error_details"], hide_index=True, use_container_width=True)
        st.download_button(
            "Download completed workbook",
            evaluation["bytes"],
            file_name=evaluation["filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_answer_improvement() -> None:
    project_id = current_project_id()
    st.subheader("Expected answer improvement review")
    if not project_id:
        st.info("Select a project in App Under Test Config before reviewing answers.")
        return

    st.caption(
        f"Text and table questions use Chroma records from project `{project_id}`; "
        "retrieval expands from the selected k up to 8 if evidence is insufficient. "
        "Graph and image questions use their cropped image(s) only. The review "
        "checks every requested part of the question against the expected answer. "
        "If the evidence cannot verify an answer, the result is marked "
        "'Unable to verify' rather than 'No'."
    )
    workbook_upload = st.file_uploader(
        "Upload the generated Q&A Excel workbook",
        type=["xlsx"],
        key="answer_improvement_workbook",
    )
    top_k = st.number_input(
        "Retrieved chunks per question (k)",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        key="answer_improvement_top_k",
    )
    run_review = st.button(
        "Review expected answers",
        type="primary",
        disabled=workbook_upload is None,
        key="run_answer_improvement",
    )

    if run_review and workbook_upload is not None:
        credentials = {
            "api_key": st.session_state.get("azure_api_key", ""),
            "endpoint": st.session_state.get("azure_endpoint", ""),
            "deployment": st.session_state.get("azure_deployment", ""),
            "api_version": st.session_state.get("azure_api_version", ""),
        }
        try:
            custom_logic.configure_azure_openai(
                credentials["api_key"],
                credentials["endpoint"],
                credentials["deployment"],
                credentials["api_version"],
            )
            progress = st.progress(0.0, text="Preparing workbook")

            def update_progress(current: int, total: int) -> None:
                progress.progress(
                    current / total if total else 1.0,
                    text=f"Reviewed {current} of {total} question(s)",
                )

            def review_answer(
                question: str,
                expected_answer: str,
                retrieved_contexts: list[dict[str, Any]],
                image_paths: list[str] | None = None,
            ) -> dict[str, Any]:
                if image_paths is None:
                    image_paths = resolve_session_object_images(
                        st.session_state.get("pipeline_objects", []),
                        st.session_state.get("pipeline_qa", {}),
                        question,
                        expected_answer,
                    )

                return review_expected_answer(
                    question,
                    expected_answer,
                    retrieved_contexts,
                    image_paths=image_paths,
                    azure_client=custom_logic.client,
                    deployment=custom_logic.MODEL_NAME,
                )

            with st.spinner("Retrieving project context and reviewing answers…"):
                workbook_bytes, summary = review_workbook(
                    workbook_upload.getvalue(),
                    project_id,
                    int(top_k),
                    review_answer,
                    update_progress,
                )

            output_filename = (
                f"{Path(workbook_upload.name).stem}_answer_reviewed.xlsx"
            )
            output_dir = (
                Path(__file__).parent
                / "streamlit_output"
                / st.session_state.pipeline_session_id
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            output_path.write_bytes(workbook_bytes)
            st.session_state.answer_improvement_result = {
                "bytes": workbook_bytes,
                "filename": output_filename,
                "path": str(output_path),
                "summary": summary,
                "project_id": project_id,
            }
        except (ValueError, RuntimeError, FileNotFoundError, OSError, ApiError) as exc:
            st.error(f"Answer review failed: {exc}")

    result = st.session_state.get("answer_improvement_result")
    if result and result.get("project_id") == project_id:
        summary = result["summary"]
        st.success(
            f"Reviewed {summary['processed']} question(s): "
            f"{summary['improvement_required']} need improvement; "
            f"{summary['no_improvement']} do not; "
            f"{summary.get('unable_to_verify', 0)} unable to verify; "
            f"{summary['errors']} error(s)."
        )
        if summary.get("error_details"):
            st.warning(
                "Some rows could not be reviewed. They are marked ERROR in the "
                "downloaded workbook; other questions were still processed."
            )
            st.dataframe(
                summary["error_details"],
                hide_index=True,
                width="stretch",
            )
        st.caption(f"Reviewed workbook saved locally to `{result['path']}`.")
        st.download_button(
            "Download reviewed workbook",
            result["bytes"],
            file_name=result["filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_answer_improvement",
        )


def render_ingestion_and_datasets(client: ApiClient) -> None:
    project_id = current_project_id()
    if not project_id:
        st.info("Select or create a project in the Projects tab first.")
        return
    st.subheader("Ingestion and test datasets")
    st.caption(f"Project {project_id}")
    ingest_tab, records_tab, dataset_tab = st.tabs(["Ingest", "Records", "Datasets"])

    with ingest_tab:
        upload = st.file_uploader("JSON records file", type=["json"], key="records_json_upload")
        ingest_cols = st.columns([1, 1, 2])
        if ingest_cols[0].button("Upload file", disabled=not upload, type="primary"):
            with st.spinner("Uploading records…"):
                result = api_result(lambda: client.ingest_file(project_id, upload.name, upload.getvalue()))
            if result:
                st.success(f"Ingested {result.get('records_ingested', 0)} record(s).")
        if ingest_cols[1].button("Read ingestion stats"):
            with st.spinner("Loading ingestion stats…"):
                show_json(api_result(lambda: client.ingestion_stats(project_id)))
        if ingest_cols[2].button("Clear ingested records"):
            with st.spinner("Clearing project data…"):
                api_result(lambda: client.clear_ingested_data(project_id), success_message="Ingested data cleared.")

        with st.expander("Paste records as JSON"):
            records_text = st.text_area("Record array", value='[{"question":"...","answer":"...","context":"...","metadata":{}}]', height=150)
            if st.button("Ingest JSON body"):
                try:
                    records = parse_json_input(records_text, "Records", [])
                    with st.spinner("Ingesting JSON records…"):
                        result = api_result(lambda: client.ingest_json(project_id, records))
                    if result:
                        st.success(f"Ingested {result.get('records_ingested', len(records))} record(s).")
                except ValueError as exc:
                    st.error(str(exc))

    with records_tab:
        search_col, page_col, size_col = st.columns([2, 1, 1])
        search = search_col.text_input("Search records", key="ingested_search")
        page = page_col.number_input("Page", min_value=1, value=1, step=1, key="records_page")
        page_size = size_col.selectbox("Rows", [25, 50, 100, 250, 500], index=1)
        if st.button("Load records"):
            with st.spinner("Loading records…"):
                response = api_result(lambda: client.ingested_data(project_id, int(page), page_size, search))
            if response:
                st.session_state.ingested_records_response = response
        if st.session_state.get("ingested_records_response"):
            st.dataframe(st.session_state.ingested_records_response.get("records", []), width="stretch", hide_index=True)
            st.caption(f"{st.session_state.ingested_records_response.get('total', 0)} total record(s)")

    with dataset_tab:
        fields_result = None
        if st.button("Load metadata fields"):
            with st.spinner("Loading fields…"):
                fields_result = api_result(lambda: client.metadata_fields(project_id))
            if fields_result:
                st.session_state.metadata_fields = fields_result.get("fields", [])
        if st.session_state.get("metadata_fields"):
            st.caption("Available metadata: " + ", ".join(st.session_state.metadata_fields))
        with st.form("generate_dataset_form"):
            pairs_per_chunk = st.number_input("Pairs per chunk", min_value=1, max_value=20, value=3)
            total_limit = st.number_input("Maximum pairs", min_value=1, max_value=5000, value=100)
            generation_mode = st.selectbox("Generation mode", ["from_ingested", "llm_augmented"])
            include_context = st.checkbox("Include source context", value=True)
            difficulty = st.multiselect("Difficulty", ["easy", "medium", "hard"], default=["easy", "medium", "hard"])
            conditions_text = st.text_input("Metadata conditions JSON", value="{}")
            generate_dataset_clicked = st.form_submit_button("Generate dataset", type="primary")
        if generate_dataset_clicked:
            try:
                config = {
                    "num_pairs_per_chunk": int(pairs_per_chunk), "total_limit": int(total_limit),
                    "query_conditions": parse_json_input(conditions_text, "Conditions", {}),
                    "include_context": include_context, "difficulty_levels": difficulty,
                    "generation_mode": generation_mode,
                }
                with st.spinner("Generating dataset…"):
                    response = api_result(lambda: client.generate_dataset(project_id, config))
                if response:
                    st.success(f"Generated {response.get('total_pairs', 0)} test pair(s).")
                    st.json(response)
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Refresh datasets"):
            with st.spinner("Loading datasets…"):
                response = api_result(lambda: client.datasets(project_id))
            if response:
                st.session_state.datasets_response = response
        dataset_items = st.session_state.get("datasets_response", {}).get("datasets", [])
        if dataset_items:
            dataset_ids = [item.get("dataset_id", "") for item in dataset_items]
            selected_dataset = st.selectbox("Dataset", dataset_ids)
            action_cols = st.columns(3)
            if action_cols[0].button("Open dataset"):
                with st.spinner("Loading dataset…"):
                    show_json(api_result(lambda: client.dataset(project_id, selected_dataset)))
            if action_cols[1].button("Delete dataset"):
                with st.spinner("Deleting dataset…"):
                    deleted = api_result(lambda: client.delete_dataset(project_id, selected_dataset), success_message="Dataset deleted.")
                if deleted:
                    st.session_state.datasets_response = api_result(lambda: client.datasets(project_id)) or {"datasets": []}
            action_cols[2].caption(f"{len(dataset_items)} dataset(s)")


def _neighbor_rank_text(rank: int | None, k: int) -> str:
    if rank is not None and rank <= k:
        return f"Rank {rank}"
    return f"Outside top-{k}"


def _render_neighbor_pair(
    item: dict[str, Any],
    index: int,
    k: int,
    failure: dict[str, Any] | None = None,
) -> None:
    pair = item.get("pair", ("?", "?"))
    status = {
        "mutual": "Mutual",
        "one_way": "One-way",
        "neither": "Neither",
        "unavailable": "Unavailable",
    }.get(item.get("status"), "Unknown")
    with st.container(border=True):
        st.caption(f"Pair {index}")
        st.text(f"{pair[0]}  ↔  {pair[1]}")
        st.caption(
            f"A to B: {_neighbor_rank_text(item.get('rank_b_from_a'), k)}  ·  "
            f"B to A: {_neighbor_rank_text(item.get('rank_a_from_b'), k)}  ·  "
            f"Result: {status}"
        )
        for label, metadata in (
            ("Chunk A metadata", item.get("metadata_a") or {}),
            ("Chunk B metadata", item.get("metadata_b") or {}),
        ):
            st.caption(label)
            st.write(metadata or "Metadata unavailable.")
        st.caption("Chunk A content")
        st.text(item.get("chunk_a") or (failure or {}).get("chunk_a") or "Text unavailable.")
        st.caption("Chunk B content")
        st.text(item.get("chunk_b") or (failure or {}).get("chunk_b") or "Text unavailable.")


def render_execution(client: ApiClient) -> None:
    project_id = current_project_id()
    if not project_id:
        st.info("Select a project in App Under Test Config before running checks.")
        return
    st.subheader("Execution")
    st.caption(
        f"Run checks against the persistent Chroma collection for project "
        f"`{project_id}`. Select a check to reveal its settings."
    )
    check_labels = {
        "vector_integrity": "Vector integrity",
        "neighbor_consistency": "Neighbor consistency",
        "duplicate_detection": "Duplicate similarity candidates",
        "metadata_completeness": "Metadata completeness",
        "index_health": "Index health",
        "negative_query_similarity": "Negative queries (off-topic)",
        "synthetic_question_retrieval": "Synthetic question retrieval",
    }
    check_descriptions = {
        "vector_integrity": "Find empty, non-finite, inconsistent, or unusually scaled vectors.",
        "neighbor_consistency": (
            "Compare adjacent chunks from the same source. Informational only; "
            "not a relevance score."
        ),
        "duplicate_detection": "Find similar vector pairs within a configurable cosine range.",
        "metadata_completeness": "Check selected metadata fields for missing or empty values.",
        "index_health": "Verify that stored vectors can be retrieved from the index.",
        "negative_query_similarity": (
            "Check that unrelated questions receive low nearest-chunk cosine scores."
        ),
        "synthetic_question_retrieval": (
            "Generate questions from a selected chunk and verify it appears in top-k retrieval."
        ),
    }
    selected_checks: list[str] = []
    required_fields = ""
    duplicate_min_similarity = 0.65
    duplicate_max_similarity = 0.98
    negative_query_texts: list[str] = []
    negative_query_max_similarity = DEFAULT_MAX_SIMILARITY
    synthetic_question_record_id = ""
    synthetic_question_record_text = ""
    synthetic_question_count = 2
    synthetic_question_top_k = 1
    st.markdown("#### Select checks")
    st.caption("Checks and their settings are arranged vertically.")
    quick_checks = (
        "vector_integrity",
        "neighbor_consistency",
        "index_health",
    )
    configured_checks = (
        "duplicate_detection",
        "metadata_completeness",
        "negative_query_similarity",
        "synthetic_question_retrieval",
    )
    for check_id in (*quick_checks, *configured_checks):
        with st.container(border=True):
            is_selected = st.checkbox(
                check_labels[check_id],
                value=check_id not in {
                    "metadata_completeness",
                    "negative_query_similarity",
                    "synthetic_question_retrieval",
                },
                key=f"check_{check_id}",
            )
            st.caption(check_descriptions[check_id])
            if not is_selected:
                continue
            selected_checks.append(check_id)

            if check_id == "metadata_completeness":
                required_fields = st.text_input(
                    "Required metadata fields",
                    value="pdf_name, pdf_sha256, page_number, chunk_index",
                    key="required_metadata_fields",
                    help="Separate field names with commas.",
                    placeholder="pdf_name, page_number, chunk_index",
                )
                st.caption(
                    "Enter the metadata keys every stored chunk should contain."
                )
            elif check_id == "duplicate_detection":
                range_columns = st.columns(2, gap="medium")
                duplicate_min_similarity = range_columns[0].number_input(
                    "Similarity from",
                    min_value=0.65,
                    max_value=0.98,
                    value=0.65,
                    step=0.01,
                    format="%.3f",
                    key="duplicate_min_similarity",
                    help="Inclusive lower bound for candidate pairs.",
                )
                duplicate_max_similarity = range_columns[1].number_input(
                    "Similarity to",
                    min_value=0.65,
                    max_value=0.98,
                    value=0.98,
                    step=0.01,
                    format="%.3f",
                    key="duplicate_max_similarity",
                    help="Inclusive upper bound for candidate pairs.",
                )
                st.caption(
                    "Candidate pairs inside this inclusive range are shown. "
                    "Similarity alone does not prove duplicate content."
                )
            elif check_id == "negative_query_similarity":
                negative_query_input = st.text_area(
                    "Off-topic questions",
                    placeholder=(
                        "What is the best recipe for chocolate cake?\n"
                        "Who won the 1998 World Cup?"
                    ),
                    height=180,
                    key="negative_query_texts",
                    help="Enter one unrelated question per line.",
                )
                negative_query_texts = [
                    query.strip()
                    for query in negative_query_input.splitlines()
                    if query.strip()
                ]
                threshold_columns = st.columns([1, 2], gap="medium")
                negative_query_max_similarity = threshold_columns[0].number_input(
                    "Maximum cosine similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_MAX_SIMILARITY,
                    step=0.01,
                    format="%.3f",
                    key="negative_query_max_similarity_v2",
                    help="Off-topic questions above this score fail the check.",
                )
                threshold_columns[1].caption(
                    "Scores above this limit may indicate a relevant match. "
                    "The 0.32 default is based on a small sample from "
                    "tech_project; recalibrate for other projects."
                )
            elif check_id == "synthetic_question_retrieval":
                st.caption(
                    "Generate 1–3 questions from one selected Chroma chunk. "
                    "A question passes only when that exact chunk ID is retrieved."
                )
                question_columns = st.columns(2, gap="medium")
                synthetic_question_count = question_columns[0].number_input(
                    "Questions to generate",
                    min_value=1,
                    max_value=3,
                    value=2,
                    step=1,
                    key="synthetic_question_count",
                )
                synthetic_question_top_k = question_columns[1].number_input(
                    "Retrieval top-k",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1,
                    key="synthetic_question_top_k_v2",
                    help="The selected source chunk must appear within these results.",
                )
                if st.button(
                    "Show records",
                    key=f"show_synthetic_records_{project_id}",
                ):
                    records_response = api_result(
                        lambda: client.request(
                            "GET",
                            f"/api/projects/{client._id(project_id)}/vector-records",
                        )
                    )
                    if records_response is not None:
                        st.session_state["synthetic_records_project_id"] = project_id
                        st.session_state["synthetic_records_response"] = records_response

                records_response = st.session_state.get(
                    "synthetic_records_response"
                )
                if (
                    records_response
                    and st.session_state.get("synthetic_records_project_id")
                    == project_id
                ):
                    records = records_response.get("records", [])
                    st.caption(
                        f"{len(records)} chunks available to select."
                    )
                    if records:
                        record_by_id = {
                            record["id"]: record for record in records
                        }

                        def record_label(record_id: str) -> str:
                            record = record_by_id[record_id]
                            preview = " ".join(record.get("text", "").split())
                            if len(preview) > 100:
                                preview = preview[:100] + "…"
                            return f"{record_id} · {preview or '[no text]'}"

                        synthetic_question_record_id = st.selectbox(
                            "Select source chunk (search by ID or text)",
                            options=list(record_by_id),
                            format_func=record_label,
                            key=f"synthetic_record_select_{project_id}",
                        )
                        selected_record = record_by_id[synthetic_question_record_id]
                        synthetic_question_record_text = selected_record.get("text", "")
                        with st.expander("Selected chunk content and metadata"):
                            st.text(selected_record.get("text") or "No text stored.")
                            st.json(selected_record.get("metadata") or {})
                        if st.button(
                            "Generate questions",
                            key=f"generate_synthetic_questions_{project_id}",
                            disabled=not synthetic_question_record_text.strip(),
                        ):
                            try:
                                with st.spinner(
                                    "Generating synthetic questions with Azure OpenAI…"
                                ):
                                    generated_questions = generate_synthetic_questions(
                                        synthetic_question_record_text,
                                        int(synthetic_question_count),
                                        st.session_state.get("azure_api_key", ""),
                                        st.session_state.get("azure_endpoint", ""),
                                        st.session_state.get("azure_deployment", ""),
                                        st.session_state.get("azure_api_version", ""),
                                    )
                                st.session_state["synthetic_question_generation"] = {
                                    "project_id": project_id,
                                    "record_id": synthetic_question_record_id,
                                    "source_text": synthetic_question_record_text,
                                    "question_count": int(synthetic_question_count),
                                    "questions": generated_questions,
                                }
                            except ValueError as exc:
                                st.error(str(exc))
                            except RuntimeError as exc:
                                st.error(str(exc))
                            except OpenAIError as exc:
                                st.error(
                                    f"Azure OpenAI question generation failed: {exc}"
                                )

                        generation = st.session_state.get(
                            "synthetic_question_generation",
                            {},
                        )
                        if (
                            generation.get("project_id") == project_id
                            and generation.get("record_id")
                            == synthetic_question_record_id
                            and generation.get("source_text")
                            == synthetic_question_record_text
                            and generation.get("question_count")
                            == int(synthetic_question_count)
                        ):
                            st.markdown("**Generated questions**")
                            for index, question in enumerate(
                                generation.get("questions", []),
                                start=1,
                            ):
                                st.markdown(f"{index}. {question}")
                    else:
                        st.warning("No text records are available in this collection.")
                else:
                    st.caption("Select **Show records** to browse the project chunks.")

    run_checks_clicked = st.button(
        "Run selected checks",
        type="primary",
        key="run_project_vector_checks",
        disabled=not selected_checks,
    )

    if run_checks_clicked:
        try:
            if duplicate_min_similarity > duplicate_max_similarity:
                raise ValueError(
                    "Minimum cosine similarity cannot be greater than maximum."
                )
            if (
                "negative_query_similarity" in selected_checks
                and not negative_query_texts
            ):
                raise ValueError(
                    "Enter at least one off-topic question for the negative-query check."
                )
            if (
                "synthetic_question_retrieval" in selected_checks
                and not synthetic_question_record_id
            ):
                raise ValueError(
                    "Select a source chunk using Show records before running the synthetic question check."
                )
            synthetic_question_texts: list[str] = []
            if "synthetic_question_retrieval" in selected_checks:
                generation = st.session_state.get(
                    "synthetic_question_generation",
                    {},
                )
                if (
                    generation.get("project_id") != project_id
                    or generation.get("record_id")
                    != synthetic_question_record_id
                    or generation.get("source_text")
                    != synthetic_question_record_text
                    or generation.get("question_count")
                    != int(synthetic_question_count)
                    or len(generation.get("questions", []))
                    != int(synthetic_question_count)
                ):
                    raise ValueError(
                        "Generate the requested questions for the selected chunk before running checks."
                    )
                synthetic_question_texts = generation["questions"]
            if (
                "negative_query_similarity" in selected_checks
                and len(negative_query_texts) > 100
            ):
                raise ValueError(
                    "Enter no more than 100 off-topic questions."
                )
            if any(len(query) > 2000 for query in negative_query_texts):
                raise ValueError(
                    "Each off-topic question must be at most 2,000 characters."
                )
            fields = [field.strip() for field in required_fields.split(",") if field.strip()]
            with st.spinner("Scanning the project vector store and building the report…"):
                result = api_result(
                    lambda: client.request(
                        "POST",
                        f"/api/projects/{client._id(project_id)}/run-vector-checks",
                        json_body={
                            "selected_checks": selected_checks,
                            "required_fields": fields,
                            "duplicate_min_similarity": duplicate_min_similarity,
                            "duplicate_max_similarity": duplicate_max_similarity,
                            "negative_query_texts": negative_query_texts,
                            "negative_query_max_similarity": negative_query_max_similarity,
                            "synthetic_question_record_id": synthetic_question_record_id,
                            "synthetic_question_texts": synthetic_question_texts,
                            "synthetic_question_top_k": synthetic_question_top_k,
                        },
                    )
                )
            if result:
                st.session_state.project_check_report = result
                st.success(
                    f"Checks complete: {result['summary']['passed']} passed, "
                    f"{result['summary']['failed']} failed, "
                    f"{result['summary']['inconclusive']} informational."
                )
        except ValueError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
        except OpenAIError as exc:
            st.error(f"Azure OpenAI question generation failed: {exc}")

    report = st.session_state.get("project_check_report")
    if report and report.get("project_id") == project_id:
        st.subheader("Check results")
        st.dataframe(
            [
                {
                    "Check": {
                        "known_similar_pairs": "Automatic adjacent-chunk consistency",
                        "near_duplicate_detection": "Duplicate similarity candidates",
                        "negative_query_similarity": "Negative-query similarity",
                        "synthetic_question_retrieval": "Synthetic question retrieval",
                    }.get(item.get("check"), item.get("check")),
                    "Status": (
                        "PASS" if item.get("passed") is True
                        else "FAIL" if item.get("passed") is False
                        else "INFO"
                    ),
                }
                for item in report.get("results", [])
            ],
            width="stretch",
            hide_index=True,
        )
        neighbor_result = next(
            (
                item for item in report.get("results", [])
                if item.get("check") == "known_similar_pairs"
            ),
            None,
        )
        negative_query_result = next(
            (
                item for item in report.get("results", [])
                if item.get("check") == "negative_query_similarity"
            ),
            None,
        )
        synthetic_result = next(
            (
                item for item in report.get("results", [])
                if item.get("check") == "synthetic_question_retrieval"
            ),
            None,
        )
        if negative_query_result:
            negative_detail = negative_query_result.get("detail", {})
            st.markdown("#### Negative-query calibration")
            st.caption(
                f"Tested {negative_detail.get('queries_tested', 0)} off-topic "
                "question(s). Lower cosine similarity is better."
            )
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "Questions tested",
                f"{negative_detail.get('queries_tested', 0)}",
            )
            metric_cols[1].metric(
                "Highest score",
                f"{negative_detail.get('maximum_negative_similarity', 0):.3f}",
            )
            metric_cols[2].metric(
                "Configured maximum",
                (
                    f"{negative_detail.get('maximum_similarity_threshold', DEFAULT_MAX_SIMILARITY):.3f}"
                ),
            )
            negative_rows = [
                {
                    "Question": item.get("query", ""),
                    "Cosine score": item.get("cosine_similarity"),
                    "Result": (
                        "Above maximum"
                        if item.get("above_threshold")
                        else "Below maximum"
                    ),
                }
                for item in negative_detail.get("query_results", [])
            ]
            if negative_rows:
                st.dataframe(
                    negative_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Question": st.column_config.TextColumn(width="large"),
                        "Cosine score": st.column_config.NumberColumn(
                            format="%.4f"
                        ),
                    },
                )
            for item in negative_detail.get("query_results", []):
                chunk = item.get("nearest_chunk")
                if item.get("above_threshold") and chunk:
                    with st.expander(
                        f"Top matching chunk for: {item.get('query', '')}"
                    ):
                        st.caption(
                            f"Chunk ID: {chunk.get('id', '')} · "
                            f"Cosine score: {item.get('cosine_similarity', 0):.4f}"
                        )
                        st.text(chunk.get("text", ""))
        if synthetic_result:
            synthetic_detail = synthetic_result.get("detail", {})
            st.markdown("#### Synthetic question retrieval")
            st.caption(
                f"Source chunk: {synthetic_detail.get('source_record_id', '')} · "
                f"top-{synthetic_detail.get('top_k', 0)} retrieval"
            )
            st.write(
                f"**Passed:** {synthetic_detail.get('passed_questions', 0)} · "
                f"**Failed:** {synthetic_detail.get('failed_questions', 0)}"
            )
            with st.expander("Selected source chunk"):
                st.text(
                    synthetic_detail.get("source_chunk_text")
                    or "No text stored."
                )
            st.dataframe(
                [
                    {
                        "Generated question": item.get("question", ""),
                        "Source chunk rank": (
                            item.get("source_rank")
                            if item.get("source_retrieved")
                            else "Not in top-k"
                        ),
                        "Retrieved record IDs": ", ".join(
                            item.get("retrieved_ids", [])
                        ),
                    }
                    for item in synthetic_detail.get("question_results", [])
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "Generated question": st.column_config.TextColumn(width="large"),
                    "Retrieved record IDs": st.column_config.TextColumn(width="large"),
                },
            )
        if neighbor_result:
            neighbor_detail = neighbor_result.get("detail", {})
            k = neighbor_detail.get("k", 10)
            st.markdown("#### Neighbor check")
            st.caption(
                f"Informational only: checks whether sampled adjacent chunks "
                f"appear in each other's top-{k} results. A miss is not an error."
            )
            st.write(
                f"**Pairs checked:** {neighbor_detail.get('pairs_checked', 0)} "
                f"of {neighbor_detail.get('pairs_sampled', 0)} sampled  ·  "
                f"**Not mutual top-{k}:** "
                f"{neighbor_detail.get('pairs_not_mutual_top_k', 0)}"
            )
            pair_results = neighbor_detail.get("pair_results", [])
            missed_pairs = neighbor_detail.get("pairs_outside_top_k", [])
            failure_by_pair = {
                tuple(item.get("pair", ())): item
                for item in missed_pairs
            }
            if pair_results:
                with st.expander(
                    f"All sampled pairs ({len(pair_results)})",
                    expanded=False,
                ):
                    for index, item in enumerate(pair_results, start=1):
                        _render_neighbor_pair(
                            item,
                            index,
                            k,
                            failure_by_pair.get(tuple(item.get("pair", ()))),
                        )
            if missed_pairs:
                with st.expander(
                    f"Outside top-{k} ({len(missed_pairs)})",
                    expanded=True,
                ):
                    for index, failure in enumerate(missed_pairs, start=1):
                        pair_result = next(
                            (
                                item for item in pair_results
                                if tuple(item.get("pair", ()))
                                == tuple(failure.get("pair", ()))
                            ),
                            {
                                "pair": failure.get("pair", ("?", "?")),
                                "rank_b_from_a": failure.get("rank_b_from_a"),
                                "rank_a_from_b": failure.get("rank_a_from_b"),
                                "status": "unavailable",
                                "metadata_a": {},
                                "metadata_b": {},
                                "chunk_a": failure.get("chunk_a", ""),
                                "chunk_b": failure.get("chunk_b", ""),
                            },
                        )
                        _render_neighbor_pair(pair_result, index, k, failure)
            if neighbor_detail.get("diagnostic_status") == "inconclusive":
                st.info(neighbor_detail.get("message", "The check is inconclusive."))

        duplicate_result = next(
            (
                item for item in report.get("results", [])
                if item.get("check") == "near_duplicate_detection"
            ),
            None,
        )
        if duplicate_result:
            duplicate_detail = duplicate_result.get("detail", {})
            duplicate_pairs = duplicate_detail.get(
                "similarity_candidates",
                duplicate_detail.get("duplicate_pairs", []),
            )
            minimum_similarity = duplicate_detail.get("minimum_similarity", 0.65)
            maximum_similarity = duplicate_detail.get("maximum_similarity", 0.98)
            st.markdown("#### Duplicate similarity candidates")
            st.caption(
                f"Shows up to {duplicate_detail.get('neighbors_per_vector', 5)} "
                f"nearest neighbors per vector; cosine range "
                f"{minimum_similarity:.2f}–{maximum_similarity:.2f}. "
                f"Found {duplicate_detail.get('candidate_pairs_found', 0)} "
                f"candidate pair(s) among {duplicate_detail.get('vectors_sampled', 0)} "
                "sampled vectors. Cosine similarity is not proof of duplicate text."
            )
            if duplicate_pairs:
                with st.expander(
                    f"Show candidate pairs ({len(duplicate_pairs)})",
                    expanded=False,
                ):
                    for index, pair in enumerate(duplicate_pairs, start=1):
                        with st.expander(
                            f"Pair {index}: {pair.get('id_a', '?')} ↔ "
                            f"{pair.get('id_b', '?')} · cosine "
                            f"{pair.get('cosine_similarity', 0):.4f}",
                            expanded=False,
                        ):
                            left, right = st.columns(2)
                            for column, label, metadata_key, chunk_key in (
                                (left, "Record A", "metadata_a", "chunk_a"),
                                (right, "Record B", "metadata_b", "chunk_b"),
                            ):
                                column.markdown(f"**{label}**")
                                column.caption(str(pair.get(
                                    "id_a" if label == "Record A" else "id_b", "?"
                                )))
                                column.write(pair.get(metadata_key) or "Metadata unavailable.")
                                column.text_area(
                                    "Chunk content",
                                    value=pair.get(chunk_key) or "Text unavailable.",
                                    height=140,
                                    disabled=True,
                                    key=f"duplicate_chunk_{index}_{label}",
                                )
                            overlap = pair.get("text_token_overlap")
                            st.caption(
                                "Text token overlap: "
                                + (f"{overlap:.4f}" if overlap is not None else "Unavailable")
                                + " · Compare the text before classifying this as a duplicate."
                            )
            if duplicate_detail.get(
                "similarity_candidates_truncated",
                duplicate_detail.get("duplicate_pairs_truncated"),
            ):
                st.info(
                    "The report displays the top 50 candidates by cosine similarity. "
                    "The total count includes all candidate pairs found."
                )

        metadata_result = next(
            (
                item for item in report.get("results", [])
                if item.get("check") == "metadata_completeness"
            ),
            None,
        )
        if metadata_result:
            metadata_detail = metadata_result.get("detail", {})
            scanned = metadata_detail.get("total_scanned", 0)
            per_field = metadata_detail.get("per_field", {})
            st.markdown("#### Metadata field counts")
            st.caption(
                f"Counts are based on {scanned} scanned records. "
                "Fields listed are the required fields selected for this check."
            )
            st.dataframe(
                [
                    {
                        "Metadata field": field,
                        "Populated": max(
                            scanned - info.get("total_incomplete", 0),
                            0,
                        ),
                        "Missing": info.get("missing", 0),
                        "Empty": info.get("empty", 0),
                        "Completeness": (
                            f'{info.get("completeness_pct", 0):g}%'
                        ),
                    }
                    for field, info in per_field.items()
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "Metadata field": st.column_config.TextColumn(
                        width="medium"
                    ),
                    "Populated": st.column_config.NumberColumn(
                        help="Records where the field is present and non-empty."
                    ),
                    "Missing": st.column_config.NumberColumn(
                        help="Records without this metadata field."
                    ),
                    "Empty": st.column_config.NumberColumn(
                        help="Records where the field exists but has an empty value."
                    ),
                    "Completeness": st.column_config.TextColumn(width="small"),
                },
            )

        other_results = [
            item for item in report.get("results", [])
            if item.get("check") not in {
                "known_similar_pairs",
                "metadata_completeness",
                "near_duplicate_detection",
                "negative_query_similarity",
            }
        ]
        if other_results:
            with st.expander("Other check details"):
                st.json(other_results)
        st.download_button(
            "Download HTML report",
            report["html_report"],
            file_name=f"{report['report_id']}.html",
            mime="text/html",
        )
        st.download_button(
            "Download JSON results",
            json.dumps(
                {
                    "report_id": report["report_id"],
                    "summary": report["summary"],
                    "results": report["results"],
                },
                indent=2,
            ),
            file_name=f"{report['report_id']}.json",
            mime="application/json",
        )


def render_vector_checks(client: ApiClient) -> None:
    st.subheader("Vector database checks")
    connect_tab, checks_tab, records_tab, report_tab = st.tabs(["Connect & map", "Checks", "Records & embeddings", "Reports"])
    with connect_tab:
        with st.form("vector_connect_form"):
            db_type = st.selectbox("Database connector", ["chroma"])
            collection_name = st.text_input("Collection", value="default")
            persist_directory = st.text_input("Chroma directory", value="./chroma_data")
            host = st.text_input("Remote host (optional)")
            port = st.number_input("Remote port", min_value=1, max_value=65535, value=8000)
            connect_clicked = st.form_submit_button("Connect database", type="primary")
        if connect_clicked:
            params = {"collection_name": collection_name, "persist_directory": persist_directory}
            if host:
                params.update({"host": host, "port": int(port)})
            with st.spinner("Connecting to vector database…"):
                result = api_result(lambda: client.connect(db_type, params))
            if result:
                st.session_state.vector_connection = result
                st.success(f"Connected · {result.get('record_count', 0)} record(s)")
                st.json(result)
        if st.session_state.get("vector_connection"):
            st.caption(f"Connection: {st.session_state.vector_connection}")
        mapper_col, mapper_button = st.columns([4, 1])
        mapper_json = mapper_col.text_area(
            "Field mapper JSON",
            value=json.dumps(st.session_state.get("field_mapper", {}), indent=2),
            height=220,
        )
        if mapper_button.button("Load mapper"):
            with st.spinner("Loading mapper…"):
                mapper = api_result(client.get_field_mapper)
            if mapper is not None:
                st.session_state.field_mapper = mapper
                st.rerun()
        if st.button("Save mapper"):
            try:
                config = parse_json_input(mapper_json, "Field mapper", {})
                with st.spinner("Saving mapper…"):
                    api_result(lambda: client.save_field_mapper(config), success_message="Field mapper saved.")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("List collections"):
            with st.spinner("Loading collections…"):
                show_json(api_result(client.collections))

    with checks_tab:
        with st.form("run_checks_form"):
            required_fields = st.text_input("Required metadata fields", value="doc_id")
            batch_size = st.number_input("Batch size", min_value=1, max_value=10000, value=500)
            sample_size = st.number_input("Sample size", min_value=1, max_value=50000, value=500)
            run_checks_clicked = st.form_submit_button("Run all checks", type="primary")
        if run_checks_clicked:
            try:
                config = {
                    "required_fields": [field.strip() for field in required_fields.split(",") if field.strip()],
                    "batch_size": int(batch_size), "sample_size": int(sample_size),
                }
                with st.spinner("Running vector-quality checks…"):
                    results = api_result(lambda: client.run_checks(config))
                if results is not None:
                    st.session_state.legacy_results = results
                    st.success("Vector checks completed.")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Load latest results"):
            with st.spinner("Loading results…"):
                result = api_result(client.results)
            if result is not None:
                st.session_state.legacy_results = result
        if st.session_state.get("legacy_results") is not None:
            st.json(st.session_state.legacy_results)

    with records_tab:
        with st.form("record_search_form"):
            search_field = st.text_input("Metadata field", value="doc_id")
            search_value = st.text_input("Exact value")
            search_limit = st.number_input("Maximum records", min_value=1, max_value=500, value=50)
            search_clicked = st.form_submit_button("Search records")
        if search_clicked:
            with st.spinner("Searching records…"):
                result = api_result(lambda: client.search_records(search_field, search_value, int(search_limit)))
            if result:
                st.dataframe(result.get("records", []), width="stretch", hide_index=True)
        record_id = st.text_input("Record ID for details")
        if st.button("Load record details", disabled=not record_id):
            with st.spinner("Loading record…"):
                show_json(api_result(lambda: client.record(record_id)))
        sample_size = st.number_input("Embedding sample size", min_value=1, max_value=50000, value=2000)
        projection_method = st.selectbox("Projection", ["pca", "random"])
        if st.button("Load embedding projection"):
            with st.spinner("Projecting embeddings…"):
                result = api_result(lambda: client.embedding_data(int(sample_size), projection_method))
            if result:
                points = result.get("points", []) if isinstance(result, dict) else result
                st.dataframe(points, width="stretch", hide_index=True)
        with st.expander("Generate automatic validation pairs"):
            group_field = st.text_input("Group field", value="doc_id")
            contrast_field = st.text_input("Contrast field", value="object_type")
            similar_count = st.number_input("Similar pairs per group", min_value=0, max_value=100, value=3)
            different_count = st.number_input("Different pairs total", min_value=0, max_value=1000, value=20)
            pair_output = st.text_input("Output path", value="auto_pairs.xlsx")
            if st.button("Generate pairs"):
                config = {"group_field": group_field, "contrast_field": contrast_field or None,
                          "similar_pairs_per_group": int(similar_count),
                          "different_pairs_total": int(different_count), "output_path": pair_output}
                with st.spinner("Generating validation pairs…"):
                    show_json(api_result(lambda: client.auto_pairs(config)))
        with st.expander("Review pairs with OpenAI"):
            review_input = st.text_input("Input workbook", value="auto_pairs.xlsx")
            review_output = st.text_input("Output workbook", value="reviewed_pairs.xlsx")
            review_model = st.text_input("OpenAI model", value="gpt-4")
            if st.button("Run pair review"):
                with st.spinner("Reviewing pairs…"):
                    show_json(api_result(lambda: client.llm_review({"input_path": review_input, "output_path": review_output, "model": review_model})))

    with report_tab:
        html_col, json_col = st.columns(2)
        if html_col.button("Build latest HTML report"):
            with st.spinner("Building HTML report…"):
                report = api_result(client.export_html)
            if report:
                html_col.download_button("Download HTML", report, file_name="vector_db_report.html", mime="text/html")
        if json_col.button("Download latest JSON results"):
            with st.spinner("Loading JSON report…"):
                report = api_result(client.export_json)
            if report is not None:
                json_col.download_button("Download JSON", json.dumps(report, indent=2), file_name="vector_db_results.json", mime="application/json")


def _selected_visual_record_ids(
    selected_points: list[dict[str, Any]],
    figure: Any,
) -> list[str]:
    selected_ids: list[str] = []
    for selected_point in selected_points:
        custom_data = selected_point.get("customdata")
        selected_id = None
        if isinstance(custom_data, (list, tuple)) and custom_data:
            selected_id = custom_data[0]
        elif isinstance(custom_data, str):
            selected_id = custom_data
        if selected_id is None:
            curve_number = selected_point.get("curve_number")
            point_index = selected_point.get("point_index")
            if (
                isinstance(curve_number, int)
                and isinstance(point_index, int)
                and 0 <= curve_number < len(figure.data)
            ):
                trace_customdata = figure.data[curve_number].customdata
                if trace_customdata is not None and point_index < len(trace_customdata):
                    trace_point_data = trace_customdata[point_index]
                    if len(trace_point_data):
                        selected_id = trace_point_data[0]
        if selected_id is not None:
            selected_ids.append(str(selected_id))
    return selected_ids


def render_visual_check(client: ApiClient) -> None:
    import pandas as pd
    import plotly.express as px

    project_id = current_project_id()
    if not project_id:
        st.info("Select a project in App Under Test Config to visualize its Chroma vectors.")
        return

    if st.session_state.get("visual_check_project_id") != project_id:
        st.session_state.pop("visual_check_data", None)
        st.session_state.pop("visual_check_selected_id", None)
        st.session_state["visual_check_project_id"] = project_id

    st.subheader("Visual Check")
    st.caption(
        f"UMAP projects vectors from project `{project_id}` into 2D. "
        "HDBSCAN groups nearby points in the displayed 2D projection; "
        "points labeled -1 are noise."
    )
    if st.session_state.get("visual_check_sample_size") == 2000:
        st.session_state["visual_check_sample_size"] = 300
    if st.session_state.get("visual_check_min_cluster_size") == 5:
        st.session_state["visual_check_min_cluster_size"] = 2
    sample_col, cluster_col, load_col = st.columns([2, 2, 1])
    sample_size = sample_col.number_input(
        "Vectors to visualize",
        min_value=3,
        max_value=10000,
        value=300,
        step=100,
        key="visual_check_sample_size",
    )
    min_cluster_size = cluster_col.number_input(
        "Minimum cluster size",
        min_value=2,
        max_value=100,
        value=2,
        step=1,
        key="visual_check_min_cluster_size",
    )
    if load_col.button("Build map", type="primary", width="stretch"):
        st.session_state.pop("visual_check_selected_id", None)
        st.session_state.pop("visual_check_data", None)
        with st.spinner("Loading Chroma records and calculating UMAP/HDBSCAN…"):
            result = api_result(
                lambda: client.project_visualization_data(
                    project_id,
                    int(sample_size),
                    int(min_cluster_size),
                )
            )
        if result is not None:
            st.session_state["visual_check_data"] = result
            st.session_state["visual_check_generation"] = (
                st.session_state.get("visual_check_generation", 0) + 1
            )

    result = st.session_state.get("visual_check_data")
    if not result:
        st.info("Choose the sample and cluster settings, then select **Build map**.")
        return

    if result.get("message"):
        st.warning(result["message"])
        return

    points = result.get("points", [])
    if not points:
        st.info("No vectors are available to visualize.")
        return

    st.caption(
        f"Collection `{result['collection_name']}` · "
        f"{result['projected_records']:,} plotted / {result['total_records']:,} total "
        f"· {result['skipped_records']:,} records skipped"
    )
    frame = pd.DataFrame(points)
    frame["cluster_label"] = frame["cluster"].map(
        lambda label: "Noise (-1)" if label == -1 else f"Cluster {label}"
    )
    frame["text_preview"] = frame["text"].map(
        lambda text: text[:160] + ("…" if len(text) > 160 else "")
    )
    cluster_options = sorted(
        frame["cluster_label"].unique().tolist(),
        key=lambda label: (-1 if label == "Noise (-1)" else int(label.split()[-1])),
    )
    cluster_counts = frame["cluster_label"].value_counts().to_dict()
    generation = st.session_state.get("visual_check_generation", 0)
    selected_clusters = st.multiselect(
        "Show clusters",
        options=cluster_options,
        default=cluster_options,
        format_func=lambda label: (
            f"{label} · {cluster_counts[label]:,} vectors"
        ),
        key=f"visual_check_clusters_{project_id}_{generation}",
        help="Choose one or more HDBSCAN clusters to display and inspect.",
    )
    visible_frame = frame[frame["cluster_label"].isin(selected_clusters)]
    visible_points = visible_frame.to_dict(orient="records")
    visible_ids = [str(point["id"]) for point in visible_points]
    visible_point_by_id = {
        str(point["id"]): point for point in visible_points
    }
    st.caption(
        f"Showing {len(visible_points):,} of {len(points):,} plotted vectors "
        f"across {len(selected_clusters)} selected cluster(s)."
    )
    if not visible_points:
        st.session_state.pop("visual_check_selected_id", None)
        st.info("Select at least one cluster to display its vectors.")
        return

    figure = px.scatter(
        visible_frame,
        x="x",
        y="y",
        color="cluster_label",
        custom_data=["id"],
        hover_name="id",
        hover_data={
            "cluster_label": True,
            "text_preview": True,
            "x": False,
            "y": False,
            "id": False,
        },
        render_mode="webgl",
        labels={"cluster_label": "Cluster", "text_preview": "Chunk text"},
        category_orders={"cluster_label": cluster_options},
    )
    figure.update_layout(
        height=550,
        margin={"l": 20, "r": 20, "t": 35, "b": 30},
        showlegend=False,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        clickmode="event+select",
        legend={"groupclick": "togglegroup"},
    )
    cluster_key = ",".join(selected_clusters)
    plot_key = f"visual_check_plot_{project_id}_{generation}_{abs(hash(cluster_key))}"
    selection = st.plotly_chart(
        figure,
        width="stretch",
        key=plot_key,
        on_select="rerun",
        selection_mode="points",
    )
    cluster_color_by_label = {
        str(trace.name): trace.marker.color
        for trace in figure.data
        if trace.name is not None
    }

    qa_upload = st.file_uploader(
        "PDF Workbench Q&A workbook",
        type=["xlsx"],
        key="visual_check_qa_workbook",
        help=(
            "Upload the Excel workbook downloaded from PDF Workbench. "
            "Matching questions and answers are shown with their chunk's "
            "UMAP cluster and color."
        ),
    )
    qa_rows: list[dict[str, Any]] = []
    qa_workbook_error = None
    if qa_upload is not None:
        workbook_bytes = qa_upload.getvalue()
        workbook_signature = hashlib.sha256(workbook_bytes).hexdigest()
        if st.session_state.get("visual_check_qa_signature") != workbook_signature:
            try:
                qa_rows = load_workbench_qa_workbook(workbook_bytes)
                st.session_state["visual_check_qa_rows"] = qa_rows
                st.session_state["visual_check_qa_signature"] = workbook_signature
                st.session_state.pop("visual_check_qa_error", None)
            except (ValueError, OSError, RuntimeError) as exc:
                st.session_state["visual_check_qa_rows"] = []
                st.session_state["visual_check_qa_signature"] = workbook_signature
                st.session_state["visual_check_qa_error"] = str(exc)
        qa_rows = st.session_state.get("visual_check_qa_rows", [])
        qa_workbook_error = st.session_state.get("visual_check_qa_error")
    else:
        st.session_state.pop("visual_check_qa_signature", None)
        st.session_state.pop("visual_check_qa_rows", None)
        st.session_state.pop("visual_check_qa_error", None)
        st.session_state.pop("visual_check_qa_match_signature", None)
        st.session_state.pop("visual_check_qa_matches", None)

    if qa_upload is not None and not qa_workbook_error:
        qa_match_signature = hashlib.sha256(
            "\0".join(
                [
                    project_id,
                    str(generation),
                    workbook_signature,
                    *visible_ids,
                ]
            ).encode("utf-8")
        ).hexdigest()
        if (
            st.session_state.get("visual_check_qa_match_signature")
            != qa_match_signature
        ):
            with st.spinner("Matching workbook Q&A to visible UMAP chunks…"):
                linked_points = [
                    (str(point["id"]), match_chunk_qa(point, qa_rows))
                    for point in visible_points
                ]
            st.session_state["visual_check_qa_matches"] = [
                (point_id, matches)
                for point_id, matches in linked_points
                if matches
            ]
            st.session_state["visual_check_qa_match_signature"] = (
                qa_match_signature
            )
        linked_points = [
            (visible_point_by_id[point_id], matches)
            for point_id, matches in st.session_state.get(
                "visual_check_qa_matches", []
            )
            if point_id in visible_point_by_id
        ]
        st.markdown("#### Q&A linked to UMAP chunks")
        if linked_points:
            st.caption(
                f"Found Q&A for {len(linked_points)} of {len(visible_points)} "
                "visible chunks. Each card uses its point's cluster color."
            )
            for point, matching_rows in linked_points:
                cluster = point["cluster"]
                cluster_label = (
                    "Noise (-1)" if cluster == -1 else f"Cluster {cluster}"
                )
                color = cluster_color_by_label.get(
                    str(cluster_label),
                    "#808080",
                )
                with st.container(border=True):
                    st.markdown(
                        f'<span style="display:inline-block;width:12px;height:12px;'
                        f'border-radius:50%;background:{color};margin-right:6px">'
                        f'</span><strong>{cluster_label}</strong>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Chunk ID: {point['id']}")
                    st.text(str(point.get("text") or "No document text stored."))
                    st.dataframe(
                        [
                            {
                                "Category": row.get("category") or "",
                                "Question": row["question"],
                                "Answer": row["expected_answer"],
                            }
                            for row in matching_rows
                        ],
                        width="stretch",
                        hide_index=True,
                    )
        else:
            st.info(
                "No workbook Q&A matched the visible chunks. Confirm the "
                "workbook and selected project contain the same PDF content."
            )
    elif qa_workbook_error:
        st.error(f"Could not read the Q&A workbook: {qa_workbook_error}")

    selected_ids = _selected_visual_record_ids(
        selection.selection.points,
        figure,
    )

    signature_key = f"visual_check_selection_signature_{plot_key}"
    selection_signature = tuple(selected_ids)
    previous_signature = st.session_state.get(signature_key)
    selection_changed = selection_signature != previous_signature
    if selection_changed:
        st.session_state[signature_key] = selection_signature
        if selected_ids:
            st.session_state["visual_check_selected_id"] = selected_ids[-1]
        elif previous_signature:
            st.session_state.pop("visual_check_selected_id", None)

    selected_id = st.session_state.get("visual_check_selected_id")
    if selected_id not in visible_ids:
        selected_id = None
        st.session_state.pop("visual_check_selected_id", None)
    selector_key = f"visual_check_record_selector_{project_id}_{generation}"
    if selection_changed and selected_id is not None:
        st.session_state[selector_key] = selected_id
    elif st.session_state.get(selector_key) not in visible_ids:
        st.session_state[selector_key] = selected_id or visible_ids[0]

    def selected_record_label(record_id: str) -> str:
        text = " ".join(visible_point_by_id[record_id]["text"].split())
        preview = text[:90] + ("…" if len(text) > 90 else "")
        return f"{record_id} · {preview or '[no text]'}"

    selected_id = st.selectbox(
        "Selected vector",
        options=visible_ids,
        format_func=selected_record_label,
        key=selector_key,
        help="Choose a vector here or select a point on the map.",
    )
    st.session_state["visual_check_selected_id"] = selected_id
    selected_record = visible_point_by_id.get(selected_id)
    with st.container(border=True):
        st.markdown("#### Chunk details")
        if selected_record is None:
            st.caption("Select a point to inspect its stored record.")
        else:
            st.caption(f"ID: {selected_record['id']}")
            cluster = selected_record["cluster"]
            st.caption("Cluster: Noise (-1)" if cluster == -1 else f"Cluster: {cluster}")
            st.markdown("**Chunk text**")
            st.text(selected_record["text"] or "No document text stored.")
            st.markdown("**Metadata**")
            st.json(selected_record["metadata"])
            st.markdown("#### Matching PDF Workbench Q&A")
            if qa_upload is None:
                st.info("Upload a PDF Workbench Q&A workbook to view chunk questions and answers.")
            elif qa_workbook_error:
                st.error(f"Could not read the Q&A workbook: {qa_workbook_error}")
            else:
                matching_qa = match_chunk_qa(selected_record, qa_rows)
                if matching_qa:
                    st.dataframe(
                        [
                            {
                                "Category": row.get("category") or "",
                                "Question": row["question"],
                                "Answer": row["expected_answer"],
                            }
                            for row in matching_qa
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                elif (
                    (selected_record.get("metadata") or {}).get("chunk_type") == "text"
                    and any(
                        row.get("source_type") == "text"
                        for row in qa_rows
                    )
                    and not any(
                        row.get("source_text")
                        for row in qa_rows
                        if row.get("source_type") == "text"
                    )
                ):
                    st.warning(
                        "This workbook does not include hidden chunk-linking data. "
                        "Download a fresh Q&A workbook from PDF Workbench, then upload it here."
                    )
                else:
                    st.info(
                        "No table/text Q&A matched this chunk. Check that the selected "
                        "project contains the same PDF and that the workbook was generated "
                        "from it."
                    )


def render_endpoint_explorer(client: ApiClient) -> None:
    st.subheader("Backend endpoint explorer")
    st.caption("All registered FastAPI operations are listed here. Dynamic path values can be entered directly in the path field.")
    st.dataframe(
        [{"Method": method, "Path": path, "Operation": description} for method, path, description in API_ENDPOINTS],
        width="stretch",
        hide_index=True,
    )
    labels = [f"{method}  {path}  ·  {description}" for method, path, description in API_ENDPOINTS]
    selected_index = st.selectbox("Operation", range(len(labels)), format_func=lambda index: labels[index])
    method, template, _ = API_ENDPOINTS[selected_index]
    path = st.text_input("Request path", value=template, key="explorer_path")
    query_text = st.text_area("Query parameters JSON", value="{}", height=75)
    body_text = st.text_area("JSON request body", value="{}" if method in {"POST", "PUT", "PATCH"} else "", height=150)
    upload = st.file_uploader("Multipart file (for upload endpoints)", key="explorer_file")
    if st.button("Send request", type="primary"):
        try:
            params = parse_json_input(query_text, "Query parameters", {})
            body = parse_json_input(body_text, "Request body", None)
            files = None
            if upload:
                files = {"file": (Path(upload.name).name, upload.getvalue(), upload.type or "application/octet-stream")}
                body = None
            with st.spinner(f"Calling {method} {path}…"):
                result = api_result(lambda: client.request(method, path, params=params, json_body=body, files=files))
            show_json(result)
        except ValueError as exc:
            st.error(str(exc))


def main() -> None:
    state_defaults()
    client = render_sidebar()
    st.markdown('<p class="eyebrow">Vector Quality Studio</p>', unsafe_allow_html=True)
    st.title("Document intelligence and vector quality")
    st.caption("Inspect PDFs, configure a PDF-backed application under test, run vector-quality checks, and explore vectors.")

    tabs = st.tabs([
        "PDF Workbench",
        "App Under Test Config",
        "Execution",
        "Answer Improvement",
        "Visual Check",
    ])
    with tabs[0]:
        render_pdf_workbench(client)
    with tabs[1]:
        render_app_under_test_config(client)
    with tabs[2]:
        render_execution(client)
    with tabs[3]:
        render_answer_improvement()
    with tabs[4]:
        render_visual_check(client)


if __name__ == "__main__":
    main()