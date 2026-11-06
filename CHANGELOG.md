# Changelog

## 2026-10-30

### Changed

- Group application service modules under `services/`, dashboard HTTP client under `clients/`, and PDF Workbench logic under `pdf_workbench/`. Preserve root app entry points and existing start commands, and centralize the project data path.
- Strengthen Mock AUT grounding rules for prompt-like content in user questions and retrieved chunks, conflicting sources, table row/column matching, unsupported calculations, and explicit insufficient-evidence answers.
- Isolate the AUT question and retrieved excerpts as JSON input data, strengthen retry instructions, and version the answer prompt as `direct-answer-v3`.
- Make the answer judge treat workbook values as data rather than instructions, clarify scoring rules, serialize evaluation inputs as JSON, and reject scores outside the defined five-point scale.

### Validation

- Compiled all 49 project Python source files and imported the Streamlit app, FastAPI backend, Mock AUT, and demo setup module. Verified services still resolve project data under the root `projects/` directory.
- Importing the standalone demo runner reaches its optional visualization dependency and fails because `matplotlib` is not installed; no dependency was added for this reorganization.
- Python compilation passed for the Mock AUT, judge, and backend prompt-version modules. Live Azure answer generation and judging were not run.

## 2026-10-29

### Fixed

- Recover an outdated Mock AUT child after backend autoreload so workbook evaluation does not send questions to a stale API process.
- Show actionable row-level workbook errors and identify when the AUT needs a restart before answer generation and judging can continue.
- Implement project-persisted drift baselines and configurable curated neighbor pairs; make index health count actual non-empty vectors and report ANN searchability separately.
- Expose source record IDs for neighbor configuration, prevent stale project reports from appearing after project switches, and classify insufficient drift/searchability samples as inconclusive.

### Validation

- Confirmed the newest evaluated workbook failed all nine rows because its AUT responses lacked the LLM-generation metadata.
- Verified the live backend discovers the orphaned AUT process after reload and reports its prompt version as outdated.
- Verified AUT status/restart process discovery, backend/AUT/dashboard Python compilation, and workbook row-level diagnostics with a stubbed AUT/judge. No live Azure request was made.
- Ran all six vector-check categories through the live API against a populated 60-vector Chroma project: integrity, duplicates, metadata, stored embeddings, and 50 sampled ANN searches passed; neighbor checks and drift were correctly marked informational until pairs/baseline are configured.
- Verified curated neighbor pairs against live Chroma, and tested saved drift baselines with deterministic vectors: unchanged data passes, a large distribution shift fails, and insufficient samples remain inconclusive.

## 2026-10-28

### Fixed

- Run synchronous Chroma ingestion in Starlette's thread pool so model embedding/upsert work does not block the FastAPI event loop or health/status requests.
- Run synchronous test executions in Starlette's thread pool so AUT HTTP calls do not block the FastAPI event loop or health/status requests.

### Changed

- Streamline the Streamlit dashboard to PDF Workbench, App Under Test Config, and Execution tabs; remove the dataset-generation, Vector QA, and API Explorer tabs from the UI.
- Add Docling PDF extraction, chunk embeddings, project-persistent Chroma storage, PDF metadata, and content-hash duplicate skipping.
- Enable Docling automatic OCR, retain section-heading context in indexed content, and split large Markdown tables into header-preserving row groups. Version the ingestion pipeline so re-uploaded PDFs are reprocessed after extraction/chunking upgrades.
- Group extracted narrative by page and section, isolate tables from narrative chunks, and enforce both an 800-character and 240-token chunk limit to stay below the embedding model's 256-token truncation.
- Make the Mock AUT retrieve Chroma chunks and use Azure OpenAI credentials supplied at startup to compose grounded answers.
- Use the sidebar Azure OpenAI credentials for the Mock AUT and LLM answer judge; remove duplicate credential entry from App Under Test Config.
- Persist PDF Workbench Excel/JSON outputs locally and add per-question workbook evaluation using the project Mock AUT plus a separate LLM judge call.
- Improve Mock AUT answer prompts to return direct, concise answers from retrieved evidence, raise default retrieval from three to five chunks, and calibrate the judge to penalize buried or incomplete answers.
- Require the Mock AUT to return validated JSON answers, retry invalid/copied-section/table output once, report LLM/retrieval metadata, and mark workbook rows as errors when LLM use or retrieved context is not confirmed. Restart the Mock AUT when its source file changes.
- Recover Mock AUT child-process discovery after backend autoreload, report outdated prompt versions, replace only verified workspace AUT processes on start, reject unrelated port collisions, and display row-level workbook errors in the dashboard.
- Make Execution run selected project vector-check categories and provide downloadable HTML and JSON reports.

### Validation

- Verified `/api/health` remains responsive during a simulated two-second ingestion and both `/api/health` and `/api/mock-aut/status` return HTTP 200 after restarting the backend.
- Verified `/health` returned HTTP 200 in 0.010 seconds during a simulated two-second execution; backend syntax compilation passed.
- Recompiled all changed Python modules and confirmed the Streamlit launcher returned HTTP 200; browser inspection confirmed exactly the three requested tabs and rendered the project-configuration and execution views.
- Verified the Mock AUT start route passes UI-submitted credentials only to the child process environment, without changing the backend process environment.
- Stubbed tests passed for PDF metadata and duplicate-content handling, selected check/report generation, Mock AUT retrieval-to-Azure answer composition, and FastAPI upload/check/report routes.
- Converted the available `test2.pdf` with Docling successfully: 11 pages, 106 extracted segments, and 3 tables. Verified section context is retained and table chunks include their header. A scanned-only PDF was not available to verify OCR recognition quality.
- Synthetic chunk tests passed for sentence-aware text boundaries, table-row splitting, repeated headers, and no lost table rows. Stubbed ingestion tests passed for metadata, duplicate skipping, and automatic reindex when the ingestion version changes.
- Verified table-vs-text structural chunking groups same-section narrative, keeps small tables as distinct chunks, and repeats table headers for oversized tables; reindexing version increased to 3.
- Verified with the actual cached MiniLM tokenizer that narrative and large-table chunks remain at or below 240 tokens, including a 120-row test table.
- Persistent-Chroma native integration remains unverified: a previous temporary native insertion probe terminated the process during `collection.add`.
- Verified the provided 24-row workbook structure is preserved while rows are evaluated sequentially; score 0.6 passes, lower scores fail, and row-level failures are recorded in the result workbook. A stubbed judge and AUT test passed; no live Azure call was made.
- Verified the judge uses the configured sidebar deployment, the Mock AUT query includes the active project, and the Streamlit health endpoint returns HTTP 200 with shared sidebar credentials visible. A live Mock AUT/project session was not started for the UI check.
- Verified with the current project's persistent Chroma collection that relevant table chunks are included in the Azure request; a stubbed Azure response that copied a section/table was rejected and retried, a concise answer passed validation, and workbook rows without confirmed LLM/retrieval metadata were marked ERROR without invoking the judge. Python compilation passed; no live Azure request was made.

## 2026-10-27

### Added

- Added this changelog and a repository instruction requiring future non-trivial changes to be recorded with validation notes.
- Added a Streamlit dashboard for the supplied multi-PDF Docling/Azure OpenAI pipeline, object filtering, Q&A generation, and Excel/JSON exports.
- Added a Requests API client and dashboard workflows for projects, ingestion, datasets, execution, vector checks, and all 40 backend API operations.
- Added `streamlit_launcher.py` to initialize Torch and the Windows Hugging Face symlink workaround before Streamlit's CLI starts.
- Added a shared Torch-backed `all-MiniLM-L6-v2` embedding function for Chroma ingestion and mock-AUT retrieval.

### Changed

- Chroma ingestion and retrieval now pass explicit embeddings so existing collections with persisted default embedding configuration can be reused.
- The Streamlit backend URL defaults to `http://127.0.0.1:8000`.

### Fixed

- Avoided importing Torch after Streamlit/backend dependencies on Windows, preventing the observed `WinError 1114` during startup/ingestion.
- Avoided Chroma's ONNX default embedder, whose native runtime terminated the Python process during ingestion.

### Validation

- Verified the Torch/Docling import sequence and converter construction in the project virtual environment.
- Verified temporary Chroma ingestion and mock-AUT retrieval, including a collection created with the prior default embedding function.
- Verified the backend health and multipart ingestion route, dashboard render, and dependency consistency.