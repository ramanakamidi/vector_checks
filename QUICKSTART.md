# Vector Quality Studio — Quick Start

The Streamlit app has three tabs:

1. **PDF Workbench** — the existing PDF object-analysis, filtering, and Q&A workflow.
2. **App Under Test Config** — create/load a project, parse PDFs with Docling, store embedded chunks in project-persistent Chroma, and start the Mock AUT.
3. **Execution** — select vector-check categories, run them against the active project's Chroma collection, and download HTML or JSON results.

## Source layout

- `app.py`, `server.py`, `mock_aut.py`, and `streamlit_launcher.py` are the app entry points and remain in the project root so the startup commands below do not change.
- `pdf_workbench/` contains PDF object analysis and Q&A logic.
- `services/` contains ingestion, project, execution, embedding, report, and workbook services.
- `clients/` contains the typed dashboard API client.
- `checks/`, `connectors/`, and `report/` contain vector-check logic, database adapters, and report components.
- `projects/`, `streamlit_output/`, and `chroma_data/` contain runtime or generated data; do not remove them as source cleanup.

## Start the application

Use two PowerShell terminals from the project root.

### Terminal 1 — FastAPI backend and Mock AUT manager

No Azure credentials are needed in this terminal. Start the backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

### Terminal 2 — Streamlit dashboard

```powershell
.\venv\Scripts\python.exe .\streamlit_launcher.py
```

Open the Streamlit URL printed in the terminal. The backend URL in the sidebar should be `http://127.0.0.1:8000`. Enter the Azure endpoint, deployment, API version, and API key in the sidebar before choosing **Start Mock AUT** in **App Under Test Config**; its default port is 8001. The Mock AUT and answer judge use the sidebar credentials. Credentials are not saved in the project. If you change them while the Mock AUT is running, stop and restart it.

Use the local backend URL and avoid sending credentials over an untrusted network.

## End-to-end workflow

### 1. Create or load a project

Open **App Under Test Config**, create a project or refresh and select an existing one.

### 2. Upload PDFs

Upload one or more PDF files and select **Parse and ingest PDFs**. Docling uses OCR for scanned text when its automatically selected OCR engine is available, and uses its accurate table-structure mode for tables. Adjacent narrative elements are grouped by page and section, then split at sentence/word boundaries. Chunks are limited to 800 characters and 240 actual model tokens to avoid the embedding model's 256-token truncation. Each table is kept separate from narrative text and remains one chunk when it fits those limits; a larger table is split into row groups with its header and the preceding data row repeated in the next chunk to preserve context. Section headings are added as context. Chunks are embedded with the local `all-MiniLM-L6-v2` model and stored in the project's persistent Chroma collection:

```text
projects\<project_id>\chroma_data\
```

Chunk metadata includes the PDF name, SHA-256 content hash, page number when Docling provides it, chunk index/type, section path, ingestion version, and upload time. Source PDFs are retained under `source_pdfs`. Re-uploading an identical PDF with the current ingestion version is skipped; new PDFs are appended. Re-upload previously ingested PDFs after an ingestion-version change to rebuild their chunks with the updated extraction/chunking pipeline. OCR and accurate table extraction can take longer than plain-text extraction.

### 3. Start the Mock AUT

Enter the Azure credentials in the sidebar and use **Start Mock AUT** in **App Under Test Config**. The credentials are provided to the Mock AUT subprocess when it starts. On a question request, the Mock AUT embeds the question locally, retrieves the five closest chunks from that project's Chroma collection, and sends those chunks plus their page/section labels to the configured Azure OpenAI deployment. It requests a JSON answer, checks that the result is concise and not a copied section/table, and retries once if the output violates that contract. If it still fails validation, the workbook records an error instead of treating copied source text as an answer. The response metadata identifies the LLM, prompt version, and number of retrieved chunks.

The Mock AUT is a testing service, not a production-ready RAG application. Its answer quality depends on PDF extraction, chunking, retrieval, and the configured model.
After changing Python code, restart the FastAPI backend and Streamlit. If the backend reloads while the Mock AUT is running, its process check can still find the old AUT child. **Check AUT status** reports an outdated child; click **Start Mock AUT** to safely replace the workspace's old AUT process with the current code. Workbook errors are also shown as row-level details in the dashboard.

### 4. Evaluate a Q&A workbook

In **App Under Test Config**, upload the PDF Workbench Excel output after starting the Mock AUT, then choose **Run AUT and judge workbook**. Each non-empty question is sent to the selected project's Mock AUT. The workbook evaluator requires the AUT response to confirm an LLM-generated answer and at least one retrieved chunk before calling the separate judge. The sidebar-configured Azure deployment compares the actual answer with `expected_answer`. The judge accepts equivalent wording but scores directness, factual correctness, and completeness; a requested fact buried in a copied passage or table is not treated as a fully correct answer. It writes `actual_answer`, `status`, `reason`, and `score` into a completed copy of the workbook. Scores range from 0 to 1; `PASS` is 0.6 or higher, `FAIL` is lower, and `ERROR` indicates a failed or unverified AUT/judge call. The original workbook is unchanged.

### 5. Run checks and download reports

Open **Execution**, select the checks you want, and run them against the project's Chroma collection. Duplicate-similarity candidates have minimum and maximum cosine-similarity inputs, defaulting to the inclusive range 0.65–0.98; only pairs inside that range are shown. Similarity indicates semantic neighbors, not necessarily duplicate content. The UI and HTML report show cosine similarity, token overlap, metadata, and chunk text for review; this diagnostic is informational rather than an automatic duplicate verdict. Neighbor consistency samples up to 25 consecutive chunks from the same source and compatible section/page/content type, then reports each pair's neighbor ranks in both directions. Chunk text is shown only for pairs outside top-k. It uses top-10 by default, based on the observed rank distribution in the current 60-record collection. It is informational, not a pass/fail relevance score. For collections with at most 11 records, it is inconclusive because every non-self record fits in top-10. Vector integrity, duplicate similarity candidates, metadata completeness, and index health run without pair setup or a drift baseline. The report includes individual results and a summary and can be downloaded as HTML or JSON.

Index health checks every stored record with a vector by querying its nearest neighbors and verifying that it can find itself. Records without vectors are counted as failures. This performs one kNN query per vector-bearing record, so it is more complete than sampling but takes longer as the collection grows.

## API endpoints added for this workflow

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/projects/{project_id}/ingest-pdfs` | Docling extraction and persistent Chroma ingestion |
| `POST` | `/api/projects/{project_id}/run-vector-checks` | Run selected vector checks, automatically sampling adjacent chunks for neighbor consistency |
| `POST` | `/api/mock-aut/start` | Start the Mock AUT subprocess |
| `POST` | `/api/mock-aut/stop` | Stop the Mock AUT subprocess |
| `GET` | `/api/mock-aut/status` | Read Mock AUT process status |
| `POST` | `http://127.0.0.1:8001/api/query` | Retrieve chunks and generate a grounded answer |

## Notes

- The previous generated easy/medium/hard dataset workflow is not part of the three-tab dashboard flow.
- PDF Workbench Excel and characteristics JSON outputs are also saved under `streamlit_output\<session_id>\` in the workspace. Evaluated workbooks are saved in that same session folder with an `_evaluated.xlsx` suffix.
- Docling uses the same OCR-enabled converter configuration for PDF Workbench and project ingestion. OCR engine availability and document quality affect scanned-page recognition; image-only figures may still need a caption or separate visual interpretation to become searchable.
- Chroma is opened by the backend and Mock AUT at different times for reads/writes. Avoid running multiple backend processes against the same project store.
