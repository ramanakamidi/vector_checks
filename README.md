# Vector Quality Studio

Vector Quality Studio is a local workspace for ingesting PDF content, building
and evaluating retrieval-augmented generation (RAG) test data, and checking
vector-store health. The interface is a React/Vite dashboard backed by a
FastAPI service, with standalone database connectors also available.

The application can:

- Detect figure images in PDFs with Docling and embed them as image-only vector records.
- Choose Chroma, FAISS, or Milvus Lite when creating a project; ingestion, Mock
  AUT retrieval, vector checks, and visualization use that project's selected
  store.
- Generate text- and image-grounded questions and Excel workbooks.
- Run a local Mock AUT that retrieves project context and uses Azure OpenAI to
  produce grounded answers.
- Evaluate generated workbooks against the Mock AUT and an LLM judge.
- Review expected answers against retrieved project evidence.
- Run vector integrity, metadata, similarity, neighbor, searchability, and
  selected retrieval-quality checks.
- Visualize project embeddings with UMAP and cluster them using HDBSCAN.

> **Local development application:** credentials, API access, PDF processing,
> and sample data are designed for a developer workstation. Do not expose the
> development API or Vite server to an untrusted network.

## Contents

- [Requirements](#requirements)
- [Install and start](#install-and-start)
- [First project workflow](#first-project-workflow)
- [Dashboard pages](#dashboard-pages)
- [Vector checks](#vector-checks)
- [PDF Workbench chunking](#pdf-workbench-chunking)
- [Standalone connector usage](#standalone-connector-usage)
- [Project structure and data](#project-structure-and-data)
- [API overview](#api-overview)
- [Tests and production build](#tests-and-production-build)
- [Troubleshooting](#troubleshooting)
- [Project history](#project-history)

## Requirements

- Windows PowerShell instructions are provided below; Python, Node.js, and npm
  must be installed.
- Python dependencies are listed in `requirements.txt`, including backend,
  PDF, embedding, and visualization packages.
- The React UI dependencies are listed in `ui/package.json`.
- Azure OpenAI credentials are only needed for features that call the model,
  including PDF object analysis, Q&A generation, answer review, Mock AUT, and
  LLM judging.

## Install and start

Run the API and React dashboard in separate PowerShell terminals. Run commands
from the repository root unless a command explicitly changes to `ui`.

### 1. Create the Python environment

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Docling and sentence-transformer models may download on first installation or
first use. PDF processing, OCR, embedding-model initialization, and semantic
chunking can take additional time on first run.

If PowerShell blocks environment activation, either allow it for the current
terminal or invoke the environment's Python executable directly:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 2. Start the FastAPI backend

In the first terminal, from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive API
documentation is at `http://127.0.0.1:8000/docs`. No Azure credentials are
needed just to start the API.

### 3. Install and start the React/Vite dashboard

In a second terminal:

```powershell
Set-Location .\ui
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. Vite
proxies `/api` requests to the FastAPI server on port 8000.

Stop either development server with **Ctrl+C** in its terminal.

## First project workflow

### 1. Configure the app under test

Create a project and choose **Chroma**, **FAISS (local)**, or **Milvus Lite
(local file)** in the project dialog, or load an existing project from the
header. Then open **Configure**.
Its **LLM Config** tab contains Azure OpenAI credentials, and
its **AUT Config** tab contains PDF ingestion and Mock AUT controls.

PDF ingestion uses Docling to identify picture/figure crops and local
`Qwen/Qwen3-VL-Embedding-2B` image embeddings. Only detected figure images are
embedded and stored as vector records; extracted PDF text, tables, and captions
are not embedded or added to the vector store:

```text
projects\<project_id>\chroma_data\   (Chroma)
projects\<project_id>\faiss_data\    (FAISS)
projects\<project_id>\milvus_lite.db\ (Milvus Lite local database directory)
```

Milvus Lite stores project vectors under a local `.db` database path and uses the same
Qwen3-VL 2048-dimensional image embeddings as the Chroma and FAISS project backends.
Figure image files are retained under `projects\<project_id>\objects\`; vector records
contain image embeddings and metadata including each local image path. Search queries
use Qwen's query encoding path to retrieve detected figure images.
The multimodal adapter follows the official
[Qwen3-VL-Embedding repository](https://github.com/QwenLM/Qwen3-VL-Embedding).
Qwen inference defaults to CPU. CPU ingestion processes and stores one record at
a time so progress advances per completed record; GPU ingestion batches four
records at a time. The 2B model can be very slow on CPU; for faster ingestion,
set `PROJECT_EMBEDDING_DEVICE=cuda`, install a CUDA-enabled PyTorch build that
matches the host GPU, and verify `torch.cuda.is_available()` is `True` before
starting the API.

Only figures Docling detects as picture items are ingested; PDFs without detected
figures produce no vector records. Native vector diagrams that Docling does not
identify as picture items are not included. Re-ingesting a PDF with the current
pipeline replaces records created by older ingestion versions. Metadata includes
PDF name, SHA-256 content hash, page number, image path, ingestion version, and
upload time. Source PDFs are retained in the project's `source_pdfs` directory.
Identical PDFs already ingested with the current pipeline are skipped; new PDFs
are appended. Start the Mock AUT from this page; it retrieves from the active
project's selected vector collection and uses the configured Azure OpenAI
deployment. Workbook evaluation against expected answers is available from
**Execution**.

PDF Workbench is a separate, interactive flow for analyzing detected PDF
objects and prose, generating Q&A, and exporting a workbook. See
[PDF Workbench chunking](#pdf-workbench-chunking).

### 2. Run project checks

Open **Execution** to choose vector-quality checks for the active project's
selected vector collection. Configure metadata requirements, similarity
thresholds, or off-topic questions as needed. Upload a
Test Data Generation `.xlsx` workbook at the top to run question-based Neighbor
Consistency, Duplicate Detection, AUT-Vector Drift, and per-question Vector
Integrity or Metadata Completeness. The latter two inspect the top three
retrieved records for each question. Neighbor Consistency, Duplicate Detection,
and Drift use the configured top-k. Hit@k evaluates each workbook question
against the top 50 retrieved text, table, image, and graph records; the workbook
must include `object_type`, `chunk`, and `object_image_paths` columns. It reports
the matching rank and Hit@1/3/5/10 for each question without averages. Drift
processes each question separately:
the vector database retrieves top-k records, the Mock AUT uses those exact
records to generate its actual answer, and the Azure OpenAI judge evaluates
that answer against the question and retrieved evidence. Per-question actual
answers, records, scores, differences, and evidence-based drift traces appear
in the results and downloadable workbook and HTML report. Select a question
row to open its details card with the workbook's expected answer, the actual AUT
answer when available, and PDF/page source information from the workbook or
retrieved records.

### 3. Work with PDFs and answers

Use **Answer Improvement** to review expected answers in a Q&A `.xlsx`
workbook against project evidence and download the annotated workbook.

### 4. Explore the embedding space

Use **Visual Check** to compute a bounded UMAP projection of project vectors
and label the displayed 2D coordinates with HDBSCAN. Select points to inspect
their stored text and metadata. Upload a PDF Workbench workbook to link
matching text/table Q&A cards to their chunk's point and cluster color.

## Dashboard workflows

| Page | Purpose |
|---|---|
| **Configure** | Set Azure OpenAI credentials under LLM Config; ingest PDFs and manage the Mock AUT under AUT Config. |
| **Execution** | Run vector checks, evaluate Q&A workbooks, and compare AUT/vector answers for drift. |
| **Test Data Generation** | Analyze PDFs, choose chunking and vision context, select detected objects inline, generate Q&A, and export an Excel workbook. |
| **Answer Improvement** | Compare expected answers against retrieved project chunks or workbook-referenced visual crops. |
| **Visual Check** | Explore project UMAP/HDBSCAN clusters and optionally link workbook Q&A to text/table points. |

Azure credentials entered in the dashboard are held in browser memory and
submitted to the local API for the requested operation. They are not stored in
browser local storage. Test Data Generation results remain available while
navigating between dashboard tabs and reset on a full page refresh. The
corresponding API session and generated files are stored under
`ui/workbench_output/react_sessions/`; restarting the backend does not clear
those files.

## Vector checks

The project-based **Execution** workflow supports these check categories:

| Check | What it reports |
|---|---|
| Vector integrity | Without a workbook, checks that scanned vectors have a consistent dimension. With a workbook, checks dimensions and basic vector validity (empty, non-finite, or zero-magnitude vectors) for the top three retrieved records per question. |
| Neighbor consistency | Per workbook question, retrieves a configurable top-k, checks every distinct pair among the retrieved records, and reports pair-level reciprocal-neighbor results. The direct project-check API can use automatically sampled adjacent pairs without workbook questions. |
| Hit@k (per question) | Embeds each workbook question using the project's query embedding model, retrieves the top 50 mixed text, table, image, and graph records, and reports the exact-match rank and Hit@1/3/5/10. Requires `question`, `object_type`, `chunk`, and `object_image_paths` workbook columns; reports no averages. |
| AUT-Vector Drift | For each workbook question, retrieves top-k vector records, gets the Mock AUT's actual answer from those records, and has an Azure OpenAI judge score the answer against the question and evidence; reports actual answers, retrieved records, a 0–1 drift score, differences, and evidence-based trace. |
| Duplicate detection | With a workbook, compares each question's retrieved records and reports per-question similarity candidates within a configurable cosine range (default 0.65–0.98). Without workbook questions, the direct project-check API samples the collection. Candidates are not automatically labeled duplicates. |
| Metadata completeness | Without a workbook, checks selected required fields across the collection. With a workbook, checks each question's top three retrieved records for missing or empty required fields and shows record-level results. |
| Negative-query similarity | Similarity of supplied off-topic questions to the project collection. |

Neighbor consistency is a heuristic diagnostic, not a relevance score or
pass/fail ground truth. Collections too small to provide a meaningful
comparison can be reported as inconclusive. Duplicate similarity candidates
are for review: high semantic similarity does not prove duplicate content.

The standalone connector framework also provides connector-agnostic vector
integrity, duplicate, metadata, index-health, and adjacent-chunk diagnostics.
See [Standalone connector usage](#standalone-connector-usage).

## PDF Workbench chunking

PDF Workbench uses Docling-extracted Markdown and offers four prose chunking
techniques:

| Technique | Controls |
|---|---|
| Recursive | Maximum token size. In PDF Workbench, Chonkie recursive splitting is performed page-wise while heading paths remain available as context. |
| Sentence | Maximum token size, token overlap, and minimum sentences per chunk. The sentence minimum is best-effort when the input or token limit cannot fit that many sentences. |
| Token | Maximum token size and token overlap. |
| Semantic | Similarity threshold. Uses Chonkie's `minishlab/potion-base-32M` embedding model for chunk segmentation; first use may download the model. |

The Chonkie token-based modes use this project's Qwen3-VL tokenizer;
semantic mode uses its corresponding Chonkie tokenizer. In PDF Workbench,
prose is chunked page-wise so section boundaries do not prevent the selected
chunker from grouping nearby text; heading paths touched by each chunk are
retained as metadata and included separately when generating Q&A. Markdown
tables and images are excluded from prose chunks because detected visual
objects are handled separately. A figure caption remains with page prose but
is not emitted as a caption-only prose chunk. Docling's page-specific Markdown
is used to assign page metadata.

After analysis, apply changed chunk settings to re-chunk the stored text
without re-analyzing the PDFs. Applying different settings clears existing
text Q&A because those questions refer to the prior chunks.

Object Q&A uses selected tables, charts, and images and their crops. Text Q&A
uses the extracted prose chunks. The exported workbook combines generated
questions, keeps `chunk` as the final source-text column, and stores linking
metadata in hidden columns. Text rows use `object_type = text` and include
page and chunk-technique metadata.

## Standalone connector usage

The connector interface supports Elasticsearch, Pinecone, Weaviate, Milvus,
PostgreSQL/pgvector, Chroma, and FAISS. Install the database client library only when
you use that connector:

```powershell
python -m pip install elasticsearch
python -m pip install pinecone
python -m pip install weaviate-client
python -m pip install pymilvus
python -m pip install milvus-lite
python -m pip install psycopg2-binary
python -m pip install chromadb
python -m pip install faiss-cpu
```

Create a connector for an existing index/collection and call the shared check
runner. For example, with Elasticsearch:

```python
from connectors.elastic_connector import ElasticConnector
from run_checks import run_full_report, print_report, safe_connect

connector = ElasticConnector(
    hosts=["https://your-cluster:9200"],
    index="your-index",
    vector_field="embedding",
    api_key="read-only-api-key",
)

connected, error = safe_connect(connector)
if not connected:
    raise RuntimeError(error)

report = run_full_report(
    connector,
    required_metadata_fields=["doc_id", "artifact_type"],
)
print_report(report)
```

The connector must point at an existing collection/index with compatible
vector fields and query support. Credentials should be provided locally (for
example, from environment variables or a local secret manager); do not commit
them into source or documentation.

To inspect a project's persistent Chroma records and optionally run a
question-retrieval probe, edit `PROJECT_ID`, `K`, and `QUESTION` near the top
of `manual_chroma_check.py`, then run:

```powershell
python .\manual_chroma_check.py
```

Set `QUESTION` to an empty string to skip question retrieval. This script
inspects the selected project collection and does not change it.

## Project structure and data

```text
requirements.txt               Python API, PDF, embedding, and visualization dependencies
server.py                      FastAPI API and Mock AUT process manager
mock_aut.py                    Mock question-answering API
run_checks.py                  Standalone connector check runner
visualize.py                   Standalone vector-space visualization
manual_chroma_check.py         Read-only project Chroma inspection

ui/                            React, Vite, TypeScript, Tailwind, Recharts UI
clients/                       Python dashboard API client
connectors/                    Shared vector-store interface and six adapters
checks/                        Vector check implementations and tests
pdf_workbench/                 PDF extraction, chunking, Q&A, workbook judge
services/                      Project, ingestion, execution, visualization,
                               workbook, and React Workbench services
report/                        HTML report components

projects/                      Project settings, vector stores, datasets,
                               source PDFs, and execution history
ui/workbench_output/            PDF Workbench uploads, crops, and session artifacts
output/                        Standalone reports and generated output
```

Treat `projects/`, `ui/workbench_output/`, and `output/` as runtime data, not
source folders. Back up project data before cleaning or moving the repository.
Avoid running multiple backend processes against the same Chroma project store.

## API overview

The FastAPI API exposes interactive documentation at `/docs`. Main route
groups include:

| Route group | Purpose |
|---|---|
| `/api/health` | Backend health. |
| `/api/projects` | Project CRUD and project settings. |
| `/api/projects/{project_id}/ingest*` | JSON, file, and PDF ingestion. |
| `/api/projects/{project_id}/datasets*` | Dataset generation, browsing, and deletion. |
| `/api/projects/{project_id}/execute*` | Run, inspect, delete executions, and download HTML reports. |
| `/api/projects/{project_id}/run-vector-checks` | Run selected checks against the project's configured vector store. |
| `/api/projects/{project_id}/run-workbook-checks` | Run workbook-question checks, including per-question Neighbor Consistency, Duplicate Detection, Hit@k, and optional AUT-Vector Drift; returns an annotated workbook when Drift is selected. |
| `/api/projects/{project_id}/visualization-data` | Project UMAP/HDBSCAN data; the `with-qa` route accepts a workbook and links Q&A. |
| `/api/workbench/*` | Analyze PDFs, apply chunking, generate Q&A, fetch crops, and export workbooks. |
| `/api/answer-improvement/review` | Review a workbook against project evidence. |
| `/api/mock-aut/*` | Start, stop, and inspect the Mock AUT. |

The Mock AUT query endpoint normally listens at
`http://127.0.0.1:8001/api/query`. See `/docs` for request fields and response
schemas.

## Tests and production build

Run the focused Python regression tests from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest checks.test_pdf_ingestion_chunking checks.test_pdf_workbench_chunking checks.test_project_visualization checks.test_react_workbench
```

Run all discovered tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s checks
```

Build and type-check the frontend:

```powershell
Set-Location .\ui
npm run build
```

The build outputs static files under `ui/dist/`. The development API and
frontend are intended for local use; deploy behind an appropriately secured
server if production hosting is required.

## Troubleshooting

### The UI reports that the API is unavailable

Confirm the FastAPI terminal is running at `127.0.0.1:8000` and visit
`http://127.0.0.1:8000/api/health`. Keep the Vite UI at its normal local
address so its `/api` proxy can reach the backend.

### PDF Workbench session expired

Workbench session data is in API memory. If the API restarted, upload the PDFs
and analyze them again. The credentials must be entered again if the browser
page was reloaded.

### PDF analysis or Q&A fails

Check the Azure endpoint, deployment, API version, and key in the relevant
page. Confirm that Docling finished model installation and that the PDFs are
valid. Semantic chunking may take longer on first use while its embedding
model is downloaded. The API returns request errors in the page; inspect the
backend terminal for detailed processing errors.

### The Mock AUT is outdated or a port is already in use

Check AUT status in **Configure** → **AUT Config**. Stop and start the managed Mock AUT again
after changing its code or credentials. If port 8001 belongs to another
service, stop that service or configure a different port.

### Visual Check has no points or fails to build

The selected project needs a Chroma collection with at least three valid,
consistently sized finite vectors. UMAP dependencies are installed through `requirements.txt`. Larger samples
and collections take longer to project and cluster.

## Project history

- **2026-10-27 to 2026-10-30:** Added the FastAPI project APIs, PDF ingestion
  and Mock AUT, vector checks, workbook evaluation, and service/package
  organization. Strengthened prompt handling and AUT response validation.
- **2026-11:** Added selectable Chonkie PDF text chunkers, chunk/source
  metadata in Excel, inline object selection, expected-answer review,
  workbook-to-UMAP Q&A links, and HDBSCAN clustering on displayed UMAP
  coordinates.
- **2026-11:** Migrated the dashboard to React/Vite and added API-backed PDF
  Workbench, Answer Improvement, and Visual Check pages.

This README is the project's consolidated setup and usage guide. Older
standalone Markdown guides and audits have been merged or removed; consult the
current implementation and tests when historical notes disagree with current
behavior.
