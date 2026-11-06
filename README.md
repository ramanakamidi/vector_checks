# Vector Database Testing Framework

A database-agnostic implementation of a five-category vector-health framework.
Six connectors are provided (Elasticsearch, Pinecone, Weaviate, Milvus,
pgvector, Chroma) and every check is written once against a common
interface, so it runs unchanged against any of them.

## Architecture

This framework covers two market-tool equivalents:

| Market Tool | What We Implement | Coverage |
|---|---|---|
| **PyVectorHound** | Connector-agnostic framework with 6 connectors, 5 active check categories, embedding quality metrics, and visualization | Full |

## Project Structure

```
app.py                    — legacy Streamlit dashboard entry point
server.py                 — FastAPI backend entry point
mock_aut.py               — Mock AUT entry point

clients/
  api_client.py           — typed dashboard API client

pdf_workbench/
  custom_logic.py         — PDF extraction, visual analysis, Q&A, and judge logic

services/
  data_generation.py      — legacy project dataset generation
  execution.py            — legacy execution service
  ingestion.py            — JSON and vector-store ingestion
  local_embeddings.py     — local embedding and tokenization
  pdf_ingestion.py        — PDF-to-Chroma ingestion
  project_chroma.py       — shared project collection lookup and Chroma access
  project_manager.py      — project lifecycle and settings
  project_visualization.py — project Chroma UMAP projection and HDBSCAN clustering
  project_vector_checks.py — selectable checks for project Chroma stores
  report_generator.py     — execution report generation
  workbook_evaluation.py  — AUT answer and judge workbook workflow
  answer_improvement.py  — Chroma-grounded expected-answer workbook review
  react_workbench.py     — session-backed PDF Workbench and answer-review API logic
  paths.py                — shared project-data paths

connectors/
  __init__.py
  base.py                  — shared interface (VectorDBConnector ABC)
  elastic_connector.py     — Elasticsearch connector
  pinecone_connector.py    — Pinecone connector
  weaviate_connector.py    — Weaviate connector
  milvus_connector.py      — Milvus connector
  pgvector_connector.py    — pgvector (PostgreSQL) connector
  chroma_connector.py      — Chroma connector

checks/
  __init__.py
  vector_integrity.py      — Category 1: dimension consistency
  neighbor_consistency.py  — Category 2: automatic adjacent-chunk consistency
  duplicate_detection.py   — Category 3: cosine-neighbor candidates with text-overlap evidence
  metadata_completeness.py — Category 4: required field presence, value distribution
  index_health.py          — Category 5: full-collection searchability

visualize.py               — PCA 2D projection plots of the embedding space
run_checks.py              — orchestrator: runs all checks, prints/exports report
manual_chroma_check.py    — read-only inspection and question retrieval for project Chroma data
```

Generated project data stays under `projects/`; the standalone React UI remains
under `ui/`. Keep these runtime/data directories separate from application
source when cleaning or reorganizing the repository.

## Run the React dashboard

The React/Vite dashboard is the primary UI. Start the API and frontend in two
PowerShell terminals from the project root:

```powershell
# Terminal 1 — Python API
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r streamlit_requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2 — React/Vite UI
Set-Location .\ui
npm install
npm run dev
```

Open the Vite URL printed in the second terminal (normally
`http://localhost:5173`). Vite proxies `/api` requests to the local FastAPI
server. Project setup, ingestion, dataset generation, execution, PDF
Workbench, answer improvement, and Visual Check use the existing project
services. Azure OpenAI credentials entered in the PDF Workbench or Answer
Improvement remain in browser memory and are sent only with the requested
operation. PDF Workbench session state is held by the API process, so restarting
the API expires the active workbench session.

Build the frontend for a production bundle with:

```powershell
Set-Location .\ui
npm run build
```

## What's Covered (Five Categories)

| # | Category | Checks | Auto Pass/Fail |
|---|---|---|---|
| 1 | Vector Integrity | Dimension consistency | ✅ Yes |
| 2 | Neighbor Consistency | Informational top-k diagnostic for sampled adjacent-chunk pairs | ❌ No; adjacent chunks are heuristic candidates |
| 3 | Similarity Candidates | kNN neighbors above a cosine threshold, with text overlap for review | ℹ️ Informational |
| 4 | Metadata Completeness | Required field presence, value distribution | ✅ Yes |
| 5 | Index Health | Searchability check for every stored record | ✅ Yes |

Duplicate-similarity candidates use a configurable inclusive cosine-similarity
range from **0.65 to 0.98** by default. Only pairs whose similarity falls
within both entered bounds are reported. This finds semantic neighbors, not
necessarily duplicates. The check is informational and shows cosine similarity,
word-token overlap, metadata, and chunk text; review the evidence before
classifying any pair as duplicate. The HTML report shows each candidate in its
own card.

The searchability check performs one nearest-neighbor query for every
record with a vector. It reports the number of records checked, vectors
queried, and records that could not be queried or did not find themselves.
This is more thorough than sampling but takes longer as the collection grows.

Neighbor consistency uses top-10 by default and samples up to 25 consecutive
chunk pairs from the same source and compatible section/page/content type. It
reports every sampled pair's rank in both directions and shows chunk text only
for pairs outside top-k. In the current 60-record collection, 21 of 25
pairs were mutual at k=10, compared with 24 of 25 at k=20; k=10 was chosen as a
stricter cutoff while still including most observed pairs. This is a small
heuristic sample, not a universal optimum. Collections with at most `k + 1`
records are marked inconclusive because every non-self record fits in top-k.
This is an informational diagnostic, not a relevance-quality score: adjacency
and overlap are not ground-truth labels.

## Manually Inspect a Project's Chroma Collection

Edit `PROJECT_ID`, `K`, and `QUESTION` near the top of the script, then run it:

```bash
python manual_chroma_check.py
```

The script does not require command-line arguments or prompt for input. Set
`QUESTION` to an empty string to skip question retrieval. It prints
collection/vector counts, vector dimensions, document and metadata coverage,
then each retrieved chunk's rank, ID, Chroma distance, metadata, and text. It
uses the project's local embedding model and does not change collection data.

## Generate Q&A in PDF Workbench

The **PDF Workbench** provides three Q&A actions: **Generate Q&A for objects**,
**Generate Q&A for objects and text**, and **Generate Q&A for text only**.
Object actions use the selected analyzed tables, charts, or images and their
crops; text actions use all prose chunks extracted from Docling Markdown during
the same PDF conversion. Before analysis, choose a text chunking technique:
**Heading-aware** (the existing behavior), **Recursive**, **Sentence**,
**Token**, or **Semantic**. Recursive, Sentence, and Token use the project's
`all-MiniLM-L6-v2` tokenizer. Semantic uses Chonkie's tokenizer for its
`minishlab/potion-base-32M` model, which is downloaded on first use. Settings
vary by technique: chunk size, overlap where supported, minimum sentences for
SentenceChunker, and similarity threshold for SemanticChunker.
SentenceChunker's sentence setting is a minimum, with chunk size as the upper
bound.

Text chunks follow document headings and exclude Markdown tables and images.
Heading-aware mode preserves nested heading paths, paragraphs, and list items;
Chonkie modes process each heading section independently and retain its path as
chunk context. Chunks are assigned the page number from Docling's page-specific
Markdown export. Text Q&A workbook rows use `object_type` = `text` and record
the selected technique in `analysis_mode`. After extraction, apply a different
technique to the stored Markdown without re-analyzing the PDFs. The preview
shows sample chunks before you generate Q&A. Text Q&A categories describe the
question intent (specific fact, definition, comparison, process step, or cause
and effect) rather than generic text retrieval. Choose 1–3 questions per text
chunk. Text questions and object questions appear together in the generated
workbook and are both included when sending Q&A to a project; duplicate
questions within a PDF prefer the existing object-generated row. Internal
source metadata is stored in hidden workbook columns for chunk linking while
keeping the normal spreadsheet view focused on the Q&A.

## Review Expected Answers

The React **Answer Improvement** page accepts the generated Q&A
`.xlsx` workbook. PDF Workbench includes each Q&A row's object type and
workspace-relative cropped image paths in the workbook. Graph and image rows
are reviewed using their crop(s) only; their text is not retrieved from Chroma.
Text and table rows use Chroma records only. The tab's `k` is the initial
retrieval count; when evidence is insufficient, retrieval expands by doubling
(for example, `2 → 4 → 8`) and stops once evidence is sufficient, the collection
is exhausted, or 8 chunks have been checked. The question and expected answer
are both used to form the retrieval query.

The model checks whether the expected answer covers every explicitly requested
part of the question. The workbook adds **Improvement Required?**,
**Improvement Reason**, and **Improved Expected Answer** columns. The status is
**Yes** only when retrieved evidence supports a needed correction, **No** when
evidence confirms the answer is already complete, and **Unable to verify** when
the evidence cannot establish the answer even after retrieval reaches its
limit. In that case the reason explains that the missing information was not
found in the available records (or crop); the reviewer does not guess. Keep
cropped-object files in the workspace so their paths in the generated workbook
continue to resolve. Row-level processing failures are marked **ERROR** so
other rows can still finish.

## Visualize Project Vectors

The React **Visual Check** page reads a bounded sample of the selected project's
persistent Chroma collection, projects its vectors to 2D with UMAP, and labels
clusters with HDBSCAN on those same 2D coordinates, so cluster colors reflect
the displayed map. Use the cluster filter to constrain the map and Q&A cards,
then select a point to inspect chunk text and metadata. The default sample is 300
records, and the default minimum cluster size is 2; both values are
configurable. Records without valid,
consistently sized finite vectors are skipped. Install `streamlit_requirements.txt` to include UMAP and Plotly; the
HDBSCAN implementation is provided by scikit-learn. Install the app dependencies
with `pip install -r streamlit_requirements.txt`.
Upload a PDF Workbench Q&A workbook below the map to show matching questions
and expected answers on cards tagged with the same cluster label and color.
Text Q&A is matched against the selected chunk's source text; table Q&A is
matched by PDF and page, with table content used to disambiguate multiple
tables on the same page. Only text and table rows are shown. The workbook keeps
the linking fields hidden, so download a fresh workbook after updating the app
to enable text matching.

## Calibrate Similarity with Negative Queries

In the **Execution** tab, enable **Negative queries (off-topic)** and enter
unrelated questions, one per line. The check embeds the unique questions
together and performs a single top-1 Chroma query batch. It reports each
question's best cosine score and flags scores above the selected maximum.
For above-threshold queries, the result also shows the closest stored chunk
to help inspect whether that question is actually relevant. The default
negative-query maximum is `0.32`, based on a small calibration sample from the `tech_project`
collection. Recalibrate if you select a different project or change its
documents. The duplicate detection range is a separate check and retains its
own `0.65`–`0.98` defaults.

## Synthetic Question Retrieval Check

In the **Execution** tab, enable **Synthetic question retrieval**, select
**Show records**, and choose a source chunk from all Chroma records in the
searchable selector. Select **Generate questions** to create and display the
requested questions in the UI before running checks. If you change the selected
chunk or question count, generate a fresh set before running.
The check uses the configured Azure OpenAI deployment to generate 1–3 questions
from that chunk, embeds the questions with the project's local embedding model,
and searches Chroma with top-k defaulting to `1`. Each question passes only if
the selected source record ID
appears in the top-k results. A check passes only when every generated question
retrieves that source chunk. Question generation uses the Azure credentials
already configured in the Streamlit session; only generated questions and the
selected record ID are sent to the vector-check API. The generated questions,
rank, and retrieved IDs are included in the results and HTML report.

## Installing Dependencies

Only install the client library for the database you're actually using:

```bash
pip install numpy scipy scikit-learn matplotlib   # needed by all checks
pip install elasticsearch                          # if using Elasticsearch
pip install pinecone-client                        # if using Pinecone
pip install weaviate-client                         # if using Weaviate
pip install pymilvus                                # if using Milvus
pip install psycopg2-binary                         # if using pgvector
pip install chromadb                                # if using Chroma
```

## Quick Start (Elasticsearch Example)

```python
from connectors.elastic_connector import ElasticConnector
from run_checks import safe_connect, run_full_report, print_report

connector = ElasticConnector(
    hosts=["https://your-cluster:9200"],
    index="orion-chunks",
    vector_field="embedding",
    api_key="...",
)

# Safe connection with clean error handling
success, error = safe_connect(connector)
if not success:
    print(f"Connection failed: {error}")
    exit(1)

# Run all checks
report = run_full_report(
    connector,
    required_metadata_fields=["doc_id", "artifact_type"],
)
print_report(report)

# Optionally export as JSON
from run_checks import export_report_json
export_report_json(report, "my_report.json")
```

## Generating Visualizations

```python
from visualize import plot_embedding_space

# Plot the current embedding space
plot_embedding_space(connector, sample_size=2000, output_path="embeddings.png",
                     color_by_field="artifact_type")

```

## What You Still Need to Do

1. **Confirm field names match your real index.** `vector_field`, `id_field`,
   metadata field names are configurable — check your actual schema.
2. **Run a small collection first** — full-collection searchability performs
   one nearest-neighbor query for each record, so runtime grows with collection
   size.
3. **Deliberately break a small test copy** before trusting any check on
   production data: insert a known duplicate or delete a required metadata
   field, then confirm the corresponding check flags it.
4. **Check the neighbor-consistency methodology** — it uses adjacent chunks as
   likely-similar examples and is inconclusive if the needed metadata is absent.
5. **Wire up scheduling/alerting** if you want these checks to run automatically
   (cron, Airflow, etc.) and send results somewhere (Slack, email, dashboard).

## What's NOT Covered (By Design)

| Original Testing Level | What It Covers | Why It's Out of Scope |
|---|---|---|
| Level 1 — Reconciliation | Source-to-DB record counts | Requires pipeline-aware code, not just vector DB checks |
| Level 3 — Chunk Self-Containedness | Semantic completeness of chunks | Requires an LLM judge |
| Level 4 — Golden Corpus Regression | Known Q&A pairs return correct answers | Requires a retrieval + generation evaluation harness |

These are valid testing levels but require fundamentally different tooling
(LLM judges, pipeline hooks, evaluation harnesses) that is outside the scope
of a vector-database-focused testing framework.
