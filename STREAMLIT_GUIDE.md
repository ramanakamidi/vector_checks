# Streamlit dashboard

The Streamlit app connects the supplied PDF object-analysis pipeline to the FastAPI backend. It includes PDF upload and analysis, natural-language filtering, Q&A generation, Excel/JSON downloads, project management, ingestion, dataset generation, executions, vector checks, and an endpoint explorer for every registered backend operation.

## Install

From the repository root, activate the project virtual environment and install the dashboard dependencies:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
python -m pip install -r streamlit_requirements.txt
```

Docling installs its own PDF-processing models and dependencies, so its first installation or first PDF conversion can take a while.

## Run

Keep the FastAPI backend running in one terminal:

```powershell
python -m uvicorn server:app --reload --port 8000
```

Start the dashboard in another terminal from the repository root:

```powershell
python streamlit_launcher.py --server.port 8501
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. The sidebar backend URL defaults to `http://127.0.0.1:8000`.

## PDF pipeline

Enter the Azure OpenAI endpoint, deployment, API version, and key in the sidebar. Upload one or more PDFs, choose crop/page/page-context analysis, and run the pipeline. Filter detected objects, select objects for grounded Q&A, then download the Excel validation workbook or `characteristics.json`.

Pipeline artifacts are written beneath `streamlit_output/<session-id>/`. Generated Q&A can be sent to the selected backend project using `POST /api/projects/{project_id}/ingest-json`.

The backend's project and vector-quality workflows are available in their respective tabs. The API Explorer lists and can call all 40 operations registered in `server.py`; dynamic path values, query JSON, request JSON, and multipart uploads can be entered there.

The Azure key is used by the local Streamlit process to call Azure OpenAI. It is not included in requests to the vector-quality backend.