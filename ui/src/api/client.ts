import type {
  Project,
  ProjectSettings,
  ConnectRequest,
  ConnectResponse,
  RunChecksResponse,
  ResultsResponse,
  EmbeddingPoint,
  RecordDetail,
  IngestResponse,
  PaginatedRecords,
  IngestionStats,
  DatasetConfig,
  GeneratedDataset,
  DatasetSummary,
  ExecutionConfig,
  ExecutionResult,
  ExecutionListItem,
  MockAutStatus,
  PaginatedVectorRecords,
} from './types';
import type { AzureCredentials, ChunkingOptions } from './workbench';

const BASE = '/api';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── Project APIs ─────────────────────────────────────────────────────

export async function listProjects(): Promise<{ projects: Project[] }> {
  return request<{ projects: Project[] }>('/projects');
}

export async function createProject(
  name: string,
  description = '',
  dbType: 'chroma' | 'faiss' | 'milvus_lite' = 'chroma',
): Promise<Project> {
  return request<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify({ name, description, db_type: dbType }),
  });
}

export async function getProject(projectId: string): Promise<any> {
  return request<any>(`/projects/${encodeURIComponent(projectId)}`);
}

export async function updateProject(
  projectId: string,
  data: { name?: string; description?: string },
): Promise<any> {
  return request<any>(`/projects/${encodeURIComponent(projectId)}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteProject(projectId: string): Promise<any> {
  return request<any>(`/projects/${encodeURIComponent(projectId)}`, {
    method: 'DELETE',
  });
}

// ── Settings APIs ────────────────────────────────────────────────────

export async function getProjectSettings(projectId: string): Promise<ProjectSettings> {
  return request<ProjectSettings>(`/projects/${encodeURIComponent(projectId)}/settings`);
}

export async function saveProjectSettings(
  projectId: string,
  settings: Partial<ProjectSettings>,
): Promise<any> {
  return request<any>(`/projects/${encodeURIComponent(projectId)}/settings`, {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

// ── Ingestion APIs (Stage 2) ─────────────────────────────────────────

export async function ingestData(
  projectId: string,
  data: any[],
): Promise<IngestResponse> {
  return request<IngestResponse>(`/projects/${encodeURIComponent(projectId)}/ingest-json`, {
    method: 'POST',
    body: JSON.stringify({ source: 'upload', data }),
  });
}

export async function uploadJsonFile(
  projectId: string,
  file: File,
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}/ingest`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<IngestResponse>;
}

export async function ingestFromFile(
  projectId: string,
  filePath: string,
): Promise<IngestResponse> {
  return request<IngestResponse>(`/projects/${encodeURIComponent(projectId)}/ingest-json`, {
    method: 'POST',
    body: JSON.stringify({ source: 'file', file_path: filePath }),
  });
}

export async function getIngestedData(
  projectId: string,
  page = 1,
  pageSize = 50,
  search = '',
): Promise<PaginatedRecords> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
    search,
  });
  return request<PaginatedRecords>(
    `/projects/${encodeURIComponent(projectId)}/ingested-data?${params}`,
  );
}

export async function deleteIngestedData(projectId: string): Promise<{ status: string; message: string }> {
  return request<{ status: string; message: string }>(
    `/projects/${encodeURIComponent(projectId)}/ingested-data`,
    { method: 'DELETE' },
  );
}

export async function getIngestionStats(projectId: string): Promise<IngestionStats> {
  return request<IngestionStats>(
    `/projects/${encodeURIComponent(projectId)}/ingestion-stats`,
  );
}

// ── Data Generation APIs (Stage 3) ───────────────────────────────────

export async function generateDataset(
  projectId: string,
  config: DatasetConfig,
): Promise<GeneratedDataset> {
  return request<GeneratedDataset>(`/projects/${encodeURIComponent(projectId)}/generate-dataset`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function listDatasets(
  projectId: string,
): Promise<{ datasets: DatasetSummary[] }> {
  return request<{ datasets: DatasetSummary[] }>(
    `/projects/${encodeURIComponent(projectId)}/datasets`,
  );
}

export async function getDataset(
  projectId: string,
  datasetId: string,
  page = 1,
  pageSize = 50,
): Promise<GeneratedDataset> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  return request<GeneratedDataset>(
    `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}?${params}`,
  );
}

export async function deleteDataset(
  projectId: string,
  datasetId: string,
): Promise<{ status: string; dataset_id: string }> {
  return request<{ status: string; dataset_id: string }>(
    `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}`,
    { method: 'DELETE' },
  );
}

export async function getMetadataFields(
  projectId: string,
): Promise<{ fields: string[] }> {
  return request<{ fields: string[] }>(
    `/projects/${encodeURIComponent(projectId)}/metadata-fields`,
  );
}

// ── Execution APIs (Stage 4 — Execute) ───────────────────────────────

export async function startExecution(
  projectId: string,
  config: ExecutionConfig,
): Promise<ExecutionResult> {
  return request<ExecutionResult>(`/projects/${encodeURIComponent(projectId)}/execute`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function listExecutions(
  projectId: string,
): Promise<{ executions: ExecutionListItem[] }> {
  return request<{ executions: ExecutionListItem[] }>(
    `/projects/${encodeURIComponent(projectId)}/executions`,
  );
}

export async function getExecution(
  projectId: string,
  executionId: string,
): Promise<ExecutionResult> {
  return request<ExecutionResult>(
    `/projects/${encodeURIComponent(projectId)}/executions/${encodeURIComponent(executionId)}`,
  );
}

export async function deleteExecution(
  projectId: string,
  executionId: string,
): Promise<{ status: string; execution_id: string }> {
  return request<{ status: string; execution_id: string }>(
    `/projects/${encodeURIComponent(projectId)}/executions/${encodeURIComponent(executionId)}`,
    { method: 'DELETE' },
  );
}

export async function downloadReport(
  projectId: string,
  executionId: string,
): Promise<Blob> {
  const url = `${BASE}/projects/${encodeURIComponent(projectId)}/executions/${encodeURIComponent(executionId)}/report`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.blob();
}

export function getExecutionReportUrl(projectId: string, executionId: string): string {
  return `${BASE}/projects/${encodeURIComponent(projectId)}/executions/${encodeURIComponent(executionId)}/report`;
}

// ── Mock AUT APIs ────────────────────────────────────────────────────

export async function startMockAut(
  projectId: string,
  credentials: AzureCredentials,
  port = 8001,
): Promise<{
  status: string;
  url: string;
  pid: number;
  project_id: string | null;
  prompt_version: string;
  restarted_stale_process?: boolean;
}> {
  return request('/mock-aut/start', {
    method: 'POST',
    body: JSON.stringify({
      project_id: projectId,
      port,
      api_key: credentials.api_key,
      endpoint: credentials.endpoint,
      deployment: credentials.deployment,
      api_version: credentials.api_version,
    }),
  });
}

export async function stopMockAut(): Promise<{ status: string }> {
  return request<{ status: string }>('/mock-aut/stop', {
    method: 'POST',
  });
}

export async function getMockAutStatus(): Promise<MockAutStatus> {
  return request<MockAutStatus>('/mock-aut/status');
}

export async function runProjectVectorChecks(
  projectId: string,
  config: ProjectVectorChecksConfig,
): Promise<{
  project_id: string;
  report_id: string;
  summary: { passed: number; failed: number; inconclusive: number; total: number };
  results: Array<{ check: string; passed: boolean | null; detail: Record<string, unknown> }>;
  html_report: string;
}> {
  return request(`/projects/${encodeURIComponent(projectId)}/run-vector-checks`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export interface ProjectVectorChecksConfig {
  selected_checks: string[];
  required_fields: string[];
  duplicate_min_similarity: number;
  duplicate_max_similarity: number;
  negative_query_texts: string[];
  negative_query_max_similarity: number;
  drift_max_score: number;
  question_retrieval_top_k: number;
}

export async function runProjectWorkbookChecks(
  projectId: string,
  file: File,
  config: ProjectVectorChecksConfig,
  credentials: AzureCredentials,
): Promise<{
  project_id: string;
  report_id: string;
  summary: { passed: number; failed: number; inconclusive: number; total: number };
  results: Array<{ check: string; passed: boolean | null; detail: Record<string, unknown> }>;
  html_report: string;
  workbook_questions: Array<{
    question_index: number;
    row: number;
    question: string;
    expected_answer?: unknown;
    actual_answer?: unknown;
    pdf_name?: unknown;
    page_number?: unknown;
    source_pages?: unknown;
  }>;
  drift_workbook: {
    filename: string;
    content_base64: string;
    summary: Record<string, unknown>;
  } | null;
}> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('config_json', JSON.stringify(config));
  formData.append('api_key', credentials.api_key);
  formData.append('endpoint', credentials.endpoint);
  formData.append('deployment', credentials.deployment);
  formData.append('api_version', credentials.api_version);

  const response = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/run-workbook-checks`,
    { method: 'POST', body: formData },
  );
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
  return response.json();
}

export async function suggestOffTopicQuestions(
  projectId: string,
  topics: string[],
  countPerTopic: number,
  credentials: AzureCredentials,
): Promise<{ suggestions: Array<{ topic: string; questions: string[] }> }> {
  return request(`/projects/${encodeURIComponent(projectId)}/off-topic-question-suggestions`, {
    method: 'POST',
    body: JSON.stringify({
      topics,
      count_per_topic: countPerTopic,
      api_key: credentials.api_key,
      endpoint: credentials.endpoint,
      deployment: credentials.deployment,
      api_version: credentials.api_version,
    }),
  });
}

export async function uploadProjectPdfs(
  projectId: string,
  files: File[],
  jobId?: string,
  chunking?: ChunkingOptions,
): Promise<{ status: 'accepted'; job_id: string }> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  if (jobId) formData.append('job_id', jobId);
  if (chunking) formData.append('chunking_json', JSON.stringify(chunking));
  const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/ingest-pdfs`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
  return response.json();
}

export interface PdfIngestionResult {
  status: string;
  job_id: string;
  files: Array<Record<string, unknown>>;
  errors: Array<Record<string, unknown>>;
  chunks_ingested: number;
  image_records_ingested: number;
  text_records_ingested: number;
  table_records_ingested: number;
}

export interface PdfIngestionProgress {
  job_id: string;
  project_id: string;
  status: 'running' | 'complete' | 'failed';
  phase: string;
  progress: number;
  message: string;
  current_file: string;
  file_index: number;
  file_total: number;
  records_current: number;
  records_total: number;
  batch_current: number;
  batch_total: number;
  elapsed_seconds: number;
  chunks_ingested?: number;
  image_records_ingested?: number;
  text_records_ingested?: number;
  table_records_ingested?: number;
  errors?: Array<Record<string, unknown>>;
  result?: PdfIngestionResult;
}

export async function getPdfIngestionProgress(
  projectId: string,
  jobId: string,
): Promise<PdfIngestionProgress> {
  return request<PdfIngestionProgress>(
    `/projects/${encodeURIComponent(projectId)}/ingestion-progress/${encodeURIComponent(jobId)}`,
  );
}

export async function getProjectVectorRecords(
  projectId: string,
  page = 1,
  pageSize = 50,
): Promise<PaginatedVectorRecords> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  return request<PaginatedVectorRecords>(
    `/projects/${encodeURIComponent(projectId)}/vector-records?${params}`,
  );
}

export async function evaluateAutWorkbook(
  projectId: string,
  file: File,
  credentials: AzureCredentials,
): Promise<{
  filename: string;
  content_base64: string;
  summary: {
    passed?: number;
    failed?: number;
    errors?: number;
    error_details?: Array<{
      row: number;
      question: string;
      error: string;
    }>;
  };
}> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('api_key', credentials.api_key);
  formData.append('endpoint', credentials.endpoint);
  formData.append('deployment', credentials.deployment);
  formData.append('api_version', credentials.api_version);

  const response = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/evaluate-workbook`,
    { method: 'POST', body: formData },
  );
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
  return response.json();
}

// ── Legacy APIs (preserved) ──────────────────────────────────────────

export async function connect(payload: ConnectRequest): Promise<ConnectResponse> {
  return request<ConnectResponse>('/connect', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function runChecks(): Promise<RunChecksResponse> {
  return request<RunChecksResponse>('/run-checks', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function getResults(): Promise<ResultsResponse> {
  return request<ResultsResponse>('/results');
}

export async function getRecordDetail(recordId: string): Promise<RecordDetail> {
  return request<RecordDetail>(`/record/${encodeURIComponent(recordId)}`);
}

export async function getEmbeddingData(sampleSize = 2000): Promise<EmbeddingPoint[]> {
  const res = await request<{ points: EmbeddingPoint[]; method: string }>(
    `/embedding-data?sample_size=${sampleSize}`,
  );
  return res.points ?? [];
}

export async function generateAutoPairs(): Promise<{ status: string }> {
  return request<{ status: string }>('/auto-pairs', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function exportHtmlUrl(): string {
  return `${BASE}/export/html`;
}

export function exportJsonUrl(): string {
  return `${BASE}/export/json`;
}

export async function healthCheck(): Promise<{ status: string }> {
  return request<{ status: string }>('/health');
}

export async function getFieldMapper(): Promise<any> {
  return request<any>('/field-mapper');
}

export async function saveFieldMapper(config: any): Promise<any> {
  return request<any>('/field-mapper', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}
