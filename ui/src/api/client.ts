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
} from './types';

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

export async function createProject(name: string, description = ''): Promise<Project> {
  return request<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify({ name, description }),
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
  port = 8001,
): Promise<MockAutStatus> {
  return request<MockAutStatus>('/mock-aut/start', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, port }),
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
