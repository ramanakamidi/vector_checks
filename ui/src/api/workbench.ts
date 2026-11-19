export interface AzureCredentials {
  api_key: string;
  endpoint: string;
  deployment: string;
  api_version: string;
}

export interface ChunkingOptions {
  technique: 'recursive' | 'sentence' | 'token' | 'semantic';
  chunk_size: number;
  overlap: number;
  min_sentences_per_chunk: number;
  semantic_threshold: number;
}

export interface WorkbenchObject {
  object_id: string;
  pdf_name: string;
  page_no: number;
  pages: number[];
  object_type: string;
  object_no: string;
  position: string;
  title: string;
  characteristics: Record<string, unknown>;
  image_urls: string[];
}

export interface WorkbenchQA {
  question: string;
  expected_answer: string;
  pdf_name: string;
  object_type: string;
  page_no: number | string | null;
  category: string;
  source_type: string;
  source_section: string;
  chunk: string;
}

export interface WorkbenchSession {
  session_id: string;
  objects: WorkbenchObject[];
  text_enabled: boolean;
  text_chunks: Array<Record<string, unknown>>;
  text_errors: Array<{ pdf_name: string; error: string }>;
  chunking: ChunkingOptions;
  qa_records: WorkbenchQA[];
}

export interface WorkbenchAnalysisProgress {
  job_id: string;
  status: 'running' | 'complete' | 'failed';
  progress: number;
  message: string;
}

export interface WorkbenchGenerationProgress extends Omit<WorkbenchAnalysisProgress, 'status'> {
  status: 'running' | 'finalizing' | 'complete' | 'failed';
  completed: number;
  total: number;
}

export interface WorkbenchQAGenerationRequest {
  object_ids: string[];
  generate_objects: boolean;
  generate_text: boolean;
  generate_single_chunk: boolean;
  generate_multi_hop: boolean;
  object_question_count: number;
  text_question_count: number;
  multi_hop_chunk_count: number;
  multi_hop_question_count: number;
  api_key: string;
  endpoint: string;
  deployment: string;
  api_version: string;
}

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let message = text || `Request failed (${response.status})`;
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch (cause) {
      if (!(cause instanceof SyntaxError)) throw cause;
    }
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail;
      message = typeof detail === 'string' ? detail : JSON.stringify(detail);
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function createWorkbenchAnalysisJob(): Promise<WorkbenchAnalysisProgress> {
  return responseJson<WorkbenchAnalysisProgress>(
    await fetch('/api/workbench/analysis-jobs', { method: 'POST' }),
  );
}

export async function getWorkbenchAnalysisProgress(
  jobId: string,
): Promise<WorkbenchAnalysisProgress> {
  return responseJson<WorkbenchAnalysisProgress>(
    await fetch(`/api/workbench/analysis-jobs/${encodeURIComponent(jobId)}`),
  );
}

export async function analyzeWorkbench(
  files: File[],
  mode: string,
  chunking: ChunkingOptions,
  objectTypes: string[],
  credentials: AzureCredentials,
  jobId: string,
): Promise<WorkbenchSession> {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  form.append('job_id', jobId);
  form.append('mode', mode);
  form.append('technique', chunking.technique);
  form.append('chunk_size', String(chunking.chunk_size));
  form.append('overlap', String(chunking.overlap));
  form.append('min_sentences_per_chunk', String(chunking.min_sentences_per_chunk));
  form.append('semantic_threshold', String(chunking.semantic_threshold));
  objectTypes.forEach((type) => form.append('object_types', type));
  Object.entries(credentials).forEach(([key, value]) => form.append(key, value));
  return responseJson<WorkbenchSession>(
    await fetch('/api/workbench/analyze', { method: 'POST', body: form }),
  );
}

export async function getWorkbenchSession(sessionId: string): Promise<WorkbenchSession> {
  return responseJson<WorkbenchSession>(
    await fetch(`/api/workbench/${encodeURIComponent(sessionId)}`),
  );
}

export async function updateWorkbenchChunking(
  sessionId: string,
  chunking: ChunkingOptions,
): Promise<WorkbenchSession> {
  return responseJson<WorkbenchSession>(
    await fetch(`/api/workbench/${encodeURIComponent(sessionId)}/chunking`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chunking),
    }),
  );
}

export async function generateWorkbenchQA(
  sessionId: string,
  payload: WorkbenchQAGenerationRequest,
): Promise<WorkbenchSession> {
  return responseJson<WorkbenchSession>(
    await fetch(`/api/workbench/${encodeURIComponent(sessionId)}/qa`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function createWorkbenchQAGenerationJob(
  sessionId: string,
  payload: WorkbenchQAGenerationRequest,
): Promise<WorkbenchGenerationProgress> {
  return responseJson<WorkbenchGenerationProgress>(
    await fetch(`/api/workbench/${encodeURIComponent(sessionId)}/qa-jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function getWorkbenchQAGenerationProgress(
  jobId: string,
): Promise<WorkbenchGenerationProgress> {
  return responseJson<WorkbenchGenerationProgress>(
    await fetch(`/api/workbench/qa-jobs/${encodeURIComponent(jobId)}`),
  );
}

export async function getWorkbenchWorkbook(
  sessionId: string,
): Promise<{ filename: string; content_base64: string }> {
  return responseJson(
    await fetch(`/api/workbench/${encodeURIComponent(sessionId)}/workbook`),
  );
}

export async function reviewAnswers(
  file: File,
  projectId: string,
  topK: number,
  credentials: AzureCredentials,
): Promise<{ filename: string; content_base64: string; summary: Record<string, unknown> }> {
  const form = new FormData();
  form.append('file', file);
  form.append('project_id', projectId);
  form.append('top_k', String(topK));
  Object.entries(credentials).forEach(([key, value]) => form.append(key, value));
  return responseJson(
    await fetch('/api/answer-improvement/review', { method: 'POST', body: form }),
  );
}

export interface VisualPoint {
  id: string;
  x: number;
  y: number;
  cluster: number;
  text: string;
  metadata: Record<string, unknown>;
  qa?: WorkbenchQA[];
}

export interface VisualizationResponse {
  points: VisualPoint[];
  total_records: number;
  sampled_records: number;
  projected_records: number;
  skipped_records: number;
  message?: string;
  qa_matched_points?: number;
  qa_rows_loaded?: number;
}

export async function getProjectVisualization(
  projectId: string,
  sampleSize: number,
  minClusterSize: number,
  workbook?: File,
): Promise<VisualizationResponse> {
  if (workbook) {
    const form = new FormData();
    form.append('file', workbook);
    form.append('sample_size', String(sampleSize));
    form.append('min_cluster_size', String(minClusterSize));
    return responseJson(
      await fetch(
        `/api/projects/${encodeURIComponent(projectId)}/visualization-data/with-qa`,
        { method: 'POST', body: form },
      ),
    );
  }
  const params = new URLSearchParams({
    sample_size: String(sampleSize),
    min_cluster_size: String(minClusterSize),
  });
  return responseJson(
    await fetch(`/api/projects/${encodeURIComponent(projectId)}/visualization-data?${params}`),
  );
}

export function downloadBase64File(filename: string, content: string, mimeType: string): void {
  const binary = atob(content);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
