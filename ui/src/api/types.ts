// ── Project Types ──

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  status: 'created' | 'configured' | 'ingested' | 'generated' | 'executed';
}

export interface LLMConfig {
  provider: string;
  api_key: string;
  api_key_set?: boolean;
  model: string;
}

export interface AUTConfig {
  url: string;
  method: string;
  endpoint: string;
  headers: Record<string, string>;
  request_template: Record<string, any>;
  response_answer_path: string;
  response_context_path: string;
}

export interface VectorStoreConfig {
  db_type: string;
  collection_name: string;
  persist_directory: string;
}

export interface FieldMapperConfig {
  doc_id_field: string;
  doc_name_field: string;
  text_field: string;
  page_field: string;
  chunk_index_field: string;
  object_type_field: string;
  display_fields: string[];
  required_fields: string[];
  similar_group_field: string;
  different_group_field: string;
  [key: string]: any;
}

export interface ProjectSettings {
  files_directory?: string;
  extraction_mode?: string;
  llm_config?: LLMConfig;
  aut_config?: AUTConfig;
  vector_store_config?: VectorStoreConfig;
  field_mapper?: FieldMapperConfig;
}

// ── Q&A Data Types ──

export interface QARecord {
  id: string;
  question: string;
  answer: string;
  context: string;
  metadata: Record<string, any>;
}

export interface IngestResponse {
  status: string;
  records_ingested: number;
  collection_name: string;
  errors: string[];
}

export interface IngestionStats {
  total_records: number;
  collections: string[];
  metadata_fields: string[];
  ingested_at: string | null;
  chroma_path: string;
}

export interface PaginatedRecords {
  records: QARecord[];
  total: number;
  page: number;
  page_size: number;
}

// ── Dataset Generation Types ──

export interface DatasetConfig {
  num_pairs_per_chunk: number;
  total_limit: number;
  query_conditions: Record<string, string>;
  include_context: boolean;
  difficulty_levels: string[];
  generation_mode: 'from_ingested' | 'llm_augmented';
}

export interface QAPair {
  id: string;
  question: string;
  expected_answer: string;
  context: string;
  source_record_id: string;
  difficulty: string;
  type: string;
  metadata: Record<string, any>;
}

export interface GeneratedDataset {
  dataset_id: string;
  generated_at: string;
  total_pairs: number;
  config_used: DatasetConfig;
  pairs: QAPair[];
  page?: number;
  page_size?: number;
  total_pages?: number;
}

export interface DatasetSummary {
  dataset_id: string;
  generated_at: string;
  total_pairs: number;
  config_used: DatasetConfig;
}

// ── Legacy Dataset Types ──

export interface TestCase {
  id: string;
  query: string;
  expected_answer: string;
  expected_context: string;
  source_chunk_id: string;
  metadata: Record<string, any>;
}

// ── Execution Types (Phase 6) ──

export interface ExecutionConfig {
  dataset_id: string;
  aut_endpoint: string;
  aut_project_id: string;
  run_vector_checks: boolean;
  checks_to_run: string[];
  collection_name?: string;
}

export interface TestResult {
  pair_id: string;
  question: string;
  expected_answer: string;
  aut_answer: string;
  aut_confidence: number;
  aut_context: any[];
  aut_sources: string[];
  is_correct: boolean;
  keyword_overlap: number;
  context_relevance: number;
  response_time_ms: number;
  metadata: Record<string, any>;
}

export interface ExecutionSummary {
  total_questions: number;
  correct_answers: number;
  accuracy: number;
  avg_confidence: number;
  avg_context_relevance: number;
  avg_response_time_ms: number;
  aut_endpoint: string;
  vector_store_health?: Record<string, any>;
}

export interface VectorCheckResult {
  check: string;
  passed: boolean;
  detail: Record<string, any>;
}

export interface VectorChecks {
  status: string;
  message?: string;
  summary: Record<string, any>;
  checks: VectorCheckResult[];
}

export interface ExecutionResult {
  execution_id: string;
  executed_at: string;
  duration_seconds: number;
  dataset_id: string;
  status: string;
  summary: ExecutionSummary;
  results: TestResult[];
  vector_checks: VectorChecks;
}

export interface ExecutionListItem {
  execution_id: string;
  executed_at: string;
  duration_seconds: number;
  dataset_id: string;
  status: string;
  summary: ExecutionSummary;
}

export interface MockAutStatus {
  running: boolean;
  url: string | null;
  pid: number | null;
  project_id: string | null;
}

// ── Existing types (preserved) ──

export interface CheckResult {
  check: string;
  passed: boolean | null;
  detail: Record<string, any>;
}

export interface Summary {
  passed: number;
  failed: number;
  inconclusive: number;
  total: number;
}

export interface EmbeddingPoint {
  id: string;
  x: number;
  y: number;
  metadata: Record<string, any>;
  fault_type: string | null;
}

export interface FaultDetail {
  check: string;
  expected: string;
  actual: string;
  severity: 'error' | 'warning';
}

export interface RecordDetail {
  id: string;
  vector_dim: number;
  vector_norm: number;
  vector_preview: number[];
  metadata: Record<string, any>;
  fault_types?: string[];
  fault_details?: FaultDetail[];
}

export interface ConnectRequest {
  db_type: string;
  params: Record<string, string>;
}

export interface ConnectResponse {
  status: string;
  record_count: number;
  db_type: string;
}

export interface RunChecksResponse {
  status: string;
  results: CheckResult[];
  summary: Summary;
}

export interface ResultsResponse {
  results: CheckResult[];
  summary: Summary;
}

export type StatusFilter = 'all' | 'pass' | 'fail' | 'info';
