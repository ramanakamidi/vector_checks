import { useEffect, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Eye,
  FileUp,
  Loader2,
  Play,
  RefreshCw,
  Square,
  X,
} from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import { useAzureCredentials, hasAzureCredentials } from '../contexts/AzureCredentialsContext';
import {
  getProjectVectorRecords,
  getMockAutStatus,
  startMockAut,
  stopMockAut,
  uploadProjectPdfs,
  getPdfIngestionProgress,
} from '../api/client';
import type { PdfIngestionProgress } from '../api/client';
import type { MockAutStatus, ProjectVectorRecord } from '../api/types';
import type { ChunkingOptions } from '../api/workbench';
import AzureCredentialFields from '../components/AzureCredentialFields';

type AutStatus = MockAutStatus & { outdated?: boolean; prompt_version?: string };
type IngestionResult = Awaited<ReturnType<typeof uploadProjectPdfs>>;
const VECTOR_RECORDS_PAGE_SIZE = 50;
const DEFAULT_CHUNKING: ChunkingOptions = {
  technique: 'sentence',
  chunk_size: 512,
  overlap: 0,
  min_sentences_per_chunk: 1,
  semantic_threshold: 0.8,
};

function recordValue(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function StoredVectorRecordsDialog({
  projectId,
  projectName,
  onClose,
}: {
  projectId: string;
  projectName: string;
  onClose: () => void;
}) {
  const [records, setRecords] = useState<ProjectVectorRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    getProjectVectorRecords(projectId, page, VECTOR_RECORDS_PAGE_SIZE)
      .then((response) => {
        if (!active) return;
        setRecords(response.records);
        setTotal(response.total);
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setError(cause instanceof Error ? cause.message : 'Could not load stored records.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId, page]);

  const pageCount = Math.max(1, Math.ceil(total / VECTOR_RECORDS_PAGE_SIZE));
  const firstRecord = total === 0 ? 0 : (page - 1) * VECTOR_RECORDS_PAGE_SIZE + 1;
  const lastRecord = Math.min(page * VECTOR_RECORDS_PAGE_SIZE, total);

  return (
    <div
      className="fixed inset-0 z-[120] flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-sm sm:p-6"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="stored-records-title"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4 sm:px-7">
          <div className="min-w-0">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-indigo-600">PDF ingestion</p>
            <h2 id="stored-records-title" className="mt-1 text-lg font-bold text-slate-900 sm:text-xl">
              Stored vector records
            </h2>
            <p className="mt-1 truncate text-sm text-slate-500">{projectName}</p>
          </div>
          <button
            type="button"
            className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-slate-200 text-slate-500 transition hover:bg-slate-50 hover:text-slate-800"
            aria-label="Close stored records"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </header>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="flex items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/70 px-5 py-3 sm:px-7">
            <p className="text-sm text-slate-600">
              {loading ? 'Loading project records…' : `${total.toLocaleString()} record${total === 1 ? '' : 's'} stored`}
            </p>
            {!loading && total > 0 && (
              <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
                Page {page} of {pageCount}
              </span>
            )}
          </div>

          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4 sm:p-6">
            {loading && (
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-500">
                Loading records…
              </div>
            )}
            {!loading && error && (
              <div className="alert-error flex items-start gap-2">
                <AlertCircle size={17} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
            {!loading && !error && total === 0 && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-12 text-center">
                <FileUp size={24} className="mx-auto text-slate-400" />
                <p className="mt-3 font-semibold text-slate-700">No records stored yet</p>
                <p className="mt-1 text-sm text-slate-500">Ingest a PDF to see its image, prose, and table records here.</p>
              </div>
            )}
            {!loading && !error && records.map((record) => {
              const expanded = expandedId === record.id;
              const metadata = record.metadata ?? {};
              const pdfName = recordValue(
                metadata.pdf_name ?? metadata.source_pdf ?? metadata.docling_document_name,
              );
              const pageNumber = recordValue(
                metadata.page_number ?? metadata.page_no ?? metadata.page,
              );
              return (
                <article key={record.id} className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition hover:bg-indigo-50/40 sm:px-5"
                    aria-expanded={expanded}
                    onClick={() => setExpandedId(expanded ? null : record.id)}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-mono text-xs font-semibold text-slate-700">{record.id}</span>
                      <span className="mt-1 block truncate text-xs text-slate-500">
                        {pdfName || 'Source document unavailable'}
                        {pageNumber ? ` · Page ${pageNumber}` : ''}
                      </span>
                    </span>
                    <ChevronDown
                      size={18}
                      className={`shrink-0 text-indigo-600 transition-transform ${expanded ? 'rotate-180' : ''}`}
                    />
                  </button>
                  {expanded && (
                    <div className="space-y-4 border-t border-slate-100 bg-slate-50/60 p-4 sm:p-5">
                      {metadata.modality === 'image' ? (
                        <div>
                          <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">Stored figure image</p>
                          <img
                            className="max-h-[32rem] max-w-full rounded-lg border border-slate-200 bg-white object-contain"
                            src={`/api/projects/${encodeURIComponent(projectId)}/vector-records/${encodeURIComponent(record.id)}/image`}
                            alt={`${pdfName || 'PDF figure'}${pageNumber ? `, page ${pageNumber}` : ''}`}
                          />
                        </div>
                      ) : (
                        <div>
                          <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">Full chunk text</p>
                          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-slate-200 bg-white p-4 font-sans text-sm leading-6 text-slate-700">
                            {record.text || 'No chunk text is available for this record.'}
                          </pre>
                        </div>
                      )}
                      <div>
                        <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">Record metadata</p>
                        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-slate-200 bg-white p-4 font-mono text-xs leading-5 text-slate-600">
                          {Object.keys(metadata).length > 0
                            ? JSON.stringify(metadata, null, 2)
                            : 'No metadata stored for this record.'}
                        </pre>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>

          <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 bg-white px-5 py-3 sm:px-7">
            <p className="text-xs text-slate-500">
              {loading ? ' ' : `Showing ${firstRecord}–${lastRecord} of ${total.toLocaleString()}`}
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="button-secondary"
                disabled={page <= 1 || loading}
                onClick={() => {
                  setExpandedId(null);
                  setPage((current) => current - 1);
                }}
              >
                Previous
              </button>
              <button
                type="button"
                className="button-secondary"
                disabled={page >= pageCount || loading}
                onClick={() => {
                  setExpandedId(null);
                  setPage((current) => current + 1);
                }}
              >
                Next
              </button>
            </div>
          </footer>
        </div>
      </section>
    </div>
  );
}

export default function AppUnderTestPage() {
  const { activeProject } = useProject();
  const { credentials, setCredentials } = useAzureCredentials();
  const projectId = activeProject?.id ?? '';
  const [pdfs, setPdfs] = useState<File[]>([]);
  const [chunking, setChunking] = useState(DEFAULT_CHUNKING);
  const [port, setPort] = useState(8001);
  const [autStatus, setAutStatus] = useState<AutStatus | null>(null);
  const [ingestion, setIngestion] = useState<IngestionResult | null>(null);
  const [ingestionProgress, setIngestionProgress] = useState<PdfIngestionProgress | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'llm' | 'aut'>('llm');
  const [showStoredRecords, setShowStoredRecords] = useState(false);
  const supportsOverlap = chunking.technique === 'sentence'
    || chunking.technique === 'token';
  const validChunking = Number.isInteger(chunking.chunk_size)
    && chunking.chunk_size >= 64
    && chunking.chunk_size <= 8000
    && (!supportsOverlap || (
      Number.isInteger(chunking.overlap)
      && chunking.overlap >= 0
      && chunking.overlap < chunking.chunk_size
    ))
    && (chunking.technique !== 'sentence'
      || (Number.isInteger(chunking.min_sentences_per_chunk)
        && chunking.min_sentences_per_chunk >= 1
        && chunking.min_sentences_per_chunk <= 20))
    && (chunking.technique !== 'semantic'
      || (chunking.semantic_threshold >= 0.05 && chunking.semantic_threshold <= 0.95));

  useEffect(() => {
    setAutStatus(null);
    setIngestion(null);
    setIngestionProgress(null);
    setPdfs([]);
    setError('');
    setShowStoredRecords(false);
    if (!projectId) return;
    getMockAutStatus()
      .then((status) => setAutStatus(status))
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : 'Could not load Mock AUT status.'));
  }, [projectId]);

  const runAction = async (label: string, action: () => Promise<void>) => {
    setBusy(label);
    setError('');
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The request failed.');
    } finally {
      setBusy('');
    }
  };

  const running = Boolean(autStatus?.running);

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Project workflow</p>
          <h1>Configure</h1>
          <p>Configure LLM credentials and the application under test.</p>
        </div>
        {activeProject && <span className="status-pill">{activeProject.name}</span>}
      </header>

      {error && <div className="alert-error flex items-start gap-2"><AlertCircle size={17} />{error}</div>}
      {showStoredRecords && activeProject && (
        <StoredVectorRecordsDialog
          projectId={projectId}
          projectName={activeProject.name}
          onClose={() => setShowStoredRecords(false)}
        />
      )}
      <div className="configure-tabs" role="tablist" aria-label="Configuration sections">
        <button
          id="llm-config-tab"
          type="button"
          role="tab"
          aria-selected={activeTab === 'llm'}
          aria-controls="llm-config-panel"
          className={activeTab === 'llm' ? 'configure-tab configure-tab-active' : 'configure-tab'}
          onClick={() => setActiveTab('llm')}
        >
          LLM Config
        </button>
        <button
          id="aut-config-tab"
          type="button"
          role="tab"
          aria-selected={activeTab === 'aut'}
          aria-controls="aut-config-panel"
          className={activeTab === 'aut' ? 'configure-tab configure-tab-active' : 'configure-tab'}
          onClick={() => setActiveTab('aut')}
        >
          AUT Config
        </button>
      </div>

      {activeTab === 'llm' ? (
        <section
          className="panel space-y-5"
          id="llm-config-panel"
          role="tabpanel"
          aria-labelledby="llm-config-tab"
        >
          <div className="section-title">
            <span className="icon-tile"><CheckCircle2 size={19} /></span>
            <div><h2>Azure OpenAI configuration</h2><p>These in-memory credentials are shared with workflows that use an LLM.</p></div>
          </div>
          <AzureCredentialFields credentials={credentials} onChange={setCredentials} />
        </section>
      ) : !activeProject ? (
        <section
          className="panel"
          id="aut-config-panel"
          role="tabpanel"
          aria-labelledby="aut-config-tab"
        >
          <h2 className="text-base font-semibold text-slate-800">Choose a project to begin</h2>
          <p className="mt-2 text-sm text-slate-500">Use the project selector in the header to create or load a project.</p>
        </section>
      ) : (
        <div id="aut-config-panel" role="tabpanel" aria-labelledby="aut-config-tab">
        <>
          <section className="panel space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div className="section-title">
                <span className="icon-tile"><FileUp size={19} /></span>
                <div><h2>PDF ingestion</h2><p>Embed detected figures, prose, and tables in this project’s vector store.</p></div>
              </div>
              <button
                type="button"
                className="button-secondary shrink-0"
                aria-label="View stored PDF records"
                title="View all stored PDF chunks"
                disabled={Boolean(busy)}
                onClick={() => setShowStoredRecords(true)}
              >
                <Eye size={16} />
                <span>View records</span>
              </button>
            </div>
            <label className="field-label">
              Upload PDF files
              <input
                type="file"
                accept=".pdf,application/pdf"
                multiple
                onChange={(event) => setPdfs(Array.from(event.target.files ?? []))}
              />
            </label>
            {pdfs.length > 0 && <p className="text-xs text-slate-500">{pdfs.map((file) => file.name).join(', ')}</p>}
            <div className="space-y-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="field-help">
                Detected figures are embedded as images; prose and tables are embedded as text chunks using the selected technique.
              </p>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <label className="field-label">Chunking technique
                  <select
                    value={chunking.technique}
                    onChange={(event) => setChunking({
                      ...DEFAULT_CHUNKING,
                      technique: event.target.value as ChunkingOptions['technique'],
                    })}
                  >
                    <option value="recursive">Recursive</option>
                    <option value="sentence">Sentence</option>
                    <option value="token">Token</option>
                    <option value="semantic">Semantic</option>
                  </select>
                </label>
                <label className="field-label">Max tokens
                  <input
                    type="number"
                    min={64}
                    max={8000}
                    value={chunking.chunk_size}
                    onChange={(event) => {
                      const chunkSize = Number(event.target.value);
                      setChunking({
                        ...chunking,
                        chunk_size: chunkSize,
                        overlap: Math.min(chunking.overlap, chunkSize - 1),
                      });
                    }}
                  />
                </label>
                {supportsOverlap && <label className="field-label">Overlap
                  <input
                    type="number"
                    min={0}
                    max={Math.min(4096, chunking.chunk_size - 1)}
                    value={chunking.overlap}
                    onChange={(event) => setChunking({
                      ...chunking,
                      overlap: Number(event.target.value),
                    })}
                  />
                </label>}
                {chunking.technique === 'sentence' && <label className="field-label">Minimum sentences
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={chunking.min_sentences_per_chunk}
                    onChange={(event) => setChunking({
                      ...chunking,
                      min_sentences_per_chunk: Number(event.target.value),
                    })}
                  />
                </label>}
                {chunking.technique === 'semantic' && <label className="field-label">Semantic threshold
                  <input
                    type="number"
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    value={chunking.semantic_threshold}
                    onChange={(event) => setChunking({
                      ...chunking,
                      semantic_threshold: Number(event.target.value),
                    })}
                  />
                </label>}
              </div>
              {!validChunking && (
                <p className="field-help text-red-600">
                  Chunk settings are invalid or exceed the embedding model&apos;s 8,000-token input budget.
                </p>
              )}
            </div>
            <button
              className="button-primary"
              disabled={!pdfs.length || Boolean(busy) || !validChunking}
              onClick={() => runAction('ingest', async () => {
                const jobId = globalThis.crypto?.randomUUID?.()
                  ?? `pdf-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                setIngestionProgress({
                  job_id: jobId,
                  project_id: projectId,
                  status: 'running',
                  phase: 'queued',
                  progress: 0,
                  message: 'Queued for PDF text, table, and figure extraction...',
                  current_file: pdfs[0]?.name ?? '',
                  file_index: 0,
                  file_total: pdfs.length,
                  records_current: 0,
                  records_total: 0,
                  batch_current: 0,
                  batch_total: 0,
                  elapsed_seconds: 0,
                });
                const poll = window.setInterval(async () => {
                  try {
                    setIngestionProgress(await getPdfIngestionProgress(projectId, jobId));
                  } catch {
                    // The job may be initialized immediately after the POST starts.
                  }
                }, 750);
                let result: IngestionResult;
                try {
                  result = await uploadProjectPdfs(projectId, pdfs, jobId, chunking);
                  setIngestionProgress(await getPdfIngestionProgress(projectId, jobId));
                } finally {
                  window.clearInterval(poll);
                }
                setIngestion(result);
                setPdfs([]);
              })}
            >
              {busy === 'ingest' ? 'Processing PDFs…' : 'Parse and ingest PDFs'}
            </button>
            {busy === 'ingest' && ingestionProgress && (
              <div className="rounded-lg border border-blue-200 bg-blue-50 p-4" role="status" aria-live="polite">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex min-w-0 items-center gap-3">
                    <Loader2 size={20} className="shrink-0 animate-spin text-blue-600" />
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-blue-900">
                        {ingestionProgress.phase === 'embedding'
                          ? 'Generating Qwen embeddings'
                          : ingestionProgress.phase === 'loading_model'
                            ? 'Loading the local Qwen model'
                          : ingestionProgress.phase === 'parsing'
                            ? 'Parsing PDF with Docling'
                            : ingestionProgress.phase === 'extracting'
                              ? 'Extracting prose, tables, and figure images'
                              : 'Preparing vector records'}
                      </p>
                      <p className="truncate text-xs text-blue-700">{ingestionProgress.message}</p>
                    </div>
                  </div>
                  <span className="shrink-0 text-sm font-bold text-blue-700">
                    {ingestionProgress.progress}%
                  </span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-blue-100">
                  <div
                    className="h-full rounded-full bg-blue-600 transition-all duration-300"
                    style={{ width: `${ingestionProgress.progress}%` }}
                  />
                </div>
                <div className="mt-3 grid gap-2 text-xs text-blue-800 sm:grid-cols-2 lg:grid-cols-4">
                  <span>File: {ingestionProgress.file_index}/{ingestionProgress.file_total}</span>
                  <span>PDF: {ingestionProgress.current_file || '—'}</span>
                  <span>
                    Stored records: {ingestionProgress.records_current}/{ingestionProgress.records_total || '—'}
                  </span>
                  <span>Elapsed: {ingestionProgress.elapsed_seconds}s</span>
                </div>
                {ingestionProgress.batch_total > 0 && (
                  <p className="mt-2 text-xs text-blue-700">
                    {ingestionProgress.phase === 'loading_model'
                      ? 'Next record'
                      : ingestionProgress.batch_total === ingestionProgress.records_total
                      ? 'Embedding record'
                      : 'Embedding batch'}{' '}
                    {ingestionProgress.batch_current}/{ingestionProgress.batch_total}
                  </p>
                )}
              </div>
            )}
            {ingestion && (
              <div className={ingestion.errors.length ? 'alert-warning' : 'alert-success'}>
                Processed {ingestion.files.length} file(s); {ingestion.image_records_ingested} image, {ingestion.text_records_ingested} text, and {ingestion.table_records_ingested} table record(s) added.
                {ingestion.errors.length > 0 && <pre className="mt-2 whitespace-pre-wrap text-xs">{JSON.stringify(ingestion.errors, null, 2)}</pre>}
              </div>
            )}
          </section>

          <section className="panel space-y-5">
            <div className="section-title">
              <span className="icon-tile"><Play size={19} /></span>
              <div><h2>Check AUT-status</h2><p>Answers questions using this project’s configured vector store.</p></div>
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <label className="field-label compact-field">
                Mock AUT port
                <input type="number" min={1024} max={65535} value={port} onChange={(event) => setPort(Number(event.target.value))} />
              </label>
              <button
                className="button-secondary"
                disabled={Boolean(busy)}
                onClick={() => runAction('status', async () => setAutStatus(await getMockAutStatus()))}
              ><RefreshCw size={15} />Check status</button>
              {running && !autStatus?.outdated ? (
                <button
                  className="button-secondary"
                  disabled={Boolean(busy)}
                  onClick={() => runAction('stop', async () => {
                    await stopMockAut();
                    setAutStatus(await getMockAutStatus());
                  })}
                ><Square size={15} />Stop Mock AUT</button>
              ) : (
                <button
                  className="button-primary"
                  disabled={Boolean(busy) || !hasAzureCredentials(credentials) || port < 1024 || port > 65535}
                  onClick={() => runAction('start', async () => {
                    const started = await startMockAut(projectId, credentials, port);
                    setAutStatus({
                      running: true,
                      url: started.url,
                      pid: started.pid,
                      project_id: started.project_id,
                      outdated: false,
                      prompt_version: started.prompt_version,
                    });
                  })}
                ><Play size={15} />{running ? 'Restart Mock AUT' : 'Start Mock AUT'}</button>
              )}
            </div>
            {autStatus && (
              <div className={running && !autStatus.outdated ? 'alert-success' : autStatus.outdated ? 'alert-warning' : 'chunk-preview'}>
                <div className="flex items-center gap-2">
                  {running && !autStatus.outdated ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                  <strong>{running ? (autStatus.outdated ? 'Restart required' : 'Mock AUT running') : 'Mock AUT stopped'}</strong>
                </div>
                {autStatus.url && <p className="mt-1 text-xs">{autStatus.url}</p>}
                {autStatus.outdated && <p className="mt-1 text-xs">The service is outdated. Restart it before running workflow checks.</p>}
              </div>
            )}
          </section>
        </>
        </div>
      )}
    </div>
  );
}
