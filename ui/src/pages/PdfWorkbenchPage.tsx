import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, Download, FileSearch, GitBranch, Layers, LoaderCircle, RefreshCw, Sparkles, X } from 'lucide-react';
import AzureCredentialFields from '../components/AzureCredentialFields';
import {
  analyzeWorkbench,
  createWorkbenchAnalysisJob,
  createWorkbenchQAGenerationJob,
  downloadBase64File,
  getWorkbenchAnalysisProgress,
  getWorkbenchQAGenerationProgress,
  getWorkbenchSession,
  getWorkbenchWorkbook,
  updateWorkbenchChunking,
} from '../api/workbench';
import type {
  ChunkingOptions,
  WorkbenchAnalysisProgress,
  WorkbenchGenerationProgress,
  WorkbenchObject,
} from '../api/workbench';
import { hasAzureCredentials } from '../contexts/AzureCredentialsContext';
import { usePdfWorkbench } from '../contexts/PdfWorkbenchContext';
import type { ObjectType } from '../contexts/PdfWorkbenchContext';

const DEFAULT_CHUNKING: ChunkingOptions = {
  technique: 'sentence',
  chunk_size: 512,
  overlap: 0,
  min_sentences_per_chunk: 1,
  semantic_threshold: 0.8,
};
const SUPPORTED_CHUNKING_TECHNIQUES: ChunkingOptions['technique'][] = [
  'recursive',
  'sentence',
  'token',
  'semantic',
];
const OBJECT_TYPES: ObjectType[] = ['table', 'image', 'graph'];
const SELECTABLE_TYPES: ObjectType[] = [...OBJECT_TYPES, 'text'];

function normalizeChunking(chunking: ChunkingOptions | undefined): ChunkingOptions {
  return chunking && SUPPORTED_CHUNKING_TECHNIQUES.includes(chunking.technique)
    ? chunking
    : DEFAULT_CHUNKING;
}

function isObjectType(value: string): value is ObjectType {
  return OBJECT_TYPES.some((type) => type === value);
}

function objectSummary(item: WorkbenchObject): string {
  const details = item.characteristics;
  return String(details.content_summary ?? details.description ?? item.title ?? 'No summary available.');
}

export default function PdfWorkbenchPage() {
  const {
    generationCredentials,
    setGenerationCredentials,
    session,
    setSession,
    selectedIds,
    setSelectedIds,
    selectedObjectTypes,
    setSelectedObjectTypes,
  } = usePdfWorkbench();
  const [previewObject, setPreviewObject] = useState<WorkbenchObject | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState('crop');
  const [chunking, setChunking] = useState(DEFAULT_CHUNKING);
  const [objectCount, setObjectCount] = useState(5);
  const [textCount, setTextCount] = useState(2);
  const [generateSingleChunk, setGenerateSingleChunk] = useState(true);
  const [generateMultiHop, setGenerateMultiHop] = useState(false);
  const [multiHopChunkCount, setMultiHopChunkCount] = useState(2);
  const [multiHopQuestionCount, setMultiHopQuestionCount] = useState(2);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [analysisProgress, setAnalysisProgress] = useState<WorkbenchAnalysisProgress | null>(null);
  const [generationProgress, setGenerationProgress] = useState<WorkbenchGenerationProgress | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!previewObject) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPreviewObject(null);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [previewObject]);

  useEffect(() => {
    const sessionId = session?.session_id ?? null;
    if (activeSessionIdRef.current === sessionId) return;
    activeSessionIdRef.current = sessionId;
    setChunking(normalizeChunking(session?.chunking));
    setGenerationProgress(null);
  }, [session]);

  const hasPendingChunking = useMemo(
    () => session !== null && JSON.stringify(chunking) !== JSON.stringify(session.chunking),
    [chunking, session],
  );
  const supportsOverlap = chunking.technique === 'sentence'
    || chunking.technique === 'token';
  const minimumChunkSize = 64;
  const validChunking = Number.isInteger(chunking.chunk_size)
    && chunking.chunk_size >= minimumChunkSize
    && chunking.chunk_size <= 8000
    && (!supportsOverlap || (
      Number.isInteger(chunking.overlap)
      && chunking.overlap >= 0
      && chunking.overlap <= 4096
      && chunking.overlap < chunking.chunk_size
    ))
    && (chunking.technique !== 'sentence'
      || (Number.isInteger(chunking.min_sentences_per_chunk)
        && chunking.min_sentences_per_chunk >= 1
        && chunking.min_sentences_per_chunk <= 20))
    && (chunking.technique !== 'semantic'
      || (chunking.semantic_threshold >= 0.05 && chunking.semantic_threshold <= 0.95));
  const textSelected = selectedObjectTypes.includes('text');
  const sessionHasText = Boolean(session?.text_enabled);
  const canGenerateText = textSelected && sessionHasText;
  const validQuestionCounts = (!selectedIds.length || (
    Number.isInteger(objectCount)
    && objectCount >= 1
    && objectCount <= 20
  ))
    && (!canGenerateText || (
      (!generateSingleChunk || (
        Number.isInteger(textCount)
        && textCount >= 1
        && textCount <= 3
      ))
      && (!generateMultiHop || (
        Number.isInteger(multiHopChunkCount)
        && multiHopChunkCount >= 2
        && multiHopChunkCount <= 10
        && Number.isInteger(multiHopQuestionCount)
        && multiHopQuestionCount >= 1
        && multiHopQuestionCount <= 3
      ))
    ));
  const visibleObjects = session?.objects.filter(
    (item) => isObjectType(item.object_type) && selectedObjectTypes.includes(item.object_type),
  ) ?? [];

  const setTechnique = (technique: ChunkingOptions['technique']) => {
    setChunking({
      ...DEFAULT_CHUNKING,
      technique,
      chunk_size: 512,
      overlap: 0,
    });
  };

  const runAnalysis = async () => {
    setBusy('analysis');
    setError('');
    setNotice('');
    setAnalysisProgress(null);
    let stopPolling = false;
    let progressTask: Promise<void> | null = null;
    try {
      const job = await createWorkbenchAnalysisJob();
      setAnalysisProgress(job);
      progressTask = (async () => {
        while (!stopPolling) {
          await new Promise((resolve) => window.setTimeout(resolve, 700));
          if (stopPolling) return;
          try {
            const progress = await getWorkbenchAnalysisProgress(job.job_id);
            if (stopPolling) return;
            setAnalysisProgress(progress);
            if (progress.status !== 'running') return;
          } catch (cause) {
            if (!stopPolling) {
              const details = cause instanceof Error ? ` (${cause.message})` : '';
              setAnalysisProgress((current) => current && ({
                ...current,
                message: `Progress updates are unavailable${details}. Analysis is still running.`,
              }));
            }
            return;
          }
        }
      })();
      const current = await analyzeWorkbench(
        files,
        mode,
        chunking,
        selectedObjectTypes,
        generationCredentials,
        job.job_id,
      );
      setSession(current);
      setSelectedIds([]);
      setAnalysisProgress({
        ...job,
        status: 'complete',
        progress: 100,
        message: `Analysis complete: ${current.objects.length} objects and ${current.text_chunks.length} text chunks.`,
      });
      setNotice(`Analysis complete: ${current.objects.length} objects and ${current.text_chunks.length} text chunks.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'PDF analysis failed.');
      setAnalysisProgress((current) => current && ({
        ...current,
        status: 'failed',
        message: 'Analysis failed. See the error message above.',
      }));
    } finally {
      stopPolling = true;
      if (progressTask) await progressTask;
      setBusy('');
    }
  };

  const applyChunking = async () => {
    if (!session) return;
    setBusy('chunking');
    setError('');
    setNotice('');
    try {
      const current = await updateWorkbenchChunking(session.session_id, chunking);
      setSession(current);
      setNotice(`Re-chunked the extracted text into ${current.text_chunks.length} chunks. Existing text Q&A was cleared.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not apply chunking settings.');
    } finally {
      setBusy('');
    }
  };

  const runGeneration = async () => {
    if (!session) return;
    setBusy('qa');
    setError('');
    setNotice('');
    setGenerationProgress({
      job_id: '',
      status: 'running',
      progress: 0,
      completed: 0,
      total: 0,
      message: 'Starting Q&A generation...',
    });
    try {
      let progress = await createWorkbenchQAGenerationJob(session.session_id, {
        object_ids: selectedIds,
        generate_objects: selectedIds.length > 0,
        generate_text: canGenerateText && (generateSingleChunk || generateMultiHop),
        generate_single_chunk: generateSingleChunk,
        generate_multi_hop: generateMultiHop,
        object_question_count: objectCount,
        text_question_count: textCount,
        multi_hop_chunk_count: multiHopChunkCount,
        multi_hop_question_count: multiHopQuestionCount,
        ...generationCredentials,
      });
      setGenerationProgress(progress);
      while (progress.status === 'running' || progress.status === 'finalizing') {
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, 700);
        });
        progress = await getWorkbenchQAGenerationProgress(progress.job_id);
        setGenerationProgress(progress);
      }
      if (progress.status === 'failed') {
        throw new Error(progress.message || 'Q&A generation failed.');
      }
      const current = await getWorkbenchSession(session.session_id);
      setSession(current);
      setNotice(`Q&A generated: ${current.qa_records.length} question-and-answer pairs.`);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Q&A generation failed.';
      setError(message);
      setGenerationProgress((current) => current && ({
        ...current,
        status: 'failed',
        message,
      }));
    } finally {
      setBusy('');
    }
  };

  const downloadWorkbook = async () => {
    if (!session) return;
    try {
      const workbook = await getWorkbenchWorkbook(session.session_id);
      downloadBase64File(
        workbook.filename,
        workbook.content_base64,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not download the workbook.');
    }
  };

  const toggleObject = (objectId: string) => {
    setSelectedIds((current) => (
      current.includes(objectId)
        ? current.filter((id) => id !== objectId)
        : [...current, objectId]
    ));
  };

  const toggleObjectType = (type: ObjectType) => {
    const nextTypes = selectedObjectTypes.includes(type)
      ? selectedObjectTypes.filter((value) => value !== type)
      : [...selectedObjectTypes, type];
    setSelectedObjectTypes(nextTypes);
    if (session) {
      const visibleIds = new Set(
        session.objects
          .filter((item) => isObjectType(item.object_type) && nextTypes.includes(item.object_type))
          .map((item) => item.object_id),
      );
      setSelectedIds((current) => current.filter((id) => visibleIds.has(id)));
    }
  };

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Document intelligence</p>
          <h1>Test Data Generation</h1>
          <p>Analyze PDF figures and prose, select objects inline, and create a linked Q&amp;A workbook.</p>
        </div>
        {session && (
          <button className="button-secondary" onClick={downloadWorkbook} disabled={!session.qa_records.length}>
            <Download size={16} /> Export Q&amp;A workbook
          </button>
        )}
      </header>

      <section className="panel space-y-5">
        <div className="section-title">
          <div className="icon-tile"><FileSearch size={19} /></div>
          <div><h2>Analyze documents</h2><p>Upload PDFs and choose how visual context and text chunks are prepared.</p></div>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <label className="dropzone">
            <input
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
            />
            <FileSearch size={25} className="text-indigo-600" />
            <span className="font-semibold">{files.length ? `${files.length} PDF file(s) selected` : 'Choose PDF files'}</span>
            <span className="text-sm text-slate-500">PDF only · Up to 10 files / 100 MB total</span>
          </label>
          <div className="space-y-4">
            <fieldset className="field-label border-0 p-0">
              <legend className="mb-2">Object types to detect</legend>
              <div className="object-type-filters" aria-label="Object types to detect">
                {SELECTABLE_TYPES.map((type) => (
                  <label className="check-label" key={type}>
                    <input
                      type="checkbox"
                      checked={selectedObjectTypes.includes(type)}
                      onChange={() => toggleObjectType(type)}
                    />
                    <span>{type === 'image' ? 'Image' : type === 'graph' ? 'Graph' : type === 'text' ? 'Text' : 'Table'}</span>
                  </label>
                ))}
              </div>
              {!selectedObjectTypes.length && (
                <span className="field-help">Select at least one object type before analyzing.</span>
              )}
            </fieldset>
            <label className="field-label">Crop mode
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="crop">Object crop</option>
                <option value="page">Crop + full page</option>
                <option value="page_context">Crop + neighboring pages</option>
              </select>
            </label>
            {textSelected && (
              <label className="field-label">Text chunking technique
                <select value={chunking.technique} onChange={(event) => setTechnique(event.target.value as ChunkingOptions['technique'])}>
                  <option value="recursive">Recursive</option>
                  <option value="sentence">Sentence</option>
                  <option value="token">Token</option>
                  <option value="semantic">Semantic</option>
                </select>
              </label>
            )}
          </div>
        </div>

        {textSelected && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="field-label">Max tokens
            <input type="number" min={minimumChunkSize} max={8000} value={chunking.chunk_size}
              onChange={(event) => {
                const chunkSize = Number(event.target.value);
                setChunking({
                  ...chunking,
                  chunk_size: chunkSize,
                  overlap: supportsOverlap
                    ? Math.max(0, Math.min(chunking.overlap, chunkSize - 1))
                    : chunking.overlap,
                });
              }} />
          </label>
          {supportsOverlap && <label className="field-label">Overlap
            <input type="number" min={0} max={Math.min(4096, chunking.chunk_size - 1)} value={chunking.overlap}
              onChange={(event) => setChunking({ ...chunking, overlap: Number(event.target.value) })} />
          </label>}
          {chunking.technique === 'sentence' && <label className="field-label">Minimum sentences
            <input type="number" min={1} max={20} value={chunking.min_sentences_per_chunk}
              onChange={(event) => setChunking({ ...chunking, min_sentences_per_chunk: Number(event.target.value) })} />
            <span className="field-help">Best-effort; short text or the token limit may yield fewer sentences.</span>
          </label>}
          {chunking.technique === 'semantic' && <label className="field-label">Semantic threshold
            <input type="number" min={0.05} max={0.95} step={0.05} value={chunking.semantic_threshold}
              onChange={(event) => setChunking({ ...chunking, semantic_threshold: Number(event.target.value) })} />
          </label>}
        </div>)}
        <details className="credential-disclosure">
          <summary>Azure OpenAI credentials <span>For this tab only; kept in memory only</span></summary>
          <div className="pt-4"><AzureCredentialFields credentials={generationCredentials} onChange={setGenerationCredentials} /></div>
        </details>
        <div className="flex flex-wrap gap-3">
          <button className="button-primary" onClick={runAnalysis} disabled={!files.length || !selectedObjectTypes.length || (textSelected && !validChunking) || !hasAzureCredentials(generationCredentials) || !!busy}>
            {busy === 'analysis' ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />}
            {busy === 'analysis' ? 'Analyzing PDFs…' : 'Analyze PDFs'}
          </button>
          {session && canGenerateText && (
            <button className="button-secondary" onClick={applyChunking} disabled={!hasPendingChunking || !validChunking || !!busy}>
              {busy === 'chunking' ? <LoaderCircle className="animate-spin" size={16} /> : <RefreshCw size={16} />}
              Apply chunking settings
            </button>
          )}
        </div>
        {busy === 'analysis' && analysisProgress && (
          <div className="workbench-progress-card" role="status" aria-live="polite">
            <div className="workbench-progress-icon">
              <FileSearch size={22} />
            </div>
            <div className="workbench-progress-content">
              <div className="workbench-progress-heading">
                <div>
                  <p className="workbench-progress-eyebrow">Document analysis</p>
                  <p className="workbench-progress-message">{analysisProgress.message}</p>
                </div>
                <strong>{analysisProgress.progress}%</strong>
              </div>
              <div
                className="workbench-progress-track"
                role="progressbar"
                aria-label="PDF analysis progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={analysisProgress.progress}
              >
                <span style={{ width: `${analysisProgress.progress}%` }} />
              </div>
              <div className="workbench-progress-footer">
                <span>{files.length} PDF{files.length === 1 ? '' : 's'} in this run</span>
                <span>Large documents may take a little longer</span>
              </div>
            </div>
          </div>
        )}
      </section>

      {error && <div className="alert-error" role="alert">{error}</div>}
      {notice && <div className="alert-success" role="status">{notice}</div>}
      {textSelected && !validChunking && <div className="alert-warning" role="alert">Chunk settings must be valid; overlap must be smaller than the chunk size.</div>}

      {session && (
        <>
          {canGenerateText && session.text_errors.length > 0 && (
            <div className="alert-error">
              {session.text_errors.map((item) => <p key={item.pdf_name}>{item.pdf_name}: {item.error}</p>)}
            </div>
          )}
          <section className="panel">
            <div className="section-title">
              <div><h2>Detected objects</h2><p>Only the object types selected before analysis are shown. Select objects for Q&amp;A.</p></div>
              <span className="status-pill">{selectedIds.length} selected</span>
            </div>
            {visibleObjects.length === 0 ? (
              <p className="empty-state">No figures, tables, or images were detected in this analysis.</p>
            ) : (
              <div className="object-grid">
                {visibleObjects.map((item) => (
                  <article className={`object-card ${selectedIds.includes(item.object_id) ? 'object-card-selected' : ''}`} key={item.object_id}>
                    <input id={`object-${item.object_id}`} type="checkbox" checked={selectedIds.includes(item.object_id)} onChange={() => toggleObject(item.object_id)} />
                    {item.image_urls[0] && (
                      <button
                        type="button"
                        className="object-image-button"
                        aria-label={`View ${item.object_no || item.object_type} image and details`}
                        onClick={() => setPreviewObject(item)}
                      >
                        <img src={item.image_urls[0]} alt={`${item.object_type} on page ${item.page_no}`} />
                      </button>
                    )}
                    <div className="min-w-0">
                      <div className="flex justify-between gap-2">
                        <label htmlFor={`object-${item.object_id}`}><strong>{item.object_no || item.object_type}</strong></label>
                        <span className="status-pill">{item.object_type}</span>
                      </div>
                      <p className="text-xs text-slate-500">{item.pdf_name} · Page {item.page_no}</p>
                      <p className="object-summary">{objectSummary(item)}</p>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          {previewObject && (
            <div className="object-preview-backdrop" onClick={() => setPreviewObject(null)}>
              <section
                className="object-preview-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="object-preview-title"
                onClick={(event) => event.stopPropagation()}
              >
                <header className="object-preview-header">
                  <div>
                    <p className="eyebrow">Detected object</p>
                    <h2 id="object-preview-title">{previewObject.object_no || previewObject.object_type}</h2>
                  </div>
                  <button
                    type="button"
                    className="object-preview-close"
                    aria-label="Close object preview"
                    onClick={() => setPreviewObject(null)}
                  >
                    <X size={20} />
                  </button>
                </header>
                {previewObject.image_urls[0] && (
                  <div className="object-preview-image-wrap">
                    <img
                      src={previewObject.image_urls[0]}
                      alt={`${previewObject.object_type} from ${previewObject.pdf_name}, page ${previewObject.page_no}`}
                    />
                  </div>
                )}
                <div className="object-preview-information">
                  <div className="object-preview-metadata">
                    <span><strong>Type</strong>{previewObject.object_type}</span>
                    <span><strong>PDF</strong>{previewObject.pdf_name}</span>
                    <span><strong>Page</strong>{previewObject.page_no}</span>
                    {previewObject.position && <span><strong>Position</strong>{previewObject.position}</span>}
                    {previewObject.title && <span><strong>Title</strong>{previewObject.title}</span>}
                  </div>
                  <div>
                    <h3>Summary</h3>
                    <p>{objectSummary(previewObject)}</p>
                  </div>
                  {Object.keys(previewObject.characteristics).length > 0 && (
                    <div>
                      <h3>Additional information</h3>
                      <dl className="object-preview-characteristics">
                        {Object.entries(previewObject.characteristics).map(([key, value]) => (
                          <div key={key}>
                            <dt>{key.replace(/_/g, ' ')}</dt>
                            <dd>{typeof value === 'string' ? value : JSON.stringify(value)}</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {textSelected && !sessionHasText && (
            <p className="alert-warning" role="status">
              Text was not selected when this analysis ran. Select Text before analyzing to extract, chunk, and generate text Q&amp;A.
            </p>
          )}

          {canGenerateText && <section className="panel space-y-4">
            <div className="section-title"><div><h2>Text chunk preview</h2><p>{session.text_chunks.length} extracted prose chunks. Apply new settings without analyzing PDFs again.</p></div></div>
            <div className="grid gap-3 md:grid-cols-2">
              {session.text_chunks.slice(0, 4).map((chunk, index) => (
                <article className="chunk-preview" key={String(chunk.chunk_id ?? index)}>
                  <div className="flex justify-between gap-3">
                    <strong>{String(chunk.pdf_name ?? 'PDF')} · {String(chunk.section ?? 'Document text')}</strong>
                    <span className="text-xs text-slate-500">Page {String(chunk.page_no ?? '—')}</span>
                  </div>
                  <p>{String(chunk.source_text ?? '')}</p>
                  {String(chunk.source_text ?? '').length > 320 && (
                    <details className="chunk-preview-details">
                      <summary>Show full chunk</summary>
                      <p>{String(chunk.source_text ?? '')}</p>
                    </details>
                  )}
                </article>
              ))}
            </div>
          </section>}

          <section className="panel qa-generation-panel">
            <div className="section-title">
              <div className="icon-tile"><Sparkles size={18} /></div>
              <div>
                <h2>Generate Q&amp;A</h2>
                <p>Choose how text questions are created{selectedIds.length ? ' and configure selected objects' : ''}.</p>
              </div>
            </div>

            {selectedIds.length > 0 && (
              <div className="qa-object-settings">
                <div>
                  <strong>Selected objects</strong>
                  <span>{selectedIds.length} object{selectedIds.length === 1 ? '' : 's'} selected</span>
                </div>
                <label className="qa-number-field compact-field">
                  <span>Questions per object</span>
                  <input type="number" min={1} max={20} step={1} value={objectCount} onChange={(event) => setObjectCount(Number(event.target.value))} />
                </label>
              </div>
            )}

            {canGenerateText && (
              <div className="qa-mode-section">
                <div className="qa-mode-section-heading">
                  <div>
                    <h3>Text generation modes</h3>
                    <p>Select either approach, or combine both in one run.</p>
                  </div>
                  <span className="qa-mode-selection-hint">Choose one or both</span>
                </div>
                <div className="qa-mode-grid">
                  <label className={`qa-mode-card ${generateSingleChunk ? 'is-selected' : ''}`}>
                    <input
                      type="checkbox"
                      checked={generateSingleChunk}
                      onChange={(event) => setGenerateSingleChunk(event.target.checked)}
                    />
                    <span className="qa-mode-icon"><Layers size={19} /></span>
                    <span className="qa-mode-copy">
                      <strong>Single chunk</strong>
                      <small>Generate questions from each chunk on its own.</small>
                    </span>
                    <span className="qa-mode-check"><Check size={15} /></span>
                  </label>
                  <label className={`qa-mode-card ${generateMultiHop ? 'is-selected' : ''}`}>
                    <input
                      type="checkbox"
                      checked={generateMultiHop}
                      onChange={(event) => setGenerateMultiHop(event.target.checked)}
                    />
                    <span className="qa-mode-icon qa-mode-icon-multihop"><GitBranch size={19} /></span>
                    <span className="qa-mode-copy">
                      <strong>Multi-hop</strong>
                      <small>Combine adjacent chunks for questions that need both.</small>
                    </span>
                    <span className="qa-mode-check"><Check size={15} /></span>
                  </label>
                </div>

                {(generateSingleChunk || generateMultiHop) ? (
                  <div className="qa-settings-grid">
                    {generateSingleChunk && (
                      <div className="qa-config-card">
                        <div className="qa-config-heading">
                          <span className="qa-config-number">01</span>
                          <div><strong>Single-chunk settings</strong><small>Generated independently for each chunk</small></div>
                        </div>
                        <label className="qa-number-field">
                          <span>Q&amp;A pairs per chunk</span>
                          <input type="number" min={1} max={3} step={1} value={textCount} onChange={(event) => setTextCount(Number(event.target.value))} />
                        </label>
                      </div>
                    )}
                    {generateMultiHop && (
                      <div className="qa-config-card qa-config-card-multihop">
                        <div className="qa-config-heading">
                          <span className="qa-config-number">02</span>
                          <div><strong>Multi-hop settings</strong><small>Uses same-document adjacent chunks</small></div>
                        </div>
                        <div className="qa-multihop-fields">
                          <label className="qa-number-field">
                            <span>Chunks per group</span>
                            <input type="number" min={2} max={10} step={1} value={multiHopChunkCount} onChange={(event) => setMultiHopChunkCount(Number(event.target.value))} />
                          </label>
                          <label className="qa-number-field">
                            <span>Q&amp;A pairs per group</span>
                            <input type="number" min={1} max={3} step={1} value={multiHopQuestionCount} onChange={(event) => setMultiHopQuestionCount(Number(event.target.value))} />
                          </label>
                        </div>
                        <p className="qa-config-help">Each group advances by one chunk. Groups never cross PDF boundaries.</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="qa-mode-empty">Select at least one text mode to generate text Q&amp;A.</p>
                )}
              </div>
            )}

            {canGenerateText && hasPendingChunking && <p className="alert-warning">Apply the changed chunk settings before generating text Q&amp;A.</p>}
            <div className="qa-generation-actions">
              <button className="button-primary" onClick={runGeneration}
                disabled={!hasAzureCredentials(generationCredentials) || !validQuestionCounts || !!busy || (!selectedIds.length && (!canGenerateText || (!generateSingleChunk && !generateMultiHop))) || (canGenerateText && session.text_chunks.length > 0 && hasPendingChunking)}>
                {busy === 'qa' ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />}
                {busy === 'qa' ? 'Generating Q&A…' : 'Generate Q&A'}
              </button>
              {!hasAzureCredentials(generationCredentials) && <span>Configure Azure OpenAI credentials to start generation.</span>}
            </div>
            {generationProgress && (
              <div className={`qa-progress-card qa-progress-${generationProgress.status}`} role="status" aria-live="polite">
                <div className="qa-progress-icon">
                  {generationProgress.status === 'running' || generationProgress.status === 'finalizing'
                    ? <LoaderCircle size={20} className="animate-spin" />
                    : generationProgress.status === 'complete'
                      ? <Check size={20} />
                      : <X size={20} />}
                </div>
                <div className="qa-progress-content">
                  <div className="qa-progress-heading">
                    <div>
                      <span className="qa-progress-eyebrow">
                        {generationProgress.status === 'running'
                          ? 'Generation in progress'
                          : generationProgress.status === 'finalizing'
                            ? 'Saving results'
                            : `Generation ${generationProgress.status}`}
                      </span>
                      <p>{generationProgress.message}</p>
                    </div>
                    <strong>{generationProgress.progress}%</strong>
                  </div>
                  <div
                    className="qa-progress-track"
                    role="progressbar"
                    aria-label="Q&A generation progress"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={generationProgress.progress}
                  >
                    <span style={{ width: `${generationProgress.progress}%` }} />
                  </div>
                  <div className="qa-progress-footer">
                    <span>{generationProgress.total ? `${generationProgress.completed} of ${generationProgress.total} steps` : 'Preparing generation steps'}</span>
                    <span>Progress updates as each object, chunk, or group finishes</span>
                  </div>
                </div>
              </div>
            )}
            {session.qa_records.length > 0 && (
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead><tr><th>Source</th><th>Page</th><th>Question</th><th>Expected answer</th></tr></thead>
                  <tbody>{session.qa_records.map((row, index) => (
                    <tr key={`${row.source_type}-${index}`}>
                      <td><span className="status-pill">{row.object_type || row.source_type}</span><div className="text-xs text-slate-500">{row.pdf_name}</div></td>
                      <td>{row.page_no ?? '—'}</td>
                      <td>{row.question}</td>
                      <td>{row.expected_answer}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
