import { useEffect, useMemo, useState } from 'react';
import { Download, FileSearch, LoaderCircle, RefreshCw, Sparkles } from 'lucide-react';
import AzureCredentialFields from '../components/AzureCredentialFields';
import {
  analyzeWorkbench,
  downloadBase64File,
  generateWorkbenchQA,
  getWorkbenchSession,
  getWorkbenchWorkbook,
  updateWorkbenchChunking,
} from '../api/workbench';
import type { ChunkingOptions, WorkbenchObject, WorkbenchSession } from '../api/workbench';
import { hasAzureCredentials, useAzureCredentials } from '../contexts/AzureCredentialsContext';

const SESSION_KEY = 'react_pdf_workbench_session';
const DEFAULT_CHUNKING: ChunkingOptions = {
  technique: 'heading_aware',
  chunk_size: 2600,
  overlap: 250,
  min_sentences_per_chunk: 1,
  semantic_threshold: 0.8,
};

function objectSummary(item: WorkbenchObject): string {
  const details = item.characteristics;
  return String(details.content_summary ?? details.description ?? item.title ?? 'No summary available.');
}

export default function PdfWorkbenchPage() {
  const { credentials } = useAzureCredentials();
  const [session, setSession] = useState<WorkbenchSession | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState('crop');
  const [chunking, setChunking] = useState(DEFAULT_CHUNKING);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [generateObjects, setGenerateObjects] = useState(true);
  const [generateText, setGenerateText] = useState(true);
  const [objectCount, setObjectCount] = useState(5);
  const [textCount, setTextCount] = useState(2);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    const savedId = sessionStorage.getItem(SESSION_KEY);
    if (!savedId) return;
    getWorkbenchSession(savedId)
      .then((current) => {
        setSession(current);
        setChunking(current.chunking);
        setSelectedIds([]);
      })
      .catch((cause) => {
        sessionStorage.removeItem(SESSION_KEY);
        setError(
          cause instanceof Error
            ? `Could not restore the previous Workbench session: ${cause.message}`
            : 'Could not restore the previous Workbench session.',
        );
      });
  }, []);

  const hasPendingChunking = useMemo(
    () => session !== null && JSON.stringify(chunking) !== JSON.stringify(session.chunking),
    [chunking, session],
  );
  const supportsOverlap = chunking.technique === 'heading_aware'
    || chunking.technique === 'sentence'
    || chunking.technique === 'token';

  const setTechnique = (technique: ChunkingOptions['technique']) => {
    setChunking({
      ...DEFAULT_CHUNKING,
      technique,
      chunk_size: technique === 'heading_aware' ? 2600 : 512,
      overlap: technique === 'heading_aware' ? 250 : 50,
    });
  };

  const runAnalysis = async () => {
    setBusy('analysis');
    setError('');
    setNotice('');
    try {
      const current = await analyzeWorkbench(files, mode, chunking, credentials);
      setSession(current);
      sessionStorage.setItem(SESSION_KEY, current.session_id);
      setSelectedIds([]);
      setNotice(`Analysis complete: ${current.objects.length} objects and ${current.text_chunks.length} text chunks.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'PDF analysis failed.');
    } finally {
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
    try {
      const current = await generateWorkbenchQA(session.session_id, {
        object_ids: selectedIds,
        generate_objects: generateObjects,
        generate_text: generateText,
        object_question_count: objectCount,
        text_question_count: textCount,
        ...credentials,
      });
      setSession(current);
      setNotice(`Q&A generated: ${current.qa_records.length} question-and-answer pairs.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Q&A generation failed.');
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

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Document intelligence</p>
          <h1>PDF Workbench</h1>
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
            <label className="field-label">Vision context
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="crop">Object crop</option>
                <option value="page">Crop + full page</option>
                <option value="page_context">Crop + neighboring pages</option>
              </select>
            </label>
            <label className="field-label">Text chunking technique
              <select value={chunking.technique} onChange={(event) => setTechnique(event.target.value as ChunkingOptions['technique'])}>
                <option value="heading_aware">Heading-aware</option>
                <option value="recursive">Recursive</option>
                <option value="sentence">Sentence</option>
                <option value="token">Token</option>
                <option value="semantic">Semantic</option>
              </select>
            </label>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="field-label">{chunking.technique === 'heading_aware' ? 'Max characters' : 'Max tokens'}
            <input type="number" min={64} max={12000} value={chunking.chunk_size}
              onChange={(event) => setChunking({ ...chunking, chunk_size: Number(event.target.value) })} />
          </label>
          {supportsOverlap && <label className="field-label">Overlap
            <input type="number" min={0} max={4096} value={chunking.overlap}
              onChange={(event) => setChunking({ ...chunking, overlap: Number(event.target.value) })} />
          </label>}
          {chunking.technique === 'sentence' && <label className="field-label">Minimum sentences
            <input type="number" min={1} max={20} value={chunking.min_sentences_per_chunk}
              onChange={(event) => setChunking({ ...chunking, min_sentences_per_chunk: Number(event.target.value) })} />
          </label>}
          {chunking.technique === 'semantic' && <label className="field-label">Semantic threshold
            <input type="number" min={0.05} max={0.95} step={0.05} value={chunking.semantic_threshold}
              onChange={(event) => setChunking({ ...chunking, semantic_threshold: Number(event.target.value) })} />
          </label>}
        </div>
        <details className="credential-disclosure">
          <summary>Azure OpenAI credentials <span>Required for analysis and Q&amp;A; kept in memory only</span></summary>
          <div className="pt-4"><AzureCredentialFields /></div>
        </details>
        <div className="flex flex-wrap gap-3">
          <button className="button-primary" onClick={runAnalysis} disabled={!files.length || !hasAzureCredentials(credentials) || !!busy}>
            {busy === 'analysis' ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />}
            {busy === 'analysis' ? 'Analyzing PDFs…' : 'Analyze PDFs'}
          </button>
          {session && (
            <button className="button-secondary" onClick={applyChunking} disabled={!hasPendingChunking || !!busy}>
              {busy === 'chunking' ? <LoaderCircle className="animate-spin" size={16} /> : <RefreshCw size={16} />}
              Apply chunking settings
            </button>
          )}
        </div>
      </section>

      {error && <div className="alert-error" role="alert">{error}</div>}
      {notice && <div className="alert-success" role="status">{notice}</div>}

      {session && (
        <>
          {session.text_errors.length > 0 && (
            <div className="alert-error">
              {session.text_errors.map((item) => <p key={item.pdf_name}>{item.pdf_name}: {item.error}</p>)}
            </div>
          )}
          <section className="panel">
            <div className="section-title">
              <div><h2>Detected objects</h2><p>Choose objects directly from this list for object Q&amp;A.</p></div>
              <span className="status-pill">{selectedIds.length} selected</span>
            </div>
            {session.objects.length === 0 ? (
              <p className="empty-state">No figures, tables, or images were detected in this analysis.</p>
            ) : (
              <div className="object-grid">
                {session.objects.map((item) => (
                  <label className={`object-card ${selectedIds.includes(item.object_id) ? 'object-card-selected' : ''}`} key={item.object_id}>
                    <input type="checkbox" checked={selectedIds.includes(item.object_id)} onChange={() => toggleObject(item.object_id)} />
                    {item.image_urls[0] && <img src={item.image_urls[0]} alt={`${item.object_type} on page ${item.page_no}`} />}
                    <div className="min-w-0">
                      <div className="flex justify-between gap-2">
                        <strong>{item.object_no || item.object_type}</strong>
                        <span className="status-pill">{item.object_type}</span>
                      </div>
                      <p className="text-xs text-slate-500">{item.pdf_name} · Page {item.page_no}</p>
                      <p className="object-summary">{objectSummary(item)}</p>
                    </div>
                  </label>
                ))}
              </div>
            )}
          </section>

          <section className="panel space-y-4">
            <div className="section-title"><div><h2>Text chunk preview</h2><p>{session.text_chunks.length} extracted prose chunks. Apply new settings without analyzing PDFs again.</p></div></div>
            <div className="grid gap-3 md:grid-cols-2">
              {session.text_chunks.slice(0, 4).map((chunk, index) => (
                <article className="chunk-preview" key={String(chunk.chunk_id ?? index)}>
                  <div className="flex justify-between gap-3">
                    <strong>{String(chunk.pdf_name ?? 'PDF')} · {String(chunk.section ?? 'Document text')}</strong>
                    <span className="text-xs text-slate-500">Page {String(chunk.page_no ?? '—')}</span>
                  </div>
                  <p>{String(chunk.source_text ?? '')}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel space-y-4">
            <div className="section-title"><div className="icon-tile"><Sparkles size={18} /></div><div><h2>Generate Q&amp;A</h2><p>Object Q&amp;A uses checked objects; text Q&amp;A uses all chunks.</p></div></div>
            <div className="flex flex-wrap gap-x-6 gap-y-3">
              <label className="check-label"><input type="checkbox" checked={generateObjects} onChange={(event) => setGenerateObjects(event.target.checked)} /> Objects</label>
              <label className="check-label"><input type="checkbox" checked={generateText} onChange={(event) => setGenerateText(event.target.checked)} /> Document text</label>
              <label className="field-label compact-field">Questions per object
                <input type="number" min={1} max={20} value={objectCount} onChange={(event) => setObjectCount(Number(event.target.value))} />
              </label>
              <label className="field-label compact-field">Questions per text chunk
                <input type="number" min={1} max={3} value={textCount} onChange={(event) => setTextCount(Number(event.target.value))} />
              </label>
            </div>
            {hasPendingChunking && <p className="alert-warning">Apply the changed chunk settings before generating text Q&amp;A.</p>}
            <button className="button-primary" onClick={runGeneration}
              disabled={!hasAzureCredentials(credentials) || !!busy || (!generateObjects && !generateText) || (generateObjects && !selectedIds.length) || (generateText && (!session.text_chunks.length || hasPendingChunking))}>
              {busy === 'qa' ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />}
              {busy === 'qa' ? 'Generating Q&A…' : 'Generate selected Q&A'}
            </button>
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
