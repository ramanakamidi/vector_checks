import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2, Download, FileCheck2, LoaderCircle, ShieldAlert } from 'lucide-react';
import { downloadBase64File, reviewAnswers } from '../api/workbench';
import { hasAzureCredentials, useAzureCredentials } from '../contexts/AzureCredentialsContext';
import { useProject } from '../contexts/ProjectContext';

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

export default function AnswerImprovementPage() {
  const { activeProject } = useProject();
  const { credentials } = useAzureCredentials();
  const projectId = activeProject?.id ?? '';
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;
  const reviewRequestIdRef = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<{
    filename: string;
    content_base64: string;
    summary: Record<string, unknown>;
  } | null>(null);

  useEffect(() => {
    reviewRequestIdRef.current += 1;
    setBusy(false);
    setError('');
    setResult(null);
    return () => {
      reviewRequestIdRef.current += 1;
    };
  }, [projectId]);

  const runReview = async () => {
    if (!activeProject || !file) return;
    const requestProjectId = activeProject.id;
    const requestId = ++reviewRequestIdRef.current;
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const reviewResult = await reviewAnswers(file, requestProjectId, topK, credentials);
      if (
        requestId === reviewRequestIdRef.current
        && projectIdRef.current === requestProjectId
      ) {
        setResult(reviewResult);
      }
    } catch (cause) {
      if (
        requestId === reviewRequestIdRef.current
        && projectIdRef.current === requestProjectId
      ) {
        setError(cause instanceof Error ? cause.message : 'Answer review failed.');
      }
    } finally {
      if (
        requestId === reviewRequestIdRef.current
        && projectIdRef.current === requestProjectId
      ) {
        setBusy(false);
      }
    }
  };

  const summary = result?.summary;
  const errorDetails = Array.isArray(summary?.error_details) ? summary.error_details : [];

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Grounded answer review</p>
          <h1>Answer Improvement</h1>
          <p>Text and table rows review the AUT answer and expected answer against project evidence, using drift notes when available. Image and graph rows keep using their cropped images.</p>
        </div>
      </header>

      <section className="panel space-y-5">
        <div className="section-title">
          <div className="icon-tile"><FileCheck2 size={19} /></div>
          <div><h2>Review a Q&amp;A workbook</h2><p>Text and table rows retrieve from the question, expanding up to 8 chunks when needed. Visual rows use the workbook’s cropped image references.</p></div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="dropzone">
            <input type="file" accept=".xlsx" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <FileCheck2 size={24} className="text-indigo-600" />
            <span className="font-semibold">{file?.name ?? 'Choose a Q&A workbook'}</span>
            <span className="text-sm text-slate-500">Excel .xlsx files</span>
          </label>
          <label className="field-label">Initial retrieved chunks (k)
            <input type="number" min={1} max={8} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            <span className="field-help">Retrieval expands up to 8 chunks when evidence is insufficient.</span>
          </label>
        </div>
        {!hasAzureCredentials(credentials) && (
          <p className="field-help">
            Configure Azure OpenAI credentials in <Link className="text-blue-700 underline" to="/app-under-test">Configure</Link> before reviewing answers.
          </p>
        )}
        <button className="button-primary" onClick={runReview}
          disabled={!file || !activeProject || !hasAzureCredentials(credentials) || busy}>
          {busy ? <LoaderCircle size={16} className="animate-spin" /> : <ShieldAlert size={16} />}
          {busy ? 'Reviewing workbook…' : 'Review answers'}
        </button>
      </section>

      {error && <div className="alert-error" role="alert">{error}</div>}
      {result && summary && (
        <section className="panel space-y-4">
          <div className="section-title">
            <div className="icon-tile icon-tile-success"><CheckCircle2 size={19} /></div>
            <div>
              <h2>Review complete</h2>
              <p>{String(summary.processed ?? 0)} questions reviewed for <strong>{activeProject?.name}</strong>.</p>
            </div>
          </div>
          <div className="metric-grid">
            <div className="metric-card"><strong>{String(summary.improvement_required ?? 0)}</strong><span>Need improvement</span></div>
            <div className="metric-card"><strong>{String(summary.no_improvement ?? 0)}</strong><span>Complete</span></div>
            <div className="metric-card"><strong>{String(summary.unable_to_verify ?? 0)}</strong><span>Unable to verify</span></div>
            <div className="metric-card"><strong>{String(summary.errors ?? 0)}</strong><span>Row errors</span></div>
          </div>
          {errorDetails.length > 0 && (
            <div className="alert-warning">
              <p>Some rows could not be reviewed; they are marked ERROR in the workbook.</p>
              <ul>{errorDetails.map((entry, index) => <li key={index}>{JSON.stringify(entry)}</li>)}</ul>
            </div>
          )}
          <button className="button-primary" onClick={() => downloadBase64File(result.filename, result.content_base64, XLSX_MIME)}>
            <Download size={16} /> Download reviewed workbook
          </button>
        </section>
      )}
    </div>
  );
}
