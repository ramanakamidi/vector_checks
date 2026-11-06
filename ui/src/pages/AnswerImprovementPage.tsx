import { useState } from 'react';
import { CheckCircle2, Download, FileCheck2, LoaderCircle, ShieldAlert } from 'lucide-react';
import AzureCredentialFields from '../components/AzureCredentialFields';
import { downloadBase64File, reviewAnswers } from '../api/workbench';
import { hasAzureCredentials, useAzureCredentials } from '../contexts/AzureCredentialsContext';
import { useProject } from '../contexts/ProjectContext';

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

export default function AnswerImprovementPage() {
  const { activeProject } = useProject();
  const { credentials } = useAzureCredentials();
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<{
    filename: string;
    content_base64: string;
    summary: Record<string, unknown>;
  } | null>(null);

  const runReview = async () => {
    if (!activeProject || !file) return;
    setBusy(true);
    setError('');
    setResult(null);
    try {
      setResult(await reviewAnswers(file, activeProject.id, topK, credentials));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Answer review failed.');
    } finally {
      setBusy(false);
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
          <p>Check expected answers against retrieved project evidence and return an annotated workbook.</p>
        </div>
      </header>

      <section className="panel space-y-5">
        <div className="section-title">
          <div className="icon-tile"><FileCheck2 size={19} /></div>
          <div><h2>Review a Q&amp;A workbook</h2><p>Text answers use project Chroma context; visual rows use the workbook’s cropped image references.</p></div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="dropzone">
            <input type="file" accept=".xlsx" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <FileCheck2 size={24} className="text-indigo-600" />
            <span className="font-semibold">{file?.name ?? 'Choose the generated Q&A workbook'}</span>
            <span className="text-sm text-slate-500">Excel .xlsx files</span>
          </label>
          <label className="field-label">Initial retrieved chunks (k)
            <input type="number" min={1} max={8} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            <span className="field-help">Retrieval expands up to 8 chunks when evidence is insufficient.</span>
          </label>
        </div>
        <details className="credential-disclosure" open>
          <summary>Azure OpenAI reviewer credentials</summary>
          <div className="pt-4"><AzureCredentialFields /></div>
        </details>
        <button className="button-primary" onClick={runReview}
          disabled={!file || !activeProject || !hasAzureCredentials(credentials) || busy}>
          {busy ? <LoaderCircle size={16} className="animate-spin" /> : <ShieldAlert size={16} />}
          {busy ? 'Reviewing workbook…' : 'Review expected answers'}
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
