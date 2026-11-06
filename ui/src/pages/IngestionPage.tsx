import { useState, useEffect, useCallback, useRef } from 'react';
import { Upload, FileJson, Database, Trash2, Loader2, CheckCircle, AlertCircle, X, Info, Clock, Tag, ChevronDown, ChevronRight, BarChart3, Search } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import type { QARecord, IngestResponse, IngestionStats, PaginatedRecords } from '../api/types';
import { uploadJsonFile, ingestData, getIngestedData, deleteIngestedData, getIngestionStats } from '../api/client';
import DataTable from '../components/shared/DataTable';

interface Toast { id: number; type: 'success' | 'error' | 'info'; message: string; }
let toastId = 0;

function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="fixed top-4 right-4 z-[100] space-y-2">
      {toasts.map((t) => (
        <div key={t.id} className={`flex items-center gap-2 px-4 py-3 rounded-lg shadow-lg text-sm font-medium max-w-md ${t.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' : t.type === 'error' ? 'bg-red-50 text-red-800 border border-red-200' : 'bg-blue-50 text-blue-800 border border-blue-200'}`}>
          {t.type === 'success' ? <CheckCircle size={16} /> : t.type === 'error' ? <AlertCircle size={16} /> : <Info size={16} />}
          <span className="flex-1">{t.message}</span>
          <button onClick={() => onDismiss(t.id)} className="text-current opacity-50 hover:opacity-100"><X size={14} /></button>
        </div>
      ))}
    </div>
  );
}

function RecordModal({ record, onClose }: { record: QARecord; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800">Record Details</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600"><X size={20} /></button>
        </div>
        <div className="px-6 py-4 space-y-4">
          <Field label="ID" value={record.id} mono />
          <Field label="Question" value={record.question} />
          <Field label="Answer" value={record.answer} />
          <Field label="Context" value={record.context} scroll />
          {record.metadata && Object.keys(record.metadata).length > 0 && (
            <div>
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Metadata</label>
              <div className="bg-slate-50 px-3 py-2 rounded-lg">
                <table className="w-full text-sm"><tbody>
                  {Object.entries(record.metadata).map(([k, v]) => (
                    <tr key={k} className="border-b border-slate-100 last:border-0">
                      <td className="py-1.5 pr-4 font-medium text-slate-600 whitespace-nowrap">{k}</td>
                      <td className="py-1.5 text-slate-700 break-all">{String(v)}</td>
                    </tr>
                  ))}
                </tbody></table>
              </div>
            </div>
          )}
        </div>
        <div className="px-6 py-3 border-t border-slate-200 flex justify-end">
          <button onClick={onClose} className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors">Close</button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, mono, scroll }: { label: string; value?: string; mono?: boolean; scroll?: boolean }) {
  return (
    <div>
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">{label}</label>
      <p className={`text-sm text-slate-700 bg-slate-50 px-3 py-2 rounded-lg whitespace-pre-wrap ${mono ? 'font-mono break-all' : ''} ${scroll ? 'max-h-60 overflow-y-auto' : ''}`}>{value || '—'}</p>
    </div>
  );
}

function ConfirmClear({ onConfirm, onCancel, loading }: { onConfirm: () => void; onCancel: () => void; loading: boolean }) {
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onCancel}>
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full" onClick={(e) => e.stopPropagation()}>
        <div className="px-6 py-4 border-b border-slate-200"><h3 className="text-lg font-semibold text-slate-800">Clear All Ingested Data</h3></div>
        <div className="px-6 py-4"><p className="text-sm text-slate-600">This will permanently delete all ingested records from the Chroma vector store and the JSON backup. This action cannot be undone.</p></div>
        <div className="px-6 py-3 border-t border-slate-200 flex justify-end gap-2">
          <button onClick={onCancel} disabled={loading} className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 disabled:opacity-50">Cancel</button>
          <button onClick={onConfirm} disabled={loading} className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50">
            {loading && <Loader2 size={14} className="animate-spin" />}{loading ? 'Clearing...' : 'Clear All Data'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function IngestionPage() {
  const { activeProject } = useProject();
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [pasteJson, setPasteJson] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<IngestResponse | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'paste'>('file');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [records, setRecords] = useState<QARecord[]>([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [stats, setStats] = useState<IngestionStats | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<QARecord | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(true);
  const pageSize = 50;

  const toast = useCallback((type: 'success' | 'error' | 'info', msg: string) => {
    const id = ++toastId;
    setToasts((p) => [...p, { id, type, message: msg }]);
    setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 5000);
  }, []);

  const loadRecords = useCallback(async () => {
    if (!activeProject) return;
    setLoadingRecords(true);
    try {
      const d: PaginatedRecords = await getIngestedData(activeProject.id, currentPage, pageSize, searchTerm);
      setRecords(d.records); setTotalRecords(d.total);
    } catch (e: any) { toast('error', `Failed to load records: ${e.message}`); }
    finally { setLoadingRecords(false); }
  }, [activeProject, currentPage, searchTerm, toast]);

  const loadStats = useCallback(async () => {
    if (!activeProject) return;
    try { setStats(await getIngestionStats(activeProject.id)); } catch { setStats(null); }
  }, [activeProject]);

  useEffect(() => { if (activeProject) { loadRecords(); loadStats(); } }, [activeProject?.id, currentPage, searchTerm]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.json')) { setSelectedFile(f); setInputMode('file'); } else { toast('error', 'Please drop a .json file'); }
  }, [toast]);

  const handleIngest = useCallback(async () => {
    if (!activeProject) return;
    setUploading(true); setUploadResult(null);
    try {
      let result: IngestResponse;
      if (inputMode === 'file' && selectedFile) {
        result = await uploadJsonFile(activeProject.id, selectedFile);
      } else if (inputMode === 'paste' && pasteJson.trim()) {
        let parsed: any;
        try { parsed = JSON.parse(pasteJson); } catch { toast('error', 'Invalid JSON'); setUploading(false); return; }
        result = await ingestData(activeProject.id, Array.isArray(parsed) ? parsed : parsed.data ?? [parsed]);
      } else { toast('error', 'Please select a file or paste JSON'); setUploading(false); return; }
      setUploadResult(result);
      if (result.status === 'success') {
        toast('success', `Ingested ${result.records_ingested} records`);
        if (result.errors.length > 0) toast('info', `${result.errors.length} warnings`);
        setSelectedFile(null); setPasteJson('');
        if (fileInputRef.current) fileInputRef.current.value = '';
        setCurrentPage(1); await loadRecords(); await loadStats();
      } else { toast('error', `Failed: ${result.errors.join(', ')}`); }
    } catch (e: any) { toast('error', e.message); }
    finally { setUploading(false); }
  }, [activeProject, inputMode, selectedFile, pasteJson, toast, loadRecords, loadStats]);

  const handleClear = useCallback(async () => {
    if (!activeProject) return;
    setClearing(true);
    try {
      await deleteIngestedData(activeProject.id);
      toast('success', 'All ingested data cleared');
      setShowClearConfirm(false); setRecords([]); setTotalRecords(0); setStats(null); setUploadResult(null);
    } catch (e: any) { toast('error', e.message); }
    finally { setClearing(false); }
  }, [activeProject, toast]);

  const trunc = (s: string | undefined | null, n = 80) => (!s ? '—' : s.length > n ? s.slice(0, n) + '…' : s);
  const totalPages = Math.ceil(totalRecords / pageSize);

  const columns = [
    { key: 'id', label: 'ID', render: (v: any) => <span className="font-mono text-xs text-slate-500">{trunc(v, 16)}</span> },
    { key: 'question', label: 'Question', render: (v: any) => <span>{trunc(v, 60)}</span> },
    { key: 'answer', label: 'Answer', render: (v: any) => <span className="text-slate-600">{trunc(v, 60)}</span> },
    { key: 'context', label: 'Context', render: (v: any) => <span className="text-slate-500 text-xs">{trunc(v, 50)}</span> },
    { key: 'source', label: 'Source', render: (_v: any, row: any) => <span className="text-xs text-slate-500">{(row as QARecord).metadata?.source || '—'}</span> },
  ];

  if (!activeProject) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Upload size={20} className="text-cognizant-blue" /></div>
          <div><h2 className="text-xl font-bold text-slate-800">Document Ingestion</h2><p className="text-sm text-slate-500">Upload Q&amp;A data as JSON and ingest into the vector store</p></div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <Info size={48} className="mx-auto text-slate-300 mb-4" />
          <h3 className="text-lg font-semibold text-slate-600 mb-2">No Project Selected</h3>
          <p className="text-sm text-slate-400">Select or create a project using the project selector in the top navigation bar.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((p) => p.filter((t) => t.id !== id))} />
      {selectedRecord && <RecordModal record={selectedRecord} onClose={() => setSelectedRecord(null)} />}
      {showClearConfirm && <ConfirmClear onConfirm={handleClear} onCancel={() => setShowClearConfirm(false)} loading={clearing} />}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Upload size={20} className="text-cognizant-blue" /></div>
          <div><h2 className="text-xl font-bold text-slate-800">Document Ingestion</h2><p className="text-sm text-slate-500">Upload Q&amp;A data as JSON and ingest into the vector store</p></div>
        </div>
        {totalRecords > 0 && (
          <button onClick={() => setShowClearConfirm(true)} className="flex items-center gap-2 px-3 py-2 text-red-600 bg-red-50 border border-red-200 rounded-lg text-sm font-medium hover:bg-red-100 transition-colors">
            <Trash2 size={14} />Clear All Data
          </button>
        )}
      </div>

      {/* Stats */}
      {stats && stats.total_records > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard icon={<Database size={14} />} label="Total Records" value={stats.total_records.toLocaleString()} />
          <StatCard icon={<Tag size={14} />} label="Metadata Fields" value={String(stats.metadata_fields.length)} sub={stats.metadata_fields.join(', ')} />
          <StatCard icon={<BarChart3 size={14} />} label="Collection" value={stats.collections[0] || '—'} small />
          <StatCard icon={<Clock size={14} />} label="Last Ingested" value={stats.ingested_at ? new Date(stats.ingested_at).toLocaleString() : '—'} small />
        </div>
      )}

      {/* Upload Panel */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <button onClick={() => setUploadOpen(!uploadOpen)} className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors">
          <div className="text-left">
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Upload Q&amp;A Data</h3>
            <p className="text-xs text-slate-400 mt-0.5">Provide a JSON file with pre-extracted Q&amp;A pairs, context, and metadata</p>
          </div>
          {uploadOpen ? <ChevronDown size={18} className="text-slate-400" /> : <ChevronRight size={18} className="text-slate-400" />}
        </button>
        {uploadOpen && (
          <div className="px-6 pb-6 border-t border-slate-100 pt-4">
            <div className="flex gap-1 mb-4 bg-slate-100 rounded-lg p-1 w-fit">
              <button onClick={() => setInputMode('file')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${inputMode === 'file' ? 'bg-white text-cognizant-blue shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>File Upload</button>
              <button onClick={() => setInputMode('paste')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${inputMode === 'paste' ? 'bg-white text-cognizant-blue shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Paste JSON</button>
            </div>
            {inputMode === 'file' ? (
              <div onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleFileDrop} onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${dragOver ? 'border-cognizant-blue bg-blue-50' : selectedFile ? 'border-green-300 bg-green-50' : 'border-slate-300 hover:border-cognizant-blue/50'}`}>
                <input ref={fileInputRef} type="file" accept=".json" onChange={(e) => { const f = e.target.files?.[0]; if (f) { setSelectedFile(f); setInputMode('file'); } }} className="hidden" />
                {selectedFile ? (<><FileJson size={40} className="mx-auto text-green-500 mb-3" /><p className="text-sm font-medium text-green-700 mb-1">{selectedFile.name}</p><p className="text-xs text-green-600">{(selectedFile.size / 1024).toFixed(1)} KB — Click to change</p></>) : (<><FileJson size={40} className="mx-auto text-slate-400 mb-3" /><p className="text-sm font-medium text-slate-600 mb-1">Drop JSON file here or click to browse</p><p className="text-xs text-slate-400">Expected: array of objects with question, answer, context, metadata</p></>)}
              </div>
            ) : (
              <div>
                <textarea value={pasteJson} onChange={(e) => setPasteJson(e.target.value)} placeholder={'[\n  { "question": "...", "answer": "...", "context": "...", "metadata": {} }\n]'} className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm font-mono focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none resize-none" rows={8} />
                <p className="text-xs text-slate-400 mt-1">Paste a JSON array of Q&amp;A records</p>
              </div>
            )}
            {uploadResult && (
              <div className={`mt-4 px-4 py-3 rounded-lg text-sm ${uploadResult.status === 'success' ? 'bg-green-50 border border-green-200 text-green-800' : 'bg-red-50 border border-red-200 text-red-800'}`}>
                <div className="flex items-center gap-2 font-medium">
                  {uploadResult.status === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                  {uploadResult.status === 'success' ? `Ingested ${uploadResult.records_ingested} records into ${uploadResult.collection_name}` : 'Ingestion failed'}
                </div>
                {uploadResult.errors.length > 0 && <div className="mt-2 text-xs space-y-0.5">{uploadResult.errors.slice(0, 5).map((e, i) => <p key={i}>• {e}</p>)}{uploadResult.errors.length > 5 && <p>...and {uploadResult.errors.length - 5} more</p>}</div>}
              </div>
            )}
            <div className="flex items-center gap-3 mt-4">
              <button onClick={handleIngest} disabled={uploading || (inputMode === 'file' && !selectedFile) || (inputMode === 'paste' && !pasteJson.trim())} className="flex items-center gap-2 px-5 py-2.5 bg-cognizant-blue text-white rounded-lg text-sm font-medium hover:bg-cognizant-blue-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                {uploading ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}{uploading ? 'Ingesting...' : 'Ingest Data'}
              </button>
              <p className="text-xs text-slate-400">Fields: <code className="bg-slate-100 px-1 rounded">question</code>, <code className="bg-slate-100 px-1 rounded">answer</code>, <code className="bg-slate-100 px-1 rounded">context</code>, <code className="bg-slate-100 px-1 rounded">metadata</code></p>
            </div>
          </div>
        )}
      </div>

      {/* Records Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <Database size={18} className="text-slate-500" />
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Ingested Records</h3>
            {totalRecords > 0 && <span className="ml-2 px-2 py-0.5 bg-cognizant-blue/10 text-cognizant-blue text-xs font-semibold rounded-full">{totalRecords.toLocaleString()}</span>}
          </div>
          {totalRecords > 0 && (
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input type="text" value={searchTerm} onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }} placeholder="Search records..." className="pl-8 pr-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none w-64" />
            </div>
          )}
        </div>
        <div className="px-6 py-4">
          {loadingRecords ? (
            <div className="text-center py-12"><Loader2 size={32} className="mx-auto text-cognizant-blue animate-spin mb-4" /><p className="text-sm text-slate-500">Loading records...</p></div>
          ) : totalRecords === 0 ? (
            <div className="text-center py-12 text-slate-400"><Database size={32} className="mx-auto mb-2 opacity-50" /><p className="text-sm font-medium text-slate-500">No data ingested yet</p><p className="text-xs mt-1">Upload a JSON file above to get started</p></div>
          ) : (
            <div>
              <DataTable columns={columns} data={records} onRowClick={(row) => setSelectedRecord(row as QARecord)} emptyMessage="No records match your search" pageSize={pageSize} />
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-3 text-sm text-slate-500">
                  <span>Page {currentPage} of {totalPages} ({totalRecords} total)</span>
                  <div className="flex gap-1">
                    <button onClick={() => setCurrentPage(Math.max(1, currentPage - 1))} disabled={currentPage === 1} className="px-3 py-1 border border-slate-300 rounded text-xs hover:bg-slate-50 disabled:opacity-50">Prev</button>
                    <button onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))} disabled={currentPage === totalPages} className="px-3 py-1 border border-slate-300 rounded text-xs hover:bg-slate-50 disabled:opacity-50">Next</button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, sub, small }: { icon: React.ReactNode; label: string; value: string; sub?: string; small?: boolean }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 px-4 py-3">
      <div className="flex items-center gap-2 text-slate-400 mb-1">{icon}<span className="text-xs font-semibold uppercase tracking-wide">{label}</span></div>
      <p className={`font-bold text-slate-800 ${small ? 'text-sm mt-1' : 'text-2xl'}`}>{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5 truncate" title={sub}>{sub}</p>}
    </div>
  );
}
