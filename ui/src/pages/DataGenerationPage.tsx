import { useState, useEffect, useCallback } from 'react';
import { Wand2, Database, Trash2, Download, Loader2, AlertCircle, Info, Plus, Minus, ChevronDown, ChevronRight, Layers, Search, Clock, Tag, BarChart3, Eye } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import type { DatasetConfig, GeneratedDataset, DatasetSummary, QAPair, IngestionStats } from '../api/types';
import { generateDataset, listDatasets, getDataset, deleteDataset, getMetadataFields, getIngestionStats } from '../api/client';
import DataTable from '../components/shared/DataTable';
import { Toasts, PairModal, ConfirmDelete, StatCard, trunc } from './DataGenHelpers';
import type { Toast } from './DataGenHelpers';

let toastId = 0;

export default function DataGenerationPage() {
  const { activeProject } = useProject();
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toast = useCallback((type: 'success' | 'error' | 'info', msg: string) => {
    const id = ++toastId;
    setToasts((p) => [...p, { id, type, message: msg }]);
    setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 5000);
  }, []);

  const [numPairs, setNumPairs] = useState(3);
  const [totalLimit, setTotalLimit] = useState(100);
  const [conds, setConds] = useState<{ key: string; value: string }[]>([]);
  const [inclCtx, setInclCtx] = useState(true);
  const [diffs, setDiffs] = useState<string[]>(['easy', 'medium', 'hard']);
  const [mode, setMode] = useState<'from_ingested' | 'llm_augmented'>('from_ingested');
  const [cfgOpen, setCfgOpen] = useState(true);
  const [mFields, setMFields] = useState<string[]>([]);
  const [stats, setStats] = useState<IngestionStats | null>(null);
  const [genning, setGenning] = useState(false);
  const [dsList, setDsList] = useState<DatasetSummary[]>([]);
  const [loadDs, setLoadDs] = useState(false);
  const [selDs, setSelDs] = useState<GeneratedDataset | null>(null);
  const [loadDet, setLoadDet] = useState(false);
  const [dsPage, setDsPage] = useState(1);
  const [dsSearch, setDsSearch] = useState('');
  const pgSz = 50;
  const [selPair, setSelPair] = useState<QAPair | null>(null);
  const [delTgt, setDelTgt] = useState<string | null>(null);
  const [deling, setDeling] = useState(false);

  const loadMeta = useCallback(async () => {
    if (!activeProject) return;
    try {
      const [f, s] = await Promise.all([getMetadataFields(activeProject.id), getIngestionStats(activeProject.id)]);
      setMFields(f.fields); setStats(s);
    } catch { /* silent */ }
  }, [activeProject]);

  const loadDatasets = useCallback(async () => {
    if (!activeProject) return;
    setLoadDs(true);
    try { setDsList((await listDatasets(activeProject.id)).datasets); }
    catch (e: unknown) { toast('error', `Load datasets failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setLoadDs(false); }
  }, [activeProject, toast]);

  const loadDetail = useCallback(async (id: string, pg = 1) => {
    if (!activeProject) return;
    setLoadDet(true);
    try { setSelDs(await getDataset(activeProject.id, id, pg, pgSz)); setDsPage(pg); }
    catch (e: unknown) { toast('error', `Load dataset failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setLoadDet(false); }
  }, [activeProject, toast]);

  useEffect(() => { if (activeProject) { loadMeta(); loadDatasets(); setSelDs(null); } }, [activeProject?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const doGen = useCallback(async () => {
    if (!activeProject) return;
    setGenning(true);
    try {
      const cfg: DatasetConfig = { num_pairs_per_chunk: numPairs, total_limit: totalLimit, query_conditions: Object.fromEntries(conds.filter((c) => c.key && c.value).map((c) => [c.key, c.value])), include_context: inclCtx, difficulty_levels: diffs, generation_mode: mode };
      const r = await generateDataset(activeProject.id, cfg);
      toast('success', `Generated ${r.total_pairs} pairs (${r.dataset_id})`);
      await loadDatasets(); setSelDs(r); setDsPage(1);
    } catch (e: unknown) { toast('error', `Generation failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setGenning(false); }
  }, [activeProject, numPairs, totalLimit, conds, inclCtx, diffs, mode, toast, loadDatasets]);

  const doDel = useCallback(async () => {
    if (!activeProject || !delTgt) return;
    setDeling(true);
    try {
      await deleteDataset(activeProject.id, delTgt);
      toast('success', `Deleted ${delTgt}`); setDelTgt(null);
      if (selDs?.dataset_id === delTgt) setSelDs(null);
      await loadDatasets();
    } catch (e: unknown) { toast('error', `Delete failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setDeling(false); }
  }, [activeProject, delTgt, selDs, toast, loadDatasets]);

  const doExport = useCallback(() => {
    if (!selDs) return;
    const b = new Blob([JSON.stringify(selDs, null, 2)], { type: 'application/json' });
    const u = URL.createObjectURL(b);
    const a = document.createElement('a'); a.href = u; a.download = `${selDs.dataset_id}.json`; a.click();
    URL.revokeObjectURL(u); toast('success', 'Exported as JSON');
  }, [selDs, toast]);

  const togDiff = (l: string) => setDiffs((p) => p.includes(l) ? p.filter((x) => x !== l) : [...p, l]);
  const addC = () => setConds((p) => [...p, { key: '', value: '' }]);
  const rmC = (i: number) => setConds((p) => p.filter((_, j) => j !== i));
  const upC = (i: number, f: 'key' | 'value', v: string) => setConds((p) => p.map((c, j) => j === i ? { ...c, [f]: v } : c));

  const fPairs = selDs?.pairs?.filter((p) => {
    if (!dsSearch) return true;
    const s = dsSearch.toLowerCase();
    return p.question.toLowerCase().includes(s) || p.expected_answer.toLowerCase().includes(s) || p.id.toLowerCase().includes(s) || p.type.toLowerCase().includes(s) || p.difficulty.toLowerCase().includes(s);
  }) ?? [];

  const cols = [
    { key: 'id', label: 'ID', render: (v: string) => <span className="font-mono text-xs text-slate-500">{v}</span> },
    { key: 'question', label: 'Question', render: (v: string) => <span>{trunc(v, 55)}</span> },
    { key: 'expected_answer', label: 'Answer', render: (v: string) => <span className="text-slate-600">{trunc(v, 45)}</span> },
    { key: 'context', label: 'Context', render: (v: string) => <span className="text-slate-500 text-xs">{trunc(v, 35)}</span> },
    { key: 'source_record_id', label: 'Source', render: (v: string) => <span className="font-mono text-xs text-slate-500">{trunc(v, 16)}</span> },
    { key: 'difficulty', label: 'Difficulty', render: (v: string) => <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${v === 'easy' ? 'bg-green-100 text-green-700' : v === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>{v}</span> },
    { key: 'type', label: 'Type', render: (v: string) => <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${v === 'original' ? 'bg-blue-100 text-blue-700' : v === 'paraphrased' ? 'bg-purple-100 text-purple-700' : 'bg-orange-100 text-orange-700'}`}>{v}</span> },
  ];

  if (!activeProject) return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Layers size={20} className="text-cognizant-blue" /></div>
        <div><h2 className="text-xl font-bold text-slate-800">Data Generation</h2><p className="text-sm text-slate-500">Generate test datasets from ingested Q&amp;A data</p></div>
      </div>
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <Info size={48} className="mx-auto text-slate-300 mb-4" />
        <h3 className="text-lg font-semibold text-slate-600 mb-2">No Project Selected</h3>
        <p className="text-sm text-slate-400">Select or create a project using the project selector.</p>
      </div>
    </div>
  );

  const hasData = stats && stats.total_records > 0;
  const totalPages = selDs ? (selDs.total_pages ?? Math.ceil(selDs.total_pairs / pgSz)) : 1;

  return (
    <div className="space-y-6">
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((p) => p.filter((t) => t.id !== id))} />
      {selPair && <PairModal pair={selPair} onClose={() => setSelPair(null)} />}
      {delTgt && <ConfirmDelete datasetId={delTgt} onConfirm={doDel} onCancel={() => setDelTgt(null)} loading={deling} />}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Wand2 size={20} className="text-cognizant-blue" /></div>
          <div><h2 className="text-xl font-bold text-slate-800">Data Generation</h2><p className="text-sm text-slate-500">Generate test Q&amp;A datasets from ingested knowledge base</p></div>
        </div>
        {dsList.length > 0 && <span className="px-3 py-1 bg-cognizant-blue/10 text-cognizant-blue text-sm font-semibold rounded-full">{dsList.length} dataset{dsList.length !== 1 ? 's' : ''}</span>}
      </div>

      {!hasData && <div className="bg-amber-50 border border-amber-200 rounded-xl px-5 py-4 flex items-start gap-3"><AlertCircle size={20} className="text-amber-500 mt-0.5 flex-shrink-0" /><div><p className="text-sm font-medium text-amber-800">No Ingested Data</p><p className="text-xs text-amber-600 mt-0.5">Ingest Q&amp;A data first in the Ingestion stage.</p></div></div>}

      {hasData && <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={<Database size={14} />} label="Ingested Records" value={stats.total_records.toLocaleString()} />
        <StatCard icon={<Tag size={14} />} label="Metadata Fields" value={String(mFields.length)} sub={mFields.join(', ')} />
        <StatCard icon={<BarChart3 size={14} />} label="Datasets" value={String(dsList.length)} />
        <StatCard icon={<Clock size={14} />} label="Last Ingested" value={stats.ingested_at ? new Date(stats.ingested_at).toLocaleString() : '—'} small />
      </div>}

      {/* Config Form */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <button onClick={() => setCfgOpen(!cfgOpen)} className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors">
          <div className="text-left"><h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Generation Configuration</h3><p className="text-xs text-slate-400 mt-0.5">Define how test Q&amp;A pairs should be generated</p></div>
          {cfgOpen ? <ChevronDown size={18} className="text-slate-400" /> : <ChevronRight size={18} className="text-slate-400" />}
        </button>
        {cfgOpen && <div className="px-6 pb-6 border-t border-slate-100 pt-4 space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div><label className="block text-sm font-medium text-slate-600 mb-1">Pairs per Chunk</label><input type="number" value={numPairs} onChange={(e) => setNumPairs(Math.max(1, parseInt(e.target.value) || 1))} min={1} max={10} className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none" /><p className="text-xs text-slate-400 mt-1">Q&amp;A pairs per ingested record</p></div>
            <div><label className="block text-sm font-medium text-slate-600 mb-1">Total Limit</label><input type="number" value={totalLimit} onChange={(e) => setTotalLimit(Math.max(0, parseInt(e.target.value) || 0))} min={0} className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none" /><p className="text-xs text-slate-400 mt-1">Max total pairs (0 = unlimited)</p></div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2"><label className="block text-sm font-medium text-slate-600">Query Conditions</label><button onClick={addC} className="flex items-center gap-1 text-xs text-cognizant-blue hover:text-cognizant-blue/80 font-medium"><Plus size={14} /> Add</button></div>
            {conds.length === 0 ? <p className="text-xs text-slate-400 bg-slate-50 px-3 py-2 rounded-lg">No conditions — all records used</p> : <div className="space-y-2">{conds.map((c, i) => <div key={i} className="flex items-center gap-2"><select value={c.key} onChange={(e) => upC(i, 'key', e.target.value)} className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue outline-none bg-white"><option value="">Select field...</option>{mFields.map((f) => <option key={f} value={f}>{f}</option>)}</select><input type="text" value={c.value} onChange={(e) => upC(i, 'value', e.target.value)} placeholder="Value..." className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue outline-none" /><button onClick={() => rmC(i)} className="p-2 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg"><Minus size={16} /></button></div>)}</div>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div><label className="block text-sm font-medium text-slate-600 mb-2">Include Context</label><label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={inclCtx} onChange={(e) => setInclCtx(e.target.checked)} className="w-4 h-4 text-cognizant-blue border-slate-300 rounded focus:ring-cognizant-blue" /><span className="text-sm text-slate-600">Include source context</span></label></div>
            <div><label className="block text-sm font-medium text-slate-600 mb-2">Difficulty Levels</label><div className="flex gap-3">{(['easy', 'medium', 'hard'] as const).map((l) => <label key={l} className="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" checked={diffs.includes(l)} onChange={() => togDiff(l)} className="w-4 h-4 text-cognizant-blue border-slate-300 rounded focus:ring-cognizant-blue" /><span className={`text-sm capitalize ${l === 'easy' ? 'text-green-600' : l === 'medium' ? 'text-yellow-600' : 'text-red-600'}`}>{l}</span></label>)}</div></div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-2">Generation Mode</label>
            <div className="flex gap-4">
              <label className={`flex items-start gap-2 cursor-pointer p-3 border rounded-lg flex-1 hover:bg-slate-50 ${mode === 'from_ingested' ? 'border-cognizant-blue bg-blue-50/30' : 'border-slate-200'}`}><input type="radio" name="gm" checked={mode === 'from_ingested'} onChange={() => setMode('from_ingested')} className="mt-0.5" /><div><p className="text-sm font-medium text-slate-700">From Ingested Data</p><p className="text-xs text-slate-400 mt-0.5">No API key needed</p></div></label>
              <label className={`flex items-start gap-2 cursor-pointer p-3 border rounded-lg flex-1 hover:bg-slate-50 ${mode === 'llm_augmented' ? 'border-cognizant-blue bg-blue-50/30' : 'border-slate-200'}`}><input type="radio" name="gm" checked={mode === 'llm_augmented'} onChange={() => setMode('llm_augmented')} className="mt-0.5" /><div><p className="text-sm font-medium text-slate-700">LLM Augmented</p><p className="text-xs text-slate-400 mt-0.5">Requires API key</p></div></label>
            </div>
            {mode === 'llm_augmented' && <div className="mt-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 flex items-center gap-2 text-xs text-amber-700"><AlertCircle size={14} /> Falls back to &quot;From Ingested&quot; if no API key configured</div>}
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button onClick={doGen} disabled={genning || !hasData || diffs.length === 0} className="flex items-center gap-2 px-5 py-2.5 bg-cognizant-blue text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed">{genning ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}{genning ? 'Generating...' : 'Generate Dataset'}</button>
            {!hasData && <p className="text-xs text-red-500">Ingest data first</p>}
            {hasData && diffs.length === 0 && <p className="text-xs text-red-500">Select at least one difficulty</p>}
          </div>
        </div>}
      </div>

      {/* Datasets List */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="flex items-center gap-2 px-6 py-4 border-b border-slate-100">
          <Database size={18} className="text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Generated Datasets</h3>
          {dsList.length > 0 && <span className="ml-2 px-2 py-0.5 bg-cognizant-blue/10 text-cognizant-blue text-xs font-semibold rounded-full">{dsList.length}</span>}
        </div>
        <div className="px-6 py-4">
          {loadDs ? <div className="text-center py-8"><Loader2 size={32} className="mx-auto text-cognizant-blue animate-spin mb-3" /><p className="text-sm text-slate-500">Loading datasets...</p></div>
          : dsList.length === 0 ? <div className="text-center py-8 text-slate-400"><Database size={32} className="mx-auto mb-2 opacity-50" /><p className="text-sm font-medium text-slate-500">No datasets generated yet</p><p className="text-xs mt-1">Configure and generate above</p></div>
          : <div className="grid gap-3">{dsList.map((ds) => (
            <div key={ds.dataset_id} className={`border rounded-lg p-4 transition-colors cursor-pointer hover:border-cognizant-blue/50 ${selDs?.dataset_id === ds.dataset_id ? 'border-cognizant-blue bg-blue-50/30' : 'border-slate-200'}`} onClick={() => loadDetail(ds.dataset_id)}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Wand2 size={14} className="text-cognizant-blue" /></div>
                  <div>
                    <p className="text-sm font-medium text-slate-700">{ds.dataset_id}</p>
                    <p className="text-xs text-slate-400">{new Date(ds.generated_at).toLocaleString()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 bg-cognizant-blue/10 text-cognizant-blue text-xs font-semibold rounded-full">{ds.total_pairs} pairs</span>
                  <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded-full">{ds.config_used.generation_mode === 'from_ingested' ? 'Ingested' : 'LLM'}</span>
                  <button onClick={(e) => { e.stopPropagation(); loadDetail(ds.dataset_id); }} className="p-1.5 text-slate-400 hover:text-cognizant-blue hover:bg-blue-50 rounded-lg"><Eye size={14} /></button>
                  <button onClick={(e) => { e.stopPropagation(); setDelTgt(ds.dataset_id); }} className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg"><Trash2 size={14} /></button>
                </div>
              </div>
            </div>
          ))}</div>}
        </div>
      </div>

      {/* Dataset Detail Table */}
      {selDs && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Eye size={18} className="text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Dataset: {selDs.dataset_id}</h3>
              <span className="ml-2 px-2 py-0.5 bg-cognizant-blue/10 text-cognizant-blue text-xs font-semibold rounded-full">{selDs.total_pairs} pairs</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" /><input type="text" value={dsSearch} onChange={(e) => setDsSearch(e.target.value)} placeholder="Search pairs..." className="pl-8 pr-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cognizant-blue outline-none w-56" /></div>
              <button onClick={doExport} className="flex items-center gap-1.5 px-3 py-1.5 border border-slate-300 rounded-lg text-sm text-slate-600 hover:bg-slate-50"><Download size={14} /> Export</button>
            </div>
          </div>
          <div className="px-6 py-4">
            {loadDet ? <div className="text-center py-8"><Loader2 size={32} className="mx-auto text-cognizant-blue animate-spin mb-3" /><p className="text-sm text-slate-500">Loading...</p></div>
            : <DataTable columns={cols} data={fPairs} onRowClick={(r) => setSelPair(r as QAPair)} emptyMessage="No pairs match your search" pageSize={pgSz} />}
            {totalPages > 1 && !loadDet && <div className="flex items-center justify-between mt-3 text-sm text-slate-500">
              <span>Page {dsPage} of {totalPages}</span>
              <div className="flex gap-1">
                <button onClick={() => loadDetail(selDs.dataset_id, Math.max(1, dsPage - 1))} disabled={dsPage === 1} className="px-3 py-1 border border-slate-300 rounded text-xs hover:bg-slate-50 disabled:opacity-50">Prev</button>
                <button onClick={() => loadDetail(selDs.dataset_id, Math.min(totalPages, dsPage + 1))} disabled={dsPage === totalPages} className="px-3 py-1 border border-slate-300 rounded text-xs hover:bg-slate-50 disabled:opacity-50">Next</button>
              </div>
            </div>}
          </div>
        </div>
      )}
    </div>
  );
}
