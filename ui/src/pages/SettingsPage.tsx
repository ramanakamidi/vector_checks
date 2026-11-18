import { useState, useEffect, useCallback } from 'react';
import { Settings, ChevronDown, ChevronRight, Save, Eye, EyeOff, Plus, Trash2, CheckCircle, AlertCircle, X, Loader2, Info } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import type { ProjectSettings, LLMConfig, AUTConfig, FieldMapperConfig } from '../api/types';
import { getProjectSettings, saveProjectSettings, updateProject } from '../api/client';

interface Toast { id: number; type: 'success' | 'error'; message: string; }
let toastId = 0;

interface MetadataField { field_name: string; display_label: string; include_in_report: boolean; }

const ic = 'w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none transition-colors';
const sc = 'w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue outline-none bg-white transition-colors';
const lc = 'block text-sm font-medium text-slate-600 mb-1';

function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="fixed top-4 right-4 z-[100] space-y-2">
      {toasts.map((t) => (
        <div key={t.id} className={`flex items-center gap-2 px-4 py-3 rounded-lg shadow-lg text-sm font-medium ${t.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
          {t.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          <span className="flex-1">{t.message}</span>
          <button onClick={() => onDismiss(t.id)} className="text-current opacity-50 hover:opacity-100"><X size={14} /></button>
        </div>
      ))}
    </div>
  );
}

function Card({ title, subtitle, defaultOpen = true, children }: { title: string; subtitle?: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors">
        <div className="text-left">
          <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">{title}</h3>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
        </div>
        {open ? <ChevronDown size={18} className="text-slate-400" /> : <ChevronRight size={18} className="text-slate-400" />}
      </button>
      {open && <div className="px-6 pb-6 border-t border-slate-100 pt-4">{children}</div>}
    </div>
  );
}

function SBtn({ onClick, saving }: { onClick: () => void; saving: boolean }) {
  return (
    <button onClick={onClick} disabled={saving} className="flex items-center gap-2 px-4 py-2 bg-cognizant-blue text-white rounded-lg text-sm font-medium hover:bg-cognizant-blue-dark transition-colors disabled:opacity-50">
      {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
      {saving ? 'Saving...' : 'Save'}
    </button>
  );
}

export default function SettingsPage() {
  const { activeProject, setActiveProject, refreshProjects } = useProject();
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [loadingSettings, setLoadingSettings] = useState(false);
  const [savingProject, setSavingProject] = useState(false);
  const [savingLLM, setSavingLLM] = useState(false);
  const [savingAUT, setSavingAUT] = useState(false);
  const [savingMeta, setSavingMeta] = useState(false);
  const [savingDoc, setSavingDoc] = useState(false);

  const [projectName, setProjectName] = useState('');
  const [projectDesc, setProjectDesc] = useState('');
  const [llmApiKey, setLlmApiKey] = useState('');
  const [llmApiKeySet, setLlmApiKeySet] = useState(false);
  const [llmModel, setLlmModel] = useState('gpt-4');
  const [showKey, setShowKey] = useState(false);
  const [autName, setAutName] = useState('');
  const [autUrl, setAutUrl] = useState('http://localhost:8000/mock-aut');
  const [autMethod, setAutMethod] = useState('POST');
  const [autEndpoint, setAutEndpoint] = useState('/ask');
  const [autType, setAutType] = useState('rest_api');
  const [autAnswerPath, setAutAnswerPath] = useState('answer');
  const [autContextPath, setAutContextPath] = useState('context');
  const [vsType, setVsType] = useState('chroma');
  const [vsDir, setVsDir] = useState('');
  const [vsCollection, setVsCollection] = useState('');
  const [metaFields, setMetaFields] = useState<MetadataField[]>([
    { field_name: 'doc_id', display_label: 'Document ID', include_in_report: true },
    { field_name: 'doc_name', display_label: 'Document Name', include_in_report: true },
    { field_name: 'page_number', display_label: 'Page Number', include_in_report: true },
    { field_name: 'object_type', display_label: 'Object Type', include_in_report: true },
    { field_name: 'text', display_label: 'Text Content', include_in_report: true },
  ]);
  const [filesDir, setFilesDir] = useState('');
  const [extractMode, setExtractMode] = useState('json_import');

  const toast = useCallback((type: 'success' | 'error', message: string) => {
    const id = ++toastId;
    setToasts((p) => [...p, { id, type, message }]);
    setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 4000);
  }, []);

  useEffect(() => {
    if (!activeProject) return;
    setProjectName(activeProject.name);
    setProjectDesc(activeProject.description || '');
    setLoadingSettings(true);
    getProjectSettings(activeProject.id)
      .then((s) => {
        if (s.llm_config) { setLlmApiKey(s.llm_config.api_key || ''); setLlmApiKeySet(s.llm_config.api_key_set || false); setLlmModel(s.llm_config.model || 'gpt-4'); }
        if (s.aut_config) { setAutUrl(s.aut_config.url || 'http://localhost:8000/mock-aut'); setAutMethod(s.aut_config.method || 'POST'); setAutEndpoint(s.aut_config.endpoint || '/ask'); setAutAnswerPath(s.aut_config.response_answer_path || 'answer'); setAutContextPath(s.aut_config.response_context_path || 'context'); setAutName((s.aut_config as any).name || ''); setAutType((s.aut_config as any).aut_type || 'rest_api'); }
        if (s.vector_store_config) { setVsType(s.vector_store_config.db_type || 'chroma'); setVsDir(s.vector_store_config.persist_directory || ''); setVsCollection(s.vector_store_config.collection_name || ''); }
        if (s.field_mapper) {
          const custom: MetadataField[] = (s.field_mapper as any).custom_metadata_fields || [];
          if (custom.length > 0) { setMetaFields(custom); }
          else if (s.field_mapper.display_fields?.length) { setMetaFields(s.field_mapper.display_fields.map((f) => ({ field_name: f, display_label: f.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()), include_in_report: true }))); }
        }
        setFilesDir(s.files_directory || '');
        setExtractMode(s.extraction_mode || 'json_import');
      })
      .catch((e: any) => toast('error', `Failed to load settings: ${e.message}`))
      .finally(() => setLoadingSettings(false));
  }, [activeProject?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveProjectInfo = async () => {
    if (!activeProject || !projectName.trim()) { toast('error', 'Project name is required'); return; }
    setSavingProject(true);
    try {
      await updateProject(activeProject.id, { name: projectName.trim(), description: projectDesc.trim() });
      const ps = await refreshProjects();
      const u = ps.find((p) => p.id === activeProject.id);
      if (u) setActiveProject(u);
      toast('success', 'Project info saved');
    } catch (e: any) { toast('error', e.message); } finally { setSavingProject(false); }
  };

  const saveLLM = async () => {
    if (!activeProject) return;
    setSavingLLM(true);
    try {
      await saveProjectSettings(activeProject.id, { llm_config: { provider: 'openai', api_key: llmApiKey, model: llmModel } });
      if (llmApiKey && llmApiKey !== '***') { setLlmApiKeySet(true); setLlmApiKey('***'); }
      toast('success', 'LLM configuration saved');
    } catch (e: any) { toast('error', e.message); } finally { setSavingLLM(false); }
  };

  const saveAUT = async () => {
    if (!activeProject) return;
    if (!autUrl.trim()) { toast('error', 'AUT Endpoint URL is required'); return; }
    setSavingAUT(true);
    try {
      await saveProjectSettings(activeProject.id, { aut_config: { name: autName, aut_type: autType, url: autUrl, method: autMethod, endpoint: autEndpoint, headers: {}, request_template: { question: '{{query}}' }, response_answer_path: autAnswerPath, response_context_path: autContextPath } as any });
      toast('success', 'AUT configuration saved');
    } catch (e: any) { toast('error', e.message); } finally { setSavingAUT(false); }
  };

  const saveMeta = async () => {
    if (!activeProject) return;
    setSavingMeta(true);
    try {
      const fm: FieldMapperConfig = { doc_id_field: 'doc_id', doc_name_field: 'doc_name', text_field: 'text', page_field: 'page_number', chunk_index_field: 'chunk_index', object_type_field: 'object_type', display_fields: metaFields.filter((f) => f.include_in_report).map((f) => f.field_name), required_fields: ['doc_id', 'object_type'], similar_group_field: 'doc_id', different_group_field: 'object_type', custom_metadata_fields: metaFields };
      await saveProjectSettings(activeProject.id, { field_mapper: fm });
      toast('success', 'Metadata field mapping saved');
    } catch (e: any) { toast('error', e.message); } finally { setSavingMeta(false); }
  };

  const saveDoc = async () => {
    if (!activeProject) return;
    setSavingDoc(true);
    try {
      await saveProjectSettings(activeProject.id, { files_directory: filesDir, extraction_mode: extractMode });
      toast('success', 'Document source configuration saved');
    } catch (e: any) { toast('error', e.message); } finally { setSavingDoc(false); }
  };

  if (!activeProject) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Settings size={20} className="text-cognizant-blue" /></div>
          <div><h2 className="text-xl font-bold text-slate-800">Settings</h2><p className="text-sm text-slate-500">Configure project settings, LLM keys, AUT endpoint, and vector store</p></div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <Info size={48} className="mx-auto text-slate-300 mb-4" />
          <h3 className="text-lg font-semibold text-slate-600 mb-2">No Project Selected</h3>
          <p className="text-sm text-slate-400">Select or create a project using the project selector in the top navigation bar.</p>
        </div>
      </div>
    );
  }

  if (loadingSettings) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Settings size={20} className="text-cognizant-blue" /></div>
          <div><h2 className="text-xl font-bold text-slate-800">Settings</h2><p className="text-sm text-slate-500">Loading settings...</p></div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <Loader2 size={32} className="mx-auto text-cognizant-blue animate-spin mb-4" />
          <p className="text-sm text-slate-500">Loading project settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((p) => p.filter((t) => t.id !== id))} />
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center"><Settings size={20} className="text-cognizant-blue" /></div>
        <div><h2 className="text-xl font-bold text-slate-800">Settings</h2><p className="text-sm text-slate-500">Configure project settings, LLM keys, AUT endpoint, and vector store</p></div>
      </div>

      <div className="grid gap-4">
        {/* Project Info */}
        <Card title="Project Information" subtitle="Name, description, and metadata">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className={lc}>Project Name <span className="text-red-500">*</span></label>
              <input type="text" value={projectName} onChange={(e) => setProjectName(e.target.value)} className={ic} placeholder="e.g., Insurance RAG Test" />
            </div>
            <div>
              <label className={lc}>Created</label>
              <input type="text" value={activeProject.created_at ? new Date(activeProject.created_at).toLocaleString() : '—'} className={`${ic} bg-slate-50 text-slate-500`} readOnly />
            </div>
            <div className="md:col-span-2">
              <label className={lc}>Description</label>
              <textarea value={projectDesc} onChange={(e) => setProjectDesc(e.target.value)} className={`${ic} resize-none`} rows={3} placeholder="Brief description of this test project..." />
            </div>
          </div>
          <div className="flex justify-end mt-4"><SBtn onClick={saveProjectInfo} saving={savingProject} /></div>
        </Card>

        {/* LLM Configuration */}
        <Card title="LLM Configuration" subtitle="OpenAI API key and model selection">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="md:col-span-2">
              <label className={lc}>OpenAI API Key{llmApiKeySet && <span className="ml-2 text-xs text-green-600 font-normal">✓ Key is set</span>}</label>
              <div className="relative">
                <input type={showKey ? 'text' : 'password'} value={llmApiKey} onChange={(e) => { setLlmApiKey(e.target.value); if (e.target.value !== '***') setLlmApiKeySet(false); }} onFocus={() => { if (llmApiKey === '***') { setLlmApiKey(''); setLlmApiKeySet(false); } }} className={`${ic} pr-10`} placeholder="sk-..." />
                <button type="button" onClick={() => setShowKey(!showKey)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                  {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
            <div>
              <label className={lc}>Model</label>
              <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)} className={sc}>
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              </select>
            </div>
            <div>
              <label className={lc}>Provider</label>
              <select className={`${sc} bg-slate-50`} disabled value="openai"><option value="openai">OpenAI</option></select>
              <p className="text-xs text-slate-400 mt-1">Only OpenAI is supported currently</p>
            </div>
          </div>
          <div className="flex justify-end mt-4"><SBtn onClick={saveLLM} saving={savingLLM} /></div>
        </Card>

        {/* AUT Configuration */}
        <Card title="Application Under Test (AUT)" subtitle="Configure the RAG application endpoint to test">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className={lc}>AUT Name</label>
              <input type="text" value={autName} onChange={(e) => setAutName(e.target.value)} className={ic} placeholder="e.g., My RAG Application" />
            </div>
            <div>
              <label className={lc}>AUT Type</label>
              <select value={autType} onChange={(e) => setAutType(e.target.value)} className={sc}>
                <option value="rest_api">REST API</option>
                <option value="mock">Mock</option>
              </select>
            </div>
            <div>
              <label className={lc}>Endpoint URL <span className="text-red-500">*</span></label>
              <input type="text" value={autUrl} onChange={(e) => setAutUrl(e.target.value)} className={ic} placeholder="http://localhost:8001/api/query" />
            </div>
            <div>
              <label className={lc}>HTTP Method</label>
              <select value={autMethod} onChange={(e) => setAutMethod(e.target.value)} className={sc}>
                <option value="POST">POST</option>
                <option value="GET">GET</option>
              </select>
            </div>
            <div>
              <label className={lc}>Response Answer Path</label>
              <input type="text" value={autAnswerPath} onChange={(e) => setAutAnswerPath(e.target.value)} className={ic} placeholder="answer" />
              <p className="text-xs text-slate-400 mt-1">JSON path to the answer field in AUT response</p>
            </div>
            <div>
              <label className={lc}>Response Context Path</label>
              <input type="text" value={autContextPath} onChange={(e) => setAutContextPath(e.target.value)} className={ic} placeholder="context" />
              <p className="text-xs text-slate-400 mt-1">JSON path to the context field in AUT response</p>
            </div>
          </div>
          <div className="flex justify-end mt-4"><SBtn onClick={saveAUT} saving={savingAUT} /></div>
        </Card>

        {/* Vector Store Configuration */}
        <Card title="Vector Store Configuration" subtitle="Selected when the project is created">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className={lc}>Vector Store Type</label>
              <select value={vsType} disabled className={`${sc} disabled:bg-slate-100 disabled:text-slate-500`}>
                <option value="chroma">Chroma</option>
                <option value="faiss">FAISS (local)</option>
                <option value="milvus_lite">Milvus Lite (local file)</option>
              </select>
            </div>
            <div>
              <label className={lc}>Collection Name</label>
              <input type="text" value={vsCollection} readOnly className={`${ic} bg-slate-100 text-slate-500`} />
            </div>
            {(vsType === 'chroma' || vsType === 'faiss' || vsType === 'milvus_lite') && (
              <div className="md:col-span-2">
                <label className={lc}>{vsType === 'milvus_lite' ? 'Database Path' : 'Storage Directory'}</label>
                <input type="text" value={vsDir} readOnly className={`${ic} bg-slate-100 text-slate-500`} />
              </div>
            )}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            To use a different vector database, create a new project and choose it there.
          </p>
        </Card>

        {/* Metadata Field Mapping */}
        <Card title="Metadata Field Mapping" subtitle="Configure which metadata fields to track and display">
          <div className="space-y-3">
            <div className="grid grid-cols-12 gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wide px-1">
              <div className="col-span-4">Field Name</div>
              <div className="col-span-4">Display Label</div>
              <div className="col-span-2 text-center">In Report</div>
              <div className="col-span-2 text-center">Actions</div>
            </div>
            {metaFields.map((f, i) => (
              <div key={i} className="grid grid-cols-12 gap-2 items-center">
                <div className="col-span-4">
                  <input type="text" value={f.field_name} onChange={(e) => setMetaFields((p) => p.map((x, j) => j === i ? { ...x, field_name: e.target.value } : x))} className={ic} placeholder="field_name" />
                </div>
                <div className="col-span-4">
                  <input type="text" value={f.display_label} onChange={(e) => setMetaFields((p) => p.map((x, j) => j === i ? { ...x, display_label: e.target.value } : x))} className={ic} placeholder="Display Label" />
                </div>
                <div className="col-span-2 flex justify-center">
                  <input type="checkbox" checked={f.include_in_report} onChange={(e) => setMetaFields((p) => p.map((x, j) => j === i ? { ...x, include_in_report: e.target.checked } : x))} className="w-4 h-4 text-cognizant-blue rounded border-slate-300 focus:ring-cognizant-blue" />
                </div>
                <div className="col-span-2 flex justify-center">
                  <button onClick={() => setMetaFields((p) => p.filter((_, j) => j !== i))} className="text-red-400 hover:text-red-600 p-1 rounded hover:bg-red-50 transition-colors"><Trash2 size={14} /></button>
                </div>
              </div>
            ))}
            <button onClick={() => setMetaFields((p) => [...p, { field_name: '', display_label: '', include_in_report: true }])} className="flex items-center gap-2 text-sm text-cognizant-blue hover:text-cognizant-blue-dark font-medium mt-2">
              <Plus size={14} /> Add Field
            </button>
          </div>
          <div className="flex justify-end mt-4"><SBtn onClick={saveMeta} saving={savingMeta} /></div>
        </Card>

        {/* Document Source Configuration */}
        <Card title="Document Source Configuration" subtitle="File paths and extraction settings">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="md:col-span-2">
              <label className={lc}>Files Directory</label>
              <input type="text" value={filesDir} onChange={(e) => setFilesDir(e.target.value)} className={ic} placeholder="C:/data/documents" />
            </div>
            <div>
              <label className={lc}>Extraction Mode</label>
              <select value={extractMode} onChange={(e) => setExtractMode(e.target.value)} className={sc}>
                <option value="json_import">JSON Import</option>
                <option value="docling" disabled>Docling (Coming Soon)</option>
              </select>
              <p className="text-xs text-amber-600 mt-1">⚠ Docling is currently blocked. Use JSON Import instead.</p>
            </div>
          </div>
          <div className="flex justify-end mt-4"><SBtn onClick={saveDoc} saving={savingDoc} /></div>
        </Card>
      </div>
    </div>
  );
}
