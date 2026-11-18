import { useState } from 'react';
import { connect } from '../api/client';

interface Props {
  connected: boolean;
  onConnected: (dbType: string, recordCount: number) => void;
  onRunChecks: () => void;
  loading: boolean;
}

const DB_TYPES = ['chroma', 'faiss', 'milvus_lite', 'pinecone', 'weaviate', 'elastic', 'milvus', 'pgvector'] as const;

interface FieldDef {
  key: string;
  label: string;
  placeholder: string;
  type?: string;
}

const DB_FIELDS: Record<string, FieldDef[]> = {
  chroma: [
    { key: 'collection_name', label: 'Collection Name', placeholder: 'my_collection' },
    { key: 'persist_directory', label: 'Persist Directory', placeholder: './chroma_data' },
  ],
  faiss: [
    { key: 'collection_name', label: 'Collection Name', placeholder: 'my_collection' },
    { key: 'persist_directory', label: 'Persist Directory', placeholder: './faiss_data' },
  ],
  milvus_lite: [
    { key: 'collection_name', label: 'Collection Name', placeholder: 'my_collection' },
    { key: 'persist_directory', label: 'Database Path', placeholder: './milvus_demo.db' },
  ],
  pinecone: [
    { key: 'api_key', label: 'API Key', placeholder: 'pk-...', type: 'password' },
    { key: 'index_name', label: 'Index Name', placeholder: 'my-index' },
    { key: 'environment', label: 'Environment', placeholder: 'us-east-1' },
  ],
  weaviate: [
    { key: 'host', label: 'Host', placeholder: 'localhost' },
    { key: 'port', label: 'Port', placeholder: '8080' },
    { key: 'class_name', label: 'Class Name', placeholder: 'Document' },
  ],
  elastic: [
    { key: 'host', label: 'Host', placeholder: 'localhost' },
    { key: 'port', label: 'Port', placeholder: '9200' },
    { key: 'index_name', label: 'Index Name', placeholder: 'vectors' },
    { key: 'api_key', label: 'API Key (optional)', placeholder: '', type: 'password' },
  ],
  milvus: [
    { key: 'host', label: 'Host', placeholder: 'localhost' },
    { key: 'port', label: 'Port', placeholder: '19530' },
    { key: 'collection_name', label: 'Collection Name', placeholder: 'my_collection' },
  ],
  pgvector: [
    { key: 'host', label: 'Host', placeholder: 'localhost' },
    { key: 'port', label: 'Port', placeholder: '5432' },
    { key: 'database', label: 'Database', placeholder: 'vectors' },
    { key: 'table_name', label: 'Table Name', placeholder: 'embeddings' },
  ],
};

export default function ConnectionPanel({ connected, onConnected, onRunChecks, loading }: Props) {
  const [expanded, setExpanded] = useState(!connected);
  const [dbType, setDbType] = useState<string>('chroma');
  const [params, setParams] = useState<Record<string, string>>({});
  const [connecting, setConnecting] = useState(false);
  const [connError, setConnError] = useState<string | null>(null);

  const fields = DB_FIELDS[dbType] ?? [];

  const handleConnect = async () => {
    setConnecting(true);
    setConnError(null);
    try {
      const res = await connect({ db_type: dbType, params });
      onConnected(res.db_type ?? dbType, res.record_count ?? 0);
      setExpanded(false);
    } catch (err: any) {
      setConnError(err.message ?? 'Connection failed');
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-lg">🔌</span>
          <span className="font-semibold text-slate-700">Database Connection</span>
          {connected && (
            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
              Connected
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="px-6 pb-6 border-t border-slate-100 pt-4 space-y-4">
          {/* DB Type selector */}
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">
              Database Type
            </label>
            <select
              value={dbType}
              onChange={(e) => {
                setDbType(e.target.value);
                setParams({});
              }}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              {DB_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t === 'milvus_lite'
                    ? 'Milvus Lite (local)'
                    : t.charAt(0).toUpperCase() + t.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Dynamic fields */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {fields.map((f) => (
              <div key={f.key}>
                <label className="block text-sm font-medium text-slate-600 mb-1">
                  {f.label}
                </label>
                <input
                  type={f.type ?? 'text'}
                  placeholder={f.placeholder}
                  value={params[f.key] ?? ''}
                  onChange={(e) => setParams({ ...params, [f.key]: e.target.value })}
                  className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                />
              </div>
            ))}
          </div>

          {connError && (
            <p className="text-sm text-red-600">{connError}</p>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleConnect}
              disabled={connecting}
              className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {connecting ? 'Connecting…' : 'Connect'}
            </button>

            {connected && (
              <button
                onClick={onRunChecks}
                disabled={loading}
                className="px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {loading ? 'Running…' : '▶ Run Checks'}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
