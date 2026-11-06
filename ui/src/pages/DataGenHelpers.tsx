import { CheckCircle, AlertCircle, X, Info, Loader2 } from 'lucide-react';
import type { QAPair } from '../api/types';

export interface Toast { id: number; type: 'success' | 'error' | 'info'; message: string }

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
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

export function PairModal({ pair, onClose }: { pair: QAPair; onClose: () => void }) {
  const F = ({ label, value, mono, scroll }: { label: string; value?: string; mono?: boolean; scroll?: boolean }) => (
    <div>
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">{label}</label>
      <p className={`text-sm text-slate-700 bg-slate-50 px-3 py-2 rounded-lg whitespace-pre-wrap ${mono ? 'font-mono break-all' : ''} ${scroll ? 'max-h-60 overflow-y-auto' : ''}`}>{value || '—'}</p>
    </div>
  );
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800">Q&amp;A Pair Details</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600"><X size={20} /></button>
        </div>
        <div className="px-6 py-4 space-y-4">
          <F label="Pair ID" value={pair.id} mono />
          <div className="grid grid-cols-2 gap-4"><F label="Difficulty" value={pair.difficulty} /><F label="Type" value={pair.type} /></div>
          <F label="Question" value={pair.question} />
          <F label="Expected Answer" value={pair.expected_answer} scroll />
          {pair.context && <F label="Context" value={pair.context} scroll />}
          <F label="Source Record ID" value={pair.source_record_id} mono />
          {pair.metadata && Object.keys(pair.metadata).length > 0 && (
            <div>
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Metadata</label>
              <div className="bg-slate-50 px-3 py-2 rounded-lg">
                <table className="w-full text-sm"><tbody>
                  {Object.entries(pair.metadata).map(([k, v]) => (
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
          <button onClick={onClose} className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200">Close</button>
        </div>
      </div>
    </div>
  );
}

export function ConfirmDelete({ datasetId, onConfirm, onCancel, loading }: { datasetId: string; onConfirm: () => void; onCancel: () => void; loading: boolean }) {
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onCancel}>
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full" onClick={(e) => e.stopPropagation()}>
        <div className="px-6 py-4 border-b border-slate-200"><h3 className="text-lg font-semibold text-slate-800">Delete Dataset</h3></div>
        <div className="px-6 py-4"><p className="text-sm text-slate-600">Delete dataset <code className="bg-slate-100 px-1 rounded text-xs">{datasetId}</code>? This cannot be undone.</p></div>
        <div className="px-6 py-3 border-t border-slate-200 flex justify-end gap-2">
          <button onClick={onCancel} disabled={loading} className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 disabled:opacity-50">Cancel</button>
          <button onClick={onConfirm} disabled={loading} className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50">
            {loading && <Loader2 size={14} className="animate-spin" />}{loading ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function StatCard({ icon, label, value, sub, small }: { icon: React.ReactNode; label: string; value: string; sub?: string; small?: boolean }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 px-4 py-3">
      <div className="flex items-center gap-2 text-slate-400 mb-1">{icon}<span className="text-xs font-semibold uppercase tracking-wide">{label}</span></div>
      <p className={`font-bold text-slate-800 ${small ? 'text-sm mt-1' : 'text-2xl'}`}>{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5 truncate" title={sub}>{sub}</p>}
    </div>
  );
}

export function trunc(s: string | undefined | null, n = 80): string {
  return !s ? '—' : s.length > n ? s.slice(0, n) + '…' : s;
}
