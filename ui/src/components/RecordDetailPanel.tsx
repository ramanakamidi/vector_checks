import { useState, useEffect } from 'react';
import { getRecordDetail } from '../api/client';
import type { RecordDetail, FaultDetail } from '../api/types';

interface Props {
  recordId: string;
  onClose: () => void;
}

export default function RecordDetailPanel({ recordId, onClose }: Props) {
  const [detail, setDetail] = useState<RecordDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getRecordDetail(recordId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [recordId]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 z-40"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed top-0 right-0 h-full w-full max-w-lg bg-white shadow-2xl z-50 slide-in flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50">
          <h2 className="font-bold text-slate-800 text-lg">Record Detail</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-200 text-slate-500 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
          {loading && (
            <p className="text-slate-400 text-center py-8">Loading…</p>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}

          {detail && (
            <>
              {/* ID */}
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                  Record ID
                </p>
                <p className="text-sm font-mono bg-slate-50 px-3 py-2 rounded break-all">
                  {detail.id}
                </p>
              </div>

              {/* Fault Analysis — Expected vs Actual */}
              {detail.fault_details && detail.fault_details.length > 0 ? (
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                    Fault Analysis
                  </p>
                  <div className="space-y-3">
                    {detail.fault_details.map((fd: FaultDetail, idx: number) => (
                      <div
                        key={`${fd.check}-${idx}`}
                        className={`rounded-lg border overflow-hidden ${
                          fd.severity === 'error'
                            ? 'border-red-200'
                            : 'border-amber-200'
                        }`}
                        style={{
                          borderLeftWidth: '4px',
                          borderLeftColor:
                            fd.severity === 'error' ? '#ef4444' : '#f59e0b',
                        }}
                      >
                        {/* Check name header */}
                        <div
                          className={`px-3 py-2 ${
                            fd.severity === 'error'
                              ? 'bg-red-50'
                              : 'bg-amber-50'
                          }`}
                        >
                          <span className="font-bold text-sm text-slate-800">
                            {fd.severity === 'error' ? '❌' : '⚠️'}{' '}
                            {fd.check}
                          </span>
                        </div>
                        {/* Expected / Actual rows */}
                        <table className="w-full text-sm">
                          <tbody>
                            <tr className="bg-emerald-50/60">
                              <td className="px-3 py-2 font-semibold text-emerald-700 w-20 align-top whitespace-nowrap">
                                Expected
                              </td>
                              <td className="px-3 py-2 text-slate-700 font-mono text-xs break-all">
                                {fd.expected}
                              </td>
                            </tr>
                            <tr className="bg-red-50/60">
                              <td className="px-3 py-2 font-semibold text-red-700 w-20 align-top whitespace-nowrap">
                                Actual
                              </td>
                              <td className="px-3 py-2 text-slate-700 font-mono text-xs break-all">
                                {fd.actual}
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    ))}
                  </div>
                </div>
              ) : detail.fault_types && detail.fault_types.length > 0 ? (
                /* Fallback: simple badges when fault_details is not available */
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                    Flagged By
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {detail.fault_types.map((ft) => (
                      <span
                        key={ft}
                        className="inline-flex items-center gap-1 bg-red-50 text-red-700 border border-red-200 text-xs font-medium px-2.5 py-1 rounded-full"
                      >
                        <span className="w-1.5 h-1.5 bg-red-500 rounded-full" />
                        {ft}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* Vector info */}
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  Vector Info
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-50 rounded-lg px-3 py-2">
                    <p className="text-xs text-slate-500">Dimension</p>
                    <p className="text-sm font-semibold">{detail.vector_dim}</p>
                  </div>
                  <div className="bg-slate-50 rounded-lg px-3 py-2">
                    <p className="text-xs text-slate-500">L2 Norm</p>
                    <p className="text-sm font-semibold">
                      {typeof detail.vector_norm === 'number'
                        ? detail.vector_norm.toFixed(4)
                        : String(detail.vector_norm)}
                    </p>
                  </div>
                </div>
                {detail.vector_preview && detail.vector_preview.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-slate-500 mb-1">
                      First {Math.min(10, detail.vector_preview.length)} values
                    </p>
                    <p className="text-xs font-mono bg-slate-50 px-3 py-2 rounded break-all">
                      [{detail.vector_preview.slice(0, 10).map((v) => v.toFixed(4)).join(', ')}
                      {detail.vector_preview.length > 10 ? ', …' : ''}]
                    </p>
                  </div>
                )}
              </div>

              {/* Text / Document content */}
              {(detail.metadata?.text || detail.metadata?.document || detail.metadata?.content || detail.metadata?.chunk) && (
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                    Text / Document Content
                  </p>
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 max-h-72 overflow-y-auto">
                    <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
                      {String(
                        detail.metadata.text ??
                        detail.metadata.document ??
                        detail.metadata.content ??
                        detail.metadata.chunk
                      )}
                    </p>
                  </div>
                </div>
              )}

              {/* Metadata (all fields) */}
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  All Metadata
                </p>
                <div className="border border-slate-200 rounded-lg overflow-hidden">
                  <table className="w-full text-sm">
                    <tbody>
                      {Object.entries(detail.metadata ?? {}).map(([k, v]) => (
                        <tr key={k} className="border-t border-slate-100 first:border-t-0">
                          <td className="px-3 py-2 font-medium text-slate-600 bg-slate-50 w-1/3 align-top whitespace-nowrap">
                            {k}
                          </td>
                          <td className="px-3 py-2 text-slate-700 break-all">
                            {typeof v === 'object' ? (
                              <pre className="text-xs font-mono whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre>
                            ) : (
                              String(v)
                            )}
                          </td>
                        </tr>
                      ))}
                      {Object.keys(detail.metadata ?? {}).length === 0 && (
                        <tr>
                          <td colSpan={2} className="px-3 py-3 text-slate-400 text-center italic">
                            No metadata fields
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
