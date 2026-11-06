import { useState } from 'react';
import type { CheckResult } from '../api/types';
import DrilldownTable from './DrilldownTable';

interface Props {
  result: CheckResult;
  onSelectRecord: (id: string) => void;
}

function statusLabel(passed: boolean | null): { text: string; cls: string; border: string } {
  if (passed === true) return { text: 'PASS', cls: 'bg-green-100 text-green-700', border: 'border-l-green-500' };
  if (passed === false) return { text: 'FAIL', cls: 'bg-red-100 text-red-700', border: 'border-l-red-500' };
  return { text: 'INFO', cls: 'bg-blue-100 text-blue-700', border: 'border-l-blue-500' };
}

export default function CheckCard({ result, onSelectRecord }: Props) {
  const [expanded, setExpanded] = useState(false);
  const status = statusLabel(result.passed);
  const detail = result.detail ?? {};

  /* Extract common detail fields */
  const stats = detail.stats ?? detail.summary ?? null;
  const failedRecords: any[] = detail.failed_records ?? detail.records ?? detail.items ?? [];
  const rootCause = detail.root_cause ?? detail.error ?? null;
  const remediation = detail.remediation ?? detail.fix ?? null;
  const description = detail.description ?? detail.help ?? null;

  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 border-l-4 ${status.border} shadow-sm overflow-hidden`}
    >
      {/* Header row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span className={`text-xs font-bold px-2 py-0.5 rounded ${status.cls}`}>
            {status.text}
          </span>
          <span className="font-medium text-slate-800">{result.check}</span>
          {description && (
            <span className="relative group">
              <span className="text-slate-400 cursor-help">ⓘ</span>
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-slate-800 text-white text-xs rounded px-3 py-2 w-64 z-10 shadow-lg">
                {description}
              </span>
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

      {/* Expanded body */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-slate-100 pt-4">
          {/* Stats row */}
          {stats && typeof stats === 'object' && (
            <div className="flex flex-wrap gap-4">
              {Object.entries(stats as Record<string, any>).map(([k, v]) => (
                <div key={k} className="bg-slate-50 rounded-lg px-3 py-2">
                  <p className="text-xs text-slate-500">{k}</p>
                  <p className="text-sm font-semibold text-slate-700">
                    {typeof v === 'number' ? v.toLocaleString() : String(v)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Root cause */}
          {rootCause && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-xs font-semibold text-red-600 mb-1">Root Cause</p>
              <p className="text-sm text-red-800 whitespace-pre-wrap">{String(rootCause)}</p>
            </div>
          )}

          {/* Remediation */}
          {remediation && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-xs font-semibold text-green-600 mb-1">Remediation</p>
              <p className="text-sm text-green-800 whitespace-pre-wrap">{String(remediation)}</p>
            </div>
          )}

          {/* Drilldown table */}
          {failedRecords.length > 0 && (
            <DrilldownTable
              records={failedRecords}
              onSelectRecord={onSelectRecord}
            />
          )}

          {/* Raw detail fallback */}
          {!stats && !rootCause && failedRecords.length === 0 && (
            <pre className="bg-slate-50 rounded-lg p-3 text-xs text-slate-600 overflow-x-auto max-h-48">
              {JSON.stringify(detail, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
