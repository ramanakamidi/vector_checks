import type { StatusFilter, Summary } from '../api/types';

interface Props {
  statusFilter: StatusFilter;
  onStatusFilter: (f: StatusFilter) => void;
  searchQuery: string;
  onSearchQuery: (q: string) => void;
  summary: Summary;
}

interface FilterBtn {
  key: StatusFilter;
  label: string;
  icon: string;
  count: number;
  activeClass: string;
}

export default function FilterToolbar({
  statusFilter,
  onStatusFilter,
  searchQuery,
  onSearchQuery,
  summary,
}: Props) {
  const buttons: FilterBtn[] = [
    { key: 'all', label: 'All', icon: '📋', count: summary.total, activeClass: 'bg-slate-700 text-white' },
    { key: 'pass', label: 'Pass', icon: '✅', count: summary.passed, activeClass: 'bg-green-600 text-white' },
    { key: 'fail', label: 'Fail', icon: '❌', count: summary.failed, activeClass: 'bg-red-600 text-white' },
    { key: 'info', label: 'Info', icon: 'ℹ️', count: summary.inconclusive, activeClass: 'bg-blue-600 text-white' },
  ];

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-2">
        {buttons.map((b) => (
          <button
            key={b.key}
            onClick={() => onStatusFilter(b.key)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              statusFilter === b.key
                ? b.activeClass
                : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            <span>{b.icon}</span>
            {b.label}
            <span
              className={`ml-1 px-1.5 py-0.5 rounded-full text-xs ${
                statusFilter === b.key
                  ? 'bg-white/20 text-white'
                  : 'bg-slate-100 text-slate-500'
              }`}
            >
              {b.count}
            </span>
          </button>
        ))}
      </div>

      <div className="flex-1 min-w-[200px]">
        <input
          type="text"
          placeholder="Search checks…"
          value={searchQuery}
          onChange={(e) => onSearchQuery(e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
        />
      </div>
    </div>
  );
}
