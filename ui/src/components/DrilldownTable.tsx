import { useState, useMemo } from 'react';

interface Props {
  records: any[];
  onSelectRecord: (id: string) => void;
}

export default function DrilldownTable({ records, onSelectRecord }: Props) {
  const [open, setOpen] = useState(false);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [filter, setFilter] = useState('');

  if (records.length === 0) return null;

  /* Derive column headers from first record */
  const columns = useMemo(() => {
    const keys = new Set<string>();
    for (const r of records.slice(0, 20)) {
      if (typeof r === 'object' && r !== null) {
        Object.keys(r).forEach((k) => keys.add(k));
      }
    }
    return Array.from(keys);
  }, [records]);

  /* Filter + sort */
  const processed = useMemo(() => {
    let rows = [...records];

    if (filter) {
      const q = filter.toLowerCase();
      rows = rows.filter((r) =>
        Object.values(r).some((v) =>
          String(v).toLowerCase().includes(q),
        ),
      );
    }

    if (sortCol) {
      rows.sort((a, b) => {
        const va = a[sortCol] ?? '';
        const vb = b[sortCol] ?? '';
        const cmp = String(va).localeCompare(String(vb), undefined, { numeric: true });
        return sortAsc ? cmp : -cmp;
      });
    }

    return rows;
  }, [records, filter, sortCol, sortAsc]);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  };

  const getRecordId = (row: any): string => {
    return String(row.id ?? row.record_id ?? row.doc_id ?? JSON.stringify(row).slice(0, 40));
  };

  const truncate = (val: any, max = 60): string => {
    const s = String(val ?? '');
    return s.length > max ? s.slice(0, max) + '…' : s;
  };

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
      >
        <svg
          className={`w-4 h-4 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        {open ? 'Hide' : 'Show'} {records.length} affected record{records.length !== 1 ? 's' : ''}
      </button>

      {open && (
        <div className="mt-3 space-y-2">
          <input
            type="text"
            placeholder="Filter records…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border border-slate-300 rounded px-3 py-1.5 text-sm w-full max-w-xs focus:ring-2 focus:ring-blue-500 outline-none"
          />

          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50">
                  {columns.map((col) => (
                    <th
                      key={col}
                      onClick={() => handleSort(col)}
                      className="px-3 py-2 text-left text-xs font-semibold text-slate-600 cursor-pointer hover:bg-slate-100 select-none whitespace-nowrap"
                    >
                      {col}
                      {sortCol === col && (
                        <span className="ml-1">{sortAsc ? '↑' : '↓'}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {processed.slice(0, 100).map((row, i) => (
                  <tr
                    key={i}
                    className="border-t border-slate-100 hover:bg-blue-50 transition-colors"
                  >
                    {columns.map((col, colIdx) => (
                      <td key={col} className="px-3 py-2 whitespace-nowrap max-w-[200px] truncate">
                        {colIdx === 0 ? (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectRecord(getRecordId(row));
                            }}
                            className="text-blue-600 hover:text-blue-800 hover:underline cursor-pointer font-medium text-left"
                            title={`Inspect record ${truncate(row[col], 80)}`}
                          >
                            {truncate(row[col])}
                          </button>
                        ) : (
                          <span className="text-slate-700">{truncate(row[col])}</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {processed.length > 100 && (
              <p className="text-xs text-slate-400 px-3 py-2 bg-slate-50">
                Showing 100 of {processed.length} records
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
