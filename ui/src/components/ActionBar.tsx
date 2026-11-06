import { exportHtmlUrl, exportJsonUrl } from '../api/client';

interface Props {
  onRerun: () => void;
  loading: boolean;
}

export default function ActionBar({ onRerun, loading }: Props) {
  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-slate-200">
      <button
        onClick={() => handleDownload(exportHtmlUrl(), 'report.html')}
        className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
      >
        <span>📄</span>
        Export HTML Report
      </button>

      <button
        onClick={() => handleDownload(exportJsonUrl(), 'results.json')}
        className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
      >
        <span>📋</span>
        Export JSON
      </button>

      <div className="flex-1" />

      <button
        onClick={onRerun}
        disabled={loading}
        className="inline-flex items-center gap-2 px-5 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
      >
        {loading ? (
          <>
            <span className="animate-spin">⏳</span>
            Running…
          </>
        ) : (
          <>
            <span>🔄</span>
            Re-run Checks
          </>
        )}
      </button>
    </div>
  );
}
