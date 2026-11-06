import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { getEmbeddingData } from '../api/client';
import type { EmbeddingPoint } from '../api/types';

interface Props {
  onSelectRecord: (id: string) => void;
}

const FAULT_COLORS: Record<string, string> = {
  healthy: '#94a3b8',
  zero_vector: '#ef4444',
  nan_vector: '#f97316',
  low_norm: '#eab308',
  high_norm: '#a855f7',
  duplicate: '#3b82f6',
  similarity_candidate: '#3b82f6',
  orphan: '#ec4899',
  outlier: '#14b8a6',
};

export default function EmbeddingPlot({ onSelectRecord }: Props) {
  const [points, setPoints] = useState<EmbeddingPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [colorBy, setColorBy] = useState<string>('fault_type');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getEmbeddingData(2000)
      .then((data) => {
        if (!cancelled) setPoints(data);
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
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-slate-400">
        Loading embedding visualization…
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-red-500">
        Failed to load embeddings: {error}
      </div>
    );
  }

  if (points.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-slate-400">
        No embedding data available.
      </div>
    );
  }

  /* Gather metadata keys for color-by dropdown */
  const metaKeys = Array.from(
    new Set(points.flatMap((p) => Object.keys(p.metadata ?? {}))),
  );

  /* Group points by the selected color dimension */
  const groups: Record<string, EmbeddingPoint[]> = {};
  for (const p of points) {
    let key: string;
    if (colorBy === 'fault_type') {
      key = p.fault_type ?? 'healthy';
    } else {
      key = String(p.metadata?.[colorBy] ?? 'unknown');
    }
    (groups[key] ??= []).push(p);
  }

  const traces = Object.entries(groups).map(([name, pts]) => ({
    x: pts.map((p) => p.x),
    y: pts.map((p) => p.y),
    text: pts.map(
      (p) =>
        `ID: ${p.id}\nFault: ${p.fault_type ?? 'none'}\n${Object.entries(p.metadata ?? {})
          .slice(0, 4)
          .map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`)
          .join('\n')}`,
    ),
    customdata: pts.map((p) => p.id),
    type: 'scattergl' as const,
    mode: 'markers' as const,
    name,
    marker: {
      size: 5,
      color: FAULT_COLORS[name] ?? undefined,
      opacity: 0.7,
    },
    hoverinfo: 'text' as const,
  }));

  const handleClick = (event: any) => {
    const point = event?.points?.[0];
    if (point?.customdata) {
      onSelectRecord(String(point.customdata));
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-slate-700">
          Embedding Space ({points.length.toLocaleString()} points)
        </h3>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-500">Color by:</label>
          <select
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value)}
            className="border border-slate-300 rounded px-2 py-1 text-sm"
          >
            <option value="fault_type">Fault Type</option>
            {metaKeys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>
      </div>

      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 450,
          margin: { l: 40, r: 20, t: 20, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: '#f8fafc',
          xaxis: { title: { text: 'UMAP-1' }, gridcolor: '#e2e8f0' },
          yaxis: { title: { text: 'UMAP-2' }, gridcolor: '#e2e8f0' },
          legend: { orientation: 'h', y: -0.15 },
          hovermode: 'closest',
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
        onClick={handleClick}
      />
    </div>
  );
}
