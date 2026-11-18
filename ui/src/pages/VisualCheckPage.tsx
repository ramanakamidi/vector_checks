import { useEffect, useMemo, useRef, useState } from 'react';
import { Eye, LoaderCircle, RefreshCw, Upload } from 'lucide-react';
import {
  CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts';
import { getProjectVisualization } from '../api/workbench';
import type { VisualPoint, VisualizationResponse } from '../api/workbench';
import { useProject } from '../contexts/ProjectContext';

const COLORS = ['#4f46e5', '#0f766e', '#e87924', '#be185d', '#0891b2', '#7c3aed', '#65a30d', '#dc2626'];
const NOISE_COLOR = '#94a3b8';
const pointColor = (cluster: number) => cluster === -1 ? NOISE_COLOR : COLORS[cluster % COLORS.length];
const clusterLabel = (cluster: number) => cluster === -1 ? 'Noise (-1)' : `Cluster ${cluster}`;

export default function VisualCheckPage() {
  const { activeProject } = useProject();
  const [sampleSize, setSampleSize] = useState(300);
  const [minClusterSize, setMinClusterSize] = useState(2);
  const [clusterFilter, setClusterFilter] = useState('all');
  const [workbook, setWorkbook] = useState<File | undefined>();
  const [visualization, setVisualization] = useState<VisualizationResponse | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const requestVersion = useRef(0);

  useEffect(() => {
    requestVersion.current += 1;
    setVisualization(null);
    setSelectedId('');
    setClusterFilter('all');
    setError('');
    setBusy(false);
  }, [activeProject?.id]);

  const loadVisualization = async () => {
    if (!activeProject) return;
    const version = ++requestVersion.current;
    setBusy(true);
    setError('');
    try {
      const result = await getProjectVisualization(activeProject.id, sampleSize, minClusterSize, workbook);
      if (requestVersion.current === version) {
        setVisualization(result);
        setSelectedId('');
        setClusterFilter('all');
      }
    } catch (cause) {
      if (requestVersion.current === version) {
        setError(cause instanceof Error ? cause.message : 'Could not build the UMAP visualization.');
      }
    } finally {
      if (requestVersion.current === version) setBusy(false);
    }
  };

  const clusters = useMemo(
    () => Array.from(new Set(visualization?.points.map((point) => point.cluster) ?? [])).sort((a, b) => a - b),
    [visualization],
  );
  const visiblePoints = useMemo(
    () => (visualization?.points ?? []).filter((point) => clusterFilter === 'all' || String(point.cluster) === clusterFilter),
    [clusterFilter, visualization],
  );
  const selected = visiblePoints.find((point) => point.id === selectedId);
  const linkedPoints = visiblePoints.filter((point) => point.qa?.length);
  const validVisualizationSettings = Number.isInteger(sampleSize)
    && sampleSize >= 3 && sampleSize <= 10000
    && Number.isInteger(minClusterSize)
    && minClusterSize >= 2 && minClusterSize <= 100;

  const series = clusters
    .filter((cluster) => clusterFilter === 'all' || String(cluster) === clusterFilter)
    .map((cluster) => ({
      cluster,
      points: visiblePoints.filter((point) => point.cluster === cluster),
    }));
  const setVisibleCluster = (cluster: number) => {
    setClusterFilter((current) => current === String(cluster) ? 'all' : String(cluster));
    setSelectedId('');
  };

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Embedding diagnostics</p>
          <h1>Embedding Explorer</h1>
          <p>Explore project chunks in 2D; cluster labels are computed on the same UMAP coordinates shown here.</p>
        </div>
      </header>

      <section className="panel space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <label className="field-label">Sample size
            <input type="number" min={3} max={10000} step={1} value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} />
          </label>
          <label className="field-label">Minimum cluster size
            <input type="number" min={2} max={100} step={1} value={minClusterSize} onChange={(event) => setMinClusterSize(Number(event.target.value))} />
          </label>
          <label className="field-label">Visible clusters
            <select value={clusterFilter} onChange={(event) => {
              setClusterFilter(event.target.value);
              setSelectedId('');
            }}>
              <option value="all">All clusters</option>
              {clusters.map((cluster) => <option key={cluster} value={String(cluster)}>{clusterLabel(cluster)}</option>)}
            </select>
          </label>
        </div>
        <button className="button-primary" onClick={loadVisualization} disabled={!activeProject || !validVisualizationSettings || busy}>
          {busy ? <LoaderCircle size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          {busy ? 'Building visualization…' : 'Build / refresh UMAP'}
        </button>
        {busy && <p className="field-help" role="status">UMAP is computing your sample. The first build after a server restart may take longer while it initializes.</p>}
        {!validVisualizationSettings && <p className="field-help">Sample size must be 3–10,000 and minimum cluster size must be 2–100.</p>}
      </section>

      {error && <div className="alert-error" role="alert">{error}</div>}
      {visualization && (
        <>
          <section className="metric-grid">
            <div className="metric-card"><strong>{visualization.total_records}</strong><span>Project records</span></div>
            <div className="metric-card"><strong>{visualization.sampled_records}</strong><span>Sampled</span></div>
            <div className="metric-card"><strong>{visualization.projected_records}</strong><span>Projected</span></div>
            <div className="metric-card"><strong>{visualization.skipped_records}</strong><span>Skipped vectors</span></div>
          </section>
          {visualization.message && <div className="alert-warning">{visualization.message}</div>}
          {visualization.points.length > 0 && (
            <section className="panel space-y-4">
              <div className="section-title"><div className="icon-tile"><Eye size={18} /></div><div><h2>UMAP · HDBSCAN</h2><p>{visiblePoints.length} visible chunks · select a point to inspect its record.</p></div></div>
              <div className="visualization-chart">
                <div className="visualization-plot">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 24, bottom: 24, left: 10 }}>
                      <CartesianGrid stroke="#e8ebf2" />
                      <XAxis type="number" dataKey="x" name="UMAP-1" tick={{ fontSize: 10 }} />
                      <YAxis type="number" dataKey="y" name="UMAP-2" tick={{ fontSize: 10 }} />
                      <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value, name, item) => {
                          const point = item.payload as VisualPoint;
                          if (name === 'UMAP-1' || name === 'UMAP-2') {
                            return [value, name];
                          }
                          return [point.id, clusterLabel(point.cluster)];
                        }}
                        labelFormatter={(_, payload) => {
                          const point = payload?.[0]?.payload as VisualPoint | undefined;
                          return point ? `${point.id} · ${String(point.metadata.pdf_name ?? '')}` : '';
                        }}
                      />
                      {series.map(({ cluster, points }) => (
                        <Scatter
                          key={cluster}
                          name={clusterLabel(cluster)}
                          data={points}
                          fill={pointColor(cluster)}
                          onClick={(entry) => {
                            const point = entry.payload as VisualPoint | undefined;
                            if (point?.id) setSelectedId(point.id);
                          }}
                          cursor="pointer"
                        />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <div className="cluster-legend" role="group" aria-label="Filter by cluster">
                  {clusters.map((cluster) => {
                    const active = clusterFilter === String(cluster);
                    const count = visualization.points.filter((point) => point.cluster === cluster).length;
                    return (
                      <button
                        key={cluster}
                        type="button"
                        className={`cluster-legend-item${active ? ' cluster-legend-item-active' : ''}`}
                        aria-pressed={active}
                        aria-label={`${clusterLabel(cluster)}, ${count} chunks`}
                        onClick={() => setVisibleCluster(cluster)}
                      >
                        <span className="cluster-dot" style={{ backgroundColor: pointColor(cluster) }} />
                        <span>{clusterLabel(cluster)}</span>
                        <span className="cluster-legend-count">{count}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
              {selected && (
                <div className="record-detail">
                  <div className="flex items-center gap-2"><span className="cluster-dot" style={{ backgroundColor: pointColor(selected.cluster) }} /><strong>{clusterLabel(selected.cluster)}</strong><span className="text-xs text-slate-500">{selected.id}</span></div>
                  <p>{selected.text || 'No document text stored.'}</p>
                  <pre>{JSON.stringify(selected.metadata, null, 2)}</pre>
                </div>
              )}
            </section>
          )}

          <section className="panel space-y-4">
            <div className="section-title"><div className="icon-tile"><Upload size={18} /></div><div><h2>Q&amp;A linked to UMAP chunks</h2><p>Upload a PDF Workbench workbook to match text and table Q&amp;A to visible points.</p></div></div>
            <label className="dropzone dropzone-compact">
              <input
                type="file"
                accept=".xlsx"
                onChange={(event) => {
                  setWorkbook(event.target.files?.[0] ?? undefined);
                  setVisualization((current) => current ? {
                    ...current,
                    points: current.points.map((point) => {
                      const unlinked = { ...point };
                      delete unlinked.qa;
                      return unlinked;
                    }),
                    qa_matched_points: undefined,
                    qa_rows_loaded: undefined,
                  } : null);
                }}
              />
              <Upload size={21} className="text-indigo-600" />
              <span className="font-semibold">{workbook?.name ?? 'Upload Q&A workbook (.xlsx)'}</span>
              <span className="text-sm text-slate-500">The chart refresh uses workbook rows to attach matching Q&amp;A to each chunk.</span>
            </label>
            {workbook && (
              <div className="flex gap-2">
                <button className="button-primary" onClick={loadVisualization} disabled={busy || !activeProject || !validVisualizationSettings}>
                  {busy ? <LoaderCircle size={16} className="animate-spin" /> : <RefreshCw size={16} />}
                  Match workbook to visible chunks
                </button>
                <button className="button-secondary" onClick={() => { setWorkbook(undefined); setVisualization(null); }}>
                  Clear workbook
                </button>
              </div>
            )}
            {workbook && visualization.qa_rows_loaded !== undefined && (
              <p className="text-sm text-slate-500">
                Found Q&amp;A for {linkedPoints.length} of {visiblePoints.length} visible chunks ({visualization.qa_rows_loaded} workbook rows loaded).
              </p>
            )}
            {linkedPoints.length > 0 && (
              <div className="grid gap-4 lg:grid-cols-2">
                {linkedPoints.map((point) => (
                  <article className="qa-cluster-card" key={point.id} style={{ borderTopColor: pointColor(point.cluster) }}>
                    <div className="flex items-center gap-2"><span className="cluster-dot" style={{ backgroundColor: pointColor(point.cluster) }} /><strong>{clusterLabel(point.cluster)}</strong></div>
                    <p className="text-xs text-slate-500">Chunk ID: {point.id}</p>
                    <p className="qa-chunk-text">{point.text || 'No document text stored.'}</p>
                    {point.qa?.map((row, index) => (
                      <div className="qa-pair" key={`${point.id}-${index}`}>
                        <span className="status-pill">{row.category || row.source_type}</span>
                        <p><strong>Q:</strong> {row.question}</p>
                        <p><strong>A:</strong> {row.expected_answer}</p>
                      </div>
                    ))}
                  </article>
                ))}
              </div>
            )}
            {workbook && visualization.qa_rows_loaded !== undefined && linkedPoints.length === 0 && (
              <p className="empty-state">No workbook Q&amp;A matched the visible chunks. Confirm the workbook and project contain the same PDF content.</p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
