import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';
import type { ExecutionResult } from '../api/types';

const COLORS = ['#059669', '#e11d48'];

export default function ExecutionPerformanceCharts({ execution }: { execution: ExecutionResult }) {
  const results = execution.results ?? [];
  const passed = results.filter((result) => result.is_correct).length;
  const failed = Math.max(0, results.length - passed);
  const ratio = [
    { name: 'Passed', value: passed },
    { name: 'Failed', value: failed },
  ];
  const average = (values: number[]) => values.length
    ? values.reduce((total, value) => total + Math.max(0, Math.min(1, value)), 0) / values.length
    : 0;
  const scores = [
    { metric: 'Answer accuracy', score: average(results.map((result) => result.is_correct ? 1 : 0)) * 100 },
    { metric: 'Confidence', score: average(results.map((result) => result.aut_confidence)) * 100 },
    { metric: 'Context relevance', score: average(results.map((result) => result.context_relevance)) * 100 },
    { metric: 'Keyword overlap', score: average(results.map((result) => result.keyword_overlap)) * 100 },
  ];

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <section className="panel">
        <div className="mb-3">
          <h2 className="text-sm font-bold text-slate-800">Passed vs failed</h2>
          <p className="mt-1 text-xs text-slate-500">Test cases in this execution</p>
        </div>
        <div className="chart-container">
          {results.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={ratio} dataKey="value" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3}>
                  {ratio.map((entry, index) => <Cell key={entry.name} fill={COLORS[index]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <p className="empty-state">No test results to chart.</p>}
          <div className="chart-center">
            <strong>{results.length ? `${Math.round((passed / results.length) * 100)}%` : '—'}</strong>
            <span>pass rate</span>
          </div>
        </div>
        <div className="chart-legend">
          <span><i style={{ backgroundColor: COLORS[0] }} />Passed <strong>{passed}</strong></span>
          <span><i style={{ backgroundColor: COLORS[1] }} />Failed <strong>{failed}</strong></span>
        </div>
      </section>
      <section className="panel">
        <div className="mb-3">
          <h2 className="text-sm font-bold text-slate-800">Evaluation metrics</h2>
          <p className="mt-1 text-xs text-slate-500">Average observed quality scores</p>
        </div>
        <div className="chart-bar">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={scores} margin={{ top: 8, right: 12, bottom: 8, left: -12 }}>
              <CartesianGrid stroke="#edf0f5" vertical={false} />
              <XAxis dataKey="metric" tick={{ fontSize: 10, fill: '#667085' }} interval={0} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#667085' }} />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Score']} />
              <Bar dataKey="score" fill="#4f46e5" radius={[5, 5, 0, 0]} maxBarSize={44} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
