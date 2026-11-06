import { useState, useEffect, useCallback } from 'react';
import {
  Play,
  Square,
  Download,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Trash2,
  Eye,
  ChevronDown,
  ChevronRight,
  Activity,
  Zap,
  BarChart3,
  AlertTriangle,
  Loader2,
} from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import type {
  ExecutionConfig,
  ExecutionResult,
  ExecutionListItem,
  TestResult,
  DatasetSummary,
  MockAutStatus,
} from '../api/types';
import {
  listDatasets,
  startExecution,
  listExecutions,
  getExecution,
  deleteExecution,
  downloadReport,
  getExecutionReportUrl,
  startMockAut,
  stopMockAut,
  getMockAutStatus,
} from '../api/client';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import ExecutionPerformanceCharts from '../components/ExecutionPerformanceCharts';

export default function ExecutionPage() {
  const { activeProject } = useProject();
  const projectId = activeProject?.id ?? '';

  // ── Config state ───────────────────────────────────────────────────
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [autEndpoint, setAutEndpoint] = useState('http://localhost:8001/api/query');
  const [runVectorChecks, setRunVectorChecks] = useState(true);
  const [checksToRun, setChecksToRun] = useState<string[]>([
    'completeness',
    'consistency',
    'statistical',
    'neighbor',
  ]);

  // ── Mock AUT state ─────────────────────────────────────────────────
  const [mockAutStatus, setMockAutStatus] = useState<MockAutStatus>({
    running: false,
    url: null,
    pid: null,
    project_id: null,
  });
  const [mockAutLoading, setMockAutLoading] = useState(false);

  // ── Execution state ────────────────────────────────────────────────
  const [executing, setExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState('');
  const [executions, setExecutions] = useState<ExecutionListItem[]>([]);
  const [viewingExecution, setViewingExecution] = useState<ExecutionResult | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [error, setError] = useState('');
  const [loadingView, setLoadingView] = useState(false);

  // ── Load data on mount ─────────────────────────────────────────────
  const loadDatasets = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await listDatasets(projectId);
      setDatasets(res.datasets);
      if (res.datasets.length > 0 && !selectedDataset) {
        setSelectedDataset(res.datasets[0].dataset_id);
      }
    } catch {
      /* ignore */
    }
  }, [projectId, selectedDataset]);

  const loadExecutions = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await listExecutions(projectId);
      setExecutions(res.executions);
    } catch {
      /* ignore */
    }
  }, [projectId]);

  const loadMockAutStatus = useCallback(async () => {
    try {
      const status = await getMockAutStatus();
      setMockAutStatus(status);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadDatasets();
    loadExecutions();
    loadMockAutStatus();
  }, [loadDatasets, loadExecutions, loadMockAutStatus]);

  // ── Handlers ───────────────────────────────────────────────────────

  const handleToggleCheck = (check: string) => {
    setChecksToRun((prev) =>
      prev.includes(check) ? prev.filter((c) => c !== check) : [...prev, check],
    );
  };

  const handleStartMockAut = async () => {
    setMockAutLoading(true);
    try {
      const status = await startMockAut(projectId, 8001);
      setMockAutStatus({
        running: true,
        url: status.url ?? 'http://localhost:8001',
        pid: status.pid ?? null,
        project_id: status.project_id ?? projectId,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setMockAutLoading(false);
    }
  };

  const handleStopMockAut = async () => {
    setMockAutLoading(true);
    try {
      await stopMockAut();
      setMockAutStatus({ running: false, url: null, pid: null, project_id: null });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setMockAutLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first.');
      return;
    }
    setError('');
    setExecuting(true);
    setExecutionProgress('Starting execution...');
    setViewingExecution(null);

    try {
      const config: ExecutionConfig = {
        dataset_id: selectedDataset,
        aut_endpoint: autEndpoint,
        aut_project_id: projectId,
        run_vector_checks: runVectorChecks,
        checks_to_run: runVectorChecks ? checksToRun : [],
      };

      setExecutionProgress('Sending questions to AUT...');
      const result = await startExecution(projectId, config);
      setViewingExecution(result);
      setExecutionProgress('');
      await loadExecutions();
    } catch (e: any) {
      setError(e.message);
      setExecutionProgress('');
    } finally {
      setExecuting(false);
    }
  };

  const handleViewExecution = async (executionId: string) => {
    setLoadingView(true);
    setError('');
    try {
      const result = await getExecution(projectId, executionId);
      setViewingExecution(result);
      setExpandedRows(new Set());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingView(false);
    }
  };

  const handleDeleteExecution = async (executionId: string) => {
    try {
      await deleteExecution(projectId, executionId);
      if (viewingExecution?.execution_id === executionId) {
        setViewingExecution(null);
      }
      await loadExecutions();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDownloadReport = async (executionId: string) => {
    try {
      const blob = await downloadReport(projectId, executionId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `rag_report_${executionId}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const toggleRow = (pairId: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(pairId)) next.delete(pairId);
      else next.add(pairId);
      return next;
    });
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 0.7) return 'text-green-600';
    if (accuracy >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAccuracyBg = (accuracy: number) => {
    if (accuracy >= 0.7) return 'bg-green-100 text-green-800';
    if (accuracy >= 0.5) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  if (!projectId) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        <p>Select a project to begin execution.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-cognizant-blue/10 rounded-lg flex items-center justify-center">
          <Play size={20} className="text-cognizant-blue" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-800">Execute</h2>
          <p className="text-sm text-slate-500">
            Run tests against the AUT, compare answers, and generate reports
          </p>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertTriangle size={18} className="text-red-500 mt-0.5 shrink-0" />
          <div className="flex-1">
            <p className="text-sm text-red-700">{error}</p>
          </div>
          <button onClick={() => setError('')} className="text-red-400 hover:text-red-600">
            <XCircle size={16} />
          </button>
        </div>
      )}

      {/* ── Section A: Execution Configuration ─────────────────────── */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
          Execution Configuration
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Dataset Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Dataset
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue"
            >
              <option value="">Select a dataset...</option>
              {datasets.map((ds) => (
                <option key={ds.dataset_id} value={ds.dataset_id}>
                  {ds.dataset_id} ({ds.total_pairs} pairs)
                </option>
              ))}
            </select>
            {datasets.length === 0 && (
              <p className="text-xs text-slate-400 mt-1">
                No datasets found. Generate one in the Data Generation stage.
              </p>
            )}
          </div>

          {/* AUT Endpoint */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              AUT Endpoint
            </label>
            <input
              type="text"
              value={autEndpoint}
              onChange={(e) => setAutEndpoint(e.target.value)}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-cognizant-blue focus:border-cognizant-blue"
              placeholder="http://localhost:8001/api/query"
            />
          </div>
        </div>

        {/* Mock AUT Controls */}
        <div className="mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity size={18} className="text-slate-500" />
              <div>
                <p className="text-sm font-medium text-slate-700">Mock AUT Server</p>
                <p className="text-xs text-slate-400">
                  {mockAutStatus.running
                    ? `Running at ${mockAutStatus.url} (PID: ${mockAutStatus.pid})`
                    : 'Not running'}
                </p>
              </div>
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                  mockAutStatus.running
                    ? 'bg-green-100 text-green-700'
                    : 'bg-slate-100 text-slate-500'
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    mockAutStatus.running ? 'bg-green-500' : 'bg-slate-400'
                  }`}
                />
                {mockAutStatus.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="flex gap-2">
              {!mockAutStatus.running ? (
                <button
                  onClick={handleStartMockAut}
                  disabled={mockAutLoading}
                  className="px-3 py-1.5 bg-green-600 text-white rounded-lg text-xs font-medium hover:bg-green-700 disabled:opacity-50 flex items-center gap-1"
                >
                  {mockAutLoading ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Play size={12} />
                  )}
                  Start
                </button>
              ) : (
                <button
                  onClick={handleStopMockAut}
                  disabled={mockAutLoading}
                  className="px-3 py-1.5 bg-red-600 text-white rounded-lg text-xs font-medium hover:bg-red-700 disabled:opacity-50 flex items-center gap-1"
                >
                  {mockAutLoading ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Square size={12} />
                  )}
                  Stop
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Vector Store Checks */}
        <div className="mt-4">
          <label className="flex items-center gap-3 cursor-pointer mb-3">
            <input
              type="checkbox"
              checked={runVectorChecks}
              onChange={(e) => setRunVectorChecks(e.target.checked)}
              className="rounded border-slate-300 text-cognizant-blue focus:ring-cognizant-blue"
            />
            <span className="text-sm font-medium text-slate-700">
              Run Vector Store Health Checks
            </span>
          </label>
          {runVectorChecks && (
            <div className="ml-8 grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { id: 'completeness', label: 'Completeness' },
                { id: 'consistency', label: 'Consistency' },
                { id: 'statistical', label: 'Statistical' },
                { id: 'neighbor', label: 'Neighbor' },
              ].map((check) => (
                <label key={check.id} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={checksToRun.includes(check.id)}
                    onChange={() => handleToggleCheck(check.id)}
                    className="rounded border-slate-300 text-cognizant-blue focus:ring-cognizant-blue"
                  />
                  <span className="text-sm text-slate-600">{check.label}</span>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Execute Button */}
        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={handleExecute}
            disabled={executing || !selectedDataset}
            className="px-8 py-3 bg-cognizant-blue text-white rounded-lg text-sm font-medium hover:bg-blue-800 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {executing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <Zap size={16} />
                Execute Tests
              </>
            )}
          </button>
          {executionProgress && (
            <div className="flex items-center gap-2 text-sm text-cognizant-blue">
              <LoadingSpinner size="sm" />
              <span>{executionProgress}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Section B: Execution History ───────────────────────────── */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
          Execution History
        </h3>

        {executions.length === 0 ? (
          <div className="text-center py-8 text-slate-400">
            <Clock size={32} className="mx-auto mb-2 opacity-50" />
            <p className="text-sm">No executions yet</p>
            <p className="text-xs mt-1">Run an execution above to see results here</p>
          </div>
        ) : (
          <div className="space-y-3">
            {executions.map((exec) => {
              const acc = exec.summary?.accuracy ?? 0;
              const accPct = Math.round(acc * 100);
              return (
                <div
                  key={exec.execution_id}
                  className={`border rounded-lg p-4 transition-colors ${
                    viewingExecution?.execution_id === exec.execution_id
                      ? 'border-cognizant-blue bg-blue-50/50'
                      : 'border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div>
                        <p className="text-sm font-medium text-slate-800">
                          {exec.execution_id}
                        </p>
                        <p className="text-xs text-slate-400">
                          {new Date(exec.executed_at).toLocaleString()} &middot;{' '}
                          {exec.duration_seconds}s &middot; Dataset: {exec.dataset_id}
                        </p>
                      </div>
                      <span
                        className={`px-2.5 py-1 rounded-full text-xs font-bold ${getAccuracyBg(acc)}`}
                      >
                        {accPct}% accuracy
                      </span>
                      <span className="text-xs text-slate-500">
                        {exec.summary?.correct_answers ?? 0}/{exec.summary?.total_questions ?? 0} correct
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleViewExecution(exec.execution_id)}
                        className="p-2 text-slate-400 hover:text-cognizant-blue hover:bg-blue-50 rounded-lg transition-colors"
                        title="View Results"
                      >
                        <Eye size={16} />
                      </button>
                      <button
                        onClick={() => handleDownloadReport(exec.execution_id)}
                        className="p-2 text-slate-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                        title="Download Report"
                      >
                        <Download size={16} />
                      </button>
                      <button
                        onClick={() => handleDeleteExecution(exec.execution_id)}
                        className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Section C: Results Panel ───────────────────────────────── */}
      {loadingView && (
        <div className="flex items-center justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {viewingExecution && !loadingView && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <SummaryCard
              icon={<BarChart3 size={24} />}
              value={`${Math.round((viewingExecution.summary?.accuracy ?? 0) * 100)}%`}
              label="Accuracy"
              colorClass={getAccuracyColor(viewingExecution.summary?.accuracy ?? 0)}
            />
            <SummaryCard
              icon={<CheckCircle size={24} />}
              value={String(viewingExecution.summary?.correct_answers ?? 0)}
              label={`of ${viewingExecution.summary?.total_questions ?? 0} Correct`}
              colorClass="text-green-600"
            />
            <SummaryCard
              icon={<Activity size={24} />}
              value={`${Math.round((viewingExecution.summary?.avg_confidence ?? 0) * 100)}%`}
              label="Avg Confidence"
              colorClass="text-cognizant-blue"
            />
            <SummaryCard
              icon={<FileText size={24} />}
              value={`${Math.round((viewingExecution.summary?.avg_context_relevance ?? 0) * 100)}%`}
              label="Avg Context Relevance"
              colorClass="text-purple-600"
            />
            <SummaryCard
              icon={<Clock size={24} />}
              value={`${Math.round(viewingExecution.summary?.avg_response_time_ms ?? 0)}ms`}
              label="Avg Response Time"
              colorClass="text-slate-600"
            />
          </div>

          <ExecutionPerformanceCharts execution={viewingExecution} />

          {/* Download Report Button */}
          <div className="flex justify-end">
            <button
              onClick={() => handleDownloadReport(viewingExecution.execution_id)}
              className="px-4 py-2 bg-cognizant-blue text-white rounded-lg text-sm font-medium hover:bg-blue-800 transition-colors flex items-center gap-2"
            >
              <Download size={16} />
              Download HTML Report
            </button>
          </div>

          {/* Results Table */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
              Test Results ({viewingExecution.results?.length ?? 0} questions)
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="px-3 py-2 text-left font-semibold text-slate-600 w-8" />
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">#</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">
                      Question
                    </th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">
                      Expected
                    </th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">
                      AUT Answer
                    </th>
                    <th className="px-3 py-2 text-center font-semibold text-slate-600">
                      Correct?
                    </th>
                    <th className="px-3 py-2 text-center font-semibold text-slate-600">
                      Confidence
                    </th>
                    <th className="px-3 py-2 text-center font-semibold text-slate-600">
                      Context Rel.
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(viewingExecution.results ?? []).map((r: TestResult, idx: number) => (
                    <ResultRow
                      key={r.pair_id}
                      result={r}
                      index={idx + 1}
                      expanded={expandedRows.has(r.pair_id)}
                      onToggle={() => toggleRow(r.pair_id)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Vector Store Health */}
          {viewingExecution.vector_checks &&
            viewingExecution.vector_checks.status === 'completed' && (
              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
                  Vector Store Health Checks
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {(viewingExecution.vector_checks.checks ?? []).map(
                    (chk: { check: string; passed: boolean; detail: Record<string, any> }) => (
                      <div
                        key={chk.check}
                        className={`rounded-lg border p-4 ${
                          chk.passed
                            ? 'bg-green-50 border-green-200'
                            : 'bg-red-50 border-red-200'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold capitalize">
                            {chk.check}
                          </span>
                          {chk.passed ? (
                            <CheckCircle size={18} className="text-green-600" />
                          ) : (
                            <XCircle size={18} className="text-red-600" />
                          )}
                        </div>
                        <div className="space-y-1">
                          {Object.entries(chk.detail).map(([k, v]) => (
                            <p key={k} className="text-xs text-slate-600">
                              <span className="font-medium">{k}:</span>{' '}
                              {String(v)}
                            </p>
                          ))}
                        </div>
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}
        </>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────

function SummaryCard({
  icon,
  value,
  label,
  colorClass,
}: {
  icon: React.ReactNode;
  value: string;
  label: string;
  colorClass: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
      <div className={`mx-auto mb-2 ${colorClass}`}>{icon}</div>
      <p className={`text-2xl font-bold ${colorClass}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  );
}

function ResultRow({
  result,
  index,
  expanded,
  onToggle,
}: {
  result: TestResult;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const truncate = (s: string, len = 60) =>
    s.length > len ? s.slice(0, len) + '...' : s;

  return (
    <>
      <tr
        onClick={onToggle}
        className={`cursor-pointer border-b transition-colors ${
          result.is_correct
            ? 'hover:bg-green-50/50'
            : 'bg-red-50/30 hover:bg-red-50/60'
        }`}
      >
        <td className="px-3 py-2">
          {expanded ? (
            <ChevronDown size={14} className="text-slate-400" />
          ) : (
            <ChevronRight size={14} className="text-slate-400" />
          )}
        </td>
        <td className="px-3 py-2 text-slate-500">{index}</td>
        <td className="px-3 py-2 max-w-[200px]" title={result.question}>
          {truncate(result.question)}
        </td>
        <td className="px-3 py-2 max-w-[200px]" title={result.expected_answer}>
          {truncate(result.expected_answer)}
        </td>
        <td className="px-3 py-2 max-w-[200px]" title={result.aut_answer}>
          {truncate(result.aut_answer)}
        </td>
        <td className="px-3 py-2 text-center">
          {result.is_correct ? (
            <CheckCircle size={16} className="inline text-green-600" />
          ) : (
            <XCircle size={16} className="inline text-red-500" />
          )}
        </td>
        <td className="px-3 py-2 text-center text-slate-600">
          {Math.round(result.aut_confidence * 100)}%
        </td>
        <td className="px-3 py-2 text-center text-slate-600">
          {Math.round(result.context_relevance * 100)}%
        </td>
      </tr>
      {expanded && (
        <tr className="border-b bg-slate-50">
          <td colSpan={8} className="px-6 py-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  Full Question
                </h4>
                <p className="text-slate-700">{result.question}</p>
              </div>
              <div>
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  Expected Answer
                </h4>
                <p className="text-slate-700">{result.expected_answer}</p>
              </div>
              <div>
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  AUT Answer
                </h4>
                <p className="text-slate-700">{result.aut_answer}</p>
              </div>
              <div>
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  Metrics
                </h4>
                <p className="text-slate-600">
                  Keyword Overlap: {Math.round(result.keyword_overlap * 100)}% &middot;
                  Response Time: {Math.round(result.response_time_ms)}ms
                </p>
                <p className="text-slate-600">
                  Sources: {result.aut_sources?.join(', ') || 'None'}
                </p>
              </div>
            </div>
            {result.aut_context && result.aut_context.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  Retrieved Context
                </h4>
                <div className="space-y-1">
                  {result.aut_context.map((ctx: any, i: number) => (
                    <div
                      key={i}
                      className="bg-white border border-slate-200 rounded p-2 text-xs text-slate-600"
                    >
                      <strong>Context {i + 1}:</strong>{' '}
                      {typeof ctx === 'string'
                        ? truncate(ctx, 200)
                        : truncate(ctx.text ?? JSON.stringify(ctx), 200)}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {result.metadata && Object.keys(result.metadata).length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-cognizant-blue text-xs uppercase mb-1">
                  Metadata Traceability
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {Object.entries(result.metadata).map(([k, v]) => (
                    <div key={k} className="text-xs">
                      <span className="font-medium text-slate-700">{k}:</span>{' '}
                      <span className="text-slate-500">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
