interface Props {
  status: 'pass' | 'fail' | 'info' | 'warning' | 'success' | 'error' | 'running' | string;
  label?: string;
}

const STYLES: Record<string, string> = {
  pass: 'bg-green-100 text-green-700',
  success: 'bg-green-100 text-green-700',
  fail: 'bg-red-100 text-red-700',
  error: 'bg-red-100 text-red-700',
  info: 'bg-blue-100 text-blue-700',
  warning: 'bg-amber-100 text-amber-700',
  running: 'bg-blue-100 text-blue-700',
  created: 'bg-slate-100 text-slate-500',
  configured: 'bg-blue-100 text-blue-700',
  ingested: 'bg-indigo-100 text-indigo-700',
  generated: 'bg-purple-100 text-purple-700',
  executed: 'bg-green-100 text-green-700',
};

export default function StatusBadge({ status, label }: Props) {
  const style = STYLES[status] ?? 'bg-slate-100 text-slate-500';
  return (
    <span className={`inline-flex items-center text-xs font-semibold px-2.5 py-0.5 rounded-full ${style}`}>
      {label ?? status.toUpperCase()}
    </span>
  );
}
