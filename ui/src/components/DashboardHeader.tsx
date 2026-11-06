import type { Summary } from '../api/types';

interface Props {
  summary: Summary;
}

interface CardDef {
  label: string;
  value: number;
  color: string;
  bg: string;
  icon: string;
}

export default function DashboardHeader({ summary }: Props) {
  const cards: CardDef[] = [
    {
      label: 'Passed',
      value: summary.passed,
      color: 'text-green-700',
      bg: 'bg-green-50 border-green-200',
      icon: '✅',
    },
    {
      label: 'Failed',
      value: summary.failed,
      color: 'text-red-700',
      bg: 'bg-red-50 border-red-200',
      icon: '❌',
    },
    {
      label: 'Info',
      value: summary.inconclusive,
      color: 'text-blue-700',
      bg: 'bg-blue-50 border-blue-200',
      icon: 'ℹ️',
    },
    {
      label: 'Total',
      value: summary.total,
      color: 'text-slate-700',
      bg: 'bg-slate-50 border-slate-200',
      icon: '📊',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className={`rounded-xl border p-4 ${c.bg} flex items-center gap-4`}
        >
          <span className="text-2xl">{c.icon}</span>
          <div>
            <p className={`text-3xl font-bold ${c.color}`}>{c.value}</p>
            <p className="text-sm text-slate-500">{c.label}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
