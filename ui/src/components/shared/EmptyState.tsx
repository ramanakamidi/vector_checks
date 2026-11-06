import type { ReactNode } from 'react';
import { Inbox } from 'lucide-react';

interface Props {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export default function EmptyState({ icon, title, description, action }: Props) {
  return (
    <div className="text-center py-12 text-slate-400">
      <div className="mb-3 flex justify-center">
        {icon ?? <Inbox size={40} className="opacity-50" />}
      </div>
      <p className="text-sm font-medium text-slate-500">{title}</p>
      {description && <p className="text-xs mt-1">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
