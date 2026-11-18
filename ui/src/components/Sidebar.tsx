import { useLocation, useNavigate } from 'react-router-dom';
import { Settings, Play, FileSearch, FileCheck2, Eye } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';

const NAV_ITEMS = [
  { path: '/pdf-workbench', label: 'Test Data Generation', icon: FileSearch, stage: 1 },
  { path: '/execution', label: 'Execution', icon: Play, stage: 2 },
  { path: '/answer-improvement', label: 'Answer Improvement', icon: FileCheck2, stage: 3 },
  { path: '/visual-check', label: 'Embedding Explorer', icon: Eye, stage: 4 },
  { path: '/app-under-test', label: 'Configure', icon: Settings, stage: 5 },
] as const;

export default function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { activeProject } = useProject();

  return (
    <aside className="w-60 bg-white border-r border-slate-200 flex flex-col">
      {/* Stage navigation */}
      <nav className="flex-1 py-4">
        <p className="px-5 mb-3 text-[10px] font-bold text-slate-400 uppercase tracking-[0.16em]">
          Workflows
        </p>
        {NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          const disabled = !activeProject && !['/pdf-workbench', '/app-under-test'].includes(item.path);

          return (
            <button
              key={item.path}
              onClick={() => !disabled && navigate(item.path)}
              disabled={disabled}
              aria-label={item.label}
              title={item.label}
              className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-indigo-50 text-cognizant-blue border-r-[3px] border-cognizant-blue font-semibold'
                  : disabled
                    ? 'text-slate-300 cursor-not-allowed'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <span
                className={`flex items-center justify-center w-7 h-7 rounded-lg text-xs font-bold ${
                  isActive
                    ? 'bg-cognizant-blue text-white'
                    : disabled
                      ? 'bg-slate-100 text-slate-300'
                      : 'bg-slate-100 text-slate-500'
                }`}
              >
                {item.stage}
              </span>
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Project info footer */}
      {activeProject && (
        <div className="p-4 border-t border-slate-200">
          <p className="text-xs text-slate-400 mb-1">Active Project</p>
          <p className="text-sm font-medium text-slate-700 truncate">
            {activeProject.name}
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            Status: <span className="capitalize">{activeProject.status}</span>
          </p>
        </div>
      )}
    </aside>
  );
}
