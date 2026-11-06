import { useState } from 'react';
import { ChevronDown, Plus, FolderOpen } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import ProjectCreateDialog from './ProjectCreateDialog';

export default function ProjectSelector() {
  const { projects, activeProject, setActiveProject } = useProject();
  const [open, setOpen] = useState(false);
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg text-sm transition-colors"
      >
        <FolderOpen size={16} />
        <span className="max-w-[200px] truncate">
          {activeProject ? activeProject.name : 'Select Project'}
        </span>
        <ChevronDown size={14} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-2 w-72 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 overflow-hidden">
            <div className="p-2 border-b border-slate-100">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide px-2 py-1">
                Projects
              </p>
            </div>

            <div className="max-h-64 overflow-y-auto">
              {projects.length === 0 && (
                <p className="text-sm text-slate-400 px-4 py-3 text-center">
                  No projects yet
                </p>
              )}
              {projects.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setActiveProject(p);
                    setOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 hover:bg-slate-50 transition-colors flex items-center gap-3 ${
                    activeProject?.id === p.id ? 'bg-blue-50' : ''
                  }`}
                >
                  <div
                    className={`w-2 h-2 rounded-full ${
                      activeProject?.id === p.id ? 'bg-cognizant-blue' : 'bg-slate-300'
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-slate-800 truncate">{p.name}</p>
                    <p className="text-xs text-slate-400 truncate">{p.description || 'No description'}</p>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      p.status === 'executed'
                        ? 'bg-green-100 text-green-700'
                        : p.status === 'configured'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-slate-100 text-slate-500'
                    }`}
                  >
                    {p.status}
                  </span>
                </button>
              ))}
            </div>

            <div className="p-2 border-t border-slate-100">
              <button
                onClick={() => {
                  setOpen(false);
                  setShowCreate(true);
                }}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-cognizant-blue hover:bg-blue-50 rounded-lg transition-colors font-medium"
              >
                <Plus size={16} />
                New Project
              </button>
            </div>
          </div>
        </>
      )}

      <ProjectCreateDialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
      />
    </div>
  );
}
