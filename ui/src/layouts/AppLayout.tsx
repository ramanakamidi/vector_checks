import { Outlet } from 'react-router-dom';
import TopBar from '../components/TopBar';
import Sidebar from '../components/Sidebar';
import { useProject } from '../contexts/ProjectContext';
import { FolderOpen } from 'lucide-react';
import { useLocation } from 'react-router-dom';

export default function AppLayout() {
  const { activeProject, loading } = useProject();
  const { pathname } = useLocation();
  const projectOptional = ['/', '/pdf-workbench', '/app-under-test'].includes(pathname);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cognizant-blue" />
            </div>
          ) : !activeProject && !projectOptional ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <FolderOpen size={48} className="mb-4" />
              <p className="text-lg font-medium">No Project Selected</p>
              <p className="text-sm mt-1">
                Select or create a project from the top bar to get started.
              </p>
            </div>
          ) : (
            <div className="workspace-content">
              <Outlet />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
