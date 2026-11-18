import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { Project } from '../api/types';
import { listProjects, getProject, createProject as apiCreateProject } from '../api/client';

const ACTIVE_PROJECT_KEY = 'rag_tester_active_project_id';

interface ProjectContextValue {
  projects: Project[];
  activeProject: Project | null;
  setActiveProject: (project: Project | null) => void;
  refreshProjects: () => Promise<Project[]>;
  createProject: (
    name: string,
    description?: string,
    dbType?: 'chroma' | 'faiss' | 'milvus_lite',
  ) => Promise<Project>;
  loading: boolean;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProjectState] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  const setActiveProject = useCallback((project: Project | null) => {
    setActiveProjectState(project);
    if (project) {
      localStorage.setItem(ACTIVE_PROJECT_KEY, project.id);
    } else {
      localStorage.removeItem(ACTIVE_PROJECT_KEY);
    }
  }, []);

  const refreshProjects = useCallback(async () => {
    try {
      const data = await listProjects();
      setProjects(data.projects);
      return data.projects;
    } catch {
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const createProject = useCallback(
    async (
      name: string,
      description = '',
      dbType: 'chroma' | 'faiss' | 'milvus_lite' = 'chroma',
    ) => {
      const project = await apiCreateProject(name, description, dbType);
      const freshProjects = await refreshProjects();
      // Find the project in the refreshed list to get the full object with status
      const found = freshProjects.find((p: Project) => p.id === project.id);
      const projectToSet = found ?? (project as Project);
      setActiveProject(projectToSet);
      return projectToSet;
    },
    [refreshProjects, setActiveProject],
  );

  // On mount: load projects and restore active project from localStorage
  useEffect(() => {
    const init = async () => {
      setLoading(true);
      try {
        const data = await listProjects();
        setProjects(data.projects);

        const savedId = localStorage.getItem(ACTIVE_PROJECT_KEY);
        if (savedId) {
          // Try to find in the loaded list first
          const found = data.projects.find((p: Project) => p.id === savedId);
          if (found) {
            setActiveProjectState(found);
          } else {
            // Try to load directly from API (might exist but not in summary)
            try {
              const full = await getProject(savedId);
              setActiveProjectState({
                id: full.id,
                name: full.name,
                description: full.description,
                created_at: full.created_at,
                updated_at: full.updated_at,
                status: full.status,
              });
            } catch {
              // Project no longer exists, clear localStorage
              localStorage.removeItem(ACTIVE_PROJECT_KEY);
            }
          }
        }
      } catch {
        /* ignore on initial load */
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  return (
    <ProjectContext.Provider
      value={{
        projects,
        activeProject,
        setActiveProject,
        refreshProjects,
        createProject,
        loading,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  const ctx = useContext(ProjectContext);
  if (!ctx) {
    throw new Error('useProject must be used within a ProjectProvider');
  }
  return ctx;
}
