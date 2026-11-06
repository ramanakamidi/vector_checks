import ProjectSelector from './ProjectSelector';

export default function TopBar() {
  return (
    <header className="topbar">
      <div className="flex items-center gap-3">
          <div className="brand-mark">
            <span>V</span>
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight leading-tight">
              Vector Quality Studio
            </h1>
            <p className="text-xs text-indigo-200 leading-tight">
              Assurance &amp; GenAI evaluation workspace
            </p>
          </div>
      </div>

      <ProjectSelector />
    </header>
  );
}
