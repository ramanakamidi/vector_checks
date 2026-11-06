import ProjectSelector from './ProjectSelector';

export default function TopBar() {
  return (
    <header className="bg-cognizant-blue text-white px-6 py-3 shadow-lg flex items-center justify-between">
      <div className="flex items-center gap-4">
        {/* Cognizant Logo Placeholder */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-white rounded flex items-center justify-center">
            <span className="text-cognizant-blue font-bold text-sm">C</span>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight leading-tight">
              RAG Testing Platform
            </h1>
            <p className="text-xs text-blue-200 leading-tight">
              Vector DB Health &amp; Evaluation
            </p>
          </div>
        </div>
      </div>

      <ProjectSelector />
    </header>
  );
}
