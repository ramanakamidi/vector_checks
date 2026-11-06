import { lazy, Suspense } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { ProjectProvider } from './contexts/ProjectContext';
import { AzureCredentialsProvider } from './contexts/AzureCredentialsContext';
import AppLayout from './layouts/AppLayout';

const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const IngestionPage = lazy(() => import('./pages/IngestionPage'));
const DataGenerationPage = lazy(() => import('./pages/DataGenerationPage'));
const ExecutionPage = lazy(() => import('./pages/ExecutionPage'));
const PdfWorkbenchPage = lazy(() => import('./pages/PdfWorkbenchPage'));
const AnswerImprovementPage = lazy(() => import('./pages/AnswerImprovementPage'));
const VisualCheckPage = lazy(() => import('./pages/VisualCheckPage'));

function PageLoader() {
  return <div className="panel text-center text-sm text-slate-500">Loading workflow…</div>;
}

function RootLayout() {
  return (
    <AzureCredentialsProvider>
      <ProjectProvider>
        <Suspense fallback={<PageLoader />}>
          <AppLayout />
        </Suspense>
      </ProjectProvider>
    </AzureCredentialsProvider>
  );
}

export const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      { path: '/', element: <Navigate to="/settings" replace /> },
      { path: '/settings', element: <SettingsPage /> },
      { path: '/ingestion', element: <IngestionPage /> },
      { path: '/generation', element: <DataGenerationPage /> },
      { path: '/execution', element: <ExecutionPage /> },
      { path: '/pdf-workbench', element: <PdfWorkbenchPage /> },
      { path: '/answer-improvement', element: <AnswerImprovementPage /> },
      { path: '/visual-check', element: <VisualCheckPage /> },
    ],
  },
]);
