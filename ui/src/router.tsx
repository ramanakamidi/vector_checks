import { lazy, Suspense } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { ProjectProvider } from './contexts/ProjectContext';
import { AzureCredentialsProvider } from './contexts/AzureCredentialsContext';
import { PdfWorkbenchProvider } from './contexts/PdfWorkbenchContext';
import AppLayout from './layouts/AppLayout';

const AppUnderTestPage = lazy(() => import('./pages/AppUnderTestPage'));
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
        <PdfWorkbenchProvider>
          <Suspense fallback={<PageLoader />}>
            <AppLayout />
          </Suspense>
        </PdfWorkbenchProvider>
      </ProjectProvider>
    </AzureCredentialsProvider>
  );
}

export const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      { path: '/', element: <Navigate to="/pdf-workbench" replace /> },
      { path: '/app-under-test', element: <AppUnderTestPage /> },
      { path: '/execution', element: <ExecutionPage /> },
      { path: '/pdf-workbench', element: <PdfWorkbenchPage /> },
      { path: '/answer-improvement', element: <AnswerImprovementPage /> },
      { path: '/visual-check', element: <VisualCheckPage /> },
      { path: '*', element: <Navigate to="/pdf-workbench" replace /> },
    ],
  },
]);
