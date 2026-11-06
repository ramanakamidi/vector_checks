import { createBrowserRouter, Navigate } from 'react-router-dom';
import { ProjectProvider } from './contexts/ProjectContext';
import AppLayout from './layouts/AppLayout';
import SettingsPage from './pages/SettingsPage';
import IngestionPage from './pages/IngestionPage';
import DataGenerationPage from './pages/DataGenerationPage';
import ExecutionPage from './pages/ExecutionPage';

function RootLayout() {
  return (
    <ProjectProvider>
      <AppLayout />
    </ProjectProvider>
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
    ],
  },
]);
