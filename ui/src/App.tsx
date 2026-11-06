import { Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import SettingsPage from './pages/SettingsPage';
import IngestionPage from './pages/IngestionPage';
import DataGenerationPage from './pages/DataGenerationPage';
import ExecutionPage from './pages/ExecutionPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to="/settings" replace />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/ingestion" element={<IngestionPage />} />
        <Route path="/generation" element={<DataGenerationPage />} />
        <Route path="/execution" element={<ExecutionPage />} />
      </Route>
    </Routes>
  );
}
