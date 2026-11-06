import { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import type { AzureCredentials } from '../api/workbench';

const EMPTY_CREDENTIALS: AzureCredentials = {
  api_key: '',
  endpoint: '',
  deployment: '',
  api_version: '2024-02-15-preview',
};

const AzureCredentialsContext = createContext<{
  credentials: AzureCredentials;
  setCredentials: (credentials: AzureCredentials) => void;
} | null>(null);

export function AzureCredentialsProvider({ children }: { children: ReactNode }) {
  const [credentials, setCredentials] = useState(EMPTY_CREDENTIALS);
  return (
    <AzureCredentialsContext.Provider value={{ credentials, setCredentials }}>
      {children}
    </AzureCredentialsContext.Provider>
  );
}

export function useAzureCredentials() {
  const value = useContext(AzureCredentialsContext);
  if (!value) {
    throw new Error('useAzureCredentials must be used within AzureCredentialsProvider');
  }
  return value;
}

export function hasAzureCredentials(credentials: AzureCredentials): boolean {
  return Object.values(credentials).every((value) => value.trim().length > 0);
}
