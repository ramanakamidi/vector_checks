import { createContext, useContext, useState } from 'react';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { AzureCredentials, WorkbenchSession } from '../api/workbench';

interface PdfWorkbenchContextValue {
  generationCredentials: AzureCredentials;
  setGenerationCredentials: Dispatch<SetStateAction<AzureCredentials>>;
  session: WorkbenchSession | null;
  setSession: Dispatch<SetStateAction<WorkbenchSession | null>>;
  selectedIds: string[];
  setSelectedIds: Dispatch<SetStateAction<string[]>>;
  selectedObjectTypes: ObjectType[];
  setSelectedObjectTypes: Dispatch<SetStateAction<ObjectType[]>>;
}

export type ObjectType = 'table' | 'image' | 'graph' | 'text';

const PdfWorkbenchContext = createContext<PdfWorkbenchContextValue | null>(null);

export function PdfWorkbenchProvider({ children }: { children: ReactNode }) {
  const [generationCredentials, setGenerationCredentials] = useState<AzureCredentials>({
    api_key: '',
    endpoint: '',
    deployment: '',
    api_version: '2024-02-15-preview',
  });
  const [session, setSession] = useState<WorkbenchSession | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedObjectTypes, setSelectedObjectTypes] = useState<ObjectType[]>([
    'table',
    'image',
    'graph',
  ]);

  return (
    <PdfWorkbenchContext.Provider value={{
      generationCredentials,
      setGenerationCredentials,
      session,
      setSession,
      selectedIds,
      setSelectedIds,
      selectedObjectTypes,
      setSelectedObjectTypes,
    }}>
      {children}
    </PdfWorkbenchContext.Provider>
  );
}

export function usePdfWorkbench() {
  const context = useContext(PdfWorkbenchContext);
  if (!context) {
    throw new Error('usePdfWorkbench must be used within PdfWorkbenchProvider');
  }
  return context;
}
