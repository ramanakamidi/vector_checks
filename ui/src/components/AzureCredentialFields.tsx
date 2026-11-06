import { useAzureCredentials } from '../contexts/AzureCredentialsContext';

export default function AzureCredentialFields() {
  const { credentials, setCredentials } = useAzureCredentials();
  const update = (key: keyof typeof credentials, value: string) => {
    setCredentials({ ...credentials, [key]: value });
  };

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <label className="field-label md:col-span-2">
        Azure OpenAI API key
        <input
          type="password"
          autoComplete="new-password"
          value={credentials.api_key}
          onChange={(event) => update('api_key', event.target.value)}
          placeholder="Enter key for this session"
        />
      </label>
      <label className="field-label">
        Endpoint
        <input
          value={credentials.endpoint}
          onChange={(event) => update('endpoint', event.target.value)}
          placeholder="https://your-resource.openai.azure.com"
        />
      </label>
      <label className="field-label">
        Deployment
        <input
          value={credentials.deployment}
          onChange={(event) => update('deployment', event.target.value)}
          placeholder="Deployment name"
        />
      </label>
      <label className="field-label md:col-span-2">
        API version
        <input
          value={credentials.api_version}
          onChange={(event) => update('api_version', event.target.value)}
          placeholder="2024-02-15-preview"
        />
      </label>
      <p className="md:col-span-2 text-xs text-slate-500">
        Credentials remain in browser memory and are sent only to this local API for the requested operation.
      </p>
    </div>
  );
}
