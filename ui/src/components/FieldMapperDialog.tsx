import { useState, useEffect, useCallback } from 'react';
import { getFieldMapper, saveFieldMapper } from '../api/client';

/* ── Default config matching Python FieldMapper dataclass ── */
const DEFAULTS = {
  display_fields: ['doc_id', 'doc_name', 'page_number', 'object_type', 'text'],
  doc_id_field: 'doc_id',
  doc_name_field: 'doc_name',
  text_field: 'text',
  page_field: 'page_number',
  object_type_field: 'object_type',
  ingestion_date_field: 'ingestion_date',
  required_fields: ['doc_id', 'artifact_type'],
  similar_group_field: 'doc_id',
  different_group_field: 'object_type',
  drift_time_field: 'ingestion_date',
};

const COMMON_DISPLAY_FIELDS = [
  'doc_id',
  'doc_name',
  'page_number',
  'chunk_index',
  'object_type',
  'artifact_type',
  'source_url',
  'text',
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function FieldMapperDialog({ open, onClose }: Props) {
  /* ── State ── */
  const [displayFields, setDisplayFields] = useState<string[]>(DEFAULTS.display_fields);
  const [customFieldInput, setCustomFieldInput] = useState('');

  const [docIdField, setDocIdField] = useState(DEFAULTS.doc_id_field);
  const [docNameField, setDocNameField] = useState(DEFAULTS.doc_name_field);
  const [textField, setTextField] = useState(DEFAULTS.text_field);
  const [pageField, setPageField] = useState(DEFAULTS.page_field);
  const [objectTypeField, setObjectTypeField] = useState(DEFAULTS.object_type_field);
  const [ingestionDateField, setIngestionDateField] = useState(DEFAULTS.ingestion_date_field);

  const [requiredFields, setRequiredFields] = useState<string[]>(DEFAULTS.required_fields);
  const [requiredFieldInput, setRequiredFieldInput] = useState('');

  const [similarGroupField, setSimilarGroupField] = useState(DEFAULTS.similar_group_field);
  const [differentGroupField, setDifferentGroupField] = useState(DEFAULTS.different_group_field);
  const [driftTimeField, setDriftTimeField] = useState(DEFAULTS.drift_time_field);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  /* ── Load current config when dialog opens ── */
  const loadConfig = useCallback(async () => {
    try {
      const cfg = await getFieldMapper();
      if (cfg.display_fields) setDisplayFields(cfg.display_fields);
      if (cfg.doc_id_field) setDocIdField(cfg.doc_id_field);
      if (cfg.doc_name_field) setDocNameField(cfg.doc_name_field);
      if (cfg.text_field) setTextField(cfg.text_field);
      if (cfg.page_field) setPageField(cfg.page_field);
      if (cfg.object_type_field) setObjectTypeField(cfg.object_type_field);
      if (cfg.ingestion_date_field) setIngestionDateField(cfg.ingestion_date_field);
      if (cfg.required_fields) setRequiredFields(cfg.required_fields);
      if (cfg.similar_group_field) setSimilarGroupField(cfg.similar_group_field);
      if (cfg.different_group_field) setDifferentGroupField(cfg.different_group_field);
      if (cfg.drift_time_field) setDriftTimeField(cfg.drift_time_field);
    } catch {
      /* use defaults on error */
    }
  }, []);

  useEffect(() => {
    if (open) {
      loadConfig();
      setError(null);
      setSuccess(false);
    }
  }, [open, loadConfig]);

  /* ── Handlers ── */
  const toggleDisplayField = (field: string) => {
    setDisplayFields((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field],
    );
  };

  const addCustomDisplayField = () => {
    const trimmed = customFieldInput.trim();
    if (trimmed && !displayFields.includes(trimmed)) {
      setDisplayFields((prev) => [...prev, trimmed]);
    }
    setCustomFieldInput('');
  };

  const addRequiredField = () => {
    const trimmed = requiredFieldInput.trim();
    if (trimmed && !requiredFields.includes(trimmed)) {
      setRequiredFields((prev) => [...prev, trimmed]);
    }
    setRequiredFieldInput('');
  };

  const removeRequiredField = (field: string) => {
    setRequiredFields((prev) => prev.filter((f) => f !== field));
  };

  const resetToDefaults = () => {
    setDisplayFields(DEFAULTS.display_fields);
    setDocIdField(DEFAULTS.doc_id_field);
    setDocNameField(DEFAULTS.doc_name_field);
    setTextField(DEFAULTS.text_field);
    setPageField(DEFAULTS.page_field);
    setObjectTypeField(DEFAULTS.object_type_field);
    setIngestionDateField(DEFAULTS.ingestion_date_field);
    setRequiredFields(DEFAULTS.required_fields);
    setSimilarGroupField(DEFAULTS.similar_group_field);
    setDifferentGroupField(DEFAULTS.different_group_field);
    setDriftTimeField(DEFAULTS.drift_time_field);
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(false);
    try {
      await saveFieldMapper({
        display_fields: displayFields,
        doc_id_field: docIdField,
        doc_name_field: docNameField,
        text_field: textField,
        page_field: pageField,
        object_type_field: objectTypeField,
        ingestion_date_field: ingestionDateField,
        required_fields: requiredFields,
        similar_group_field: similarGroupField,
        different_group_field: differentGroupField,
        drift_time_field: driftTimeField,
      });
      setSuccess(true);
      setTimeout(() => onClose(), 800);
    } catch (err: any) {
      setError(err.message ?? 'Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto mx-4">
        {/* Title bar */}
        <div className="sticky top-0 bg-white border-b border-slate-200 px-6 py-4 rounded-t-2xl flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
            <span>⚙️</span> Field Mapper Settings
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-xl leading-none"
          >
            ✕
          </button>
        </div>

        <div className="px-6 py-5 space-y-6">
          {/* ── Section 1: Display Fields ── */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              Fields to Show in Report Tables
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {COMMON_DISPLAY_FIELDS.map((f) => (
                <label
                  key={f}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={displayFields.includes(f)}
                    onChange={() => toggleDisplayField(f)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm font-mono text-slate-700">{f}</span>
                </label>
              ))}
            </div>
            {/* Custom fields already added (not in common list) */}
            {displayFields
              .filter((f) => !COMMON_DISPLAY_FIELDS.includes(f))
              .map((f) => (
                <label
                  key={f}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked
                    onChange={() => toggleDisplayField(f)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm font-mono text-slate-700">{f}</span>
                  <span className="text-xs text-slate-400">(custom)</span>
                </label>
              ))}
            <div className="flex gap-2 mt-3">
              <input
                type="text"
                value={customFieldInput}
                onChange={(e) => setCustomFieldInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addCustomDisplayField()}
                placeholder="Add custom field name…"
                className="flex-1 text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={addCustomDisplayField}
                className="px-3 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700"
              >
                Add
              </button>
            </div>
          </section>

          <hr className="border-slate-200" />

          {/* ── Section 2: Field Mapping ── */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              Map Your Fields to Framework Concepts
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[
                { label: 'Document ID field', value: docIdField, setter: setDocIdField },
                { label: 'Document Name field', value: docNameField, setter: setDocNameField },
                { label: 'Text / Content field', value: textField, setter: setTextField },
                { label: 'Page Number field', value: pageField, setter: setPageField },
                { label: 'Object Type field', value: objectTypeField, setter: setObjectTypeField },
                { label: 'Ingestion Date field', value: ingestionDateField, setter: setIngestionDateField },
              ].map(({ label, value, setter }) => (
                <div key={label}>
                  <label className="block text-xs font-medium text-slate-500 mb-1">
                    {label}
                  </label>
                  <input
                    type="text"
                    value={value}
                    onChange={(e) => setter(e.target.value)}
                    className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              ))}
            </div>
          </section>

          <hr className="border-slate-200" />

          {/* ── Section 3: Required Fields ── */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              Required Metadata Fields (for completeness check)
            </h3>
            <div className="flex flex-wrap gap-2 mb-3">
              {requiredFields.map((f) => (
                <span
                  key={f}
                  className="inline-flex items-center gap-1 bg-blue-50 text-blue-700 text-sm font-mono px-3 py-1 rounded-full"
                >
                  {f}
                  <button
                    onClick={() => removeRequiredField(f)}
                    className="text-blue-400 hover:text-blue-600 ml-1"
                  >
                    ×
                  </button>
                </span>
              ))}
              {requiredFields.length === 0 && (
                <span className="text-sm text-slate-400 italic">No required fields set</span>
              )}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={requiredFieldInput}
                onChange={(e) => setRequiredFieldInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addRequiredField()}
                placeholder="Add required field…"
                className="flex-1 text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={addRequiredField}
                className="px-3 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700"
              >
                Add
              </button>
            </div>
          </section>

          <hr className="border-slate-200" />

          {/* ── Section 4: Auto-Pair Grouping ── */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              Grouping for Auto-Pair Generation
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">
                  Similar group field
                </label>
                <input
                  type="text"
                  value={similarGroupField}
                  onChange={(e) => setSimilarGroupField(e.target.value)}
                  className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">
                  Different group field
                </label>
                <input
                  type="text"
                  value={differentGroupField}
                  onChange={(e) => setDifferentGroupField(e.target.value)}
                  className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">
                  Drift time field
                </label>
                <input
                  type="text"
                  value={driftTimeField}
                  onChange={(e) => setDriftTimeField(e.target.value)}
                  className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </section>
        </div>

        {/* ── Footer ── */}
        <div className="sticky bottom-0 bg-white border-t border-slate-200 px-6 py-4 rounded-b-2xl">
          {error && (
            <p className="text-sm text-red-600 mb-3">⚠️ {error}</p>
          )}
          {success && (
            <p className="text-sm text-green-600 mb-3">✅ Settings saved successfully!</p>
          )}
          <div className="flex items-center justify-between">
            <button
              onClick={resetToDefaults}
              className="text-sm text-slate-500 hover:text-slate-700 underline"
            >
              Reset to Defaults
            </button>
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
