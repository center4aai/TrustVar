import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, CheckCircle, FileText, ChevronDown, Table } from 'lucide-react';
import { useCreateDataset } from '@/api/hooks';

interface FilePreview {
  columns: string[];
  rows: any[];
  format: string;
}

const TASK_TYPES = [
  { value: 'mcq', label: 'Multiple Choice (MCQ)' },
  { value: 'classification', label: 'Classification' },
  { value: 'open_qa', label: 'Open Question Answering' },
  { value: 'generation', label: 'Text Generation' },
];

const TASK_SEMANTICS = [
  { value: '', label: '(auto — same as task type)' },
  { value: 'multi_label_classification', label: 'Multi-label Classification' },
  { value: 'sentiment_classification', label: 'Sentiment Classification' },
  { value: 'set_membership', label: 'Set Membership' },
  { value: 'summarization', label: 'Summarization' },
  { value: 'negation_detection', label: 'Negation Detection' },
  { value: 'translation', label: 'Translation' },
  { value: 'tense_discrimination', label: 'Tense Discrimination' },
];


const MCQ_OPTION_HINTS = ['options', 'option_labels', 'choices', 'labels', 'classes'];

interface UploadSchemaInputs {
  taskType: string;
  columns: string[];
  promptColumn: string;
  targetColumn: string;
}

export function validateUploadSchema(inputs: UploadSchemaInputs): string[] {
  const errors: string[] = [];
  const { taskType, columns, promptColumn, targetColumn } = inputs;
  const colSet = new Set(columns);
  const lowerCols = columns.map((c) => c.toLowerCase());

  if (promptColumn && !colSet.has(promptColumn)) {
    errors.push(`Prompt column '${promptColumn}' is not present in the uploaded file.`);
  }
  if (targetColumn && !colSet.has(targetColumn)) {
    errors.push(`Target column '${targetColumn}' is not present in the uploaded file.`);
  }

  if (taskType === 'classification' || taskType === 'open_qa') {
    if (!targetColumn) {
      errors.push(
        `Task type '${taskType}' requires a Target Column for per-item scoring. ` +
        'Select a column containing the expected answer (or gold label).',
      );
    }
  }

  if (taskType === 'mcq') {
    const hasMcqCol = MCQ_OPTION_HINTS.some((h) => lowerCols.includes(h));
    if (!hasMcqCol) {
      errors.push(
        `MCQ datasets usually need a column with answer options ` +
        `(one of: ${MCQ_OPTION_HINTS.join(', ')}). None found in the uploaded file.`,
      );
    }
  }

  return errors;
}

export default function DatasetUploadModal({ onClose, onSuccess }: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [step, setStep] = useState<'upload' | 'configure' | 'uploading'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [taskType, setTaskType] = useState('mcq');
  const [taskSemantics, setTaskSemantics] = useState('');
  const [tags, setTags] = useState('');

  const [promptColumn, setPromptColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [includeColumn, setIncludeColumn] = useState('');
  const [excludeColumn, setExcludeColumn] = useState('');
  const [templateColumn, setTemplateColumn] = useState('');
  const [variablesColumns, setVariablesColumns] = useState<string[]>([]);
  const [variablesDropdownOpen, setVariablesDropdownOpen] = useState(false);

  const [error, setError] = useState('');
  const createDataset = useCreateDataset();
  const variablesDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (variablesDropdownRef.current && !variablesDropdownRef.current.contains(e.target as Node)) {
        setVariablesDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const detectFormat = (filename: string): string => {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 'csv') return 'csv';
    if (ext === 'parquet') return 'parquet';
    if (ext === 'jsonl') return 'jsonl';
    if (ext === 'json') return 'json';
    return 'json';
  };

  const parseFile = async (file: File): Promise<FilePreview> => {
    const format = detectFormat(file.name);
    const text = await file.text();

    try {
      if (format === 'jsonl') {
        const lines = text.trim().split('\n').filter(l => l.trim());
        const firstLine = JSON.parse(lines[0]);
        const columns = Object.keys(firstLine);
        const rows = lines.slice(0, 5).map(line => JSON.parse(line));
        return { columns, rows, format };
      } else if (format === 'json') {
        const data = JSON.parse(text);
        let items = Array.isArray(data) ? data : data.data || data.items || [];
        if (items.length === 0) throw new Error('No items found in JSON');
        const columns = Object.keys(items[0]);
        const rows = items.slice(0, 5);
        return { columns, rows, format };
      } else if (format === 'csv') {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const rows = lines.slice(1, 6).map(line => {
          const values = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
          return headers.reduce((obj, header, i) => {
            obj[header] = values[i] || '';
            return obj;
          }, {} as any);
        });
        return { columns: headers, rows, format };
      }
    } catch (err: any) {
      throw new Error(`Failed to parse ${format}: ${err.message}`);
    }

    throw new Error('Unsupported format');
  };

  const handleFile = async (file: File) => {
    setError('');
    setFile(file);

    try {
      const preview = await parseFile(file);
      setPreview(preview);

      const { columns } = preview;
      const promptCandidates = ['prompt', 'question', 'input', 'query', 'text'];
      const promptCol = columns.find(c => promptCandidates.includes(c.toLowerCase())) || columns[0];
      setPromptColumn(promptCol);

      const targetCandidates = ['target', 'answer', 'output', 'response', 'completion'];
      const targetCol = columns.find(c => targetCandidates.includes(c.toLowerCase())) || '';
      setTargetColumn(targetCol);

      if (!name) {
        const basename = file.name.replace(/\.[^/.]+$/, '');
        setName(basename);
      }

      setStep('configure');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file || !name || !promptColumn) {
      setError('Please fill in all required fields');
      return;
    }

    const schemaErrors = validateUploadSchema({
      taskType,
      columns: preview?.columns ?? [],
      promptColumn,
      targetColumn,
    });
    if (schemaErrors.length > 0) {
      setError('Schema validation failed:\n• ' + schemaErrors.join('\n• '));
      return;
    }

    setStep('uploading');
    setError('');

    try {
      await createDataset.mutateAsync({
        datasetInfo: {
          name,
          description,
          task_type: taskType,
          task_semantics: taskSemantics || undefined,
          tags,
        },
        uploadInfo: {
          file,
          file_format: preview!.format,
          prompt_column: promptColumn,
          target_column: targetColumn || undefined,
          include_column: includeColumn || undefined,
          exclude_column: excludeColumn || undefined,
          template_column: templateColumn || undefined,
          variables_columns: variablesColumns.length > 0 ? variablesColumns.join(",") : undefined,
        },
      });

      onSuccess();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
      setStep('configure');
    }
  };

  const toggleVariableColumn = (col: string) => {
    if (variablesColumns.includes(col)) {
      setVariablesColumns(variablesColumns.filter(c => c !== col));
    } else {
      setVariablesColumns([...variablesColumns, col]);
    }
  };

  const availableVariablesColumns = preview?.columns.filter(col => col !== templateColumn) || [];

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-800 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-800">
          <div>
            <h2 className="text-xl font-semibold text-white">Upload Dataset</h2>
            <p className="text-sm text-gray-400 mt-1">
              {step === 'upload' && 'Select a file to upload'}
              {step === 'configure' && 'Configure dataset and column mapping'}
              {step === 'uploading' && 'Uploading...'}
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6">

          {step === 'upload' && (
            <div>
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${dragActive ? 'border-violet-500 bg-violet-500/5' : 'border-gray-700 hover:border-gray-600'
                  }`}
              >
                <Upload size={48} className="mx-auto text-gray-600 mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">
                  Drop your file here, or click to browse
                </h3>
                <p className="text-sm text-gray-400 mb-6">Supports: JSON, JSONL, CSV, Parquet</p>
                <input
                  type="file"
                  accept=".json,.jsonl,.csv,.parquet"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium cursor-pointer transition-colors"
                >
                  <FileText size={16} />
                  Select File
                </label>
              </div>
            </div>
          )}

          {step === 'configure' && preview && (
            <div className="space-y-6">
              {/* File parsed success banner */}
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4 flex items-start gap-3">
                <CheckCircle size={20} className="text-emerald-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-emerald-400">File parsed successfully</div>
                  <div className="text-sm text-gray-400 mt-1">
                    {file?.name} • {preview.format.toUpperCase()} • {preview.columns.length} columns
                  </div>
                </div>
              </div>

              {/* Dataset Information */}
              <div>
                <h3 className="text-sm font-medium text-white mb-4">Dataset Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Name <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g., QA Test Set v1"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Task Type</label>
                    <select
                      value={taskType}
                      onChange={(e) => setTaskType(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                    >
                      {TASK_TYPES.map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Task Semantics (optional)</label>
                    <select
                      value={taskSemantics}
                      onChange={(e) => setTaskSemantics(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                    >
                      {TASK_SEMANTICS.map(s => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-2">Description</label>
                    <textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Brief description..."
                      rows={2}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500 resize-none"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-2">Tags (comma-separated)</label>
                    <input
                      type="text"
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                      placeholder="e.g., qa, english, test"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
                    />
                  </div>
                </div>
              </div>

              {/* Column Mapping */}
              <div>
                <h3 className="text-sm font-medium text-white mb-4">Column Mapping</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Prompt Column <span className="text-red-400">*</span>
                    </label>
                    <select
                      value={promptColumn}
                      onChange={(e) => setPromptColumn(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      {preview.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Target Column (optional)
                    </label>
                    <select
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      <option value="">None</option>
                      {preview.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Include List Column (optional)
                    </label>
                    <select
                      value={includeColumn}
                      onChange={(e) => setIncludeColumn(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      <option value="">None</option>
                      {preview.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Exclude List Column (optional)
                    </label>
                    <select
                      value={excludeColumn}
                      onChange={(e) => setExcludeColumn(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      <option value="">None</option>
                      {preview.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Prompt Templating */}
                <div className="border-t border-gray-800 pt-6 mt-6">
                  <h3 className="text-sm font-medium text-white mb-4">
                    Prompt Templating (optional)
                  </h3>
                  <p className="text-xs text-gray-500 mb-4">
                    If your dataset has separate columns for template and variables, configure them
                    here for Jinja2 templating at runtime.
                  </p>
                  <div className="grid grid-cols-2 gap-4">
                    {/* Template Column — single select */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Template Column</label>
                      <select
                        value={templateColumn}
                        onChange={(e) => setTemplateColumn(e.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                      >
                        <option value="">None</option>
                        {preview.columns.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    {/* Variables Columns — multi-select dropdown */}
                    <div ref={variablesDropdownRef} className="relative">
                      <label className="block text-sm text-gray-400 mb-2">Variables Columns</label>
                      <button
                        type="button"
                        onClick={() => setVariablesDropdownOpen(!variablesDropdownOpen)}
                        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-left flex items-center justify-between focus:outline-none focus:border-violet-500 transition-colors"
                      >
                        <span className={variablesColumns.length > 0 ? 'text-white' : 'text-gray-500'}>
                          {variablesColumns.length > 0
                            ? `${variablesColumns.length} column${variablesColumns.length > 1 ? 's' : ''} selected`
                            : 'None'}
                        </span>
                        <ChevronDown
                          size={16}
                          className={`text-gray-400 transition-transform ${variablesDropdownOpen ? 'rotate-180' : ''}`}
                        />
                      </button>

                      {/* Dropdown panel */}
                      {variablesDropdownOpen && (
                        <div className="absolute z-10 mt-1 w-full bg-gray-800 border border-gray-700 rounded-lg shadow-xl max-h-48 overflow-y-auto">
                          {availableVariablesColumns.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-gray-500">No columns available</div>
                          ) : (
                            <>
                              {/* Select all / Deselect all */}
                              <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
                                <span className="text-xs text-gray-500">
                                  {variablesColumns.length}/{availableVariablesColumns.length} selected
                                </span>
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (variablesColumns.length === availableVariablesColumns.length) {
                                      setVariablesColumns([]);
                                    } else {
                                      setVariablesColumns([...availableVariablesColumns]);
                                    }
                                  }}
                                  className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
                                >
                                  {variablesColumns.length === availableVariablesColumns.length
                                    ? 'Deselect all'
                                    : 'Select all'}
                                </button>
                              </div>

                              {availableVariablesColumns.map(col => (
                                <button
                                  key={col}
                                  type="button"
                                  onClick={() => toggleVariableColumn(col)}
                                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-white hover:bg-gray-700/50 transition-colors text-left"
                                >
                                  <div
                                    className={`w-4 h-4 rounded border flex-shrink-0 flex items-center justify-center transition-colors ${variablesColumns.includes(col)
                                      ? 'bg-violet-500 border-violet-500'
                                      : 'border-gray-600 bg-gray-700'
                                      }`}
                                  >
                                    {variablesColumns.includes(col) && (
                                      <CheckCircle size={12} className="text-white" />
                                    )}
                                  </div>
                                  {col}
                                </button>
                              ))}
                            </>
                          )}
                        </div>
                      )}

                      {/* Selected tags */}
                      {variablesColumns.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {variablesColumns.map(col => (
                            <span
                              key={col}
                              className="inline-flex items-center gap-1 px-2 py-0.5 bg-violet-500/20 text-violet-300 text-xs rounded"
                            >
                              {col}
                              <button
                                type="button"
                                onClick={() => setVariablesColumns(variablesColumns.filter(c => c !== col))}
                                className="hover:text-violet-200"
                              >
                                <X size={12} />
                              </button>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>{/* end grid */}
                </div>{/* end Prompt Templating */}
              </div>{/* end Column Mapping */}

              {/* Data Preview Table */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Table size={16} className="text-gray-400" />
                  <h3 className="text-sm font-medium text-white">Data Preview</h3>
                  <span className="text-xs text-gray-500">(first {preview.rows.length} rows)</span>
                </div>
                <div className="border border-gray-800 rounded-lg overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-gray-800/50">
                          {preview.columns.map(col => (
                            <th
                              key={col}
                              className="px-4 py-2.5 text-left text-xs font-medium text-gray-400 whitespace-nowrap border-b border-gray-800"
                            >
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((row, i) => (
                          <tr
                            key={i}
                            className={`${i % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800/20'} hover:bg-gray-800/40 transition-colors`}
                          >
                            {preview.columns.map(col => (
                              <td
                                key={col}
                                className="px-4 py-2 text-gray-300 whitespace-nowrap max-w-[200px] truncate border-b border-gray-800/50"
                                title={String(row[col] ?? '')}
                              >
                                {String(row[col] ?? '')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

            </div>
          )}

          {step === 'uploading' && (
            <div className="flex flex-col items-center justify-center py-12 gap-4">
              <div className="w-10 h-10 border-4 border-violet-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-gray-400">Uploading dataset...</p>
            </div>
          )}

          {/* Footer buttons */}
          {step !== 'uploading' && (
            <div className="mt-6 pt-6 border-t border-gray-800 space-y-3">
              {error && step === 'configure' && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-sm text-red-300 whitespace-pre-line">
                  {error}
                </div>
              )}
              <div className="flex justify-end gap-3">
                <button
                  onClick={onClose}
                  className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                {step === 'configure' && (
                  <button
                    onClick={handleUpload}
                    disabled={!name || !promptColumn}
                    className="px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium transition-colors"
                  >
                    Upload Dataset
                  </button>
                )}
              </div>
            </div>
          )}
        </div>{/* end body */}
      </div>
    </div>
  );
}