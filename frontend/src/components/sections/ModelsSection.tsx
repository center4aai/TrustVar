import React, { useState, useEffect, useRef } from 'react';
import { Cpu, Plus, Trash2, X, TestTube, Loader, RefreshCw } from 'lucide-react';
import { useModels, useDeleteModel, useRegisterModel, useTestModel, useAvailableModels, useBulkRegisterModels } from '@/api/hooks';
import { apiClient } from '@/api/client';
import StatusBadge from '@/components/common/StatusBadge';
import { formatDate } from '@/utils/format';
import { useAppStore } from '@/stores/useAppStore';

const PROVIDERS = ['ollama', 'huggingface', 'openai', 'vllm', 'llamacpp'];

export default function ModelsSection() {
  const { data: models = [] } = useModels();
  const deleteModel = useDeleteModel();
  const registerModel = useRegisterModel();
  const testModel = useTestModel();
  const selectedItem = useAppStore(s => s.selectedItem);
  const clearSelectedItem = useAppStore(s => s.setSelectedItem);

  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showDiscoverModal, setShowDiscoverModal] = useState(false);
  const [testingModelId, setTestingModelId] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [testTaskIds, setTestTaskIds] = useState<Record<string, string>>({});
  const [pollingModelId, setPollingModelId] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (selectedItem?.type === 'model') {
      const element = document.getElementById(`model-${selectedItem.id}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        element.classList.add('ring-2', 'ring-violet-500');
        setTimeout(() => {
          element.classList.remove('ring-2', 'ring-violet-500');
        }, 3000);
      }
      clearSelectedItem(null);
    }
  }, [selectedItem, clearSelectedItem]);

  // Poll for test results
  useEffect(() => {
    if (pollingModelId && testTaskIds[pollingModelId]) {
      pollIntervalRef.current = setInterval(async () => {
        try {
          const result = await apiClient.getTestResult(pollingModelId, testTaskIds[pollingModelId]);
          if (result.status === 'completed') {
            setTestResults(prev => ({
              ...prev,
              [pollingModelId]: {
                success: result.result?.success || false,
                response: result.result?.response,
                duration: result.result?.duration,
                error: result.result?.error,
                test_prompt: result.result?.test_prompt
              }
            }));
            setPollingModelId(null);
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
            }
          } else if (result.status === 'failed') {
            setTestResults(prev => ({
              ...prev,
              [pollingModelId]: {
                success: false,
                error: result.error || 'Test failed'
              }
            }));
            setPollingModelId(null);
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
            }
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 2000);

      return () => {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
        }
      };
    }
  }, [pollingModelId, testTaskIds]);

  return (
    <div className="p-6 space-y-6 animate-slideIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white mb-1">Models</h1>
          <p className="text-sm text-gray-400">Manage your language models</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowDiscoverModal(true)}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-sm text-white font-medium flex items-center gap-2 transition-colors"
          >
            <RefreshCw size={16} />
            Sync with Ollama & API
          </button>
          <button
            onClick={() => setShowRegisterModal(true)}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-sm text-white font-medium flex items-center gap-2 transition-colors"
          >
            <Plus size={16} />
            Register Model
          </button>
        </div>
      </div>

      {/* Models List */}
      <div className="grid gap-4">
        {models.map(model => (
          <div key={model.id} id={`model-${model.id}`} className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all group">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-base font-semibold text-white">{model.name}</h3>
                  <StatusBadge status={model.status} size="sm" />
                </div>
                <p className="text-sm text-gray-400 mb-4 font-mono">{model.model_name}</p>
                <div className="flex items-center gap-6 text-xs text-gray-500">
                  <span>Provider: <span className="text-gray-400">{model.provider}</span></span>
                  <span>Temp: <span className="text-gray-400">{model.config.temperature}</span></span>
                  <span>Max tokens: <span className="text-gray-400">{model.config.max_tokens}</span></span>
                  {model.config.top_p && <span>Top-p: <span className="text-gray-400">{model.config.top_p}</span></span>}
                  {model.config.top_k && <span>Top-k: <span className="text-gray-400">{model.config.top_k}</span></span>}
                </div>
                {model.description && (
                  <p className="text-xs text-gray-500 mt-3">{model.description}</p>
                )}
                <p className="text-xs text-gray-600 mt-2">Created {formatDate(model.created_at)}</p>

                {/* Test Result */}
                {testResults[model.id] && (
                  <div className={`mt-4 p-4 rounded-lg text-sm ${
                    testResults[model.id].status === 'testing'
                      ? 'bg-violet-500/10 border border-violet-500/30'
                      : testResults[model.id].success
                        ? 'bg-emerald-500/10 border border-emerald-500/20'
                        : 'bg-red-500/10 border border-red-500/20'
                  }`}>
                    {testResults[model.id].status === 'testing' ? (
                      <div className="flex items-center gap-2 text-violet-300">
                        <Loader size={16} className="animate-spin" />
                        <span>Testing inference...</span>
                      </div>
                    ) : testResults[model.id].success ? (
                      <div>
                        <div className="text-emerald-400 font-medium mb-2 flex items-center gap-2">
                          <span>✓</span>
                          <span>Test successful ({testResults[model.id].duration?.toFixed(2)}s)</span>
                        </div>
                        {testResults[model.id].test_prompt && (
                          <div className="mb-2">
                            <div className="text-gray-500 text-xs mb-1">Prompt:</div>
                            <div className="text-gray-300 bg-gray-900/50 rounded p-2 font-mono text-xs">{testResults[model.id].test_prompt}</div>
                          </div>
                        )}
                        <div>
                          <div className="text-gray-500 text-xs mb-1">Response:</div>
                          <div className="text-gray-300 bg-gray-900/50 rounded p-3 font-mono text-xs whitespace-pre-wrap">{testResults[model.id].response}</div>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-red-400 font-medium mb-2 flex items-center gap-2">
                          <span>✗</span>
                          <span>Test failed</span>
                        </div>
                        <div className="text-red-300 bg-gray-900/50 rounded p-3 font-mono text-xs whitespace-pre-wrap">{testResults[model.id].error}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={async () => {
                    setTestingModelId(model.id);
                    setTestResults(prev => ({
                      ...prev,
                      [model.id]: {
                        status: 'testing',
                        success: true,
                        response: '',
                        duration: 0
                      }
                    }));
                    try {
                      const result = await testModel.mutateAsync({
                        id: model.id,
                        prompt: 'Write only 1 symbol.'
                      });

                      if (result.celery_task_id) {
                        setTestTaskIds(prev => ({
                          ...prev,
                          [model.id]: result.celery_task_id
                        }));
                        setPollingModelId(model.id);
                      }
                    } catch (err: any) {
                      const errorMessage = err.response?.data?.detail || err.message;
                      setTestResults(prev => ({
                        ...prev,
                        [model.id]: { success: false, error: errorMessage }
                      }));
                      setTestingModelId(null);
                    }
                  }}
                  disabled={testingModelId === model.id || pollingModelId === model.id}
                  className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-blue-400 transition-colors disabled:opacity-50"
                  title="Test model"
                >
                  {testingModelId === model.id || pollingModelId === model.id ? (
                    <Loader size={16} className="animate-spin" />
                  ) : (
                    <TestTube size={16} />
                  )}
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Delete model "${model.name}"?`)) {
                      deleteModel.mutate(model.id);
                    }
                  }}
                  className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-red-400 transition-colors"
                  title="Delete model"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          </div>
        ))}

        {models.length === 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
            <Cpu size={48} className="mx-auto text-gray-700 mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No models yet</h3>
            <p className="text-sm text-gray-500 mb-6">Register your first model to get started</p>
            <button
              onClick={() => setShowRegisterModal(true)}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium transition-colors"
            >
              Register Model
            </button>
          </div>
        )}
      </div>

      {/* Register Modal */}
      {showRegisterModal && (
        <RegisterModelModal
          onClose={() => setShowRegisterModal(false)}
          onSuccess={() => setShowRegisterModal(false)}
          registerModel={registerModel}
        />
      )}

      {/* Discover Ollama Modal */}
      {showDiscoverModal && (
        <DiscoverOllamaModal
          onClose={() => setShowDiscoverModal(false)}
          existingModels={models}
        />
      )}
    </div>
  );
}

function RegisterModelModal({
  onClose,
  onSuccess,
  registerModel
}: {
  onClose: () => void;
  onSuccess: () => void;
  registerModel: any;
}) {
  const [name, setName] = useState('');
  const [provider, setProvider] = useState<'ollama' | 'huggingface' | 'openai'>('ollama');
  const [modelName, setModelName] = useState('');
  const [description, setDescription] = useState('');
  const [temperature, setTemperature] = useState(0.0);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [topP, setTopP] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!name || !modelName) {
      setError('Please fill in all required fields');
      return;
    }

    setError('');
    setIsSubmitting(true);

    try {
      await registerModel.mutateAsync({
        name,
        provider,
        model_name: modelName,
        description,
        config: {
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          top_k: topK,
        },
      });
      onSuccess();
    } catch (err: any) {
      setError(err.message || 'Failed to register model');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-800 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-800">
          <div>
            <h2 className="text-xl font-semibold text-white">Register New Model</h2>
            <p className="text-sm text-gray-400 mt-1">Add a model to your evaluation pipeline</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Basic Info */}
          <div>
            <h3 className="text-sm font-medium text-white mb-4">Basic Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Model Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Llama 2 7B Chat"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Provider</label>
                    <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value as any)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                >
                  {PROVIDERS.map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div className="col-span-2">
                <label className="block text-sm text-gray-400 mb-2">
                  Model Identifier <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="e.g., llama2:7b or gpt-4"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
                />
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
            </div>
          </div>

          {/* Configuration */}
          <div>
            <h3 className="text-sm font-medium text-white mb-4">Configuration</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Temperature ({temperature})
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Precise</span>
                  <span>Creative</span>
                </div>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  min="1"
                  max="128536"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Top P ({topP})
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Top K</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="100"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-white text-sm font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || !name || !modelName}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-2"
          >
            {isSubmitting ? (
              <>
                <Loader size={16} className="animate-spin" />
                Registering...
              </>
            ) : (
              <>
                <Plus size={16} />
                Register Model
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function DiscoverOllamaModal({
  onClose,
  existingModels,
}: {
  onClose: () => void;
  existingModels: any[];
}) {
  const [manualInput, setManualInput] = useState('');
  const [pullIfMissing, setPullIfMissing] = useState(true);
  const [provider, setProvider] = useState<'ollama' | 'openai' | 'huggingface' | 'vllm' | 'llamacpp'>('ollama');
  const { data: availableData, isLoading, error, refetch } = useAvailableModels(provider);
  const bulkRegister = useBulkRegisterModels();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [description, setDescription] = useState('Auto-discovered from Ollama');
  const [result, setResult] = useState<{ created: number; skipped: number; downloading: number } | null>(null);

  const existingNames = new Set(existingModels.map(m => m.model_name));
  const models = availableData?.models || [];

  const formatSize = (bytes: number) => {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
    return `${(bytes / 1e3).toFixed(0)} KB`;
  };

  const toggleModel = (name: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const toggleAll = () => {
    const registerable = models.filter(m => !existingNames.has(m.name));
    if (selected.size === registerable.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(registerable.map(m => m.name)));
    }
  };

  const parseManualNames = () => {
    return manualInput
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);
  };

  const handleRegister = async () => {
    const manualNames = parseManualNames();
    const allNames = [...new Set([...Array.from(selected), ...manualNames])];
    if (allNames.length === 0) return;
    try {
      const res = await bulkRegister.mutateAsync({
        model_names: allNames,
        config: { temperature, max_tokens: maxTokens, top_p: topP, top_k: topK },
        description,
        provider,
        pull_if_missing: pullIfMissing,
      });
      setResult({
        created: res.created.length,
        skipped: res.skipped.length,
        downloading: res.downloading.length,
      });
    } catch (err: any) {
      console.error('Bulk register error:', err);
    }
  };

  if (result) {
    return (
      <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 max-w-md w-full p-6 text-center">
          <div className="text-emerald-400 text-4xl mb-4">✓</div>
          <h3 className="text-lg font-semibold text-white mb-2">Sync Complete</h3>
          <p className="text-sm text-gray-400 mb-1">Registered: {result.created} model(s)</p>
          {result.downloading > 0 && (
            <p className="text-sm text-violet-400 mb-1">Downloading: {result.downloading} model(s)</p>
          )}
          {result.skipped > 0 && (
            <p className="text-sm text-gray-500 mb-4">Skipped: {result.skipped} (already registered)</p>
          )}
          {result.downloading === 0 && result.skipped === 0 && <div className="mb-4" />}
          <button
            onClick={onClose}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-800 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-800">
          <div>
            <h2 className="text-xl font-semibold text-white">Add Models</h2>
            <p className="text-sm text-gray-400 mt-1">
              {provider === 'ollama'
                ? 'Discover and register models from your Ollama server'
                : `Register ${provider} models`}
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Provider</label>
            <select
              value={provider}
              onChange={(e) => {
                setProvider(e.target.value as typeof provider);
                setSelected(new Set());
              }}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
            >
              <option value="ollama">Ollama</option>
              <option value="openai">OpenAI</option>
              <option value="huggingface">HuggingFace</option>
              <option value="vllm">vLLM</option>
              <option value="llamacpp">LLaMA.cpp</option>
            </select>
          </div>

          {provider === 'ollama' && isLoading && (
            <div className="flex items-center justify-center py-12">
              <Loader size={24} className="animate-spin text-violet-400" />
              <span className="ml-3 text-gray-400">Fetching models from Ollama...</span>
            </div>
          )}

          {provider === 'ollama' && error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <p className="text-sm text-red-400 mb-3">Cannot connect to Ollama. Check if Ollama is running.</p>
              <button
                onClick={() => refetch()}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm text-white transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {provider === 'ollama' && !isLoading && !error && models.length === 0 && (
            <div className="text-center py-12">
              <Cpu size={48} className="mx-auto text-gray-700 mb-4" />
              <p className="text-gray-400">No models found in Ollama.</p>
              <p className="text-sm text-gray-600 mt-1">Pull models first using <code>ollama pull</code></p>
            </div>
          )}

          {provider === 'ollama' && !isLoading && !error && models.length > 0 && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">
                  {models.length} model(s) found, {selected.size} selected
                </span>
                <button
                  onClick={toggleAll}
                  className="text-sm text-violet-400 hover:text-violet-300 transition-colors"
                >
                  {selected.size === models.filter(m => !existingNames.has(m.name)).length
                    ? 'Deselect All'
                    : 'Select All'}
                </button>
              </div>

              <div className="border border-gray-800 rounded-lg divide-y divide-gray-800">
                {models.map(model => {
                  const alreadyRegistered = existingNames.has(model.name);
                  return (
                    <label
                      key={model.name}
                      className={`flex items-center gap-4 p-4 cursor-pointer transition-colors ${
                        alreadyRegistered ? 'opacity-50' : 'hover:bg-gray-800/50'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(model.name)}
                        onChange={() => toggleModel(model.name)}
                        disabled={alreadyRegistered}
                        className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-violet-500 focus:ring-violet-500 focus:ring-offset-0 disabled:opacity-50"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white font-mono">{model.name}</div>
                        <div className="text-xs text-gray-500">
                          {formatSize(model.size)}
                          {alreadyRegistered && (
                            <span className="ml-2 text-gray-600">(already registered)</span>
                          )}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </>
          )}

          <div className="border-t border-gray-800 pt-4">
            <h3 className="text-sm font-medium text-white mb-2">
              {provider === 'ollama' ? 'Or enter models manually' : 'Model names'}
            </h3>
            <p className="text-xs text-gray-500 mb-3">One model name per line (e.g., <code>{provider === 'ollama' ? 'deepseek-r1:8b' : provider === 'openai' ? 'gpt-4o' : 'model-name'}</code>)</p>
            <textarea
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder={provider === 'ollama'
                ? "deepseek-r1:8b\nqwen2.5:7b\ncodellama:13b"
                : provider === 'openai'
                ? "gpt-4o\ngpt-4o-mini\ngpt-3.5-turbo"
                : "model-name-1\nmodel-name-2"}
              rows={4}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white font-mono placeholder-gray-600 focus:outline-none focus:border-violet-500 resize-none"
            />
          </div>

          {provider === 'ollama' && (
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={pullIfMissing}
                onChange={(e) => setPullIfMissing(e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-violet-500 focus:ring-violet-500 focus:ring-offset-0"
              />
              <span className="text-sm text-gray-300">Pull missing models from Ollama</span>
            </label>
          )}

          <div>
            <h3 className="text-sm font-medium text-white mb-4">Default Configuration</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Temperature ({temperature})</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  min="1"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Top P ({topP})</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Top K</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="100"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
                />
              </div>
            </div>
            <div className="mt-4">
              <label className="block text-sm text-gray-400 mb-2">Description</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>
        </div>

        {(provider !== 'ollama' || models.length > 0 || manualInput.trim().length > 0) && (
          <div className="flex items-center justify-between p-6 border-t border-gray-800">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-white text-sm font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleRegister}
              disabled={(selected.size === 0 && parseManualNames().length === 0) || bulkRegister.isPending}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-2"
            >
              {bulkRegister.isPending ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  Registering...
                </>
              ) : (
                <>
                  <RefreshCw size={16} />
                  Register Selected ({selected.size + parseManualNames().length})
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}