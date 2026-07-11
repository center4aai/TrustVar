import React, { useState } from 'react';
import { Zap, Plus, Pause, Play, X, Eye, ChevronDown, ChevronRight, Trash2 } from 'lucide-react';
import { useTasks, usePauseTask, useResumeTask, useCancelTask, useDeleteTask, useDatasets, useModels, useCreateTask } from '@/api/hooks';
import StatusBadge from '@/components/common/StatusBadge';
import { useAppStore } from '@/stores/useAppStore';
import PromptSelector from '@/components/common/PromptSelector';
import type { Prompt } from '@/api/types';

const TASK_TYPES = [
  { value: 'standard', label: 'Standard QA', description: 'Simple question answering' },
  { value: 'variation', label: 'Prompt Variations', description: 'Test with prompt variations' },
  { value: 'judged', label: 'LLM as a Judge', description: 'Evaluate with LLM judge' },
  { value: 'refuse_to_answer', label: 'Refuse-to-Answer Detection', description: 'Detect refusal to answer' },
];

interface VariationStrategyInfo {
  value: string;
  label: string;
  tier: 'A' | 'B' | 'C';
  description: string;
}

const TIER_COLORS: Record<string, string> = {
  A: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
  B: 'text-violet-400 border-violet-500/30 bg-violet-500/10',
  C: 'text-orange-400 border-orange-500/30 bg-orange-500/10',
};

const TIER_TAB_COLORS: Record<string, { active: string; inactive: string }> = {
  A: { active: 'border-blue-500 text-blue-400 bg-blue-500/10', inactive: 'border-transparent text-gray-500 hover:text-blue-300' },
  B: { active: 'border-violet-500 text-violet-400 bg-violet-500/10', inactive: 'border-transparent text-gray-500 hover:text-violet-300' },
  C: { active: 'border-orange-500 text-orange-400 bg-orange-500/10', inactive: 'border-transparent text-gray-500 hover:text-orange-300' },
  all: { active: 'border-gray-500 text-gray-200 bg-gray-500/10', inactive: 'border-transparent text-gray-500 hover:text-gray-300' },
};

const VARIATION_STRATEGIES: VariationStrategyInfo[] = [
  // Tier A — symbolic, no LLM needed
  { value: 'format_normalization', label: 'Format Normalization', tier: 'A', description: 'Unicode NFC, whitespace, case normalization' },
  { value: 'orthographic_normalization_ru', label: 'Orthographic Norm (RU)', tier: 'A', description: 'Russian yo/e letter normalization' },
  { value: 'mcq_option_permutation', label: 'MCQ Option Permutation', tier: 'A', description: 'Answer option order permutation' },
  { value: 'list_reordering', label: 'List Reordering', tier: 'A', description: 'Commutative list item reordering' },
  { value: 'typed_parametric_substitution', label: 'Parametric Substitution', tier: 'A', description: 'Typed template slot substitution' },
  // Tier B — LLM-guided, surface restructuring
  { value: 'active_passive_voice', label: 'Active/Passive Voice', tier: 'B', description: 'Voice flip with UD verification' },
  { value: 'monosemic_synonym_substitution', label: 'Monosemic Synonyms', tier: 'B', description: 'Single-sense synonym substitution' },
  { value: 'nominalisation', label: 'Nominalisation', tier: 'B', description: 'Verb↔noun phrase restructuring' },
  { value: 'controlled_syntactic_transformations', label: 'Syntactic Transformations', tier: 'B', description: 'Clefting, topicalization, RC reshaping' },
  { value: 'sentence_split_merge', label: 'Sentence Split/Merge', tier: 'B', description: 'Split or merge adjacent sentences' },
  { value: 'controlled_descriptive_modifier_insertion', label: 'Descriptive Modifiers', tier: 'B', description: 'Controlled adjective/adverb insertion' },
  // Tier C — LLM-based, semantically constrained
  { value: 'paraphrase_lexico_syntactic_constrained', label: 'Constrained Paraphrase', tier: 'C', description: 'Lexico-syntactic constrained paraphrase' },
  { value: 'paraphrase_free', label: 'Free Paraphrase', tier: 'C', description: 'Free semantic paraphrase' },
  { value: 'length_variation', label: 'Length Variation', tier: 'C', description: 'Shorten or lengthen text' },
  { value: 'register_formal_informal', label: 'Register Shift', tier: 'C', description: 'Formal↔informal register change' },
  { value: 'tone_shift', label: 'Tone Shift', tier: 'C', description: 'Emotional tone transformation' },
  { value: 'negation_scope_preserving_rephrasing', label: 'Negation Rephrasing', tier: 'C', description: 'Negation-preserving rephrase' },
  { value: 'wsd_synonym_substitution', label: 'WSD Synonyms', tier: 'C', description: 'Word-sense disambiguated substitution' },
  { value: 'back_translation_single_pivot', label: 'Back Translation', tier: 'C', description: 'Translate→pivot→back translate' },
];

export default function TasksSection() {
  const { data: tasks = [] } = useTasks();
  const { data: datasets = [] } = useDatasets();
  const { data: models = [] } = useModels();
  const pauseTask = usePauseTask();
  const resumeTask = useResumeTask();
  const cancelTask = useCancelTask();
  const deleteTask = useDeleteTask();
  const createTask = useCreateTask();
  const setActiveSection = useAppStore(s => s.setActiveSection);
  const setSelectedTaskId = useAppStore(s => s.setSelectedTaskId);
  const preselectedDatasetId = useAppStore(s => s.preselectedDatasetId);
  const clearPreselectedDatasetId = useAppStore(s => s.setPreselectedDatasetId);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [localDatasetId, setLocalDatasetId] = useState<string | null>(null);

  React.useEffect(() => {
    if (preselectedDatasetId) {
      setLocalDatasetId(preselectedDatasetId);
      setShowCreateModal(true);
      clearPreselectedDatasetId(null);
    }
  }, [preselectedDatasetId, clearPreselectedDatasetId]);

  const handleTaskCreated = () => {
    setShowCreateModal(false);
    setLocalDatasetId(null);
    setActiveSection('dashboard');
  };

  return (
    <div className="p-6 space-y-6 animate-slideIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white mb-1">Tasks</h1>
          <p className="text-sm text-gray-400">Create and monitor evaluation tasks</p>
        </div>
        <button
          onClick={() => {
            setLocalDatasetId(null);
            setShowCreateModal(true);
          }}
          className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-sm text-white font-medium flex items-center gap-2 transition-colors"
        >
          <Plus size={16} />
          Create Task
        </button>
      </div>

      <div className="grid gap-4">
        {tasks.map(task => (
          <div key={task.id} className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-base font-semibold text-white">{task.name}</h3>
                  <StatusBadge status={task.status} />
                </div>
                <p className="text-sm text-gray-500">
                  {task.processed_samples ?? 0} / {task.total_samples ?? 0} samples ({((task.progress ?? 0)).toFixed(1)}%)
                </p>
              </div>
              <div className="flex gap-2">
                {task.status === 'running' && (
                  <>
                    <button
                      onClick={() => pauseTask.mutate(task.id)}
                      className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-orange-400"
                    >
                      <Pause size={16} />
                    </button>
                    <button
                      onClick={() => {
                        if (confirm('Cancel task?')) {
                          cancelTask.mutate(task.id);
                        }
                      }}
                      className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-red-400"
                    >
                      <X size={16} />
                    </button>
                  </>
                )}
                {task.status === 'paused' && (
                  <button
                    onClick={() => resumeTask.mutate(task.id)}
                    className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-green-400"
                  >
                    <Play size={16} />
                  </button>
                )}
                {task.status === 'completed' && (
                  <button
                    onClick={() => {
                      setSelectedTaskId(task.id);
                      setActiveSection('results');
                    }}
                    className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-blue-400"
                  >
                    <Eye size={16} />
                  </button>
                )}
                <button
                  onClick={() => {
                    if (confirm(`Delete task "${task.name}"?`)) {
                      deleteTask.mutate(task.id);
                    }
                  }}
                  className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-red-400 transition-colors"
                  title="Delete task"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-violet-600 to-purple-600"
                style={{ width: `${task.progress}%` }}
              />
            </div>
          </div>
        ))}

        {tasks.length === 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
            <Zap size={48} className="mx-auto text-gray-700 mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No tasks yet</h3>
            <p className="text-sm text-gray-500 mb-6">Create your first evaluation task</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium"
            >
              Create Task
            </button>
          </div>
        )}
      </div>

      {showCreateModal && (
        <CreateTaskModal
          onClose={() => {
            setShowCreateModal(false);
            setLocalDatasetId(null);
          }}
          onSuccess={handleTaskCreated}
          datasets={datasets}
          models={models}
          createTask={createTask}
          initialDatasetId={localDatasetId}
        />
      )}
    </div>
  );
}

function CreateTaskModal({ onClose, onSuccess, datasets, models, createTask, initialDatasetId }: any) {
  const [step, setStep] = useState(1);
  const [name, setName] = useState('');
  const [datasetId, setDatasetId] = useState(initialDatasetId || '');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [taskType, setTaskType] = useState('standard');

  React.useEffect(() => {
    if (initialDatasetId) {
      setDatasetId(initialDatasetId);
    }
  }, [initialDatasetId]);

  // Variations config
  const [variationModelId, setVariationModelId] = useState('');
  const [numVariations, setNumVariations] = useState(3);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [variationPromptId, setVariationPromptId] = useState<string | null>(null);
  const [variationPromptContent, setVariationPromptContent] = useState<string>('');
  const [variationTierFilter, setVariationTierFilter] = useState<'A' | 'B' | 'C' | 'all'>('all');

  // Judge config
  const [judgeModelId, setJudgeModelId] = useState('');
  const [judgePromptId, setJudgePromptId] = useState<string | null>(null);
  const [judgePromptContent, setJudgePromptContent] = useState<string>('');

  // RTA config
  const [rtaModelId, setRtaModelId] = useState('');
  const [rtaPromptId, setRtaPromptId] = useState<string | null>(null);
  const [rtaPromptContent, setRtaPromptContent] = useState<string>('');

  const [batchSize, setBatchSize] = useState(1);
  const [error, setError] = useState('');
  const [variationPromptError, setVariationPromptError] = useState('');
  const [judgePromptError, setJudgePromptError] = useState('');
  const [rtaPromptError, setRtaPromptError] = useState('');
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleSubmit = async () => {
    if (!name || !datasetId || selectedModels.length === 0) {
      setError('Please fill in all required fields');
      return;
    }

    // Clear previous errors
    setError('');
    setVariationPromptError('');
    setJudgePromptError('');
    setRtaPromptError('');

    // Validate task-specific requirements
    if (taskType === 'variation' && selectedStrategies.length === 0) {
      setError('Please select at least one variation strategy');
      return;
    }
    if (taskType === 'variation') {
      const hasLLM = selectedStrategies.some(s => {
        const info = VARIATION_STRATEGIES.find(vs => vs.value === s);
        return info && info.tier !== 'A';
      });
      if (hasLLM && !variationModelId) {
        setError('Variation model is required for Tier B/C strategies');
        return;
      }
      if (hasLLM && !variationPromptContent) {
        setVariationPromptError('Custom prompt is required for Tier B/C strategies');
        return;
      }
    }
    const selectedDataset = datasets.find((ds: any) => ds.id === datasetId);
    const generativeTaskTypes = ['open_qa', 'generation'];
    const needsJudge = taskType === 'judged'
      || (taskType === 'variation' && selectedDataset && generativeTaskTypes.includes(selectedDataset.task_type));
    if (needsJudge && !judgeModelId) {
      setError('Judge model is required for open_qa/generation tasks. Enable Judge Configuration below.');
      return;
    }
    if (taskType === 'judged' && !judgePromptContent) {
      setJudgePromptError('Please select or create a judge prompt');
      return;
    }
    if (taskType === 'refuse_to_answer' && !rtaModelId) {
      setError('Please select an RTA judge model');
      return;
    }
    if (taskType === 'refuse_to_answer' && !rtaPromptContent) {
      setRtaPromptError('Please select or create an RTA prompt');
      return;
    }

    try {
      const config: any = {
        batch_size: batchSize,
        evaluate: true,
        evaluation_metrics: ['accuracy'],
        variations: {
          enabled: taskType === 'variation',
          model_id: variationModelId || null,
          strategies: selectedStrategies,
          count_per_strategy: numVariations,
          custom_prompt: variationPromptContent || null,
        },
        judge: {
          enabled: taskType === 'judged' || (taskType === 'variation' && !!judgeModelId),
          model_id: judgeModelId || null,
          criteria: ['accuracy', 'relevance', 'completeness'],
          custom_prompt_template: judgePromptContent || null,
        },
        rta: {
          enabled: taskType === 'refuse_to_answer',
          rta_judge_model_id: rtaModelId || null,
          rta_prompt_template: rtaPromptContent || null,
        },
        ab_test: {
          enabled: false,
          statistical_test: 't_test',
        },
      };

      await createTask.mutateAsync({
        name,
        dataset_id: datasetId,
        model_ids: selectedModels,
        task_type: taskType,
        config,
      });

      onSuccess();
    } catch (err: any) {
      const detail = err?.response?.data?.detail || err?.message || 'Unknown error';
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-800 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-800 sticky top-0 bg-gray-900 z-10">
          <div>
            <h2 className="text-xl font-semibold text-white">Create New Task</h2>
            <p className="text-sm text-gray-400 mt-1">Configure your evaluation task</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg">
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

          {/* Step 1: Basic Info */}
          <div>
            <h3 className="text-sm font-medium text-white mb-4">1. Basic Information</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Task Name *</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., QA Evaluation v1"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Dataset *</label>
                <select
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                >
                  <option value="">Select dataset</option>
                  {datasets.map((ds: any) => (
                    <option key={ds.id} value={ds.id}>{ds.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Models * (select one or more)</label>
                <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-700 rounded-lg p-3 bg-gray-800/50">
                  {models.map((model: any) => (
                    <label key={model.id} className="flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-800">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, model.id]);
                          } else {
                            setSelectedModels(selectedModels.filter(id => id !== model.id));
                          }
                        }}
                        className="w-4 h-4"
                      />
                      <span className="text-sm text-white">{model.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Step 2: Task Type */}
          <div>
            <h3 className="text-sm font-medium text-white mb-4">2. Task Type *</h3>
            <div className="grid grid-cols-2 gap-3">
              {TASK_TYPES.map(type => (
                <button
                  key={type.value}
                  onClick={() => setTaskType(type.value)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${taskType === type.value
                    ? 'border-violet-600 bg-violet-600/10'
                    : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                    }`}
                >
                  <div className="text-sm font-medium text-white mb-1">{type.label}</div>
                  <div className="text-xs text-gray-400">{type.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Step 3: Type-specific Configuration */}
          {taskType === 'variation' && (
            <div className="border border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('variation')}
                className="w-full p-4 bg-gray-800/50 hover:bg-gray-800 flex items-center justify-between transition-colors"
              >
                <span className="text-sm font-medium text-white">3. Variation Configuration</span>
                {expandedSections.variation ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
              {expandedSections.variation && (
                <div className="p-4 space-y-4 bg-gray-900/50">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Num Variations per Strategy</label>
                    <input
                      type="number"
                      value={numVariations}
                      onChange={(e) => setNumVariations(parseInt(e.target.value) || 1)}
                      min="1"
                      max="10"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    />
                  </div>

                  {/* Tier tabs + strategy selection */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm text-gray-400">Variation Strategies <span className="text-red-400">*</span></label>
                      <span className="text-xs text-gray-500">
                        {selectedStrategies.length} selected
                      </span>
                    </div>

                    {/* Tier filter tabs + Select All */}
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex gap-1">
                        {(['all', 'A', 'B', 'C'] as const).map(tier => {
                          const isActive = variationTierFilter === tier;
                          const colors = TIER_TAB_COLORS[tier];
                          return (
                            <button
                              key={tier}
                              onClick={() => setVariationTierFilter(tier)}
                              className={`px-3 py-1 rounded text-xs font-medium border transition-colors ${isActive ? colors.active : colors.inactive}`}
                            >
                              {tier === 'all' ? 'All Tiers' : `Tier ${tier}`}
                            </button>
                          );
                        })}
                      </div>
                      <button
                        onClick={() => {
                          const visible = VARIATION_STRATEGIES.filter(
                            s => variationTierFilter === 'all' || s.tier === variationTierFilter
                          );
                          const allVisibleSelected = visible.every(s => selectedStrategies.includes(s.value));
                          if (allVisibleSelected) {
                            setSelectedStrategies(selectedStrategies.filter(
                              s => !visible.some(v => v.value === s)
                            ));
                          } else {
                            const toAdd = visible.filter(s => !selectedStrategies.includes(s.value)).map(s => s.value);
                            setSelectedStrategies([...selectedStrategies, ...toAdd]);
                          }
                        }}
                        className="text-xs text-violet-400 hover:text-violet-300 transition-colors whitespace-nowrap"
                      >
                        {(() => {
                          const visible = VARIATION_STRATEGIES.filter(
                            s => variationTierFilter === 'all' || s.tier === variationTierFilter
                          );
                          return visible.every(s => selectedStrategies.includes(s.value))
                            ? 'Deselect All'
                            : 'Select All';
                        })()}
                      </button>
                    </div>

                    {/* Strategy checkboxes, filtered by tier */}
                    <div className="grid grid-cols-2 gap-1.5 max-h-64 overflow-y-auto border border-gray-700 rounded-lg p-3 bg-gray-800">
                      {VARIATION_STRATEGIES
                        .filter(s => variationTierFilter === 'all' || s.tier === variationTierFilter)
                        .map(strategy => (
                          <label
                            key={strategy.value}
                            className={`flex items-start gap-2 p-1.5 rounded cursor-pointer hover:bg-gray-700/50 transition-colors ${selectedStrategies.includes(strategy.value) ? TIER_COLORS[strategy.tier] : ''}`}
                          >
                            <input
                              type="checkbox"
                              checked={selectedStrategies.includes(strategy.value)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedStrategies([...selectedStrategies, strategy.value]);
                                } else {
                                  setSelectedStrategies(selectedStrategies.filter(s => s !== strategy.value));
                                }
                              }}
                              className="w-3.5 h-3.5 mt-0.5"
                            />
                            <div className="min-w-0">
                              <div className="text-xs text-gray-200 leading-tight">{strategy.label}</div>
                              <div className="text-[10px] text-gray-500 leading-tight mt-0.5">{strategy.description}</div>
                            </div>
                          </label>
                        ))}
                    </div>
                  </div>

                  {/* Variation Model — only for Tier B/C */}
                  {(() => {
                    const hasLLM = selectedStrategies.some(s => {
                      const info = VARIATION_STRATEGIES.find(vs => vs.value === s);
                      return info && info.tier !== 'A';
                    });
                    if (!hasLLM) return null;
                    return (
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">
                          Variation Model <span className="text-red-400">*</span>
                          <span className="text-xs text-gray-500 ml-1">(required for Tier B/C)</span>
                        </label>
                        <select
                          value={variationModelId}
                          onChange={(e) => setVariationModelId(e.target.value)}
                          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                        >
                          <option value="">Select model for variations</option>
                          {models.map((m: any) => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                          ))}
                        </select>
                      </div>
                    );
                  })()}

                  {/* Custom prompt — only for Tier B/C */}
                  {(() => {
                    const hasLLM = selectedStrategies.some(s => {
                      const info = VARIATION_STRATEGIES.find(vs => vs.value === s);
                      return info && info.tier !== 'A';
                    });
                    if (!hasLLM) return null;
                    return (
                      <div>
                        <PromptSelector
                          selectedPromptId={variationPromptId}
                          onSelect={(prompt: Prompt | null) => {
                            setVariationPromptId(prompt?.id || null);
                            setVariationPromptContent(prompt?.content || '');
                            setVariationPromptError('');
                          }}
                          onContentChange={(content: string) => setVariationPromptContent(content)}
                          label="Variation Prompt"
                          error={variationPromptError}
                        />
                        <p className="text-xs text-gray-500 mt-1">Required for Tier B/C. Jinja2 template for LLM-based variation generation.</p>
                      </div>
                    );
                  })()}

                  {/* Tier A info */}
                  {selectedStrategies.length > 0 && selectedStrategies.every(s => {
                    const info = VARIATION_STRATEGIES.find(vs => vs.value === s);
                    return info && info.tier === 'A';
                  }) && (
                      <div className="text-xs text-blue-400/70 bg-blue-500/5 border border-blue-500/20 rounded-lg p-3">
                        Tier A strategies are deterministic and symbolic — no variation model or custom prompt needed.
                      </div>
                    )}
                </div>
              )}
            </div>
          )}

          {(taskType === 'judged' || taskType === 'variation') && (
            <div className="border border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('judge')}
                className="w-full p-4 bg-gray-800/50 hover:bg-gray-800 flex items-center justify-between transition-colors"
              >
                <span className="text-sm font-medium text-white">3. LLM Judge Configuration</span>
                {expandedSections.judge ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
              {expandedSections.judge && (
                <div className="p-4 space-y-4 bg-gray-900/50">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Judge Model *</label>
                    <select
                      value={judgeModelId}
                      onChange={(e) => setJudgeModelId(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      <option value="">Select judge model</option>
                      {models.map((m: any) => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <PromptSelector
                      selectedPromptId={judgePromptId}
                      onSelect={(prompt: Prompt | null) => {
                        setJudgePromptId(prompt?.id || null);
                        setJudgePromptContent(prompt?.content || '');
                        setJudgePromptError('');
                      }}
                      onContentChange={(content: string) => setJudgePromptContent(content)}
                      label="Judge Prompt *"
                      error={judgePromptError}
                    />
                    <p className="text-xs text-gray-500 mt-1">Judge response should be valid JSON with metrics</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {taskType === 'refuse_to_answer' && (
            <div className="border border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('rta')}
                className="w-full p-4 bg-gray-800/50 hover:bg-gray-800 flex items-center justify-between transition-colors"
              >
                <span className="text-sm font-medium text-white">3. RTA Judge Configuration</span>
                {expandedSections.rta ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
              {expandedSections.rta && (
                <div className="p-4 space-y-4 bg-gray-900/50">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">RTA Judge Model *</label>
                    <select
                      value={rtaModelId}
                      onChange={(e) => setRtaModelId(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                    >
                      <option value="">Select RTA judge model</option>
                      {models.map((m: any) => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <PromptSelector
                      selectedPromptId={rtaPromptId}
                      onSelect={(prompt: Prompt | null) => {
                        setRtaPromptId(prompt?.id || null);
                        setRtaPromptContent(prompt?.content || '');
                        setRtaPromptError('');
                      }}
                      onContentChange={(content: string) => setRtaPromptContent(content)}
                      label="RTA Judge Prompt *"
                      error={rtaPromptError}
                    />
                    <p className="text-xs text-gray-500 mt-1">RTA judge response should be valid JSON with refusal detection</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Advanced Settings */}
          <div>
            <h3 className="text-sm font-medium text-white mb-4">4. Advanced Settings</h3>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                min="1"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-800 sticky bottom-0 bg-gray-900 space-y-4">

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-sm text-red-400 whitespace-pre-line">
              {error}
            </div>
          )}
          <div className="flex items-center justify-between">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-white text-sm"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={!name || !datasetId || selectedModels.length === 0}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 rounded-lg text-white text-sm flex items-center gap-2"
            >
              <Plus size={16} />
              Create Task
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}