import React, { useEffect } from 'react';
import { Database, Cpu, Zap, CheckCircle, Pause, X, Play } from 'lucide-react';
import { useDatasets, useModels, useTasks, usePauseTask, useResumeTask, useCancelTask } from '@/api/hooks';
import { useAppStore } from '@/stores/useAppStore';
import StatusBadge from '@/components/common/StatusBadge';
import { formatDate } from '@/utils/format';

export default function DashboardSection() {
  const { data: datasets = [], isLoading: datasetsLoading } = useDatasets();
  const { data: models = [], isLoading: modelsLoading } = useModels();
  const { data: tasks = [], isLoading: tasksLoading } = useTasks();
  const pauseTask = usePauseTask();
  const resumeTask = useResumeTask();
  const cancelTask = useCancelTask();

  const isLoading = datasetsLoading || modelsLoading || tasksLoading;

  const safeFilter = <T,>(arr: T[], predicate: (item: T) => boolean): T[] => {
    if (!Array.isArray(arr)) return [];
    return arr.filter(predicate);
  };

const stats = [
    {
      label: 'Total Datasets',
      value: datasets.length,
      icon: Database,
      color: 'text-blue-400'
    },
    {
      label: 'Active Models',
      value: safeFilter(models, m => m.status === 'registered').length,
      icon: Cpu,
      color: 'text-violet-400'
    },
    {
      label: 'Running Tasks',
      value: safeFilter(tasks, t => t.status === 'running').length,
      icon: Zap,
      color: 'text-amber-400'
    },
    {
      label: 'Completed',
      value: safeFilter(tasks, t => t.status === 'completed').length,
      icon: CheckCircle,
      color: 'text-emerald-400'
    },
  ];

  const runningTasks = safeFilter(tasks, t => t.status === 'running');
  const pausedTasks = safeFilter(tasks, t => t.status === 'paused');

  return (
    <div className="p-6 space-y-6 animate-slideIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white mb-1">Dashboard</h1>
        <p className="text-sm text-gray-400">Monitor your evaluation pipeline</p>
      </div>

      {isLoading && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center">
          <div className="animate-pulse flex items-center justify-center gap-3">
            <div className="w-4 h-4 bg-violet-500 rounded-full animate-bounce" />
            <span className="text-gray-400">Loading data...</span>
          </div>
        </div>
      )}

      {!isLoading && (
        <>
          {/* Stats Grid */}
          <div className="grid grid-cols-4 gap-4">
            {stats.map((stat, i) => {
              const Icon = stat.icon;
              return (
                <div
                  key={i}
                  className="bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-gray-700 transition-all"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-gray-400 text-sm">{stat.label}</span>
                    <Icon size={18} className={`${stat.color} opacity-60`} />
                  </div>
                  <div className="text-3xl font-semibold text-white">{stat.value}</div>
                </div>
              );
            })}
          </div>

      {/* Active Tasks */}
      {runningTasks.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Active Tasks</h2>
            <span className="text-xs text-gray-500">Auto-refreshing every 3s</span>
          </div>

          <div className="space-y-4">
            {runningTasks.map(task => (
              <div key={task.id} className="bg-black/40 border border-gray-800 rounded-lg p-4">
                {/* Task Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm font-medium text-white">{task.name}</h3>
                      <StatusBadge status={task.status} size="sm" />
                    </div>
                    <p className="text-xs text-gray-500">
                      {task.processed_samples ?? 0} / {task.total_samples ?? 0} samples ({((task.progress ?? 0)).toFixed(1)}%)
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => pauseTask.mutate(task.id)}
                      className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded text-orange-400 transition-colors"
                      title="Pause task"
                    >
                      <Pause size={14} />
                    </button>
                    <button
                      onClick={() => cancelTask.mutate(task.id)}
                      className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded text-red-400 transition-colors"
                      title="Cancel task"
                    >
                      <X size={14} />
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden mb-3">
                  <div
                    className="h-full bg-gradient-to-r from-violet-600 to-purple-600 transition-all duration-300"
                    style={{ width: `${task.progress}%` }}
                  />
                </div>

                {/* Current Execution */}
                {task.current_execution && (
                  <div className="bg-gray-900/50 border border-gray-800 rounded p-3 mb-3">
                    <span className="text-xs text-blue-400 font-medium block mb-2">Currently Processing</span>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <span className="text-gray-500">Model:</span>
                      <span className="text-gray-200">{task.current_execution.model_name || '—'}</span>
                      <span className="text-gray-500">Models:</span>
                      <span className="text-gray-200">{task.current_execution.model_progress || '—'}</span>
                      <span className="text-gray-500">Throughput:</span>
                      <span className="text-gray-200">{task.current_execution.throughput ? `${task.current_execution.throughput}/s` : '—'}</span>
                      <span className="text-gray-500">ETA:</span>
                      <span className="text-gray-200">{task.current_execution.eta_seconds != null ? `${Math.round(task.current_execution.eta_seconds)}s` : '—'}</span>
                    </div>
                  </div>
                )}

                {/* Recent Executions */}
                {task.recent_executions && task.recent_executions.length > 0 && (
                  <div className="space-y-2">
                    <span className="text-xs text-emerald-400 font-medium">✅ Recent Completions</span>
                    {task.recent_executions.slice(0, 2).map((recent, idx) => (
                      <div key={idx} className="bg-gray-900/30 border border-gray-800 rounded p-2">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-400">
                            Item #{recent.index ?? '?'} • {recent.model_name ?? '?'}
                          </span>
                          {recent.completed_at && (
                            <span className="text-xs text-gray-500">
                              {(() => {
                                try {
                                  const d = new Date(recent.completed_at);
                                  return isNaN(d.getTime()) ? '' : d.toLocaleTimeString();
                                } catch { return ''; }
                              })()}
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">Input:</span>
                            <p className="text-gray-400 truncate">{(recent.prompt ?? '').slice(0, 40)}...</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Output:</span>
                            <p className="text-gray-400 truncate">{(recent.output ?? '').slice(0, 40)}...</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Paused Tasks */}
      {pausedTasks.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Paused Tasks</h2>
          <div className="space-y-3">
            {pausedTasks.map(task => (
              <div key={task.id} className="bg-black/40 border border-gray-800 rounded-lg p-4 flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-sm font-medium text-white">{task.name}</h3>
                    <StatusBadge status={task.status} size="sm" />
                  </div>
                  <p className="text-xs text-gray-500">
                    Progress: {task.processed_samples ?? 0}/{task.total_samples ?? 0} ({((task.progress ?? 0)).toFixed(1)}%)
                    {task.paused_at && ` • Paused ${formatDate(task.paused_at)}`}
                  </p>
                </div>
                <button
                  onClick={() => resumeTask.mutate(task.id)}
                  className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded-lg text-white text-sm font-medium flex items-center gap-1.5 transition-colors"
                >
                  <Play size={14} />
                  Resume
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {runningTasks.length === 0 && pausedTasks.length === 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <Zap size={48} className="mx-auto text-gray-700 mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No active tasks</h3>
          <p className="text-sm text-gray-500 mb-6">Create a new task to start evaluating models</p>
          <button
            onClick={() => useAppStore.getState().setActiveSection('tasks')}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-700 rounded-lg text-white text-sm font-medium transition-colors"
          >
            Create Task
          </button>
        </div>
      )}
        </>
      )}
    </div>
  );
}