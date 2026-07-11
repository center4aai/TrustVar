import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';
import type { CreateDatasetPayload, UploadDatasetPayload } from './types';

// ===== Datasets =====
export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: () => apiClient.listDatasets(),
    refetchInterval: 30000,
  });
}

export function useDataset(id: string | null) {
  return useQuery({
    queryKey: ['dataset', id],
    queryFn: () => apiClient.getDataset(id!),
    enabled: !!id,
  });
}

export function useDatasetItems(id: string | null, skip = 0, limit = 10) {
  return useQuery({
    queryKey: ['dataset-items', id, skip, limit],
    queryFn: () => apiClient.getDatasetItems(id!, skip, limit),
    enabled: !!id,
  });
}

export function useDatasetStats(id: string | null) {
  return useQuery({
    queryKey: ['dataset-stats', id],
    queryFn: () => apiClient.getDatasetStats(id!),
    enabled: !!id,
  });
}

export function useCreateDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ datasetInfo, uploadInfo }: {
      datasetInfo: CreateDatasetPayload;
      uploadInfo: UploadDatasetPayload;
    }) => apiClient.createDatasetAndUpload(datasetInfo, uploadInfo),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

// ===== Models =====
export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => apiClient.listModels(),
    refetchInterval: 30000,
  });
}

export function useModel(id: string | null) {
  return useQuery({
    queryKey: ['model', id],
    queryFn: () => apiClient.getModel(id!),
    enabled: !!id,
  });
}

export function useRegisterModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelData: any) => apiClient.registerModel(modelData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

export function useTestModel() {
  return useMutation({
    mutationFn: ({ id, prompt }: { id: string; prompt: string }) =>
      apiClient.testModel(id, prompt),
  });
}

// ===== Tasks =====
export function useTasks() {
  return useQuery({
    queryKey: ['tasks'],
    queryFn: () => apiClient.listTasks(),
    refetchInterval: 3000,
  });
}

export function useTask(id: string | null) {
  return useQuery({
    queryKey: ['task', id],
    queryFn: () => apiClient.getTask(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      return query.state.data?.status === 'running'
        ? 3000
        : false;
    },
  });
}

export function useCreateTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (taskData: any) => apiClient.createTask(taskData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function usePauseTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.pauseTask(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useResumeTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.resumeTask(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useCancelTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.cancelTask(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useDeleteTask() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteTask(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useTaskResults(taskId: string | null, skip = 0, limit = 10000) {
  return useQuery({
    queryKey: ['task-results', taskId, skip, limit],
    queryFn: () => apiClient.getTaskResults(taskId!, skip, limit),
    enabled: !!taskId,
    staleTime: 30000,
  });
}

export function useCompareModels(taskId: string | null) {
  return useQuery({
    queryKey: ['compare-models', taskId],
    queryFn: () => apiClient.compareModels(taskId!),
    enabled: !!taskId,
  });
}

// ===== Prompts =====
export function usePrompts() {
  return useQuery({
    queryKey: ['prompts'],
    queryFn: () => apiClient.listPrompts(),
    refetchInterval: 30000,
  });
}

export function useCreatePrompt() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (promptData: any) => apiClient.createPrompt(promptData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] });
    },
  });
}

export function useDeletePrompt() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.deletePrompt(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] });
    },
  });
}
export function useTaskTrustVarMetrics(taskId: string | null) {
  return useQuery({
    queryKey: ['trustvar-metrics', taskId],
    queryFn: () => apiClient.getTaskTrustVarMetrics(taskId!),
    enabled: !!taskId,
    staleTime: Infinity,
  });
}

// ===== Ollama Discovery =====
export function useAvailableModels(provider?: string) {
  return useQuery({
    queryKey: ['availableModels', provider],
    queryFn: () => apiClient.getAvailableModels(),
    staleTime: 30_000,
    enabled: provider === 'ollama',
  });
}

export function useBulkRegisterModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      model_names: string[];
      config?: Record<string, any>;
      description?: string;
      provider?: string;
      pull_if_missing?: boolean;
    }) => apiClient.bulkRegisterModels(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
