import axios, { AxiosInstance } from 'axios';
import type {
  Dataset,
  Model,
  Task,
  TaskResult,
  ResultsPage,
  DatasetItem,
  DatasetStats,
  CreateDatasetPayload,
  UploadDatasetPayload,
  Prompt,
  CreatePromptPayload,
  TrustVarMetrics,
  AvailableModel,
  BulkRegisterResponse,
} from './types';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // ===== Datasets =====
  async listDatasets(): Promise<Dataset[]> {
    const { data } = await this.client.get('/api/v1/datasets/');
    return data;
  }

  async getDataset(id: string): Promise<Dataset> {
    const { data } = await this.client.get(`/api/v1/datasets/${id}`);
    return data;
  }

  async deleteDataset(id: string): Promise<void> {
    await this.client.delete(`/api/v1/datasets/${id}`);
  }

  async getDatasetItems(id: string, skip = 0, limit = 10): Promise<DatasetItem[]> {
    const { data } = await this.client.get(`/api/v1/datasets/${id}/items`, {
      params: { skip, limit }
    });
    return data;
  }

  async getDatasetStats(id: string): Promise<DatasetStats> {
    const { data } = await this.client.get(`/api/v1/datasets/${id}/stats`);
    return data;
  }

  async createDatasetAndUpload(
    datasetInfo: CreateDatasetPayload,
    uploadInfo: UploadDatasetPayload
  ): Promise<any> {
    // 1. Create dataset with FormData
    const createFormData = new FormData();
    createFormData.append('name', datasetInfo.name);
    createFormData.append('description', datasetInfo.description);
    createFormData.append('task_type', datasetInfo.task_type);
    if (datasetInfo.task_semantics) {
      createFormData.append('task_semantics', datasetInfo.task_semantics);
    }
    createFormData.append('tags', datasetInfo.tags);

    const { data: dataset } = await this.client.post('/api/v1/datasets/', createFormData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    const datasetId = dataset.id;

    // 2. Upload file with FormData
    const uploadFormData = new FormData();
    uploadFormData.append('file', uploadInfo.file);
    uploadFormData.append('file_format', uploadInfo.file_format);
    uploadFormData.append('prompt_column', uploadInfo.prompt_column);
    if (uploadInfo.target_column) {
      uploadFormData.append('target_column', uploadInfo.target_column);
    }
    if (uploadInfo.include_column) {
      uploadFormData.append('include_column', uploadInfo.include_column);
    }
    if (uploadInfo.exclude_column) {
      uploadFormData.append('exclude_column', uploadInfo.exclude_column);
    }
    if (uploadInfo.template_column) {
      uploadFormData.append('template_column', uploadInfo.template_column);
    }
    if (uploadInfo.variables_columns) {
      uploadFormData.append('variables_columns', uploadInfo.variables_columns);
    }

    const { data: uploadResult } = await this.client.post(
      `/api/v1/datasets/${datasetId}/upload`,
      uploadFormData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return uploadResult;
  }

  // ===== Models =====
  async listModels(): Promise<Model[]> {
    const { data } = await this.client.get('/api/v1/models/');
    return data;
  }

  async getModel(id: string): Promise<Model> {
    const { data } = await this.client.get(`/api/v1/models/${id}/get`);
    return data;
  }

  async registerModel(modelData: Partial<Model>): Promise<Model> {
    const { data } = await this.client.post('/api/v1/models/', modelData);
    return data;
  }

  async deleteModel(id: string): Promise<void> {
    await this.client.delete(`/api/v1/models/${id}`);
  }

  async testModel(id: string, testPrompt: string): Promise<any> {
    const { data } = await this.client.post(`/api/v1/models/${id}/test`, {
      test_prompt: testPrompt
    });
    return data;
  }

  async getTestResult(modelId: string, celeryTaskId: string): Promise<any> {
    const { data } = await this.client.get(`/api/v1/models/${modelId}/test/${celeryTaskId}`);
    return data;
  }

  async getAvailableModels(): Promise<{ models: AvailableModel[] }> {
    const { data } = await this.client.get('/api/v1/models/available');
    return data;
  }

  async bulkRegisterModels(payload: {
    model_names: string[];
    config?: Partial<Model['config']>;
    description?: string;
    provider?: string;
    pull_if_missing?: boolean;
  }): Promise<BulkRegisterResponse> {
    const { data } = await this.client.post('/api/v1/models/bulk-register', payload);
    return data;
  }

  // ===== Tasks =====
  async listTasks(status?: string): Promise<Task[]> {
    const { data } = await this.client.get('/api/v1/tasks/', {
      params: status ? { status } : {},
    });
    return data;
  }

  async getTask(id: string): Promise<Task> {
    const { data } = await this.client.get(`/api/v1/tasks/${id}`);
    return data;
  }

  async createTask(taskData: any): Promise<Task> {
    // Backend ожидает структуру TaskCreate
    const payload = {
      name: taskData.name,
      dataset_id: taskData.dataset_id,
      model_ids: taskData.model_ids, // Это массив
      task_type: taskData.task_type || 'standard',
      config: taskData.config || {
        batch_size: 1,
        max_samples: null,
        evaluate: true,
        evaluation_metrics: ['accuracy'],
        variations: {
          enabled: false,
          strategies: [],
          count_per_strategy: 0
        },
        judge: {
          enabled: false,
          criteria: []
        },
        rta: {
          enabled: false
        },
        ab_test: {
          enabled: false,
          statistical_test: 't_test'
        }
      }
    };

    const { data } = await this.client.post('/api/v1/tasks/', payload);
    return data;
  }

  async pauseTask(id: string): Promise<void> {
    await this.client.post(`/api/v1/tasks/${id}/pause`);
  }

  async resumeTask(id: string): Promise<void> {
    await this.client.post(`/api/v1/tasks/${id}/resume`);
  }

  async cancelTask(id: string): Promise<void> {
    await this.client.post(`/api/v1/tasks/${id}/cancel`);
  }

  async deleteTask(id: string): Promise<void> {
    await this.client.delete(`/api/v1/tasks/${id}`);
  }

  async getTaskResults(taskId: string, skip = 0, limit = 10000): Promise<ResultsPage> {
    const { data } = await this.client.get(`/api/v1/tasks/${taskId}/results`, {
      params: { skip, limit },
    });
    return data;
  }

  async compareModels(taskId: string): Promise<any> {
    const { data } = await this.client.get(`/api/v1/tasks/${taskId}/compare-models`);
    return data;
  }

  async getTaskTrustVarMetrics(taskId: string): Promise<TrustVarMetrics> {
    const { data } = await this.client.get(`/api/v1/tasks/${taskId}/trustvar-metrics`);
    return data;
  }

  async downloadTaskResults(taskId: string): Promise<void> {
    const response = await this.client.get(`/api/v1/tasks/${taskId}/export`, {
      responseType: 'blob',
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `task_${taskId}_results.json`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }

  // ===== Prompts =====
  async listPrompts(): Promise<Prompt[]> {
    const { data } = await this.client.get('/api/v1/prompts/');
    return data;
  }

  async getPrompt(id: string): Promise<Prompt> {
    const { data } = await this.client.get(`/api/v1/prompts/${id}`);
    return data;
  }

  async createPrompt(promptData: CreatePromptPayload): Promise<Prompt> {
    const { data } = await this.client.post('/api/v1/prompts/', promptData);
    return data;
  }

  async updatePrompt(id: string, updateData: Partial<CreatePromptPayload>): Promise<Prompt> {
    const { data } = await this.client.put(`/api/v1/prompts/${id}`, updateData);
    return data;
  }

  async deletePrompt(id: string): Promise<void> {
    await this.client.delete(`/api/v1/prompts/${id}`);
  }
}

export const apiClient = new ApiClient();