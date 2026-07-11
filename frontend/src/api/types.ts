export interface Dataset {
  id: string;
  name: string;
  description: string;
  size: number;
  task_type: string;
  format: string;
  created_at: string;
  updated_at?: string;
  tags?: string[];
  prompt_column?: string;
  target_column?: string;
  include_column?: string;
  exclude_column?: string;
  template_column?: string;
  variables_columns?: string[];
  metadata?: Record<string, any>;
}

export interface DatasetItem {
  id: string;
  prompt: string;
  target?: string;
  metadata?: Record<string, any>;
}

export interface DatasetStats {
  total_items: number;
  avg_prompt_length: number;
  items_with_target: number;
  coverage: number;
}

export interface Model {
  id: string;
  name: string;
  provider: 'ollama' | 'huggingface' | 'openai';
  model_name: string;
  status: 'registered' | 'downloading' | 'failed';
  config: {
    temperature: number;
    max_tokens: number;
    top_p?: number;
    top_k?: number;
  };
  description?: string;
  created_at: string;
}

export interface AvailableModel {
  name: string;
  size: number;
  modified_at: string;
  digest?: string;
  details?: Record<string, any>;
}

export interface BulkRegisterResponse {
  created: Model[];
  skipped: string[];
  downloading: string[];
}

export interface CompletionSummary {
  is_complete: boolean;
  final_count: number;
  expected_total: number;
  n_models_total: number;
  n_models_with_results: number;
  failed_models: string[];
  per_model_generated: Record<string, number>;
  per_model_errors: Record<string, number>;
}

export interface Task {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  task_type: string;
  progress: number;
  processed_samples: number;
  total_samples: number;
  dataset_id: string;
  model_ids: string[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  paused_at?: string;
  error?: string;
  // RUN-3: сбой пост-расчёта trustvar метрик (отсутствует = OK)
  metrics_error?: string;
  current_execution?: {
    operation_type?: string;
    index?: number;
    model_name?: string;
    model_progress?: string;
    throughput?: number;
    eta_seconds?: number;
    last_item_index?: number;
    prompt?: string;
    output?: string;
    started_at?: string;
    execution_time?: number;
    prompt_variation?: string;
  };
  recent_executions?: Array<{
    operation_type: string;
    index: number;
    model_name: string;
    prompt: string;
    output?: string;
    error?: string;
    completed_at: string;
    execution_time?: number;
    prompt_variation?: string;
  }>;
  aggregated_metrics?: Record<string, Record<string, number>>;
  // WEB-5/WEB-6: run completeness + per-model coverage (null until run ends)
  completion_summary?: CompletionSummary | null;
  config?: {
    batch_size: number;
    max_samples?: number;
    evaluate: boolean;
    evaluation_metrics: string[];
    variations: {
      enabled: boolean;
      model_id?: string;
      strategies: string[];
      count_per_strategy: number;
      custom_prompt?: string;
    };
    judge: {
      enabled: boolean;
      model_id?: string;
      criteria: string[];
      custom_prompt_template?: string;
    };
    rta: {
      enabled: boolean;
      rta_judge_model_id?: string;
      rta_prompt_template?: string;
    };
    ab_test: {
      enabled: boolean;
      strategy?: string;
      prompt_variants?: Record<string, string>;
      temperatures?: number[];
      system_prompts?: Record<string, string>;
      sample_size_per_variant?: number;
      statistical_test: string;
    };
  };
  // results is no longer embedded in Task; fetch via GET /tasks/{id}/results
  results?: TaskResult[];
  ab_test_results?: any;
}

export interface ResultsPage {
  task_id: string;
  total: number;
  skip: number;
  limit: number;
  results: TaskResult[];
}

export interface TaskResult {
  id: string;
  task_id: string;
  model_id: string;
  input: string;
  output: string;
  target?: string;
  variation_type?: string;
  judge_score?: number;
  judge_results?: Record<string, any>;
  include_score?: number;
  refused?: string;
  metadata?: Record<string, any>;
  execution_time?: number;
}

export interface Prompt {
  id: string;
  name: string;
  content: string;
  prompt_type: 'judge' | 'rta' | 'variation';
  description?: string;
  created_at: string;
  updated_at?: string;
}

export interface CreatePromptPayload {
  name: string;
  content: string;
  prompt_type: 'judge' | 'rta' | 'variation';
  description?: string;
}

export interface CreateDatasetPayload {
  name: string;
  description: string;
  task_type: string;
  task_semantics?: string;
  tags: string;
}

export interface UploadDatasetPayload {
  file: File;
  file_format: string;
  prompt_column: string;
  target_column?: string;
  include_column?: string;
  exclude_column?: string;
  template_column?: string;
  variables_columns?: string;
}
export interface TierAggregate {
  mean: number;
  std: number;
  ci_low: number;
  ci_high: number;
  n_tasks: number;
}

export interface TrustVarMetrics {
  per_task_tsi: Record<string, Record<string, number>>;
  per_task_ear: Record<string, Record<string, number>>;
  per_task_cv: Record<string, Record<string, number>>;
  per_task_iqr_cv: Record<string, Record<string, number>>;
  per_task_uninformative: Record<string, Record<string, boolean>>;
  per_task_ear_flags?: Record<string, Record<string, string>>;
  per_task_cv_unreliable?: Record<string, Record<string, boolean>>;
  model_cv_star: Record<string, Record<string, number>>;
  aggregate_tsi: Record<string, TierAggregate>;
  aggregate_ear: Record<string, TierAggregate>;
  variance_decomposition: Record<string, any>;
  tier_comparison: Record<string, any>;
  bootstrap_replicates: {
    tsi: { benchmark?: Record<string, number[]>; per_task?: Record<string, number[]> };
    ear: { benchmark?: Record<string, number[]>; per_task?: Record<string, number[]> };
  };
  n_models: number;
  n_resamples: number;
  ci_level: number;
}
