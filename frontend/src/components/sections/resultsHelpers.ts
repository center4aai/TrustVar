import type { CompletionSummary, Model, Task, TaskResult } from '../../api/types';

export interface FailedModelInfo {
  modelId: string;
  modelName: string;
  errorCount: number;
}

export interface PartialRunInfo {
  isPartial: boolean;
  failedModels: string[];
  modelsWithResults: number;
  modelsTotal: number;
  message: string | null;
}

type PartialRunTask = Pick<
  Task,
  'status' | 'completion_summary' | 'processed_samples' | 'model_ids'
>;

export function getPartialRunInfo(
  task: PartialRunTask,
  models: Pick<Model, 'id' | 'name'>[] = [],
): PartialRunInfo {
  const cs: CompletionSummary | null | undefined = task.completion_summary;
  if (cs) {
    const failed = cs.failed_models ?? [];
    const resolveName = (id: string): string => {
      const m = models.find((mm) => mm.id === id);
      return m?.name || `${id.slice(0, 12)}…`;
    };
    const failedNames = failed.map(resolveName);
    return {
      isPartial: !cs.is_complete,
      failedModels: failed,
      modelsWithResults: cs.n_models_with_results,
      modelsTotal: cs.n_models_total,
      message: cs.is_complete
        ? null
        : `Partial run: ${cs.n_models_with_results}/${cs.n_models_total} model(s) produced results` +
        (failedNames.length ? ` (no rows from: ${failedNames.join(', ')})` : '') +
        '. TrustVar metrics may be unreliable — interpret with care.',
    };
  }
  // Legacy run without completion_summary: zero processed rows ⇒ partial.
  const isPartial = task.status === 'completed' && task.processed_samples === 0;
  return {
    isPartial,
    failedModels: [],
    modelsWithResults: 0,
    modelsTotal: task.model_ids?.length ?? 0,
    message: isPartial
      ? 'Partial run: no results were recorded — TrustVar metrics are not available.'
      : null,
  };
}

export interface VariantYieldInfo {
  enabled: boolean;
  expectedStrategies: number;
  realizedStrategies: number;
  totalVariantRows: number;
  warn: boolean;
  message: string | null;
}


export function getVariantYieldInfo(
  config: Task['config'] | undefined,
  results: Pick<TaskResult, 'variation_type'>[] | undefined,
): VariantYieldInfo {
  const enabled = !!config?.variations?.enabled;
  const expectedStrategies = config?.variations?.strategies?.length ?? 0;
  const rows = results ?? [];
  const variantRows = rows.filter((r) => !!r.variation_type);
  const realized = new Set(variantRows.map((r) => r.variation_type as string));
  const warn = enabled && expectedStrategies > 0 && realized.size === 0;
  return {
    enabled,
    expectedStrategies,
    realizedStrategies: realized.size,
    totalVariantRows: variantRows.length,
    warn,
    message: warn
      ? `No prompt variants were generated (0 of ${expectedStrategies} configured strategies). ` +
      'TSI/EAR cannot be computed — the task has no variation to measure.'
      : null,
  };
}


export function getFailedModelInfo(
  completionSummary: CompletionSummary | null | undefined,
  models: Pick<Model, 'id' | 'name'>[],
): FailedModelInfo[] {
  if (!completionSummary || !completionSummary.failed_models?.length) return [];
  return completionSummary.failed_models.map((id) => {
    const model = models.find((m) => m.id === id);
    return {
      modelId: id,
      modelName: model?.name || `${id.slice(0, 12)}…`,
      errorCount: completionSummary.per_model_errors?.[id] || 0,
    };
  });
}


export function isExactMatchMeaningful(taskType: string | undefined): boolean {
  return taskType === 'mcq' || taskType === 'classification';
}


export const NON_MODEL_AGGREGATION_KEYS: ReadonlySet<string> = new Set([
  '_trustvar',
  '_trustvar_error',
  'judge',
  'rta',
  'include_exclude',
]);

export function isModelAggregationKey(key: string): boolean {
  return !NON_MODEL_AGGREGATION_KEYS.has(key);
}

export function filterModelEntries<K extends string, V>(
  entries: ReadonlyArray<readonly [K, V]>,
): Array<[K, V]> {
  return entries.filter(([k]) => isModelAggregationKey(k)) as Array<[K, V]>;
}


export function getGenerationEarSignature(
  canonicalTaskType: string | undefined,
): string | null {
  if (canonicalTaskType !== 'generation') return null;
  return (
    'EAR is N/A for `generation` tasks: ' +
    'equivalence can be defined for free-form generation. Headline metrics ' +
    'are TSI, IQR-CV, variance decomposition, and the JSD-companion ' +
    '(output-distribution divergence across tiers).'
  );
}


export function isRefused(refused: unknown): boolean {
  if (refused == null) return false;
  if (typeof refused === 'boolean') return refused;
  if (typeof refused === 'number') return refused === 1;
  const v = String(refused).trim().toLowerCase();
  return v === '1' || v === 'yes' || v === 'true';
}

export type ValidatorCategory =
  | 'ACCEPT'
  | 'FLAG'
  | 'REJECT'
  | 'BYPASSED'
  | 'UNKNOWN';


export function categorizeValidatorVerdict(
  verdict: string | null | undefined,
): ValidatorCategory {
  if (!verdict) return 'UNKNOWN';
  const v = verdict.trim().toLowerCase();
  if (v.startsWith('accept')) return 'ACCEPT';
  if (v.startsWith('reject')) return 'REJECT';
  if (v.startsWith('flag')) return 'FLAG';
  if (v.startsWith('bypass')) return 'BYPASSED';
  return 'UNKNOWN';
}
