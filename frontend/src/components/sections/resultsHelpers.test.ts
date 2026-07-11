import { describe, it, expect } from 'vitest';
import {
  getFailedModelInfo,
  getPartialRunInfo,
  getVariantYieldInfo,
  isExactMatchMeaningful,
} from './resultsHelpers';

describe('getPartialRunInfo (WEB-5 / WEB-6)', () => {
  const modelList = [
    { id: 'm1', name: 'Model One' },
    { id: 'm2', name: 'Model Two' },
    { id: 'm3', name: 'Model Three' },
  ];

  it('reports complete when completion_summary.is_complete', () => {
    const info = getPartialRunInfo({
      status: 'completed',
      processed_samples: 78,
      model_ids: ['m1', 'm2'],
      completion_summary: {
        is_complete: true,
        final_count: 78,
        expected_total: 80,
        n_models_total: 2,
        n_models_with_results: 2,
        failed_models: [],
        per_model_generated: { m1: 40, m2: 38 },
        per_model_errors: {},
      },
    } as any, modelList);
    expect(info.isPartial).toBe(false);
    expect(info.message).toBeNull();
  });

  it('flags partial + resolves failed model IDs to names in the message', () => {
    const info = getPartialRunInfo({
      status: 'completed',
      processed_samples: 78,
      model_ids: ['m1', 'm2', 'm3'],
      completion_summary: {
        is_complete: false,
        final_count: 78,
        expected_total: 120,
        n_models_total: 3,
        n_models_with_results: 2,
        failed_models: ['m3'],
        per_model_generated: { m1: 40, m2: 38 },
        per_model_errors: { m3: 17 },
      },
    } as any, modelList);
    expect(info.isPartial).toBe(true);
    expect(info.failedModels).toEqual(['m3']);
    expect(info.message).toContain('Model Three');
    expect(info.message).not.toContain('m3,');
    expect(info.message).toContain('2/3');
  });

  it('falls back to truncated ID when the failed model is not in the list', () => {
    const info = getPartialRunInfo({
      status: 'completed',
      processed_samples: 0,
      model_ids: ['m1', 'orphan-abcdef0123'],
      completion_summary: {
        is_complete: false,
        final_count: 0,
        expected_total: 10,
        n_models_total: 2,
        n_models_with_results: 0,
        failed_models: ['orphan-abcdef0123'],
        per_model_generated: {},
        per_model_errors: { 'orphan-abcdef0123': 5 },
      },
    } as any, [{ id: 'm1', name: 'Model One' }]);
    expect(info.message).toContain('orphan-abcde…');
  });

  it('falls back to processed==0 for legacy runs without completion_summary', () => {
    const info = getPartialRunInfo({
      status: 'completed',
      processed_samples: 0,
      model_ids: ['m1'],
      completion_summary: null,
    } as any, modelList);
    expect(info.isPartial).toBe(true);
    expect(info.message).toContain('no results');
  });

  it('legacy run with results is not flagged partial', () => {
    const info = getPartialRunInfo({
      status: 'completed',
      processed_samples: 50,
      model_ids: ['m1'],
      completion_summary: null,
    } as any, modelList);
    expect(info.isPartial).toBe(false);
  });
});

describe('getFailedModelInfo (WEB-6)', () => {
  it('returns empty when no completion_summary', () => {
    expect(getFailedModelInfo(undefined, [{ id: 'm1', name: 'Model 1' }])).toEqual([]);
    expect(getFailedModelInfo(null, [{ id: 'm1', name: 'Model 1' }])).toEqual([]);
  });

  it('returns empty when no failed models', () => {
    const cs = {
      is_complete: true,
      final_count: 10,
      expected_total: 10,
      n_models_total: 1,
      n_models_with_results: 1,
      failed_models: [],
      per_model_generated: { m1: 10 },
      per_model_errors: {},
    };
    expect(getFailedModelInfo(cs as any, [{ id: 'm1', name: 'Model 1' }])).toEqual([]);
  });

  it('resolves IDs to names and error counts', () => {
    const cs = {
      is_complete: false,
      final_count: 5,
      expected_total: 10,
      n_models_total: 2,
      n_models_with_results: 1,
      failed_models: ['m2'],
      per_model_generated: { m1: 5 },
      per_model_errors: { m2: 7 },
    };
    expect(getFailedModelInfo(cs as any, [
      { id: 'm1', name: 'Model 1' },
      { id: 'm2', name: 'Model 2' },
    ])).toEqual([
      { modelId: 'm2', modelName: 'Model 2', errorCount: 7 },
    ]);
  });

  it('falls back to truncated ID when model is unknown', () => {
    const cs = {
      is_complete: false,
      final_count: 0,
      expected_total: 5,
      n_models_total: 1,
      n_models_with_results: 0,
      failed_models: ['unknown-model-id-123'],
      per_model_generated: {},
      per_model_errors: {},
    };
    const info = getFailedModelInfo(cs as any, []);
    expect(info).toHaveLength(1);
    expect(info[0].modelName).toBe('unknown-mode…');
  });
});

describe('getVariantYieldInfo (WEB-7)', () => {
  const config = { variations: { enabled: true, strategies: ['a', 'b', 'c'] } } as any;

  it('warns when a variation task produced no variants', () => {
    const info = getVariantYieldInfo(config, [
      { variation_type: null },
      { variation_type: undefined },
    ] as any);
    expect(info.warn).toBe(true);
    expect(info.realizedStrategies).toBe(0);
    expect(info.message).toContain('0 of 3');
  });

  it('does not warn when variants were realized', () => {
    const info = getVariantYieldInfo(config, [
      { variation_type: 'a' },
      { variation_type: 'a' },
      { variation_type: 'b' },
      { variation_type: null },
    ] as any);
    expect(info.warn).toBe(false);
    expect(info.realizedStrategies).toBe(2);
    expect(info.totalVariantRows).toBe(3);
  });

  it('is inert when variations are disabled', () => {
    const info = getVariantYieldInfo({ variations: { enabled: false, strategies: [] } } as any, []);
    expect(info.enabled).toBe(false);
    expect(info.warn).toBe(false);
    expect(info.message).toBeNull();
  });
});

describe('isExactMatchMeaningful (WEB-4)', () => {
  it('true for closed-form task types', () => {
    expect(isExactMatchMeaningful('mcq')).toBe(true);
    expect(isExactMatchMeaningful('classification')).toBe(true);
  });
  it('false for generative task types', () => {
    expect(isExactMatchMeaningful('open_qa')).toBe(false);
    expect(isExactMatchMeaningful('generation')).toBe(false);
    expect(isExactMatchMeaningful(undefined)).toBe(false);
  });
});


import {
  isModelAggregationKey,
  filterModelEntries,
  getGenerationEarSignature,
  NON_MODEL_AGGREGATION_KEYS,
  isRefused,
  categorizeValidatorVerdict,
} from './resultsHelpers';

describe('isModelAggregationKey / filterModelEntries (N1)', () => {
  it('isModelAggregationKey filters non-model keys', () => {
    for (const k of NON_MODEL_AGGREGATION_KEYS) {
      expect(isModelAggregationKey(k as string)).toBe(false);
    }
    expect(isModelAggregationKey('gpt-4')).toBe(true);
    expect(isModelAggregationKey('llama-3')).toBe(true);
  });

  it('filterModelEntries strips non-model entries', () => {
    const entries: [string, any][] = [
      ['gpt-4', { acc: 1 }],
      ['_trustvar', { tsi: 0 }],
      ['_trustvar_error', 'msg'],
      ['llama-3', { acc: 0.5 }],
    ];
    expect(filterModelEntries(entries)).toEqual([
      ['gpt-4', { acc: 1 }],
      ['llama-3', { acc: 0.5 }],
    ]);
  });
});

describe('getGenerationEarSignature (§4.2)', () => {
  it('returns note for generation only', () => {
    expect(getGenerationEarSignature('generation')).toBeTypeOf('string');
    expect(getGenerationEarSignature('generation')).toContain('N/A');
    expect(getGenerationEarSignature('open_qa')).toBeNull();
    expect(getGenerationEarSignature(undefined)).toBeNull();
  });
});

describe('isRefused (fix п.12 — canonical "1"/"0" domain)', () => {
  it('treats the canonical stored value "1" as refused', () => {
    // The real bug: Leaderboard used `=== "yes"`, so "1" (the stored value) read
    // as not-refused and the refusal rate was always 0.
    expect(isRefused('1')).toBe(true);
    expect(isRefused('0')).toBe(false);
  });

  it('is defensive to custom RTA prompts emitting yes/true/boolean', () => {
    expect(isRefused('yes')).toBe(true);
    expect(isRefused('YES')).toBe(true);
    expect(isRefused(' true ')).toBe(true);
    expect(isRefused(true)).toBe(true);
    expect(isRefused(1)).toBe(true);
    expect(isRefused('no')).toBe(false);
    expect(isRefused('false')).toBe(false);
    expect(isRefused(false)).toBe(false);
    expect(isRefused(0)).toBe(false);
  });

  it('treats null/undefined (RTA not run) as not refused', () => {
    expect(isRefused(null)).toBe(false);
    expect(isRefused(undefined)).toBe(false);
    expect(isRefused('')).toBe(false);
  });
});

describe('categorizeValidatorVerdict (fix п.15 — ValidationStatus mapping)', () => {
  it('maps each cascade verdict prefix to a display category', () => {
    expect(categorizeValidatorVerdict('accept')).toBe('ACCEPT');
    expect(categorizeValidatorVerdict('reject_lexical')).toBe('REJECT');
    expect(categorizeValidatorVerdict('reject_semantic')).toBe('REJECT');
    expect(categorizeValidatorVerdict('reject_logic')).toBe('REJECT');
    expect(categorizeValidatorVerdict('reject_lineage')).toBe('REJECT');
    expect(categorizeValidatorVerdict('flag_disagreement')).toBe('FLAG');
    expect(categorizeValidatorVerdict('flag_marginal')).toBe('FLAG');
    expect(categorizeValidatorVerdict('bypassed')).toBe('BYPASSED');
  });

  it('null / unknown verdict → UNKNOWN (caller renders nothing)', () => {
    expect(categorizeValidatorVerdict(null)).toBe('UNKNOWN');
    expect(categorizeValidatorVerdict(undefined)).toBe('UNKNOWN');
    expect(categorizeValidatorVerdict('something_else')).toBe('UNKNOWN');
  });
});
