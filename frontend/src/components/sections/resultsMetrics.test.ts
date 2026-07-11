import { describe, it, expect } from 'vitest';
import type { TrustVarMetrics } from '@/api/types';
import {
  calculateCV,
  calculateIQRCV,
  buildMetricsRows,
  getTierFromVariationType,
  calculateAccuracy,
  buildTaskTierRadar,
  buildModelCvStarRadar,
  buildImpactHeatmap,
} from './resultsMetrics';

// Minimal TrustVarMetrics factory — only the fields buildMetricsRows reads.
function makeMetrics(partial: Partial<TrustVarMetrics>): TrustVarMetrics {
  return {
    per_task_tsi: {},
    per_task_ear: {},
    per_task_cv: {},
    per_task_iqr_cv: {},
    per_task_uninformative: {},
    model_cv_star: {},
    aggregate_tsi: {},
    aggregate_ear: {},
    variance_decomposition: {},
    tier_comparison: {},
    bootstrap_replicates: { tsi: {}, ear: {} },
    n_models: 0,
    n_resamples: 0,
    ci_level: 0.95,
    ...partial,
  };
}

describe('calculateIQRCV (fix 1.2 — median denominator, not midhinge)', () => {
  it('returns NaN for fewer than 2 values', () => {
    expect(calculateIQRCV([])).toBeNaN();
    expect(calculateIQRCV([5])).toBeNaN();
  });

  it('returns 0 when the median is 0 and all values are identical', () => {
    expect(calculateIQRCV([0, 0, 0, 0])).toBe(0);
  });

  it('divides IQR by the MEDIAN, not the midhinge (Q1+Q3)/2', () => {
    const values = [1, 2, 3, 4, 100];
    const median = 3;
    const midhinge = (2 + 4) / 2;
    expect(calculateIQRCV(values)).toBeCloseTo(((4 - 2) / median) * 100, 6);


    const skewed = [1, 1, 1, 9, 9];
    const viaMedian = ((9 - 1) / 1) * 100; // 800
    const viaMidhinge = ((9 - 1) / 5) * 100; // 160
    expect(viaMedian).not.toBeCloseTo(viaMidhinge, 6);
    expect(calculateIQRCV(skewed)).toBeCloseTo(viaMedian, 6);
    expect(midhinge).toBe(3); // sanity for the first sample
  });
});

describe('calculateCV', () => {
  it('returns NaN for fewer than 2 values or zero-mean non-identical', () => {
    expect(calculateCV([7])).toBeNaN();
  });

  it('returns 0 when all values are identical and zero', () => {
    expect(calculateCV([0, 0])).toBe(0);
  });

  it('computes std/mean * 100', () => {
    expect(calculateCV([40, 60])).toBeCloseTo(20, 6);
  });
});

describe('getTierFromVariationType', () => {
  it('maps strategy ids to tiers and null for empty', () => {
    expect(getTierFromVariationType('format_normalization')).toBe('A');
    expect(getTierFromVariationType('active_passive_voice')).toBe('B');
    expect(getTierFromVariationType('paraphrase_free')).toBe('C');
    expect(getTierFromVariationType(null)).toBeNull();
    expect(getTierFromVariationType(undefined)).toBeNull();
  });

  it('covers the full canonical 5A/6B/8C taxonomy (registry.py)', () => {
    for (const s of [
      'format_normalization', 'orthographic_normalization_ru', 'mcq_option_permutation',
      'list_reordering', 'typed_parametric_substitution',
    ]) expect(getTierFromVariationType(s)).toBe('A');
    for (const s of [
      'active_passive_voice', 'monosemic_synonym_substitution', 'nominalisation',
      'controlled_syntactic_transformations', 'sentence_split_merge',
      'controlled_descriptive_modifier_insertion',
    ]) expect(getTierFromVariationType(s)).toBe('B');
    for (const s of [
      'paraphrase_lexico_syntactic_constrained', 'paraphrase_free', 'length_variation',
      'register_formal_informal', 'tone_shift', 'negation_scope_preserving_rephrasing',
      'wsd_synonym_substitution', 'back_translation_single_pivot',
    ]) expect(getTierFromVariationType(s)).toBe('C');
  });

  it('fix п.14: unknown / non-taxonomy strategy is null, NOT a default "C"', () => {
    expect(getTierFromVariationType('some_new_operator')).toBeNull();
    expect(getTierFromVariationType('original')).toBeNull();
    expect(getTierFromVariationType('typo_paraphrase')).toBeNull();
  });
});

describe('buildMetricsRows (O.2 fix — keyed by per_task prompt text)', () => {
  it('returns [] for undefined metrics', () => {
    expect(buildMetricsRows(undefined)).toEqual([]);
  });

  it('returns [] when there are no per_task entries', () => {
    expect(buildMetricsRows(makeMetrics({}))).toEqual([]);
  });

  it('builds one row per per_task key (the benchmark prompt), NOT per job name', () => {
    const prompt = 'What is the capital of France?';
    const metrics = makeMetrics({
      per_task_tsi: { [prompt]: { A: 12.5, B: 8.0 } },
      per_task_ear: { [prompt]: { A: 0.9 } },
      per_task_cv: { [prompt]: { A: 11.0 } },
      per_task_iqr_cv: { [prompt]: { A: 10.0, B: 7.0, C: 5.0 } },
    });

    const rows = buildMetricsRows(metrics);
    expect(rows).toHaveLength(1);
    expect(rows[0].key).toBe(prompt);
    expect(rows[0].label).toContain(prompt);
    expect(rows[0].label.startsWith('#1')).toBe(true);
    expect(rows[0].tsi.A).toBe(12.5);
    expect(rows[0].ear.A).toBe(0.9);
    expect(rows[0].iqr.C).toBe(5.0);
  });

  it('keys come from the union of all per_task_* dicts', () => {
    const metrics = makeMetrics({
      per_task_tsi: { taskOnlyInTsi: { A: 1 } },
      per_task_ear: { taskOnlyInEar: { A: 0.5 } },
    });
    const keys = buildMetricsRows(metrics).map((r) => r.key).sort();
    expect(keys).toEqual(['taskOnlyInEar', 'taskOnlyInTsi']);
  });

  it('suppresses TSI (null + flagged) when per_task_uninformative is set, keeping EAR/CV/IQR', () => {
    const k = 'fragile task';
    const metrics = makeMetrics({
      per_task_tsi: { [k]: { A: 99.0 } },
      per_task_ear: { [k]: { A: 0.8 } },
      per_task_iqr_cv: { [k]: { A: 15.0 } },
      per_task_uninformative: { [k]: { A: true } },
    });
    const [row] = buildMetricsRows(metrics);
    expect(row.tsi.A).toBeNull();
    expect(row.tsiSuppressed.A).toBe(true);
    expect(row.ear.A).toBe(0.8);
    expect(row.iqr.A).toBe(15.0);
  });

  it('fills missing tiers with null', () => {
    const k = 'partial';
    const [row] = buildMetricsRows(makeMetrics({ per_task_tsi: { [k]: { A: 5 } } }));
    expect(row.tsi.A).toBe(5);
    expect(row.tsi.B).toBeNull();
    expect(row.tsi.C).toBeNull();
  });

  it('sorts rows by peak TSI across tiers, descending', () => {
    const metrics = makeMetrics({
      per_task_tsi: {
        low: { A: 1 },
        high: { A: 50, B: 2 },
        mid: { C: 20 },
      },
    });
    const order = buildMetricsRows(metrics).map((r) => r.key);
    expect(order).toEqual(['high', 'mid', 'low']);
  });
});

describe('buildTaskTierRadar (variant A — server per-item TSI by tier)', () => {
  it('returns empty for undefined metrics', () => {
    expect(buildTaskTierRadar(undefined)).toEqual({ points: [], total: 0, shown: 0 });
  });

  it('one point per item, tiers as series, sorted by peak TSI; missing tier → null', () => {
    const metrics = makeMetrics({
      per_task_tsi: { low: { A: 1 }, high: { A: 50, B: 2 }, mid: { C: 20 } },
    });
    const { points, total, shown } = buildTaskTierRadar(metrics);
    expect(total).toBe(3);
    expect(shown).toBe(3);
    expect(points.map((p) => p.fullKey)).toEqual(['high', 'mid', 'low']);
    expect(points[0].item).toBe('#1');
    expect(points[0].A).toBe(50);
    expect(points[0].B).toBe(2);
    expect(points[0].C).toBeNull();
  });

  it('respects topN and reports total vs shown (no silent cap)', () => {
    const per_task_tsi: Record<string, Record<string, number>> = {};
    for (let i = 0; i < 10; i++) per_task_tsi[`t${i}`] = { A: i };
    const { total, shown, points } = buildTaskTierRadar(makeMetrics({ per_task_tsi }), 4);
    expect(total).toBe(10);
    expect(shown).toBe(4);
    expect(points).toHaveLength(4);
    expect(points[0].A).toBe(9);
  });

  it('suppressed TSI (uninformative) → null', () => {
    const metrics = makeMetrics({
      per_task_tsi: { k: { A: 99 } },
      per_task_uninformative: { k: { A: true } },
    });
    expect(buildTaskTierRadar(metrics).points[0].A).toBeNull();
  });
});

describe('buildModelCvStarRadar (variant A — server model_cv_star by tier)', () => {
  it('reads model_cv_star[modelId][tier]; one series per task; 3 tier points', () => {
    const m1 = makeMetrics({ model_cv_star: { mdl: { A: 5, B: 10, C: 15 } } });
    const m2 = makeMetrics({ model_cv_star: { mdl: { A: 1, B: 2, C: 3 } } });
    const { points, taskNames, anyData } = buildModelCvStarRadar(
      [{ taskName: 'T1', metrics: m1 }, { taskName: 'T2', metrics: m2 }],
      'mdl',
    );
    expect(anyData).toBe(true);
    expect(taskNames).toEqual(['T1', 'T2']);
    expect(points).toHaveLength(3);
    const tierA = points.find((p) => p.tier === 'Tier A')!;
    expect(tierA.T1).toBe(5);
    expect(tierA.T2).toBe(1);
  });

  it('NaN / missing model / missing metrics → null; anyData=false when nothing finite', () => {
    const { points, anyData } = buildModelCvStarRadar(
      [
        { taskName: 'T1', metrics: makeMetrics({ model_cv_star: { mdl: { A: NaN } } }) },
        { taskName: 'T2', metrics: undefined },
        { taskName: 'T3', metrics: makeMetrics({ model_cv_star: { other: { A: 9 } } }) },
      ],
      'mdl',
    );
    expect(anyData).toBe(false);
    const tierA = points.find((p) => p.tier === 'Tier A')!;
    expect(tierA.T1).toBeNull();
    expect(tierA.T2).toBeNull();
    expect(tierA.T3).toBeNull();
  });
});

describe('buildImpactHeatmap (model × variation matrix)', () => {
  const models = [{ id: 'm1', name: 'Alpha' }, { id: 'm2', name: 'Beta' }];

  it('builds the matrix with original first, judge domain, gaps as null (not 0)', () => {
    const results = [
      { model_id: 'm1', variation_type: null, judge_score: 5 },
      { model_id: 'm1', variation_type: 'format_normalization', judge_score: 3 },
      { model_id: 'm2', variation_type: 'format_normalization', judge_score: 1 },
    ];
    const h = buildImpactHeatmap(results, ['m1', 'm2'], models);
    expect(h.hasJudge).toBe(true);
    expect(h.domain).toEqual([1, 5]);
    expect(h.variations).toEqual(['original', 'format_normalization']);
    const alpha = h.rows.find((r) => r.model === 'Alpha')!;
    expect(alpha.cells).toEqual([5, 3]);
    const beta = h.rows.find((r) => r.model === 'Beta')!;
    expect(beta.cells[0]).toBeNull();
    expect(beta.cells[1]).toBe(1);
  });

  it('ignores models outside modelIds and resolves ids to names', () => {
    const results = [
      { model_id: 'm1', variation_type: 'a', judge_score: 4 },
      { model_id: 'other', variation_type: 'a', judge_score: 2 },
    ];
    const h = buildImpactHeatmap(results, ['m1'], models);
    expect(h.rows.map((r) => r.model)).toEqual(['Alpha']);
    expect(h.rows[0].cells).toEqual([4]);
  });

  it('uses the accuracy [0,100] domain when there are no judge scores', () => {
    const results = [
      {
        model_id: 'm1',
        variation_type: 'a',
        target: 'A',
        output: 'A',
        metadata: { task_type: 'mcq', option_labels: ['A', 'B'] },
      },
    ];
    const h = buildImpactHeatmap(results, ['m1'], models);
    expect(h.hasJudge).toBe(false);
    expect(h.domain).toEqual([0, 100]);
    expect(h.rows[0].cells[0]).toBe(100);
  });
});

describe('calculateAccuracy (F1 — multi-label MCQ extraction mirrors backend)', () => {
  it('counts a multi-label MCQ answer correct when all labels are present', () => {
    const r = {
      target: '123',
      output: '1, 2, 3',
      metadata: {
        task_type: 'mcq',
        task_semantics: 'multi_label_classification',
        option_labels: ['1', '2', '3', '4', '5'],
      },
    };
    expect(calculateAccuracy([r])).toBe(100);
  });

  it('regression: single-answer MCQ keeps the last matched label', () => {
    const r = {
      target: 'C',
      output: '...Answer: C) 24',
      metadata: { task_type: 'mcq', option_labels: ['A', 'B', 'C', 'D'] },
    };
    expect(calculateAccuracy([r])).toBe(100);
  });
});
