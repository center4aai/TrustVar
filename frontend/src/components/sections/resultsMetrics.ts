import type { TrustVarMetrics } from '@/api/types';

export const TIER_COLORS: Record<string, string> = {
  A: '#3b82f6',
  B: '#8b5cf6',
  C: '#f97316',
};

export const TIER_ORDER = ['A', 'B', 'C'];

export function getTierFromVariationType(
  variationType: string | null | undefined,
): string | null {
  if (!variationType) return null;

  const tierAMatch = /^(format_normalization|orthographic_normalization_ru|mcq_option_permutation|list_reordering|typed_parametric_substitution)/;
  const tierBMatch = /^(active_passive_voice|monosemic_synonym_substitution|nominalisation|controlled_syntactic_transformations|sentence_split_merge|controlled_descriptive_modifier_insertion)/;
  const tierCMatch = /^(paraphrase_lexico_syntactic_constrained|paraphrase_free|length_variation|register_formal_informal|tone_shift|negation_scope_preserving_rephrasing|wsd_synonym_substitution|back_translation_single_pivot)/;
  if (tierAMatch.test(variationType)) return 'A';
  if (tierBMatch.test(variationType)) return 'B';
  if (tierCMatch.test(variationType)) return 'C';
  return null;
}


export function calculateCV(values: number[]): number {
  if (values.length < 2) return NaN;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length,
  );
  if (mean === 0) return std === 0 ? 0 : NaN;
  return (std / mean) * 100;
}

export function calculateIQRCV(values: number[]): number {
  if (values.length < 2) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const mid = sorted[Math.floor(sorted.length / 2)];
  const iqr = q3 - q1;
  if (mid === 0) return iqr === 0 ? 0 : NaN;
  return (iqr / mid) * 100;
}

const COMMON_SUFFIXES = [
  'ment', 'tion', 'sion', 'ing', 'ed', 'ly', 'ness', 'ity', 'ive', 'al',
  'ic', 'ism', 'ist', 'able', 'ible', 'ful', 'less', 'ous', 'er', 'est',
  'ize', 'ise', 'ify', 'en', 'ate', 'ion', 'or',
];

function stemWord(word: string): string {
  const w = word.trim().toLowerCase();
  for (const suffix of [...COMMON_SUFFIXES].sort((a, b) => b.length - a.length)) {
    if (w.endsWith(suffix) && w.length - suffix.length >= 3) {
      return word.slice(0, -suffix.length);
    }
  }
  return word;
}

function extractAnswerLabels(
  output: string,
  optionLabels: string[],
  taskType: string,
  options?: string[],
  multiLabel: boolean = false,
  allowStemFallback: boolean = false,
): string {
  if (!optionLabels.length || !output) return output.trim();

  const escaped = [...optionLabels]
    .map((lbl) => lbl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .sort((a, b) => b.length - a.length);
  const pattern = new RegExp('\\b(?:' + escaped.join('|') + ')\\b', 'gi');
  const found = output.match(pattern) || [];

  if (!found.length && options && options.length === optionLabels.length) {
    const escapedVals = options
      .map((v) => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .sort((a, b) => b.length - a.length);
    const valPattern = new RegExp('\\b(?:' + escapedVals.join('|') + ')\\b', 'g');
    const foundVals = output.match(valPattern) || [];
    const mapped: string[] = [];
    for (const val of foundVals) {
      const idx = options.findIndex((o) => String(o).trim() === val.trim());
      if (idx !== -1) mapped.push(optionLabels[idx]);
      else mapped.push(val);
    }

    return !multiLabel
      ? (mapped[mapped.length - 1] || '').trim()
      : [...new Set(mapped)].sort().join('');
  }

  if (!found.length && allowStemFallback && output) {
    const allAlpha = optionLabels.every(
      (lbl) => /^[a-zA-Z]+$/.test(String(lbl)) && String(lbl).length >= 3,
    );
    if (allAlpha) {
      const stemToLabel: Record<string, string> = {};
      for (const lbl of optionLabels) {
        const s = stemWord(String(lbl)).toLowerCase();
        if (!stemToLabel[s]) stemToLabel[s] = String(lbl);
      }
      const stems = [...new Set(Object.keys(stemToLabel))]
        .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .sort((a, b) => b.length - a.length);
      const stemPattern = new RegExp('\\b(?:' + stems.join('|') + ')\\w*\\b', 'gi');
      const stemFound = output.match(stemPattern);
      if (stemFound?.length) {
        const mapped: string[] = [];
        for (const w of stemFound) {
          const wl = w.toLowerCase();
          let matched = w;
          for (const [stem, label] of Object.entries(stemToLabel).sort(([, a], [, b]) => b.length - a.length)) {
            if (wl.startsWith(stem)) { matched = label; break; }
          }
          mapped.push(matched);
        }
        if (!multiLabel) return mapped[mapped.length - 1].trim();
        return [...new Set(mapped.map((f) => f.trim()))].sort().join('');
      }
    }
  }

  if (!found.length) return output.trim();
  if (!multiLabel) return found[found.length - 1].trim();
  return [...new Set(found.map((f) => f.trim()))].sort().join('');
}

function getOptionLabels(result: any): string[] | undefined {
  if (result.metadata?.option_labels?.length) return result.metadata.option_labels;
  const classes = result.metadata?.classes;
  if (classes && typeof classes === 'object' && !Array.isArray(classes)) {
    const vals = Object.values(classes) as string[];
    if (vals.length) return vals.sort();
  }
  return undefined;
}

function getOptions(result: any): string[] | undefined {
  if (result.metadata?.options?.length) return result.metadata.options;
  const classes = result.metadata?.classes;
  if (classes && typeof classes === 'object' && !Array.isArray(classes)) {
    return Object.keys(classes).sort();
  }
  return undefined;
}

function normalizeTarget(result: any): string {
  const target = String(result.target).trim();
  const classes = result.metadata?.classes;
  if (classes && typeof classes === 'object' && !Array.isArray(classes)) {
    for (const [k, v] of Object.entries(classes)) {
      if (String(k).trim() === target) return String(v).trim();
    }
  }
  return target;
}

function isMultiLabel(result: any): boolean {
  const semantics = result.metadata?.task_semantics;
  return typeof semantics === 'string' && semantics.toLowerCase().includes('multi_label');
}


export function calculateModelMetric(results: any[]): number {
  if (!results.length) return NaN;
  const tt = results[0]?.metadata?.task_type;
  const hasJudge = results.some(r => r.judge_score != null);
  if (hasJudge && (tt === 'open_qa' || tt === 'generation' || !tt)) {
    const scores = results.map(r => r.judge_score).filter(s => s != null);
    if (!scores.length) return NaN;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  }
  return calculateAccuracy(results);
}

export function calculateAccuracy(results: any[]): number {
  if (!results.length) return 0;
  let correct = 0;
  let total = 0;
  for (const r of results) {
    if (!r.target) continue;
    const tt = r.metadata?.task_type;
    if (tt === 'mcq' || tt === 'classification') {
      const target = normalizeTarget(r);
      const output = (r.output || '').trim();
      const optionLabels = getOptionLabels(r);
      const options = getOptions(r);
      if (optionLabels && optionLabels.length) {
        const extracted = extractAnswerLabels(output, optionLabels, tt, options, isMultiLabel(r), tt === 'classification');
        if (target.toLowerCase() === extracted.toLowerCase()) correct++;
      } else {
        if (target.toLowerCase() === output.toLowerCase()) correct++;
      }
      total++;
    } else if (tt === 'open_qa' || tt === 'generation') {
      if (r.judge_score != null) {
        if (r.judge_score >= 4) correct++;
        total++;
      }
    } else {
      if (String(r.target).trim().toLowerCase() === (r.output || '').trim().toLowerCase()) correct++;
      total++;
    }
  }
  if (total === 0) return NaN;
  return (correct / total) * 100;
}

// ── MetricsTable row construction (server-driven) ───────────────────────────

const LABEL_MAX_CHARS = 60;

export interface MetricsRow {

  key: string;

  label: string;
  cv: Record<string, number | null>;
  iqr: Record<string, number | null>;

  tsi: Record<string, number | null>;
  tsiSuppressed: Record<string, boolean>;
  ear: Record<string, number | null>;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function pickTiers(
  m: Record<string, Record<string, number>> | undefined,
  key: string,
): Record<string, number | null> {
  const tierVals = m?.[key] ?? {};
  const out: Record<string, number | null> = {};
  for (const t of TIER_ORDER) out[t] = tierVals[t] ?? null;
  return out;
}

function maxTsi(row: MetricsRow): number {
  const vals = TIER_ORDER.map((t) => row.tsi[t]).filter(
    (v): v is number => v != null,
  );
  return vals.length ? Math.max(...vals) : -Infinity;
}


export function buildMetricsRows(metrics?: TrustVarMetrics): MetricsRow[] {
  if (!metrics) return [];

  const tsiMap = metrics.per_task_tsi ?? {};
  const earMap = metrics.per_task_ear ?? {};
  const cvMap = metrics.per_task_cv ?? {};
  const iqrMap = metrics.per_task_iqr_cv ?? {};
  const uninfMap = metrics.per_task_uninformative ?? {};

  const keys = Array.from(
    new Set([
      ...Object.keys(tsiMap),
      ...Object.keys(earMap),
      ...Object.keys(cvMap),
      ...Object.keys(iqrMap),
    ]),
  );

  const rows: MetricsRow[] = keys.map((key) => {
    const uninf = uninfMap[key] ?? {};
    const rawTsi = pickTiers(tsiMap, key);
    const tsi: Record<string, number | null> = {};
    const tsiSuppressed: Record<string, boolean> = {};
    for (const t of TIER_ORDER) {
      tsiSuppressed[t] = !!uninf[t];
      tsi[t] = uninf[t] ? null : rawTsi[t];
    }
    return {
      key,
      label: key,
      cv: pickTiers(cvMap, key),
      iqr: pickTiers(iqrMap, key),
      tsi,
      tsiSuppressed,
      ear: pickTiers(earMap, key),
    };
  });

  const sorted = [...rows].sort((a, b) => maxTsi(b) - maxTsi(a));

  return sorted.map((r, i) => ({
    ...r,
    label: `#${i + 1}  ${truncate(r.key, LABEL_MAX_CHARS)}`,
  }));
}


export interface TaskTierRadarPoint {

  item: string;

  fullKey: string;
  A: number | null;
  B: number | null;
  C: number | null;
}

export function buildTaskTierRadar(
  metrics: TrustVarMetrics | undefined,
  topN = 8,
): { points: TaskTierRadarPoint[]; total: number; shown: number } {
  const rows = buildMetricsRows(metrics);
  const total = rows.length;
  const points = rows.slice(0, Math.max(0, topN)).map((r, i) => ({
    item: `#${i + 1}`,
    fullKey: r.key,
    A: r.tsi['A'] ?? null,
    B: r.tsi['B'] ?? null,
    C: r.tsi['C'] ?? null,
  }));
  return { points, total, shown: points.length };
}


export interface ModelCvStarRadar {

  points: Array<Record<string, string | number | null>>;
  taskNames: string[];
  anyData: boolean;
}

export function buildModelCvStarRadar(
  perTask: ReadonlyArray<{ taskName: string; metrics?: TrustVarMetrics }>,
  modelId: string,
): ModelCvStarRadar {
  const taskNames = perTask.map((t) => t.taskName);
  let anyData = false;
  const points = TIER_ORDER.map((tier) => {
    const point: Record<string, string | number | null> = { tier: `Tier ${tier}` };
    for (const { taskName, metrics } of perTask) {
      const raw = metrics?.model_cv_star?.[modelId]?.[tier];
      const val = typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
      if (val != null) anyData = true;
      point[taskName] = val;
    }
    return point;
  });
  return { points, taskNames, anyData };
}


export interface ImpactHeatmap {

  variations: string[];

  rows: { model: string; cells: (number | null)[] }[];
  hasJudge: boolean;

  domain: [number, number];
}

export function buildImpactHeatmap(
  results: any[],
  modelIds: string[],
  models: any[],
): ImpactHeatmap {
  const hasJudge = results.some((r) => r.judge_score != null);
  const domain: [number, number] = hasJudge ? [1, 5] : [0, 100];
  const resolveName = (id: string): string =>
    models.find((m: any) => m.id === id)?.name || `${id.slice(0, 12)}...`;

  const groups = new Map<string, any[]>();
  for (const r of results) {
    if (!modelIds.includes(r.model_id)) continue;
    const k = `${r.model_id}||${r.variation_type ?? 'original'}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(r);
  }

  const perModel: Record<string, Record<string, number | null>> = {};
  const variationSet = new Set<string>();
  for (const [k, group] of groups) {
    const sep = k.indexOf('||');
    const name = resolveName(k.slice(0, sep));
    const varType = k.slice(sep + 2);
    variationSet.add(varType);
    const metric = calculateModelMetric(group);
    (perModel[name] ??= {})[varType] = Number.isNaN(metric) ? null : metric;
  }

  const others = [...variationSet].filter((v) => v !== 'original').sort();
  const variations = variationSet.has('original') ? ['original', ...others] : others;
  const rows = Object.keys(perModel)
    .sort()
    .map((model) => ({
      model,
      cells: variations.map((v) => perModel[model][v] ?? null),
    }));
  return { variations, rows, hasJudge, domain };
}
