import React, { useState, useMemo, useEffect } from 'react';
import { BarChart3, ChevronDown, ChevronRight, Eye, Trophy, ArrowUp, ArrowDown, Minus, GitCompare, Download } from 'lucide-react';
import { useTasks, useModels, useTaskResults, useTaskTrustVarMetrics } from '@/api/hooks';
import { apiClient } from '@/api/client';
import { useAppStore } from '@/stores/useAppStore';
import type { TrustVarMetrics, TierAggregate } from '@/api/types';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, Cell } from 'recharts';
import { TIER_COLORS, TIER_ORDER, buildMetricsRows, buildTaskTierRadar, buildModelCvStarRadar, buildImpactHeatmap, getTierFromVariationType, calculateAccuracy } from './resultsMetrics';
import { getFailedModelInfo, getPartialRunInfo, getVariantYieldInfo, getGenerationEarSignature, isModelAggregationKey, isRefused, categorizeValidatorVerdict } from './resultsHelpers';
import { AlertTriangle } from 'lucide-react';

type SortKey = 'model' | 'accuracy' | 'avg_judge_score' | 'avg_metrics' | 'rta_rate' | 'sample_count';
type SortDir = 'asc' | 'desc';

function computeLeaderboard(results: any[]) {
  const byModel: Record<string, {
    judge_scores: number[];
    rta_refused: number;
    rta_total: number;
    metric_scores: number[];
    sample_count: number;
    results: any[];
  }> = {};

  for (const r of results) {
    const m = r.model_name;
    if (!byModel[m]) {
      byModel[m] = { judge_scores: [], rta_refused: 0, rta_total: 0, metric_scores: [], sample_count: 0, results: [] };
    }
    const entry = byModel[m];
    entry.sample_count++;
    entry.results.push(r);

    if (r.judge_score != null) entry.judge_scores.push(r.judge_score);
    if (r.refused != null) {
      entry.rta_total++;

      if (isRefused(r.refused)) entry.rta_refused++;
    }
    if (r.include_score != null) entry.metric_scores.push(r.include_score);
  }

  return Object.entries(byModel).map(([model, d]) => {
    const avg_judge = d.judge_scores.length ? d.judge_scores.reduce((a, b) => a + b, 0) / d.judge_scores.length : null;
    const avg_metric = d.metric_scores.length ? d.metric_scores.reduce((a, b) => a + b, 0) / d.metric_scores.length : null;
    const rta_rate = d.rta_total > 0 ? d.rta_refused / d.rta_total : null;

    const accRaw = calculateAccuracy(d.results);
    const accuracy = Number.isNaN(accRaw) ? null : accRaw;
    return { model, accuracy, avg_judge, avg_metric, rta_rate, sample_count: d.sample_count };
  });
}


function ScoreBadge({ value, type }: { value: number | null; type: 'judge' | 'metric' | 'rta' | 'accuracy' }) {
  if (value == null) return <span className="text-gray-600 text-xs">—</span>;

  let color = 'text-gray-400';
  if (type === 'rta') {
    color = value > 0.5 ? 'text-red-400' : value > 0.2 ? 'text-yellow-400' : 'text-emerald-400';
  } else if (type === 'accuracy') {

    color = value >= 70 ? 'text-emerald-400' : value >= 40 ? 'text-yellow-400' : 'text-red-400';
  } else if (type === 'judge') {

    color = value >= 4 ? 'text-emerald-400' : value >= 3 ? 'text-yellow-400' : 'text-red-400';
  } else {

    color = value >= 0.8 ? 'text-emerald-400' : value >= 0.5 ? 'text-yellow-400' : 'text-red-400';
  }

  if (type === 'rta') {
    return <span className={`font-semibold tabular-nums ${color}`}>{(value * 100).toFixed(1)}%</span>;
  }
  if (type === 'accuracy') {
    return <span className={`font-semibold tabular-nums ${color}`}>{value.toFixed(1)}%</span>;
  }

  return <span className={`font-semibold tabular-nums ${color}`}>{value.toFixed(type === 'judge' ? 2 : 3)}</span>;
}

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) return <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-yellow-500/20 text-yellow-400 text-xs font-bold">1</span>;
  if (rank === 2) return <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-400/20 text-gray-300 text-xs font-bold">2</span>;
  if (rank === 3) return <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-orange-700/20 text-orange-400 text-xs font-bold">3</span>;
  return <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-gray-800 text-gray-500 text-xs">{rank}</span>;
}

function SortIcon({ col, sortKey, sortDir }: { col: SortKey; sortKey: SortKey; sortDir: SortDir }) {
  if (col !== sortKey) return <Minus size={12} className="text-gray-700" />;
  return sortDir === 'desc' ? <ArrowDown size={12} className="text-violet-400" /> : <ArrowUp size={12} className="text-violet-400" />;
}

function FailedBadge() {
  return (
    <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20">
      FAILED
    </span>
  );
}

function Leaderboard({
  results,
  aggregated_metrics,
  completionSummary,
  models,
}: {
  results: any[];
  aggregated_metrics?: any;
  completionSummary?: any;
  models?: any[];
}) {

  const [sortKey, setSortKey] = useState<SortKey>('accuracy');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const rows = useMemo(() => computeLeaderboard(results), [results]);
  const failedModels = useMemo(
    () => getFailedModelInfo(completionSummary, models || []),
    [completionSummary, models]
  );

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      let av: number | null = null, bv: number | null = null;
      if (sortKey === 'model') return sortDir === 'asc' ? a.model.localeCompare(b.model) : b.model.localeCompare(a.model);
      if (sortKey === 'accuracy') { av = a.accuracy; bv = b.accuracy; }
      if (sortKey === 'avg_judge_score') { av = a.avg_judge; bv = b.avg_judge; }
      if (sortKey === 'avg_metrics') { av = a.avg_metric; bv = b.avg_metric; }
      if (sortKey === 'rta_rate') { av = a.rta_rate; bv = b.rta_rate; }
      if (sortKey === 'sample_count') { av = a.sample_count; bv = b.sample_count; }
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return sortDir === 'desc' ? bv - av : av - bv;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const toggle = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortKey(key); setSortDir('desc'); }
  };


  const aggMetricKeys: string[] = useMemo(() => {
    if (!aggregated_metrics) return [];
    const keys = new Set<string>();
    Object.entries(aggregated_metrics)
      .filter(([k]) => isModelAggregationKey(k))
      .forEach(([, metrics]) => {
        const m = metrics as any;
        if (m && typeof m === 'object' && !Array.isArray(m)) {
          Object.keys(m).filter(k => k !== 'execution_time').forEach(k => keys.add(k));
        }
      });
    return Array.from(keys);
  }, [aggregated_metrics]);

  const headers: { key: SortKey; label: string }[] = [
    { key: 'model', label: 'Model' },
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'avg_judge_score', label: 'Avg Judge Score' },
    { key: 'avg_metrics', label: 'Avg Include Score' },
    { key: 'rta_rate', label: 'Refusal Rate' },
    { key: 'sample_count', label: 'Samples' },
  ];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div className="flex items-center gap-2 mb-5">
        <Trophy size={18} className="text-yellow-500" />
        <h3 className="text-lg font-semibold text-white">Model Leaderboard</h3>
        <span className="ml-auto text-xs text-gray-500">{sorted.length} model{sorted.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-500 w-8">#</th>
              {headers.map(h => (
                <th key={h.key} className="py-2 px-3 text-left">
                  <button
                    onClick={() => toggle(h.key)}
                    className="flex items-center gap-1 text-xs font-medium text-gray-400 hover:text-white transition-colors group"
                  >
                    {h.label}
                    <SortIcon col={h.key} sortKey={sortKey} sortDir={sortDir} />
                  </button>
                </th>
              ))}
              {/* Extra aggregated metric columns */}
              {aggMetricKeys.map(k => (
                <th key={k} className="py-2 px-3 text-left text-xs font-medium text-gray-500 whitespace-nowrap">
                  {k.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, idx) => (
              <tr
                key={row.model}
                className={`border-b border-gray-800/60 transition-colors hover:bg-gray-800/30 ${idx === 0 ? 'bg-yellow-500/5' : ''}`}
              >
                <td className="py-3 px-3">
                  <RankBadge rank={idx + 1} />
                </td>
                <td className="py-3 px-3">
                  <span className="font-medium text-white">{row.model}</span>
                </td>
                <td className="py-3 px-3">
                  <ScoreBadge value={row.accuracy} type="accuracy" />
                </td>
                <td className="py-3 px-3">
                  <ScoreBadge value={row.avg_judge} type="judge" />
                </td>
                <td className="py-3 px-3">
                  <ScoreBadge value={row.avg_metric} type="metric" />
                </td>
                <td className="py-3 px-3">
                  <ScoreBadge value={row.rta_rate} type="rta" />
                </td>
                <td className="py-3 px-3 text-gray-400 tabular-nums">
                  {row.sample_count}
                </td>
                {/* Extra aggregated metric values */}
                {aggMetricKeys.map(k => {
                  const val = aggregated_metrics?.[row.model]?.[k];
                  return (
                    <td key={k} className="py-3 px-3 text-gray-300 tabular-nums text-xs">
                      {val != null ? (typeof val === 'number' ? val.toFixed(3) : val) : <span className="text-gray-700">—</span>}
                    </td>
                  );
                })}
              </tr>
            ))}

            {failedModels.map((failed) => (
              <tr
                key={failed.modelId}
                className="border-b border-gray-800/60 transition-colors hover:bg-gray-800/30 bg-red-500/5"
              >
                <td className="py-3 px-3">
                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-red-500/20 text-red-400 text-xs font-bold">—</span>
                </td>
                <td className="py-3 px-3">
                  <span className="font-medium text-white">{failed.modelName}</span>
                  <FailedBadge />
                </td>
                <td className="py-3 px-3"><span className="text-gray-700">—</span></td>
                <td className="py-3 px-3"><span className="text-gray-700">—</span></td>
                <td className="py-3 px-3"><span className="text-gray-700">—</span></td>
                <td className="py-3 px-3"><span className="text-gray-700">—</span></td>
                <td className="py-3 px-3 text-gray-500 tabular-nums">
                  0{failed.errorCount > 0 ? ` (${failed.errorCount} errors)` : ''}
                </td>
                {aggMetricKeys.map(k => (
                  <td key={k} className="py-3 px-3 text-gray-700 tabular-nums text-xs">—</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function getModelName(modelId: string, models: any[]): string {
  const model = models.find((m: any) => m.id === modelId);
  return model?.name || modelId.slice(0, 12) + '...';
}


function TaskCentricSpiderChart({ trustVarMetrics }: { trustVarMetrics?: TrustVarMetrics }) {
  const { points, total, shown } = useMemo(
    () => buildTaskTierRadar(trustVarMetrics),
    [trustVarMetrics],
  );

  if (points.length < 3) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-2">🕸️ Task Sensitivity (TSI) by Tier</h3>
        <div className="text-center py-8 text-gray-500">
          Need at least 3 benchmark items with server-side TSI to draw this radar.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-1">🕸️ Task Sensitivity (TSI) by Tier</h3>
      <p className="text-sm text-gray-400 mb-1">
        Server-side TSI per benchmark item, split by tier (A/B/C). Larger radius = more sensitive to invariance-preserving rewrites.
      </p>
      {shown < total && (
        <p className="text-xs text-amber-400/80 mb-3">Showing the top {shown} of {total} items by peak TSI.</p>
      )}
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={points}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="item" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <PolarRadiusAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
          {TIER_ORDER.map((tier) => (
            <Radar
              key={tier}
              name={`Tier ${tier}`}
              dataKey={tier}
              stroke={TIER_COLORS[tier]}
              fill={TIER_COLORS[tier]}
              fillOpacity={0.2}
            />
          ))}
          <Legend wrapperStyle={{ color: '#9ca3af' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#f3f4f6' }}
            formatter={(value: any, name: any) => [value == null ? 'n/a' : `${Number(value).toFixed(2)}%`, name]}
            labelFormatter={(label: any) => {
              const p = points.find((pt) => pt.item === label);
              return p ? p.fullKey.slice(0, 80) : label;
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

function AugmentationImpactChart({ results, taskName, modelIds, models }: { results: any[]; taskName: string; modelIds: string[]; models: any[] }) {
  const { variations, rows, hasJudge, domain } = useMemo(
    () => buildImpactHeatmap(results, modelIds, models),
    [results, modelIds, models],
  );
  const metricLabel = hasJudge ? 'Avg Judge Score' : 'Accuracy';
  const [lo, hi] = domain;

  if (!rows.length || !variations.length) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-2">Impact of Variations on Metrics</h3>
        <div className="text-center py-8 text-gray-500">No per-variation results to compare.</div>
      </div>
    );
  }

  const cellColor = (v: number | null): string => {
    if (v == null) return 'transparent';
    const frac = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)));
    return `hsl(${Math.round(120 * frac)} 55% 38%)`; // red (low) → amber → green (high)
  };
  const cellText = (v: number | null): string => (v == null ? '·' : hasJudge ? v.toFixed(1) : v.toFixed(0));
  const cellTip = (v: number | null): string => (v == null ? 'no data' : hasJudge ? v.toFixed(2) : `${v.toFixed(0)}%`);
  const shortVar = (v: string): string => (v.length > 11 ? `${v.slice(0, 10)}…` : v);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-1">Impact of Variations on Metrics</h3>
      <p className="text-sm text-gray-400 mb-1">
        Heatmap of per-model {metricLabel.toLowerCase()} across variation prompts — green = higher, red = lower, empty = no data.
      </p>
      <p className="text-sm text-gray-400 mb-3">{taskName}</p>

      <div className="flex flex-wrap items-center gap-2 mb-3 text-xs text-gray-500">
        <span>{hasJudge ? lo.toFixed(0) : `${lo}%`}</span>
        <span
          className="h-2 w-40 rounded"
          style={{ background: 'linear-gradient(90deg, hsl(0 55% 38%), hsl(60 55% 38%), hsl(120 55% 38%))' }}
        />
        <span>{hasJudge ? hi.toFixed(0) : `${hi}%`}</span>
        <span className="ml-3">tier:</span>
        {TIER_ORDER.map((t) => (
          <span key={t} className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: TIER_COLORS[t] }} />
            {t}
          </span>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="text-xs border-separate" style={{ borderSpacing: '3px' }}>
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-gray-900 px-2 py-1 text-left font-medium text-gray-500">Model</th>
              {variations.map((v) => {
                const tier = getTierFromVariationType(v === 'original' ? null : v);
                return (
                  <th
                    key={v}
                    title={v}
                    className="px-1 py-1 text-center font-medium whitespace-nowrap"
                    style={{ color: tier ? TIER_COLORS[tier] : '#6b7280' }}
                  >
                    {shortVar(v)}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model}>
                <td
                  className="sticky left-0 z-10 bg-gray-900 px-2 py-1 text-gray-300 whitespace-nowrap max-w-[12rem] truncate"
                  title={row.model}
                >
                  {row.model}
                </td>
                {row.cells.map((v, i) => (
                  <td
                    key={i}
                    title={`${row.model} · ${variations[i]}: ${cellTip(v)}`}
                    className="text-center tabular-nums rounded"
                    style={{
                      background: cellColor(v),
                      color: v == null ? '#4b5563' : '#f9fafb',
                      minWidth: '2.4rem',
                      padding: '0.35rem 0.4rem',
                      border: v == null ? '1px dashed #374151' : 'none',
                    }}
                  >
                    {cellText(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RTASpiderChart({ task, models }: { task: any; models: any[] }) {
  const rawVariations: string[] = (task.results || [])
    .map((r: any) => r.variation_type as string)
    .filter((v: string) => Boolean(v));
  const uniqueVariations = Array.from(new Set(rawVariations)).sort();

  const hasOriginal = (task.results || []).some((r: any) => !r.variation_type);
  const variations = hasOriginal ? ['original', ...uniqueVariations] : uniqueVariations;

  if (variations.length < 3) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="text-center py-8 text-gray-500">
          Need at least 3 variations for spider chart
        </div>
      </div>
    );
  }

  const colors = ['#f97316', '#fb923c', '#fdba74', '#fed7aa', '#c2410c', '#9a3412'];

  const traces: { name: string; variation_names: string[]; variation_scores: number[] }[] = [];

  for (const model of models) {
    if (!task.model_ids.includes(model.id)) continue;

    const modelName = getModelName(model.id, models);
    const variationScores: number[] = [];
    const variationNames: string[] = [];

    for (const variation of variations) {
      let varResults: any[];
      if (variation === 'original') {
        varResults = (task.results || []).filter(
          (r: any) => r.model_id === model.id && !r.variation_type
        );
      } else {
        varResults = (task.results || []).filter(
          (r: any) => r.model_id === model.id && r.variation_type === variation
        );
      }

      if (varResults.length > 0) {

        const refusedCount = varResults.filter((r: any) => isRefused(r.refused)).length;
        const refusalRate = (refusedCount / varResults.length) * 100;
        variationScores.push(refusalRate);
        variationNames.push(variation);
      }
    }

    if (variationScores.length > 0) {
      traces.push({ name: modelName, variation_names: variationNames, variation_scores: variationScores });
    }
  }

  if (traces.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="text-center py-8 text-gray-500">No RTA data available</div>
      </div>
    );
  }

  const allVariations = Array.from(new Set(traces.flatMap(t => t.variation_names)));
  const chartData = allVariations.map(variation => {
    const point: Record<string, any> = { variation };
    for (const trace of traces) {
      const idx = trace.variation_names.indexOf(variation);
      point[trace.name] = idx >= 0 ? trace.variation_scores[idx] : null;
    }
    return point;
  });

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-2">🕸️ RTA Rates Across Variations</h3>
      <p className="text-sm text-gray-400 mb-4">Shows refusal rates for each model across different variations</p>
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="variation" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} />
          {traces.map((trace, i) => (
            <Radar
              key={trace.name}
              name={trace.name}
              dataKey={trace.name}
              stroke={colors[i % colors.length]}
              fill={colors[i % colors.length]}
              fillOpacity={0.2}
            />
          ))}
          <Legend wrapperStyle={{ color: '#9ca3af' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#f3f4f6' }}
            formatter={(value: any) => [`${Number(value).toFixed(1)}%`, 'Refusal Rate']}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}


function ModelCentricSpiderChart({ completedTasks, selectedTaskId, selectedTaskNames, models }: { completedTasks: any[]; selectedTaskId: string; selectedTaskNames: string[]; models: any[] }) {
  const [perTask, setPerTask] = useState<Array<{ taskName: string; metrics?: TrustVarMetrics }>>([]);
  const [loading, setLoading] = useState(false);
  const selectedRef = React.useRef('');

  useEffect(() => {
    const key = selectedTaskNames.join(',');
    if (key === selectedRef.current) return;
    selectedRef.current = key;

    let cancelled = false;
    const fetchAll = async () => {
      setLoading(true);
      const rows: Array<{ taskName: string; metrics?: TrustVarMetrics }> = [];
      for (const task of completedTasks) {
        if (!selectedTaskNames.includes(task.name)) continue;
        try {
          const metrics = await apiClient.getTaskTrustVarMetrics(task.id);
          rows.push({ taskName: task.name, metrics });
        } catch {
          rows.push({ taskName: task.name, metrics: undefined });
        }
      }
      if (!cancelled) {
        setPerTask(rows);
        setLoading(false);
      }
    };
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [selectedTaskNames, completedTasks]);

  const modelName = getModelName(selectedTaskId, models);
  const { points, taskNames, anyData } = useMemo(
    () => buildModelCvStarRadar(perTask, selectedTaskId),
    [perTask, selectedTaskId],
  );

  if (!selectedTaskNames.length) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="text-center py-8 text-gray-500">Please select at least one task</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="text-center py-8 text-gray-500">Loading server metrics…</div>
      </div>
    );
  }

  if (!anyData) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="text-center py-8 text-gray-500">
          No server CV* available for {modelName} on the selected task(s).
        </div>
      </div>
    );
  }

  const colors = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#059669', '#047857'];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-1">🕸️ Model Reliability — {modelName}</h3>
      <p className="text-sm text-gray-400 mb-4">
        Server-side CV* (bias-corrected dispersion across variants, at fixed model) by tier — one series per task. Larger radius = less stable.
      </p>
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={points}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="tier" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <PolarRadiusAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
          {taskNames.map((name, i) => (
            <Radar
              key={name}
              name={name}
              dataKey={name}
              stroke={colors[i % colors.length]}
              fill={colors[i % colors.length]}
              fillOpacity={0.2}
            />
          ))}
          <Legend wrapperStyle={{ color: '#9ca3af' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#f3f4f6' }}
            formatter={(value: any, name: any) => [value == null ? 'n/a' : `${Number(value).toFixed(2)}%`, name]}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

const SUPPRESS_TITLE = 'Suppressed: >5% of bootstrap replicates dropped (undefined CV)';

function TsiCell({ value, suppressed, color }: { value: number | null; suppressed: boolean; color: string }) {
  if (suppressed) return <span className="text-gray-500 text-xs" title={SUPPRESS_TITLE}>†</span>;
  if (value == null) return <span className="text-gray-600">—</span>;
  return <span className={color}>{value.toFixed(2)}</span>;
}


export function MetricsTable({
  trustVarMetrics,
  canonicalTaskType,
}: {
  trustVarMetrics?: TrustVarMetrics;
  canonicalTaskType?: string;
}) {
  const rows = useMemo(() => buildMetricsRows(trustVarMetrics), [trustVarMetrics]);

  if (!rows.length) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-2">Task Metrics Table</h3>
        <p className="text-sm text-gray-500">No task-level TrustVar metrics available for this task.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-1">Task Metrics Table</h3>
      <p className="text-xs text-gray-500 mb-3">
        Server-side TrustVar metrics, one row per benchmark task • TSI averaged over the model consilium • sorted by peak TSI
      </p>
      {(() => {
        const signature = getGenerationEarSignature(canonicalTaskType);
        if (!signature) return null;
        return (
          <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90">
            <span className="font-medium text-amber-100">EAR is N/A for this task type.</span>{' '}
            {signature}
          </div>
        );
      })()}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-500">Task</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-gray-500">IQR-CV % (A/B/C)</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-blue-400">TSI-A (%)</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-purple-400">TSI-B (%)</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-orange-400">TSI-C (%)</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-blue-400">EAR-A</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-purple-400">EAR-B</th>
              <th className="py-2 px-3 text-left text-xs font-medium text-orange-400">EAR-C</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.key} className="border-b border-gray-800/60">
                <td className="py-3 px-3 text-white max-w-md truncate" title={row.key}>{row.label}</td>
                <td className="py-3 px-3 text-gray-300 tabular-nums">

                  {(row.iqr['A'] == null && row.iqr['B'] == null && row.iqr['C'] == null)
                    ? '—'
                    : `${row.iqr['A']?.toFixed(1) ?? '—'} / ${row.iqr['B']?.toFixed(1) ?? '—'} / ${row.iqr['C']?.toFixed(1) ?? '—'}`}
                </td>
                <td className="py-3 px-3 tabular-nums"><TsiCell value={row.tsi['A']} suppressed={row.tsiSuppressed['A']} color="text-blue-400" /></td>
                <td className="py-3 px-3 tabular-nums"><TsiCell value={row.tsi['B']} suppressed={row.tsiSuppressed['B']} color="text-purple-400" /></td>
                <td className="py-3 px-3 tabular-nums"><TsiCell value={row.tsi['C']} suppressed={row.tsiSuppressed['C']} color="text-orange-400" /></td>
                <td className="py-3 px-3 tabular-nums">{row.ear['A'] != null ? <span className="text-emerald-400">{row.ear['A']!.toFixed(2)}</span> : <span className="text-gray-600">—</span>}</td>
                <td className="py-3 px-3 tabular-nums">{row.ear['B'] != null ? <span className="text-emerald-400">{row.ear['B']!.toFixed(2)}</span> : <span className="text-gray-600">—</span>}</td>
                <td className="py-3 px-3 tabular-nums">{row.ear['C'] != null ? <span className="text-emerald-400">{row.ear['C']!.toFixed(2)}</span> : <span className="text-gray-600">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}



function SubsamplingStabilityChart({ trustVarMetrics, canonicalTaskType }: { trustVarMetrics?: TrustVarMetrics; canonicalTaskType?: string }) {
  if (!trustVarMetrics || !trustVarMetrics.aggregate_tsi || Object.keys(trustVarMetrics.aggregate_tsi).length === 0) {
    return null;
  }

  const tiers = ['A', 'B', 'C'];
  const aggregateTsi = trustVarMetrics.aggregate_tsi ?? {};

  const aggregateEar = trustVarMetrics.aggregate_ear ?? {};
  const earSignature = getGenerationEarSignature(canonicalTaskType);

  const tsiData = tiers.filter(t => aggregateTsi[t] && aggregateTsi[t].n_tasks > 0).map(t => ({
    tier: `Tier ${t}`,
    metric: 'TSI',
    mean: aggregateTsi[t].mean,
    ci_low: aggregateTsi[t].ci_low,
    ci_high: aggregateTsi[t].ci_high,
    fill: TIER_COLORS[t],
  }));
  const earData = earSignature ? [] : tiers.filter(t => aggregateEar[t] && aggregateEar[t].n_tasks > 0).map(t => ({
    tier: `Tier ${t}`,
    metric: 'EAR',
    mean: aggregateEar[t].mean,
    ci_low: aggregateEar[t].ci_low,
    ci_high: aggregateEar[t].ci_high,
    fill: TIER_COLORS[t],
  }));

  if (!tsiData.length && !earData.length && !earSignature) return null;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-2">Bootstrap CI <span className="text-xs font-normal text-yellow-500">(provisional — §F aggregation rule not yet locked)</span></h3>
      <p className="text-sm text-gray-400 mb-4">
        Model-resampled BCa 95% CI, full N={trustVarMetrics.n_models} models, {trustVarMetrics.n_resamples} replicates
      </p>
      <div className="grid grid-cols-2 gap-6">
        {/* TSI by Tier */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">TSI by Tier</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={tsiData} layout="vertical" margin={{ left: 40 }}>
              <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis type="category" dataKey="tier" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                formatter={(value: any, _: any, props: any) => [
                  `${Number(value).toFixed(2)} [${Number(props.payload.ci_low).toFixed(2)}, ${Number(props.payload.ci_high).toFixed(2)}]`,
                  props.payload.tier
                ]}
              />
              <Bar dataKey="mean" fill="#8b5cf6">
                {tsiData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-500 mt-1">
            {tsiData.map(d => (
              <span key={d.tier} className="mr-4">
                {d.tier}: {d.mean.toFixed(2)} [{d.ci_low.toFixed(2)}, {d.ci_high.toFixed(2)}]
              </span>
            ))}
          </div>
        </div>
        {/* EAR Stability */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">EAR by Tier</h4>

          {earSignature ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90">
              <span className="font-medium text-amber-100">EAR is N/A for this task type.</span>{' '}
              {earSignature}
            </div>
          ) : earData.length ? (
            <>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={earData} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} domain={[0, 1]} />
                  <YAxis type="category" dataKey="tier" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                    formatter={(value: any, _: any, props: any) => [
                      `${Number(value).toFixed(3)} [${Number(props.payload.ci_low).toFixed(3)}, ${Number(props.payload.ci_high).toFixed(3)}]`,
                      props.payload.tier
                    ]}
                  />
                  <Bar dataKey="mean" fill="#8b5cf6">
                    {earData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-500 mt-1">
                {earData.map(d => (
                  <span key={d.tier} className="mr-4">
                    {d.tier}: {d.mean.toFixed(3)} [{d.ci_low.toFixed(3)}, {d.ci_high.toFixed(3)}]
                  </span>
                ))}
              </div>
            </>
          ) : (
            <p className="text-xs text-gray-500">No EAR data recorded for this run.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ResultsSection() {
  const { data: tasks = [] } = useTasks();
  const { data: models = [] } = useModels();
  const storeSelectedTaskId = useAppStore(s => s.selectedTaskId);
  const clearSelectedTaskId = useAppStore(s => s.setSelectedTaskId);
  const completedTasks = tasks.filter(t => t.status === 'completed');
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedResultIndex, setSelectedResultIndex] = useState<number | null>(null);
  const [showSpiderCharts, setShowSpiderCharts] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [selectedTaskNames, setSelectedTaskNames] = useState<string[]>([]);

  useEffect(() => {
    if (storeSelectedTaskId) {
      setSelectedTaskId(storeSelectedTaskId);
      clearSelectedTaskId(null);
    }
  }, [storeSelectedTaskId, clearSelectedTaskId]);

  const selectedTask = completedTasks.find(t => t.id === selectedTaskId);
  const { data: taskResultsData } = useTaskResults(selectedTaskId);
  const { data: trustVarMetrics } = useTaskTrustVarMetrics(selectedTaskId);


  const selectedTaskWithResults = selectedTask
    ? { ...selectedTask, results: taskResultsData?.results ?? [] }
    : undefined;

  const canonicalTaskType = useMemo(
    () => selectedTaskWithResults?.results?.[0]?.metadata?.task_type as string | undefined,
    [selectedTaskWithResults?.results],
  );

  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = async (taskId: string) => {
    setExportError(null);
    setIsExporting(true);
    try {
      await apiClient.downloadTaskResults(taskId);
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Unknown error';
      setExportError(`Export failed: ${detail}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="p-6 space-y-6 animate-slideIn">
      <div>
        <h1 className="text-2xl font-semibold text-white mb-1">Results & Analytics</h1>
        <p className="text-sm text-gray-400">Analyze model performance and responses</p>
      </div>

      {completedTasks.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <BarChart3 size={48} className="mx-auto text-gray-700 mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No results yet</h3>
          <p className="text-sm text-gray-500">Complete a task to see results</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Task Selector */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Select Task</label>
            <div className="flex items-center gap-3">
              <select
                value={selectedTaskId || ''}
                onChange={(e) => setSelectedTaskId(e.target.value)}
                className="flex-1 max-w-md bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
              >
                <option value="">Choose a task...</option>
                {completedTasks.map(task => (
                  <option key={task.id} value={task.id}>{task.name}</option>
                ))}
              </select>
              {selectedTask && (
                <button
                  onClick={() => handleExport(selectedTask.id)}
                  disabled={isExporting}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-700 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Download size={16} />
                  {isExporting ? 'Exporting…' : 'Export'}
                </button>
              )}
            </div>
            {exportError && (
              <p className="mt-2 text-sm text-red-400">{exportError}</p>
            )}
          </div>

          {selectedTask && (
            <>
              {/* Task Overview */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <h2 className="text-xl font-semibold text-white mb-4">{selectedTask.name}</h2>


                {(() => {
                  const partial = getPartialRunInfo(selectedTask, models);
                  if (!partial.isPartial) return null;
                  return (
                    <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-700/50 bg-amber-900/20 px-4 py-3 text-sm text-amber-300">
                      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                      <span>{partial.message}</span>
                    </div>
                  );
                })()}


                {(() => {
                  const yieldInfo = getVariantYieldInfo(
                    selectedTask.config,
                    selectedTaskWithResults?.results,
                  );
                  if (!yieldInfo.warn) return null;
                  return (
                    <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-700/50 bg-amber-900/20 px-4 py-3 text-sm text-amber-300">
                      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                      <span>{yieldInfo.message}</span>
                    </div>
                  );
                })()}

                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-black/40 rounded-lg p-4">
                    <div className="text-2xl font-semibold text-white">{selectedTask.total_samples}</div>
                    <div className="text-xs text-gray-500">Total Samples</div>
                  </div>
                  <div className="bg-black/40 rounded-lg p-4">
                    <div className="text-2xl font-semibold text-white">{selectedTask.processed_samples}</div>
                    <div className="text-xs text-gray-500">Processed</div>
                  </div>
                  <div className="bg-black/40 rounded-lg p-4">
                    <div className="text-2xl font-semibold text-white">{selectedTask.model_ids.length}</div>
                    <div className="text-xs text-gray-500">Models</div>
                  </div>
                  <div className="bg-black/40 rounded-lg p-4">
                    <div className="text-2xl font-semibold text-violet-400">{selectedTask.task_type}</div>
                    <div className="text-xs text-gray-500">Task Type</div>
                  </div>
                </div>


              </div>

              {/* Leaderboard */}
              {(selectedTaskWithResults?.results?.length ?? 0) > 0 && (
                <Leaderboard
                  results={selectedTaskWithResults!.results!}
                  aggregated_metrics={selectedTaskWithResults!.aggregated_metrics}
                  completionSummary={selectedTask.completion_summary}
                  models={models}
                />
              )}


              {selectedTask.metrics_error && (
                <div className="mb-4 p-4 rounded-lg border border-amber-600 bg-amber-950/40">
                  <div className="text-sm font-medium text-amber-300 mb-2">
                    TrustVar metrics could not be computed for this run
                  </div>
                  <details className="text-xs text-amber-200/80">
                    <summary className="cursor-pointer">Show details</summary>
                    <pre className="mt-2 whitespace-pre-wrap break-words">{selectedTask.metrics_error}</pre>
                  </details>
                </div>
              )}

              {/* Spider Charts Section for Prompt Variations */}
              {selectedTask.config?.variations?.enabled && (selectedTaskWithResults?.results?.length ?? 0) > 0 && (
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                  <button
                    onClick={() => setShowSpiderCharts(!showSpiderCharts)}
                    className="w-full flex items-center justify-between hover:bg-gray-800/50 transition-colors rounded-lg p-2 -mx-2"
                  >
                    <div className="flex items-center gap-3">
                      <GitCompare size={18} className="text-violet-400" />
                      <div className="text-left">
                        <div className="text-lg font-semibold text-white">Spider Charts & Variations</div>
                        <div className="text-sm text-gray-500">
                          {selectedTask.config?.variations?.strategies?.length || 0} variations • Task Stability Analysis
                        </div>
                      </div>
                    </div>
                    {showSpiderCharts ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                  </button>

                  {showSpiderCharts && (
                    <div className="mt-6 space-y-6">
                      {/* Task-Centric Spider Chart */}
                      <TaskCentricSpiderChart trustVarMetrics={trustVarMetrics} />

                      <AugmentationImpactChart
                        results={selectedTaskWithResults?.results ?? []}
                        taskName={selectedTask.name}
                        modelIds={selectedTask.model_ids}
                        models={models}
                      />

                      {/* Task-level TrustVar metrics (per benchmark item, consilium-aggregated) */}
                      <MetricsTable
                        trustVarMetrics={trustVarMetrics}
                        canonicalTaskType={canonicalTaskType}
                      />

                      {/* RTA Spider Chart */}
                      {selectedTask.task_type === 'rta' && (
                        <RTASpiderChart task={selectedTaskWithResults} models={models} />
                      )}

                      {/* Model-Centric Analysis */}
                      <div className="border border-gray-800 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-white mb-4">Model-Centric Analysis</h3>
                        <p className="text-sm text-gray-400 mb-4">Compare tasks for a selected model</p>

                        <div className="mb-4">
                          <label className="block text-xs text-gray-500 mb-2">Select Model</label>
                          <select
                            value={selectedModelId}
                            onChange={(e) => {
                              setSelectedModelId(e.target.value);
                              setSelectedTaskNames([]);
                            }}
                            className="w-full max-w-xs bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                          >
                            <option value="">Choose a model...</option>
                            {models.filter((m: any) => m.status === 'registered').map((model: any) => (
                              <option key={model.id} value={model.id}>{model.name}</option>
                            ))}
                          </select>
                        </div>

                        {selectedModelId && (
                          <>
                            <div className="mb-4">
                              <label className="block text-xs text-gray-500 mb-2">Select Tasks</label>
                              <div className="flex flex-wrap gap-2">
                                {completedTasks.filter(t => t.config?.variations?.enabled).map(task => (
                                  <button
                                    key={task.id}
                                    onClick={() => {
                                      setSelectedTaskNames(prev =>
                                        prev.includes(task.name)
                                          ? prev.filter(n => n !== task.name)
                                          : [...prev, task.name]
                                      );
                                    }}
                                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${selectedTaskNames.includes(task.name)
                                      ? 'bg-violet-600 text-white'
                                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                                      }`}
                                  >
                                    {task.name}
                                  </button>
                                ))}
                              </div>
                            </div>

                            {selectedTaskNames.length > 0 && (
                              <>
                                <ModelCentricSpiderChart
                                  completedTasks={completedTasks.filter(t => t.config?.variations?.enabled)}
                                  selectedTaskId={selectedModelId}
                                  selectedTaskNames={selectedTaskNames}
                                  models={models}
                                />
                                <SubsamplingStabilityChart trustVarMetrics={trustVarMetrics} canonicalTaskType={canonicalTaskType} />
                              </>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Individual Results */}
              {(selectedTaskWithResults?.results?.length ?? 0) > 0 && (
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Individual Responses</h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Click on any result to view detailed responses
                  </p>

                  <div className="space-y-2">
                    {(selectedTaskWithResults?.results ?? []).map((result: any, idx: number) => (
                      <div key={idx} className="border border-gray-800 rounded-lg overflow-hidden">
                        <button
                          onClick={() => setSelectedResultIndex(selectedResultIndex === idx ? null : idx)}
                          className="w-full p-4 bg-gray-800/50 hover:bg-gray-800 flex items-center justify-between transition-colors"
                        >
                          <div className="flex items-center gap-3 flex-1 text-left">
                            <Eye size={16} className="text-gray-500" />
                            <div className="flex-1">
                              <div className="text-sm text-white font-medium truncate">
                                {result.input.slice(0, 80)}...
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                Model: {result.model_name} • Time: {result.execution_time?.toFixed(2)}s
                              </div>
                            </div>
                          </div>
                          {selectedResultIndex === idx ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                        </button>

                        {selectedResultIndex === idx && (
                          <div className="p-4 bg-gray-900/50 border-t border-gray-800 space-y-4">
                            {/* Input */}
                            <div>
                              <div className="text-xs font-medium text-gray-400 mb-2">Input Prompt</div>
                              <div className="bg-black/50 rounded-lg p-3 text-sm text-gray-300 font-mono">
                                {result.input}
                              </div>
                            </div>

                            {/* Output */}
                            <div>
                              <div className="text-xs font-medium text-gray-400 mb-2">Model Output</div>
                              <div className="bg-black/50 rounded-lg p-3 text-sm text-gray-300 font-mono whitespace-pre-wrap">
                                {result.output}
                              </div>
                            </div>

                            {/* Target (if exists) */}
                            {result.target && (
                              <div>
                                <div className="text-xs font-medium text-gray-400 mb-2">Target/Expected</div>
                                <div className="bg-black/50 rounded-lg p-3 text-sm text-emerald-300 font-mono">
                                  {result.target}
                                </div>
                              </div>
                            )}

                            {/* Variation Info */}
                            {result.variation_type && (
                              <div className="bg-violet-500/10 border border-violet-500/20 rounded-lg p-3">
                                <div className="text-xs font-medium text-violet-400 mb-1">Variation Applied</div>
                                <div className="text-sm text-gray-300">Type: {result.variation_type}</div>
                                {result.original_input && (
                                  <div className="text-xs text-gray-500 mt-2">
                                    Original: {result.original_input.slice(0, 60)}...
                                  </div>
                                )}
                              </div>
                            )}
                            {result.validator_verdict != null && (() => {
                              const cat = categorizeValidatorVerdict(result.validator_verdict);
                              const styles: Record<string, string> = {
                                ACCEPT: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
                                FLAG: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
                                REJECT: 'bg-red-500/10 border-red-500/20 text-red-400',
                                BYPASSED: 'bg-gray-500/10 border-gray-600/30 text-gray-400',
                                UNKNOWN: 'bg-gray-500/10 border-gray-600/30 text-gray-400',
                              };
                              const hasLayers =
                                result.validator_layers &&
                                Object.keys(result.validator_layers).length > 0;
                              return (
                                <div className={`border rounded-lg p-3 ${styles[cat]}`}>
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs font-semibold uppercase tracking-wide">
                                      Validator: {cat}
                                    </span>
                                    <span className="text-xs opacity-70">
                                      ({result.validator_verdict}
                                      {result.valid === false ? ', retained as REJECT' : ''})
                                    </span>
                                  </div>
                                  {hasLayers && (
                                    <details className="text-xs opacity-80">
                                      <summary className="cursor-pointer">Cascade layers</summary>
                                      <pre className="mt-2 bg-black/30 rounded p-2 overflow-auto max-h-48 font-mono leading-relaxed whitespace-pre-wrap">
                                        {JSON.stringify(result.validator_layers, null, 2)}
                                      </pre>
                                    </details>
                                  )}
                                </div>
                              );
                            })()}

                            {/* Judge Evaluation */}
                            {result.judge_score !== null && result.judge_score !== undefined && (
                              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                                <div className="text-xs font-medium text-blue-400 mb-2">Judge Evaluation</div>
                                <div className="flex items-center gap-4 mb-2">
                                  <div>
                                    <div className="text-xs text-gray-500">Score</div>
                                    <div className="text-lg font-semibold text-white">{result.judge_score.toFixed(2)}</div>
                                  </div>
                                </div>
                                {result.judge_results && (
                                  <pre className="bg-black/30 rounded p-3 text-xs text-gray-300 mt-2 overflow-auto max-h-48 font-mono leading-relaxed">
                                    {JSON.stringify(result.judge_results, null, 4)}
                                  </pre>
                                )}
                              </div>
                            )}

                            {/* RTA Detection */}
                            {result.refused !== null && result.refused !== undefined && (() => {

                              const refused = isRefused(result.refused);
                              return (
                                <div className={`border rounded-lg p-3 ${refused
                                  ? 'bg-red-500/10 border-red-500/20'
                                  : 'bg-green-500/10 border-green-500/20'
                                  }`}>
                                  <div className="text-xs font-medium mb-1">
                                    <span className={refused ? 'text-red-400' : 'text-green-400'}>
                                      Refuse-to-Answer Detection
                                    </span>
                                  </div>
                                  <div className="text-sm text-white">
                                    Refused: <span className="font-semibold">{result.refused}</span>
                                  </div>
                                </div>
                              );
                            })()}

                            {/* Include/Exclude Scores */}
                            {(result.include_score !== null || result.exclude_violations !== null) && (
                              <div className="grid grid-cols-2 gap-3">
                                {result.include_score !== null && (
                                  <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
                                    <div className="text-xs text-emerald-400 mb-1">Include Score</div>
                                    <div className="text-lg font-semibold text-white">{result.include_score.toFixed(2)}</div>
                                  </div>
                                )}
                                {result.exclude_violations !== null && (
                                  <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                                    <div className="text-xs text-red-400 mb-1">Exclude Violations</div>
                                    <div className="text-lg font-semibold text-white">{result.exclude_violations}</div>
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Metrics */}
                            {result.metrics && result.metrics.length > 0 && (
                              <div>
                                <div className="text-xs font-medium text-gray-400 mb-2">Metrics</div>
                                <div className="flex flex-wrap gap-2">
                                  {result.metrics.map((metric: string, i: number) => (
                                    <span key={i} className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-300">
                                      {metric}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Metadata */}
                            {result.metadata && Object.keys(result.metadata).length > 0 && (
                              <div>
                                <div className="text-xs font-medium text-gray-400 mb-2">Additional Metadata</div>
                                <pre className="bg-black/30 rounded-lg p-3 text-xs text-gray-400 overflow-auto max-h-48 font-mono leading-relaxed">
                                  {JSON.stringify(result.metadata, null, 2)}
                                </pre>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}