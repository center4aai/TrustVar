import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { TrustVarMetrics } from '@/api/types';
import { MetricsTable } from './ResultsSection';

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

describe('<MetricsTable />', () => {
  it('shows the empty-state message when there are no per_task metrics', () => {
    render(<MetricsTable trustVarMetrics={makeMetrics({})} />);
    expect(screen.getByText(/No task-level TrustVar metrics/i)).toBeInTheDocument();
  });

  it('renders a row labelled by the benchmark prompt and shows TSI', () => {
    const prompt = 'Solve: 2 + 2 = ?';
    render(
      <MetricsTable
        trustVarMetrics={makeMetrics({
          per_task_tsi: { [prompt]: { A: 12.5 } },
          per_task_ear: { [prompt]: { A: 0.9 } },
        })}
      />,
    );
    expect(screen.getByText(new RegExp(prompt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))).toBeInTheDocument();
    expect(screen.getByText('12.50')).toBeInTheDocument();
  });

  it('renders a † marker (not a number) for suppressed TSI', () => {
    const prompt = 'fragile task';
    render(
      <MetricsTable
        trustVarMetrics={makeMetrics({
          per_task_tsi: { [prompt]: { A: 99 } },
          per_task_uninformative: { [prompt]: { A: true } },
        })}
      />,
    );
    expect(screen.getByText('†')).toBeInTheDocument();
    expect(screen.queryByText('99.00')).not.toBeInTheDocument();
  });
});
