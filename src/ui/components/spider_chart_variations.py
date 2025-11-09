# src/ui/components/spider_chart_variations.py
from collections import defaultdict
from typing import List

import plotly.graph_objects as go
import streamlit as st

from src.core.schemas.task import TaskResult

plotly_config = dict(use_container_width=True)


def plot_variation_spider_chart(
    results: List[TaskResult], selected_model_id: str, metric_names: List[str]
):
    """
    Создать паутинный график для анализа вариаций

    Args:
        results: Результаты задачи
        selected_model_id: ID выбранной модели для анализа
        metric_names: Список метрик для отображения
    """

    # Фильтруем результаты по модели
    model_results = [r for r in results if r.model_id == selected_model_id]

    if not model_results:
        st.warning("No results found for selected model")
        return

    # Группируем по типам вариаций
    variation_groups = defaultdict(list)

    for result in model_results:
        variation_type = result.variation_type or "original"
        variation_groups[variation_type].append(result)

    if len(variation_groups) < 2:
        st.info("Need at least 2 variation types for spider chart (including original)")
        return

    # Вычисляем средние метрики для каждого типа вариации
    variation_metrics = {}

    for variation_type, var_results in variation_groups.items():
        metrics = {}

        # Judge score
        judge_scores = [r.judge_score for r in var_results if r.judge_score is not None]
        if judge_scores:
            metrics["Judge Score"] = sum(judge_scores) / len(judge_scores)

        # Include/Exclude score
        include_scores = [
            r.include_score for r in var_results if r.include_score is not None
        ]
        if include_scores:
            metrics["Include Score"] = sum(include_scores) / len(include_scores)

        # Execution time (инвертируем для графика - меньше = лучше)
        exec_times = [r.execution_time for r in var_results]
        if exec_times:
            avg_time = sum(exec_times) / len(exec_times)
            # Нормализуем: 10 - (time / max_time * 10)
            max_time = max(r.execution_time for r in model_results)
            metrics["Speed"] = 10 - (avg_time / max_time * 10) if max_time > 0 else 5

        # Custom metrics
        for metric_name in metric_names:
            values = [
                r.metrics.get(metric_name)
                for r in var_results
                if metric_name in r.metrics
            ]
            if values:
                # Нормализуем к шкале 0-10
                avg_val = sum(values) / len(values)
                metrics[metric_name.replace("_", " ").title()] = (
                    avg_val / 10
                )  # Предполагаем что метрики в процентах

        variation_metrics[variation_type] = metrics

    # Получаем все уникальные метрики
    all_metrics = set()
    for metrics in variation_metrics.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(all_metrics)

    if not all_metrics:
        st.warning("No metrics available for spider chart")
        return

    # Создаем паутинный график
    fig = go.Figure()

    for variation_type, metrics in variation_metrics.items():
        # Значения для каждой метрики
        values = [metrics.get(metric, 0) for metric in all_metrics]

        # Замыкаем круг (добавляем первое значение в конец)
        values_closed = values + [values[0]]
        metrics_closed = list(all_metrics) + [all_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill="toself",
                name=variation_type.replace("_", " ").title(),
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title=f"Variation Analysis - {selected_model_id[:12]}...",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config=plotly_config)

    # Таблица с детальными значениями
    st.markdown("### Detailed Metrics")

    import pandas as pd

    table_data = []
    for variation_type, metrics in variation_metrics.items():
        row = {"Variation": variation_type.replace("_", " ").title()}
        row.update({k: f"{v:.2f}" for k, v in metrics.items()})
        table_data.append(row)

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)


def plot_multi_model_comparison_spider(
    results: List[TaskResult], model_ids: List[str], metric_names: List[str]
):
    """
    Паутинный график для сравнения нескольких моделей

    Args:
        results: Все результаты
        model_ids: Список ID моделей для сравнения
        metric_names: Метрики для отображения
    """

    # Группируем по моделям
    model_groups = defaultdict(list)

    for result in results:
        if result.model_id in model_ids:
            model_groups[result.model_id].append(result)

    if len(model_groups) < 2:
        st.info("Need at least 2 models for comparison spider chart")
        return

    # Вычисляем метрики для каждой модели
    model_metrics = {}

    for model_id, model_results in model_groups.items():
        metrics = {}

        # Judge score
        judge_scores = [
            r.judge_score for r in model_results if r.judge_score is not None
        ]
        if judge_scores:
            metrics["Judge Score"] = sum(judge_scores) / len(judge_scores)

        # Include score
        include_scores = [
            r.include_score for r in model_results if r.include_score is not None
        ]
        if include_scores:
            metrics["Include Score"] = (
                sum(include_scores) / len(include_scores) / 10
            )  # Нормализация

        # Speed
        exec_times = [r.execution_time for r in model_results]
        if exec_times:
            avg_time = sum(exec_times) / len(exec_times)
            max_time = max(r.execution_time for r in results)
            metrics["Speed"] = 10 - (avg_time / max_time * 10) if max_time > 0 else 5

        # Custom metrics
        for metric_name in metric_names:
            values = [
                r.metrics.get(metric_name)
                for r in model_results
                if metric_name in r.metrics
            ]
            if values:
                avg_val = sum(values) / len(values)
                metrics[metric_name.replace("_", " ").title()] = avg_val / 10

        model_metrics[model_id] = metrics

    # Получаем все метрики
    all_metrics = set()
    for metrics in model_metrics.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(all_metrics)

    if not all_metrics:
        st.warning("No metrics available")
        return

    # Создаем график
    fig = go.Figure()

    for model_id, metrics in model_metrics.items():
        values = [metrics.get(metric, 0) for metric in all_metrics]
        values_closed = values + [values[0]]
        metrics_closed = list(all_metrics) + [all_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill="toself",
                name=model_id[:12] + "...",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title="Model Comparison",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config=plotly_config)
