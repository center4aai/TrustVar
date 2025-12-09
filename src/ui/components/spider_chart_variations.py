from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.schemas.task import Task, TaskResult
from src.core.services.eval_service import EvaluationService

plotly_config = dict(width="stretch")

eval_service = EvaluationService()


def get_model_name(model_id: str, models: List) -> str:
    """Get model name from model_id"""
    model = next((m for m in models if m.id == model_id), None)
    return model.name if model else model_id[:12] + "..."


def plot_task_centric_spider(
    task: Task,
    models: List,
):
    """
    Task-centric spider chart with dispersion metrics:
    - Axes (theta): Variations
    - Colors: Different models (one line per model)
    - Metrics: CV, IQR_CV, JSD

    Для каждой вариации собираем accuracy по всем моделям и вычисляем дисперсию.
    """
    st.markdown(f"#### 🕸️ Task: {task.name}")

    # Get all variations
    variations = sorted(task.config.variations.strategies)

    if len(variations) < 3:
        st.info("Need at least 3 variations for spider chart")
        return

    # Dispersion metrics to display
    metric_names = {
        "cv": "Coefficient of Variation",
        "iqr_cv": "IQR-based CV",
    }

    # Create spider chart
    fig = go.Figure()

    # For each metric, create a trace
    for metric_key, metric_name in metric_names.items():
        variation_scores = []
        variation_names = []

        # For each variation, collect accuracy from all models
        for variation in variations:
            values = []

            # Collect accuracy for each model on this variation
            for model in models:
                if model.id not in task.model_ids:
                    continue

                # Get results for this model and variation
                per_model_var_results = [
                    r
                    for r in task.results
                    if r.model_id == model.id and r.variation_type == variation
                ]

                if per_model_var_results:
                    # Calculate accuracy for this model on this variation
                    accuracy = eval_service._accuracy(per_model_var_results)
                    values.append(accuracy)

            # Calculate dispersion metric for this variation
            if len(values) >= 2:  # Need at least 2 values for dispersion
                if metric_key == "cv":
                    score = eval_service._calculate_cv(values)
                elif metric_key == "iqr_cv":
                    score = eval_service._calculate_iqr_cv(values)
                else:
                    score = 0

                if not np.isnan(score):
                    variation_scores.append(score)
                    variation_names.append(variation)

        if not variation_scores:
            continue

        # Close the loop
        scores_closed = variation_scores + [variation_scores[0]]
        variations_closed = variation_names + [variation_names[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=variations_closed,
                fill="toself",
                name=metric_name,
                hovertemplate="%{theta}<br>%{fullData.name}<br>"
                + "Score: %{r:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title="Variation Stability Across Models",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config=plotly_config)


def plot_model_centric_spider(
    all_tasks: List[Task],
    selected_model_id: str,
    models: List,
    selected_task_names: List[str],
):
    """
    Model-centric spider chart with dispersion metrics:
    - Axes (theta): Variations
    - Colors: Different tasks (one line per selected task)
    - Selected model: User selects ONE model to analyze

    Для выбранной модели, для каждой задачи вычисляем CV по вариациям.

    Args:
        all_tasks: List of all completed tasks
        selected_model_id: ID of the model to analyze
        models: List of all models
        selected_task_names: List of task names selected by user
    """
    model_name = get_model_name(selected_model_id, models)

    st.markdown(f"#### 🕸️ Model: {model_name}")

    if not selected_task_names:
        st.warning("Please select at least one task")
        return

    # Collect all variations across selected tasks
    all_variations = set()

    # First pass: collect all variations
    for task in all_tasks:
        if task.name not in selected_task_names:
            continue
        if not task.config.variations.enabled:
            continue

        for result in task.results:
            if result.model_id == selected_model_id:
                variation = result.variation_type or "original"
                all_variations.add(variation)

    if len(all_variations) < 3:
        st.info("Need at least 3 variations for spider chart")
        return

    variations_sorted = sorted(list(all_variations))

    # Create spider chart - one trace per task
    fig = go.Figure()

    for task in all_tasks:
        if task.name not in selected_task_names:
            continue
        if not task.config.variations.enabled:
            continue

        # Filter results for selected model
        model_results = [r for r in task.results if r.model_id == selected_model_id]

        if not model_results:
            continue

        # Calculate CV for each variation
        variation_scores = []
        variation_names = []

        # Collect accuracy values across all variations for CV calculation
        values = []

        for variation in variations_sorted:
            var_results = [r for r in model_results if r.variation_type == variation]

            if var_results:
                # Calculate accuracy for this variation
                accuracy = eval_service._accuracy(var_results)
                values.append(accuracy)
                variation_names.append(variation)

        # Calculate CV across all variations for this task
        if len(values) >= 2:
            cv_score = eval_service._calculate_cv(values)

            if not np.isnan(cv_score):
                # For spider chart, we need one value per variation
                # So we'll show the same CV for all variations of this task
                variation_scores = [cv_score] * len(
                    variation_names
                )
            else:
                continue
        else:
            continue

        if not variation_scores:
            continue

        # Close the loop
        scores_closed = variation_scores + [variation_scores[0]]
        variations_closed = variation_names + [variation_names[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=variations_closed,
                fill="toself",
                name=task.name,
                hovertemplate="%{theta}<br>%{fullData.name}<br>"
                + "CV: %{r:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title=f"Task Stability Index for Model: {model_name}",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config=plotly_config)


def plot_augmentation_impact_chart(
    results: List[TaskResult],
    task_name: str,
    model_ids: List[str],
    models: List,
):
    """
    Bar chart comparing augmentation impact on metrics
    """
    # Group by model and variation, calculate accuracy
    grouped_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        if result.model_id in model_ids:
            variation = result.variation_type or "original"
            grouped_data[result.model_id][variation].append(result)

    # Calculate metrics
    data_rows = []

    for model_id, variations in grouped_data.items():
        model_name = get_model_name(model_id, models)

        for variation, var_results in variations.items():
            # Calculate accuracy (всегда есть)
            accuracy = eval_service._accuracy(var_results)  # TODO: не только accuracy

            if accuracy >= 0:  # Даже 0 - валидное значение
                data_rows.append(
                    {
                        "Model": model_name,
                        "Variation": variation,
                        "Metric": "Accuracy",
                        "Value": accuracy,
                    }
                )

            # Also add judge scores if available
            judge_scores = [
                r.judge_score for r in var_results if r.judge_score is not None
            ]
            if judge_scores:
                avg_judge = sum(judge_scores) / len(judge_scores)
                data_rows.append(
                    {
                        "Model": model_name,
                        "Variation": variation,
                        "Metric": "Judge Score",
                        "Value": avg_judge,
                    }
                )

            # Add include scores if available
            include_scores = [
                r.include_score for r in var_results if r.include_score is not None
            ]
            if include_scores:
                avg_include = sum(include_scores) / len(include_scores)
                data_rows.append(
                    {
                        "Model": model_name,
                        "Variation": variation,
                        "Metric": "Include Score",
                        "Value": avg_include,
                    }
                )

    if not data_rows:
        st.warning("No data available for augmentation impact chart")
        return

    df = pd.DataFrame(data_rows)

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="Variation",
        y="Value",
        color="Model",
        facet_col="Metric",
        barmode="group",
        title=f"Impact of Variations on Metrics - {task_name}",
        labels={"Value": "Metric Value", "Variation": "Variation Type"},
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config=plotly_config)


def create_task_metrics_table(
    all_tasks_results: Dict[str, List[TaskResult]],
    selected_model_id: str,
    selected_task_names: List[str],
) -> pd.DataFrame:
    """
    Create a table showing dispersion metrics for each selected task

    Args:
        all_tasks_results: Dict mapping task_name -> list of TaskResult
        selected_model_id: ID of the model to analyze
        models: List of all models
        selected_task_names: List of task names selected by user
    """
    table_data = []

    for task_name in selected_task_names:
        if task_name not in all_tasks_results:
            continue

        task_results = all_tasks_results[task_name]
        model_results = [r for r in task_results if r.model_id == selected_model_id]

        if not model_results:
            continue

        # Get all variations for this task
        variations = set()
        for result in model_results:
            variation = result.variation_type or "original"
            variations.add(variation)

        # Collect accuracy for each variation
        values = []
        for variation in variations:
            var_results = [r for r in model_results if r.variation_type == variation]
            if var_results:
                accuracy = eval_service._accuracy(var_results)
                values.append(accuracy)

        if len(values) < 2:
            continue

        # Calculate dispersion metrics
        cv = eval_service._calculate_cv(values)
        iqr_cv = eval_service._calculate_iqr_cv(values)
        jsd = eval_service._calculate_jsd_divergence(values)

        row = {
            "Task": task_name,
            "Result Count": len(model_results),
            "Mean Accuracy": f"{np.mean(values):.2f}",
            "TSI (%)": cv if not np.isnan(cv) else np.nan,
            "IQR-CV (%)": iqr_cv if not np.isnan(iqr_cv) else np.nan,
            "JSD": jsd if not np.isnan(jsd) else np.nan,
        }

        table_data.append(row)

    return pd.DataFrame(table_data)


def plot_rta_spider_chart(
    task: Task,
    models: List,
):
    """
    RTA spider chart:
    - Axes (theta): Variations
    - Colors: Different models (one line per model)
    - Metric: RTA (Refuse-to-Answer) rate per variation

    For each variation, we calculate the refusal rate for each model.
    """
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown("#### 🕸️ RTA Rates Across Variations")
    st.caption("Shows refusal rates for each model across different variations")

    # Get all variations
    variations = sorted(set(r.variation_type for r in task.results if r.variation_type))

    # Add 'original' if there are results without variation
    if any(not r.variation_type for r in task.results):
        variations = ["original"] + variations

    if len(variations) < 3:
        st.info("Need at least 3 variations for spider chart")
        return

    # Create spider chart
    fig = go.Figure()

    # For each model, create a trace
    for model in models:
        if model.id not in task.model_ids:
            continue

        model_name = get_model_name(model.id, models)
        variation_scores = []
        variation_names = []

        # For each variation, calculate RTA rate for this model
        for variation in variations:
            # Get results for this model and variation
            if variation == "original":
                var_results = [
                    r
                    for r in task.results
                    if r.model_id == model.id and not r.variation_type
                ]
            else:
                var_results = [
                    r
                    for r in task.results
                    if r.model_id == model.id and r.variation_type == variation
                ]

            if var_results:
                # Calculate refusal rate
                refused_count = sum(1 for r in var_results if r.refused == "1")
                refusal_rate = (refused_count / len(var_results)) * 100

                variation_scores.append(refusal_rate)
                variation_names.append(variation)

        if not variation_scores:
            continue

        # Close the loop for spider chart
        scores_closed = variation_scores + [variation_scores[0]]
        variations_closed = variation_names + [variation_names[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=variations_closed,
                fill="toself",
                name=model_name,
                hovertemplate="%{theta}<br>%{fullData.name}<br>"
                + "Refusal Rate: %{r:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickfont=dict(size=10),
                range=[0, 100],  # RTA rate is 0-100%
            ),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        title="Refusal Rates Across Variations by Model",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, config={"width": "stretch"})
