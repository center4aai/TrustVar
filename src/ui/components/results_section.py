# src/ui/components/results_section.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.schemas.task import TaskStatus
from src.ui.api_client import get_api_client


def _render_results_overview(api_client):
    """Обзор всех результатов"""

    st.markdown("### 📊 Overview of Completed Tasks")

    # Селектор задачи
    try:
        completed_tasks = api_client.list_tasks(status=TaskStatus.COMPLETED)
        # completed_tasks = [t for t in tasks if t.status == "completed"]

        if completed_tasks:
            # Выбор задачи
            task_options = {
                f"{t.name} ({t.created_at.strftime('%Y-%m-%d %H:%M')})": t.id
                for t in completed_tasks
            }

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_label = st.selectbox(
                    "Select a task to view results",
                    list(task_options.keys()),
                    key="results_task_selector",
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📊 View Results", width="stretch"):
                    st.session_state.selected_task_id = task_options[selected_label]
                    st.rerun()

            st.divider()

            # Сводная таблица всех задач
            st.markdown("### 📋 All Completed Tasks")

            tasks_data = []
            for task in completed_tasks:
                duration = (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.completed_at and task.started_at
                    else 0
                )

                tasks_data.append(
                    {
                        "Task": task.name,
                        "Samples": task.total_samples,
                        "Duration (s)": f"{duration:.1f}",
                        "Avg Time/Sample": f"{duration / task.total_samples:.2f}s"
                        if task.total_samples > 0
                        else "N/A",
                        **{
                            # k.replace("_", " ").title(): f"{v:.2f}"
                            # for k, v in task.aggregated_metrics.items()
                            metric.replace("_", " ").title(): f"{value:.2f}"
                            for model_id, metrics in task.aggregated_metrics.items()
                            for metric, value in metrics.items()
                        },
                        "Created": task.created_at.strftime("%Y-%m-%d %H:%M"),
                    }
                )

            df = pd.DataFrame(tasks_data)
            st.dataframe(df, width="stretch", height=400)

            # Сравнительные графики
            if len(completed_tasks) > 1:
                st.divider()
                st.markdown("### 📊 Comparative Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # График метрик
                    _plot_metrics_comparison(completed_tasks)

                with col2:
                    # График производительности
                    _plot_performance_comparison(completed_tasks)

        else:
            st.info(
                "📭 No completed tasks yet. Complete some tasks to see results here!"
            )

    except Exception as e:
        st.error(f"❌ Error loading results: {e}")


def _plot_metrics_comparison(tasks):
    """График сравнения метрик"""

    st.markdown("**Metrics Comparison**")

    # Собираем все метрики
    all_metrics = set()
    for task in tasks:
        all_metrics.update(task.aggregated_metrics.keys())

    if not all_metrics:
        st.info("No metrics to compare")
        return

    # Готовим данные для графика
    data = []
    for task in tasks:
        for metric in all_metrics:
            data.append(
                {
                    "Task": task.name[:20],
                    "Metric": metric.replace("_", " ").title(),
                    "Value": task.aggregated_metrics.get(metric, 0),
                }
            )

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Task",
        y="Value",
        color="Metric",
        barmode="group",
        title="Metrics Comparison Across Tasks",
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Task",
        yaxis_title="Score (%)",
    )

    st.plotly_chart(fig, width="stretch")


def _plot_performance_comparison(tasks):
    """График сравнения производительности"""

    st.markdown("**Performance Comparison**")

    # Готовим данные
    data = []
    for task in tasks:
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            avg_time = duration / task.total_samples if task.total_samples > 0 else 0

            data.append(
                {
                    "Task": task.name[:20],
                    "Total Duration (s)": duration,
                    "Avg Time/Sample (s)": avg_time,
                    "Throughput (samples/s)": task.total_samples / duration
                    if duration > 0
                    else 0,
                }
            )

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Avg Time/Sample",
            x=df["Task"],
            y=df["Avg Time/Sample (s)"],
            marker_color="rgb(102, 126, 234)",
        )
    )

    fig.update_layout(
        title="Average Time per Sample",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Task",
        yaxis_title="Time (s)",
    )

    st.plotly_chart(fig, width="stretch")


# src/ui/components/results_section.py (дополнение)
from collections import defaultdict


def render_results_section():
    """Рендер секции результатов"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📈 Results & Analytics")

    api_client = get_api_client()

    if "selected_task_id" in st.session_state:
        _render_task_results(st.session_state.selected_task_id, api_client)
    else:
        _render_results_overview(api_client)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_task_results(task_id, api_client):
    """Детальные результаты с поддержкой множественных моделей"""

    try:
        task = api_client.get_task(task_id)

        if not task:
            st.error("Task not found")
            return

        # Заголовок
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"## 📊 {task.name}")
            st.caption(f"Task ID: {task.id}")
            if len(task.model_ids) > 1:
                st.info(f"ℹ️ This task compared {len(task.model_ids)} models")

        with col2:
            if st.button("⬅️ Back to Overview", width="stretch"):
                del st.session_state.selected_task_id
                st.rerun()

        st.divider()

        # Tabs для разных видов анализа
        if len(task.model_ids) > 1:
            tabs = st.tabs(
                [
                    "📊 Model Comparison",
                    "📈 Metrics",
                    "📄 Detailed Results",
                    "🔄 Variations Analysis",
                ]
            )
        else:
            tabs = st.tabs(
                ["📈 Metrics", "📄 Detailed Results", "🔄 Variations Analysis"]
            )

        # TAB: Model Comparison (только если > 1 модели)
        if len(task.model_ids) > 1:
            with tabs[0]:
                _render_model_comparison(task, api_client)
            tab_offset = 1
        else:
            tab_offset = 0

        # TAB: Metrics
        with tabs[tab_offset]:
            _render_metrics_tab(task, api_client)

        # TAB: Detailed Results
        with tabs[tab_offset + 1]:
            _render_detailed_results(task, api_client)

        # TAB: Variations Analysis
        with tabs[tab_offset + 2]:
            if task.config.enable_variations:
                _render_variations_analysis(task)
            else:
                st.info("Variations were not enabled for this task")

    except Exception as e:
        st.error(f"❌ Error loading task results: {e}")


def _render_model_comparison(task, api_client):
    """Сравнение моделей"""

    st.markdown("### 🏆 Model Performance Comparison")

    try:
        # Получаем сравнение
        response = api_client.compare_results(task.id)

        if not response:
            st.warning("Comparison data not available")
            return

        # Лучшая модель
        if "best_model" in response:
            best = response["best_model"]
            st.success(
                f"🥇 Best Model: {best['model_id'][:12]}... ({best['reason']}) - Score: {best['score']:.2f}"
            )

        st.divider()

        # Таблица сравнения
        comparison_data = []
        for model_id, stats in response["models"].items():
            row = {
                "Model ID": model_id[:12] + "...",
                "Results": stats["total_results"],
                "Avg Time": f"{stats['avg_execution_time']:.3f}s",
            }

            # Метрики
            for metric, value in stats.get("metrics", {}).items():
                row[metric.title()] = f"{value:.2f}"

            # Judge score
            if "avg_judge_score" in stats:
                row["Judge Score"] = f"{stats['avg_judge_score']:.2f}"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width="stretch")

        st.divider()

        # Визуализации
        col1, col2 = st.columns(2)

        with col1:
            # График метрик
            _plot_model_metrics_comparison(response["models"])

        with col2:
            # График производительности
            _plot_model_performance(response["models"])

        # Judge scores (если есть)
        if any("avg_judge_score" in stats for stats in response["models"].values()):
            st.divider()
            st.markdown("### 🎯 LLM Judge Scores")
            _plot_judge_scores(response["models"])

    except Exception as e:
        st.error(f"Error loading comparison: {e}")


def _render_metrics_tab(task, api_client):
    """Вкладка метрик"""

    st.markdown("### 📊 Performance Metrics")

    # Группируем результаты по моделям
    results_by_model = defaultdict(list)
    for result in task.results:
        results_by_model[result.model_id].append(result)

    # Для каждой модели
    for model_id, model_results in results_by_model.items():
        with st.expander(f"📦 Model: {model_id[:12]}...", expanded=True):
            # Базовые метрики
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Results", len(model_results))

            avg_time = sum(r.execution_time for r in model_results) / len(model_results)
            col2.metric("Avg Time", f"{avg_time:.3f}s")

            # Standard metrics
            if model_id in task.aggregated_metrics:
                metrics = task.aggregated_metrics[model_id]
                for i, (metric, value) in enumerate(metrics.items()):
                    if i < 2:
                        [col3, col4][i].metric(metric.title(), f"{value:.2f}")

            # Judge scores
            judge_scores = [
                r.judge_score for r in model_results if r.judge_score is not None
            ]
            if judge_scores:
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Avg Judge Score", f"{sum(judge_scores) / len(judge_scores):.2f}"
                )
                col2.metric("Min Judge Score", f"{min(judge_scores):.2f}")
                col3.metric("Max Judge Score", f"{max(judge_scores):.2f}")


def _render_detailed_results(task, api_client):
    """Детальные результаты"""

    st.markdown("### 📄 Detailed Results")

    # Фильтры
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Фильтр по модели
        model_options = ["All"] + list(set(r.model_id for r in task.results))
        selected_model = st.selectbox("Model", model_options, key="filter_model")

    with col2:
        # Фильтр по вариациям
        variation_types = ["All"] + list(
            set(r.variation_type for r in task.results if r.variation_type)
        )
        selected_variation = st.selectbox(
            "Variation", variation_types, key="filter_variation"
        )

    with col3:
        search = st.text_input("🔍 Search", key="search_results")

    with col4:
        limit = st.number_input("Limit", 10, 100, 20, key="limit_results")

    # Фильтрация
    filtered_results = task.results[:limit]

    if selected_model != "All":
        filtered_results = [r for r in filtered_results if r.model_id == selected_model]

    if selected_variation != "All":
        filtered_results = [
            r for r in filtered_results if r.variation_type == selected_variation
        ]

    if search:
        filtered_results = [
            r
            for r in filtered_results
            if search.lower() in r.input.lower() or search.lower() in r.output.lower()
        ]

    # Отображение
    for i, result in enumerate(filtered_results, 1):
        is_correct = result.output == str(result.target) if result.target else None

        title = f"**Result {i}:** {result.input[:60]}..."
        if result.variation_type:
            title += f" [Variation: {result.variation_type}]"
        if is_correct is not None:
            title += f" {'✅' if is_correct else '❌'}"
        if result.judge_score:
            title += f" [Judge: {result.judge_score:.1f}/10]"

        with st.expander(title, expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Model:** {result.model_id[:12]}...")

                if result.variation_type:
                    st.markdown(f"**Variation:** {result.variation_type}")
                    if result.original_input:
                        st.markdown("**Original Prompt:**")
                        st.code(result.original_input, language=None)

                st.markdown("**Input:**")
                st.code(result.input, language=None)

                st.markdown("**Output:**")
                st.code(result.output, language=None)

            with col2:
                if result.target:
                    st.markdown("**Expected:**")
                    st.code(result.target, language=None)

                st.markdown(f"**⏱️ Time:** {result.execution_time:.3f}s")

                if result.judge_score:
                    st.markdown(f"**🎯 Judge Score:** {result.judge_score:.2f}/10")
                    if result.judge_reasoning:
                        st.markdown("**Judge Reasoning:**")
                        st.info(result.judge_reasoning)


def _render_variations_analysis(task):
    """Анализ вариаций"""

    st.markdown("### 🔄 Variations Analysis")

    # Группируем по типам вариаций
    variation_results = defaultdict(lambda: defaultdict(list))

    for result in task.results:
        if result.variation_type:
            variation_results[result.variation_type][result.model_id].append(result)

    if not variation_results:
        st.info("No variations found in results")
        return

    # Для каждого типа вариации
    for variation_type, models_data in variation_results.items():
        st.markdown(f"#### {variation_type.replace('_', ' ').title()}")

        # Статистика
        total_variations = sum(len(results) for results in models_data.values())
        st.write(f"Total variations: {total_variations}")

        # Сравнение моделей на этих вариациях
        col1, col2 = st.columns(2)

        with col1:
            # Avg execution time
            avg_times = {}
            for model_id, results in models_data.items():
                avg_time = sum(r.execution_time for r in results) / len(results)
                avg_times[model_id[:12]] = avg_time

            fig = px.bar(
                x=list(avg_times.keys()),
                y=list(avg_times.values()),
                title=f"Avg Time - {variation_type}",
                labels={"x": "Model", "y": "Time (s)"},
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Judge scores (если есть)
            if any(r.judge_score for results in models_data.values() for r in results):
                avg_scores = {}
                for model_id, results in models_data.items():
                    scores = [r.judge_score for r in results if r.judge_score]
                    if scores:
                        avg_scores[model_id[:12]] = sum(scores) / len(scores)

                fig = px.bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    title=f"Avg Judge Score - {variation_type}",
                    labels={"x": "Model", "y": "Score"},
                )
                st.plotly_chart(fig, width="stretch")

        st.divider()


def _plot_model_metrics_comparison(models_data):
    """График сравнения метрик моделей"""

    st.markdown("**Metrics Comparison**")

    data = []
    for model_id, stats in models_data.items():
        for metric, value in stats.get("metrics", {}).items():
            data.append(
                {
                    "Model": model_id[:12],
                    "Metric": metric.replace("_", " ").title(),
                    "Value": value,
                }
            )

    if data:
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Metrics by Model",
        )
        st.plotly_chart(fig, width="stretch")


def _plot_model_performance(models_data):
    """График производительности моделей"""

    st.markdown("**Performance Comparison**")

    data = {"Model": [], "Avg Time (s)": [], "Total Results": []}

    for model_id, stats in models_data.items():
        data["Model"].append(model_id[:12])
        data["Avg Time (s)"].append(stats["avg_execution_time"])
        data["Total Results"].append(stats["total_results"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Avg Time",
            x=data["Model"],
            y=data["Avg Time (s)"],
            marker_color="rgb(102, 126, 234)",
        )
    )

    fig.update_layout(
        title="Average Execution Time by Model",
        xaxis_title="Model",
        yaxis_title="Time (s)",
    )
    st.plotly_chart(fig, width="stretch")


def _plot_judge_scores(models_data):
    """График Judge scores"""

    data = {"Model": [], "Avg Score": [], "Min Score": [], "Max Score": []}

    for model_id, stats in models_data.items():
        if "avg_judge_score" in stats:
            data["Model"].append(model_id[:12])
            data["Avg Score"].append(stats["avg_judge_score"])
            data["Min Score"].append(stats.get("min_judge_score", 0))
            data["Max Score"].append(stats.get("max_judge_score", 10))

    if data["Model"]:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Avg Judge Score",
                x=data["Model"],
                y=data["Avg Score"],
                marker_color="rgb(76, 175, 80)",
            )
        )

        fig.update_layout(
            title="LLM Judge Scores by Model",
            xaxis_title="Model",
            yaxis_title="Score (1-10)",
        )
        st.plotly_chart(fig, width="stretch")
