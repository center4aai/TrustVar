# src/ui/components/results_section.py
import asyncio

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.services.dataset_service import DatasetService
from src.core.services.model_service import ModelService
from src.core.services.task_service import TaskService


def render_results_section():
    """Рендер секции результатов"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📈 Results & Analytics")

    # Инициализация
    if "task_service" not in st.session_state:
        st.session_state.task_service = TaskService()
    if "dataset_service" not in st.session_state:
        st.session_state.dataset_service = DatasetService()
    if "model_service" not in st.session_state:
        st.session_state.model_service = ModelService()

    task_service = st.session_state.task_service
    dataset_service = st.session_state.dataset_service
    model_service = st.session_state.model_service

    # Проверяем, выбрана ли задача
    if "selected_task_id" in st.session_state:
        _render_task_results(
            st.session_state.selected_task_id,
            task_service,
            dataset_service,
            model_service,
        )
    else:
        _render_results_overview(task_service, model_service)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_results_overview(task_service, model_service):
    """Обзор всех результатов"""

    st.markdown("### 📊 Overview of Completed Tasks")

    # Селектор задачи
    try:
        tasks = asyncio.run(task_service.list_tasks(limit=100))
        completed_tasks = [t for t in tasks if t.status == "completed"]

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
                if st.button("📊 View Results", use_container_width=True):
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
                            k.replace("_", " ").title(): f"{v:.2f}"
                            for k, v in task.aggregated_metrics.items()
                        },
                        "Created": task.created_at.strftime("%Y-%m-%d %H:%M"),
                    }
                )

            df = pd.DataFrame(tasks_data)
            st.dataframe(df, use_container_width=True, height=400)

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


def _render_task_results(task_id, task_service, dataset_service, model_service):
    """Детальные результаты конкретной задачи"""

    try:
        task = asyncio.run(task_service.get_task(task_id))

        if not task:
            st.error("Task not found")
            return

        # Заголовок
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"## 📊 {task.name}")
            st.caption(f"Task ID: {task.id}")

        with col2:
            if st.button("⬅️ Back to Overview", use_container_width=True):
                del st.session_state.selected_task_id
                st.rerun()

        st.divider()

        # Информация о задаче
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📝 Task Details**")
            st.write(f"Type: {task.task_type}")
            st.write(f"Status: {task.status}")
            st.write(f"Created: {task.created_at.strftime('%Y-%m-%d %H:%M')}")

        with col2:
            st.markdown("**📊 Dataset**")
            dataset = asyncio.run(dataset_service.get_dataset(task.dataset_id))
            if dataset:
                st.write(f"Name: {dataset.name}")
                st.write(f"Size: {dataset.size} items")

        with col3:
            st.markdown("**🤖 Model**")
            model = asyncio.run(model_service.get_model(task.model_id))
            if model:
                st.write(f"Name: {model.name}")
                st.write(f"Provider: {model.provider}")

        st.divider()

        # Метрики
        if task.aggregated_metrics:
            st.markdown("### 📈 Performance Metrics")

            cols = st.columns(len(task.aggregated_metrics))
            for col, (metric, value) in zip(cols, task.aggregated_metrics.items()):
                col.metric(metric.replace("_", " ").title(), f"{value:.2f}%")

            st.divider()

        # Временные метрики
        if task.started_at and task.completed_at:
            st.markdown("### ⏱️ Execution Statistics")

            duration = (task.completed_at - task.started_at).total_seconds()
            avg_time = duration / task.total_samples if task.total_samples > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", task.total_samples)
            col2.metric("Total Duration", f"{duration:.2f}s")
            col3.metric("Avg Time/Sample", f"{avg_time:.2f}s")
            col4.metric(
                "Throughput",
                f"{task.total_samples / duration:.2f} samples/s"
                if duration > 0
                else "N/A",
            )

            st.divider()

        # Детальные результаты
        if task.results:
            st.markdown("### 📄 Detailed Results")

            # Фильтры
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_result = st.text_input(
                    "🔍 Search in results", key="search_results"
                )
            with col2:
                show_only = st.selectbox(
                    "Show", ["All", "Correct", "Incorrect"], key="filter_results"
                )
            with col3:
                limit = st.number_input("Limit", 10, 100, 20, key="limit_results")

            # Фильтрация результатов
            filtered_results = task.results[:limit]

            if search_result:
                filtered_results = [
                    r
                    for r in filtered_results
                    if search_result.lower() in r.input.lower()
                    or search_result.lower() in r.output.lower()
                ]

            # Отображение результатов
            for i, result in enumerate(filtered_results, 1):
                is_correct = None
                if result.expected_output:
                    is_correct = (
                        result.output.strip().lower()
                        == result.expected_output.strip().lower()
                    )

                # Пропускаем если фильтр активен
                if show_only == "Correct" and not is_correct:
                    continue
                if show_only == "Incorrect" and is_correct:
                    continue

                with st.expander(
                    f"**Result {i}:** {result.input[:80]}... "
                    f"{'✅' if is_correct else '❌' if is_correct is False else '➖'}",
                    expanded=False,
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Input:**")
                        st.code(result.input, language=None)

                        st.markdown("**Output:**")
                        st.code(result.output, language=None)

                    with col2:
                        if result.expected_output:
                            st.markdown("**Expected Output:**")
                            st.code(result.expected_output, language=None)

                            if is_correct:
                                st.success("✅ Match!")
                            else:
                                st.error("❌ Mismatch")

                        st.markdown("**Execution Time:**")
                        st.write(f"{result.execution_time:.3f}s")

                        if result.metrics:
                            st.markdown("**Metrics:**")
                            for metric, value in result.metrics.items():
                                st.write(f"{metric}: {value:.2f}")

            # График распределения времени выполнения
            st.divider()
            st.markdown("### 📊 Execution Time Distribution")

            execution_times = [r.execution_time for r in task.results]
            fig = px.histogram(
                x=execution_times,
                nbins=30,
                labels={"x": "Execution Time (s)", "y": "Count"},
                title="Distribution of Execution Times",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No detailed results available")

    except Exception as e:
        st.error(f"❌ Error loading task results: {e}")


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

    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)
