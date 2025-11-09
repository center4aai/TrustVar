# src/ui/components/results_section.py
from collections import defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.schemas.task import TaskStatus
from src.ui.api_client import get_api_client
from src.ui.components.spider_chart_variations import (
    plot_multi_model_comparison_spider,
    plot_variation_spider_chart,
)

plotly_config = dict(use_container_width=True)


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
    """Детальные результаты с поддержкой всех новых функций"""

    try:
        task = api_client.get_task(task_id)
        models = api_client.list_models()

        if not task:
            st.error("Task not found")
            return

        # Заголовок
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"## 📊 {task.name}")
            st.caption(f"Task ID: {task.id}")

            # Информация о типе задачи
            badges = []
            if len(task.model_ids) > 1:
                badges.append(f"**{len(task.model_ids)} Models**")
            if task.config.variations.enabled:
                badges.append("**Variations ✓**")
            if task.config.judge.enabled:
                badges.append("**Judge ✓**")
            if task.config.rta.enabled:
                badges.append("**RTA ✓**")
            if task.config.ab_test.enabled:
                badges.append("**A/B Test ✓**")

            if badges:
                st.markdown(" | ".join(badges))

        with col2:
            if st.button("⬅️ Back", use_container_width=True):
                del st.session_state.selected_task_id
                st.rerun()

        st.divider()

        # Динамические табы в зависимости от конфигурации
        tab_names = ["📊 Overview", "📈 Metrics"]

        if len(task.model_ids) > 1:
            tab_names.append("🔍 Model Comparison")

        if task.config.variations.enabled:
            tab_names.append("🔄 Variations Analysis")

        if task.config.rta.enabled:
            tab_names.append("🛑 RTA Analysis")

        if task.config.ab_test.enabled:
            tab_names.append("🧪 A/B Test Results")

        tab_names.append("📄 Detailed Results")

        tabs = st.tabs(tab_names)
        tab_idx = 0

        # ===== TAB: Overview =====
        with tabs[tab_idx]:
            _render_overview_tab(task, models, api_client)
        tab_idx += 1

        # ===== TAB: Metrics =====
        with tabs[tab_idx]:
            _render_metrics_tab(task, models)
        tab_idx += 1

        # ===== TAB: Model Comparison (если > 1 модели) =====
        if len(task.model_ids) > 1:
            with tabs[tab_idx]:
                _render_model_comparison_tab(task, models, api_client)
            tab_idx += 1

        # ===== TAB: Variations Analysis (если включены) =====
        if task.config.variations.enabled:
            with tabs[tab_idx]:
                _render_variations_tab(task, models)
            tab_idx += 1

        # ===== TAB: RTA Analysis (если включен) =====
        if task.config.rta.enabled:
            with tabs[tab_idx]:
                _render_rta_tab(task, models)
            tab_idx += 1

        # ===== TAB: A/B Test Results (если включен) =====
        if task.config.ab_test.enabled:
            with tabs[tab_idx]:
                _render_ab_test_tab(task, models)
            tab_idx += 1

        # ===== TAB: Detailed Results =====
        with tabs[tab_idx]:
            _render_detailed_results_tab(task, models)

    except Exception as e:
        st.error(f"❌ Error loading task results: {e}")
        import traceback

        st.code(traceback.format_exc())


def _render_overview_tab(task, models, api_client):
    """Вкладка обзора"""
    st.markdown("### Task Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📝 Task Details**")
        st.write(f"**Type:** {task.task_type.replace('_', ' ').title()}")
        st.write(f"**Status:** {task.status.upper()}")
        st.write(f"**Created:** {task.created_at.strftime('%Y-%m-%d %H:%M')}")

    with col2:
        st.markdown("**📊 Dataset**")
        try:
            dataset = api_client.get_dataset(task.dataset_id)
            if dataset:
                st.write(f"**Name:** {dataset.name}")
                st.write(f"**Size:** {dataset.size} items")
                st.write(f"**Task Type:** {dataset.task_type}")
        except:
            st.write(f"**ID:** {task.dataset_id[:12]}...")

    with col3:
        st.markdown("**🤖 Models**")
        model_names = []
        for model_id in task.model_ids:
            model = next((m for m in models if m.id == model_id), None)
            model_names.append(model.name if model else model_id[:12] + "...")

        for name in model_names:
            st.write(f"• {name}")

    st.divider()

    # Прогресс
    if task.status in ["running", "completed"]:
        st.markdown("### Execution Progress")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Samples", task.total_samples)
        col2.metric("Processed", task.processed_samples)
        col3.metric("Progress", f"{task.progress:.0f}%")

        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            col4.metric("Duration", f"{duration:.1f}s")

        if task.status == "running":
            st.progress(task.progress / 100)

    # Ошибки
    if task.error:
        st.error(f"**Error:** {task.error}")


def _render_metrics_tab(task, models):
    """Вкладка метрик"""
    st.markdown("### 📊 Performance Metrics")

    if not task.aggregated_metrics:
        st.info("No metrics available yet")
        return

    # Группируем по моделям
    for model_id, model_metrics in task.aggregated_metrics.items():
        model = next((m for m in models if m.id == model_id), None)
        model_name = model.name if model else model_id[:12] + "..."

        with st.expander(f"📦 {model_name}", expanded=True):
            # Разделяем метрики по категориям
            standard_metrics = {}
            special_metrics = {}

            for metric_name, value in model_metrics.items():
                if metric_name in [
                    "include_exclude_score",
                    "include_success_rate",
                    "exclude_violation_rate",
                ]:
                    special_metrics[metric_name] = value
                elif metric_name in [
                    "refusal_rate",
                    "explicit_refusals",
                    "implicit_refusals",
                ]:
                    special_metrics[metric_name] = value
                else:
                    standard_metrics[metric_name] = value

            # Стандартные метрики
            if standard_metrics:
                st.markdown("**Standard Metrics**")
                cols = st.columns(min(len(standard_metrics), 4))
                for i, (metric, value) in enumerate(standard_metrics.items()):
                    cols[i % 4].metric(
                        metric.replace("_", " ").title(),
                        f"{value:.2f}%" if value < 200 else f"{value:.2f}",
                    )

            # Специальные метрики
            if special_metrics:
                st.markdown("**Special Metrics**")
                cols = st.columns(min(len(special_metrics), 4))
                for i, (metric, value) in enumerate(special_metrics.items()):
                    cols[i % 4].metric(
                        metric.replace("_", " ").title(),
                        f"{value:.2f}%"
                        if "rate" in metric or "score" in metric
                        else f"{value:.0f}",
                    )

    # Сравнительные графики
    if len(task.aggregated_metrics) > 1:
        st.divider()
        st.markdown("### Comparative Analysis")
        _plot_comparative_metrics(task, models)


def _render_model_comparison_tab(task, models, api_client):
    """Вкладка сравнения моделей"""
    st.markdown("### 🏆 Model Performance Comparison")

    try:
        comparison = api_client.compare_models(task.id)

        if not comparison:
            st.warning("Comparison data not available")
            return

        # Победитель
        if "best_model" in comparison:
            best = comparison["best_model"]
            best_model = next((m for m in models if m.id == best["model_id"]), None)
            best_name = best_model.name if best_model else best["model_id"][:12]

            st.success(
                f"🥇 **Best Model:** {best_name} — {best['reason'].replace('_', ' ').title()} (Score: {best['score']:.2f})"
            )

        st.divider()

        # Таблица сравнения
        st.markdown("### Detailed Comparison")

        comparison_data = []
        for model_id, stats in comparison["models"].items():
            model = next((m for m in models if m.id == model_id), None)
            model_name = model.name if model else model_id[:12] + "..."

            row = {
                "Model": model_name,
                "Results": stats["total_results"],
                "Avg Time (s)": f"{stats['avg_execution_time']:.3f}",
            }

            # Метрики
            for metric, value in stats.get("metrics", {}).items():
                row[metric.replace("_", " ").title()] = f"{value:.2f}"

            # Judge score
            if "avg_judge_score" in stats:
                row["Judge Score"] = f"{stats['avg_judge_score']:.2f}/10"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        st.divider()

        # Spider chart для сравнения моделей
        st.markdown("### Spider Chart Comparison")
        plot_multi_model_comparison_spider(
            task.results,
            list(comparison["models"].keys()),
            task.config.evaluation_metrics,
        )

    except Exception as e:
        st.error(f"Error loading comparison: {e}")


def _render_variations_tab(task, models):
    """Вкладка анализа вариаций"""
    st.markdown("### 🔄 Variations Analysis")

    # Выбор модели для анализа
    model_options = {}
    for model_id in task.model_ids:
        model = next((m for m in models if m.id == model_id), None)
        model_name = model.name if model else model_id[:12] + "..."
        model_options[model_name] = model_id

    selected_model_name = st.selectbox(
        "Select Model for Analysis", list(model_options.keys())
    )
    selected_model_id = model_options[selected_model_name]

    st.divider()

    # Spider chart
    st.markdown("### Spider Chart - Performance by Variation")
    plot_variation_spider_chart(
        task.results, selected_model_id, task.config.evaluation_metrics
    )

    st.divider()

    # Детальная статистика по вариациям
    st.markdown("### Detailed Statistics")

    variation_stats = defaultdict(lambda: defaultdict(list))

    for result in task.results:
        if result.model_id == selected_model_id:
            var_type = result.variation_type or "original"

            variation_stats[var_type]["execution_time"].append(result.execution_time)

            if result.judge_score:
                variation_stats[var_type]["judge_score"].append(result.judge_score)

            if result.include_score is not None:
                variation_stats[var_type]["include_score"].append(result.include_score)

    # Таблица статистики
    stats_data = []
    for var_type, stats in variation_stats.items():
        row = {"Variation": var_type.replace("_", " ").title()}
        row["Count"] = len(stats["execution_time"])
        row["Avg Time (s)"] = (
            f"{sum(stats['execution_time']) / len(stats['execution_time']):.3f}"
        )

        if stats["judge_score"]:
            row["Avg Judge Score"] = (
                f"{sum(stats['judge_score']) / len(stats['judge_score']):.2f}"
            )

        if stats["include_score"]:
            row["Avg Include Score"] = (
                f"{sum(stats['include_score']) / len(stats['include_score']):.2f}"
            )

        stats_data.append(row)

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, use_container_width=True)


def _render_rta_tab(task, models):
    """Вкладка RTA анализа"""
    st.markdown("### 🛑 Refuse-to-Answer Analysis")

    # Группируем по моделям
    for model_id in task.model_ids:
        model = next((m for m in models if m.id == model_id), None)
        model_name = model.name if model else model_id[:12] + "..."

        model_results = [r for r in task.results if r.model_id == model_id]

        with st.expander(f"📦 {model_name}", expanded=True):
            refused_count = sum(1 for r in model_results if r.refused)
            total = len(model_results)
            refusal_rate = (refused_count / total * 100) if total > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Responses", total)
            col2.metric("Refusals", refused_count)
            col3.metric("Refusal Rate", f"{refusal_rate:.1f}%")

            # Примеры отказов
            if refused_count > 0:
                st.markdown("**Refusal Examples:**")
                refused_examples = [r for r in model_results if r.refused][:5]

                for i, result in enumerate(refused_examples, 1):
                    with st.expander(f"Example {i}: {result.input[:50]}..."):
                        st.markdown("**Input:**")
                        st.code(result.input, language=None)
                        st.markdown("**Output:**")
                        st.code(result.output, language=None)
                        if "rta_reasoning" in result.metadata:
                            st.markdown("**RTA Reasoning:**")
                            st.info(result.metadata["rta_reasoning"])


def _render_ab_test_tab(task, models):
    """Вкладка A/B тестов"""
    st.markdown("### 🧪 A/B Test Results")

    if not task.ab_test_results:
        st.info("A/B test results not available yet")
        return

    ab_results = task.ab_test_results

    # Победитель
    if "winner" in ab_results and ab_results["winner"].get("variant"):
        winner = ab_results["winner"]
        st.success(
            f"🏆 **Winner:** {winner['variant']} — {winner['reason']} (Confidence: {winner['confidence']})"
        )
    else:
        st.warning("No clear winner found")

    st.divider()

    # Статистика по вариантам
    st.markdown("### Variant Statistics")

    variant_data = []
    for variant, metrics in ab_results.get("variant_metrics", {}).items():
        row = {"Variant": variant}
        row.update(
            {k: f"{v:.2f}" for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        variant_data.append(row)

    if variant_data:
        df_variants = pd.DataFrame(variant_data)
        st.dataframe(df_variants, use_container_width=True)

    st.divider()

    # Статистические тесты
    st.markdown("### Statistical Tests")

    tests = ab_results.get("statistical_tests", {})

    if tests:
        for test_name, test_result in tests.items():
            if test_result.get("significant"):
                badge = "✅ Significant"
                color = "green"
            else:
                badge = "❌ Not Significant"
                color = "red"

            with st.expander(f"{test_name} - {badge}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**P-Value:** {test_result.get('p_value', 'N/A')}")
                    st.write(f"**Test:** {test_result.get('test', 'N/A')}")

                with col2:
                    if "mean_a" in test_result:
                        st.write(f"**Mean A:** {test_result['mean_a']:.2f}")
                        st.write(f"**Mean B:** {test_result['mean_b']:.2f}")
                    elif "median_a" in test_result:
                        st.write(f"**Median A:** {test_result['median_a']:.2f}")
                        st.write(f"**Median B:** {test_result['median_b']:.2f}")


def _render_detailed_results_tab(task, models):
    """Вкладка детальных результатов"""
    st.markdown("### 📄 Detailed Results")

    # Фильтры
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Фильтр по модели
        model_options = {"All": None}
        for model_id in task.model_ids:
            model = next((m for m in models if m.id == model_id), None)
            model_name = model.name if model else model_id[:12] + "..."
            model_options[model_name] = model_id

        selected_model_name = st.selectbox(
            "Model", list(model_options.keys()), key="filter_model"
        )
        selected_model = model_options[selected_model_name]

    with col2:
        # Фильтр по вариации
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
    filtered_results = task.results[: limit * 10]  # Берем больше для фильтрации

    if selected_model:
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

    filtered_results = filtered_results[:limit]

    # Отображение
    for i, result in enumerate(filtered_results, 1):
        # Определяем модель
        model = next((m for m in models if m.id == result.model_id), None)
        model_name = model.name if model else result.model_id[:12] + "..."

        # Формируем заголовок
        title_parts = [f"**Result {i}**", f"[{model_name}]"]

        if result.variation_type:
            title_parts.append(f"*{result.variation_type}*")

        if result.judge_score:
            title_parts.append(f"⭐ {result.judge_score:.1f}/10")

        if result.refused:
            title_parts.append("🛑 REFUSED")

        title = " — ".join(title_parts)

        with st.expander(f"{title}: {result.input[:50]}...", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
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
                        with st.expander("Judge Reasoning"):
                            st.write(result.judge_reasoning)

                if result.include_score is not None:
                    st.markdown(f"**📊 Include Score:** {result.include_score:.2f}")

                if result.refused:
                    st.warning("🛑 Model refused to answer")


def _plot_comparative_metrics(task, models):
    """График сравнительных метрик"""
    data = []

    for model_id, metrics in task.aggregated_metrics.items():
        model = next((m for m in models if m.id == model_id), None)
        model_name = model.name if model else model_id[:12] + "..."

        for metric, value in metrics.items():
            data.append(
                {
                    "Model": model_name,
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
            title="Metrics Comparison",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, config=plotly_config)


def _render_results_overview(api_client):
    """Обзор результатов (список задач)"""
    st.markdown("### 📊 Select a Completed Task")

    try:
        completed_tasks = api_client.list_tasks(status=TaskStatus.COMPLETED)
        models = api_client.list_models()

        if completed_tasks:
            for task in completed_tasks[:10]:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.markdown(f"**{task.name}**")
                        st.caption(
                            f"{task.task_type.replace('_', ' ').title()} — {task.created_at.strftime('%Y-%m-%d %H:%M')}"
                        )

                    with col2:
                        # Показываем модели
                        model_names = []
                        for model_id in task.model_ids[:2]:
                            model = next((m for m in models if m.id == model_id), None)
                            model_names.append(model.name if model else model_id[:8])

                        st.write(
                            ", ".join(model_names)
                            + ("..." if len(task.model_ids) > 2 else "")
                        )

                    with col3:
                        if st.button("📊 View", key=f"view_{task.id}"):
                            st.session_state.selected_task_id = task.id
                            st.rerun()

                    st.divider()
        else:
            st.info("📭 No completed tasks yet")

    except Exception as e:
        st.error(f"Error: {e}")
