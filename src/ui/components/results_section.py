# src/ui/components/results_section.py
from collections import defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.schemas.task import TaskStatus
from src.ui.api_client import get_api_client
from src.ui.components.spider_chart_variations import (
    create_task_metrics_table,
    get_model_name,
    plot_augmentation_impact_chart,
    plot_model_centric_spider,
    plot_rta_spider_chart,
    plot_task_centric_spider,
)

plotly_config = dict(width="stretch")


def render_results_section():
    """Рендер секции результатов с правильной навигацией"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📈 Results & Analytics")

    api_client = get_api_client()

    # Всегда показываем выбор задачи
    _render_task_selector(api_client)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_task_selector(api_client):
    """Выбор задачи для анализа с правильной навигацией"""

    try:
        # Получаем все завершенные задачи
        completed_tasks = api_client.list_tasks(status=TaskStatus.COMPLETED)
        models = api_client.list_models()

        if not completed_tasks:
            st.info("📭 No completed tasks yet")
            return

        preselected_task_id = st.session_state.get("selected_task_id")

        # Создаем словарь задач
        task_options = {}
        preselected_index = 0

        for idx, task in enumerate(completed_tasks):
            task_label = (
                f"{task.name} ({task.task_type.replace('_', ' ').title()}) [{task.id}]"
            )
            task_options[task_label] = task

            # Находим индекс предвыбранной задачи
            if preselected_task_id and task.id == preselected_task_id:
                preselected_index = idx

        # Селектор с предвыбранной задачей
        selected_task_label = st.selectbox(
            "Select Task for Analysis",
            list(task_options.keys()),
            index=preselected_index,
            key="task_selector_results",
        )

        selected_task = task_options[selected_task_label]

        st.session_state.selected_task_id = selected_task.id

        st.divider()

        # Показываем результаты выбранной задачи
        _render_task_results_with_tabs(selected_task, models, api_client)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback

        st.code(traceback.format_exc())


def _render_task_results_with_tabs(task, models, api_client):
    """Отображение результатов задачи с фиксированными табами"""

    # Заголовок
    st.markdown(f"## 📊 {task.name}")
    st.caption(f"Task ID: {task.id}")

    # Информация о задаче
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

    st.divider()

    # Фиксированные табы (всегда все показываем)
    tabs = st.tabs(
        [
            "📊 Overview",
            "📈 Metrics",
            "🔍 Task-Centric Analysis",
            "🤖 Model-Centric Analysis",
            "🛑 RTA Analysis",
            "🧪 A/B Test Results",
            "📄 Detailed Results",
        ]
    )

    # ===== TAB 0: Overview =====
    with tabs[0]:
        _render_overview_tab(task, models, api_client)

    # ===== TAB 1: Metrics =====
    with tabs[1]:
        _render_metrics_tab(task, models)

    # ===== TAB 2: Task-Centric Analysis =====
    with tabs[2]:
        _render_task_centric_tab(task, models)

    # ===== TAB 3: Model-Centric Analysis =====
    with tabs[3]:
        _render_model_centric_tab(task, models, api_client)

    # ===== TAB 4: RTA Analysis =====
    with tabs[4]:
        _render_rta_tab(task, models)

    # ===== TAB 5: A/B Test Results =====
    with tabs[5]:
        _render_ab_test_tab(task, models)

    # ===== TAB 6: Detailed Results =====
    with tabs[6]:
        _render_detailed_results_tab(task, models)


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
        for model_id in task.model_ids:
            model_name = get_model_name(model_id, models)
            st.write(f"• {model_name}")

    st.divider()

    # Прогресс
    if task.status in ["running", "completed"]:
        st.markdown("### Execution Progress")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Samples", task.total_samples)
        col2.metric("Processed", task.processed_samples)
        col3.metric("Progress", f"{task.progress:.0f}%")

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
        model_name = get_model_name(model_id, models)

        with st.expander(f"📦 {model_name}", expanded=True):
            # Разделяем метрики по категориям
            standard_metrics = {}
            special_metrics = {}

            for metric_name, value in model_metrics.items():
                # Пропускаем execution_time
                if metric_name == "execution_time":
                    continue

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


def _render_task_centric_tab(task, models):
    """Task-centric вкладка анализа вариаций"""

    if not task.config.variations.enabled:
        st.info("Variations are not enabled for this task")
        return

    st.markdown("### 🔄 Task-Centric Variations Analysis")
    st.caption(
        "Axes: Variations | Colors: Models | Metrics: Dispersion indices (TSI, CV, IQR-CV, JSD)"
    )

    task_display_name = task.name

    st.divider()

    # Spider charts с метриками дисперсии
    st.markdown("### 🕸️ Dispersion Metrics Across Variations")
    st.caption("Lower values indicate more stable performance across variations")

    plot_task_centric_spider(
        task,
        models,
    )

    st.divider()

    # График влияния вариаций
    st.markdown("### 📊 Impact of Variations on Metrics")
    plot_augmentation_impact_chart(
        task.results,
        task_display_name,
        task.model_ids,
        models,
    )

    st.divider()

    # Таблица с метриками дисперсии
    st.markdown("### 📈 Variation Stability Metrics")

    variation_stats = defaultdict(lambda: defaultdict(list))

    for result in task.results:
        var_type = result.variation_type or "original"

        if result.judge_score is not None:
            variation_stats[var_type]["judge_score"].append(result.judge_score)
        if result.include_score is not None:
            variation_stats[var_type]["include_score"].append(result.include_score)

    # Создаем таблицу со статистикой
    # stats_rows = []
    # for var_type, stats in variation_stats.items():
    #     # Collect all values for dispersion calculation
    #     all_values = []
    #     for values_list in stats.values():
    #         all_values.extend(values_list)

    #     if not all_values:
    #         continue
    #     # TODO: this table

    # dispersion = compute_dispersion_indices(all_values)

    # row = {
    #     "Variation": var_type.replace("_", " ").title(),
    #     "Count": len(all_values),
    #     "Mean": f"{np.mean(all_values):.2f}",
    #     "TSI (%)": f"{dispersion['tsi']:.2f}",
    #     "Corrected CV (%)": f"{dispersion['cv_corrected']:.2f}",
    #     "IQR-CV (%)": f"{dispersion['iqr_cv']:.2f}",
    #     "JSD": f"{dispersion['jsd']:.4f}",
    # }

    # stats_rows.append(row)

    # if stats_rows:
    #     df_stats = pd.DataFrame(stats_rows)
    #     st.dataframe(df_stats, use_container_width=True)

    # Interpretation guide
    with st.expander("📚 Metric Interpretation Guide"):
        st.markdown("""
        **Task Sensitivity Index (TSI) / Coefficient of Variation:**
        - TSI < 10%: Very stable variation
        - TSI 10-20%: Stable variation
        - TSI 20-30%: Moderately stable
        - TSI > 30%: Unstable variation
        
        **Corrected CV:**
        - Adjusted for small sample bias using Everitt's correction
        - More accurate for small datasets
        - Interpretation same as TSI
        
        **IQR-CV (Interquartile Range CV):**
        - Based on interquartile range, robust to outliers
        - Good for non-normal distributions
        - Lower values = more consistent performance
        
        **Jensen-Shannon Divergence (JSD):**
        - Measures distribution heterogeneity (0-1 scale, shown as %)
        - Lower values indicate more uniform performance
        - JSD < 0.1: Very uniform
        - JSD > 0.3: High variability
        
        **Spider Chart Interpretation:**
        - Each axis represents a different variation type
        - Each colored line represents a different model
        - Smaller area = more stable (lower dispersion)
        - Compare shapes to see which model is most consistent
        """)


def _render_model_centric_tab(task, models, api_client):
    """Model-centric вкладка сравнения моделей"""

    st.markdown("### 🏆 Model-Centric Performance Comparison")
    st.caption(
        "Axes: Variations | Colors: Tasks | Metrics: Dispersion indices (TSI, CV, IQR-CV, JSD)"
    )

    # Выбор модели для анализа
    model_options = {}
    for model_id in task.model_ids:
        model_name = get_model_name(model_id, models)
        model_options[model_name] = model_id

    selected_model_name = st.selectbox(
        "Select Model for Analysis",
        list(model_options.keys()),
        key="model_centric_model_select",
    )
    selected_model_id = model_options[selected_model_name]

    st.divider()

    # Получаем все доступные задачи
    # В реальной системе это должны быть разные задачи из API
    # Пока используем текущую задачу как пример
    try:
        all_tasks = api_client.list_tasks(status=TaskStatus.COMPLETED)

        # Создаем словарь: task_name -> results
        all_tasks_results = {}
        available_task_names = []

        for t in all_tasks:
            # Проверяем, что у задачи есть результаты для выбранной модели
            model_results = [r for r in t.results if r.model_id == selected_model_id]
            if model_results and t.config.variations.enabled:
                all_tasks_results[t.name] = t.results
                available_task_names.append(t.name)

        if not available_task_names:
            st.warning("No tasks with variations found for this model")
            return

        # Мультиселект для выбора задач
        st.markdown("### 📋 Select Tasks for Comparison")
        selected_task_names = st.multiselect(
            "Choose tasks to display on spider chart:",
            options=available_task_names,
            default=available_task_names[
                : min(3, len(available_task_names))
            ],  # По умолчанию первые 3
            key="model_centric_task_select",
            help="Select multiple tasks to compare how the model performs across different tasks",
        )

        if not selected_task_names:
            st.info("Please select at least one task to display charts")
            return

        st.divider()

        # Spider charts для модели по разным задачам
        st.markdown("### 🕸️ Model Performance Across Tasks")
        st.caption("Lower values indicate more stable performance")

        plot_model_centric_spider(
            all_tasks,
            selected_model_id,
            models,
            selected_task_names,
        )

        # Таблица с задачами и метриками
        st.markdown("### 📊 Task Dispersion Metrics Table")

        task_metrics_df = create_task_metrics_table(
            all_tasks_results,
            selected_model_id,
            selected_task_names,
        )

        if not task_metrics_df.empty:
            st.dataframe(task_metrics_df, use_container_width=True)

            # Ranking by stability
            st.markdown("### 🏆 Task Stability Ranking")

            ranking_metric = st.selectbox(
                "Rank by:",
                ["TSI", "IQR-CV", "JSD"],
                key="task_ranking_metric",
            )

            metric_col_map = {
                "TSI": "TSI (%)",
                "IQR-CV": "IQR-CV (%)",
                "JSD": "JSD",
            }

            col_name = metric_col_map[ranking_metric]

            # Filter out NaN values and sort
            df_sorted = task_metrics_df.copy()
            df_sorted = df_sorted.dropna(subset=[col_name])

            if df_sorted.empty:
                st.warning("No valid data available for ranking")
            else:
                # Sort by selected metric (values are already numeric)
                df_sorted = df_sorted.sort_values(col_name)

                for _, row in df_sorted.iterrows():
                    metric_value = row[col_name]
                    task_name = row["Task"]

                    # Determine stability level
                    if ranking_metric in ["TSI", "IQR-CV"]:
                        if metric_value < 10:
                            badge = "🟢 Very Stable"
                        elif metric_value < 20:
                            badge = "🟡 Stable"
                        elif metric_value < 30:
                            badge = "🟠 Moderately Stable"
                        else:
                            badge = "🔴 Unstable"
                    else:  # JSD
                        if metric_value < 0.1:
                            badge = "🟢 Very Stable"
                        elif metric_value < 0.2:
                            badge = "🟡 Stable"
                        elif metric_value < 0.3:
                            badge = "🟠 Moderately Stable"
                        else:
                            badge = "🔴 Unstable"

                    st.write(
                        f"**{task_name}**: {ranking_metric} = {metric_value:.2f} — {badge}"
                    )
        else:
            st.info("No task metrics available")

        st.divider()

        # Сравнительная статистика
        comparison = api_client.compare_models(task.id)

        if comparison:
            # Победитель
            if "best_model" in comparison:
                best = comparison["best_model"]
                best_name = get_model_name(best["model_id"], models)

                st.success(
                    f"🥇 **Best Model (Current Task):** {best_name} — {best['reason'].replace('_', ' ').title()} (Score: {best['score']:.2f})"
                )

            st.divider()

            # Детальное сравнение
            st.markdown("### 📊 Detailed Model Comparison (Current Task)")

            comparison_data = []
            for model_id, stats in comparison["models"].items():
                model_name = get_model_name(model_id, models)

                row = {
                    "Model": model_name,
                    "Results": stats["total_results"],
                }

                for metric, value in stats.get("metrics", []).items():
                    if metric != "execution_time":
                        row[metric.replace("_", " ").title()] = f"{value:.2f}"

                if "avg_judge_score" in stats:
                    row["Judge Score"] = f"{stats['avg_judge_score']:.2f}/10"

                comparison_data.append(row)

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading model-centric analysis: {e}")
        import traceback

        st.code(traceback.format_exc())


def _render_rta_tab(task, models):
    """Вкладка RTA анализа с spider chart"""

    if not task.config.rta.enabled:
        st.info("RTA (Refuse-to-Answer) is not enabled for this task")
        return

    st.markdown("### 🛑 Refuse-to-Answer Analysis")

    # Spider chart в начале (если есть вариации)
    if task.config.variations.enabled:
        variations = set(r.variation_type for r in task.results if r.variation_type)
        if len(variations) >= 3 or (
            len(variations) >= 2 and any(not r.variation_type for r in task.results)
        ):
            plot_rta_spider_chart(task, models)
            st.divider()

    # Общая статистика по моделям
    st.markdown("### 📊 Model-wise Refusal Statistics")

    for model_id in task.model_ids:
        model_name = get_model_name(model_id, models)
        model_results = [r for r in task.results if r.model_id == model_id]

        with st.expander(f"📦 {model_name}", expanded=True):
            refused_count = sum(1 for r in model_results if r.refused)
            total = len(model_results)
            refusal_rate = (refused_count / total * 100) if total > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Responses", total)
            col2.metric("Refusals", refused_count)
            col3.metric("Refusal Rate", f"{refusal_rate:.1f}%")

            # Если есть вариации, показываем разбивку по вариациям
            if task.config.variations.enabled:
                st.markdown("**Refusal Rate by Variation:**")

                variation_stats = {}

                # Original (без вариации)
                original_results = [r for r in model_results if not r.variation_type]
                if original_results:
                    orig_refused = sum(1 for r in original_results if r.refused)
                    orig_rate = (orig_refused / len(original_results)) * 100
                    variation_stats["Original"] = {
                        "total": len(original_results),
                        "refused": orig_refused,
                        "rate": orig_rate,
                    }

                # Вариации
                variations = set(
                    r.variation_type for r in model_results if r.variation_type
                )
                for variation in sorted(variations):
                    var_results = [
                        r for r in model_results if r.variation_type == variation
                    ]
                    var_refused = sum(1 for r in var_results if r.refused)
                    var_rate = (var_refused / len(var_results)) * 100
                    variation_stats[variation] = {
                        "total": len(var_results),
                        "refused": var_refused,
                        "rate": var_rate,
                    }

                # Показываем в колонках
                if variation_stats:
                    var_cols = st.columns(min(len(variation_stats), 4))
                    for idx, (var_name, stats) in enumerate(variation_stats.items()):
                        with var_cols[idx % 4]:
                            st.metric(
                                var_name.replace("_", " ").title(),
                                f"{stats['rate']:.1f}%",
                                f"{stats['refused']}/{stats['total']}",
                            )

            # Примеры отказов
            if refused_count > 0:
                st.markdown("**Refusal Examples:**")
                refused_examples = [r for r in model_results if r.refused][:5]

                for i, result in enumerate(refused_examples, 1):
                    variation_label = ""
                    if result.variation_type:
                        variation_label = f" [{result.variation_type}]"

                    with st.expander(
                        f"Example {i}{variation_label}: {result.input[:50]}..."
                    ):
                        st.markdown("**Input:**")
                        st.code(result.input, language=None)
                        st.markdown("**Output:**")
                        st.code(result.output, language=None)
                        if "rta_reasoning" in result.metadata:
                            st.markdown("**RTA Reasoning:**")
                            st.info(result.metadata["rta_reasoning"])


def _render_ab_test_tab(task, models):
    """Вкладка A/B тестов"""

    if not task.config.ab_test.enabled:
        st.info("A/B Testing is not enabled for this task")
        return

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
            {
                k: f"{v:.2f}"
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "execution_time"
            }
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
            else:
                badge = "❌ Not Significant"

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
    col1, col2, col3 = st.columns(3)

    with col1:
        model_options = {"All": None}
        for model_id in task.model_ids:
            model_name = get_model_name(model_id, models)
            model_options[model_name] = model_id

        selected_model_name = st.selectbox(
            "Model", list(model_options.keys()), key="filter_model"
        )
        selected_model = model_options[selected_model_name]

    with col2:
        variation_types = ["All"] + list(
            set(r.variation_type for r in task.results if r.variation_type)
        )
        selected_variation = st.selectbox(
            "Variation", variation_types, key="filter_variation"
        )

    with col3:
        search = st.text_input("🔍 Search", key="search_results")

    limit = st.number_input("Limit", 10, 100, 20, key="limit_results")

    # Фильтрация
    filtered_results = task.results[: limit * 10]

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
        model_name = get_model_name(result.model_id, models)

        title_parts = [f"**Result {i}**", f"[{model_name}]"]

        if result.variation_type:
            title_parts.append(f"*{result.variation_type}*")

        if result.judge_score:
            title_parts.append(f"⭐ {result.judge_score:.1f}/10")

        if result.refused == "1":
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

                if result.judge_score:
                    st.markdown(f"**🎯 Judge Score:** {result.judge_score:.2f}/10")
                    if result.judge_reasoning:
                        with st.expander("Judge Reasoning"):
                            st.write(result.judge_reasoning)

                if result.include_score is not None:
                    st.markdown(f"**📊 Include Score:** {result.include_score:.2f}")

                if result.refused == "1":
                    st.warning("🛑 Model refused to answer")


def _plot_comparative_metrics(task, models):
    """График сравнительных метрик"""
    data = []

    for model_id, metrics in task.aggregated_metrics.items():
        model_name = get_model_name(model_id, models)

        for metric, value in metrics.items():
            # Пропускаем execution_time
            if metric != "execution_time":
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


# def _render_model_centric_tab_ranking_fix(task_metrics_df):
#     """Исправленная версия Task Stability Ranking"""

#     st.markdown("### 🏆 Task Stability Ranking")

#     ranking_metric = st.selectbox(
#         "Rank by:",
#         ["TSI", "IQR-CV", "JSD"],
#         key="task_ranking_metric",
#     )

#     metric_col_map = {
#         "TSI": "TSI (%)",
#         "IQR-CV": "IQR-CV (%)",
#         "JSD": "JSD",
#     }

#     col_name = metric_col_map[ranking_metric]

#     # Удаляем строки с NaN перед сортировкой
#     df_sorted = task_metrics_df.copy()
#     df_sorted = df_sorted.dropna(subset=[col_name])

#     if df_sorted.empty:
#         st.warning("No valid data available for ranking")
#         return

#     # Сортируем (значения уже числовые, не строки)
#     df_sorted = df_sorted.sort_values(col_name)

#     for _, row in df_sorted.iterrows():
#         metric_value = row[col_name]
#         task_name = row["Task"]

#         # Determine stability level
#         if ranking_metric in ["TSI", "IQR-CV"]:
#             if metric_value < 10:
#                 badge = "🟢 Very Stable"
#             elif metric_value < 20:
#                 badge = "🟡 Stable"
#             elif metric_value < 30:
#                 badge = "🟠 Moderately Stable"
#             else:
#                 badge = "🔴 Unstable"
#         else:  # JSD
#             if metric_value < 0.1:
#                 badge = "🟢 Very Stable"
#             elif metric_value < 0.2:
#                 badge = "🟡 Stable"
#             elif metric_value < 0.3:
#                 badge = "🟠 Moderately Stable"
#             else:
#                 badge = "🔴 Unstable"

#         st.write(f"**{task_name}**: {ranking_metric} = {metric_value:.2f} — {badge}")
