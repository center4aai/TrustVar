# src/ui/components/general_section.py
import pandas as pd
import streamlit as st

from src.ui.api_client import get_api_client


def render_general_section():
    """Рендер секции General (обзор + туториал)"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📊 General Overview")

    api_client = get_api_client()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📋 All Tasks", "📖 Tutorial"])

    # ===== TAB 1: Dashboard =====
    with tab1:
        st.markdown("### System Overview")

        try:
            # Загружаем данные
            datasets = api_client.list_datasets()
            models = api_client.list_models()
            tasks = api_client.list_tasks()

            # Статистика
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Datasets", len(datasets))

            with col2:
                active_models = [m for m in models if m.status == "registered"]
                st.metric("🤖 Models", f"{len(active_models)}/{len(models)}")

            with col3:
                running_tasks = [t for t in tasks if t.status == "running"]
                st.metric("⚡ Running Tasks", len(running_tasks))

            with col4:
                completed_tasks = [t for t in tasks if t.status == "completed"]
                st.metric("✅ Completed", len(completed_tasks))

            st.divider()

            # Недавняя активность
            st.markdown("### 📅 Recent Activity")

            if tasks:
                recent_tasks = sorted(tasks, key=lambda x: x.created_at, reverse=True)[
                    :10
                ]

                activity_data = []
                for task in recent_tasks:
                    # Получаем имена моделей
                    model_names = []
                    for model_id in task.model_ids:
                        model = next((m for m in models if m.id == model_id), None)
                        if model:
                            model_names.append(model.name)

                    activity_data.append(
                        {
                            "Task": task.name,
                            "Type": task.task_type.replace("_", " ").title(),
                            "Models": ", ".join(model_names[:2])
                            + ("..." if len(model_names) > 2 else ""),
                            "Status": task.status.upper(),
                            "Progress": f"{task.progress:.0f}%",
                            "Created": task.created_at.strftime("%Y-%m-%d %H:%M"),
                        }
                    )

                df = pd.DataFrame(activity_data)
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("No tasks yet. Create your first task!")

            st.divider()

            # Статистика моделей
            st.markdown("### 🤖 Models Performance")

            if completed_tasks and models:
                model_stats = []

                for model in models:
                    # Находим задачи с этой моделью
                    model_tasks = [
                        t for t in completed_tasks if model.id in t.model_ids
                    ]

                    if model_tasks:
                        total_results = sum(len(t.results) for t in model_tasks)

                        # Средний judge score
                        all_judge_scores = []
                        for task in model_tasks:
                            for result in task.results:
                                if result.model_id == model.id and result.judge_score:
                                    all_judge_scores.append(result.judge_score)

                        avg_judge = (
                            sum(all_judge_scores) / len(all_judge_scores)
                            if all_judge_scores
                            else 0
                        )

                        model_stats.append(
                            {
                                "Model": model.name,
                                "Provider": model.provider,
                                "Tasks": len(model_tasks),
                                "Results": total_results,
                                "Avg Judge Score": f"{avg_judge:.2f}"
                                if avg_judge > 0
                                else "N/A",
                            }
                        )

                if model_stats:
                    df_models = pd.DataFrame(model_stats)
                    st.dataframe(df_models, use_container_width=True)
                else:
                    st.info("No completed tasks with judge scores yet")

        except Exception as e:
            st.error(f"Error loading dashboard: {e}")

    # ===== TAB 2: All Tasks Table =====
    with tab2:
        st.markdown("### 📋 All Tasks")

        try:
            tasks = api_client.list_tasks()
            models = api_client.list_models()
            datasets = api_client.list_datasets()

            if tasks:
                # Фильтры
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    status_filter = st.multiselect(
                        "Status",
                        ["pending", "running", "completed", "failed", "cancelled"],
                        default=["pending", "running", "completed"],
                        key="general_status_filter",
                    )

                with col2:
                    type_filter = st.multiselect(
                        "Task Type",
                        [
                            "standard",
                            "variation",
                            "comparison",
                            "judged",
                            "refuse_to_answer",
                            "ab_test",
                        ],
                        key="general_type_filter",
                    )

                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Refresh", key="general_refresh"):
                        st.rerun()

                # Фильтрация
                filtered_tasks = tasks
                if status_filter:
                    filtered_tasks = [
                        t for t in filtered_tasks if t.status in status_filter
                    ]
                if type_filter:
                    filtered_tasks = [
                        t for t in filtered_tasks if t.task_type in type_filter
                    ]

                # Таблица задач
                task_data = []
                for task in filtered_tasks:
                    # Получаем имена моделей
                    model_names = []
                    for model_id in task.model_ids:
                        model = next((m for m in models if m.id == model_id), None)
                        if model:
                            model_names.append(model.name)

                    # Получаем имя датасета
                    dataset = next(
                        (d for d in datasets if d.id == task.dataset_id), None
                    )
                    dataset_name = dataset.name if dataset else "Unknown"

                    # Форматируем данные
                    row = {
                        "Name": task.name,
                        "Type": task.task_type.replace("_", " ").title(),
                        "Dataset": dataset_name,
                        "Models": ", ".join(model_names),
                        "Status": task.status.upper(),
                        "Progress": f"{task.progress:.0f}%",
                        "Samples": f"{task.processed_samples}/{task.total_samples}",
                        "Created": task.created_at.strftime("%Y-%m-%d %H:%M"),
                    }

                    # Добавляем метрики для завершенных
                    if task.status == "completed" and task.aggregated_metrics:
                        # Берем первую модель и первую метрику
                        first_model_metrics = next(
                            iter(task.aggregated_metrics.values()), {}
                        )
                        if first_model_metrics:
                            first_metric = next(iter(first_model_metrics.items()), None)
                            if first_metric:
                                row[first_metric[0].title()] = f"{first_metric[1]:.1f}"

                    task_data.append(row)

                df = pd.DataFrame(task_data)

                # Сортировка
                sort_column = st.selectbox(
                    "Sort by",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index("Created"),
                    key="general_sort",
                )

                df_sorted = df.sort_values(by=sort_column, ascending=False)

                st.dataframe(
                    df_sorted, use_container_width=True, height=600, hide_index=True
                )

                # Кнопки действий
                st.markdown("### Actions")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 View Results"):
                        st.session_state.selected_section = "results"
                        st.rerun()

                with col2:
                    if st.button("➕ Create New Task"):
                        st.session_state.selected_section = "tasks"
                        st.rerun()

            else:
                st.info("📭 No tasks yet. Create your first task!")

                if st.button("➕ Create Task"):
                    st.session_state.selected_section = "tasks"
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading tasks: {e}")

    # ===== TAB 3: Tutorial =====
    with tab3:
        st.markdown("### 📖 Quick Start Guide")

        st.markdown("""
        Welcome to **TrustVar** - A Dynamic Framework for Trustworthiness Evaluation!
        
        ---
        
        #### 🚀 Getting Started
        
        **1. Upload a Dataset** 📊
        - Go to **DATASETS** section
        - Click **Upload New**
        - Choose your file (JSONL, JSON, CSV)
        - Configure columns:
          - `prompt_column`: Column with prompts/questions
          - `target_column` (optional): Column with expected answers
          - For Include/Exclude tasks: specify `include_column` and `exclude_column`
        
        **2. Register Models** 🤖
        - Go to **MODELS** section
        - Click **Register New**
        - Choose provider: Ollama, HuggingFace, OpenAI
        - Configure parameters (temperature, max_tokens, etc.)
        - For HuggingFace/Ollama: model will be downloaded automatically
        
        **3. Create a Task** ⚡
        - Go to **TASKS** section
        - Click **Create New Task**
        - Select dataset and one or more models
        - Choose task type:
          - **Standard**: Basic inference
          - **Comparison**: Compare multiple models
          - **Variation**: Test with prompt variations
          - **Judged**: Use LLM as a judge
          - **RTA**: Refuse-to-Answer analysis
          - **A/B Test**: Statistical comparison
        
        **4. Analyze Results** 📈
        - Go to **RESULTS** section
        - Select your completed task
        - View:
          - Model comparison
          - Detailed metrics
          - Variations analysis (spider charts!)
          - A/B test results
        
        ---
        
        #### 🎯 Task Types Explained
        
        **Standard** 📝
        - Basic inference on your dataset
        - Good for: Initial testing, baseline metrics
        
        **Comparison** 🔍
        - Run multiple models on same data
        - Automatic statistical comparison
        - Good for: Choosing best model
        
        **Variation** 🔄
        - Generates prompt variations (paraphrase, style change, etc.)
        - Tests model robustness
        - Spider charts for visualization
        - Good for: Testing prompt sensitivity
        
        **Judged** 🎯
        - Uses another LLM to evaluate outputs
        - Custom evaluation criteria
        - Good for: Complex evaluation scenarios
        
        **Refuse-to-Answer (RTA)** 🛑
        - Detects when models refuse to answer
        - Keyword + LLM-based detection
        - Good for: Safety and refusal testing
        
        **A/B Test** 🧪
        - Statistical comparison of variants
        - T-test or Mann-Whitney U test
        - Good for: Scientific model evaluation
        
        ---
        
        #### 💡 Pro Tips
        
        1. **Start Small**: Test with 10-50 samples first
        2. **Use Variations**: Test prompt robustness with 2-3 strategies
        3. **Enable Judge**: For subjective tasks (creativity, safety)
        4. **Monitor Progress**: Check GENERAL → Dashboard regularly
        5. **Spider Charts**: Great for visualizing variation performance
        
        ---
        
        #### 🔧 Advanced Features
        
        **Include/Exclude Lists**
        - Dataset column: `include_list` - words that must appear
        - Dataset column: `exclude_list` - words that must not appear
        - Automatic scoring
        
        **Custom Variation Prompts**
        - Select variation model
        - Write custom instruction
        - Generate unique variations
        
        **A/B Testing Strategies**
        - Prompt Variants: Test different prompts
        - Temperature Test: Find optimal temperature
        - System Prompt Test: Compare system prompts
        - Parameter Sweep: Test multiple parameters
        
        ---
        
        #### ❓ Need Help?
        
        - Check GENERAL → Dashboard for system status
        - All tasks are tracked in GENERAL → All Tasks
        - Results available in RESULTS section
        - Contact support if issues persist
        
        ---
        
        **Happy Testing!** 🎉
        """)

    st.markdown("</div>", unsafe_allow_html=True)
