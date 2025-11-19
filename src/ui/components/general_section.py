# src/ui/components/general_section.py
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.ui.api_client import get_api_client


def render_general_section():
    """Рендер секции General без дублирования контента"""

    # ВАЖНО: Используем уникальный контейнер
    container_key = "general_section_container"

    # Очищаем предыдущее содержимое (если есть)
    if container_key in st.session_state:
        del st.session_state[container_key]

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📊 General Overview")

    api_client = get_api_client()

    # Tabs с уникальными ключами
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

            # НОВАЯ СЕКЦИЯ: Детальный мониторинг активных задач
            st.markdown("### ⚡ Active Tasks Monitor")

            running_tasks = [t for t in tasks if t.status == "running"]

            if running_tasks:
                for task in running_tasks:
                    with st.expander(
                        f"🔄 {task.name} - {task.processed_samples}/{task.total_samples} inferences",
                        expanded=True,
                    ):
                        # Прогресс BAR
                        progress_pct = (
                            (task.processed_samples / task.total_samples * 100)
                            if task.total_samples > 0
                            else 0
                        )
                        st.progress(progress_pct / 100)

                        col1, col2 = st.columns([2, 1])

                        # Определяем иконку для типа операции
                        operation_icons = {
                            "standard": "📝",
                            "variation": "🔄",
                            "ab_test": "🧪",
                            "judge_evaluation": "⚖️",
                            "rta_evaluation": "🛑",
                        }

                        with col1:
                            # Статистика
                            st.caption(
                                f"**Progress:** {task.processed_samples}/{task.total_samples} inferences ({progress_pct:.1f}%)"
                            )
                            st.caption(
                                f"**Task Type:** {task.task_type.replace('_', ' ').title()}"
                            )

                            # Текущее выполнение
                            if task.current_execution:
                                st.markdown("**🔵 Currently Processing:**")

                                exec_data = task.current_execution

                                operation_type = exec_data.get(
                                    "operation_type", "standard"
                                )
                                icon = operation_icons.get(operation_type, "📝")

                                with st.container(border=True):
                                    # Заголовок с типом операции
                                    st.markdown(
                                        f"**{icon} {operation_type.replace('_', ' ').title()}**"
                                    )

                                    # Детали
                                    col_a, col_b = st.columns(2)

                                    with col_a:
                                        st.caption(
                                            f"**Item Index:** {exec_data.get('index', 'N/A')}"
                                        )
                                        st.caption(
                                            f"**Model:** {exec_data.get('model_name', 'N/A')}"
                                        )

                                        # Вариация (если есть)
                                        variation = exec_data.get(
                                            "prompt_variation", "original"
                                        )
                                        if variation != "original":
                                            st.caption(f"**Variation:** {variation}")

                                    with col_b:
                                        if "started_at" in exec_data:
                                            try:
                                                started = datetime.fromisoformat(
                                                    exec_data["started_at"]
                                                )
                                                duration = (
                                                    datetime.utcnow() - started
                                                ).total_seconds()
                                                st.caption(
                                                    f"**Duration:** {duration:.1f}s"
                                                )
                                            except:
                                                pass

                                        # Execution time (если уже завершено)
                                        if "execution_time" in exec_data:
                                            st.caption(
                                                f"**Exec Time:** {exec_data['execution_time']:.2f}s"
                                            )

                                    # Input prompt
                                    st.markdown("**Input:**")
                                    st.code(
                                        exec_data.get("prompt", "N/A"), language=None
                                    )

                                    # Output (если есть)
                                    if exec_data.get("output"):
                                        st.markdown("**Output:**")
                                        st.code(exec_data["output"], language=None)

                            # Последние 2 выполненных
                            if task.recent_executions:
                                st.markdown("**✅ Recent Completions:**")

                                for idx, recent in enumerate(
                                    task.recent_executions[:2], 1
                                ):
                                    # Определяем иконку
                                    operation_type = recent.get(
                                        "operation_type", "standard"
                                    )
                                    icon = operation_icons.get(operation_type, "📝")

                                    with st.container(border=True):
                                        # Заголовок
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.caption(
                                                f"**{icon} Item #{recent.get('index', 'N/A')}** - {recent.get('model_name', 'N/A')}"
                                            )
                                        with col_b:
                                            exec_time = recent.get("execution_time")
                                            if exec_time:
                                                st.caption(f"⏱️ {exec_time:.2f}s")

                                        # Вариация
                                        variation = recent.get(
                                            "prompt_variation", "original"
                                        )
                                        if variation != "original":
                                            st.caption(f"**Variation:** {variation}")

                                        # Input/Output
                                        col_c, col_d = st.columns(2)
                                        with col_c:
                                            st.caption("**Input:**")
                                            st.text(
                                                recent.get("prompt", "N/A")[:80] + "..."
                                            )
                                        with col_d:
                                            st.caption("**Output:**")
                                            output_text = recent.get(
                                                "output", recent.get("error", "N/A")
                                            )
                                            st.text(output_text[:80] + "...")

                                        # Время завершения
                                        if "completed_at" in recent:
                                            try:
                                                completed = datetime.fromisoformat(
                                                    recent["completed_at"]
                                                )
                                                st.caption(
                                                    f"⏱️ {completed.strftime('%H:%M:%S')}"
                                                )
                                            except:
                                                pass

                        with col2:
                            # Кнопки управления
                            st.markdown("**Actions:**")

                            if st.button(
                                "⏸️ Pause",
                                key=f"pause_{task.id}",
                                use_container_width=True,
                            ):
                                try:
                                    api_client.pause_task(task.id)
                                    st.success("Task paused!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

                            if st.button(
                                "❌ Cancel",
                                key=f"cancel_dash_{task.id}",
                                use_container_width=True,
                            ):
                                confirm_key = f"confirm_cancel_dash_{task.id}"
                                if st.session_state.get(confirm_key):
                                    try:
                                        api_client.cancel_task(task.id)
                                        st.success("Cancelled!")
                                        if confirm_key in st.session_state:
                                            del st.session_state[confirm_key]
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                                else:
                                    st.session_state[confirm_key] = True
                                    st.warning("Click again")

                # Auto-refresh для running задач
                time.sleep(2)
                st.rerun()

            else:
                st.info("No active tasks running")

            # Паузированные задачи
            paused_tasks = [t for t in tasks if t.status == "paused"]

            if paused_tasks:
                st.markdown("### ⏸️ Paused Tasks")

                for task in paused_tasks:
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([3, 2, 1])

                        with col1:
                            st.write(f"**{task.name}**")
                            st.caption(
                                f"Progress: {task.processed_samples}/{task.total_samples} ({task.progress:.1f}%)"
                            )

                        with col2:
                            if task.paused_at:
                                try:
                                    paused_time = datetime.fromisoformat(
                                        str(task.paused_at)
                                    )
                                    st.caption(
                                        f"⏸️ Paused at {paused_time.strftime('%H:%M:%S')}"
                                    )
                                except:
                                    st.caption("⏸️ Paused")

                        with col3:
                            if st.button(
                                "▶️ Resume",
                                key=f"resume_{task.id}",
                                use_container_width=True,
                            ):
                                try:
                                    api_client.resume_task(task.id)
                                    st.success("Resumed!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"Error in dashboard: {e}")
            import traceback

            st.code(traceback.format_exc())

    # ===== TAB 2: All Tasks Table =====
    with tab2:
        # Устанавливаем текущий таб
        st.session_state["current_general_tab"] = "all_tasks"

        all_tasks_placeholder = st.empty()

        with all_tasks_placeholder.container():
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
                                first_metric = next(
                                    iter(first_model_metrics.items()), None
                                )
                                if first_metric:
                                    row[first_metric[0].title()] = (
                                        f"{first_metric[1]:.1f}"
                                    )

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
                        df_sorted, width="stretch", height=600, hide_index=True
                    )

                    # Кнопки действий
                    st.markdown("### Actions")
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("📊 View Results", key="view_results"):
                            st.session_state.selected_section = "results"
                            st.rerun()

                    with col2:
                        if st.button("➕ Create New Task", key="create_new_task"):
                            st.session_state.selected_section = "tasks"
                            st.rerun()

                else:
                    st.info("📭 No tasks yet. Create your first task!")

                    if st.button("➕ Create Task", key="create_new_task_else"):
                        st.session_state.selected_section = "tasks"
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading tasks: {e}")

    # ===== TAB 3: Tutorial =====
    with tab3:
        st.session_state["current_general_tab"] = "tutorial"

        tutorial_placeholder = st.empty()

        with tutorial_placeholder.container():
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
