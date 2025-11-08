# src/ui/components/tasks_section.py
import time

import streamlit as st

from src.config.constants import EVALUATION_METRICS
from src.core.schemas.task import TaskConfig, TaskType, VariationStrategy
from src.ui.api_client import get_api_client
from src.ui.components.task_monitor import TaskMonitor


def render_tasks_section():
    """Рендер секции управления задачами"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## ⚡ Task Management")

    api_client = get_api_client()
    task_monitor = TaskMonitor(api_client)

    # Tabs
    tab1, tab2 = st.tabs(["📋 Active Tasks", "➕ Create New Task"])

    # ===== TAB 1: Список задач =====
    with tab1:

        def on_view_task(task):
            st.session_state.selected_task_id = task.id
            st.session_state.selected_section = "results"
            st.rerun()

        def on_cancel_task(task):
            api_client.cancel_task(task.id)
            st.success(f"Task '{task.name}' cancelled!")
            st.rerun()

        task_monitor.render(
            auto_refresh=True,
            refresh_interval=2,
            on_view_click=on_view_task,
            on_cancel_click=on_cancel_task,
            show_filters=True,
            compact_mode=False,
        )

    # ===== TAB 2: Создание новой задачи =====
    with tab2:
        st.markdown("### ➕ Create New Task")

        with st.form("create_task_form", clear_on_submit=True):
            # Базовая информация
            st.markdown("#### 📝 Basic Information")
            col1, col2 = st.columns(2)

            with col1:
                task_name = st.text_input(
                    "Task Name*",
                    placeholder="e.g., Multi-Model QA Evaluation",
                    help="A descriptive name for this task",
                )

            with col2:
                task_type = st.selectbox(
                    "Task Type*",
                    options=[t.value for t in TaskType],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Type of task to perform",
                )

            st.divider()

            # Датасет и модели
            st.markdown("#### 🗂️ Dataset & Models")

            col1, col2 = st.columns(2)

            with col1:
                # Датасет
                datasets = api_client.list_datasets()
                if datasets:
                    dataset_options = {
                        f"{ds.name} ({ds.size} items)": ds.id for ds in datasets
                    }
                    selected_dataset_label = st.selectbox(
                        "Dataset*",
                        list(dataset_options.keys()),
                        help="Select the dataset to use",
                    )
                    selected_dataset = dataset_options[selected_dataset_label]
                else:
                    st.warning("⚠️ No datasets available")
                    selected_dataset = None

            with col2:
                # Модели (множественный выбор)
                models = api_client.list_models()
                if models:
                    model_options = {f"{m.name} ({m.provider})": m.id for m in models}
                    selected_model_labels = st.multiselect(
                        "Models* (select one or more)",
                        list(model_options.keys()),
                        help="Select models to test",
                    )
                    selected_models = [
                        model_options[label] for label in selected_model_labels
                    ]
                else:
                    st.warning("⚠️ No models available")
                    selected_models = []

            st.divider()

            # Execution Settings
            st.markdown("#### ⚙️ Execution Settings")
            col1, col2, col3 = st.columns(3)

            with col1:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help="Number of samples to process in parallel",
                )

            with col2:
                max_samples = st.number_input(
                    "Max Samples (0 = all)",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    help="Limit the number of samples",
                )

            with col3:
                evaluate = st.checkbox(
                    "Enable Standard Evaluation",
                    value=True,
                    help="Calculate standard metrics",
                )

            if evaluate:
                metrics = st.multiselect(
                    "Evaluation Metrics",
                    EVALUATION_METRICS,
                    default=["exact_match", "accuracy"],
                    help="Metrics to calculate",
                )
            else:
                metrics = []

            st.divider()

            # Prompt Variations
            st.markdown("#### 🔄 Prompt Variations (Optional)")

            enable_variations = st.checkbox(
                "Enable Prompt Variations",
                value=False,
                help="Generate variations of each prompt",
            )

            if enable_variations:
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Модель для вариаций
                    if models:
                        variation_model_label = st.selectbox(
                            "Variation Model",
                            list(model_options.keys()),
                            help="Model to generate variations",
                        )
                        variation_model_id = model_options[variation_model_label]
                    else:
                        variation_model_id = None

                with col2:
                    variations_per_prompt = st.number_input(
                        "Variations per Prompt",
                        min_value=1,
                        max_value=5,
                        value=1,
                        help="How many variations to generate",
                    )

                with col3:
                    variation_strategies = st.multiselect(
                        "Variation Strategies",
                        [s.value for s in VariationStrategy],
                        default=["paraphrase"],
                        format_func=lambda x: x.replace("_", " ").title(),
                        help="How to vary the prompts",
                    )
            else:
                variation_model_id = None
                variations_per_prompt = 0
                variation_strategies = []

            st.divider()

            # LLM Judge
            st.markdown("#### 🎯 LLM as a Judge (Optional)")

            enable_judge = st.checkbox(
                "Enable LLM Judge", value=False, help="Use an LLM to evaluate outputs"
            )

            if enable_judge:
                col1, col2 = st.columns(2)

                with col1:
                    # Модель-судья
                    if models:
                        judge_model_label = st.selectbox(
                            "Judge Model",
                            list(model_options.keys()),
                            help="Model to use as judge",
                        )
                        judge_model_id = model_options[judge_model_label]
                    else:
                        judge_model_id = None

                with col2:
                    judge_criteria = st.multiselect(
                        "Evaluation Criteria",
                        [
                            "accuracy",
                            "relevance",
                            "completeness",
                            "clarity",
                            "coherence",
                            "correctness",
                        ],
                        default=["accuracy", "relevance", "completeness"],
                        help="Criteria for the judge to evaluate",
                    )
            else:
                judge_model_id = None
                judge_criteria = []

            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button(
                    "🚀 Launch Task", type="primary", width="stretch"
                )

            if submitted:
                if not task_name or not selected_dataset or not selected_models:
                    st.error("⚠️ Please fill in all required fields")
                else:
                    try:
                        # Создаем конфиг
                        config = TaskConfig(
                            batch_size=batch_size,
                            max_samples=max_samples if max_samples > 0 else None,
                            evaluate=evaluate,
                            evaluation_metrics=metrics,
                            enable_variations=enable_variations,
                            variation_model_id=variation_model_id,
                            variation_strategies=variation_strategies,
                            variations_per_prompt=variations_per_prompt,
                            enable_judge=enable_judge,
                            judge_model_id=judge_model_id,
                            judge_criteria=judge_criteria,
                        )

                        task = api_client.create_task(
                            task_data=dict(
                                name=task_name,
                                dataset_id=selected_dataset,
                                model_ids=selected_models,
                                task_type=task_type,
                                config=config.model_dump(),
                            )
                        )

                        st.success(f"✅ Task '{task_name}' created and launched!")

                        # Показываем информацию
                        info_text = f"""
**Task Created:**
- {len(selected_models)} model(s) selected
- Variations: {"Enabled" if enable_variations else "Disabled"}
- LLM Judge: {"Enabled" if enable_judge else "Disabled"}
                        """
                        st.info(info_text)

                        st.balloons()
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error creating task: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
