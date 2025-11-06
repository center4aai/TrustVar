# src/ui/components/tasks_section.py (обновленная версия)
import time

import streamlit as st

from src.config.constants import EVALUATION_METRICS, SUPPORTED_TASKS
from src.ui.api_client import get_api_client
from src.ui.components.task_monitor import TaskMonitor


def render_tasks_section():
    """Рендер секции управления задачами"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## ⚡ Task Management")

    api_client = get_api_client()

    # Создаем экземпляр TaskMonitor
    task_monitor = TaskMonitor(api_client)

    # Tabs
    tab1, tab2 = st.tabs(["📋 Active Tasks", "➕ Create New Task"])

    # ===== TAB 1: Список задач с монитором =====
    with tab1:
        # Callbacks для монитора
        def on_view_task(task):
            st.session_state.selected_task_id = task.id
            st.session_state.selected_section = "results"
            st.rerun()

        def on_cancel_task(task):
            api_client.cancel_task(task.id)
            st.success(f"Task '{task.name}' cancelled!")
            st.rerun()

        # Используем TaskMonitor
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
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Task Configuration**")

                task_name = st.text_input(
                    "Task Name*",
                    placeholder="e.g., QA Evaluation on GPT-4",
                    help="A descriptive name for this task",
                )

                task_type = st.selectbox(
                    "Task Type*", SUPPORTED_TASKS, help="The type of task to perform"
                )

                # Датасеты
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
                    st.warning(
                        "⚠️ No datasets available. Please upload a dataset first."
                    )
                    selected_dataset = None

                # Модели
                models = api_client.list_models()
                if models:
                    model_options = {f"{m.name} ({m.provider})": m.id for m in models}
                    selected_model_label = st.selectbox(
                        "Model*",
                        list(model_options.keys()),
                        help="Select the model to test",
                    )
                    selected_model = model_options[selected_model_label]
                else:
                    st.warning("⚠️ No models available. Please register a model first.")
                    selected_model = None

            with col2:
                st.markdown("**Execution Settings**")

                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help="Number of samples to process in parallel",
                )

                max_samples = st.number_input(
                    "Max Samples (0 = all)",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    help="Limit the number of samples to process (0 for all)",
                )

                st.markdown("**Evaluation**")

                evaluate = st.checkbox(
                    "Enable Evaluation",
                    value=True,
                    help="Evaluate results against expected outputs",
                )

                if evaluate:
                    metrics = st.multiselect(
                        "Metrics",
                        EVALUATION_METRICS,
                        default=["exact_match", "accuracy"],
                        help="Metrics to calculate",
                    )
                else:
                    metrics = []

            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button(
                    "🚀 Launch Task", type="primary", use_container_width=True
                )

            if submitted:
                if not task_name or not selected_dataset or not selected_model:
                    st.error("⚠️ Please fill in all required fields")
                else:
                    try:
                        task = api_client.create_task(
                            task_data=dict(
                                name=task_name,
                                dataset_id=selected_dataset,
                                model_id=selected_model,
                                task_type=task_type,
                                batch_size=batch_size,
                                max_samples=max_samples if max_samples > 0 else None,
                                evaluate=evaluate,
                                evaluation_metrics=metrics
                            )
                        )

                        st.success(f"✅ Task '{task_name}' created and launched!")
                        st.balloons()

                        # Переключаемся на список задач
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error creating task: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
