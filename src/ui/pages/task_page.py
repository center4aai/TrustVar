# src/ui/pages/03_⚡_Tasks.py
import asyncio
from datetime import datetime

import streamlit as st

from src.config.constants import EVALUATION_METRICS, SUPPORTED_TASKS
from src.core.services.dataset_service import DatasetService
from src.core.services.model_service import ModelService
from src.core.services.task_service import TaskService

st.set_page_config(page_title="Tasks", page_icon="⚡", layout="wide")

# Auto-refresh every 2 seconds
st_autorefresh = st.empty()
import time

refresh_interval = 2000  # ms

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

# Header
st.title("⚡ Task Management")
st.markdown("Create and monitor LLM testing tasks")

# Tabs
tab1, tab2 = st.tabs(["📋 All Tasks", "➕ Create New"])

# Tab 1: List Tasks
with tab1:
    st.subheader("Your Tasks")

    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search tasks", placeholder="Enter task name...")
    with col2:
        status_filter = st.multiselect(
            "Status",
            ["pending", "running", "completed", "failed", "cancelled"],
            default=["pending", "running"],
        )
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    # Fetch tasks
    try:
        tasks = asyncio.run(task_service.list_tasks(limit=50))

        if status_filter:
            tasks = [t for t in tasks if t.status in status_filter]

        if tasks:
            for task in tasks:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 2])

                    with col1:
                        st.markdown(f"**{task.name}**")
                        st.caption(f"ID: {task.id[:8]}...")

                    with col2:
                        # Status badge
                        status_colors = {
                            "pending": "🟡",
                            "running": "🔵",
                            "completed": "🟢",
                            "failed": "🔴",
                            "cancelled": "⚫",
                        }
                        icon = status_colors.get(task.status, "⚪")
                        st.markdown(f"{icon} **{task.status.upper()}**")

                    with col3:
                        if task.status == "running":
                            st.progress(task.progress / 100)
                            st.caption(
                                f"{task.progress:.1f}% ({task.processed_samples}/{task.total_samples})"
                            )
                        elif task.status == "completed":
                            st.success("100%")
                        elif task.status == "failed":
                            st.error("Failed")
                            if task.error:
                                with st.expander("Error details"):
                                    st.code(task.error)

                    with col4:
                        # Duration
                        if task.completed_at and task.started_at:
                            duration = (
                                task.completed_at - task.started_at
                            ).total_seconds()
                            st.caption(f"⏱️ {duration:.0f}s")
                        elif task.started_at:
                            duration = (
                                datetime.utcnow() - task.started_at
                            ).total_seconds()
                            st.caption(f"⏱️ {duration:.0f}s")

                    with col5:
                        col_view, col_cancel = st.columns(2)

                        with col_view:
                            if task.status == "completed":
                                if st.button("📊 Results", key=f"view_{task.id}"):
                                    st.session_state.selected_task = task.id
                                    st.switch_page("pages/04_📈_Results.py")

                        with col_cancel:
                            if task.status in ["pending", "running"]:
                                if st.button("❌", key=f"cancel_{task.id}"):
                                    asyncio.run(task_service.cancel_task(task.id))
                                    st.rerun()

                    st.divider()

            # Auto-refresh
            if auto_refresh:
                time.sleep(2)
                st.rerun()

        else:
            st.info("No tasks found. Create your first task!")

    except Exception as e:
        st.error(f"Error loading tasks: {e}")

# Tab 2: Create New Task
with tab2:
    st.subheader("Create New Task")

    with st.form("create_task_form"):
        col1, col2 = st.columns(2)

        with col1:
            task_name = st.text_input("Task Name*", placeholder="e.g., QA Test Run")

            # Fetch datasets
            datasets = asyncio.run(dataset_service.list_datasets())
            dataset_options = {f"{ds.name} ({ds.size} items)": ds.id for ds in datasets}

            if dataset_options:
                selected_dataset_label = st.selectbox(
                    "Dataset*", list(dataset_options.keys())
                )
                selected_dataset = dataset_options[selected_dataset_label]
            else:
                st.warning("No datasets available. Please upload a dataset first.")
                selected_dataset = None

            # Fetch models
            models = asyncio.run(model_service.list_models(active_only=True))
            model_options = {f"{m.name} ({m.provider})": m.id for m in models}

            if model_options:
                selected_model_label = st.selectbox(
                    "Model*", list(model_options.keys())
                )
                selected_model = model_options[selected_model_label]
            else:
                st.warning("No models available. Please register a model first.")
                selected_model = None

        with col2:
            task_type = st.selectbox("Task Type*", SUPPORTED_TASKS)
            batch_size = st.number_input("Batch Size", 1, 100, 1)
            max_samples = st.number_input(
                "Max Samples (0 = all)",
                0,
                100000,
                0,
                help="Limit the number of samples to process",
            )

            evaluate = st.checkbox("Evaluate Results", value=True)

            if evaluate:
                metrics = st.multiselect(
                    "Evaluation Metrics", EVALUATION_METRICS, default=["exact_match"]
                )
            else:
                metrics = []

        submitted = st.form_submit_button("🚀 Launch Task", type="primary")

        if submitted:
            if not task_name or not selected_dataset or not selected_model:
                st.error("Please fill in all required fields")
            else:
                try:
                    task = asyncio.run(
                        task_service.create_task(
                            name=task_name,
                            dataset_id=selected_dataset,
                            model_id=selected_model,
                            task_type=task_type,
                            batch_size=batch_size,
                            max_samples=max_samples if max_samples > 0 else None,
                            evaluate=evaluate,
                            evaluation_metrics=metrics,
                        )
                    )

                    st.success(f"✅ Task '{task_name}' created and launched!")
                    st.balloons()

                    # Switch to tasks list
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error creating task: {e}")
