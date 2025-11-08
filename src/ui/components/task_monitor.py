# src/ui/components/task_monitor.py
import time
from datetime import datetime
from typing import Callable, List, Optional

import streamlit as st

from src.core.schemas.task import Task, TaskStatus
from src.ui.api_client import ApiClient


class TaskMonitor:
    """Компонент для мониторинга задач в реальном времени"""

    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

    def render(
        self,
        filters: Optional[dict] = None,
        auto_refresh: bool = True,
        refresh_interval: int = 2,
        on_view_click: Optional[Callable] = None,
        on_cancel_click: Optional[Callable] = None,
        show_filters: bool = True,
        compact_mode: bool = False,
    ):
        """
        Отрисовка монитора задач

        Args:
            filters: Фильтры для задач (статус, поиск и т.д.)
            auto_refresh: Автообновление
            refresh_interval: Интервал обновления в секундах
            on_view_click: Callback при клике на просмотр
            on_cancel_click: Callback при отмене задачи
            show_filters: Показывать ли фильтры
            compact_mode: Компактный режим отображения
        """

        # Фильтры
        if show_filters:
            search, status_filter, sort_by = self._render_filters()

            if filters is None:
                filters = {}

            filters["search"] = search
            filters["status"] = status_filter
            filters["sort"] = sort_by

        # Загрузка задач
        try:
            tasks = self.api_client.list_tasks(limit=100)

            # Применяем фильтры
            tasks = self._apply_filters(tasks, filters)

            if tasks:
                if compact_mode:
                    self._render_compact(tasks, on_view_click, on_cancel_click)
                else:
                    self._render_full(tasks, on_view_click, on_cancel_click)

                # Auto-refresh для running задач
                if auto_refresh and any(t.status == TaskStatus.RUNNING for t in tasks):
                    time.sleep(refresh_interval)
                    st.rerun()
            else:
                st.info("📭 No tasks found")

        except Exception as e:
            st.error(f"❌ Error loading tasks: {e}")

    def _render_filters(self) -> tuple:
        """Отрисовка панели фильтров"""

        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            search = st.text_input(
                "🔍 Search",
                placeholder="Search tasks...",
                key="task_monitor_search",
                label_visibility="collapsed",
            )

        with col2:
            status_filter = st.multiselect(
                "Status",
                ["pending", "running", "completed", "failed", "cancelled"],
                default=["pending", "running", "completed"],
                key="task_monitor_status",
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["created_desc", "created_asc", "status", "progress"],
                format_func=lambda x: {
                    "created_desc": "Newest First",
                    "created_asc": "Oldest First",
                    "status": "Status",
                    "progress": "Progress",
                }[x],
                key="task_monitor_sort",
            )

        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄", width="stretch", key="task_monitor_refresh"):
                st.rerun()

        return search, status_filter, sort_by

    def _apply_filters(self, tasks: List[Task], filters: dict) -> List[Task]:
        """Применение фильтров к задачам"""

        filtered_tasks = tasks

        # Фильтр по статусу
        if filters.get("status"):
            filtered_tasks = [
                t for t in filtered_tasks if t.status in filters["status"]
            ]

        # Поиск
        if filters.get("search"):
            search_term = filters["search"].lower()
            filtered_tasks = [
                t
                for t in filtered_tasks
                if search_term in t.name.lower() or search_term in t.id.lower()
            ]

        # Сортировка
        sort_by = filters.get("sort", "created_desc")

        if sort_by == "created_desc":
            filtered_tasks.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "created_asc":
            filtered_tasks.sort(key=lambda x: x.created_at)
        elif sort_by == "status":
            status_order = {
                TaskStatus.RUNNING: 0,
                TaskStatus.PENDING: 1,
                TaskStatus.COMPLETED: 2,
                TaskStatus.FAILED: 3,
                TaskStatus.CANCELLED: 4,
            }
            filtered_tasks.sort(key=lambda x: status_order.get(x.status, 5))
        elif sort_by == "progress":
            filtered_tasks.sort(key=lambda x: x.progress, reverse=True)

        return filtered_tasks

    def _render_full(
        self,
        tasks: List[Task],
        on_view_click: Optional[Callable] = None,
        on_cancel_click: Optional[Callable] = None,
    ):
        """Полный режим отображения"""

        for task in tasks:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)

                col1, col2, col3, col4, col5 = st.columns([3, 2, 2.5, 1.5, 1.5])

                with col1:
                    st.markdown(f"### {task.name}")
                    st.caption(f"🆔 {task.id[:12]}...")
                    st.caption(f"📝 {task.task_type}")

                with col2:
                    # Status badge
                    status_html = self._get_status_badge(task.status)
                    st.markdown(status_html, unsafe_allow_html=True)

                    # Время выполнения
                    time_info = self._get_time_info(task)
                    if time_info:
                        st.caption(time_info)

                with col3:
                    # Progress
                    self._render_progress(task)

                with col4:
                    # Метрики
                    if task.aggregated_metrics:
                        for model_id, metrics in task.aggregated_metrics.items():
                            for metric, value in metrics.items():
                                st.metric(
                                    metric.replace("_", " ").title(), f"{value:.2f}"
                                )
                        # for metric, value in list(task.aggregated_metrics.items())[:2]:
                        #     st.metric(metric.replace("_", " ").title(), f"{value:.1f}")

                with col5:
                    # Кнопки действий
                    self._render_actions(task, on_view_click, on_cancel_click)

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

    def _render_compact(
        self,
        tasks: List[Task],
        on_view_click: Optional[Callable] = None,
        on_cancel_click: Optional[Callable] = None,
    ):
        """Компактный режим отображения"""

        for task in tasks:
            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])

            with col1:
                st.markdown(f"**{task.name}**")
                st.caption(f"{task.task_type} • {task.id[:8]}...")

            with col2:
                status_html = self._get_status_badge(task.status)
                st.markdown(status_html, unsafe_allow_html=True)

            with col3:
                if task.status == TaskStatus.RUNNING:
                    st.progress(task.progress / 100)
                    st.caption(f"{task.progress:.0f}%")
                elif task.status == TaskStatus.COMPLETED:
                    st.success("✅ Done")

            with col4:
                if task.status == TaskStatus.COMPLETED and on_view_click:
                    if st.button("👁️", key=f"view_compact_{task.id}"):
                        on_view_click(task)

            st.divider()

    def _render_progress(self, task: Task):
        """Отрисовка прогресса задачи"""

        if task.status == TaskStatus.RUNNING:
            st.progress(task.progress / 100)
            st.caption(
                f"{task.progress:.1f}% • "
                f"{task.processed_samples}/{task.total_samples} samples"
            )

        elif task.status == TaskStatus.COMPLETED:
            st.progress(1.0)
            st.caption(f"✅ {task.total_samples} samples completed")

        elif task.status == TaskStatus.FAILED:
            st.error("❌ Failed")
            if task.error:
                with st.expander("Error details"):
                    st.code(task.error, language=None)

        elif task.status == TaskStatus.PENDING:
            st.info("⏳ Waiting to start...")

        elif task.status == TaskStatus.CANCELLED:
            st.warning("⚫ Cancelled")

    def _render_actions(
        self,
        task: Task,
        on_view_click: Optional[Callable] = None,
        on_cancel_click: Optional[Callable] = None,
    ):
        """Отрисовка кнопок действий"""

        # Кнопка просмотра результатов
        if task.status == TaskStatus.COMPLETED:
            if st.button("📊 View", key=f"view_{task.id}", width="stretch"):
                if on_view_click:
                    on_view_click(task)
                else:
                    st.session_state.selected_task_id = task.id
                    st.session_state.selected_section = "results"
                    st.rerun()

        # Кнопка отмены
        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if st.button("❌ Cancel", key=f"cancel_{task.id}", width="stretch"):
                confirm_key = f"confirm_cancel_{task.id}"

                if st.session_state.get(confirm_key):
                    if on_cancel_click:
                        on_cancel_click(task)
                    else:
                        self.api_client.cancel_task(task.id)
                        st.success("Cancelled!")
                        st.rerun()
                else:
                    st.session_state[confirm_key] = True
                    st.warning("Click again to confirm")

        # Кнопка повтора для failed задач
        if task.status == TaskStatus.FAILED:
            if st.button("🔄 Retry", key=f"retry_{task.id}", width="stretch"):
                try:
                    self.api_client.create_task(
                        task_data=dict(
                            name=f"{task.name} (Retry)",
                            dataset_id=task.dataset_id,
                            model_id=task.model_id,
                            task_type=task.task_type,
                            batch_size=task.batch_size,
                            max_samples=task.max_samples,
                            evaluate=task.evaluate,
                            evaluation_metrics=task.evaluation_metrics,
                        )
                    )

                    st.success("Task retried!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error retrying task: {e}")

    def _get_status_badge(self, status: str) -> str:
        """Генерация HTML для бейджа статуса"""

        badges = {
            "pending": '<span class="status-badge status-pending">🟡 PENDING</span>',
            "running": '<span class="status-badge status-running">🔵 RUNNING</span>',
            "completed": '<span class="status-badge status-completed">🟢 COMPLETED</span>',
            "failed": '<span class="status-badge status-failed">🔴 FAILED</span>',
            "cancelled": '<span class="status-badge">⚫ CANCELLED</span>',
        }

        return badges.get(status, '<span class="status-badge">⚪ UNKNOWN</span>')

    def _get_time_info(self, task: Task) -> Optional[str]:
        """Получение информации о времени выполнения"""

        if task.started_at:
            if task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                return f"⏱️ Completed in {duration:.0f}s"
            else:
                duration = (datetime.utcnow() - task.started_at).total_seconds()
                return f"⏱️ Running {duration:.0f}s"

        return None

    def render_mini(self, task_id: str):
        """Мини-виджет для отображения одной задачи"""

        try:
            task = self.api_client.get_task(task_id)

            if not task:
                st.error("Task not found")
                return

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**{task.name}**")
                st.caption(f"{task.task_type}")

            with col2:
                status_html = self._get_status_badge(task.status)
                st.markdown(status_html, unsafe_allow_html=True)

            with col3:
                if task.status == TaskStatus.RUNNING:
                    st.progress(task.progress / 100)
                    st.caption(f"{task.progress:.0f}%")

        except Exception as e:
            st.error(f"Error: {e}")

    def render_summary(self, limit: int = 5):
        """Виджет сводки последних задач"""

        try:
            tasks = self.api_client.list_tasks(limit=limit)

            if not tasks:
                st.info("No recent tasks")
                return

            st.markdown("### 📊 Recent Tasks")

            for task in tasks:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"**{task.name}**")

                with col2:
                    status_html = self._get_status_badge(task.status)
                    st.markdown(status_html, unsafe_allow_html=True)

                with col3:
                    if task.status == TaskStatus.COMPLETED:
                        if st.button("View", key=f"summary_view_{task.id}"):
                            st.session_state.selected_task_id = task.id
                            st.session_state.selected_section = "results"
                            st.rerun()

                st.divider()

        except Exception as e:
            st.error(f"Error: {e}")
