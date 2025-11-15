# src/ui/components/datasets_section.py
import streamlit as st

from src.ui.api_client import get_api_client
from src.ui.components.dataset_uploader import DatasetUploader


def render_datasets_section():
    """Рендер секции управления датасетами"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Management")

    api_client = get_api_client()

    # --- Колбэк-функции для управления состоянием навигации ---
    def view_dataset_details(dataset_id):
        """Сохраняет ID датасета и переключает вкладку на 'Details'."""
        st.session_state.selected_dataset_id = dataset_id
        st.session_state.datasets_tab_selection = "🔍 Details"

    def back_to_list():
        """Очищает ID и возвращает на вкладку со списком."""
        if "selected_dataset_id" in st.session_state:
            del st.session_state.selected_dataset_id
        st.session_state.datasets_tab_selection = "📋 All Datasets"

    # --- Навигация с сохранением состояния ---
    tab_names = ["📋 All Datasets", "➕ Upload New", "🔍 Details"]

    # Инициализация состояния вкладки, если его еще нет
    if "datasets_tab_selection" not in st.session_state:
        st.session_state.datasets_tab_selection = "📋 All Datasets"

    # Используем st.radio, которое сохраняет свое состояние через `key`
    selected_tab = st.radio(
        "Dataset Management Navigation",
        options=tab_names,
        key="datasets_tab_selection",
        horizontal=True,
        label_visibility="collapsed",
    )

    # --- Отрисовка контента в зависимости от выбранной "вкладки" ---
    if selected_tab == "📋 All Datasets":
        # ===== TAB 1: Список датасетов =====
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.text_input(
                "🔍 Search", placeholder="Search datasets...", key="dataset_search"
            )
        with col2:
            st.selectbox(
                "Task Type",
                [
                    "All",
                    "text-generation",
                    "question-answering",
                    "summarization",
                    "classification",
                ],
                key="dataset_task_filter",
            )
        with col3:
            st.selectbox("Sort by", ["Created", "Name", "Size"], key="dataset_sort")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        st.divider()

        try:
            datasets = api_client.list_datasets()
            if datasets:
                for dataset in datasets:
                    with st.container(border=True):
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 1.5, 2, 1.5])
                        with col1:
                            st.markdown(f"**{dataset.name}**")
                            if dataset.description:
                                st.caption(dataset.description, unsafe_allow_html=False)
                            st.caption(f"🆔 `{dataset.id}`")
                        with col2:
                            st.markdown(f"**Task:** `{dataset.task_type}`")
                            st.caption(f"**Format:** `{dataset.format}`")
                        with col3:
                            st.metric("Items", dataset.size)
                        with col4:
                            st.caption("📅 Created")
                            st.write(dataset.created_at.strftime("%Y-%m-%d %H:%M"))
                        with col5:
                            st.button(
                                "👁️ View",
                                key=f"view_ds_{dataset.id}",
                                use_container_width=True,
                                on_click=view_dataset_details,
                                args=(dataset.id,),
                            )
                            if st.button(
                                "🗑️ Delete",
                                type="secondary",
                                key=f"del_ds_{dataset.id}",
                                use_container_width=True,
                            ):
                                api_client.delete_dataset(dataset.id)
                                st.toast(f"Dataset '{dataset.name}' deleted.")
                                st.rerun()
            else:
                st.info("📭 No datasets found. Upload your first dataset!")
        except Exception as e:
            st.error(f"❌ Error loading datasets: {e}")

    elif selected_tab == "➕ Upload New":
        # ===== TAB 2: Загрузка нового датасета =====
        uploader = DatasetUploader(api_client)
        uploader.render()

    elif selected_tab == "🔍 Details":
        # ===== TAB 3: Детали датасета =====
        if (
            "selected_dataset_id" in st.session_state
            and st.session_state.selected_dataset_id
        ):
            dataset_id = st.session_state.selected_dataset_id
            try:
                dataset = api_client.get_dataset(dataset_id)
                if dataset:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"## 📊 {dataset.name}")
                        st.markdown(dataset.description or "*No description*")
                    with col2:
                        st.button(
                            "⬅️ Back to list",
                            use_container_width=True,
                            on_click=back_to_list,
                        )
                    st.divider()

                    stats = api_client.get_dataset_stats(dataset_id)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📝 Total Items", stats.get("total_items", 0))
                    col2.metric(
                        "📏 Avg Prompt Len", f"{stats.get('avg_prompt_length', 0):.0f}"
                    )
                    col3.metric("🎯 With Target", stats.get("items_with_target", 0))
                    col4.metric(
                        "📈 Target Coverage", f"{stats.get('coverage', 0):.1f}%"
                    )
                    st.divider()

                    st.markdown("### 📄 Sample Items")
                    items = api_client.get_dataset_items(dataset_id, limit=20)
                    if not items:
                        st.info("No items found in this dataset.")
                    for i, item in enumerate(items, 1):
                        with st.expander(
                            f"**Item #{i}:** `{item.prompt[:70].strip()}...`"
                        ):
                            st.markdown("**Prompt:**")
                            st.code(item.prompt, language=None)
                            if item.target:
                                st.markdown("**Target:**")
                                st.code(item.target, language=None)
                            else:
                                st.info("This item does not have a target.")
                            if item.metadata:
                                st.markdown("**Metadata:**")
                                st.json(item.metadata)
                else:
                    st.error("Dataset not found. It might have been deleted.")
                    back_to_list()
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading dataset details: {e}")
        else:
            st.info(
                "👈 Select a dataset from the 'All Datasets' tab to view its details."
            )

    st.markdown("</div>", unsafe_allow_html=True)
