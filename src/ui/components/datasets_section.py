# src/ui/components/datasets_section.py
import streamlit as st

from src.ui.api_client import get_api_client
from src.ui.components.dataset_uploader import DatasetUploader


def render_datasets_section():
    """Рендер секции управления датасетами"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Management")

    api_client = get_api_client()

    # --- Инициализация состояния ---
    if "datasets_current_tab" not in st.session_state:
        st.session_state.datasets_current_tab = "📋 All Datasets"

    if "selected_dataset_id" not in st.session_state:
        st.session_state.selected_dataset_id = None

    # --- Колбэк-функции для управления состоянием навигации ---
    def view_dataset_details(dataset_id):
        """Сохраняет ID датасета и переключает вкладку на 'Details'."""
        st.session_state.selected_dataset_id = dataset_id
        st.session_state.datasets_current_tab = "🔍 Details"

    def back_to_list():
        """Очищает ID и возвращает на вкладку со списком."""
        st.session_state.selected_dataset_id = None
        st.session_state.datasets_current_tab = "📋 All Datasets"

    # --- Навигация ---
    tab_names = ["📋 All Datasets", "➕ Upload New", "🔍 Details"]

    # Находим индекс текущей вкладки
    try:
        current_index = tab_names.index(st.session_state.datasets_current_tab)
    except ValueError:
        current_index = 0
        st.session_state.datasets_current_tab = tab_names[0]

    # Используем st.radio с отдельным ключом и callback
    def on_tab_change():
        """Синхронизирует изменение radio с internal state"""
        st.session_state.datasets_current_tab = st.session_state.datasets_tab_radio

    selected_tab = st.radio(
        "Dataset Management Navigation",
        options=tab_names,
        index=current_index,
        key="datasets_tab_radio",
        horizontal=True,
        label_visibility="collapsed",
        on_change=on_tab_change,
    )

    # Используем актуальное значение из session_state
    active_tab = st.session_state.datasets_current_tab

    # --- Отрисовка контента в зависимости от выбранной вкладки ---
    if active_tab == "📋 All Datasets":
        _render_all_datasets_tab(api_client, view_dataset_details)

    elif active_tab == "➕ Upload New":
        _render_upload_tab(api_client)

    elif active_tab == "🔍 Details":
        _render_details_tab(api_client, back_to_list)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_all_datasets_tab(api_client, view_dataset_details):
    """Рендер вкладки со списком датасетов"""

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
                col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1, 1.5, 1.5])

                with col1:
                    st.markdown(f"**{dataset.name}**")
                    if dataset.description:
                        desc_short = (
                            dataset.description[:60] + "..."
                            if len(dataset.description) > 60
                            else dataset.description
                        )
                        st.caption(desc_short)

                with col2:
                    st.caption(f"**Type:** `{dataset.task_type}`")
                    st.caption(f"**Items:** {dataset.size}")

                with col3:
                    st.caption(f"**Format:** `{dataset.format}`")

                with col4:
                    st.caption(dataset.created_at.strftime("%Y-%m-%d %H:%M"))

                with col5:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(
                            "👁️",
                            key=f"view_ds_{dataset.id}",
                            use_container_width=True,
                            help="View details",
                        ):
                            view_dataset_details(dataset.id)
                            st.rerun()

                    with btn_col2:
                        if st.button(
                            "🗑️",
                            key=f"del_ds_{dataset.id}",
                            use_container_width=True,
                            help="Delete dataset",
                        ):
                            confirm_key = f"confirm_del_{dataset.id}"
                            if st.session_state.get(confirm_key):
                                api_client.delete_dataset(dataset.id)
                                st.toast(f"Dataset '{dataset.name}' deleted.")
                                del st.session_state[confirm_key]
                                st.rerun()
                            else:
                                st.session_state[confirm_key] = True
                                st.rerun()

                st.markdown(
                    "<hr style='margin: 8px 0; border: 0; border-top: 1px solid #333;'>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("📭 No datasets found. Upload your first dataset!")

    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")


def _render_upload_tab(api_client):
    """Рендер вкладки загрузки датасета"""
    uploader = DatasetUploader(api_client)
    uploader.render()


def _render_details_tab(api_client, back_to_list):
    """Рендер вкладки деталей датасета"""

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
                    if st.button(
                        "⬅️ Back to list",
                        use_container_width=True,
                        key="back_to_list_btn",
                    ):
                        back_to_list()
                        st.rerun()

                st.divider()

                stats = api_client.get_dataset_stats(dataset_id)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📝 Total Items", stats.get("total_items", 0))
                col2.metric(
                    "📏 Avg Prompt Len", f"{stats.get('avg_prompt_length', 0):.0f}"
                )
                col3.metric("🎯 With Target", stats.get("items_with_target", 0))
                col4.metric("📈 Target Coverage", f"{stats.get('coverage', 0):.1f}%")
                st.divider()

                st.markdown("### 📄 Sample Items")
                items = api_client.get_dataset_items(dataset_id, limit=20)
                if not items:
                    st.info("No items found in this dataset.")
                for i, item in enumerate(items, 1):
                    with st.expander(f"**Item #{i}:** `{item.prompt[:70].strip()}...`"):
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
        st.info("👈 Select a dataset from the 'All Datasets' tab to view its details.")
