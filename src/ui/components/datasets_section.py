# src/ui/components/datasets_section.py
import streamlit as st

from src.ui.api_client import get_api_client
from src.ui.components.dataset_uploader import DatasetUploader


def render_datasets_section():
    """Рендер секции управления датасетами"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Management")

    api_client = get_api_client()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 All Datasets", "➕ Upload New", "🔍 Details"])

    # ===== TAB 1: Список датасетов =====
    with tab1:
        # Фильтры в одной строке
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            search = st.text_input(
                "🔍 Search", placeholder="Search datasets...", key="dataset_search"
            )

        with col2:
            task_filter = st.selectbox(
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
            sort_by = st.selectbox(
                "Sort by", ["Created", "Name", "Size"], key="dataset_sort"
            )

        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", width="stretch"):
                st.rerun()

        st.divider()

        # Загрузка датасетов
        try:
            datasets = api_client.list_datasets()

            if datasets:
                for dataset in datasets:
                    with st.container():
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

                        col1, col2, col3, col4, col5 = st.columns([3, 2, 1.5, 2, 1.5])

                        with col1:
                            st.markdown(f"### {dataset.name}")
                            if dataset.description:
                                st.caption(dataset.description)
                            st.caption(f"🆔 {dataset.id[:12]}...")

                        with col2:
                            st.markdown(f"**Task:** {dataset.task_type}")
                            st.caption(f"Format: {dataset.format}")

                        with col3:
                            st.metric("Items", dataset.size)

                        with col4:
                            st.caption("📅 Created")
                            st.write(dataset.created_at.strftime("%Y-%m-%d %H:%M"))

                        with col5:
                            if st.button(
                                "👁️ View",
                                key=f"view_ds_{dataset.id}",
                                width="stretch",
                            ):
                                st.session_state.selected_dataset_id = dataset.id
                                st.rerun()

                            if st.button(
                                "🗑️ Delete",
                                key=f"del_ds_{dataset.id}",
                                width="stretch",
                            ):
                                if st.session_state.get(f"confirm_delete_{dataset.id}"):
                                    api_client.delete_dataset(dataset.id)
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.session_state[f"confirm_delete_{dataset.id}"] = (
                                        True
                                    )
                                    st.warning("Click again to confirm")

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("📭 No datasets found. Upload your first dataset!")

        except Exception as e:
            st.error(f"❌ Error loading datasets: {e}")

    # ===== TAB 2: Загрузка нового датасета =====
    with tab2:
        uploader = DatasetUploader(api_client)
        uploader.render()

    # ===== TAB 3: Детали датасета =====
    with tab3:
        if "selected_dataset_id" in st.session_state:
            dataset_id = st.session_state.selected_dataset_id

            try:
                dataset = api_client.get_dataset(dataset_id)

                if dataset:
                    # Заголовок
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"## 📊 {dataset.name}")
                        st.markdown(dataset.description or "*No description*")

                    with col2:
                        if st.button("⬅️ Back", width="stretch"):
                            del st.session_state.selected_dataset_id
                            st.rerun()

                    st.divider()

                    # Статистика
                    stats = api_client.get_dataset_stats(dataset_id)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📝 Total Items", stats.get("total_items", 0))
                    col2.metric(
                        "📏 Avg Length", f"{stats.get('avg_prompt_length', 0):.0f}"
                    )
                    col3.metric("✅ With Expected", stats.get("items_with_target", 0))
                    col4.metric("📊 Coverage", f"{stats.get('coverage', 0):.1f}%")

                    st.divider()

                    # Примеры данных
                    st.markdown("### 📄 Sample Items")

                    items = api_client.get_dataset_items(dataset_id)

                    for i, item in enumerate(items, 1):
                        with st.expander(f"**Item {i}:** {item.prompt[:60]}..."):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Prompt:**")
                                st.code(item.prompt, language=None)

                            with col2:
                                if item.target:
                                    st.markdown("**Expected Output:**")
                                    st.code(item.target, language=None)
                                else:
                                    st.info("No expected output")

                            if item.metadata:
                                st.markdown("**Metadata:**")
                                st.json(item.metadata)
                else:
                    st.error("Dataset not found")

            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        else:
            st.info("👈 Select a dataset from the 'All Datasets' tab")

    st.markdown("</div>", unsafe_allow_html=True)
