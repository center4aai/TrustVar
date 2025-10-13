# src/ui/pages/01_📊_Datasets.py
import asyncio

import pandas as pd
import streamlit as st
from src.ui.components.dataset_uploader import DatasetUploader

from src.core.services.dataset_service import DatasetService

st.set_page_config(page_title="Datasets", page_icon="📊", layout="wide")

# Инициализация
if "dataset_service" not in st.session_state:
    st.session_state.dataset_service = DatasetService()

dataset_service = st.session_state.dataset_service

# Header
st.title("📊 Dataset Management")
st.markdown("Upload and manage your test datasets")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 All Datasets", "➕ Upload New", "🔍 Dataset Details"])

# Tab 1: List Datasets
with tab1:
    st.subheader("Your Datasets")

    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input(
            "🔍 Search datasets", placeholder="Enter dataset name..."
        )
    with col2:
        task_type_filter = st.selectbox(
            "Task Type",
            [
                "All",
                "text-generation",
                "question-answering",
                "summarization",
                "classification",
            ],
        )
    with col3:
        sort_by = st.selectbox("Sort by", ["Created Date", "Name", "Size"])

    # Fetch datasets
    try:
        datasets = asyncio.run(dataset_service.list_datasets())

        if datasets:
            # Convert to DataFrame
            datasets_data = []
            for ds in datasets:
                datasets_data.append(
                    {
                        "Name": ds.name,
                        "Task Type": ds.task_type,
                        "Size": ds.size,
                        "Format": ds.format,
                        "Created": ds.created_at.strftime("%Y-%m-%d %H:%M"),
                        "ID": ds.id,
                    }
                )

            df = pd.DataFrame(datasets_data)

            # Display datasets
            for idx, row in df.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 2, 1])

                    with col1:
                        st.markdown(f"**{row['Name']}**")
                        st.caption(f"ID: {row['ID'][:8]}...")

                    with col2:
                        st.markdown(f"📝 {row['Task Type']}")

                    with col3:
                        st.markdown(f"**{row['Size']}** items")

                    with col4:
                        st.markdown(f"🕐 {row['Created']}")

                    with col5:
                        if st.button("👁️ View", key=f"view_{row['ID']}"):
                            st.session_state.selected_dataset = row["ID"]
                            st.rerun()

                    st.divider()
        else:
            st.info("No datasets found. Upload your first dataset!")

    except Exception as e:
        st.error(f"Error loading datasets: {e}")

# Tab 2: Upload New Dataset
with tab2:
    st.subheader("Upload New Dataset")

    uploader = DatasetUploader(dataset_service)
    uploader.render()

# Tab 3: Dataset Details
with tab3:
    if "selected_dataset" in st.session_state:
        dataset_id = st.session_state.selected_dataset

        try:
            dataset = asyncio.run(dataset_service.get_dataset(dataset_id))

            if dataset:
                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"📊 {dataset.name}")
                    st.markdown(dataset.description or "No description")

                with col2:
                    if st.button("🗑️ Delete Dataset", type="secondary"):
                        if st.session_state.get("confirm_delete"):
                            asyncio.run(dataset_service.delete_dataset(dataset_id))
                            st.success("Dataset deleted!")
                            del st.session_state.selected_dataset
                            st.rerun()
                        else:
                            st.session_state.confirm_delete = True
                            st.warning("Click again to confirm deletion")

                # Stats
                st.divider()
                stats = asyncio.run(dataset_service.get_stats(dataset_id))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Items", stats.get("total_items", 0))
                col2.metric(
                    "Avg Prompt Length", f"{stats.get('avg_prompt_length', 0):.0f}"
                )
                col3.metric(
                    "With Expected Output", stats.get("items_with_expected_output", 0)
                )
                col4.metric("Coverage", f"{stats.get('coverage', 0):.1f}%")

                # Sample items
                st.divider()
                st.subheader("Sample Items")

                items = asyncio.run(dataset_service.get_items(dataset_id, limit=10))

                for i, item in enumerate(items):
                    with st.expander(f"Item {i + 1}: {item.prompt[:50]}..."):
                        st.markdown("**Prompt:**")
                        st.code(item.prompt)

                        if item.expected_output:
                            st.markdown("**Expected Output:**")
                            st.code(item.expected_output)

                        if item.metadata:
                            st.markdown("**Metadata:**")
                            st.json(item.metadata)
            else:
                st.error("Dataset not found")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    else:
        st.info("Select a dataset from the 'All Datasets' tab to view details")
