# src/ui/components/dataset_uploader.py
import asyncio

import streamlit as st

from src.config.constants import SUPPORTED_TASKS, DatasetFormat
from src.core.services.dataset_service import DatasetService


class DatasetUploader:
    """Компонент для загрузки датасетов"""

    def __init__(self, dataset_service: DatasetService):
        self.dataset_service = dataset_service

    def render(self):
        """Отрисовка компонента"""

        st.markdown("### ➕ Upload New Dataset")

        with st.form("upload_dataset_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Dataset Information**")

                name = st.text_input(
                    "Dataset Name*",
                    placeholder="e.g., QA Test Set v1",
                    help="A unique name for your dataset",
                )

                description = st.text_area(
                    "Description",
                    placeholder="Brief description of the dataset...",
                    height=100,
                )

                task_type = st.selectbox(
                    "Task Type*",
                    SUPPORTED_TASKS,
                    help="The type of task this dataset is for",
                )

                tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="e.g., qa, english, test",
                    help="Tags to help organize datasets",
                )

            with col2:
                st.markdown("**Upload File**")

                file_format = st.selectbox(
                    "File Format*",
                    [f.value for f in DatasetFormat],
                    help="Format of your dataset file",
                )

                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=["jsonl", "json", "csv", "parquet"],
                    help="Upload your dataset file",
                )

                st.markdown("**Expected Format:**")

                if file_format == "jsonl":
                    st.code(
                        """{"prompt": "Question?", "expected_output": "Answer"}
{"prompt": "Another question?", "expected_output": "Another answer"}""",
                        language="json",
                    )
                elif file_format == "json":
                    st.code(
                        """[
  {"prompt": "Question?", "expected_output": "Answer"},
  {"prompt": "Another question?", "expected_output": "Another answer"}
]""",
                        language="json",
                    )
                elif file_format == "csv":
                    st.code(
                        '''prompt,expected_output
"Question?","Answer"
"Another question?","Another answer"''',
                        language="csv",
                    )

            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button(
                    "📤 Upload Dataset", type="primary", use_container_width=True
                )

            if submitted:
                if not name or not uploaded_file:
                    st.error("⚠️ Please provide dataset name and upload a file")
                else:
                    try:
                        # Создаем датасет
                        tag_list = [t.strip() for t in tags.split(",")] if tags else []

                        dataset = asyncio.run(
                            self.dataset_service.create_dataset(
                                name=name,
                                description=description,
                                task_type=task_type,
                                tags=tag_list,
                            )
                        )

                        # Загружаем данные
                        with st.spinner("Uploading data..."):
                            count = asyncio.run(
                                self.dataset_service.upload_from_file(
                                    dataset.id, uploaded_file, file_format
                                )
                            )

                        st.success(
                            f"✅ Dataset '{name}' uploaded successfully with {count} items!"
                        )
                        st.balloons()

                        # Очищаем и перезагружаем
                        asyncio.run(asyncio.sleep(1))
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error uploading dataset: {e}")
