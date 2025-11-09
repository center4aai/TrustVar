# src/ui/components/dataset_uploader.py
import csv
import io
import json
import time

import streamlit as st
from requests.exceptions import RequestException

from src.config.constants import SUPPORTED_TASKS, DatasetFormat
from src.ui.api_client import ApiClient


class DatasetUploader:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

    def render(self):
        st.markdown("### ➕ Upload New Dataset")

        # Инициализируем состояние
        if "dataset_columns" not in st.session_state:
            st.session_state.dataset_columns = []
        if "dataset_preview" not in st.session_state:
            st.session_state.dataset_preview = []

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
                    key="dataset_file_uploader",
                )

                # Анализируем файл при загрузке
                if uploaded_file is not None:
                    try:
                        columns = self._extract_columns(uploaded_file, file_format)
                        st.session_state.dataset_columns = columns

                        # Показываем превью
                        preview = self._get_preview(uploaded_file, file_format)
                        st.session_state.dataset_preview = preview

                        if columns:
                            st.success(f"✅ Found {len(columns)} columns")
                            st.caption(
                                "Detected columns: "
                                + ", ".join(columns[:5])
                                + ("..." if len(columns) > 5 else "")
                            )
                    except Exception as e:
                        st.error(f"Error analyzing file: {e}")

                st.markdown("**Expected Format:**")
                if file_format == "jsonl":
                    st.code(
                        """{"prompt": "Question?", "target": "Answer"}
{"prompt": "Another question?", "target": "Another answer"}""",
                        language="json",
                    )
                elif file_format == "json":
                    st.code(
                        """[
  {"prompt": "Question?", "target": "Answer"},
  {"prompt": "Another question?", "target": "Another answer"}
]""",
                        language="json",
                    )
                elif file_format == "csv":
                    st.code(
                        '''prompt,target
"Question?","Answer"
"Another question?","Another answer"''',
                        language="csv",
                    )

            # Конфигурация столбцов (если файл загружен)
            if st.session_state.dataset_columns:
                st.divider()
                st.markdown("**Column Configuration**")

                col1, col2 = st.columns(2)

                with col1:
                    prompt_column = st.selectbox(
                        "Prompt Column*",
                        options=st.session_state.dataset_columns,
                        index=st.session_state.dataset_columns.index("prompt")
                        if "prompt" in st.session_state.dataset_columns
                        else 0,
                        help="Column containing the input prompts/questions",
                    )

                    include_column = st.selectbox(
                        "Include List Column (optional)",
                        options=["None"] + st.session_state.dataset_columns,
                        help="Column with words that must appear in output",
                    )

                with col2:
                    target_column = st.selectbox(
                        "Target Column (optional)",
                        options=["None"] + st.session_state.dataset_columns,
                        index=st.session_state.dataset_columns.index("target") + 1
                        if "target" in st.session_state.dataset_columns
                        else 0,
                        help="Column containing expected answers/outputs",
                    )

                    exclude_column = st.selectbox(
                        "Exclude List Column (optional)",
                        options=["None"] + st.session_state.dataset_columns,
                        help="Column with words that must NOT appear in output",
                    )

                # Показываем превью данных
                if st.session_state.dataset_preview:
                    st.markdown("**Data Preview:**")
                    import pandas as pd

                    df_preview = pd.DataFrame(st.session_state.dataset_preview[:3])
                    st.dataframe(df_preview, use_container_width=True)

            else:
                prompt_column = "prompt"
                target_column = "None"
                include_column = "None"
                exclude_column = "None"

            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "📤 Upload Dataset", type="primary", use_container_width=True
            )

            if submitted:
                if not name or not uploaded_file:
                    st.error("⚠️ Please provide dataset name and upload a file")
                else:
                    try:
                        tag_str = (
                            ",".join([t.strip() for t in tags.split(",")])
                            if tags
                            else ""
                        )

                        with st.spinner("Uploading data..."):
                            result = self.api_client.create_dataset_and_upload(
                                name=name,
                                description=description,
                                task_type=task_type,
                                tags=tag_str,
                                file=uploaded_file,
                                file_format=file_format,
                                prompt_column=prompt_column,
                                target_column=None
                                if target_column == "None"
                                else target_column,
                                include_column=None
                                if include_column == "None"
                                else include_column,
                                exclude_column=None
                                if exclude_column == "None"
                                else exclude_column,
                            )

                        count = result.get("items_uploaded", 0)
                        st.success(
                            f"✅ Dataset '{name}' uploaded successfully with {count} items!"
                        )
                        st.balloons()

                        # Очищаем состояние
                        st.session_state.dataset_columns = []
                        st.session_state.dataset_preview = []

                        time.sleep(1)
                        st.rerun()

                    except RequestException:
                        pass
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    def _extract_columns(self, file, file_format: str) -> list:
        """Извлечь названия столбцов из файла"""
        file.seek(0)  # Сброс позиции файла

        try:
            if file_format == "jsonl":
                line = file.readline().decode("utf-8")
                if line:
                    data = json.loads(line)
                    return list(data.keys())

            elif file_format == "json":
                content = file.read().decode("utf-8")
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0:
                    return list(data[0].keys())
                elif isinstance(data, dict):
                    items = data.get("data", data.get("items", []))
                    if items and len(items) > 0:
                        return list(items[0].keys())

            elif file_format == "csv":
                content = file.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(content))
                return reader.fieldnames or []

            elif file_format == "parquet":
                import pandas as pd

                df = pd.read_parquet(file)
                return df.columns.tolist()

        finally:
            file.seek(0)  # Снова сбрасываем позицию

        return []

    def _get_preview(self, file, file_format: str, n_rows: int = 3) -> list:
        """Получить превью данных"""
        file.seek(0)

        try:
            if file_format == "jsonl":
                preview = []
                for i, line in enumerate(file):
                    if i >= n_rows:
                        break
                    data = json.loads(line.decode("utf-8"))
                    preview.append(data)
                return preview

            elif file_format == "json":
                content = file.read().decode("utf-8")
                data = json.loads(content)
                if isinstance(data, list):
                    return data[:n_rows]
                elif isinstance(data, dict):
                    items = data.get("data", data.get("items", []))
                    return items[:n_rows]

            elif file_format == "csv":
                content = file.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(content))
                return [row for i, row in enumerate(reader) if i < n_rows]

            elif file_format == "parquet":
                import pandas as pd

                df = pd.read_parquet(file)
                return df.head(n_rows).to_dict("records")

        finally:
            file.seek(0)

        return []
