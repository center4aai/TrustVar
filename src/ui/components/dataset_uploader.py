# src/ui/components/dataset_uploader.py
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

            # Submit button
            # st.markdown("<br>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "📤 Upload Dataset", type="primary", width="stretch"
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
                            )

                        count = result.get("items_uploaded", 0)
                        st.success(
                            f"✅ Dataset '{name}' uploaded successfully with {count} items!"
                        )
                        st.balloons()

                        time.sleep(1)
                        st.rerun()

                    except RequestException:
                        # Ошибка уже будет показана в api_client, здесь можно не дублировать
                        pass
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
