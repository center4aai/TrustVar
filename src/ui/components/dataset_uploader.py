# src/ui/components/dataset_uploader.py
import csv
import io
import json
from pathlib import Path

import streamlit as st
from requests.exceptions import RequestException

from src.config.constants import SUPPORTED_TASKS
from src.ui.api_client import ApiClient


class DatasetUploader:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

    def _detect_format_from_file(self, file) -> str:
        """Автоматически определить формат файла по расширению и содержимому"""
        filename = file.name
        ext = Path(filename).suffix.lower()

        # Простые случаи - по расширению
        if ext == ".csv":
            return "csv"
        elif ext == ".parquet":
            return "parquet"
        elif ext == ".jsonl":
            return "jsonl"
        elif ext == ".json":
            return self._detect_json_format(file)

        return "json"

    def _detect_json_format(self, file) -> str:
        """Определить, является ли JSON файл обычным JSON или JSONL"""
        try:
            file.seek(0)
            # Читаем первые несколько байт
            first_line = file.readline().decode("utf-8").strip()
            file.seek(0)

            # Если первая строка начинается с '[' - это JSON array
            if first_line.startswith("["):
                return "json"

            # Если первая строка - валидный JSON объект (начинается с '{')
            if first_line.startswith("{"):
                try:
                    # Проверяем, можем ли мы распарсить первую строку как объект
                    json.loads(first_line)

                    # Читаем вторую строку
                    file.seek(0)
                    file.readline()  # Пропускаем первую
                    second_line = file.readline().decode("utf-8").strip()
                    file.seek(0)

                    # Если есть вторая строка и она тоже JSON объект - это JSONL
                    if second_line and second_line.startswith("{"):
                        try:
                            json.loads(second_line)
                            return "jsonl"
                        except:
                            pass

                    # Иначе пытаемся прочитать весь файл как JSON
                    content = file.read().decode("utf-8")
                    file.seek(0)
                    json.loads(content)
                    return "json"

                except json.JSONDecodeError:
                    pass

        except Exception:
            pass
        finally:
            file.seek(0)

        return "json"

    def render(self):
        st.markdown("### ➕ Upload New Dataset")

        # Проверяем, был ли недавно загружен датасет
        if "upload_success_message" in st.session_state:
            st.success(st.session_state.upload_success_message)
            st.balloons()
            del st.session_state.upload_success_message

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 View All Datasets", type="primary", width="stretch"):
                    st.session_state.active_dataset_tab = 0
                    st.rerun()
            with col2:
                if st.button("➕ Upload Another", width="stretch"):
                    st.rerun()

            st.markdown("---")

        # File uploader ВНЕ формы
        st.markdown("**📁 Upload File**")

        uploaded_file = st.file_uploader(
            "Choose your dataset file",
            type=["jsonl", "json", "csv", "parquet"],
            help="Upload your dataset file (format will be detected automatically)",
            key="dataset_file_uploader",
        )

        # Переменные для хранения данных анализа (локальные, не в session_state)
        detected_format = None
        columns = []
        preview = []

        # Автоматическое определение формата и анализ файла
        if uploaded_file is not None:
            # Определяем формат
            detected_format = self._detect_format_from_file(uploaded_file)

            # Анализируем файл
            try:
                columns = self._extract_columns(uploaded_file, detected_format)
                preview = self._get_preview(uploaded_file, detected_format)
            except Exception as e:
                st.error(f"❌ Error analyzing file: {e}")

            # Показываем информацию о файле
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f"📄 **File:** {uploaded_file.name}")
            with col2:
                # Показываем детектированный формат с пояснением для JSON
                format_display = (
                    detected_format.upper() if detected_format else "Unknown"
                )
                if (
                    Path(uploaded_file.name).suffix.lower() == ".json"
                    and detected_format
                ):
                    if detected_format == "jsonl":
                        format_display = "JSONL (JSON Lines)"
                    else:
                        format_display = "JSON (Array)"
                st.success(f"**Format:** {format_display}")
            with col3:
                size_kb = uploaded_file.size / 1024
                size_str = (
                    f"{size_kb:.1f} KB"
                    if size_kb < 1024
                    else f"{size_kb / 1024:.1f} MB"
                )
                st.metric("Size", size_str)

            # Показываем результат анализа
            if columns:
                st.success(
                    f"✅ Found {len(columns)} columns: {', '.join(columns[:5])}{('...' if len(columns) > 5 else '')}"
                )
            else:
                st.warning("⚠️ No columns detected. Please check your file format.")

        # Показываем превью данных (вне формы для лучшей видимости)
        if preview:
            st.markdown("---")
            st.markdown("**📊 Data Preview**")
            import pandas as pd

            df_preview = pd.DataFrame(preview[:5])
            st.dataframe(df_preview, width="stretch", hide_index=True)

        # Показываем expected format только если файл не загружен
        if uploaded_file is None:
            st.markdown("---")
            st.markdown("**💡 Supported Formats & Examples**")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["JSON Lines", "JSON Array", "CSV", "Parquet"]
            )

            with tab1:
                st.markdown("**JSON Lines format** (`.jsonl` or `.json`)")
                st.caption("Each line is a separate JSON object")
                st.code(
                    """{"prompt": "What is AI?", "target": "Artificial Intelligence..."}
{"prompt": "Explain ML", "target": "Machine Learning..."}""",
                    language="json",
                )

            with tab2:
                st.markdown("**JSON array format** (`.json`)")
                st.caption("Array of JSON objects")
                st.code(
                    """[
  {"prompt": "What is AI?", "target": "Artificial Intelligence..."},
  {"prompt": "Explain ML", "target": "Machine Learning..."}
]""",
                    language="json",
                )

            with tab3:
                st.markdown("**CSV format** (`.csv`)")
                st.caption("Comma-separated values with header")
                st.code(
                    '''prompt,target
"What is AI?","Artificial Intelligence..."
"Explain ML","Machine Learning..."''',
                    language="csv",
                )

            with tab4:
                st.markdown("**Parquet format** (`.parquet`)")
                st.info(
                    "Apache Parquet binary columnar format - efficient for large datasets"
                )

        st.markdown("---")

        # Форма начинается здесь
        with st.form("upload_dataset_form", clear_on_submit=True):
            st.markdown("**📝 Dataset Information**")

            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input(
                    "Dataset Name*",
                    placeholder="e.g., QA Test Set v1",
                    help="A unique name for your dataset",
                )

                task_type = st.selectbox(
                    "Task Type*",
                    SUPPORTED_TASKS,
                    help="The type of task this dataset is for",
                )

            with col2:
                description = st.text_area(
                    "Description",
                    placeholder="Brief description of the dataset...",
                    height=100,
                )

                tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="e.g., qa, english, test",
                    help="Tags to help organize datasets",
                )

            # Конфигурация столбцов (если файл загружен)
            if columns:
                st.markdown("---")
                st.markdown("**⚙️ Column Mapping**")
                st.caption("Map your dataset columns to the required fields")

                col1, col2 = st.columns(2)

                with col1:
                    # Безопасный выбор индекса для prompt
                    prompt_default_idx = 0
                    prompt_candidates = ["prompt", "question", "input", "query", "text"]
                    for candidate in prompt_candidates:
                        if candidate in columns:
                            prompt_default_idx = columns.index(candidate)
                            break

                    prompt_column = st.selectbox(
                        "🎯 Prompt Column* (Required)",
                        options=columns,
                        index=prompt_default_idx,
                        help="Column containing the input prompts/questions",
                    )

                    # Безопасный выбор индекса для target
                    target_options = ["None"] + columns
                    target_default_idx = 0
                    target_candidates = [
                        "target",
                        "answer",
                        "output",
                        "response",
                        "completion",
                    ]
                    for candidate in target_candidates:
                        if candidate in columns:
                            target_default_idx = columns.index(candidate) + 1
                            break

                    target_column = st.selectbox(
                        "✓ Target Column (Optional)",
                        options=target_options,
                        index=target_default_idx,
                        help="Column containing expected answers/outputs",
                    )

                    target_column_default_value = st.text_input(
                        "Default Target value (Optional)",
                        placeholder="e.g., '1' or '5' or any other text",
                    )

                with col2:
                    include_column = st.selectbox(
                        "➕ Include List Column (Optional)",
                        options=["None"] + columns,
                        help="Column with words that must appear in output",
                    )

                    exclude_column = st.selectbox(
                        "➖ Exclude List Column (Optional)",
                        options=["None"] + columns,
                        help="Column with words that must NOT appear in output",
                    )

            else:
                prompt_column = "prompt"
                target_column = "None"
                include_column = "None"
                exclude_column = "None"

                if uploaded_file is None:
                    st.warning(
                        "👆 Please upload a file first to configure column mapping"
                    )
                else:
                    st.error(
                        "⚠️ Could not detect columns in the uploaded file. Please check the file format."
                    )

            # Submit button
            st.markdown("---")

            submitted = st.form_submit_button(
                "🚀 Upload Dataset", type="primary", width="stretch"
            )

            if submitted:
                if not name:
                    st.error("⚠️ Please provide a dataset name")
                elif not uploaded_file:
                    st.error("⚠️ Please upload a file")
                elif not columns:
                    st.error("⚠️ Could not detect columns in the uploaded file")
                else:
                    try:
                        tag_str = (
                            ",".join([t.strip() for t in tags.split(",")])
                            if tags
                            else ""
                        )

                        with st.spinner("⏳ Uploading dataset..."):
                            result = self.api_client.create_dataset_and_upload(
                                name=name,
                                description=description,
                                task_type=task_type,
                                tags=tag_str,
                                file=uploaded_file,
                                file_format=detected_format,
                                prompt_column=prompt_column,
                                target_column=target_column
                                or str(target_column_default_value) + "_default",
                                include_column=None
                                if include_column == "None"
                                else include_column,
                                exclude_column=None
                                if exclude_column == "None"
                                else exclude_column,
                            )

                        count = result.get("items_uploaded", 0)

                        # Сохраняем сообщение об успехе и остаемся на вкладке Upload
                        st.session_state.upload_success_message = f"✅ Dataset '{name}' uploaded successfully with {count} items!"
                        st.session_state.active_dataset_tab = (
                            1  # Остаемся на Upload New
                        )

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
                if line.strip():
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
                return list(reader.fieldnames) if reader.fieldnames else []

            elif file_format == "parquet":
                import pandas as pd

                df = pd.read_parquet(file)
                return df.columns.tolist()

        except Exception as e:
            raise Exception(f"Failed to extract columns: {str(e)}")
        finally:
            file.seek(0)  # Снова сбрасываем позицию

        return []

    def _get_preview(self, file, file_format: str, n_rows: int = 5) -> list:
        """Получить превью данных"""
        file.seek(0)

        try:
            if file_format == "jsonl":
                preview = []
                for i, line in enumerate(file):
                    if i >= n_rows:
                        break
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        data = json.loads(line_str)
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

        except Exception as e:
            raise Exception(f"Failed to generate preview: {str(e)}")
        finally:
            file.seek(0)

        return []
