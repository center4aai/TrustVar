# src/ui/app.py
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st

from src.ui.components.datasets_section import render_datasets_section
from src.ui.components.models_section import render_models_section
from src.ui.components.results_section import render_results_section
from src.ui.components.tasks_section import render_tasks_section
from src.ui.styles.custom_styles import apply_custom_styles

st.set_page_config(
    page_title="TrustVar",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Применяем кастомные стили
apply_custom_styles()

# Инициализация состояния
if "selected_section" not in st.session_state:
    st.session_state.selected_section = None

# Заголовок приложения
st.markdown(
    """
<div class="main-header">
    <h1 class="main-title">TrustVar</h1>
    <p class="main-subtitle">A Dynamic Framework for Trustworthiness Evaluation and Task Variation Analysis in LLMs</p>
</div>
""",
    unsafe_allow_html=True,
)

# Карточки-кнопки для навигации
features = [
    {
        "icon": "\U0001f4c2",
        "title": "DATASETS",
        "description": "Upload and manage test datasets",
        "key": "datasets",
    },
    {
        "icon": "\U0001f9e9",
        "title": "MODELS",
        "description": "Register and configure LLM models",
        "key": "models",
    },
    {
        "icon": "\U0001f6e1",
        "title": "TEMPLATES",
        "description": "Define evaluation templates and metrics",
        "key": "templates",
    },
    {
        "icon": "\U0001f680",
        "title": "TASKS",
        "description": "Create and monitor tasks for testing",
        "key": "tasks",
    },
    {
        "icon": "\U0001f3af",
        "title": "RESULTS",
        "description": "Analyze performance and metrics",
        "key": "results",
    },
]

# Создаем 4 колонки для карточек
cols = st.columns(5)
for idx, (col, feature) in enumerate(zip(cols, features)):
    with col:
        button_text = f"{feature['icon']}\n\n ### {feature['title']}  \n\n {feature['description']}"

        if st.button(button_text, key=f"nav_{feature['key']}", width="stretch"):
            st.session_state.selected_section = feature["key"]
            # st.rerun()

# Отображаем выбранную секцию
if st.session_state.selected_section is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    # Кнопка закрытия секции
    col1, col2, col3 = st.columns([1, 10, 1])
    with col3:
        if st.button("✕ Close", key="close_section", help="Close current section"):
            st.session_state.selected_section = None
            st.rerun()

    # Отображаем соответствующий контент
    if st.session_state.selected_section == "datasets":
        render_datasets_section()
    elif st.session_state.selected_section == "models":
        render_models_section()
    elif st.session_state.selected_section == "templates":
        render_tasks_section()  # TODO: think about templates
    elif st.session_state.selected_section == "tasks":
        render_tasks_section()
    elif st.session_state.selected_section == "results":
        render_results_section()


else:
    # Показываем приветственную информацию, когда ничего не выбрано
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<h3 style="text-align: center;">\U0001f9e0 Quick Start Guide</h3><br>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="info-card">
            <h3>1️⃣ Prepare Data</h3>
            <p>Upload your test datasets in JSONL, JSON, or CSV format</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="info-card">
            <h3>2️⃣ Add Models</h3>
            <p>Register models from Ollama, HuggingFace, or API providers</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="info-card">
            <h3>3️⃣ Run Tests</h3>
            <p>Create tasks to evaluate models on your datasets</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="info-card">
            <h3>4️⃣ Analyze</h3>
            <p>Review results with interactive visualizations</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
