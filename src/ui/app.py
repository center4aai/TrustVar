# src/ui/app.py
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st

from src.ui.components.datasets_section import render_datasets_section
from src.ui.components.general_section import render_general_section
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

# ВАЖНО: Инициализация session state с защитой от дублирования
if "selected_section" not in st.session_state:
    st.session_state.selected_section = None

if "previous_section" not in st.session_state:
    st.session_state.previous_section = None

# Заголовок приложения
st.markdown(
    """
<div class="main-header">
    <h1 class="main-title">TrustVar</h1>
    <p class="main-subtitle">A Dynamic Framework for Trustworthiness Evaluation</p>
</div>
""",
    unsafe_allow_html=True,
)

# Карточки-кнопки для навигации
features = [
    {
        "icon": "🛡",
        "title": "GENERAL",
        "description": "Guide and monitoring",
        "key": "general",
    },
    {
        "icon": "📂",
        "title": "DATASETS",
        "description": "Upload and manage",
        "key": "datasets",
    },
    {
        "icon": "🧩",
        "title": "MODELS",
        "description": "Register and configure",
        "key": "models",
    },
    {
        "icon": "🚀",
        "title": "TASKS",
        "description": "Create and monitor",
        "key": "tasks",
    },
    {
        "icon": "🎯",
        "title": "RESULTS",
        "description": "Analyze performance",
        "key": "results",
    },
]


cols = st.columns(5)
for idx, (col, feature) in enumerate(zip(cols, features)):
    with col:
        button_text = f"{feature['icon']}\n\n ### {feature['title']}  \n\n {feature['description']}"

        if st.button(
            button_text, key=f"nav_{feature['key']}", use_container_width=True
        ):
            # Сохраняем предыдущую секцию
            st.session_state.previous_section = st.session_state.selected_section
            st.session_state.selected_section = feature["key"]

            # Очищаем временные данные при смене секции
            keys_to_clear = [
                "current_general_tab",
                "selected_task_id_for_results",
                "dashboard_placeholder",
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

# Отображаем выбранную секцию
if st.session_state.selected_section is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    # Кнопка закрытия секции
    col1, col2, col3 = st.columns([1, 10, 1])
    with col3:
        if st.button("✕ Close", key="close_section", help="Close current section"):
            st.session_state.previous_section = st.session_state.selected_section
            st.session_state.selected_section = None

            # Очищаем все временные данные
            keys_to_clear = [
                k for k in st.session_state.keys() if k.startswith("temp_")
            ]
            for key in keys_to_clear:
                del st.session_state[key]

            st.rerun()

    section_container = st.container()

    with section_container:
        # Отображаем соответствующий контент
        current_section = st.session_state.selected_section

        if current_section == "datasets":
            render_datasets_section()
        elif current_section == "models":
            render_models_section()
        elif current_section == "general":
            render_general_section()
        elif current_section == "tasks":
            render_tasks_section()
        elif current_section == "results":
            render_results_section()

else:
    # Показываем приветственную информацию
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<h3 style="text-align: center;">🧠 Quick Start Guide</h3><br>',
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


def cleanup_session_state():
    """Очистка временных данных из session state"""
    temp_keys = [
        "current_general_tab",
        "dashboard_placeholder",
        "confirm_cancel_dash_",
        "confirm_cancel_paused_",
    ]

    for key in list(st.session_state.keys()):
        if any(temp_key in key for temp_key in temp_keys):
            del st.session_state[key]


# Вызываем при каждом рендере
cleanup_session_state()
