# src/ui/components/models_section.py
import time

import streamlit as st

from src.config.constants import ModelProvider
from src.core.schemas.model import ModelConfig, ModelStatus
from src.core.services.model_service import ModelService
from src.ui.api_client import get_api_client


def render_models_section():
    """Рендер секции управления моделями"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 🤖 Model Management")

    # Инициализация
    if "model_service" not in st.session_state:
        st.session_state.model_service = ModelService()

    api_client = get_api_client()

    # Tabs
    tab1, tab2 = st.tabs(["📋 Registered Models", "➕ Register New"])

    # ===== TAB 1: Список моделей =====
    with tab1:
        # Фильтры
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            search = st.text_input(
                "🔍 Search", placeholder="Search models...", key="model_search"
            )

        with col2:
            provider_filter = st.selectbox(
                "Provider",
                ["All", "ollama", "huggingface", "openai"],
                key="model_provider_filter",
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", width="stretch"):
                st.rerun()

        st.divider()

        # Загрузка моделей
        try:
            models = api_client.list_models()

            if models:
                for model in models:
                    with st.container():
                        # st.markdown('<div class="custom-card">', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                        with col1:
                            st.markdown(f"### {model.name}")
                            st.caption(f"🔧 {model.model_name}")
                            st.caption(f"🆔 {model.id[:12]}...")

                        with col2:
                            # Provider icon
                            provider_icons = {
                                "ollama": "🦙",
                                "huggingface": "🤗",
                                "openai": "🟢",
                            }
                            icon = provider_icons.get(model.provider, "🤖")
                            st.markdown(f"{icon} **{model.provider.upper()}**")

                            # Status
                            if model.status == ModelStatus.REGISTERED:
                                st.markdown(
                                    '<span class="status-badge status-completed">🟢 Active</span>',
                                    unsafe_allow_html=True,
                                )
                            elif model.status == ModelStatus.DOWNLOADING:
                                st.markdown(
                                    '<span class="status-badge status-running">🟠 Downloading</span>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<span class="status-badge status-failed">🔴 Failed</span>',
                                    unsafe_allow_html=True,
                                )

                        with col3:
                            st.caption("Configuration")
                            st.write(f"🌡️ Temp: {model.config.temperature}")
                            st.write(f"📏 Max tokens: {model.config.max_tokens}")

                        with col4:
                            # Кнопка тестирования
                            if st.button(
                                "🧪 Test",
                                key=f"test_model_{model.id}",
                                width="stretch",
                            ):
                                # Запускаем тест через Celery
                                with st.spinner("Starting test..."):
                                    result = api_client.test_model(
                                        model.id, "Hello, how are you?"
                                    )

                                if result.get("celery_task_id"):
                                    st.session_state[f"test_task_{model.id}"] = result[
                                        "celery_task_id"
                                    ]
                                    st.info("⏳ Test started. Refresh to see results.")
                                else:
                                    st.error("Failed to start test")

                            # Проверяем результат теста (если есть задача)
                            test_task_id = st.session_state.get(f"test_task_{model.id}")
                            if test_task_id:
                                try:
                                    test_result = api_client.get_test_result(
                                        model.id, test_task_id
                                    )

                                    if test_result["status"] == "completed":
                                        result_data = test_result["result"]
                                        if result_data["success"]:
                                            st.success(
                                                f"✅ {result_data['duration']:.2f}s"
                                            )
                                            with st.expander("See response"):
                                                st.write(result_data["response"])
                                            # Очищаем задачу
                                            del st.session_state[
                                                f"test_task_{model.id}"
                                            ]
                                        else:
                                            st.error(f"❌ {result_data['error']}")
                                            del st.session_state[
                                                f"test_task_{model.id}"
                                            ]
                                    elif test_result["status"] == "pending":
                                        st.info(
                                            f"⏳ Testing... ({test_result.get('state', 'PENDING')})"
                                        )
                                    elif test_result["status"] == "failed":
                                        st.error("❌ Test failed")
                                        del st.session_state[f"test_task_{model.id}"]
                                except Exception as e:
                                    st.warning(f"Could not get test result: {e}")

                            # Кнопка удаления
                            if st.button(
                                "🗑️ Delete",
                                key=f"del_model_{model.id}",
                                width="stretch",
                            ):
                                if st.session_state.get(
                                    f"confirm_delete_model_{model.id}"
                                ):
                                    api_client.delete_model(model.id)
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.session_state[
                                        f"confirm_delete_model_{model.id}"
                                    ] = True
                                    st.warning("Click again")

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.divider()
            else:
                st.info("🤖 No models registered. Add your first model!")

        except Exception as e:
            st.error(f"❌ Error loading models: {e}")

    # ===== TAB 2: Регистрация новой модели =====
    with tab2:
        st.markdown("### ➕ Register New Model")

        with st.form("register_model_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Basic Information**")

                name = st.text_input(
                    "Model Name*",
                    placeholder="e.g., Llama 2 7B Chat",
                    help="A friendly name for the model",
                )

                provider = st.selectbox(
                    "Provider*",
                    options=[p.value for p in ModelProvider],
                    help="Select the model provider",
                )

                model_name = st.text_input(
                    "Model Identifier*",
                    placeholder="e.g., llama2:7b or gpt-4",
                    help="The exact model name/ID used by the provider",
                )

                description = st.text_area(
                    "Description",
                    placeholder="Brief description of the model...",
                    height=100,
                )

            with col2:
                st.markdown("**Configuration**")

                temperature = st.slider(
                    "Temperature",
                    0.0,
                    2.0,
                    0.7,
                    0.1,
                    help="Higher values make output more random",
                )

                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1,
                    max_value=65536,
                    value=8192,
                    help="Maximum length of generated text",
                )

                top_p = st.slider(
                    "Top P", 0.0, 1.0, 1.0, 0.05, help="Nucleus sampling parameter"
                )

                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=50,
                    help="Top-k sampling parameter",
                )

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button(
                    "🚀 Register Model", type="primary", width="stretch"
                )

            if submitted:
                if not name or not model_name:
                    st.error("⚠️ Please fill in all required fields (marked with *)")
                else:
                    try:
                        config = ModelConfig(
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k,
                        )

                        model = api_client.register_model(
                            model_data=dict(
                                name=name,
                                provider=provider,
                                model_name=model_name,
                                description=description,
                                config=config.model_dump(),
                            )
                        )

                        st.success(f"✅ Model '{name}' registered successfully!")

                        if provider in ["huggingface", "ollama"]:
                            st.info(
                                "🔄 Model downloading started in background. Check status in model list."
                            )

                        st.balloons()
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error registering model: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
