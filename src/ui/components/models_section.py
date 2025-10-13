# src/ui/components/models_section.py
import asyncio

import streamlit as st

from src.config.constants import ModelProvider
from src.core.models.model import ModelConfig
from src.core.services.model_service import ModelService


def render_models_section():
    """Рендер секции управления моделями"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## 🤖 Model Management")

    # Инициализация
    if "model_service" not in st.session_state:
        st.session_state.model_service = ModelService()

    model_service = st.session_state.model_service

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
                ["All", "ollama", "huggingface", "openai", "anthropic"],
                key="model_provider_filter",
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        st.divider()

        # Загрузка моделей
        try:
            models = asyncio.run(model_service.list_models())

            if models:
                for model in models:
                    with st.container():
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

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
                                "anthropic": "🔵",
                            }
                            icon = provider_icons.get(model.provider, "🤖")
                            st.markdown(f"{icon} **{model.provider.upper()}**")

                            # Status
                            if model.is_active:
                                st.markdown(
                                    '<span class="status-badge status-completed">🟢 Active</span>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<span class="status-badge status-pending">⚫ Inactive</span>',
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
                                use_container_width=True,
                            ):
                                with st.spinner("Testing model..."):
                                    result = asyncio.run(
                                        model_service.test_model(
                                            model.id, "Hello, how are you?"
                                        )
                                    )

                                    if result["success"]:
                                        st.success(f"✅ {result['duration']:.2f}s")
                                        with st.expander("See response"):
                                            st.write(result["response"])
                                    else:
                                        st.error(f"❌ {result['error']}")

                            # Кнопка удаления
                            if st.button(
                                "🗑️ Delete",
                                key=f"del_model_{model.id}",
                                use_container_width=True,
                            ):
                                if st.session_state.get(
                                    f"confirm_delete_model_{model.id}"
                                ):
                                    asyncio.run(model_service.delete_model(model.id))
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.session_state[
                                        f"confirm_delete_model_{model.id}"
                                    ] = True
                                    st.warning("Click again")

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
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
                    max_value=8192,
                    value=512,
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
                    "🚀 Register Model", type="primary", use_container_width=True
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

                        model = asyncio.run(
                            model_service.register_model(
                                name=name,
                                provider=provider,
                                model_name=model_name,
                                description=description,
                                config=config,
                            )
                        )

                        st.success(f"✅ Model '{name}' registered successfully!")
                        st.balloons()

                        # Переключаемся на вкладку списка
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error registering model: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
