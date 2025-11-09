# src/ui/components/tasks_section.py
import time

import streamlit as st

from src.config.constants import EVALUATION_METRICS
from src.core.schemas.task import (
    ABTestConfig,
    ABTestStrategy,
    JudgeConfig,
    RTAConfig,
    TaskConfig,
    TaskType,
    VariationConfig,
    VariationStrategy,
)
from src.ui.api_client import get_api_client


def render_tasks_section():
    """Рендер секции создания задач"""

    st.markdown('<div class="animated">', unsafe_allow_html=True)
    st.markdown("## ➕ Create New Task")

    api_client = get_api_client()

    # Форма создания задачи
    with st.form("create_task_form", clear_on_submit=True):
        # ===== Базовая информация =====
        st.markdown("#### 📝 Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            task_name = st.text_input(
                "Task Name*",
                placeholder="e.g., Multi-Model Evaluation with Variations",
                help="A descriptive name for this task",
            )

        with col2:
            task_type = st.selectbox(
                "Task Type*",
                options=[t.value for t in TaskType],
                format_func=lambda x: x.replace("_", " ").title(),
                help="Type of task to perform",
            )

        st.divider()

        # ===== Датасет и модели =====
        st.markdown("#### 🗂️ Dataset & Models")

        col1, col2 = st.columns(2)

        with col1:
            # Датасет
            datasets = api_client.list_datasets()
            if datasets:
                dataset_options = {
                    f"{ds.name} ({ds.size} items)": ds.id for ds in datasets
                }
                selected_dataset_label = st.selectbox(
                    "Dataset*",
                    list(dataset_options.keys()),
                    help="Select the dataset to use",
                )
                selected_dataset = dataset_options[selected_dataset_label]
            else:
                st.warning("⚠️ No datasets available. Upload a dataset first.")
                selected_dataset = None

        with col2:
            # Модели (множественный выбор)
            models = api_client.list_models()
            if models:
                # Создаем словарь с именами моделей
                model_options = {f"{m.name} ({m.provider})": m.id for m in models}
                selected_model_labels = st.multiselect(
                    "Models* (select one or more)",
                    list(model_options.keys()),
                    help="Select models to test",
                )
                selected_models = [
                    model_options[label] for label in selected_model_labels
                ]
            else:
                st.warning("⚠️ No models available. Register a model first.")
                selected_models = []

        st.divider()

        # ===== Execution Settings =====
        st.markdown("#### ⚙️ Execution Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=100,
                value=1,
                help="Number of samples to process in parallel",
            )

        with col2:
            max_samples = st.number_input(
                "Max Samples (0 = all)",
                min_value=0,
                max_value=100000,
                value=0,
                help="Limit the number of samples",
            )

        with col3:
            evaluate = st.checkbox(
                "Enable Standard Evaluation",
                value=True,
                help="Calculate standard metrics",
            )

        if evaluate:
            available_metrics = EVALUATION_METRICS + ["include_exclude", "rta"]
            metrics = st.multiselect(
                "Evaluation Metrics",
                available_metrics,
                default=["exact_match", "accuracy"],
                help="Metrics to calculate",
            )
        else:
            metrics = []

        st.divider()

        # ===== Prompt Variations =====
        st.markdown("#### 🔄 Prompt Variations (Optional)")

        enable_variations = st.checkbox(
            "Enable Prompt Variations",
            value=False,
            help="Generate variations of each prompt",
        )

        if enable_variations:
            col1, col2 = st.columns(2)

            with col1:
                # Модель для вариаций
                if models:
                    variation_model_label = st.selectbox(
                        "Variation Model*",
                        list(model_options.keys()),
                        help="Model to generate variations",
                    )
                    variation_model_id = model_options[variation_model_label]
                else:
                    variation_model_id = None
                    st.warning("No models available for variations")

                variations_per_prompt = st.number_input(
                    "Variations per Prompt",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="How many variations to generate per strategy",
                )

            with col2:
                variation_strategies = st.multiselect(
                    "Variation Strategies*",
                    [s.value for s in VariationStrategy],
                    default=["paraphrase"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="How to vary the prompts",
                )

                # Кастомный промпт для вариаций
                use_custom_prompt = st.checkbox(
                    "Use Custom Variation Prompt",
                    help="Provide your own prompt template for variations",
                )

            if (
                use_custom_prompt
                and VariationStrategy.CUSTOM.value in variation_strategies
            ):
                variation_custom_prompt = st.text_area(
                    "Custom Variation Prompt",
                    placeholder="Transform the following prompt: {prompt}\n\nTransformed prompt:",
                    help="Use {prompt} as placeholder. You can also use {style}, {complexity}, etc.",
                )
            else:
                variation_custom_prompt = None
        else:
            variation_model_id = None
            variations_per_prompt = 0
            variation_strategies = []
            variation_custom_prompt = None

        st.divider()

        # ===== LLM Judge =====
        st.markdown("#### 🎯 LLM as a Judge (Optional)")

        enable_judge = st.checkbox(
            "Enable LLM Judge", value=False, help="Use an LLM to evaluate outputs"
        )

        if enable_judge:
            col1, col2 = st.columns(2)

            with col1:
                # Модель-судья
                if models:
                    judge_model_label = st.selectbox(
                        "Judge Model*",
                        list(model_options.keys()),
                        help="Model to use as judge",
                    )
                    judge_model_id = model_options[judge_model_label]
                else:
                    judge_model_id = None

                judge_criteria = st.multiselect(
                    "Evaluation Criteria",
                    [
                        "accuracy",
                        "relevance",
                        "completeness",
                        "clarity",
                        "coherence",
                        "correctness",
                        "safety",
                    ],
                    default=["accuracy", "relevance", "completeness"],
                    help="Criteria for the judge to evaluate",
                )

            with col2:
                use_custom_judge_prompt = st.checkbox(
                    "Use Custom Judge Prompt", help="Provide your own evaluation prompt"
                )

                if use_custom_judge_prompt:
                    judge_custom_prompt = st.text_area(
                        "Custom Judge Prompt Template",
                        placeholder="Evaluate this output...\nInput: {input_prompt}\nOutput: {model_output}\n...",
                        help="Use {input_prompt}, {model_output}, {criteria} as placeholders",
                    )
                else:
                    judge_custom_prompt = None
        else:
            judge_model_id = None
            judge_criteria = []
            judge_custom_prompt = None

        st.divider()

        # ===== Refuse-to-Answer =====
        st.markdown("#### 🛑 Refuse-to-Answer Detection (Optional)")

        enable_rta = st.checkbox(
            "Enable RTA Detection",
            value=False,
            help="Detect when models refuse to answer",
        )

        if enable_rta:
            col1, col2 = st.columns(2)

            with col1:
                # RTA судья
                if models:
                    rta_model_label = st.selectbox(
                        "RTA Judge Model*",
                        list(model_options.keys()),
                        help="Model to detect refusals",
                    )
                    rta_model_id = model_options[rta_model_label]
                else:
                    rta_model_id = None

                rta_keywords = st.text_area(
                    "Refusal Keywords (one per line)",
                    value="I cannot\nI can't\nI'm not able to\nI refuse\nI apologize, but\nI'm sorry, but",
                    help="Keywords that indicate refusal",
                )
                rta_keywords_list = [
                    k.strip() for k in rta_keywords.split("\n") if k.strip()
                ]

            with col2:
                use_custom_rta_prompt = st.checkbox(
                    "Use Custom RTA Prompt", help="Custom prompt for RTA detection"
                )

                if use_custom_rta_prompt:
                    rta_custom_prompt = st.text_area(
                        "Custom RTA Prompt",
                        placeholder="Analyze if this is a refusal:\nInput: {input_prompt}\nOutput: {model_output}\n...",
                        help="Use {input_prompt} and {model_output} as placeholders",
                    )
                else:
                    rta_custom_prompt = None
        else:
            rta_model_id = None
            rta_keywords_list = []
            rta_custom_prompt = None

        st.divider()

        # ===== A/B Tests =====
        st.markdown("#### 🧪 A/B Testing (Optional)")

        enable_ab_test = st.checkbox(
            "Enable A/B Testing", value=False, help="Statistical comparison of variants"
        )

        if enable_ab_test:
            col1, col2 = st.columns(2)

            with col1:
                ab_strategy = st.selectbox(
                    "A/B Test Strategy*",
                    [s.value for s in ABTestStrategy],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Type of A/B test to perform",
                )

                statistical_test = st.selectbox(
                    "Statistical Test",
                    ["t_test", "mann_whitney"],
                    format_func=lambda x: "T-Test"
                    if x == "t_test"
                    else "Mann-Whitney U",
                    help="Statistical test for significance",
                )

            with col2:
                sample_size_per_variant = st.number_input(
                    "Sample Size per Variant (0 = all)",
                    min_value=0,
                    max_value=10000,
                    value=0,
                    help="Number of samples for each variant",
                )

            # Специфичные настройки для каждой стратегии
            if ab_strategy == ABTestStrategy.PROMPT_VARIANTS.value:
                st.markdown("**Prompt Variants Configuration**")
                n_variants = st.number_input(
                    "Number of Variants", min_value=2, max_value=5, value=2
                )

                prompt_variants = {}
                for i in range(n_variants):
                    variant_name = st.text_input(
                        f"Variant {chr(65 + i)} Name",
                        value=f"variant_{chr(97 + i)}",
                        key=f"variant_name_{i}",
                    )
                    variant_prompt = st.text_area(
                        f"Variant {chr(65 + i)} Prompt Template",
                        placeholder=f"Prompt template {i + 1}... Use {{input}} as placeholder",
                        key=f"variant_prompt_{i}",
                    )
                    if variant_prompt:
                        prompt_variants[variant_name] = variant_prompt

            elif ab_strategy == ABTestStrategy.TEMPERATURE_TEST.value:
                st.markdown("**Temperature Values**")
                temp_input = st.text_input(
                    "Temperatures (comma-separated)",
                    value="0.3, 0.7, 1.0",
                    help="E.g., 0.3, 0.7, 1.0",
                )
                temperatures = [
                    float(t.strip()) for t in temp_input.split(",") if t.strip()
                ]
                prompt_variants = None

            elif ab_strategy == ABTestStrategy.SYSTEM_PROMPT_TEST.value:
                st.markdown("**System Prompt Variants**")
                n_system_prompts = st.number_input(
                    "Number of System Prompts", min_value=2, max_value=4, value=2
                )

                system_prompts = {}
                for i in range(n_system_prompts):
                    sp_name = st.text_input(
                        f"System Prompt {i + 1} Name",
                        value=f"system_{i + 1}",
                        key=f"sp_name_{i}",
                    )
                    sp_text = st.text_area(f"System Prompt {i + 1}", key=f"sp_text_{i}")
                    if sp_text:
                        system_prompts[sp_name] = sp_text

                temperatures = None
                prompt_variants = None

            else:
                temperatures = None
                prompt_variants = None
                system_prompts = None
        else:
            ab_strategy = None
            statistical_test = "t_test"
            sample_size_per_variant = 0
            temperatures = None
            prompt_variants = None
            system_prompts = None

        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 Launch Task", type="primary", use_container_width=True
            )

        if submitted:
            if not task_name or not selected_dataset or not selected_models:
                st.error("⚠️ Please fill in all required fields (name, dataset, models)")
            else:
                try:
                    # Создаем конфигурацию
                    config = TaskConfig(
                        batch_size=batch_size,
                        max_samples=max_samples if max_samples > 0 else None,
                        evaluate=evaluate,
                        evaluation_metrics=metrics,
                        # Variations
                        variations=VariationConfig(
                            enabled=enable_variations,
                            model_id=variation_model_id,
                            strategies=variation_strategies,
                            count_per_strategy=variations_per_prompt,
                            custom_prompt=variation_custom_prompt,
                        ),
                        # Judge
                        judge=JudgeConfig(
                            enabled=enable_judge,
                            model_id=judge_model_id,
                            criteria=judge_criteria,
                            custom_prompt_template=judge_custom_prompt,
                        ),
                        # RTA
                        rta=RTAConfig(
                            enabled=enable_rta,
                            rta_judge_model_id=rta_model_id,
                            rta_prompt_template=rta_custom_prompt,
                            refusal_keywords=rta_keywords_list,
                        ),
                        # A/B Test
                        ab_test=ABTestConfig(
                            enabled=enable_ab_test,
                            strategy=ab_strategy,
                            prompt_variants=prompt_variants
                            if ab_strategy == ABTestStrategy.PROMPT_VARIANTS.value
                            else None,
                            temperatures=temperatures
                            if ab_strategy == ABTestStrategy.TEMPERATURE_TEST.value
                            else None,
                            system_prompts=system_prompts
                            if ab_strategy == ABTestStrategy.SYSTEM_PROMPT_TEST.value
                            else None,
                            sample_size_per_variant=sample_size_per_variant
                            if sample_size_per_variant > 0
                            else None,
                            statistical_test=statistical_test,
                        ),
                    )

                    task = api_client.create_task(
                        task_data=dict(
                            name=task_name,
                            dataset_id=selected_dataset,
                            model_ids=selected_models,
                            task_type=task_type,
                            config=config.model_dump(),
                        )
                    )

                    st.success(f"✅ Task '{task_name}' created and launched!")

                    # Показываем информацию
                    info_parts = [f"- {len(selected_models)} model(s) selected"]
                    if enable_variations:
                        info_parts.append(
                            f"- Variations: {len(variation_strategies)} strategies"
                        )
                    if enable_judge:
                        info_parts.append("- LLM Judge: Enabled")
                    if enable_rta:
                        info_parts.append("- RTA Detection: Enabled")
                    if enable_ab_test:
                        info_parts.append(
                            f"- A/B Test: {ab_strategy.replace('_', ' ').title()}"
                        )

                    st.info("**Task Configuration:**\n" + "\n".join(info_parts))

                    st.balloons()
                    time.sleep(2)

                    # Переходим в GENERAL для просмотра
                    st.session_state.selected_section = "general"
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating task: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    st.markdown("</div>", unsafe_allow_html=True)
