# src/core/tasks/variation_task.py
from typing import Dict, List

from src.adapters.factory import LLMFactory
from src.core.schemas.task import VariationStrategy
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class PromptVariationGenerator:
    """Генератор вариаций промптов"""

    VARIATION_TEMPLATES = {
        VariationStrategy.PARAPHRASE: """Rephrase the following prompt while keeping the same meaning:

Original prompt: {prompt}

Rephrased prompt:""",
        VariationStrategy.STYLE_CHANGE: """Rewrite the following prompt in a {style} style:

Original prompt: {prompt}

Rewritten prompt:""",
        VariationStrategy.COMPLEXITY: """Rewrite the following prompt making it {complexity}:

Original prompt: {prompt}

Rewritten prompt:""",
        VariationStrategy.LANGUAGE: """Translate the following prompt to {language}:

Original prompt: {prompt}

Translated prompt:""",
        VariationStrategy.PERSPECTIVE: """Rewrite the following prompt from a {perspective} perspective:

Original prompt: {prompt}

Rewritten prompt:""",
        VariationStrategy.CUSTOM: """{custom_instruction}

Original prompt: {prompt}

Modified prompt:""",
    }

    STYLE_OPTIONS = ["formal", "casual", "technical", "simple", "academic"]
    COMPLEXITY_OPTIONS = ["simpler", "more complex", "more detailed", "more concise"]
    PERSPECTIVE_OPTIONS = ["first person", "third person", "objective", "empathetic"]

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.adapter = None

    async def initialize(self):
        """Инициализация модели"""
        model_repo = ModelRepository()
        self.model = await model_repo.find_by_id(self.model_id)
        if not self.model:
            raise ValueError(f"Variation model {self.model_id} not found")

        self.adapter = LLMFactory.create(self.model)
        logger.info(f"Variation generator initialized with model: {self.model.name}")

    async def generate_variations(
        self,
        prompt: str,
        strategies: List[VariationStrategy],
        count_per_strategy: int = 1,
        custom_params: Dict = None,
    ) -> List[Dict[str, str]]:
        """
        Генерация вариаций промпта

        Returns:
            List of dicts with 'text', 'strategy', 'metadata'
        """
        if not self.adapter:
            await self.initialize()

        variations = []
        custom_params = custom_params or {}

        for strategy in strategies:
            for i in range(count_per_strategy):
                try:
                    variation_text = await self._generate_single_variation(
                        prompt, strategy, i, custom_params
                    )

                    variations.append(
                        {
                            "text": variation_text,
                            "strategy": strategy.value,
                            "metadata": {
                                "original": prompt,
                                "iteration": i,
                                "params": custom_params.get(strategy.value, {}),
                            },
                        }
                    )

                    logger.info(f"Generated variation {i + 1} using {strategy.value}")

                except Exception as e:
                    logger.error(f"Failed to generate variation with {strategy}: {e}")
                    continue

        return variations

    async def _generate_single_variation(
        self,
        prompt: str,
        strategy: VariationStrategy,
        iteration: int,
        custom_params: Dict,
    ) -> str:
        """Генерация одной вариации"""

        template = self.VARIATION_TEMPLATES.get(strategy)
        if not template:
            raise ValueError(f"Unknown variation strategy: {strategy}")

        # Подготавливаем параметры для шаблона
        template_params = {"prompt": prompt}

        if strategy == VariationStrategy.STYLE_CHANGE:
            styles = custom_params.get("styles", self.STYLE_OPTIONS)
            template_params["style"] = styles[iteration % len(styles)]

        elif strategy == VariationStrategy.COMPLEXITY:
            complexities = custom_params.get("complexities", self.COMPLEXITY_OPTIONS)
            template_params["complexity"] = complexities[iteration % len(complexities)]

        elif strategy == VariationStrategy.LANGUAGE:
            languages = custom_params.get("languages", ["Spanish", "French", "German"])
            template_params["language"] = languages[iteration % len(languages)]

        elif strategy == VariationStrategy.PERSPECTIVE:
            perspectives = custom_params.get("perspectives", self.PERSPECTIVE_OPTIONS)
            template_params["perspective"] = perspectives[iteration % len(perspectives)]

        elif strategy == VariationStrategy.CUSTOM:
            template_params["custom_instruction"] = custom_params.get(
                "custom_instruction", "Rewrite this prompt in a different way:"
            )

        # Форматируем промпт
        variation_prompt = template.format(**template_params)

        # Генерируем вариацию
        variation = await self.adapter.generate(
            variation_prompt,
            temperature=0.8,  # Более высокая температура для разнообразия
            max_tokens=self.model.config.max_tokens,
        )

        return variation.strip()
