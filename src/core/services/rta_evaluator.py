# src/core/services/rta_evaluator.py
from typing import Any, Dict, Optional

from src.adapters.factory import LLMFactory
from src.config.constants import RTA_PROMPT
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class RTAEvaluator:
    """Оценщик для Refuse-to-Answer задач"""

    def __init__(
        self,
        rta_judge_model_id: str,
        rta_prompt_template: Optional[str] = None,
        # refusal_keywords: Optional[List[str]] = None,
    ):
        self.rta_judge_model_id = rta_judge_model_id
        self.rta_prompt_template = rta_prompt_template or RTA_PROMPT
        self.model = None
        self.adapter = None

    async def initialize(self):
        """Инициализация модели RTA судьи"""
        model_repo = ModelRepository()
        self.model = await model_repo.find_by_id(self.rta_judge_model_id)
        if not self.model:
            raise ValueError(f"RTA judge model {self.rta_judge_model_id} not found")

        self.adapter = LLMFactory.create(self.model)
        logger.info(f"RTA Judge initialized with model: {self.model.name}")

    async def evaluate_output(
        self, input_prompt: str, model_output: str
    ) -> Dict[str, Any]:
        """
        Оценить, отказалась ли модель отвечать

        Returns:
            Dict with 'refused', 'confidence', 'reasoning'
        """
        if not self.adapter:
            await self.initialize()

        rta_input_prompt = self.rta_prompt_template.format(
            input_prompt=input_prompt, model_output=model_output
        )
        # logger.info(f"Rta input: {rta_input_prompt}")
        try:
            response = await self.adapter.generate(
                rta_input_prompt,
                temperature=0.2,  # Низкая температура для консистентности
                max_tokens=32768,
            )
            # logger.info(f"Rta response: {response}")
            # Парсим ответ
            result = self._parse_judge_response(response)

            logger.info(
                f"RTA evaluation: refused={result['refused']}, confidence={result['confidence']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during RTA evaluation: {e}")
            # Fallback на проверку по ключевым словам
            return {
                "refused": "0",
                "confidence": 0.5,
                "reasoning": f"Error occured: {str(e)}",
                "raw_response": response if response else "",
            }

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Парсинг ответа RTA судьи"""
        import json
        import re

        try:
            # Пытаемся найти JSON в ответе
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return {
                    "refused": data.get("refused", "0"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                    "raw_response": response,
                }
        except Exception as e:
            logger.warning(f"Failed to parse RTA judge response as JSON: {e}")

        # Fallback: простой парсинг
        refused = "1" if "1" in response.lower() else "0"

        return {
            "refused": refused,
            "confidence": 0.5,
            "reasoning": response,
            "raw_response": response,
        }
