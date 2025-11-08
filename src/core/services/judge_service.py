# src/core/services/judge_service.py
import json
import re
from typing import Dict, List, Optional

from src.adapters.factory import LLMFactory
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class LLMJudgeService:
    """Сервис для оценки результатов с помощью LLM"""

    DEFAULT_JUDGE_TEMPLATE = """You are an expert evaluator assessing AI model outputs.

Task: {task_description}

Input Prompt: {input_prompt}

Model Output: {model_output}

{reference_section}

Evaluation Criteria:
{criteria}

Please evaluate the output on a scale of 1-10 for each criterion and provide your reasoning.
Respond in the following JSON format:
{{
    "scores": {{
        "criterion1": score,
        "criterion2": score,
        ...
    }},
    "overall_score": average_score,
    "reasoning": "Your detailed reasoning here"
}}"""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.adapter = None

    async def initialize(self):
        """Инициализация модели судьи"""
        model_repo = ModelRepository()
        self.model = await model_repo.find_by_id(self.model_id)
        if not self.model:
            raise ValueError(f"Judge model {self.model_id} not found")

        self.adapter = LLMFactory.create(self.model)
        logger.info(f"LLM Judge initialized with model: {self.model.name}")

    async def evaluate_output(
        self,
        input_prompt: str,
        model_output: str,
        task_description: str = "General text generation",
        reference_output: Optional[str] = None,
        criteria: List[str] = None,
        custom_template: Optional[str] = None,
    ) -> Dict:
        """
        Оценка выхода модели

        Returns:
            Dict with 'score', 'criteria_scores', 'reasoning'
        """
        if not self.adapter:
            await self.initialize()

        criteria = criteria or ["accuracy", "relevance", "completeness", "clarity"]

        # Подготавливаем промпт для судьи
        judge_prompt = await self._prepare_judge_prompt(
            input_prompt=input_prompt,
            model_output=model_output,
            task_description=task_description,
            reference_output=reference_output,
            criteria=criteria,
            custom_template=custom_template,
        )

        # Получаем оценку от LLM
        try:
            response = await self.adapter.generate(
                judge_prompt,
                temperature=0.3,  # Низкая температура для консистентности
                max_tokens=1000,
            )

            # Парсим ответ
            result = self._parse_judge_response(response, criteria)

            logger.info(
                f"Judge evaluation completed. Score: {result['overall_score']:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error during LLM judge evaluation: {e}")
            return {
                "overall_score": 0.0,
                "criteria_scores": {c: 0.0 for c in criteria},
                "reasoning": f"Evaluation failed: {str(e)}",
                "error": str(e),
            }

    async def evaluate_batch(
        self,
        evaluations: List[Dict],
        task_description: str = "General text generation",
        criteria: List[str] = None,
    ) -> List[Dict]:
        """
        Пакетная оценка

        Args:
            evaluations: List of dicts with 'input', 'output', 'reference' (optional)
        """
        results = []

        for eval_item in evaluations:
            result = await self.evaluate_output(
                input_prompt=eval_item["input"],
                model_output=eval_item["output"],
                task_description=task_description,
                reference_output=eval_item.get("reference"),
                criteria=criteria,
            )
            results.append(result)

        return results

    async def compare_outputs(
        self,
        input_prompt: str,
        outputs: Dict[str, str],  # model_name -> output
        task_description: str = "General text generation",
        reference_output: Optional[str] = None,
        criteria: List[str] = None,
    ) -> Dict:
        """
        Сравнение выходов разных моделей

        Returns:
            Dict with rankings and detailed comparisons
        """
        if not self.adapter:
            await self.initialize()

        criteria = criteria or ["accuracy", "relevance", "completeness", "clarity"]

        # Формируем промпт для сравнения
        comparison_prompt = self._prepare_comparison_prompt(
            input_prompt=input_prompt,
            outputs=outputs,
            task_description=task_description,
            reference_output=reference_output,
            criteria=criteria,
        )

        try:
            response = await self.adapter.generate(
                comparison_prompt, temperature=0.3, max_tokens=1500
            )

            result = self._parse_comparison_response(
                response, list(outputs.keys()), criteria
            )

            logger.info(
                f"Comparison completed. Winner: {result.get('best_model', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"Error during comparison: {e}")
            return {
                "rankings": {},
                "reasoning": f"Comparison failed: {str(e)}",
                "error": str(e),
            }

    async def _prepare_judge_prompt(
        self,
        input_prompt: str,
        model_output: str,
        task_description: str,
        reference_output: Optional[str],
        criteria: List[str],
        custom_template: Optional[str],
    ) -> str:
        """Подготовка промпта для судьи"""

        template = custom_template or self.DEFAULT_JUDGE_TEMPLATE

        # Форматируем критерии
        criteria_text = "\n".join(
            [f"- {c.replace('_', ' ').title()}" for c in criteria]
        )

        # Референс (если есть)
        reference_section = ""
        if reference_output:
            reference_section = f"\nExpected/Reference Output: {reference_output}\n"

        return template.format(
            task_description=task_description,
            input_prompt=input_prompt,
            model_output=model_output,
            reference_section=reference_section,
            criteria=criteria_text,
        )

    def _prepare_comparison_prompt(
        self,
        input_prompt: str,
        outputs: Dict[str, str],
        task_description: str,
        reference_output: Optional[str],
        criteria: List[str],
    ) -> str:
        """Подготовка промпта для сравнения"""

        outputs_text = "\n\n".join(
            [f"Model {name}:\n{output}" for name, output in outputs.items()]
        )

        reference_section = ""
        if reference_output:
            reference_section = f"\nExpected Output: {reference_output}\n"

        criteria_text = "\n".join(
            [f"- {c.replace('_', ' ').title()}" for c in criteria]
        )

        return f"""You are an expert evaluator comparing outputs from different AI models.

Task: {task_description}

Input Prompt: {input_prompt}
{reference_section}

Model Outputs:
{outputs_text}

Evaluation Criteria:
{criteria_text}

Please:
1. Evaluate each model's output on each criterion (scale 1-10)
2. Rank the models from best to worst
3. Provide detailed reasoning for your rankings

Respond in JSON format:
{{
    "rankings": {{
        "model_name": {{"rank": 1, "overall_score": 8.5, "scores": {{"criterion": score, ...}}}},
        ...
    }},
    "best_model": "model_name",
    "reasoning": "Detailed comparison and reasoning"
}}"""

    def _parse_judge_response(self, response: str, criteria: List[str]) -> Dict:
        """Парсинг ответа судьи"""

        try:
            # Пытаемся найти JSON в ответе
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return {
                    "overall_score": float(data.get("overall_score", 0)),
                    "criteria_scores": {
                        k: float(v) for k, v in data.get("scores", {}).items()
                    },
                    "reasoning": data.get("reasoning", ""),
                    "raw_response": response,
                }
        except Exception as e:
            logger.warning(f"Failed to parse judge response as JSON: {e}")

        # Fallback: простой парсинг
        overall_score = self._extract_score_from_text(response)

        return {
            "overall_score": overall_score,
            "criteria_scores": {c: overall_score for c in criteria},
            "reasoning": response,
            "raw_response": response,
        }

    def _parse_comparison_response(
        self, response: str, model_names: List[str], criteria: List[str]
    ) -> Dict:
        """Парсинг ответа сравнения"""

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "rankings": data.get("rankings", {}),
                    "best_model": data.get("best_model", ""),
                    "reasoning": data.get("reasoning", ""),
                    "raw_response": response,
                }
        except Exception as e:
            logger.warning(f"Failed to parse comparison response: {e}")

        return {
            "rankings": {},
            "best_model": model_names[0] if model_names else "",
            "reasoning": response,
            "raw_response": response,
        }

    def _extract_score_from_text(self, text: str) -> float:
        """Извлечение оценки из текста"""

        # Ищем паттерны типа "8/10", "8 out of 10", "score: 8"
        patterns = [
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s+out of\s+10",
            r"score:?\s*(\d+(?:\.\d+)?)",
            r"rating:?\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return 5.0  # Default средняя оценка
