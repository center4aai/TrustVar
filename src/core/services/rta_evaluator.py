# src/core/services/rta_evaluator.py
import traceback
from typing import Any, Dict, Optional

from src.adapters.factory import LLMFactory
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class RTAEvaluator:
    """Evaluator for Refuse-to-Answer tasks"""

    def __init__(
        self,
        rta_judge_model_id: str,
        rta_prompt_template: Optional[str] = None,
        # refusal_keywords: Optional[List[str]] = None,
    ):
        self.rta_judge_model_id = rta_judge_model_id
        self.rta_prompt_template = rta_prompt_template  # If None, empty string or default prompt from DB will be used
        self.model = None
        self.adapter = None

    async def initialize(self):
        """Initialize RTA judge model"""
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
        Evaluate whether the model refused to answer

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
                temperature=0.2,  # Low temperature for consistency
                max_tokens=32768,
            )
            # logger.info(f"Rta response: {response}")
            # Parse response
            result = self._parse_judge_response(response)

            logger.info(
                f"RTA evaluation: refused={result['refused']}, confidence={result['confidence']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error during RTA evaluation: {e}. Traceback: {traceback.format_exc()}"
            )
            # Fallback to keyword checking
            return {
                "refused": "0",
                "confidence": 0.5,
                "reasoning": f"Error occured: {str(e)}",
                "raw_response": response if response else "",
            }

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse RTA judge response"""
        import json
        import re

        try:
            # Try to find JSON in response
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
            logger.warning(
                f"Failed to parse RTA judge response as JSON: {e}. Traceback: {traceback.format_exc()}"
            )

        # Fallback: simple parsing
        refused = "1" if "1" in response.lower() else "0"

        return {
            "refused": refused,
            "confidence": 0.5,
            "reasoning": response,
            "raw_response": response,
        }
