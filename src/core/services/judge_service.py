# src/core/services/judge_service.py
import json
import re
import traceback
from typing import Any, Dict, List, Optional

from src.adapters.factory import LLMFactory
from src.config.default_prompts import DEFAULT_JUDGE_EN
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger
from jinja2 import Template


def _parse_trustvar(data: dict, criteria: List[str], response: str) -> Dict:
    """Back-compat: TrustVar rubric parser."""
    scores = data.get("scores", {}) or {}
    overall_score: Optional[float] = None
    if scores:
        numeric = [
            float(v) for k, v in scores.items()
            if "score" in k and isinstance(v, (int, float))
        ]
        if numeric:
            overall_score = sum(numeric) / len(numeric)
    return {
        "overall_score": overall_score,
        "criteria_scores": {
            k: float(v)
            for k, v in scores.items()
            if "score" in k and isinstance(v, (int, float))
        },
        "results": data,
        "raw_response": response,
        "verdict": data.get("verdict"),
        "verdict_reasoning": data.get("verdict_reasoning"),
        "confidence": data.get("confidence"),
        "reasoning": None,
    }


def _parse_by_schema(
    data: dict, response: str, schema: dict, criteria: List[str]
) -> Dict:
    """Schema-aware extraction. Slots via x-slot; *_score fields -> criteria_scores."""
    slots: Dict[str, Any] = {
        "overall_score": None, "verdict": None, "verdict_reasoning": None,
        "confidence": None, "reasoning": None,
    }
    criteria_scores: Dict[str, float] = {}
    try:
        _walk_schema(data, schema, slots, criteria_scores)
    except Exception as e:
        return {
            "overall_score": None, "criteria_scores": {},
            "results": data, "raw_response": response,
            "verdict": None, "verdict_reasoning": None,
            "confidence": None, "reasoning": None,
            "error": f"schema_walk: {e}",
        }

    overall = slots.get("overall_score")
    if overall is not None:
        try:
            overall = float(overall)
        except (TypeError, ValueError):
            overall = None
            if isinstance(data, dict):
                data["error"] = "non_numeric_overall_score"
    if overall is None and criteria_scores:
        overall = sum(criteria_scores.values()) / len(criteria_scores)

    verdict = slots.get("verdict")
    if isinstance(verdict, bool):
        verdict = "PASS" if verdict else "FAIL"

    return {
        "overall_score": overall,
        "criteria_scores": criteria_scores,
        "results": data,
        "raw_response": response,
        "verdict": verdict,
        "verdict_reasoning": slots.get("verdict_reasoning"),
        "confidence": slots.get("confidence"),
        "reasoning": slots.get("reasoning"),
    }


def _walk_schema(data, schema, slots, criteria_scores, path=""):
    """DFS traverse JSON Schema + data; extract x-slot fields + *_score fields."""
    if not isinstance(schema, dict):
        return
    if isinstance(data, dict):
        if "properties" in schema and isinstance(schema["properties"], dict):
            for k, sub_schema in schema["properties"].items():
                if k not in data:
                    continue
                slot = sub_schema.get("x-slot") if isinstance(sub_schema, dict) else None
                if slot in slots and slots[slot] is None:
                    slots[slot] = data[k]
                if isinstance(k, str) and k.endswith("_score"):
                    v = data[k]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        criteria_scores[k] = float(v)
                _walk_schema(data[k], sub_schema, slots, criteria_scores,
                             f"{path}.{k}")
    elif isinstance(data, list):
        items_schema = schema.get("items", {}) if isinstance(schema, dict) else {}
        for el in data:
            _walk_schema(el, items_schema, slots, criteria_scores, f"{path}[]")


class LLMJudgeService:
    """Service for evaluating results with LLM"""

    # [AUGMENT 2026-07-11 default-prompts-seed] Was an inline RU rubric; now
    # points at the shared EN seed body. Name + usages unchanged; the
    # back-compat parser handles its JSON schema.
    DEFAULT_JUDGE_TEMPLATE = DEFAULT_JUDGE_EN

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.adapter = None

    async def initialize(self):
        """Initialize judge model"""
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
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Evaluate model output

        Returns:
            Dict with 'score', 'criteria_scores', 'reasoning'
        """
        if not self.adapter:
            await self.initialize()

        criteria = criteria or ["accuracy", "relevance", "completeness", "clarity"]

        # Prepare judge prompt
        judge_prompt = await self._prepare_judge_prompt(
            input_prompt=input_prompt,
            model_output=model_output,
            task_description=task_description,
            reference_output=reference_output,
            criteria=criteria,
            custom_template=custom_template,
            metadata=metadata,
        )

        # logger.info(f"Judge prompt {judge_prompt}")
        # Get evaluation from LLM
        try:
            response = await self.adapter.generate(
                judge_prompt,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=10000,
            )

            # Parse response
            result = self._parse_judge_response(response, criteria)

            # F3: overall_score is None on the no_verdict fallback — formatting it
            # with :.2f raised TypeError, and the outer except then swallowed the
            # structured payload (raw_response, error='no_verdict'). Guard the
            # format; do NOT widen the except.
            score = result.get("overall_score")
            logger.info(
                "Judge evaluation completed. Score: "
                + (f"{score:.2f}" if score is not None else "no verdict")
            )
            return result

        except Exception as e:
            logger.error(
                f"Error during LLM judge evaluation: {e}. Traceback: {traceback.format_exc()}"
            )
            return {
                "overall_score": None,
                "criteria_scores": {},
                "results": {"error": f"Evaluation failed: {str(e)}"},
                "error": str(e),
            }

    # async def evaluate_batch(
    #     self,
    #     evaluations: List[Dict],
    #     task_description: str = "General text generation",
    #     criteria: List[str] = None,
    # ) -> List[Dict]:
    #     """
    #     Batch evaluation
    #
    #     Args:
    #         evaluations: List of dicts with 'input', 'output', 'reference' (optional)
    #     """
    #     results = []
    #
    #     for eval_item in evaluations:
    #         result = await self.evaluate_output(
    #             input_prompt=eval_item["input"],
    #             model_output=eval_item["output"],
    #             task_description=task_description,
    #             reference_output=eval_item.get("reference"),
    #             criteria=criteria,
    #         )
    #         results.append(result)
    #
    #     return results

    async def compare_outputs(
        self,
        input_prompt: str,
        outputs: Dict[str, str],  # model_name -> output
        task_description: str = "General text generation",
        reference_output: Optional[str] = None,
        criteria: List[str] = None,
    ) -> Dict:
        """
        Compare outputs from different models

        Returns:
            Dict with rankings and detailed comparisons
        """
        if not self.adapter:
            await self.initialize()

        criteria = criteria or ["accuracy", "relevance", "completeness", "clarity"]

        # Build comparison prompt
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
            logger.error(
                f"Error during comparison: {e}. Traceback: {traceback.format_exc()}"
            )
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
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Prepare judge prompt (TrustVar rubric)."""

        template = custom_template or self.DEFAULT_JUDGE_TEMPLATE
        template_obj = Template(template)
        criteria_text = "\n".join(
            [f"- {c.replace('_', ' ').title()}" for c in criteria]
        )
        md = metadata or {}

        return template_obj.render(
            task_description=task_description,
            input_prompt=input_prompt,
            model_output=model_output,
            model_response=model_output,
            reference_output=reference_output,
            reference_present=bool(reference_output),
            criteria=criteria_text,
            task_type=md.get("task_type", "generation"),
            variation_type=md.get("variation_type"),
            original_input=md.get("original_input"),
        )

    def _parse_judge_response(
        self, response: str, criteria: List[str], output_schema: Optional[dict] = None
    ) -> Dict:
        """Dispatch: schema-aware if output_schema, else TrustVar back-compat."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if output_schema is None:
                    return _parse_trustvar(data, criteria, response)
                return _parse_by_schema(data, response, output_schema, criteria)
        except Exception as e:
            logger.warning(f"Failed to parse judge response as JSON: {e}")

        return {
            "overall_score": None, "criteria_scores": {},
            "results": {"response": response, "error": "no_verdict"},
            "raw_response": response,
            "verdict": None, "verdict_reasoning": None,
            "confidence": None, "reasoning": None,
        }

    def _parse_comparison_response(
        self, response: str, model_names: List[str], criteria: List[str]
    ) -> Dict:
        """Parse comparison response"""

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

    def _extract_score_from_text(self, text: str) -> Optional[float]:
        """Extract score from text by numeric pattern.

        F7: not called from main parsing path (JSON-first,
        ``_parse_judge_response``); kept as a utility with unit coverage.
        Returns ``None`` if no pattern found — annotation matches behavior.
        """

        # Look for patterns like "8/10", "8 out of 10", "score: 8"
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

        return None
