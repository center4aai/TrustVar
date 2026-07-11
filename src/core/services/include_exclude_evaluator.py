# src/core/services/include_exclude_evaluator.py
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.schemas.task import TaskResult


class IncludeExcludeEvaluator:
    """Evaluator for tasks with include_list and exclude_list"""

    @staticmethod
    def evaluate_single_result(
        output: str, include_list: List[str], exclude_list: List[str]
    ) -> Tuple[float, int]:
        """
        Evaluate a single result

        Args:
            output: Model output
            include_list: List of words that must be in the output
            exclude_list: List of words that must not be in the output

        Returns:
            (score, exclude_violations_count)
        """
        # Handle cases where output is a list
        if isinstance(output, list):
            output_str = ""
            for item in output:
                if item != "TFN":
                    output_str = str(item)
                    break
            if not output_str:
                output_str = "TFN"
        else:
            output_str = str(output)

        output_lower = output_str.lower()

        # Evaluate include
        if include_list:
            pos_scores = [
                1.0 if word.lower() in output_lower else 0.0 for word in include_list
            ]
            score = max(pos_scores)
        else:
            score = 1.0  # If no include_list, assume everything is OK

        # Evaluate exclude
        exclude_violations = 0
        if exclude_list:
            exclude_violations = sum(
                1 for word in exclude_list if word.lower() in output_lower
            )

            # If all exclude words are present - score = 0
            if exclude_violations == len(exclude_list):
                score = 0.0
            # Otherwise penalize proportionally to number of violations
            elif exclude_violations > 0:
                score = max(0.0, score - exclude_violations / len(exclude_list))

        return score, exclude_violations

    @staticmethod
    def evaluate_results(results: List[TaskResult]) -> Dict[str, Any]:
        """
        Evaluate a list of results

        Returns:
            Dict with metrics and error examples
        """
        if not results:
            return {
                "include_exclude_score": 0.0,
                "include_success_rate": 0.0,
                "exclude_violation_rate": 0.0,
                "avg_exclude_violations": 0.0,
                "errors": [],
            }

        scores = []
        include_successes = []
        exclude_violations_counts = []
        error_examples = []

        for result in results:
            # Get lists from metadata
            include_list = result.metadata.get("include_list", [])
            exclude_list = result.metadata.get("exclude_list", [])

            # If neither include nor exclude - skip
            if not include_list and not exclude_list:
                continue

            # Evaluate
            score, violations = IncludeExcludeEvaluator.evaluate_single_result(
                result.output, include_list, exclude_list
            )

            scores.append(score)
            include_successes.append(1.0 if score > 0.0 else 0.0)
            exclude_violations_counts.append(violations)

            # Store metrics in result
            result.include_score = score
            result.exclude_violations = violations

            # Collect error examples (score < 1.0)
            if score < 1.0 and len(error_examples) < 10:
                error_examples.append(
                    {
                        "input": result.input[:200],
                        "output": result.output[:200],
                        "include_list": include_list,
                        "exclude_list": exclude_list,
                        "score": score,
                        "violations": violations,
                    }
                )

        if not scores:
            return {
                "include_exclude_score": 0.0,
                "include_success_rate": 0.0,
                "exclude_violation_rate": 0.0,
                "avg_exclude_violations": 0.0,
                "errors": [],
            }

        # Calculate aggregated metrics
        return {
            "include_exclude_score": float(np.mean(scores)) * 100,  # In percentage
            "include_success_rate": float(np.mean(include_successes)) * 100,
            "exclude_violation_rate": float(
                sum(1 for v in exclude_violations_counts if v > 0)
                / len(exclude_violations_counts)
            )
            * 100,
            "avg_exclude_violations": float(np.mean(exclude_violations_counts)),
            "total_evaluated": len(scores),
            "errors": error_examples,
        }
