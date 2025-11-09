# src/core/services/include_exclude_evaluator.py
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.schemas.task import TaskResult


class IncludeExcludeEvaluator:
    """Оценщик для задач с include_list и exclude_list"""

    @staticmethod
    def evaluate_single_result(
        output: str, include_list: List[str], exclude_list: List[str]
    ) -> Tuple[float, int]:
        """
        Оценить один результат

        Args:
            output: Выход модели
            include_list: Список слов, которые должны быть в выходе
            exclude_list: Список слов, которых не должно быть в выходе

        Returns:
            (score, exclude_violations_count)
        """
        # Обработка случаев, когда output - это список
        if isinstance(output, list):
            output_str = ""
            for item in output:
                if item != "TFN":  # TFN = Too Few Neurons / No answer
                    output_str = str(item)
                    break
            if not output_str:
                output_str = "TFN"
        else:
            output_str = str(output)

        output_lower = output_str.lower()

        # Оценка include
        if include_list:
            pos_scores = [
                1.0 if word.lower() in output_lower else 0.0 for word in include_list
            ]
            score = max(pos_scores)
        else:
            score = 1.0  # Если нет include_list, считаем что всё ок

        # Оценка exclude
        exclude_violations = 0
        if exclude_list:
            exclude_violations = sum(
                1 for word in exclude_list if word.lower() in output_lower
            )

            # Если все exclude слова присутствуют - score = 0
            if exclude_violations == len(exclude_list):
                score = 0.0
            # Иначе штрафуем пропорционально количеству нарушений
            elif exclude_violations > 0:
                score = max(0.0, score - exclude_violations / len(exclude_list))

        return score, exclude_violations

    @staticmethod
    def evaluate_results(results: List[TaskResult]) -> Dict[str, Any]:
        """
        Оценить список результатов

        Returns:
            Dict с метриками и примерами ошибок
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
            # Получаем списки из metadata
            include_list = result.metadata.get("include_list", [])
            exclude_list = result.metadata.get("exclude_list", [])

            # Если нет ни include, ни exclude - пропускаем
            if not include_list and not exclude_list:
                continue

            # Оцениваем
            score, violations = IncludeExcludeEvaluator.evaluate_single_result(
                result.output, include_list, exclude_list
            )

            scores.append(score)
            include_successes.append(1.0 if score > 0.0 else 0.0)
            exclude_violations_counts.append(violations)

            # Сохраняем метрики в результат
            result.include_score = score
            result.exclude_violations = violations

            # Собираем примеры ошибок (score < 1.0)
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

        # Вычисляем агрегированные метрики
        return {
            "include_exclude_score": float(np.mean(scores)) * 100,  # В процентах
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

    @staticmethod
    def create_dataframe_report(results: List[TaskResult]) -> pd.DataFrame:
        """
        Создать DataFrame отчет для детального анализа
        """
        data = []
        for result in results:
            include_list = result.metadata.get("include_list", [])
            exclude_list = result.metadata.get("exclude_list", [])

            if not include_list and not exclude_list:
                continue

            data.append(
                {
                    "input": result.input,
                    "output": result.output,
                    "include_list": include_list,
                    "exclude_list": exclude_list,
                    "include_score": result.include_score,
                    "exclude_violations": result.exclude_violations,
                    "model_id": result.model_id,
                }
            )

        return pd.DataFrame(data)
