# src/core/services/ab_test_analyzer.py
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from src.core.schemas.task import TaskResult
from src.utils.logger import logger


class ABTestAnalyzer:
    """Анализатор A/B тестов с статистическими тестами"""

    @staticmethod
    def group_results_by_variant(
        results: List[TaskResult],
    ) -> Dict[str, List[TaskResult]]:
        """Группировка результатов по вариантам"""
        grouped = defaultdict(list)
        for result in results:
            variant = result.ab_variant or "default"
            grouped[variant].append(result)
        return dict(grouped)

    @staticmethod
    def compute_variant_metrics(
        results: List[TaskResult], metric_names: List[str]
    ) -> Dict[str, float]:
        """Вычислить метрики для одного варианта"""
        if not results:
            return {}

        metrics = {}

        # Базовые метрики
        metrics["count"] = len(results)
        metrics["avg_execution_time"] = np.mean([r.execution_time for r in results])

        # Judge scores (если есть)
        judge_scores = [r.judge_score for r in results if r.judge_score is not None]
        if judge_scores:
            metrics["avg_judge_score"] = np.mean(judge_scores)
            metrics["std_judge_score"] = np.std(judge_scores)

        # Include/Exclude scores (если есть)
        include_scores = [
            r.include_score for r in results if r.include_score is not None
        ]
        if include_scores:
            metrics["avg_include_score"] = np.mean(include_scores)

        # Custom metrics из результатов
        for metric_name in metric_names:
            values = [
                r.metrics.get(metric_name) for r in results if metric_name in r.metrics
            ]
            if values:
                metrics[f"avg_{metric_name}"] = np.mean(values)
                metrics[f"std_{metric_name}"] = np.std(values)

        return metrics

    @staticmethod
    def t_test(
        variant_a_scores: List[float], variant_b_scores: List[float]
    ) -> Dict[str, Any]:
        """T-test для сравнения двух вариантов"""
        if len(variant_a_scores) < 2 or len(variant_b_scores) < 2:
            return {
                "test": "t_test",
                "statistic": None,
                "p_value": None,
                "significant": False,
                "error": "Not enough samples",
            }

        try:
            statistic, p_value = stats.ttest_ind(variant_a_scores, variant_b_scores)

            return {
                "test": "t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "confidence_level": 0.95,
                "mean_a": np.mean(variant_a_scores),
                "mean_b": np.mean(variant_b_scores),
                "std_a": np.std(variant_a_scores),
                "std_b": np.std(variant_b_scores),
            }
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return {"test": "t_test", "error": str(e), "significant": False}

    @staticmethod
    def mann_whitney_test(
        variant_a_scores: List[float], variant_b_scores: List[float]
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (непараметрический)"""
        if len(variant_a_scores) < 2 or len(variant_b_scores) < 2:
            return {
                "test": "mann_whitney",
                "statistic": None,
                "p_value": None,
                "significant": False,
                "error": "Not enough samples",
            }

        try:
            statistic, p_value = stats.mannwhitneyu(
                variant_a_scores, variant_b_scores, alternative="two-sided"
            )

            return {
                "test": "mann_whitney",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "confidence_level": 0.95,
                "median_a": np.median(variant_a_scores),
                "median_b": np.median(variant_b_scores),
            }
        except Exception as e:
            logger.error(f"Mann-Whitney test failed: {e}")
            return {"test": "mann_whitney", "error": str(e), "significant": False}

    @staticmethod
    def analyze_ab_test(
        results: List[TaskResult],
        metric_names: List[str] = None,
        test_type: str = "t_test",
    ) -> Dict[str, Any]:
        """
        Полный анализ A/B теста

        Args:
            results: Результаты с заполненным ab_variant
            metric_names: Метрики для анализа
            test_type: 't_test' или 'mann_whitney'
        """
        metric_names = metric_names or []

        # Группируем по вариантам
        grouped = ABTestAnalyzer.group_results_by_variant(results)

        if len(grouped) < 2:
            return {
                "error": "Need at least 2 variants for A/B test",
                "variants": list(grouped.keys()),
            }

        # Вычисляем метрики для каждого варианта
        variant_metrics = {}
        for variant_name, variant_results in grouped.items():
            variant_metrics[variant_name] = ABTestAnalyzer.compute_variant_metrics(
                variant_results, metric_names
            )

        # Статистические тесты между всеми парами вариантов
        statistical_tests = {}
        variants = list(grouped.keys())

        for i, variant_a in enumerate(variants):
            for variant_b in variants[i + 1 :]:
                pair_key = f"{variant_a}_vs_{variant_b}"

                # Для judge scores
                scores_a = [
                    r.judge_score
                    for r in grouped[variant_a]
                    if r.judge_score is not None
                ]
                scores_b = [
                    r.judge_score
                    for r in grouped[variant_b]
                    if r.judge_score is not None
                ]

                if scores_a and scores_b:
                    if test_type == "t_test":
                        test_result = ABTestAnalyzer.t_test(scores_a, scores_b)
                    else:
                        test_result = ABTestAnalyzer.mann_whitney_test(
                            scores_a, scores_b
                        )

                    statistical_tests[f"{pair_key}_judge_score"] = test_result

                # Для каждой кастомной метрики
                for metric_name in metric_names:
                    values_a = [
                        r.metrics.get(metric_name)
                        for r in grouped[variant_a]
                        if metric_name in r.metrics
                    ]
                    values_b = [
                        r.metrics.get(metric_name)
                        for r in grouped[variant_b]
                        if metric_name in r.metrics
                    ]

                    if values_a and values_b:
                        if test_type == "t_test":
                            test_result = ABTestAnalyzer.t_test(values_a, values_b)
                        else:
                            test_result = ABTestAnalyzer.mann_whitney_test(
                                values_a, values_b
                            )

                        statistical_tests[f"{pair_key}_{metric_name}"] = test_result

        # Определяем победителя
        winner = ABTestAnalyzer._determine_winner(variant_metrics, statistical_tests)

        return {
            "variants": list(grouped.keys()),
            "variant_metrics": variant_metrics,
            "statistical_tests": statistical_tests,
            "winner": winner,
            "test_type": test_type,
            "total_samples": len(results),
        }

    @staticmethod
    def _determine_winner(
        variant_metrics: Dict[str, Dict[str, float]],
        statistical_tests: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Определить победителя A/B теста"""

        # Ищем значимые различия
        significant_tests = {
            k: v for k, v in statistical_tests.items() if v.get("significant", False)
        }

        if not significant_tests:
            return {
                "variant": None,
                "reason": "No statistically significant differences found",
                "confidence": "low",
            }

        # Определяем лучший вариант по judge_score
        best_variant = None
        best_score = -float("inf")

        for variant, metrics in variant_metrics.items():
            score = metrics.get("avg_judge_score", 0)
            if score > best_score:
                best_score = score
                best_variant = variant

        # Проверяем, что победитель статистически значимо лучше
        winner_confirmed = False
        for test_name, test_result in significant_tests.items():
            if best_variant in test_name and test_result.get("significant"):
                winner_confirmed = True
                break

        return {
            "variant": best_variant,
            "score": best_score,
            "reason": "Highest average score with statistical significance"
            if winner_confirmed
            else "Highest average score (check significance)",
            "confidence": "high" if winner_confirmed else "medium",
            "significant_tests": len(significant_tests),
        }
