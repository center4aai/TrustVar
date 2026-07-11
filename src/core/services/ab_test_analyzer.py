# src/core/services/ab_test_analyzer.py
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from src.core.schemas.task import TaskResult
from src.utils.logger import logger
import traceback


class ABTestAnalyzer:
    """A/B test analyzer with statistical tests"""

    @staticmethod
    def group_results_by_variant(
        results: List[TaskResult],
    ) -> Dict[str, List[TaskResult]]:
        """Group results by variants"""
        grouped = defaultdict(list)
        for result in results:
            variant = result.ab_variant or "default"
            grouped[variant].append(result)
        return dict(grouped)

    @staticmethod
    def compute_variant_metrics(
        results: List[TaskResult], metric_names: List[str]
    ) -> Dict[str, float]:
        """Compute metrics for a single variant"""
        if not results:
            return {}

        metrics = {}

        # Basic metrics
        metrics["count"] = len(results)
        metrics["avg_execution_time"] = np.mean([r.execution_time for r in results])

        # Judge scores (if available)
        judge_scores = [r.judge_score for r in results if r.judge_score is not None]
        if judge_scores:
            metrics["avg_judge_score"] = np.mean(judge_scores)
            metrics["std_judge_score"] = np.std(judge_scores)

        # Include/Exclude scores (if available)
        include_scores = [
            r.include_score for r in results if r.include_score is not None
        ]
        if include_scores:
            metrics["avg_include_score"] = np.mean(include_scores)

        # Custom metrics from results
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
        """T-test for comparing two variants"""
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
            logger.error(f"T-test failed: {e}. Traceback: {traceback.format_exc()}")
            return {"test": "t_test", "error": str(e), "significant": False}

    @staticmethod
    def mann_whitney_test(
        variant_a_scores: List[float], variant_b_scores: List[float]
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (non-parametric)"""
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
            logger.error(
                f"Mann-Whitney test failed: {e}. Traceback: {traceback.format_exc()}"
            )
            return {"test": "mann_whitney", "error": str(e), "significant": False}

    @staticmethod
    def analyze_ab_test(
        results: List[TaskResult],
        metric_names: List[str] = None,
        test_type: str = "t_test",
    ) -> Dict[str, Any]:
        """Full A/B test analysis with improved validation"""

        metric_names = metric_names or []

        # Validate input data
        if not results:
            return {
                "error": "No results provided",
                "variants": [],
                "total_samples": 0,
            }

        # Check for variant markers
        unmarked = [r for r in results if r.ab_variant is None]
        if unmarked:
            logger.warning(f"Found {len(unmarked)} results without ab_variant marker")

        # Group by variants
        grouped = ABTestAnalyzer.group_results_by_variant(results)

        # Detailed grouping info
        variant_counts = {k: len(v) for k, v in grouped.items()}
        logger.info(f"Variant distribution: {variant_counts}")

        if len(grouped) < 2:
            return {
                "error": "Need at least 2 variants for A/B test",
                "variants": list(grouped.keys()),
                "variant_counts": variant_counts,
                "total_samples": len(results),
            }

        # Check data sufficiency
        min_samples = min(variant_counts.values())
        if min_samples < 2:
            logger.warning(f"Some variants have less than 2 samples: {variant_counts}")

        # Compute metrics for each variant
        variant_metrics = {}
        for variant_name, variant_results in grouped.items():
            variant_metrics[variant_name] = ABTestAnalyzer.compute_variant_metrics(
                variant_results, metric_names
            )

        # Statistical tests between all variant pairs
        statistical_tests = {}
        variants = list(grouped.keys())

        for i, variant_a in enumerate(variants):
            for variant_b in variants[i + 1 :]:
                pair_key = f"{variant_a}_vs_{variant_b}"

                # For judge scores
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

                # For each custom metric
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

        # Determine winner
        winner = ABTestAnalyzer._determine_winner(variant_metrics, statistical_tests)

        return {
            "variants": list(grouped.keys()),
            "variant_counts": variant_counts,  # NEW
            "variant_metrics": variant_metrics,
            "statistical_tests": statistical_tests,
            "winner": winner,
            "test_type": test_type,
            "total_samples": len(results),
            "warnings": ABTestAnalyzer._generate_warnings(grouped, statistical_tests),
        }

    @staticmethod
    def _generate_warnings(
        grouped: Dict[str, List[TaskResult]],
        statistical_tests: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate A/B test quality warnings"""

        warnings = []

        # Check sample sizes
        counts = {k: len(v) for k, v in grouped.items()}
        min_count = min(counts.values())
        max_count = max(counts.values())

        if min_count < 30:
            warnings.append(
                f"Small sample size detected (min={min_count}). Results may not be reliable."
            )

        if max_count / min_count > 2:
            warnings.append(
                f"Imbalanced sample sizes: {counts}. Consider equal distribution."
            )

        # Check test power
        failed_tests = [k for k, v in statistical_tests.items() if v.get("error")]
        if failed_tests:
            warnings.append(
                f"{len(failed_tests)} statistical tests failed: {failed_tests[:3]}"
            )

        return warnings

    @staticmethod
    def _determine_winner(
        variant_metrics: Dict[str, Dict[str, float]],
        statistical_tests: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Determine A/B test winner"""

        # Find significant differences
        significant_tests = {
            k: v for k, v in statistical_tests.items() if v.get("significant", False)
        }

        if not significant_tests:
            return {
                "variant": None,
                "reason": "No statistically significant differences found",
                "confidence": "low",
            }

        # Determine best variant by judge_score
        best_variant = None
        best_score = -float("inf")

        for variant, metrics in variant_metrics.items():
            score = metrics.get("avg_judge_score", 0)
            if score > best_score:
                best_score = score
                best_variant = variant

        # Check that winner is statistically significantly better
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
