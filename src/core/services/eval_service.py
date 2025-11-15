# src/core/services/eval_service.py
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import jensenshannon

from src.core.schemas.task import TaskResult
from src.core.services.include_exclude_evaluator import IncludeExcludeEvaluator
from src.core.services.rta_evaluator import RTAEvaluator
from src.utils.logger import logger


class EvaluationService:
    """Сервис для оценки результатов"""

    def evaluate_results(
        self, results: List[TaskResult], metrics: List[str]
    ) -> Dict[str, float]:
        """Оценить результаты по метрикам"""
        aggregated = {}

        for metric in metrics:
            if metric == "exact_match":
                aggregated[metric] = self._exact_match(results)
            elif metric == "bleu":
                aggregated[metric] = self._bleu_score(results)
            elif metric == "rouge":
                aggregated[metric] = self._rouge_score(results)
            elif metric == "accuracy":
                aggregated[metric] = self._accuracy(results)
            elif metric == "include_exclude":
                # Интегрируем Include/Exclude оценщик
                ie_results = IncludeExcludeEvaluator.evaluate_results(results)
                aggregated["include_exclude_score"] = ie_results[
                    "include_exclude_score"
                ]
                aggregated["include_success_rate"] = ie_results["include_success_rate"]
                aggregated["exclude_violation_rate"] = ie_results[
                    "exclude_violation_rate"
                ]
            elif metric == "rta":
                # Интегрируем RTA метрики
                rta_results = RTAEvaluator.compute_rta_metrics(results)
                aggregated["refusal_rate"] = rta_results["refusal_rate"]
                aggregated["explicit_refusals"] = rta_results["explicit_refusals"]
                aggregated["implicit_refusals"] = rta_results["implicit_refusals"]

        return aggregated

    # def compute_dispersion_indices(
    #     self,
    #     results: List[TaskResult],
    #     models: List[str],
    #     variations: List[str],
    #     metrics_to_compute: List[str],
    #     centricity: str,
    # ) -> Dict[str, float]:
    #     if centricity == "var-centric":
    #         scores_per_metric = defaultdict(
    #             lambda: defaultdict(dict)
    #         )  # ['Metric']['Variation']['Values']

    #         for metric in metrics_to_compute:
    #             for v in variations:
    #                 values = []
    #                 per_augment_results = [r for r in results if r.variation_type == v]

    #                 for model in models:
    #                     per_model_results = [
    #                         r for r in per_augment_results if r.model_id == model.id
    #                     ]

    #                     values.append(self._accuracy(per_model_results))

    #                 if metric == "cv":
    #                     scores_per_metric[metric][v] = self._calculate_cv(values)

    #                 elif metric == "iqr_cv":
    #                     scores_per_metric[metric][v] = self._calculate_iqr_cv(values)

    #                 elif metric == "jsd":
    #                     scores_per_metric[metric][v] = self._calculate_jsd_divergence(
    #                         values
    #                     )

    #         return scores_per_metric

    #     elif centricity == "model-centric":
    #         scores_per_metric = defaultdict(
    #             lambda: defaultdict(dict)
    #         )  # ['Metric']['Model']['Values']
    #         logger.info(scores_per_metric)
    #         for metric in metrics_to_compute:
    #             for model in models:
    #                 per_model_results = [r for r in results if r.model_id == model.id]

    #                 for v in variations:
    #                     values = []
    #                     per_augment_results = [
    #                         r for r in per_model_results if r.variation_type == v
    #                     ]
    #                     values.append(self._accuracy(per_augment_results))

    #                 if metric == "cv_star":
    #                     scores_per_metric[metric][model.name] = self._calculate_cv(
    #                         values
    #                     )

    #                 elif metric == "iqr_cv":
    #                     scores_per_metric[metric][model.name] = self._calculate_iqr_cv(
    #                         values
    #                     )

    #                 elif metric == "jsd":
    #                     scores_per_metric[metric][model.name] = (
    #                         self._calculate_jsd_divergence(values)
    #                     )

    #                 logger.info(scores_per_metric)

    #         return scores_per_metric

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculates the coefficient of variation (CV = std/mean * 100%)."""
        if not values or len(values) < 2:
            return np.nan
        mean_val = np.mean(values)
        if mean_val == 0:
            return np.nan
        return (np.std(values) / mean_val) * 100

    def _calculate_corrected_cv(self, values: List[float]) -> float:
        """Calculates the corrected coefficient of variation for small sample sizes."""
        if not values or len(values) < 2:
            return np.nan

        arr = np.array(values)
        n = arr.size
        mean_val = arr.mean()
        std_val = arr.std(ddof=0)

        if mean_val == 0:
            return np.nan

        cv = std_val / mean_val
        return (1 + 1 / (4 * n)) * cv * 100

    def _calculate_iqr_cv(self, values: List[float]) -> float:
        """Calculates IQR-based coefficient of variation: (Q3-Q1) / midhinge."""
        if not values or len(values) < 2:
            return np.nan

        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        midhinge = (q1 + q3) / 2

        if midhinge == 0:
            return np.nan

        return ((q3 - q1) / midhinge) * 100

    def _calculate_jsd_divergence(self, values: List[float]) -> float:
        """Calculates Jensen-Shannon Divergence for measuring distribution heterogeneity."""
        if not values or len(values) < 2:
            return np.nan

        arr = np.array(values)
        arr_sum = arr.sum()
        if arr_sum == 0:
            return np.nan

        P = arr / arr_sum
        P_mean = P.mean()

        jsd = jensenshannon(P, [P_mean] * len(P)) ** 2
        return jsd * 100

    def _exact_match(self, results: List[TaskResult]) -> float:
        """Exact Match метрика"""
        if not results:
            return 0.0

        matches = sum(
            1
            for r in results
            if r.target and r.output.strip().lower() == str(r.target).strip().lower()
        )

        total = sum(1 for r in results if r.target)

        return (matches / total * 100) if total > 0 else 0.0

    def _accuracy(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        if not results:
            return 0.0

        correct = sum(
            1 for r in results if r.target and str(r.target).lower() in r.output.lower()
        )

        total = sum(1 for r in results if r.target)

        return (correct / total * 100) if total > 0 else 0.0

    def _bleu_score(self, results: List[TaskResult]) -> float:
        """BLEU score (упрощенная версия)"""
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            scores = []
            smooth = SmoothingFunction()

            for r in results:
                if r.target:
                    reference = [str(r.target).split()]
                    candidate = r.output.split()
                    score = sentence_bleu(
                        reference, candidate, smoothing_function=smooth.method1
                    )
                    scores.append(score)

            return (sum(scores) / len(scores) * 100) if scores else 0.0
        except ImportError:
            logger.warning("NLTK not installed, BLEU score unavailable")
            return 0.0

    def _rouge_score(self, results: List[TaskResult]) -> float:
        """ROUGE score (упрощенная версия)"""
        try:
            from rouge import Rouge

            rouge = Rouge()
            scores = []

            for r in results:
                if r.target and r.output:
                    score = rouge.get_scores(r.output, str(r.target))[0]
                    scores.append(score["rouge-l"]["f"])

            return (sum(scores) / len(scores) * 100) if scores else 0.0
        except ImportError:
            logger.warning("rouge not installed, ROUGE score unavailable")
            return 0.0
