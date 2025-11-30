# src/core/services/eval_service.py
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, precision_score, recall_score

from src.core.schemas.task import TaskResult
from src.core.services.include_exclude_evaluator import IncludeExcludeEvaluator
from src.utils.logger import logger


class EvaluationService:
    """Сервис для оценки результатов"""

    def evaluate_results(
        self, results: List[TaskResult], metrics: List[str]
    ) -> Dict[str, float]:
        """Оценить результаты по метрикам"""
        aggregated = {}

        for metric in metrics:
            if metric == "bleu":
                aggregated[metric] = self._bleu_score(results)
            elif metric == "rouge":
                aggregated[metric] = self._rouge_score(results)
            elif metric == "accuracy":
                aggregated[metric] = self._accuracy(results)
            elif metric == "f1_score":
                aggregated[metric] = self._f1_score(results)
            elif metric == "precision":
                aggregated[metric] = self._precision(results)
            elif metric == "recall":
                aggregated[metric] = self._recall(results)
            elif metric == "cv":
                aggregated[metric] = self._calculate_cv(results)
            elif metric == "rta":
                aggregated[metric] = self._rta_score(results)
            elif metric == "iqr_cv":
                aggregated[metric] = self._calculate_iqr_cv(results)
            elif metric == "jsd":
                aggregated[metric] = self._calculate_jsd_divergence(results)
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

        return aggregated

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

    # def _exact_match(self, results: List[TaskResult]) -> float:
    #     """Exact Match метрика"""
    #     if not results:
    #         return 0.0

    #     matches = sum(
    #         1
    #         for r in results
    #         if r.target and r.output.strip().lower() == str(r.target).strip().lower()
    #     )

    #     total = sum(1 for r in results if r.target)

    #     return (matches / total * 100) if total > 0 else 0.0

    def _accuracy(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        if not results:
            return 0.0

        correct = sum(
            1
            for r in results
            if r.target and str(r.target).strip().lower() == r.output.strip().lower()
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

    def _rta_score(self, results: List[TaskResult]) -> float:
        """Refuse to answer score"""
        if not results:
            return 0.0

        refused = sum(int(r.refused) for r in results if r.refused)

        total = sum(1 for r in results if r.refused)

        return (refused / total * 100) if total > 0 else 0.0

    def _f1_score(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        pass
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1 * 100

    def _precision(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        pass
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        precision = precision_score(y_true, y_pred, average="macro")
        return precision * 100

    def _recall(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        pass
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        recall = recall_score(y_true, y_pred, average="macro")
        return recall * 100
