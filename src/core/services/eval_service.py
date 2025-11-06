# src/core/services/evaluation_service.py
from typing import Dict, List

from src.core.schemas.task import TaskResult
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

        return aggregated

    def _exact_match(self, results: List[TaskResult]) -> float:
        """Exact Match метрика"""
        if not results:
            return 0.0

        matches = sum(
            1
            for r in results
            if r.expected_output
            and r.output.strip().lower() == r.expected_output.strip().lower()
        )

        total = sum(1 for r in results if r.expected_output)

        return (matches / total * 100) if total > 0 else 0.0

    def _accuracy(self, results: List[TaskResult]) -> float:
        """Accuracy для классификации"""
        if not results:
            return 0.0

        correct = sum(
            1
            for r in results
            if r.expected_output and r.expected_output.lower() in r.output.lower()
        )

        total = sum(1 for r in results if r.expected_output)

        return (correct / total * 100) if total > 0 else 0.0

    def _bleu_score(self, results: List[TaskResult]) -> float:
        """BLEU score (упрощенная версия)"""
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            scores = []
            smooth = SmoothingFunction()

            for r in results:
                if r.expected_output:
                    reference = [r.expected_output.split()]
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
                if r.expected_output and r.output:
                    score = rouge.get_scores(r.output, r.expected_output)[0]
                    scores.append(score["rouge-l"]["f"])

            return (sum(scores) / len(scores) * 100) if scores else 0.0
        except ImportError:
            logger.warning("rouge not installed, ROUGE score unavailable")
            return 0.0
