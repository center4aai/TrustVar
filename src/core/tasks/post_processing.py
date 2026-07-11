# src/core/tasks/post_processing.py

import asyncio
import traceback
from typing import Any, Dict

from src.core.schemas.task import Task, TaskResult
from src.utils.logger import logger


async def run_post_processing(
    task: Task,
    task_id: str,
    task_repo: Any,
    result_repo: Any,
) -> None:
    """
    Runs once after all inferences.
    Only reads from task_results — no new inferences.
    Idempotent: safe to re-run on resume.
    """
    aggregated_metrics: Dict = {}

    # ── 1. Standard metrics ────────────────────────────────────────────────
    if task.config.evaluate and task.config.evaluation_metrics:
        from src.core.services.eval_service import EvaluationService

        evaluation_service = EvaluationService()
        results_by_model: Dict[str, list] = {}

        async for page in _iter_pages(result_repo, task_id):
            for r in page:
                results_by_model.setdefault(r.model_name, []).append(r)

        for model_name, model_results in results_by_model.items():
            metrics = evaluation_service.evaluate_results(
                model_results,
                task.config.evaluation_metrics,
            )
            aggregated_metrics[model_name] = metrics
            logger.info(f"Metrics [{model_name}]: {metrics}")

    # ── 2. Include/Exclude ─────────────────────────────────────────────────
    if (
        hasattr(task.config, "dataset_config")
        and task.config.dataset_config
        and (
            task.config.dataset_config.include_column
            or task.config.dataset_config.exclude_column
        )
    ):
        from src.core.services.include_exclude_evaluator import (
            IncludeExcludeEvaluator,
        )

        all_results = await result_repo.find_all_by_task(task_id)
        ie_metrics = IncludeExcludeEvaluator.evaluate_results(all_results)
        aggregated_metrics["include_exclude"] = ie_metrics
        logger.info(f"Include/Exclude metrics: {ie_metrics}")

    # ── 3. Judge aggregation ───────────────────────────────────────────────
    if task.config.judge.enabled:
        judge_agg = await _aggregate_judge_scores(result_repo, task_id)
        aggregated_metrics["judge"] = judge_agg
        logger.info(f"Judge aggregation: {judge_agg}")

    # ── 4. RTA aggregation ─────────────────────────────────────────────────
    if task.config.rta.enabled:
        rta_agg = await _aggregate_rta_scores(result_repo, task_id)
        aggregated_metrics["rta"] = rta_agg
        logger.info(f"RTA aggregation: {rta_agg}")

    # ── 5. A/B analysis ────────────────────────────────────────────────────
    if task.config.ab_test.enabled:
        marked = await result_repo.find_by_task_with_ab(task_id)
        if len(marked) >= 2:
            from src.core.services.ab_test_analyzer import ABTestAnalyzer

            ab_results = ABTestAnalyzer.analyze_ab_test(
                marked,
                task.config.evaluation_metrics,
                task.config.ab_test.statistical_test,
            )
            await task_repo.update(task_id, {"ab_test_results": ab_results})
            logger.info(f"A/B analysis done: winner={ab_results.get('winner')}")

    # ── 6. TrustVar metrics (TSI, EAR, CV) — variation tasks only ─────────
    if task.config.variations.enabled:
        try:
            from src.core.services.eval_service import EvaluationService

            all_results = await result_repo.find_all_by_task(task_id)
            if all_results:
                # StoredTaskResult → TaskResult (reset internal fields)
                task_results = [
                    TaskResult(
                        **{
                            k: v
                            for k, v in r.model_dump().items()
                            if k not in {"id", "task_id", "sequence_num", "created_at"}
                        }
                    )
                    for r in all_results
                ]
                trustvar_metrics = await asyncio.to_thread(
                    EvaluationService().compute_trustvar_metrics,
                    task_results,
                )
                aggregated_metrics["_trustvar"] = trustvar_metrics
                logger.info(
                    f"TrustVar metrics: TSI={trustvar_metrics.get('aggregate_tsi')}, "
                    f"EAR={trustvar_metrics.get('aggregate_ear')}"
                )
        except Exception as e:
            msg = f"TrustVar metrics computation failed: {e}\n{traceback.format_exc()}"
            logger.error(msg)
            aggregated_metrics["_trustvar_error"] = msg
            await task_repo.update(task_id, {"metrics_error": msg})

    await task_repo.set_aggregated_metrics(task_id, aggregated_metrics)
    logger.info(f"Post-processing complete for task {task_id}")


async def _iter_pages(result_repo: Any, task_id: str, page_size: int = 500):
    skip = 0
    while True:
        page = await result_repo.find_by_task(task_id, skip=skip, limit=page_size)
        if not page:
            break
        yield page
        skip += page_size


async def _aggregate_judge_scores(result_repo: Any, task_id: str) -> Dict:
    pipeline = [
        {"$match": {"task_id": task_id, "judge_score": {"$ne": None}}},
        {
            "$group": {
                "_id": "$model_name",
                "mean_score": {"$avg": "$judge_score"},
                "min_score": {"$min": "$judge_score"},
                "max_score": {"$max": "$judge_score"},
                "std_score": {"$stdDevPop": "$judge_score"},
                "count": {"$sum": 1},
            }
        },
    ]
    results = {}
    async for doc in result_repo.collection.aggregate(pipeline):
        model_name = doc.pop("_id")
        results[model_name] = doc
    return results


async def _aggregate_rta_scores(result_repo: Any, task_id: str) -> Dict:
    pipeline = [
        {"$match": {"task_id": task_id, "refused": {"$ne": None}}},
        {
            "$group": {
                "_id": "$model_name",
                "total": {"$sum": 1},
                "refused_count": {
                    "$sum": {"$cond": [{"$eq": ["$refused", "1"]}, 1, 0]}
                },
            }
        },
        {"$addFields": {"refusal_rate": {"$divide": ["$refused_count", "$total"]}}},
    ]
    results = {}
    async for doc in result_repo.collection.aggregate(pipeline):
        model_name = doc.pop("_id")
        results[model_name] = doc
    return results
