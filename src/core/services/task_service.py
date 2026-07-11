# src/core/services/task_service.py
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.core.schemas.task import Task, TaskConfig, TaskStatus, TaskType
from src.core.taxonomy import normalize_task_type
from src.database.repositories.dataset_repository import DatasetRepository
from src.database.repositories.task_repository import TaskRepository
from src.database.repositories.task_result_repository import TaskResultRepository
from src.utils.logger import logger


class TaskService:
    """Service for working with tasks"""

    def __init__(self):
        self.repository = TaskRepository()
        self.dataset_repository = DatasetRepository()

    async def create_task(
        self,
        name: str,
        dataset_id: str,
        model_ids: List[str],
        task_type: TaskType = TaskType.STANDARD,
        config: Any = None,
    ) -> Task:
        """Create and run a task"""

        if not model_ids:
            raise ValueError("At least one model is required")

        # Validate prompts for specific task types
        if isinstance(config, dict):
            config = TaskConfig(**config)
        else:
            config = config or TaskConfig()
        if task_type == TaskType.VARIATION:
            _validate_variation_config(config)
            # WEB-3: hard guard — a generative dataset (open_qa/generation) needs
            # a scoring path (judge or include/exclude); otherwise scores are NaN
            # and TSI/EAR cannot be computed. Resolve the dataset's canonical
            # task_type to apply the guard only where it matters (mcq/
            # classification score by exact-match and are not blocked).
            dataset = await self.dataset_repository.find_by_id(dataset_id)
            if dataset is not None:
                canon = normalize_task_type(getattr(dataset, "task_type", None))
                err = _check_generative_scoring(canon, config)
                if err:
                    raise ValueError(err)
            elif config.variations.enabled and not config.judge.enabled:
                # Dataset unresolved — fall back to a soft warning.
                logger.warning(
                    "Judge is disabled for a variation task and the dataset could "
                    "not be resolved; open_qa/generation items would be unscorable "
                    "(NaN) without a judge."
                )
        if task_type == TaskType.JUDGED and not config.judge.custom_prompt_template:
            raise ValueError("Judge prompt is required for judged tasks")
        if task_type == TaskType.RTA and not config.rta.rta_prompt_template:
            raise ValueError("RTA prompt is required for refuse-to-answer tasks")

        task = Task(
            name=name,
            dataset_id=dataset_id,
            model_ids=model_ids,
            task_type=task_type,
            config=config,
        )

        # Save to DB
        created = await self.repository.create(task)

        # Start Celery task
        from src.core.tasks.inference_task import run_inference_task

        celery_task = run_inference_task.delay(created.id)

        # Save Celery task ID
        await self.repository.update(created.id, {"celery_task_id": celery_task.id})

        logger.info(
            f"Task created and scheduled: {created.id} with {len(model_ids)} models"
        )

        return created

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task"""
        return await self.repository.find_by_id(task_id)

    async def list_tasks(
        self, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Task]:
        """List tasks"""
        filters = {}
        if status:
            filters["status"] = status

        return await self.repository.find_all(filters, skip, limit)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        task = await self.get_task(task_id)

        if not task:
            return False

        # Cancel Celery task
        if task.celery_task_id:
            from src.core.tasks.celery_app import celery_app

            celery_app.control.revoke(task.celery_task_id, terminate=True)

        # Update status
        await self.repository.update_status(task_id, TaskStatus.CANCELLED)

        logger.info(f"Task cancelled: {task_id}")

        return True

    async def delete_task(self, task_id: str) -> bool:
        """Delete task and all its results"""
        result = await self.repository.delete(task_id)

        if result:
            result_repo = TaskResultRepository()
            deleted_count = await result_repo.delete_by_task(task_id)
            logger.info(f"Task deleted: {task_id}, removed {deleted_count} results")

        return result

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        task = await self.get_task(task_id)

        if not task:
            return {"error": "Task not found"}

        return {
            "id": task.id,
            "name": task.name,
            "status": task.status,
            "progress": task.progress,
            "processed_samples": task.processed_samples,
            "total_samples": task.total_samples,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error,
        }

    async def compare_model_results(self, task_id: str) -> Dict[str, Any]:
        """Compare results of different models within a task"""
        task = await self.get_task(task_id)

        if not task or task.status != TaskStatus.COMPLETED:
            return None

        # Streaming: load results by pages
        result_repo = TaskResultRepository()
        results_by_model = defaultdict(list)
        page_skip, page_limit = 0, 500
        while True:
            page = await result_repo.find_by_task(
                task_id, skip=page_skip, limit=page_limit
            )
            if not page:
                break
            for r in page:
                results_by_model[r.model_id].append(r)
            page_skip += page_limit

        total_results = task.processed_samples

        comparison = {
            "task_id": task.id,
            "task_name": task.name,
            "models": {},
            "summary": {
                "total_models": len(results_by_model),
                "total_results": total_results,
            },
        }

        # Statistics per model
        for model_id, model_results in results_by_model.items():
            avg_time = sum(r.execution_time for r in model_results) / len(model_results)

            model_stats = {
                "total_results": len(model_results),
                "avg_execution_time": avg_time,
                "metrics": task.aggregated_metrics.get(model_id, {}),
            }

            # If judge was used
            judge_scores = [
                r.judge_score for r in model_results if r.judge_score is not None
            ]
            if judge_scores:
                model_stats["avg_judge_score"] = sum(judge_scores) / len(judge_scores)
                model_stats["min_judge_score"] = min(judge_scores)
                model_stats["max_judge_score"] = max(judge_scores)

            # Variations
            variation_counts = defaultdict(int)
            for r in model_results:
                if r.variation_type:
                    variation_counts[r.variation_type] += 1

            if variation_counts:
                model_stats["variations"] = dict(variation_counts)

            comparison["models"][model_id] = model_stats

        # Determine best model
        if task.config.judge.enabled:
            # By judge score
            best_model = max(
                comparison["models"].items(),
                key=lambda x: x[1].get("avg_judge_score", 0),
            )
            comparison["best_model"] = {
                "model_id": best_model[0],
                "reason": "highest_judge_score",
                "score": best_model[1].get("avg_judge_score", 0),
            }
        elif task.aggregated_metrics:
            # By first metric
            first_metric = list(list(task.aggregated_metrics.values())[0].keys())[0]
            best_model = max(
                comparison["models"].items(),
                key=lambda x: x[1]["metrics"].get(first_metric, 0),
            )
            comparison["best_model"] = {
                "model_id": best_model[0],
                "reason": f"highest_{first_metric}",
                "score": best_model[1]["metrics"].get(first_metric, 0),
            }

        return comparison

    async def pause_task(self, task_id: str) -> bool:
        """Pause task"""
        task = await self.get_task(task_id)

        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        # Update status to PAUSED
        from datetime import datetime

        await self.repository.update(
            task_id,
            {"status": TaskStatus.PAUSED, "paused_at": datetime.now()},
        )

        logger.info(f"Task paused: {task_id}")
        return True

    async def resume_task(self, task_id: str) -> bool:
        """Resume task with recovery"""
        task = await self.get_task(task_id)

        if not task:
            return False

        if task.status != TaskStatus.PAUSED:
            logger.warning(f"Task {task_id} is not paused (status: {task.status})")
            return False

        # Resume task
        from datetime import datetime

        await self.repository.update(
            task_id, {"status": TaskStatus.RUNNING, "resumed_at": datetime.utcnow()}
        )

        logger.info(f"Task {task_id} status set to RUNNING, launching Celery task...")

        # Start Celery task again
        # Recovery will be performed inside _run_inference_async
        from src.core.tasks.inference_task import run_inference_task

        celery_task = run_inference_task.delay(task_id)

        await self.repository.update(task_id, {"celery_task_id": celery_task.id})

        logger.info(f"Task resumed: {task_id}, celery_task_id: {celery_task.id}")

        return True


def _check_generative_scoring(
    canon_task_type: Optional[str], config: TaskConfig
) -> Optional[str]:
    """WEB-3: return an error message if a generative task_type has no scoring
    path, else None. Pure and model-free.

    open_qa / generation per-item scores need either an LLM judge or an
    include/exclude criterion; otherwise ``_extract_score`` returns NaN (S1.2)
    and TSI/EAR cannot be computed. Non-generative types (mcq/classification)
    score by exact-match and are never blocked; an unknown type is not blocked
    (generativeness cannot be determined)."""
    if canon_task_type not in ("open_qa", "generation"):
        return None
    if config.judge.enabled:
        return None
    if "include_exclude" in (config.evaluation_metrics or []):
        return None
    return (
        f"Task type '{canon_task_type}' is generative: per-item scoring needs "
        f"either an LLM judge (enable judge) or an include/exclude criterion "
        f"('include_exclude' in evaluation_metrics). Without one, TSI/EAR cannot "
        f"be computed (scores are NaN)."
    )


def _validate_variation_config(config: TaskConfig) -> None:
    """Validate variation task configuration.

    Tier A (symbolic operators): model_id not required.
    Tier B/C (LLM-based): model_id required (variation generation model).

    [AUGMENT 2026-07-11 default-prompts-seed] custom_prompt requirement removed:
    Tier B/C operators use their own built-in templates, and the
    variations.custom_prompt field never affected the generated text.
    """
    strategies = config.variations.strategies
    if not strategies:
        raise ValueError("At least one variation strategy is required")

    has_llm = any(s.requires_llm for s in strategies)

    if has_llm:
        if not config.variations.model_id:
            raise ValueError(
                "Variation model is required for Tier B/C strategies. "
                "Select at least one variation model or use only Tier A strategies."
            )
