# src/core/services/task_service.py
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.core.schemas.task import Task, TaskConfig, TaskStatus, TaskType
from src.core.tasks.inference_task import run_inference_task
from src.database.repositories.task_repository import TaskRepository
from src.utils.logger import logger


class TaskService:
    """Сервис для работы с задачами"""

    def __init__(self):
        self.repository = TaskRepository()

    async def create_task(
        self,
        name: str,
        dataset_id: str,
        model_ids: List[str],  # Изменено
        task_type: TaskType = TaskType.STANDARD,
        config: TaskConfig = None,
    ) -> Task:
        """Создать и запустить задачу"""

        if not model_ids:
            raise ValueError("At least one model is required")

        task = Task(
            name=name,
            dataset_id=dataset_id,
            model_ids=model_ids,
            task_type=task_type,
            config=config or TaskConfig(),
        )

        # Сохраняем в БД
        created = await self.repository.create(task)

        # Запускаем Celery задачу
        celery_task = run_inference_task.delay(created.id)

        # Сохраняем Celery task ID
        await self.repository.update(created.id, {"celery_task_id": celery_task.id})

        logger.info(
            f"Task created and scheduled: {created.id} with {len(model_ids)} models"
        )

        return created

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получить задачу"""
        return await self.repository.find_by_id(task_id)

    async def list_tasks(
        self, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Task]:
        """Список задач"""
        filters = {}
        if status:
            filters["status"] = status

        return await self.repository.find_all(filters, skip, limit)

    async def cancel_task(self, task_id: str) -> bool:
        """Отменить задачу"""
        task = await self.get_task(task_id)

        if not task:
            return False

        # Отменяем Celery задачу
        if task.celery_task_id:
            from src.core.tasks.celery_app import celery_app

            celery_app.control.revoke(task.celery_task_id, terminate=True)

        # Обновляем статус
        await self.repository.update_status(task_id, TaskStatus.CANCELLED)

        logger.info(f"Task cancelled: {task_id}")

        return True

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Получить статус задачи"""
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
        """Сравнить результаты разных моделей в рамках одной задачи"""
        task = await self.get_task(task_id)

        if not task or task.status != TaskStatus.COMPLETED:
            return None

        # Группируем результаты по моделям
        results_by_model = defaultdict(list)
        for result in task.results:
            results_by_model[result.model_id].append(result)

        comparison = {
            "task_id": task.id,
            "task_name": task.name,
            "models": {},
            "summary": {
                "total_models": len(results_by_model),
                "total_results": len(task.results),
            },
        }

        # Статистика по каждой модели
        for model_id, model_results in results_by_model.items():
            avg_time = sum(r.execution_time for r in model_results) / len(model_results)

            model_stats = {
                "total_results": len(model_results),
                "avg_execution_time": avg_time,
                "metrics": task.aggregated_metrics.get(model_id, {}),
            }

            # Если использовался judge
            judge_scores = [
                r.judge_score for r in model_results if r.judge_score is not None
            ]
            if judge_scores:
                model_stats["avg_judge_score"] = sum(judge_scores) / len(judge_scores)
                model_stats["min_judge_score"] = min(judge_scores)
                model_stats["max_judge_score"] = max(judge_scores)

            # Вариации
            variation_counts = defaultdict(int)
            for r in model_results:
                if r.variation_type:
                    variation_counts[r.variation_type] += 1

            if variation_counts:
                model_stats["variations"] = dict(variation_counts)

            comparison["models"][model_id] = model_stats

        # Определяем лучшую модель
        if task.config.judge.enabled:
            # По judge score
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
            # По первой метрике
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
        """Приостановить задачу"""
        task = await self.get_task(task_id)

        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        # Обновляем статус на PAUSED
        from datetime import datetime

        await self.repository.update(
            task_id,
            {"status": TaskStatus.PAUSED, "paused_at": datetime.now()},
        )

        logger.info(f"Task paused: {task_id}")
        return True

    async def resume_task(self, task_id: str) -> bool:
        """Возобновить задачу с recovery"""
        task = await self.get_task(task_id)

        if not task:
            return False

        if task.status != TaskStatus.PAUSED:
            logger.warning(f"Task {task_id} is not paused (status: {task.status})")
            return False

        # Возобновляем задачу
        from datetime import datetime

        await self.repository.update(
            task_id, {"status": TaskStatus.RUNNING, "resumed_at": datetime.utcnow()}
        )

        logger.info(f"Task {task_id} status set to RUNNING, launching Celery task...")

        # Запускаем Celery задачу снова
        # Recovery будет выполнено внутри _run_inference_async
        from src.core.tasks.inference_task import run_inference_task

        celery_task = run_inference_task.delay(task_id)

        await self.repository.update(task_id, {"celery_task_id": celery_task.id})

        logger.info(f"Task resumed: {task_id}, celery_task_id: {celery_task.id}")

        return True
