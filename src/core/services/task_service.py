# src/core/services/task_service.py
from typing import Any, Dict, List, Optional

from src.core.tasks.inference_task import run_inference_task
from src.utils.logger import logger

from src.core.models.task import Task, TaskStatus
from src.database.repositories.task_repository import TaskRepository


class TaskService:
    """Сервис для работы с задачами"""

    def __init__(self):
        self.repository = TaskRepository()

    async def create_task(
        self,
        name: str,
        dataset_id: str,
        model_id: str,
        task_type: str,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        evaluate: bool = True,
        evaluation_metrics: List[str] = None,
    ) -> Task:
        """Создать и запустить задачу"""

        task = Task(
            name=name,
            dataset_id=dataset_id,
            model_id=model_id,
            task_type=task_type,
            batch_size=batch_size,
            max_samples=max_samples,
            evaluate=evaluate,
            evaluation_metrics=evaluation_metrics or [],
        )

        # Сохраняем в БД
        created = await self.repository.create(task)

        # Запускаем Celery задачу
        celery_task = run_inference_task.delay(created.id)

        # Сохраняем Celery task ID
        await self.repository.update(created.id, {"celery_task_id": celery_task.id})

        logger.info(f"Task created and scheduled: {created.id}")

        return created

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Получить задачу"""
        return await self.repository.find_by_id(task_id)

    async def list_tasks(
        self, status: Optional[TaskStatus] = None, skip: int = 0, limit: int = 100
    ) -> List[Task]:
        """Список задач"""
        filters = {}
        if status:
            filters["status"] = status.value

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

    async def get_task_results(self, task_id: str) -> Dict[str, Any]:
        """Получить результаты задачи"""
        task = await self.get_task(task_id)

        if not task:
            return {"error": "Task not found"}

        if task.status != TaskStatus.COMPLETED:
            return {"error": "Task not completed"}

        return {
            "task_id": task.id,
            "task_name": task.name,
            "results": task.results,
            "metrics": task.aggregated_metrics,
            "total_samples": task.total_samples,
            "duration": (task.completed_at - task.started_at).total_seconds()
            if task.completed_at and task.started_at
            else None,
        }
