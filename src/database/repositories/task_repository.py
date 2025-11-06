# src/database/repositories/task_repository.py
from datetime import datetime
from typing import List, Optional

from src.core.schemas.task import Task, TaskResult, TaskStatus
from src.utils.logger import logger

from .base import BaseRepository


class TaskRepository(BaseRepository[Task]):
    """Репозиторий для задач"""

    def __init__(self):
        super().__init__("tasks", Task)

    async def update_status(
        self, task_id: str, status: TaskStatus, error: Optional[str] = None
    ):
        """Обновить статус задачи"""
        update_data = {"status": status.value}

        if status == TaskStatus.RUNNING and not error:
            update_data["started_at"] = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            update_data["completed_at"] = datetime.utcnow()

        if error:
            update_data["error"] = error

        logger.info(f"Updating task {task_id} status to {status}")
        return await self.update(task_id, update_data)

    async def update_progress(self, task_id: str, progress: float, processed: int):
        """Обновить прогресс задачи"""
        return await self.update(
            task_id, {"progress": progress, "processed_samples": processed}
        )

    async def add_result(self, task_id: str, result: TaskResult):
        """Добавить результат"""
        collection = await self._get_collection()
        await collection.update_one(
            {"id": task_id}, {"$push": {"results": result.model_dump()}}
        )

    async def set_results(self, task_id: str, results: List[TaskResult], metrics: dict):
        """Установить результаты и метрики"""
        return await self.update(
            task_id,
            {
                "results": [r.model_dump() for r in results],
                "aggregated_metrics": metrics,
                "processed_samples": len(results),
            },
        )

    async def find_by_status(self, status: TaskStatus) -> List[Task]:
        """Найти задачи по статусу"""
        return await self.find_all({"status": status.value})
