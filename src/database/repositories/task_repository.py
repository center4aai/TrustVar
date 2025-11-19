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
            update_data["started_at"] = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            update_data["completed_at"] = datetime.now()

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

    async def save_intermediate_results(
        self,
        task_id: str,
        results: List[TaskResult],
        processed_count: int,
        last_index: int,
    ) -> bool:
        """
        Сохранение промежуточных результатов во время выполнения задачи

        Args:
            task_id: ID задачи
            results: Список результатов для сохранения
            processed_count: Количество обработанных инференсов
            last_index: Последний обработанный индекс датасета

        Returns:
            bool: Успешность операции
        """
        try:
            collection = await self._get_collection()

            # Конвертируем результаты в dict
            results_dict = [r.model_dump() for r in results]

            # Обновляем документ
            result = await collection.update_one(
                {"id": task_id},
                {
                    "$set": {
                        "results": results_dict,
                        "processed_samples": processed_count,
                        "last_processed_index": last_index,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error saving intermediate results for task {task_id}: {e}")
            return False

    async def append_result(self, task_id: str, result: TaskResult) -> bool:
        """
        Добавить один результат к существующим (атомарная операция)

        Более эффективно для real-time обновлений
        """
        try:
            collection = self._get_collection()

            result_dict = result.model_dump()

            update_result = await collection.update_one(
                {"id": task_id},
                {
                    "$push": {"results": result_dict},
                    "$inc": {"processed_samples": 1},
                    "$set": {"updated_at": datetime.utcnow()},
                },
            )

            return update_result.modified_count > 0

        except Exception as e:
            logger.error(f"Error appending result for task {task_id}: {e}")
            return False
