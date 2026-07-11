# src/database/repositories/task_repository.py
# No logic changes — just ensuring compatibility

from datetime import datetime
from typing import Dict, List, Optional

from src.core.schemas.task import Task, TaskStatus

from .base import BaseRepository


class TaskRepository(BaseRepository[Task]):
    def __init__(self):
        super().__init__("tasks", Task)

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
    ) -> Optional[Task]:
        update_data: Dict = {"status": status.value}

        if status == TaskStatus.RUNNING and not error:
            update_data["started_at"] = datetime.now()
        elif status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            update_data["completed_at"] = datetime.now()

        if error:
            update_data["error"] = error

        return await self.update(task_id, update_data)

    async def update_progress(
        self,
        task_id: str,
        progress: float,
        processed: int,
    ) -> Optional[Task]:
        return await self.update(
            task_id,
            {
                "progress": progress,
                "processed_samples": processed,
            },
        )

    async def set_aggregated_metrics(
        self,
        task_id: str,
        metrics: Dict[str, Dict],
    ) -> Optional[Task]:
        return await self.update(task_id, {"aggregated_metrics": metrics})

    async def set_completion_summary(
        self,
        task_id: str,
        summary: Dict,
    ) -> Optional[Task]:
        """Persist the run completeness / per-model coverage summary (WEB-5/6)."""
        return await self.update(task_id, {"completion_summary": summary})

    async def find_by_status(self, status: TaskStatus) -> List[Task]:
        return await self.find_all({"status": status.value})
