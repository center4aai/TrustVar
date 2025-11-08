# src/api/routes/tasks.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.config.constants import TaskStatus
from src.core.schemas.task import Task, TaskConfig, TaskType
from src.core.services.task_service import TaskService

router = APIRouter()


def get_task_service():
    return TaskService()


class TaskCreate(BaseModel):
    name: str
    dataset_id: str
    model_ids: List[str]  # Изменено: теперь список
    task_type: TaskType = TaskType.STANDARD
    config: TaskConfig


class TaskList(BaseModel):
    status: Optional[TaskStatus] = None
    skip: int = 0
    limit: int = 100


@router.post("/", response_model=Task, status_code=202)
async def create_task(
    task_data: TaskCreate,
    service: TaskService = Depends(get_task_service),
):
    """Создать и запустить задачу"""
    try:
        task = await service.create_task(**task_data.model_dump())
        return task
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Task])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    skip: int = 0,
    limit: int = 100,
    service: TaskService = Depends(get_task_service),
):
    """Список задач"""
    return await service.list_tasks(status=status, skip=skip, limit=limit)


@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Получить задачу"""
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Отменить задачу"""
    cancelled = await service.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(
            status_code=404, detail="Task not found or could not be cancelled"
        )
    return {"message": "Task cancellation requested"}


@router.get("/{task_id}/compare-models")
async def compare_models(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Сравнить результаты разных моделей в задаче"""
    comparison = await service.compare_model_results(task_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Task not found or not completed")
    return comparison
