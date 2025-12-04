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
    status: Optional[str] = None,
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


@router.post("/{task_id}/pause")
async def pause_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Приостановить выполнение задачи"""
    paused = await service.pause_task(task_id)
    if not paused:
        raise HTTPException(
            status_code=404, detail="Task not found or could not be paused"
        )
    return {"message": "Task paused successfully"}


@router.post("/{task_id}/resume")
async def resume_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Возобновить выполнение задачи"""
    resumed = await service.resume_task(task_id)
    if not resumed:
        raise HTTPException(
            status_code=404, detail="Task not found or could not be resumed"
        )
    return {"message": "Task resumed successfully"}


@router.post("/{task_id}/recover")
async def recover_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """
    Восстановить состояние задачи вручную

    Полезно если задача зависла или данные повреждены
    """
    from src.core.tasks.inference_task import _recover_task_state
    from src.database.repositories.task_repository import TaskRepository

    task_repo = TaskRepository()

    try:
        recovery_info = await _recover_task_state(task_id, task_repo)

        return {
            "status": "success",
            "recovery_info": recovery_info,
            "message": "Task state recovered successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {str(e)}")
