# src/api/routes/tasks.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.config.constants import TaskStatus
from src.core.schemas.task import Task
from src.core.services.task_service import TaskService

router = APIRouter()


def get_task_service():
    return TaskService()


class TaskCreate(BaseModel):
    name: str
    dataset_id: str
    model_id: str
    task_type: str
    batch_size: int = 1
    max_samples: Optional[int] = None
    evaluate: bool = True
    evaluation_metrics: Optional[List[str]] = []


class TaskList(BaseModel):
    status: Optional[TaskStatus] = None
    skip: int = 0
    limit: int = 100


@router.post(
    "/", response_model=Task, status_code=202
)  # 202 Accepted, т.к. задача запускается в фоне
async def create_task(
    task_data: TaskCreate,
    service: TaskService = Depends(get_task_service),
):
    try:
        task = await service.create_task(**task_data.model_dump())
        return task
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Task])
async def list_tasks(
    request: TaskList,
    service: TaskService = Depends(get_task_service),
):
    return await service.list_tasks(
        status=request.status, skip=request.skip, limit=request.limit
    )


@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    cancelled = await service.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(
            status_code=404, detail="Task not found or could not be cancelled"
        )
    return {"message": "Task cancellation requested"}
