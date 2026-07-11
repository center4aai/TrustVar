# src/api/routes/tasks.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config.constants import TaskStatus
from src.core.schemas.task import Task, TaskConfig, TaskResult, TaskType
from src.core.services.task_service import TaskService
from src.database.repositories.task_result_repository import TaskResultRepository
from src.utils.json_safe import to_json_safe
from src.utils.logger import logger

router = APIRouter()

# Empty TrustVar envelope (shape == EvaluationService.compute_trustvar_metrics([]))
# returned when metrics are neither cached nor recomputable — so the Results UI
# renders its honest "no metrics" state instead of the endpoint 500-ing.
_EMPTY_TRUSTVAR_METRICS = {
    "per_task_tsi": {},
    "per_task_ear": {},
    "per_task_cv": {},
    "per_task_iqr_cv": {},
    "per_task_uninformative": {},
    "per_task_ear_flags": {},
    "per_task_cv_unreliable": {},
    "model_cv_star": {},
    "aggregate_tsi": {},
    "aggregate_ear": {},
    "variance_decomposition": {"per_tier": {}, "pooled": {}},
    "tier_comparison": {},
    "bootstrap_replicates": {"tsi": {}, "ear": {}},
    "n_models": 0,
    "n_resamples": 0,
    "ci_level": 0.95,
}


def get_task_service():
    return TaskService()


class TaskCreate(BaseModel):
    name: str
    dataset_id: str
    model_ids: List[str]
    task_type: TaskType = TaskType.STANDARD
    config: TaskConfig


class TaskList(BaseModel):
    status: Optional[TaskStatus] = None
    skip: int = 0
    limit: int = 100


class ResultsPage(BaseModel):
    task_id: str
    total: int
    skip: int
    limit: int
    results: List[TaskResult]


@router.post("/", response_model=Task, status_code=202)
async def create_task(
    task_data: TaskCreate,
    service: TaskService = Depends(get_task_service),
):
    """Create and run a task"""
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
    """List tasks"""
    return await service.list_tasks(status=status, skip=skip, limit=limit)


@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Get task (metadata without results)"""
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}/results")
async def get_task_results(
    task_id: str,
    skip: int = 0,
    limit: int = 100,
):
    """Get task results with pagination"""
    result_repo = TaskResultRepository()
    stored = await result_repo.find_by_task(task_id, skip=skip, limit=limit)
    total = await result_repo.count_by_task(task_id)

    results = [
        TaskResult(
            **{
                k: v
                for k, v in r.model_dump().items()
                if k not in {"id", "task_id", "sequence_num", "created_at"}
            }
        )
        for r in stored
    ]
    return ResultsPage(task_id=task_id, total=total, skip=skip, limit=limit, results=results)


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Cancel a task"""
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
    """Compare results of different models in a task"""
    comparison = await service.compare_model_results(task_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Task not found or not completed")
    return comparison


@router.post("/{task_id}/pause")
async def pause_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Pause task execution"""
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
    """Resume task execution"""
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
    Manually recover task state

    Useful if the task is stuck or data is corrupted
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


@router.delete("/{task_id}", status_code=204)
async def delete_dataset(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    deleted = await service.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return {}


@router.get("/{task_id}/export")
async def export_task_results(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Export task results to JSON"""
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    result_repo = TaskResultRepository()
    all_results = await result_repo.find_all_by_task(task_id)

    export_data = {
        "task_id": task.id,
        "task_name": task.name,
        "task_type": task.task_type,
        "status": task.status,
        "dataset_id": task.dataset_id,
        "model_ids": task.model_ids,
        "config": task.config.model_dump() if task.config else None,
        "total_samples": task.total_samples,
        "processed_samples": task.processed_samples,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "aggregated_metrics": task.aggregated_metrics,
        "results": [
            {
                "input": r.input,
                "output": r.output,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "execution_time": r.execution_time,
                "variation_type": r.variation_type,
                "original_input": r.original_input,
                "target": r.target,
                "judge_score": r.judge_score,
                "judge_results": r.judge_results,
                "refused": r.refused,
                "include_score": r.include_score,
                "exclude_violations": r.exclude_violations,
                "metrics": r.metrics,
                "metadata": r.metadata,
            }
            for r in all_results
        ],
    }

    return JSONResponse(
        content=to_json_safe(export_data),
        headers={
            "Content-Disposition": f'attachment; filename="task_{task_id}_results.json"'
        },
    )


@router.get("/{task_id}/trustvar-metrics")
async def get_trustvar_metrics(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    """Get TrustVar metrics for task (TSI, EAR, bootstrap CI, tier breakdown)."""
    task = await service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    cached = task.aggregated_metrics.get("_trustvar") if task.aggregated_metrics else None
    if cached:
        return JSONResponse(content=to_json_safe(cached))

    try:
        from fastapi.concurrency import run_in_threadpool

        from src.core.services.eval_service import EvaluationService
    except ImportError as exc:
        logger.warning(
            "trustvar-metrics recompute unavailable (stats stack absent: %s); "
            "returning empty metrics — they are normally cached by post-processing.",
            exc,
        )
        return JSONResponse(content=to_json_safe(_EMPTY_TRUSTVAR_METRICS))

    all_results = await TaskResultRepository().find_all_by_task(task_id)

    if not all_results:
        empty = EvaluationService().compute_trustvar_metrics([])
        return JSONResponse(content=to_json_safe(empty))

    # task.task_type — execution-type enum (standard/variation/…); answer-type
    # (mcq/open_qa/…) and language are inferred per-item inside the service from
    # result.metadata["task_type"] and result.metadata["language"].  No override
    # is passed here so the service uses _infer_task_type per task group.
    results = [
        TaskResult(
            **{
                k: v
                for k, v in r.model_dump().items()
                if k not in {"id", "task_id", "sequence_num", "created_at"}
            }
        )
        for r in all_results
    ]

    metrics = await run_in_threadpool(
        EvaluationService().compute_trustvar_metrics, results
    )
    return JSONResponse(content=to_json_safe(metrics))
