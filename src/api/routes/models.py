# src/api/routes/models.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.config.constants import ModelProvider
from src.core.schemas.model import Model, ModelConfig
from src.core.services.model_service import ModelService

router = APIRouter()


def get_model_service():
    return ModelService()


class ModelCreate(BaseModel):
    name: str
    provider: ModelProvider
    model_name: str
    description: str | None = None
    config: ModelConfig


class TestModelRequest(BaseModel):
    test_prompt: str = "Hello, how are you?"


class TaskStatusResponse(BaseModel):
    celery_task_id: str
    status: str
    message: str


class TaskResultResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None
    state: str | None = None


class BulkRegisterRequest(BaseModel):
    model_names: List[str]
    config: ModelConfig = Field(default_factory=ModelConfig)
    description: str | None = None
    provider: ModelProvider = ModelProvider.OLLAMA
    pull_if_missing: bool = True


@router.get("/available")
async def list_available_models(
    service: ModelService = Depends(get_model_service),
):
    try:
        models = await service.list_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {e}",
        )


@router.post("/bulk-register")
async def bulk_register_models(
    request: BulkRegisterRequest,
    service: ModelService = Depends(get_model_service),
):
    try:
        result = await service.bulk_register_models(
            model_names=request.model_names,
            config=request.config,
            description=request.description,
            provider=request.provider,
            pull_if_missing=request.pull_if_missing,
        )
        return {
            "created": [m.model_dump() for m in result["created"]],
            "skipped": result["skipped"],
            "downloading": result["downloading"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/", response_model=Model, status_code=201)
async def register_model(
    model_data: ModelCreate,
    service: ModelService = Depends(get_model_service),
):
    try:
        model = await service.register_model(
            name=model_data.name,
            provider=model_data.provider,
            model_name=model_data.model_name,
            description=model_data.description,
            config=model_data.config,
        )
        return model
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Model])
async def list_models(
    active_only: bool = False,
    service: ModelService = Depends(get_model_service),
):
    return await service.list_models(active_only=active_only)


@router.get("/{model_id}/get", response_model=Model)
async def get_model(
    model_id: str,
    service: ModelService = Depends(get_model_service),
):
    model = await service.get_model(model_id=model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/{model_id}/test", response_model=TaskStatusResponse)
async def test_model(
    model_id: str,
    request: TestModelRequest,
    service: ModelService = Depends(get_model_service),
):
    """
    Start model test inference (asynchronously via Celery)

    Returns celery_task_id for tracking the result
    """
    result = service.test_model(model_id, request.test_prompt)
    return result


@router.get("/{model_id}/test/{celery_task_id}", response_model=TaskResultResponse)
async def get_test_result(
    model_id: str,
    celery_task_id: str,
    service: ModelService = Depends(get_model_service),
):
    """
    Get test inference result

    Statuses:
    - pending: task is still running
    - completed: task completed successfully
    - failed: task failed with error
    """
    result = service.get_test_result(celery_task_id)
    return result


@router.post("/{model_id}/health", response_model=TaskStatusResponse)
async def health_check(
    model_id: str,
    service: ModelService = Depends(get_model_service),
):
    """
    Check model availability (asynchronously via Celery)

    Returns celery_task_id for tracking the result
    """
    result = service.health_check(model_id)
    return result


@router.get("/{model_id}/health/{celery_task_id}", response_model=TaskResultResponse)
async def get_health_check_result(
    model_id: str,
    celery_task_id: str,
    service: ModelService = Depends(get_model_service),
):
    """
    Get health check result

    Statuses:
    - pending: check is still running
    - completed: check completed
    - failed: check failed with error
    """
    result = service.get_health_check_result(celery_task_id)
    return result


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    service: ModelService = Depends(get_model_service),
):
    deleted = await service.delete_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    return {}
