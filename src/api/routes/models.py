# src/api/routes/models.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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
    return await service.get_model(model_id=model_id)


@router.post("/{model_id}/test", response_model=TaskStatusResponse)
async def test_model(
    model_id: str,
    request: TestModelRequest,
    service: ModelService = Depends(get_model_service),
):
    """
    Запустить тестовый инференс модели (асинхронно через Celery)

    Возвращает celery_task_id для отслеживания результата
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
    Получить результат тестового инференса

    Статусы:
    - pending: задача еще выполняется
    - completed: задача завершена успешно
    - failed: задача завершилась с ошибкой
    """
    result = service.get_test_result(celery_task_id)
    return result


@router.post("/{model_id}/health", response_model=TaskStatusResponse)
async def health_check(
    model_id: str,
    service: ModelService = Depends(get_model_service),
):
    """
    Проверить доступность модели (асинхронно через Celery)

    Возвращает celery_task_id для отслеживания результата
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
    Получить результат health check

    Статусы:
    - pending: проверка еще выполняется
    - completed: проверка завершена
    - failed: проверка завершилась с ошибкой
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
