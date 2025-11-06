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
    test_prompt: str


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


@router.post("/{model_id}/test")
async def test_model(
    model_id: str,
    request: TestModelRequest,
    service: ModelService = Depends(get_model_service),
):
    result = await service.test_model(model_id, request.test_prompt)
    print(result)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
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
