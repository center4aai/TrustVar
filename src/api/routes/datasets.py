# src/api/routes/datasets.py
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.config.constants import DatasetFormat, TaskStatus
from src.core.schemas.dataset import Dataset, DatasetItem
from src.core.services.dataset_service import DatasetService

router = APIRouter()


# Dependency Injection для сервиса
def get_dataset_service():
    return DatasetService()


class DatasetList(BaseModel):
    status: Optional[TaskStatus] = None
    skip: int = 0
    limit: int = 100


@router.post("/", response_model=Dataset, status_code=201)
async def create_dataset(
    name: str = Form(...),
    description: str = Form(None),
    task_type: str = Form(...),
    tags: str = Form(None),  # Теги передаются как строка "tag1,tag2"
    service: DatasetService = Depends(get_dataset_service),
):
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    try:
        dataset = await service.create_dataset(
            name=name, description=description, task_type=task_type, tags=tag_list
        )
        return dataset
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/upload")
async def upload_dataset_file(
    dataset_id: str,
    file: UploadFile = File(...),
    file_format: DatasetFormat = Form(...),
    service: DatasetService = Depends(get_dataset_service),
):
    if not await service.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    # try:
    count = await service.upload_from_file(dataset_id, file.file, file_format.value)
    return {"filename": file.filename, "items_uploaded": count}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")


@router.get("/", response_model=List[Dataset])
async def list_datasets(
    request: DatasetList,
    service: DatasetService = Depends(get_dataset_service),
):
    return await service.list_datasets(skip=request.skip, limit=request.limit)


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
):
    dataset = await service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/{dataset_id}/items", response_model=List[DatasetItem])
async def get_dataset_items(
    dataset_id: str,
    request: DatasetList,
    service: DatasetService = Depends(get_dataset_service),
):
    items = await service.get_items(dataset_id, skip=request.skip, limit=request.limit)
    return items


@router.get("/{dataset_id}/stats", response_model=dict)
async def get_dataset_stats(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
):
    stats = await service.get_stats(dataset_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")
    return stats


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
):
    deleted = await service.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {}
