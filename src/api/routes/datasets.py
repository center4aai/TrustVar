# src/api/routes/datasets.py
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.config.constants import DatasetFormat, TaskStatus
from src.core.schemas.dataset import Dataset, DatasetItem
from src.core.services.dataset_service import DatasetService
from src.core.taxonomy import CANONICAL_TASK_TYPES, normalize_task_type

router = APIRouter()


# Dependency Injection for service
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
    task_semantics: Optional[str] = Form(None),
    tags: str = Form(None),  # Tags passed as comma-separated string "tag1,tag2"
    service: DatasetService = Depends(get_dataset_service),
):
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    canon = normalize_task_type(task_type, default=None)
    if canon is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_type '{task_type}'. Must be one of {list(CANONICAL_TASK_TYPES)}.",
        )
    try:
        dataset = await service.create_dataset(
            name=name, description=description, task_type=canon,
            task_semantics=task_semantics, tags=tag_list
        )
        return dataset
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/upload")
async def upload_dataset_file(
    dataset_id: str,
    file: UploadFile = File(...),
    file_format: DatasetFormat = Form(...),
    prompt_column: str = Form("prompt"),
    target_column: Optional[str] = Form(None),
    include_column: Optional[str] = Form(None),
    exclude_column: Optional[str] = Form(None),
    template_column: Optional[str] = Form(None),
    variables_columns: Optional[str] = Form(None),
    service: DatasetService = Depends(get_dataset_service),
):
    if not await service.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    variables_list = None
    if variables_columns:
        variables_list = [v.strip() for v in variables_columns.split(",") if v.strip()]

    count = await service.upload_from_file(
        dataset_id,
        file.file,
        file_format.value,
        prompt_column,
        target_column,
        include_column,
        exclude_column,
        template_column,
        variables_list,
    )
    return {"filename": file.filename, "items_uploaded": count}


@router.get("/", response_model=List[Dataset])
async def list_datasets(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=0, le=1000, description="Max number of records to return"
    ),
    service: DatasetService = Depends(get_dataset_service),
):
    return await service.list_datasets(skip=skip, limit=limit)


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
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=0, le=1000, description="Max number of records to return"
    ),
    service: DatasetService = Depends(get_dataset_service),
):
    items = await service.get_items(dataset_id, skip=skip, limit=limit)
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
