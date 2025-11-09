# src/core/schemas/dataset.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """Элемент датасета"""

    prompt: str
    target: Optional[str] = None  # Может быть None если target_column не указан

    # Для задач Include/Exclude
    include_list: Optional[List[str]] = None
    exclude_list: Optional[List[str]] = None

    metadata: Dict[str, Any] = {}

    model_config = {"coerce_numbers_to_str": True}


class Dataset(BaseModel):
    """Модель датасета"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    format: str = "jsonl"
    size: int = 0
    task_type: str
    tags: List[str] = []

    # Конфигурация столбцов
    prompt_column: str = "prompt"  # Имя столбца с промптами
    target_column: Optional[str] = None  # Имя столбца с целевыми значениями (если есть)
    include_column: Optional[str] = None  # Имя столбца с include_list
    exclude_column: Optional[str] = None  # Имя столбца с exclude_list

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Dataset",
                "description": "Test dataset for QA",
                "format": "jsonl",
                "task_type": "question-answering",
                "tags": ["qa", "test"],
                "prompt_column": "prompt",
                "target_column": "answer",
            }
        }
