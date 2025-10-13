# src/core/models/dataset.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """Элемент датасета"""

    prompt: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Dataset(BaseModel):
    """Модель датасета"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    format: str = "jsonl"
    size: int = 0
    task_type: str
    tags: List[str] = []
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
            }
        }
