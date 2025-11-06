# src/core/schemas/task.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.constants import TaskStatus


class TaskResult(BaseModel):
    """Результат выполнения задачи"""

    input: str
    output: str
    target: Optional[int] = None
    metrics: Dict[str, float] = {}
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}


class Task(BaseModel):
    """Модель задачи"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0

    # Параметры выполнения
    batch_size: int = 1
    max_samples: Optional[int] = None
    evaluate: bool = True
    evaluation_metrics: List[str] = []

    # Временные метки
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Результаты
    total_samples: int = 0
    processed_samples: int = 0
    results: List[TaskResult] = []
    aggregated_metrics: Dict[str, float] = {}

    # Ошибки
    error: Optional[str] = None

    # Метаданные
    celery_task_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        use_enum_values = True
