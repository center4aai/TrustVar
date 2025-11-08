# src/core/schemas/task.py
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.constants import TaskStatus


class TaskType(str, Enum):
    """Типы задач"""

    STANDARD = "standard"  # Обычный инференс
    VARIATION = "variation"  # С вариациями промптов
    COMPARISON = "comparison"  # Сравнение моделей
    JUDGED = "judged"  # С использованием LLM judge


class VariationStrategy(str, Enum):
    """Стратегии вариаций"""

    PARAPHRASE = "paraphrase"  # Перефразирование
    STYLE_CHANGE = "style_change"  # Изменение стиля
    COMPLEXITY = "complexity"  # Изменение сложности
    LANGUAGE = "language"  # Перевод на другой язык
    PERSPECTIVE = "perspective"  # Изменение перспективы
    CUSTOM = "custom"  # Кастомная инструкция


class TaskResult(BaseModel):
    """Результат выполнения задачи"""

    input: str
    output: str
    model_id: str  # ID модели, которая сгенерировала результат
    target: Optional[str] = None
    metrics: Dict[str, float] = {}
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}

    # Для вариаций
    original_input: Optional[str] = None  # Оригинальный промпт
    variation_type: Optional[str] = None  # Тип вариации

    # Для LLM judge
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None


class TaskConfig(BaseModel):
    """Конфигурация задачи"""

    # Базовые настройки
    batch_size: int = 1
    max_samples: Optional[int] = None

    # Оценка
    evaluate: bool = True
    evaluation_metrics: List[str] = []

    # Вариации
    enable_variations: bool = False
    variation_model_id: Optional[str] = None  # Модель для создания вариаций
    variation_strategies: List[VariationStrategy] = []
    variations_per_prompt: int = 1  # Количество вариаций на промпт

    # LLM Judge
    enable_judge: bool = False
    judge_model_id: Optional[str] = None
    judge_prompt_template: Optional[str] = None  # Кастомный промпт для judge
    judge_criteria: List[str] = ["accuracy", "relevance", "completeness"]


class Task(BaseModel):
    """Модель задачи"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_ids: List[str]  # Теперь список моделей
    task_type: TaskType = TaskType.STANDARD
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0

    # Конфигурация
    config: TaskConfig = Field(default_factory=TaskConfig)

    # Временные метки
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Результаты
    total_samples: int = 0
    processed_samples: int = 0
    results: List[TaskResult] = []
    aggregated_metrics: Dict[str, Dict[str, float]] = {}  # model_id -> metrics

    # Ошибки
    error: Optional[str] = None

    # Метаданные
    celery_task_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        use_enum_values = True
