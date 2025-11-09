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
    RTA = "refuse_to_answer"  # Refuse-to-Answer задачи
    AB_TEST = "ab_test"  # A/B тестирование


class VariationStrategy(str, Enum):
    """Стратегии вариаций"""

    PARAPHRASE = "paraphrase"  # Перефразирование
    STYLE_CHANGE = "style_change"  # Изменение стиля
    COMPLEXITY = "complexity"  # Изменение сложности
    LANGUAGE = "language"  # Перевод на другой язык
    PERSPECTIVE = "perspective"  # Изменение перспективы
    CUSTOM = "custom"  # Кастомная инструкция


class ABTestStrategy(str, Enum):
    """Стратегии A/B тестирования"""

    PROMPT_VARIANTS = "prompt_variants"  # Сравнение разных промптов
    MODEL_COMPARISON = "model_comparison"  # Сравнение моделей на одних данных
    TEMPERATURE_TEST = "temperature_test"  # Тестирование температур
    SYSTEM_PROMPT_TEST = "system_prompt_test"  # Тестирование системных промптов
    PARAMETER_SWEEP = "parameter_sweep"  # Перебор параметров


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

    # Для LLM judge и RTA
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None

    # Для RTA
    refused: Optional[bool] = None  # Отказалась ли модель отвечать

    # Для Include/Exclude
    include_score: Optional[float] = None
    exclude_violations: Optional[int] = None

    # Для A/B тестов
    ab_variant: Optional[str] = None  # Вариант A/B теста


class VariationConfig(BaseModel):
    """Конфигурация вариаций"""

    enabled: bool = False
    model_id: Optional[str] = None  # Модель для генерации вариаций
    strategies: List[VariationStrategy] = []
    count_per_strategy: int = 1
    custom_prompt: Optional[str] = None  # Кастомный промпт для вариаций


class JudgeConfig(BaseModel):
    """Конфигурация LLM Judge"""

    enabled: bool = False
    model_id: Optional[str] = None
    criteria: List[str] = ["accuracy", "relevance", "completeness"]
    custom_prompt_template: Optional[str] = None


class RTAConfig(BaseModel):
    """Конфигурация Refuse-to-Answer"""

    enabled: bool = False
    rta_judge_model_id: Optional[str] = None
    rta_prompt_template: Optional[str] = None  # Промпт для RTA судьи
    refusal_keywords: List[str] = [
        "I cannot",
        "I can't",
        "I'm not able to",
        "I refuse",
        "I apologize, but",
        "I'm sorry, but",
        "I cannot assist",
    ]


class ABTestConfig(BaseModel):
    """Конфигурация A/B тестов"""

    enabled: bool = False
    strategy: ABTestStrategy = ABTestStrategy.MODEL_COMPARISON

    # Для PROMPT_VARIANTS
    prompt_variants: Optional[Dict[str, str]] = None  # variant_name -> prompt

    # Для TEMPERATURE_TEST
    temperatures: Optional[List[float]] = None  # [0.3, 0.7, 1.0]

    # Для SYSTEM_PROMPT_TEST
    system_prompts: Optional[Dict[str, str]] = None  # variant_name -> system_prompt

    # Для PARAMETER_SWEEP
    parameter_ranges: Optional[Dict[str, List[Any]]] = None  # param_name -> [values]

    # Общие настройки
    sample_size_per_variant: Optional[int] = None  # Размер выборки для каждого варианта
    statistical_test: str = "t_test"  # t_test, chi_square, mann_whitney


class TaskConfig(BaseModel):
    """Конфигурация задачи"""

    # Базовые настройки
    batch_size: int = 1
    max_samples: Optional[int] = None

    # Оценка
    evaluate: bool = True
    evaluation_metrics: List[str] = []

    # Вариации
    variations: VariationConfig = Field(default_factory=VariationConfig)

    # LLM Judge
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # Refuse-to-Answer
    rta: RTAConfig = Field(default_factory=RTAConfig)

    # A/B тесты
    ab_test: ABTestConfig = Field(default_factory=ABTestConfig)


class Task(BaseModel):
    """Модель задачи"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_ids: List[str]  # Список моделей
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

    # Для A/B тестов
    ab_test_results: Optional[Dict[str, Any]] = None  # Статистический анализ

    # Ошибки
    error: Optional[str] = None

    # Метаданные
    celery_task_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        use_enum_values = True
