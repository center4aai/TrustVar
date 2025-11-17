# src/core/schemas/dataset.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

TDatasetItem = TypeVar("TDatasetItem", bound="DatasetItem")


class DatasetItem(BaseModel):
    """Элемент датасета"""

    prompt: str
    target: Optional[str] = "1"

    include_list: Optional[List[str]] = None
    exclude_list: Optional[List[str]] = None

    metadata: Dict[str, Any] = {}

    model_config = {"coerce_numbers_to_str": True}

    @classmethod
    def from_row(
        cls: Type[TDatasetItem],
        row: Dict[str, Any],
        prompt_column: str,
        target_column: Optional[str] = None,
        include_column: Optional[str] = None,
        exclude_column: Optional[str] = None,
    ) -> TDatasetItem:
        """
        Создает экземпляр DatasetItem из строки данных (dict) и маппинга колонок.
        Все неиспользованные колонки автоматически попадают в metadata.
        """
        if prompt_column not in row or row[prompt_column] is None:
            raise ValueError(
                f"Required prompt column '{prompt_column}' not found or is empty in the row."
            )

        item_data = {"prompt": row.get(prompt_column)}

        # Маппинг опциональных полей
        if target_column:
            if "_default" in target_column:
                item_data["target"] = target_column.split("_default")[0]
            elif row.get(target_column) is not None:
                item_data["target"] = row.get(target_column)

        if include_column and row.get(include_column) is not None:
            item_data["include_list"] = row.get(include_column)

        if exclude_column and row.get(exclude_column) is not None:
            item_data["exclude_list"] = row.get(exclude_column)

        # Собираем все остальные данные в metadata
        mapped_columns = {
            prompt_column,
            target_column,
            include_column,
            exclude_column,
        }
        # Удаляем None, если какие-то колонки не были выбраны
        mapped_columns.discard(None)

        item_data["metadata"] = {
            key: value for key, value in row.items() if key not in mapped_columns
        }

        try:
            # Валидируем преобразованные данные через Pydantic
            return cls(**item_data)
        except ValidationError as e:
            # Добавляем контекст к ошибке валидации
            raise ValueError(f"Failed to validate row after mapping: {e}") from e


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
