# src/core/models/model.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.constants import ModelProvider


class ModelConfig(BaseModel):
    """Конфигурация модели"""

    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = 50
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = []


class Model(BaseModel):
    """Модель LLM"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    provider: ModelProvider
    model_name: str  # Имя модели у провайдера
    description: Optional[str] = None
    config: ModelConfig = Field(default_factory=ModelConfig)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Llama 2 7B",
                "provider": "ollama",
                "model_name": "llama2:7b",
                "description": "Llama 2 7B via Ollama",
            }
        }
