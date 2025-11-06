# src/adapters/base.py
from abc import ABC, abstractmethod
from typing import List

from src.core.schemas.model import Model


class BaseLLMAdapter(ABC):
    """Базовый адаптер для LLM"""

    def __init__(self, model: Model):
        self.model = model
        self.config = model.config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерация ответа на один промпт"""
        pass

    @abstractmethod
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Пакетная генерация"""
        pass

    async def health_check(self) -> bool:
        """Проверка доступности модели"""
        pass
