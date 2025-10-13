# src/database/repositories/model_repository.py
from typing import List

from src.config.constants import ModelProvider
from src.core.models.model import Model

from .base import BaseRepository


class ModelRepository(BaseRepository[Model]):
    """Репозиторий для моделей"""

    def __init__(self):
        super().__init__("models", Model)

    async def find_by_provider(self, provider: ModelProvider) -> List[Model]:
        """Найти модели по провайдеру"""
        return await self.find_all({"provider": provider.value})

    async def find_active(self) -> List[Model]:
        """Найти активные модели"""
        return await self.find_all({"is_active": True})
