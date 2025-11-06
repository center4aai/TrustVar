# src/database/repositories/model_repository.py
from typing import List

from src.config.constants import ModelProvider
from src.core.schemas.model import Model, ModelStatus
from src.utils.logger import logger

from .base import BaseRepository


class ModelRepository(BaseRepository[Model]):
    """Репозиторий для моделей"""

    def __init__(self):
        super().__init__("models", Model)

    async def find_by_provider(self, provider: ModelProvider) -> List[Model]:
        """Найти модели по провайдеру"""
        return await self.find_all({"provider": provider.value})

    async def find_active(self) -> List[Model]:
        """Найти зарегистрированные модели"""
        return await self.find_all({"status": "registered"})

    async def update_status(self, model_id: str, status: ModelStatus):
        """Обновить статус задачи"""
        update_data = {"status": status.value}

        logger.info(f"Updating task {model_id} status to {status}")
        return await self.update(model_id, update_data)
