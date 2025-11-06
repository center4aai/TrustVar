# src/core/services/model_service.py
from typing import List, Optional

from src.adapters.factory import LLMFactory
from src.config.constants import ModelProvider
from src.core.schemas.model import Model, ModelConfig
from src.core.tasks.model_download_task import run_download_model_task
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class ModelService:
    """Сервис для работы с моделями"""

    def __init__(self):
        self.repository = ModelRepository()

    async def register_model(
        self,
        name: str,
        provider: str,
        model_name: str,
        description: str = None,
        config: ModelConfig = None,
    ) -> Model:
        """Зарегистрировать модель"""
        model = Model(
            name=name,
            provider=provider,
            model_name=model_name,
            description=description,
            config=config or ModelConfig(),
        )

        created = await self.repository.create(model)

        if (
            model.provider == ModelProvider.HUGGINGFACE
            or model.provider == ModelProvider.OLLAMA
        ):
            # Запускаем Celery задачу
            celery_task = run_download_model_task.delay(created.id)

            # Сохраняем Celery task ID
            await self.repository.update(created.id, {"celery_task_id": celery_task.id})

        logger.info(f"Model registered: {created.id} - {created.name}")

        return created

    async def get_model(self, model_id: str) -> Optional[Model]:
        """Получить модель"""
        return await self.repository.find_by_id(model_id)

    async def list_models(self, active_only: bool = False) -> List[Model]:
        """Список моделей"""
        if active_only:
            return await self.repository.find_active()
        return await self.repository.find_all()

    async def update_model(self, model_id: str, **kwargs) -> Optional[Model]:
        """Обновить модель"""
        result = await self.repository.update(model_id, kwargs)

        if result:
            # cache.delete(f"model:{model_id}")
            logger.info(f"Model updated: {model_id}")

        return result

    async def delete_model(self, model_id: str) -> bool:
        """Удалить модель"""
        result = await self.repository.delete(model_id)

        if result:
            # cache.clear_pattern(f"model:{model_id}*")
            logger.info(f"Model deleted: {model_id}")

        return result

    async def test_model(self, model_id: str, test_prompt: str = "Hello") -> dict:
        """Тестировать модель"""
        model = await self.get_model(model_id)
        if not model:
            return {"success": False, "error": "Model not found"}

        try:
            adapter = LLMFactory.create(model)

            # Проверяем health
            is_healthy = await adapter.health_check()
            if not is_healthy:
                return {"success": False, "error": "Model not available"}

            # Тестовая генерация
            import time

            start = time.time()
            response = await adapter.generate(test_prompt)
            duration = time.time() - start

            return {
                "success": True,
                "response": response,
                "duration": duration,
                "model": model.name,
            }
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return {"success": False, "error": str(e)}
