# src/core/services/model_service.py
from typing import List, Optional

from celery.result import AsyncResult

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

        # Автоматическая загрузка для HF и Ollama
        if (
            model.provider == ModelProvider.HUGGINGFACE
            or model.provider == ModelProvider.OLLAMA
        ):
            logger.info(f"Starting downloading model {model.name}")
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
            logger.info(f"Model updated: {model_id}")

        return result

    async def delete_model(self, model_id: str) -> bool:
        """Удалить модель"""
        result = await self.repository.delete(model_id)

        if result:
            logger.info(f"Model deleted: {model_id}")

        return result

    def test_model(
        self, model_id: str, test_prompt: str = "Hello, how are you?"
    ) -> dict:
        """
        Тестировать модель через Celery (асинхронно)

        Returns:
            Dict с celery_task_id для отслеживания
        """
        from src.core.tasks.health_check_task import run_test_inference_task

        logger.info(f"Scheduling test inference for model {model_id}")

        # Запускаем Celery задачу
        celery_task = run_test_inference_task.delay(model_id, test_prompt)

        return {
            "celery_task_id": celery_task.id,
            "status": "scheduled",
            "message": "Test inference task has been scheduled",
        }

    def get_test_result(self, celery_task_id: str) -> dict:
        """
        Получить результат тестового инференса

        Args:
            celery_task_id: ID Celery задачи

        Returns:
            Dict с результатами или статусом
        """
        from src.core.tasks.celery_app import celery_app

        result = AsyncResult(celery_task_id, app=celery_app)

        if result.ready():
            if result.successful():
                return {"status": "completed", "result": result.result}
            else:
                return {"status": "failed", "error": str(result.result)}
        else:
            return {"status": "pending", "state": result.state}

    def health_check(self, model_id: str) -> dict:
        """
        Проверить доступность модели через Celery (асинхронно)

        Returns:
            Dict с celery_task_id для отслеживания
        """
        from src.core.tasks.health_check_task import run_health_check_task

        logger.info(f"Scheduling health check for model {model_id}")

        # Запускаем Celery задачу
        celery_task = run_health_check_task.delay(model_id)

        return {
            "celery_task_id": celery_task.id,
            "status": "scheduled",
            "message": "Health check task has been scheduled",
        }

    def get_health_check_result(self, celery_task_id: str) -> dict:
        """
        Получить результат health check

        Args:
            celery_task_id: ID Celery задачи

        Returns:
            Dict с результатами или статусом
        """
        from src.core.tasks.celery_app import celery_app

        result = AsyncResult(celery_task_id, app=celery_app)

        if result.ready():
            if result.successful():
                return {"status": "completed", "result": result.result}
            else:
                return {"status": "failed", "error": str(result.result)}
        else:
            return {"status": "pending", "state": result.state}
