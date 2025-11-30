# src/core/tasks/health_check_task.py
import asyncio
from typing import Dict

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.tasks.celery_app import celery_app
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class HealthCheckTask(CeleryTask):
    """Базовый класс для задач проверки здоровья"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка ошибки"""
        logger.error(f"Health check task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=HealthCheckTask, time_limit=60)
def run_health_check_task(self, model_id: str) -> Dict:
    """
    Проверка доступности модели через Celery

    Args:
        model_id: ID модели для проверки

    Returns:
        Dict с результатами проверки
    """
    return asyncio.run(_health_check_async(model_id))


@celery_app.task(bind=True, base=HealthCheckTask, time_limit=120)
def run_test_inference_task(
    self, model_id: str, test_prompt: str = "Hello, how are you?"
) -> Dict:
    """
    Тестовый инференс модели через Celery
    """
    return asyncio.run(_test_inference_async(model_id, test_prompt))


async def _health_check_async(model_id: str) -> Dict:
    """Асинхронная проверка здоровья модели"""

    model_repo = ModelRepository()

    try:
        # Получаем модель
        model = await model_repo.find_by_id(model_id)
        if not model:
            return {
                "success": False,
                "error": f"Model {model_id} not found",
                "model_id": model_id,
            }

        logger.info(f"Running health check for model: {model.name}")

        # Создаем адаптер
        adapter = LLMFactory.create(model)

        # Проверяем доступность
        is_healthy = await adapter.health_check()

        if is_healthy:
            logger.info(f"Model {model.name} is healthy")
            return {
                "success": True,
                "model_id": model_id,
                "model_name": model.name,
                "provider": model.provider,
                "status": "healthy",
            }
        else:
            logger.warning(f"Model {model.name} is not available")
            return {
                "success": False,
                "error": "Model is not available",
                "model_id": model_id,
                "model_name": model.name,
                "status": "unhealthy",
            }

    except Exception as e:
        logger.error(f"Health check failed for model {model_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "status": "error",
        }


async def _test_inference_async(model_id: str, test_prompt: str) -> Dict:
    """Асинхронный тестовый инференс"""

    model_repo = ModelRepository()

    try:
        # Получаем модель
        model = await model_repo.find_by_id(model_id)
        if not model:
            return {"success": False, "error": f"Model {model_id} not found"}

        logger.info(f"Running test inference for model: {model.name}")

        # Создаем адаптер
        adapter = LLMFactory.create(model)

        # Сначала проверяем доступность
        is_healthy = await adapter.health_check()
        if not is_healthy:
            logger.error(
                f"ERROR: Test inference failed for model {model.name}: Model is not healthy"
            )
            return {
                "success": False,
                "error": "Model is not available",
                "model": model.name,
            }

        # Выполняем тестовую генерацию
        import time

        start = time.time()

        response = await adapter.generate(test_prompt)

        duration = time.time() - start

        logger.info(f"Test inference completed in {duration:.2f}s")

        return {
            "success": True,
            "response": response,
            "duration": duration,
            "model": model.name,
            "model_id": model_id,
            "provider": model.provider,
            "test_prompt": test_prompt,
        }

    except Exception as e:
        logger.error(f"Test inference failed for model {model.name}: {e}")
        return {"success": False, "error": str(e), "model_id": model_id}
