# src/core/tasks/model_download_task.py

import asyncio
import time

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.schemas.model import ModelStatus
from src.core.tasks.celery_app import celery_app
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class ModelDownloadTask(CeleryTask):
    """Базовый класс для задач загрузки моделей"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка ошибки"""
        # Логируем ошибку до попытки обновления статуса
        logger.error(f"Download task {task_id} failed: {exc}")

        # Обновляем статус модели в БД
        if args:
            model_id = args[0]
            logger.info(
                f"Attempting to set FAILED status for model {model_id} on task failure."
            )
            try:
                asyncio.run(self._update_status_on_failure(model_id))
            except Exception as e:
                # Логируем, если даже обновление статуса упало
                logger.error(
                    f"Could not update status for model {model_id} on failure: {e}"
                )

    # Вспомогательная async-функция для on_failure
    async def _update_status_on_failure(self, model_id: str):
        model_repo = ModelRepository()
        await model_repo.update_status(model_id, ModelStatus.FAILED)
        logger.info(f"Successfully set FAILED status for model {model_id}.")


@celery_app.task(bind=True, base=ModelDownloadTask)
def run_download_model_task(self, model_id: str):
    """Выполнить задачу загрузки модели"""
    try:
        return asyncio.run(_download_model_async(self, model_id))
    except Exception as e:
        # Логируем исключение на уровне задачи, чтобы Celery мог его обработать
        # и вызвать on_failure.
        logger.error(
            f"Exception in run_download_model_task for model {model_id}: {e}",
            exc_info=True,
        )
        # Перевыбрасываем исключение, чтобы Celery понял, что задача провалилась
        raise


async def _download_model_async(celery_task, model_id: str):
    """Асинхронное выполнение загрузки модели"""
    # ВАЖНО: Репозиторий создается здесь, внутри асинхронной функции.
    # Это гарантирует, что он будет использовать правильный event loop.
    model_repo = ModelRepository()

    model = await model_repo.find_by_id(model_id)
    if not model:
        # Лучше выбрасывать более конкретную ошибку
        raise ValueError(f"Model {model_id} not found in database")

    logger.info(
        f"Starting download for model {model_id}: {model.name} (provider: {model.provider})"
    )

    await model_repo.update_status(model_id, ModelStatus.DOWNLOADING)

    # Выбираем адаптер и запускаем загрузку
    adapter = LLMFactory.create(model)

    start_time = time.time()

    is_downloaded = await adapter.download_model()

    if is_downloaded:
        execution_time = time.time() - start_time

        # Завершаем задачу
        await model_repo.update_status(model_id, ModelStatus.REGISTERED)

        logger.info(
            f"Model {model_id} downloaded successfully in {execution_time:.2f}s"
        )

        return {
            "model_id": model_id,
            "status": "registered",
            "execution_time": execution_time,
        }
    else:
        execution_time = time.time() - start_time

        # Завершаем задачу
        await model_repo.update_status(model_id, ModelStatus.FAILED)

        logger.info(f"Model {model_id} was NOT downloaded in {execution_time:.2f}s")

        return {
            "model_id": model_id,
            "status": "failed",
            "execution_time": execution_time,
        }
