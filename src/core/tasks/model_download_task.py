# src/core/tasks/model_download_task.py

import asyncio
import time
import traceback

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.schemas.model import ModelStatus
from src.core.tasks.celery_app import celery_app
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class ModelDownloadTask(CeleryTask):
    """Base class for model download tasks"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Error handling"""
        # Log error before attempting status update
        logger.error(
            f"Download task {task_id} failed: {exc}. Traceback: {traceback.format_exc()}"
        )

        # Update model status in DB
        if args:
            model_id = args[0]
            logger.info(
                f"Attempting to set FAILED status for model {model_id} on task failure."
            )
            try:
                asyncio.run(self._update_status_on_failure(model_id))
            except Exception as e:
                # Log if even status update failed
                logger.error(
                    f"Could not update status for model {model_id} on failure: {e}. Traceback: {traceback.format_exc()}"
                )

    # Helper async function for on_failure
    async def _update_status_on_failure(self, model_id: str):
        model_repo = ModelRepository()
        await model_repo.update_status(model_id, ModelStatus.FAILED)
        logger.info(f"Successfully set FAILED status for model {model_id}.")


@celery_app.task(bind=True, base=ModelDownloadTask)
def run_download_model_task(self, model_id: str):
    """Execute model download task"""
    try:
        return asyncio.run(_download_model_async(self, model_id))
    except Exception as e:
        # Log exception at task level so Celery can handle it
        # and call on_failure.
        logger.error(
            f"Exception in run_download_model_task for model {model_id}: {e}. Traceback: {traceback.format_exc()}",
            exc_info=True,
        )
        # Re-raise exception so Celery knows task failed
        raise


async def _download_model_async(celery_task, model_id: str):
    """Asynchronous model download execution"""
    model_repo = ModelRepository()

    model = await model_repo.find_by_id(model_id)
    if not model:
        # Better to throw a more specific error
        raise ValueError(f"Model {model_id} not found in database")

    logger.info(
        f"Starting download for model {model_id}: {model.name} (provider: {model.provider})"
    )

    await model_repo.update_status(model_id, ModelStatus.DOWNLOADING)

    # Select adapter and start download
    adapter = LLMFactory.create(model)

    start_time = time.time()

    is_downloaded = await adapter.download_model()

    if is_downloaded:
        execution_time = time.time() - start_time

        # Complete task
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

        # Complete task
        await model_repo.update_status(model_id, ModelStatus.FAILED)

        logger.info(f"Model {model_id} was NOT downloaded in {execution_time:.2f}s")

        return {
            "model_id": model_id,
            "status": "failed",
            "execution_time": execution_time,
        }
