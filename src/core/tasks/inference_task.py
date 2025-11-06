# src/core/tasks/inference_task.py
import asyncio
import time
from typing import List

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.schemas.task import TaskResult, TaskStatus
from src.core.services.eval_service import EvaluationService
from src.core.tasks.celery_app import celery_app
from src.database.repositories.dataset_repository import DatasetRepository
from src.database.repositories.model_repository import ModelRepository
from src.database.repositories.task_repository import TaskRepository
from src.utils.logger import logger


class InferenceTask(CeleryTask):
    """Базовый класс для задач инференса"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка ошибки"""
        logger.error(f"Task {task_id} failed: {exc}")

        # Обновляем статус задачи в БД
        if args:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            task_repo = TaskRepository()
            loop.run_until_complete(
                task_repo.update_status(args[0], TaskStatus.FAILED, error=str(exc))
            )
            loop.close()


@celery_app.task(bind=True, base=InferenceTask)
def run_inference_task(self, task_id: str):
    """Выполнить задачу инференса"""

    # Создаем event loop для async операций
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_run_inference_async(self, task_id))
        return result
    finally:
        loop.close()


async def _run_inference_async(celery_task, task_id: str):
    """Асинхронное выполнение инференса"""

    # Репозитории
    task_repo = TaskRepository()
    dataset_repo = DatasetRepository()
    model_repo = ModelRepository()
    evaluation_service = EvaluationService()

    # Получаем задачу
    task = await task_repo.find_by_id(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    logger.info(f"Starting inference for task {task_id}: {task.name}")

    # Обновляем статус
    await task_repo.update_status(task_id, TaskStatus.RUNNING)

    try:
        # Загружаем датасет
        dataset = await dataset_repo.find_by_id(task.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {task.dataset_id} not found")

        # Загружаем модель
        model = await model_repo.find_by_id(task.model_id)
        if not model:
            raise ValueError(f"Model {task.model_id} not found")

        # Получаем адаптер
        adapter = LLMFactory.create(model)

        # Получаем элементы датасета
        max_samples = task.max_samples or dataset.size
        items = await dataset_repo.get_items(task.dataset_id, limit=max_samples)

        total_items = len(items)
        await task_repo.update(task_id, {"total_samples": total_items})

        logger.info(f"Processing {total_items} items for task {task_id}")

        # Обрабатываем элементы
        results: List[TaskResult] = []
        batch_size = task.batch_size

        for i in range(0, total_items, batch_size):
            batch = items[i : i + batch_size]
            prompts = [item.prompt for item in batch]

            # Генерируем ответы
            start_time = time.time()

            if batch_size > 1:
                responses = await adapter.batch_generate(prompts)
            else:
                responses = [await adapter.generate(prompts[0])]

            execution_time = time.time() - start_time

            # Сохраняем результаты
            for j, (item, response) in enumerate(zip(batch, responses)):
                result = TaskResult(
                    input=item.prompt,
                    output=response,
                    expected_output=item.expected_output,
                    execution_time=execution_time / len(batch),
                    metadata=item.metadata,
                )
                results.append(result)

            # Обновляем прогресс
            processed = min(i + batch_size, total_items)
            progress = (processed / total_items) * 100

            await task_repo.update_progress(task_id, progress, processed)

            # Обновляем состояние Celery задачи
            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "processed": processed,
                    "total": total_items,
                },
            )

            logger.info(f"Task {task_id}: {processed}/{total_items} ({progress:.1f}%)")

        # Оценка результатов
        aggregated_metrics = {}
        if task.evaluate and task.evaluation_metrics:
            logger.info(f"Evaluating results for task {task_id}")
            aggregated_metrics = evaluation_service.evaluate_results(
                results, task.evaluation_metrics
            )

        # Сохраняем результаты
        await task_repo.set_results(task_id, results, aggregated_metrics)

        # Завершаем задачу
        await task_repo.update_status(task_id, TaskStatus.COMPLETED)

        logger.info(f"Task {task_id} completed successfully")

        return {
            "task_id": task_id,
            "status": "completed",
            "total_samples": total_items,
            "metrics": aggregated_metrics,
        }

    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}", exc_info=True)
        await task_repo.update_status(task_id, TaskStatus.FAILED, error=str(e))
        raise
