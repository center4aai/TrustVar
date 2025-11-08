# src/core/tasks/inference_task.py
import asyncio
import time
from typing import List

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.schemas.task import TaskResult, TaskStatus
from src.core.services.eval_service import EvaluationService
from src.core.services.judge_service import LLMJudgeService
from src.core.tasks.celery_app import celery_app
from src.core.tasks.variation_task import PromptVariationGenerator
from src.database.repositories.dataset_repository import DatasetRepository
from src.database.repositories.model_repository import ModelRepository
from src.database.repositories.task_repository import TaskRepository
from src.utils.logger import logger


class InferenceTask(CeleryTask):
    """Базовый класс для задач инференса"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка ошибки"""
        logger.error(f"Task {task_id} failed: {exc}")

        if args:
            task_repo = TaskRepository()
            asyncio.run(
                task_repo.update_status(args[0], TaskStatus.FAILED, error=str(exc))
            )


@celery_app.task(bind=True, base=InferenceTask)
def run_inference_task(self, task_id: str):
    """Выполнить задачу инференса"""
    return asyncio.run(_run_inference_async(self, task_id))


async def _run_inference_async(celery_task, task_id: str):
    """Асинхронное выполнение инференса с поддержкой вариаций и judge"""

    # Репозитории и сервисы
    task_repo = TaskRepository()
    dataset_repo = DatasetRepository()
    model_repo = ModelRepository()
    evaluation_service = EvaluationService()

    # Получаем задачу
    task = await task_repo.find_by_id(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    logger.info(f"Starting inference for task {task_id}: {task.name}")
    await task_repo.update_status(task_id, TaskStatus.RUNNING)

    try:
        # Загружаем датасет
        dataset = await dataset_repo.find_by_id(task.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {task.dataset_id} not found")

        # Загружаем модели
        models = []
        adapters = {}
        for model_id in task.model_ids:
            model = await model_repo.find_by_id(model_id)
            if not model:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            models.append(model)
            adapters[model_id] = LLMFactory.create(model)

        if not models:
            raise ValueError("No valid models found for task")

        logger.info(f"Loaded {len(models)} models for inference")

        # Инициализируем генератор вариаций (если нужно)
        variation_generator = None
        if task.config.enable_variations and task.config.variation_model_id:
            variation_generator = PromptVariationGenerator(
                task.config.variation_model_id
            )
            await variation_generator.initialize()
            logger.info("Variation generator initialized")

        # Инициализируем LLM judge (если нужно)
        judge_service = None
        if task.config.enable_judge and task.config.judge_model_id:
            judge_service = LLMJudgeService(task.config.judge_model_id)
            await judge_service.initialize()
            logger.info("LLM Judge initialized")

        # Получаем элементы датасета
        max_samples = task.config.max_samples or dataset.size
        items = await dataset_repo.get_items(task.dataset_id, limit=max_samples)
        total_items = len(items)
        await task_repo.update(task_id, {"total_samples": total_items})

        logger.info(f"Processing {total_items} items")

        # Обрабатываем элементы
        all_results: List[TaskResult] = []
        batch_size = task.config.batch_size

        for i in range(0, total_items, batch_size):
            batch = items[i : i + batch_size]

            # Для каждого элемента батча
            for item in batch:
                prompts_to_process = []

                # Основной промпт
                prompts_to_process.append(
                    {
                        "text": item.prompt,
                        "original": item.prompt,
                        "variation_type": None,
                    }
                )

                # Генерируем вариации (если включено)
                if variation_generator:
                    variations = await variation_generator.generate_variations(
                        prompt=item.prompt,
                        strategies=task.config.variation_strategies,
                        count_per_strategy=task.config.variations_per_prompt,
                    )

                    for var in variations:
                        prompts_to_process.append(
                            {
                                "text": var["text"],
                                "original": item.prompt,
                                "variation_type": var["strategy"],
                            }
                        )

                    logger.info(f"Generated {len(variations)} variations for item")

                # Прогоняем каждый промпт (оригинал + вариации) через все модели
                for prompt_data in prompts_to_process:
                    for model in models:
                        adapter = adapters[model.id]

                        try:
                            start_time = time.time()

                            # Генерация
                            response = await adapter.generate(prompt_data["text"])

                            execution_time = time.time() - start_time

                            # Создаем результат
                            result = TaskResult(
                                input=prompt_data["text"],
                                output=response,
                                model_id=model.id,
                                target=item.target,
                                execution_time=execution_time,
                                metadata={
                                    **item.metadata,
                                    "model_name": model.name,
                                    "model_provider": model.provider,
                                },
                                original_input=prompt_data["original"],
                                variation_type=prompt_data["variation_type"],
                            )

                            # LLM Judge оценка (если включено)
                            if judge_service:
                                judge_result = await judge_service.evaluate_output(
                                    input_prompt=prompt_data["text"],
                                    model_output=response,
                                    task_description=task.name,
                                    reference_output=item.target,
                                    criteria=task.config.judge_criteria,
                                )

                                result.judge_score = judge_result["overall_score"]
                                result.judge_reasoning = judge_result["reasoning"]
                                result.metadata["judge_criteria_scores"] = judge_result[
                                    "criteria_scores"
                                ]

                                logger.info(f"Judge score: {result.judge_score:.2f}")

                            all_results.append(result)

                        except Exception as e:
                            logger.error(
                                f"Error processing with model {model.name}: {e}"
                            )
                            continue

            # Обновляем прогресс
            processed = min(i + batch_size, total_items)
            progress = (processed / total_items) * 100

            await task_repo.update_progress(task_id, progress, processed)

            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "processed": processed,
                    "total": total_items,
                },
            )

            logger.info(f"Task {task_id}: {processed}/{total_items} ({progress:.1f}%)")

        # Классическая оценка метрик
        aggregated_metrics = {}
        if task.config.evaluate and task.config.evaluation_metrics:
            logger.info("Evaluating results")

            # Группируем результаты по моделям
            results_by_model = {}
            for result in all_results:
                if result.model_id not in results_by_model:
                    results_by_model[result.model_id] = []
                results_by_model[result.model_id].append(result)

            # Оцениваем каждую модель отдельно
            for model_id, model_results in results_by_model.items():
                model_metrics = evaluation_service.evaluate_results(
                    model_results, task.config.evaluation_metrics
                )
                aggregated_metrics[model_id] = model_metrics

        # Сохраняем результаты
        await task_repo.set_results(task_id, all_results, aggregated_metrics)

        # Завершаем задачу
        await task_repo.update_status(task_id, TaskStatus.COMPLETED)

        logger.info(f"Task {task_id} completed successfully")

        return {
            "task_id": task_id,
            "status": "completed",
            "total_samples": total_items,
            "total_results": len(all_results),
            "models_tested": len(models),
            "variations_used": variation_generator is not None,
            "judge_used": judge_service is not None,
            "metrics": aggregated_metrics,
        }

    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}", exc_info=True)
        await task_repo.update_status(task_id, TaskStatus.FAILED, error=str(e))
        raise
