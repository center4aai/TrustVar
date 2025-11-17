# src/core/tasks/inference_task.py
import asyncio
import time
from typing import Any, Dict, List, Optional

from celery import Task as CeleryTask

from src.adapters.factory import LLMFactory
from src.core.schemas.model import Model
from src.core.schemas.task import (
    ABTestConfig,
    ABTestStrategy,
    Task,
    TaskResult,
    TaskStatus,
)
from src.core.services.ab_test_analyzer import ABTestAnalyzer
from src.core.services.eval_service import EvaluationService
from src.core.services.include_exclude_evaluator import IncludeExcludeEvaluator
from src.core.services.judge_service import LLMJudgeService
from src.core.services.rta_evaluator import RTAEvaluator
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
        if task.config.variations.enabled and task.config.variations.model_id:
            variation_generator = PromptVariationGenerator(
                task.config.variations.model_id
            )
            await variation_generator.initialize()
            logger.info("Variation generator initialized")

        # Инициализируем LLM judge (если нужно)
        judge_service = None
        if task.config.judge.enabled and task.config.judge.model_id:
            judge_service = LLMJudgeService(task.config.judge_model_id)
            await judge_service.initialize()
            logger.info("LLM Judge initialized")

        # После инициализации judge_service
        rta_evaluator = None
        if task.config.rta.enabled and task.config.rta.rta_judge_model_id:
            rta_evaluator = RTAEvaluator(
                task.config.rta.rta_judge_model_id,
                task.config.rta.rta_prompt_template,
                # task.config.rta.refusal_keywords,
            )
            await rta_evaluator.initialize()

        ab_variants = None
        if task.config.ab_test.enabled:
            ab_variants = _prepare_ab_test_variants(task.config.ab_test, models)
            logger.info(f"A/B test enabled with {len(ab_variants)} variants")

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
                        variation_prompt=task.config.variations.custom_prompt,
                        strategies=task.config.variations.strategies,
                        count_per_strategy=task.config.variations.count_per_strategy,
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

                    import json

                    with open(
                        "var.json", "w", encoding="utf-8"
                    ) as f:  # TODO: delete this strings
                        json.dump(variations, f, ensure_ascii=False, indent=4)

                    with open("prompts.json", "w", encoding="utf-8") as f:
                        json.dump(prompts_to_process, f, ensure_ascii=False, indent=4)

                logger.info(f"Start inference for item. {prompts_to_process}\n")
                if ab_variants:
                    # Режим A/B тестирования
                    results = await _process_with_ab_test(
                        item=item,
                        prompts_to_process=prompts_to_process,
                        adapters=adapters,
                        ab_variants=ab_variants,
                        task=task,
                        judge_service=judge_service,
                        rta_evaluator=rta_evaluator,
                    )
                else:
                    # Обычный режим
                    results = await _process_standard(
                        item=item,
                        prompts_to_process=prompts_to_process,
                        models=models,
                        adapters=adapters,
                        task=task,
                        judge_service=judge_service,
                        rta_evaluator=rta_evaluator,
                    )

                all_results.extend(results)

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

        # После основного цикла - Include/Exclude
        if dataset.include_column or dataset.exclude_column:
            ie_metrics = IncludeExcludeEvaluator.evaluate_results(all_results)
            aggregated_metrics["include_exclude"] = ie_metrics

        # Если A/B тест
        if task.config.ab_test.enabled and all_results:
            logger.info(f"Running A/B analysis on {len(all_results)} results")

            # Проверяем, что есть маркированные результаты
            marked_results = [r for r in all_results if r.ab_variant is not None]
            logger.info(f"Found {len(marked_results)} results with A/B variant markers")

            if len(marked_results) >= 2:
                ab_results = ABTestAnalyzer.analyze_ab_test(
                    marked_results,
                    task.config.evaluation_metrics,
                    task.config.ab_test.statistical_test,
                )
                await task_repo.update(task_id, {"ab_test_results": ab_results})
                logger.info(f"A/B test completed: {ab_results.get('winner', {})}")
            else:
                logger.warning(
                    f"Not enough marked results for A/B test: {len(marked_results)}"
                )

        # Сохраняем результаты

        logger.info(f"Agg metrics: {aggregated_metrics}")
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


def _prepare_ab_test_variants(
    ab_config: ABTestConfig, models: List[Any]
) -> List[Dict[str, Any]]:
    """Подготовка вариантов для A/B тестирования"""

    variants = []

    if ab_config.strategy == ABTestStrategy.PROMPT_VARIANTS:
        # Разные промпты
        for variant_name, prompt_template in ab_config.prompt_variants.items():
            for model in models:
                variants.append(
                    {
                        "name": f"{variant_name}_{model.id}",
                        "model_id": model.id,
                        "model_name": model.name,
                        "prompt_template": prompt_template,
                        "system_prompt": None,
                        "temperature": None,
                    }
                )

    elif ab_config.strategy == ABTestStrategy.MODEL_COMPARISON:
        # Сравнение моделей на одинаковых данных
        for model in models:
            variants.append(
                {
                    "name": f"model_{model.id}",
                    "model_id": model.id,
                    "model_name": model.name,
                    "prompt_template": None,
                    "system_prompt": None,
                    "temperature": None,
                }
            )

    elif ab_config.strategy == ABTestStrategy.TEMPERATURE_TEST:
        # Разные температуры
        for temp in ab_config.temperatures:
            for model in models:
                variants.append(
                    {
                        "name": f"temp_{temp}_{model.id}",
                        "model_id": model.id,
                        "model_name": model.name,
                        "prompt_template": None,
                        "system_prompt": None,
                        "temperature": temp,
                    }
                )

    # elif ab_config.strategy == ABTestStrategy.SYSTEM_PROMPT_TEST:
    #     # Разные системные промпты
    #     for variant_name, sys_prompt in ab_config.system_prompts.items():
    #         for model in models:
    #             variants.append(
    #                 {
    #                     "name": f"{variant_name}_{model.id}",
    #                     "model_id": model.id,
    #                     "model_name": model.name,
    #                     "prompt_template": None,
    #                     "system_prompt": sys_prompt,
    #                     "temperature": None,
    #                 }
    #             )

    # elif ab_config.strategy == ABTestStrategy.PARAMETER_SWEEP:
    #     # Перебор параметров
    #     import itertools

    #     param_combinations = list(
    #         itertools.product(*ab_config.parameter_ranges.values())
    #     )
    #     param_names = list(ab_config.parameter_ranges.keys())

    #     for combo in param_combinations:
    #         params = dict(zip(param_names, combo))
    #         for model in models:
    #             variant_name = "_".join([f"{k}_{v}" for k, v in params.items()])
    #             variants.append(
    #                 {
    #                     "name": f"{variant_name}_{model.id}",
    #                     "model_id": model.id,
    #                     "model_name": model.name,
    #                     "prompt_template": None,
    #                     "system_prompt": None,
    #                     "parameters": params,
    #                 }
    #             )

    return variants


async def _process_with_ab_test(
    item: Any,
    prompts_to_process: List[Dict],
    adapters: Dict[str, Model],
    ab_variants: List[Dict[str, Any]],
    task: Task,
    judge_service: Optional[Any],
    rta_evaluator: Optional[Any],
) -> List[TaskResult]:
    """Обработка элемента в режиме A/B теста"""

    results = []

    # Определяем, сколько семплов на вариант
    samples_per_variant = task.config.ab_test.sample_size_per_variant or 1

    for prompt_data in prompts_to_process:
        for variant in ab_variants:
            # Ограничиваем количество запросов на вариант
            for sample_idx in range(samples_per_variant):
                try:
                    adapter = adapters[variant["model_id"]]

                    # Формируем промпт для варианта
                    final_prompt = prompt_data["text"]
                    if variant.get("prompt_template"):
                        final_prompt = variant["prompt_template"].format(
                            input=prompt_data["text"]
                        )

                    # Параметры генерации
                    gen_params = {}
                    if variant.get("temperature") is not None:
                        gen_params["temperature"] = variant["temperature"]
                    if variant.get("system_prompt"):
                        gen_params["system_prompt"] = variant["system_prompt"]
                    if variant.get("parameters"):
                        gen_params.update(variant["parameters"])

                    start_time = time.time()

                    # Генерация
                    response = await adapter.generate(final_prompt, **gen_params)

                    execution_time = time.time() - start_time

                    # Создаем результат с маркировкой варианта
                    result = TaskResult(
                        input=final_prompt,
                        output=response,
                        model_id=variant["model_id"],
                        target=item.target,
                        execution_time=execution_time,
                        metadata={
                            **item.metadata,
                            "model_name": variant["model_name"],
                            "ab_variant_config": variant,
                            "sample_index": sample_idx,
                        },
                        metrics=task.config.evaluation_metrics,
                        original_input=prompt_data["original"],
                        variation_type=prompt_data["variation_type"],
                        ab_variant=variant["name"],
                    )

                    # Judge оценка
                    if judge_service:
                        judge_result = await judge_service.evaluate_output(
                            input_prompt=final_prompt,
                            model_output=response,
                            task_description=task.name,
                            reference_output=item.target,
                            criteria=task.config.judge.criteria,
                        )
                        result.judge_score = judge_result["overall_score"]
                        result.judge_reasoning = judge_result["reasoning"]
                        result.metadata["judge_criteria_scores"] = judge_result[
                            "criteria_scores"
                        ]

                    # RTA оценка
                    if rta_evaluator:
                        rta_result = await rta_evaluator.evaluate_output(
                            input_prompt=final_prompt,
                            model_output=response,
                        )
                        result.refused = rta_result["refused"]
                        result.metadata["rta_reasoning"] = rta_result["reasoning"]

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error in A/B variant {variant['name']}: {e}")
                    continue

    return results


async def _process_standard(
    item: Any,
    prompts_to_process: List[Dict],
    models: List[Any],
    adapters: Dict[str, Any],
    task: Task,
    judge_service: Optional[Any],
    rta_evaluator: Optional[Any],
) -> List[TaskResult]:
    """Стандартная обработка без A/B теста"""

    results = []

    for prompt_data in prompts_to_process:
        for model in models:
            adapter = adapters[model.id]

            try:
                start_time = time.time()
                response = await adapter.generate(prompt_data["text"])
                execution_time = time.time() - start_time

                result = TaskResult(
                    input=prompt_data["text"],
                    output=response,
                    model_id=model.id,
                    target=item.target,
                    execution_time=execution_time,
                    metrics=task.config.evaluation_metrics,
                    metadata={
                        **item.metadata,
                        "model_name": model.name,
                        "model_provider": model.provider,
                    },
                    original_input=prompt_data["original"],
                    variation_type=prompt_data["variation_type"],
                )

                # Judge оценка
                if judge_service:
                    judge_result = await judge_service.evaluate_output(
                        input_prompt=prompt_data["text"],
                        model_output=response,
                        task_description=task.name,
                        reference_output=item.target,
                        criteria=task.config.judge.criteria,
                    )
                    result.judge_score = judge_result["overall_score"]
                    result.judge_reasoning = judge_result["reasoning"]
                    result.metadata["judge_criteria_scores"] = judge_result[
                        "criteria_scores"
                    ]

                # RTA оценка
                if rta_evaluator:
                    rta_result = await rta_evaluator.evaluate_output(
                        input_prompt=prompt_data["text"],
                        model_output=response,
                    )
                    result.refused = rta_result["refused"]
                    result.metadata["rta_reasoning"] = rta_result["reasoning"]
                    result.metadata["rta_confidence"] = rta_result["confidence"]
                    result.metadata["raw_response"] = rta_result["raw_response"]

                logger.info(f"Success inference: {result}")
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing with model {model.name}: {e}")
                continue

    return results
