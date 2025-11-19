# src/core/tasks/inference_task.py
import asyncio
import time
from datetime import datetime
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

    return variants


def get_model_name(model_id: str, models: List) -> str:
    """Вспомогательная функция"""
    model = next((m for m in models if m.id == model_id), None)
    return model.name if model else model_id[:12]


def calculate_total_inferences(
    total_items: int,
    num_models: int,
    variations_enabled: bool,
    variations_per_prompt: int,
    num_variation_strategies: int,
    ab_test_enabled: bool,
    ab_variants_count: int,
) -> int:
    """
    Вычисляет общее количество инференсов с учетом вариаций, моделей и AB тестов

    Returns:
        int: Ожидаемое количество инференсов
    """
    if ab_test_enabled:
        # В AB тесте количество определяется вариантами
        return total_items * ab_variants_count

    # Базовое количество: items * models
    base_count = total_items * num_models

    if variations_enabled:
        # Для каждого элемента:
        # - 1 оригинальный промпт * num_models
        # - (variations_per_prompt * num_variation_strategies) вариаций * num_models
        variations_count = variations_per_prompt * num_variation_strategies
        total_per_item = (1 + variations_count) * num_models
        return total_items * total_per_item

    return base_count


async def _recover_task_state(task_id: str, task_repo: TaskRepository) -> Dict:
    """
    Восстановление состояния задачи при возобновлении

    Проверяет целостность данных и восстанавливает если нужно

    Args:
        task_id: ID задачи
        task_repo: Репозиторий задач

    Returns:
        Dict с информацией о восстановлении
    """
    task = await task_repo.find_by_id(task_id)

    if not task:
        raise ValueError(f"Task {task_id} not found")

    recovery_info = {
        "recovered_results": 0,
        "last_index": task.last_processed_index,
        "status": "ok",
        "warnings": [],
    }

    # Проверяем consistency
    if task.results:
        recovery_info["recovered_results"] = len(task.results)

        # Валидация 1: processed_samples должен совпадать с len(results)
        if task.processed_samples != len(task.results):
            logger.warning(
                f"Mismatch detected: processed_samples={task.processed_samples}, "
                f"results count={len(task.results)}. Fixing..."
            )

            await task_repo.update(task_id, {"processed_samples": len(task.results)})

            recovery_info["status"] = "fixed_mismatch"
            recovery_info["warnings"].append(
                f"Fixed processed_samples mismatch: {task.processed_samples} -> {len(task.results)}"
            )

        # Валидация 2: Проверяем дубликаты (по input + model_id)
        seen = set()
        duplicates = 0
        unique_results = []

        for result in task.results:
            key = (result.input, result.model_id, result.variation_type or "")
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
            else:
                duplicates += 1

        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate results. Removing...")

            await task_repo.update(
                task_id,
                {
                    "results": [r.model_dump() for r in unique_results],
                    "processed_samples": len(unique_results),
                },
            )

            recovery_info["status"] = "fixed_duplicates"
            recovery_info["warnings"].append(f"Removed {duplicates} duplicate results")
            recovery_info["recovered_results"] = len(unique_results)

        # Валидация 3: Проверяем что last_processed_index не больше total_samples
        if task.last_processed_index > task.total_samples:
            logger.warning(
                f"Invalid last_processed_index: {task.last_processed_index} > {task.total_samples}. "
                f"Resetting to results count..."
            )

            # Вычисляем правильный индекс на основе результатов
            # Примерная оценка: количество обработанных элементов датасета
            estimated_index = (
                len(unique_results) // len(task.model_ids) if task.model_ids else 0
            )

            await task_repo.update(task_id, {"last_processed_index": estimated_index})

            recovery_info["last_index"] = estimated_index
            recovery_info["warnings"].append(
                f"Reset last_processed_index: {task.last_processed_index} -> {estimated_index}"
            )
    else:
        logger.info(f"No previous results found for task {task_id}, starting fresh")

    # Очищаем current_execution (если осталось от прошлого запуска)
    if task.current_execution:
        await task_repo.update(task_id, {"current_execution": None})
        recovery_info["warnings"].append("Cleared stale current_execution")

    logger.info(f"Task state recovered: {recovery_info}")

    return recovery_info


# async def _process_standard(
#     item: Any,
#     prompts_to_process: List[Dict],
#     models: List[Any],
#     adapters: Dict[str, Any],
#     task: Task,
#     judge_service: Optional[Any],
#     rta_evaluator: Optional[Any],
#     task_id: str,  # НОВЫЙ параметр
#     task_repo: Any,  # НОВЫЙ параметр
#     item_index: int,  # НОВЫЙ параметр
# ) -> List[TaskResult]:
#     """Стандартная обработка с детальным отслеживанием"""

#     results = []

#     for prompt_data in prompts_to_process:
#         for model in models:
#             adapter = adapters[model.id]

#             # Определяем тип операции
#             operation_type = "standard"
#             if prompt_data["variation_type"]:
#                 operation_type = "variation"

#             try:
#                 # ===== ШАГ 1: ИНФЕРЕНС =====
#                 # Обновляем current_execution перед генерацией
#                 current_execution = {
#                     "index": item_index,
#                     "prompt": prompt_data["text"][:100],
#                     "prompt_variation": prompt_data["variation_type"] or "original",
#                     "model_id": model.id,
#                     "model_name": model.name,
#                     "operation_type": operation_type,
#                     "started_at": datetime.utcnow().isoformat(),
#                     "output": None,
#                 }

#                 await task_repo.update(
#                     task_id, {"current_execution": current_execution}
#                 )

#                 # Генерация
#                 start_time = time.time()
#                 response = await adapter.generate(prompt_data["text"])
#                 execution_time = time.time() - start_time

#                 # Обновляем с результатом
#                 current_execution["output"] = response[:100]
#                 current_execution["execution_time"] = execution_time
#                 await task_repo.update(
#                     task_id, {"current_execution": current_execution}
#                 )

#                 # Создаем результат
#                 result = TaskResult(
#                     input=prompt_data["text"],
#                     output=response,
#                     model_id=model.id,
#                     target=item.target,
#                     execution_time=execution_time,
#                     metrics=task.config.evaluation_metrics,
#                     metadata={
#                         **item.metadata,
#                         "model_name": model.name,
#                         "model_provider": model.provider,
#                         "operation_type": operation_type,
#                     },
#                     original_input=prompt_data["original"],
#                     variation_type=prompt_data["variation_type"],
#                 )

#                 # ===== ШАГ 2: JUDGE ОЦЕНКА (если включено) =====
#                 if judge_service:
#                     # Обновляем тип операции
#                     current_execution["operation_type"] = "judge_evaluation"
#                     current_execution["output"] = None  # Очищаем для judge
#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                     judge_result = await judge_service.evaluate_output(
#                         input_prompt=prompt_data["text"],
#                         model_output=response,
#                         task_description=task.name,
#                         reference_output=item.target,
#                         criteria=task.config.judge.criteria,
#                     )

#                     result.judge_score = judge_result["overall_score"]
#                     result.judge_reasoning = judge_result["reasoning"]
#                     result.metadata["judge_criteria_scores"] = judge_result[
#                         "criteria_scores"
#                     ]

#                     # Обновляем с результатом judge
#                     current_execution["output"] = (
#                         f"Score: {judge_result['overall_score']:.2f}"
#                     )
#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                 # ===== ШАГ 3: RTA ОЦЕНКА (если включено) =====
#                 if rta_evaluator:
#                     # Обновляем тип операции
#                     current_execution["operation_type"] = "rta_evaluation"
#                     current_execution["output"] = None
#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                     rta_result = await rta_evaluator.evaluate_output(
#                         input_prompt=prompt_data["text"],
#                         model_output=response,
#                     )

#                     result.refused = rta_result["refused"]
#                     result.metadata["rta_reasoning"] = rta_result["reasoning"]
#                     result.metadata["rta_confidence"] = rta_result["confidence"]

#                     # Обновляем с результатом RTA
#                     refused_text = (
#                         "REFUSED" if rta_result["refused"] == "1" else "ANSWERED"
#                     )
#                     current_execution["output"] = f"RTA: {refused_text}"
#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                 # ===== ШАГ 4: СОХРАНЕНИЕ В RECENT_EXECUTIONS =====
#                 execution_summary = {
#                     "index": item_index,
#                     "prompt": prompt_data["text"][:100],
#                     "prompt_variation": prompt_data["variation_type"] or "original",
#                     "output": response[:100],
#                     "model_id": model.id,
#                     "model_name": model.name,
#                     "operation_type": operation_type,
#                     "execution_time": execution_time,
#                     "completed_at": datetime.utcnow().isoformat(),
#                 }

#                 # Добавляем judge/RTA инфо если есть
#                 if result.judge_score:
#                     execution_summary["judge_score"] = result.judge_score
#                 if result.refused:
#                     execution_summary["refused"] = result.refused

#                 # Получаем текущую задачу для recent_executions
#                 current_task = await task_repo.find_by_id(task_id)
#                 recent = current_task.recent_executions or []
#                 recent.insert(0, execution_summary)
#                 recent = recent[:2]

#                 # Обновляем recent и очищаем current
#                 await task_repo.update(
#                     task_id,
#                     {
#                         "recent_executions": recent,
#                         "current_execution": None,
#                     },
#                 )

#                 logger.info(f"Successfully processed: {model.name} - {operation_type}")
#                 results.append(result)

#             except Exception as e:
#                 logger.error(f"Error processing with model {model.name}: {e}")

#                 # Сохраняем ошибку в recent_executions
#                 error_summary = {
#                     "index": item_index,
#                     "prompt": prompt_data["text"][:100],
#                     "prompt_variation": prompt_data["variation_type"] or "original",
#                     "model_name": model.name,
#                     "operation_type": operation_type,
#                     "error": str(e)[:200],
#                     "completed_at": datetime.utcnow().isoformat(),
#                 }

#                 current_task = await task_repo.find_by_id(task_id)
#                 recent = current_task.recent_executions or []
#                 recent.insert(0, error_summary)
#                 recent = recent[:2]

#                 await task_repo.update(
#                     task_id, {"recent_executions": recent, "current_execution": None}
#                 )

#                 continue

#     return results


# async def _process_with_ab_test(
#     item: Any,
#     prompts_to_process: List[Dict],
#     adapters: Dict[str, Model],
#     ab_variants: List[Dict[str, Any]],
#     task: Task,
#     judge_service: Optional[Any],
#     rta_evaluator: Optional[Any],
#     task_id: str,  # НОВЫЙ параметр
#     task_repo: Any,  # НОВЫЙ параметр
#     item_index: int,  # НОВЫЙ параметр
# ) -> List[TaskResult]:
#     """Обработка элемента в режиме A/B теста с детальным отслеживанием"""

#     results = []

#     # Определяем количество семплов на вариант
#     samples_per_variant = task.config.ab_test.sample_size_per_variant or 1

#     for prompt_data in prompts_to_process:
#         for variant in ab_variants:
#             # Ограничиваем количество запросов на вариант
#             for sample_idx in range(samples_per_variant):
#                 try:
#                     adapter = adapters[variant["model_id"]]

#                     # Формируем промпт для варианта
#                     final_prompt = prompt_data["text"]
#                     if variant.get("prompt_template"):
#                         final_prompt = variant["prompt_template"].format(
#                             input=prompt_data["text"]
#                         )

#                     # Определяем тип операции
#                     operation_type = "ab_test"
#                     if prompt_data["variation_type"]:
#                         operation_type = "ab_test_variation"

#                     # Параметры генерации
#                     gen_params = {}
#                     if variant.get("temperature") is not None:
#                         gen_params["temperature"] = variant["temperature"]
#                     if variant.get("system_prompt"):
#                         gen_params["system_prompt"] = variant["system_prompt"]
#                     if variant.get("parameters"):
#                         gen_params.update(variant["parameters"])

#                     # ===== ШАГ 1: ИНФЕРЕНС =====
#                     # Обновляем current_execution
#                     current_execution = {
#                         "index": item_index,
#                         "prompt": final_prompt[:100],
#                         "prompt_variation": prompt_data["variation_type"] or "original",
#                         "model_id": variant["model_id"],
#                         "model_name": variant["model_name"],
#                         "operation_type": operation_type,
#                         "ab_variant": variant["name"],
#                         "ab_sample_index": sample_idx,
#                         "started_at": datetime.utcnow().isoformat(),
#                         "output": None,
#                     }

#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                     # Генерация
#                     start_time = time.time()
#                     response = await adapter.generate(final_prompt, **gen_params)
#                     execution_time = time.time() - start_time

#                     # Обновляем с результатом
#                     current_execution["output"] = response[:100]
#                     current_execution["execution_time"] = execution_time
#                     await task_repo.update(
#                         task_id, {"current_execution": current_execution}
#                     )

#                     # Создаем результат с маркировкой варианта
#                     result = TaskResult(
#                         input=final_prompt,
#                         output=response,
#                         model_id=variant["model_id"],
#                         target=item.target,
#                         execution_time=execution_time,
#                         metadata={
#                             **item.metadata,
#                             "model_name": variant["model_name"],
#                             "ab_variant_config": variant,
#                             "sample_index": sample_idx,
#                             "operation_type": operation_type,
#                         },
#                         metrics=task.config.evaluation_metrics,
#                         original_input=prompt_data["original"],
#                         variation_type=prompt_data["variation_type"],
#                         ab_variant=variant["name"],
#                     )

#                     # ===== ШАГ 2: JUDGE ОЦЕНКА =====
#                     if judge_service:
#                         current_execution["operation_type"] = "ab_judge_evaluation"
#                         current_execution["output"] = None
#                         await task_repo.update(
#                             task_id, {"current_execution": current_execution}
#                         )

#                         judge_result = await judge_service.evaluate_output(
#                             input_prompt=final_prompt,
#                             model_output=response,
#                             task_description=task.name,
#                             reference_output=item.target,
#                             criteria=task.config.judge.criteria,
#                         )

#                         result.judge_score = judge_result["overall_score"]
#                         result.judge_reasoning = judge_result["reasoning"]
#                         result.metadata["judge_criteria_scores"] = judge_result[
#                             "criteria_scores"
#                         ]

#                         current_execution["output"] = (
#                             f"Score: {judge_result['overall_score']:.2f}"
#                         )
#                         await task_repo.update(
#                             task_id, {"current_execution": current_execution}
#                         )

#                     # ===== ШАГ 3: RTA ОЦЕНКА =====
#                     if rta_evaluator:
#                         current_execution["operation_type"] = "ab_rta_evaluation"
#                         current_execution["output"] = None
#                         await task_repo.update(
#                             task_id, {"current_execution": current_execution}
#                         )

#                         rta_result = await rta_evaluator.evaluate_output(
#                             input_prompt=final_prompt,
#                             model_output=response,
#                         )

#                         result.refused = rta_result["refused"]
#                         result.metadata["rta_reasoning"] = rta_result["reasoning"]

#                         refused_text = (
#                             "REFUSED" if rta_result["refused"] == "1" else "ANSWERED"
#                         )
#                         current_execution["output"] = f"RTA: {refused_text}"
#                         await task_repo.update(
#                             task_id, {"current_execution": current_execution}
#                         )

#                     # ===== ШАГ 4: СОХРАНЕНИЕ В RECENT_EXECUTIONS =====
#                     execution_summary = {
#                         "index": item_index,
#                         "prompt": final_prompt[:100],
#                         "prompt_variation": prompt_data["variation_type"] or "original",
#                         "output": response[:100],
#                         "model_id": variant["model_id"],
#                         "model_name": variant["model_name"],
#                         "operation_type": operation_type,
#                         "ab_variant": variant["name"],
#                         "execution_time": execution_time,
#                         "completed_at": datetime.utcnow().isoformat(),
#                     }

#                     if result.judge_score:
#                         execution_summary["judge_score"] = result.judge_score
#                     if result.refused:
#                         execution_summary["refused"] = result.refused

#                     current_task = await task_repo.find_by_id(task_id)
#                     recent = current_task.recent_executions or []
#                     recent.insert(0, execution_summary)
#                     recent = recent[:2]

#                     await task_repo.update(
#                         task_id,
#                         {
#                             "recent_executions": recent,
#                             "current_execution": None,
#                         },
#                     )

#                     logger.info(
#                         f"AB test processed: {variant['name']} - sample {sample_idx}"
#                     )
#                     results.append(result)

#                 except Exception as e:
#                     logger.error(f"Error in A/B variant {variant['name']}: {e}")

#                     # Сохраняем ошибку
#                     error_summary = {
#                         "index": item_index,
#                         "prompt": final_prompt[:100]
#                         if "final_prompt" in locals()
#                         else "N/A",
#                         "model_name": variant.get("model_name", "N/A"),
#                         "operation_type": operation_type,
#                         "ab_variant": variant["name"],
#                         "error": str(e)[:200],
#                         "completed_at": datetime.utcnow().isoformat(),
#                     }

#                     current_task = await task_repo.find_by_id(task_id)
#                     recent = current_task.recent_executions or []
#                     recent.insert(0, error_summary)
#                     recent = recent[:2]

#                     await task_repo.update(
#                         task_id,
#                         {"recent_executions": recent, "current_execution": None},
#                     )

#                     continue

#     return results


# async def _run_inference_async(celery_task, task_id: str):
#     """Асинхронное выполнение инференса с правильной интеграцией"""

#     # Репозитории и сервисы
#     task_repo = TaskRepository()
#     dataset_repo = DatasetRepository()
#     model_repo = ModelRepository()
#     evaluation_service = EvaluationService()

#     # Получаем задачу
#     task = await task_repo.find_by_id(task_id)
#     if not task:
#         raise ValueError(f"Task {task_id} not found")

#     logger.info(f"Starting inference for task {task_id}: {task.name}")
#     await task_repo.update_status(task_id, TaskStatus.RUNNING)

#     try:
#         # Загружаем датасет
#         dataset = await dataset_repo.find_by_id(task.dataset_id)
#         if not dataset:
#             raise ValueError(f"Dataset {task.dataset_id} not found")

#         # Загружаем модели
#         models = []
#         adapters = {}
#         for model_id in task.model_ids:
#             model = await model_repo.find_by_id(model_id)
#             if not model:
#                 logger.warning(f"Model {model_id} not found, skipping")
#                 continue
#             models.append(model)
#             adapters[model_id] = LLMFactory.create(model)

#         if not models:
#             raise ValueError("No valid models found for task")

#         logger.info(f"Loaded {len(models)} models for inference")

#         # Инициализация генератора вариаций
#         variation_generator = None
#         if task.config.variations.enabled and task.config.variations.model_id:
#             variation_generator = PromptVariationGenerator(
#                 task.config.variations.model_id
#             )
#             await variation_generator.initialize()
#             logger.info("Variation generator initialized")

#         # Инициализация LLM judge
#         judge_service = None
#         if task.config.judge.enabled and task.config.judge.model_id:
#             judge_service = LLMJudgeService(task.config.judge.model_id)
#             await judge_service.initialize()
#             logger.info("LLM Judge initialized")

#         # Инициализация RTA evaluator
#         rta_evaluator = None
#         if task.config.rta.enabled and task.config.rta.rta_judge_model_id:
#             rta_evaluator = RTAEvaluator(
#                 task.config.rta.rta_judge_model_id,
#                 task.config.rta.rta_prompt_template,
#             )
#             await rta_evaluator.initialize()
#             logger.info("RTA Evaluator initialized")

#         # Подготовка AB тестов
#         ab_variants = None
#         if task.config.ab_test.enabled:
#             ab_variants = _prepare_ab_test_variants(task.config.ab_test, models)
#             logger.info(f"A/B test enabled with {len(ab_variants)} variants")

#         # Получаем элементы датасета
#         max_samples = task.config.max_samples or dataset.size
#         items = await dataset_repo.get_items(task.dataset_id, limit=max_samples)
#         total_items = len(items)

#         # Вычисляем ожидаемое количество инференсов
#         expected_total_inferences = calculate_total_inferences(
#             total_items=total_items,
#             num_models=len(models),
#             variations_enabled=task.config.variations.enabled,
#             variations_per_prompt=task.config.variations.count_per_strategy
#             if task.config.variations.enabled
#             else 0,
#             num_variation_strategies=len(task.config.variations.strategies)
#             if task.config.variations.enabled
#             else 0,
#             ab_test_enabled=task.config.ab_test.enabled,
#             ab_variants_count=len(ab_variants) if ab_variants else 0,
#         )

#         # Обновляем total_samples
#         await task_repo.update(task_id, {"total_samples": expected_total_inferences})

#         logger.info(f"Expected total inferences: {expected_total_inferences}")
#         logger.info(f"Processing {total_items} dataset items")

#         # ВАЖНО: Загружаем предыдущие результаты (для resume)
#         all_results: List[TaskResult] = []
#         if task.results:
#             all_results = task.results.copy()
#             logger.info(f"Loaded {len(all_results)} previous results")

#         # Начинаем с последнего обработанного индекса
#         start_index = task.last_processed_index
#         batch_size = task.config.batch_size

#         # Настройки сохранения
#         SAVE_INTERVAL = 5
#         results_since_last_save = 0

#         # Обрабатываем элементы
#         for i in range(start_index, total_items, batch_size):
#             # Проверяем статус задачи
#             current_task = await task_repo.find_by_id(task_id)

#             if current_task.status == TaskStatus.PAUSED:
#                 logger.info(f"Task {task_id} paused at index {i}")

#                 # Сохраняем ВСЁ перед паузой
#                 await task_repo.save_intermediate_results(
#                     task_id=task_id,
#                     results=all_results,
#                     processed_count=len(all_results),
#                     last_index=i,
#                 )

#                 logger.info(f"Saved {len(all_results)} results before pause")

#                 return {
#                     "task_id": task_id,
#                     "status": "paused",
#                     "last_processed_index": i,
#                     "saved_results": len(all_results),
#                 }

#             if current_task.status == TaskStatus.CANCELLED:
#                 logger.info(f"Task {task_id} cancelled at index {i}")

#                 # Сохраняем перед отменой
#                 await task_repo.save_intermediate_results(
#                     task_id=task_id,
#                     results=all_results,
#                     processed_count=len(all_results),
#                     last_index=i,
#                 )

#                 logger.info(f"Saved {len(all_results)} results before cancellation")

#                 return {
#                     "task_id": task_id,
#                     "status": "cancelled",
#                     "saved_results": len(all_results),
#                 }

#             batch = items[i : i + batch_size]

#             for item in batch:
#                 prompts_to_process = []

#                 # Основной промпт
#                 prompts_to_process.append(
#                     {
#                         "text": item.prompt,
#                         "original": item.prompt,
#                         "variation_type": None,
#                     }
#                 )

#                 # Генерируем вариации
#                 if variation_generator:
#                     variations = await variation_generator.generate_variations(
#                         prompt=item.prompt,
#                         variation_prompt=task.config.variations.custom_prompt,
#                         strategies=task.config.variations.strategies,
#                         count_per_strategy=task.config.variations.count_per_strategy,
#                     )

#                     for var in variations:
#                         prompts_to_process.append(
#                             {
#                                 "text": var["text"],
#                                 "original": item.prompt,
#                                 "variation_type": var["strategy"],
#                             }
#                         )

#                     logger.info(f"Generated {len(variations)} variations for item {i}")

#                 # ВЫЗЫВАЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ с нужными параметрами
#                 if ab_variants:
#                     # Режим A/B тестирования
#                     results = await _process_with_ab_test(
#                         item=item,
#                         prompts_to_process=prompts_to_process,
#                         adapters=adapters,
#                         ab_variants=ab_variants,
#                         task=current_task,
#                         judge_service=judge_service,
#                         rta_evaluator=rta_evaluator,
#                         task_id=task_id,  # НОВЫЙ параметр
#                         task_repo=task_repo,  # НОВЫЙ параметр
#                         item_index=i,  # НОВЫЙ параметр
#                     )
#                 else:
#                     # Обычный режим
#                     results = await _process_standard(
#                         item=item,
#                         prompts_to_process=prompts_to_process,
#                         models=models,
#                         adapters=adapters,
#                         task=current_task,
#                         judge_service=judge_service,
#                         rta_evaluator=rta_evaluator,
#                         task_id=task_id,  # НОВЫЙ параметр
#                         task_repo=task_repo,  # НОВЫЙ параметр
#                         item_index=i,  # НОВЫЙ параметр
#                     )

#                 all_results.extend(results)
#                 results_since_last_save += len(results)

#                 # ПЕРИОДИЧЕСКОЕ СОХРАНЕНИЕ
#                 if results_since_last_save >= SAVE_INTERVAL:
#                     logger.info(
#                         f"Saving intermediate results: {len(all_results)} total"
#                     )

#                     await task_repo.save_intermediate_results(
#                         task_id=task_id,
#                         results=all_results,
#                         processed_count=len(all_results),
#                         last_index=i,
#                     )

#                     results_since_last_save = 0

#             # Обновляем прогресс после батча
#             processed = len(all_results)
#             progress = (
#                 (processed / expected_total_inferences * 100)
#                 if expected_total_inferences > 0
#                 else 0
#             )

#             await task_repo.update_progress(task_id, progress, processed)
#             await task_repo.update(
#                 task_id, {"last_processed_index": min(i + batch_size, total_items)}
#             )

#             # Сохраняем после каждого батча
#             if results_since_last_save > 0:
#                 await task_repo.save_intermediate_results(
#                     task_id=task_id,
#                     results=all_results,
#                     processed_count=len(all_results),
#                     last_index=min(i + batch_size, total_items),
#                 )
#                 results_since_last_save = 0

#             celery_task.update_state(
#                 state="PROGRESS",
#                 meta={
#                     "progress": progress,
#                     "processed": processed,
#                     "total": expected_total_inferences,
#                 },
#             )

#             logger.info(
#                 f"Task {task_id}: {processed}/{expected_total_inferences} ({progress:.1f}%)"
#             )

#         # Классическая оценка метрик
#         aggregated_metrics = {}
#         if task.config.evaluate and task.config.evaluation_metrics:
#             logger.info("Evaluating results")

#             # Группируем результаты по моделям
#             results_by_model = {}
#             for result in all_results:
#                 if result.model_id not in results_by_model:
#                     results_by_model[result.model_id] = []
#                 results_by_model[result.model_id].append(result)

#             # Оцениваем каждую модель отдельно
#             for model_id, model_results in results_by_model.items():
#                 model_metrics = evaluation_service.evaluate_results(
#                     model_results, task.config.evaluation_metrics
#                 )
#                 aggregated_metrics[model_id] = model_metrics

#         # Include/Exclude оценка
#         if dataset.include_column or dataset.exclude_column:
#             ie_metrics = IncludeExcludeEvaluator.evaluate_results(all_results)
#             aggregated_metrics["include_exclude"] = ie_metrics

#         # A/B тест анализ
#         if task.config.ab_test.enabled and all_results:
#             logger.info(f"Running A/B analysis on {len(all_results)} results")

#             marked_results = [r for r in all_results if r.ab_variant is not None]
#             logger.info(f"Found {len(marked_results)} results with A/B variant markers")

#             if len(marked_results) >= 2:
#                 ab_results = ABTestAnalyzer.analyze_ab_test(
#                     marked_results,
#                     task.config.evaluation_metrics,
#                     task.config.ab_test.statistical_test,
#                 )
#                 await task_repo.update(task_id, {"ab_test_results": ab_results})
#                 logger.info(f"A/B test completed: {ab_results.get('winner', {})}")

#         # ФИНАЛЬНОЕ сохранение
#         logger.info(f"Saving final results: {len(all_results)} total")
#         await task_repo.set_results(task_id, all_results, aggregated_metrics)

#         # Завершаем задачу
#         await task_repo.update_status(task_id, TaskStatus.COMPLETED)

#         logger.info(f"Task {task_id} completed successfully")

#         return {
#             "task_id": task_id,
#             "status": "completed",
#             "total_samples": expected_total_inferences,
#             "total_results": len(all_results),
#             "models_tested": len(models),
#         }

#     except Exception as e:
#         logger.error(f"Error in task {task_id}: {e}", exc_info=True)
#         await task_repo.update_status(task_id, TaskStatus.FAILED, error=str(e))
#         raise

# src/core/tasks/inference_task.py - Исправленные функции обработки


async def _process_standard(
    item: Any,
    prompts_to_process: List[Dict],
    models: List[Any],
    adapters: Dict[str, Any],
    task: Task,
    judge_service: Optional[Any],
    rta_evaluator: Optional[Any],
    task_id: str,  # НОВЫЙ параметр
    task_repo: Any,  # НОВЫЙ параметр
    item_index: int,  # НОВЫЙ параметр
) -> List[TaskResult]:
    """Стандартная обработка с детальным отслеживанием"""

    results = []

    for prompt_data in prompts_to_process:
        for model in models:
            adapter = adapters[model.id]

            # Определяем тип операции
            operation_type = "standard"
            if prompt_data["variation_type"]:
                operation_type = "variation"

            try:
                # ===== ШАГ 1: ИНФЕРЕНС =====
                # Обновляем current_execution перед генерацией
                current_execution = {
                    "index": item_index,
                    "prompt": prompt_data["text"][:100],
                    "prompt_variation": prompt_data["variation_type"] or "original",
                    "model_id": model.id,
                    "model_name": model.name,
                    "operation_type": operation_type,
                    "started_at": datetime.utcnow().isoformat(),
                    "output": None,
                }

                await task_repo.update(
                    task_id, {"current_execution": current_execution}
                )

                # Генерация
                start_time = time.time()
                response = await adapter.generate(prompt_data["text"])
                execution_time = time.time() - start_time

                # Обновляем с результатом
                current_execution["output"] = response[:100]
                current_execution["execution_time"] = execution_time
                await task_repo.update(
                    task_id, {"current_execution": current_execution}
                )

                # Создаем результат
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
                        "operation_type": operation_type,
                    },
                    original_input=prompt_data["original"],
                    variation_type=prompt_data["variation_type"],
                )

                # ===== ШАГ 2: JUDGE ОЦЕНКА (если включено) =====
                if judge_service:
                    # Обновляем тип операции
                    current_execution["operation_type"] = "judge_evaluation"
                    current_execution["output"] = None  # Очищаем для judge
                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

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

                    # Обновляем с результатом judge
                    current_execution["output"] = (
                        f"Score: {judge_result['overall_score']:.2f}"
                    )
                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

                # ===== ШАГ 3: RTA ОЦЕНКА (если включено) =====
                if rta_evaluator:
                    # Обновляем тип операции
                    current_execution["operation_type"] = "rta_evaluation"
                    current_execution["output"] = None
                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

                    rta_result = await rta_evaluator.evaluate_output(
                        input_prompt=prompt_data["text"],
                        model_output=response,
                    )

                    result.refused = rta_result["refused"]
                    result.metadata["rta_reasoning"] = rta_result["reasoning"]
                    result.metadata["rta_confidence"] = rta_result["confidence"]

                    # Обновляем с результатом RTA
                    refused_text = (
                        "REFUSED" if rta_result["refused"] == "1" else "ANSWERED"
                    )
                    current_execution["output"] = f"RTA: {refused_text}"
                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

                # ===== ШАГ 4: СОХРАНЕНИЕ В RECENT_EXECUTIONS =====
                execution_summary = {
                    "index": item_index,
                    "prompt": prompt_data["text"][:100],
                    "prompt_variation": prompt_data["variation_type"] or "original",
                    "output": response[:100],
                    "model_id": model.id,
                    "model_name": model.name,
                    "operation_type": operation_type,
                    "execution_time": execution_time,
                    "completed_at": datetime.utcnow().isoformat(),
                }

                # Добавляем judge/RTA инфо если есть
                if result.judge_score:
                    execution_summary["judge_score"] = result.judge_score
                if result.refused:
                    execution_summary["refused"] = result.refused

                # Получаем текущую задачу для recent_executions
                current_task = await task_repo.find_by_id(task_id)
                recent = current_task.recent_executions or []
                recent.insert(0, execution_summary)
                recent = recent[:2]

                # Обновляем recent и очищаем current
                await task_repo.update(
                    task_id,
                    {
                        "recent_executions": recent,
                        "current_execution": None,
                    },
                )

                logger.info(f"Successfully processed: {model.name} - {operation_type}")
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing with model {model.name}: {e}")

                # Сохраняем ошибку в recent_executions
                error_summary = {
                    "index": item_index,
                    "prompt": prompt_data["text"][:100],
                    "prompt_variation": prompt_data["variation_type"] or "original",
                    "model_name": model.name,
                    "operation_type": operation_type,
                    "error": str(e)[:200],
                    "completed_at": datetime.utcnow().isoformat(),
                }

                current_task = await task_repo.find_by_id(task_id)
                recent = current_task.recent_executions or []
                recent.insert(0, error_summary)
                recent = recent[:2]

                await task_repo.update(
                    task_id, {"recent_executions": recent, "current_execution": None}
                )

                continue

    return results


async def _process_with_ab_test(
    item: Any,
    prompts_to_process: List[Dict],
    adapters: Dict[str, Model],
    ab_variants: List[Dict[str, Any]],
    task: Task,
    judge_service: Optional[Any],
    rta_evaluator: Optional[Any],
    task_id: str,  # НОВЫЙ параметр
    task_repo: Any,  # НОВЫЙ параметр
    item_index: int,  # НОВЫЙ параметр
) -> List[TaskResult]:
    """Обработка элемента в режиме A/B теста с детальным отслеживанием"""

    results = []

    # Определяем количество семплов на вариант
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

                    # Определяем тип операции
                    operation_type = "ab_test"
                    if prompt_data["variation_type"]:
                        operation_type = "ab_test_variation"

                    # Параметры генерации
                    gen_params = {}
                    if variant.get("temperature") is not None:
                        gen_params["temperature"] = variant["temperature"]
                    if variant.get("system_prompt"):
                        gen_params["system_prompt"] = variant["system_prompt"]
                    if variant.get("parameters"):
                        gen_params.update(variant["parameters"])

                    # ===== ШАГ 1: ИНФЕРЕНС =====
                    # Обновляем current_execution
                    current_execution = {
                        "index": item_index,
                        "prompt": final_prompt[:100],
                        "prompt_variation": prompt_data["variation_type"] or "original",
                        "model_id": variant["model_id"],
                        "model_name": variant["model_name"],
                        "operation_type": operation_type,
                        "ab_variant": variant["name"],
                        "ab_sample_index": sample_idx,
                        "started_at": datetime.utcnow().isoformat(),
                        "output": None,
                    }

                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

                    # Генерация
                    start_time = time.time()
                    response = await adapter.generate(final_prompt, **gen_params)
                    execution_time = time.time() - start_time

                    # Обновляем с результатом
                    current_execution["output"] = response[:100]
                    current_execution["execution_time"] = execution_time
                    await task_repo.update(
                        task_id, {"current_execution": current_execution}
                    )

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
                            "operation_type": operation_type,
                        },
                        metrics=task.config.evaluation_metrics,
                        original_input=prompt_data["original"],
                        variation_type=prompt_data["variation_type"],
                        ab_variant=variant["name"],
                    )

                    # ===== ШАГ 2: JUDGE ОЦЕНКА =====
                    if judge_service:
                        current_execution["operation_type"] = "ab_judge_evaluation"
                        current_execution["output"] = None
                        await task_repo.update(
                            task_id, {"current_execution": current_execution}
                        )

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

                        current_execution["output"] = (
                            f"Score: {judge_result['overall_score']:.2f}"
                        )
                        await task_repo.update(
                            task_id, {"current_execution": current_execution}
                        )

                    # ===== ШАГ 3: RTA ОЦЕНКА =====
                    if rta_evaluator:
                        current_execution["operation_type"] = "ab_rta_evaluation"
                        current_execution["output"] = None
                        await task_repo.update(
                            task_id, {"current_execution": current_execution}
                        )

                        rta_result = await rta_evaluator.evaluate_output(
                            input_prompt=final_prompt,
                            model_output=response,
                        )

                        result.refused = rta_result["refused"]
                        result.metadata["rta_reasoning"] = rta_result["reasoning"]

                        refused_text = (
                            "REFUSED" if rta_result["refused"] == "1" else "ANSWERED"
                        )
                        current_execution["output"] = f"RTA: {refused_text}"
                        await task_repo.update(
                            task_id, {"current_execution": current_execution}
                        )

                    # ===== ШАГ 4: СОХРАНЕНИЕ В RECENT_EXECUTIONS =====
                    execution_summary = {
                        "index": item_index,
                        "prompt": final_prompt[:100],
                        "prompt_variation": prompt_data["variation_type"] or "original",
                        "output": response[:100],
                        "model_id": variant["model_id"],
                        "model_name": variant["model_name"],
                        "operation_type": operation_type,
                        "ab_variant": variant["name"],
                        "execution_time": execution_time,
                        "completed_at": datetime.utcnow().isoformat(),
                    }

                    if result.judge_score:
                        execution_summary["judge_score"] = result.judge_score
                    if result.refused:
                        execution_summary["refused"] = result.refused

                    current_task = await task_repo.find_by_id(task_id)
                    recent = current_task.recent_executions or []
                    recent.insert(0, execution_summary)
                    recent = recent[:2]

                    await task_repo.update(
                        task_id,
                        {
                            "recent_executions": recent,
                            "current_execution": None,
                        },
                    )

                    logger.info(
                        f"AB test processed: {variant['name']} - sample {sample_idx}"
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error in A/B variant {variant['name']}: {e}")

                    # Сохраняем ошибку
                    error_summary = {
                        "index": item_index,
                        "prompt": final_prompt[:100]
                        if "final_prompt" in locals()
                        else "N/A",
                        "model_name": variant.get("model_name", "N/A"),
                        "operation_type": operation_type,
                        "ab_variant": variant["name"],
                        "error": str(e)[:200],
                        "completed_at": datetime.utcnow().isoformat(),
                    }

                    current_task = await task_repo.find_by_id(task_id)
                    recent = current_task.recent_executions or []
                    recent.insert(0, error_summary)
                    recent = recent[:2]

                    await task_repo.update(
                        task_id,
                        {"recent_executions": recent, "current_execution": None},
                    )

                    continue

    return results


async def _run_inference_async(celery_task, task_id: str):
    """Асинхронное выполнение инференса с правильной интеграцией"""

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

    # ВАЖНО: Если задача возобновляется (resumed_at установлено или есть results)
    # то выполняем recovery
    if task.resumed_at or (task.results and len(task.results) > 0):
        logger.info(f"Task {task_id} is being resumed, running recovery...")

        recovery_info = await _recover_task_state(task_id, task_repo)

        if recovery_info["warnings"]:
            logger.warning(f"Recovery warnings: {recovery_info['warnings']}")

        logger.info(
            f"Recovery completed: recovered {recovery_info['recovered_results']} results, "
            f"starting from index {recovery_info['last_index']}"
        )

        # Перезагружаем задачу после recovery
        task = await task_repo.find_by_id(task_id)

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

        # Инициализация генератора вариаций
        variation_generator = None
        if task.config.variations.enabled and task.config.variations.model_id:
            variation_generator = PromptVariationGenerator(
                task.config.variations.model_id
            )
            await variation_generator.initialize()
            logger.info("Variation generator initialized")

        # Инициализация LLM judge
        judge_service = None
        if task.config.judge.enabled and task.config.judge.model_id:
            judge_service = LLMJudgeService(task.config.judge.model_id)
            await judge_service.initialize()
            logger.info("LLM Judge initialized")

        # Инициализация RTA evaluator
        rta_evaluator = None
        if task.config.rta.enabled and task.config.rta.rta_judge_model_id:
            rta_evaluator = RTAEvaluator(
                task.config.rta.rta_judge_model_id,
                task.config.rta.rta_prompt_template,
            )
            await rta_evaluator.initialize()
            logger.info("RTA Evaluator initialized")

        # Подготовка AB тестов
        ab_variants = None
        if task.config.ab_test.enabled:
            ab_variants = _prepare_ab_test_variants(task.config.ab_test, models)
            logger.info(f"A/B test enabled with {len(ab_variants)} variants")

        # Получаем элементы датасета
        max_samples = task.config.max_samples or dataset.size
        items = await dataset_repo.get_items(task.dataset_id, limit=max_samples)
        total_items = len(items)

        # Вычисляем ожидаемое количество инференсов
        expected_total_inferences = calculate_total_inferences(
            total_items=total_items,
            num_models=len(models),
            variations_enabled=task.config.variations.enabled,
            variations_per_prompt=task.config.variations.count_per_strategy
            if task.config.variations.enabled
            else 0,
            num_variation_strategies=len(task.config.variations.strategies)
            if task.config.variations.enabled
            else 0,
            ab_test_enabled=task.config.ab_test.enabled,
            ab_variants_count=len(ab_variants) if ab_variants else 0,
        )

        # Обновляем total_samples
        await task_repo.update(task_id, {"total_samples": expected_total_inferences})

        logger.info(f"Expected total inferences: {expected_total_inferences}")
        logger.info(f"Processing {total_items} dataset items")

        # ВАЖНО: Загружаем предыдущие результаты (для resume)
        all_results: List[TaskResult] = []
        if task.results:
            all_results = task.results.copy()
            logger.info(f"Loaded {len(all_results)} previous results")

        # Начинаем с последнего обработанного индекса
        start_index = task.last_processed_index
        batch_size = task.config.batch_size

        # Настройки сохранения
        SAVE_INTERVAL = 5
        results_since_last_save = 0

        logger.info(f"Starting processing from index {start_index}/{total_items}")

        # Обрабатываем элементы
        for i in range(start_index, total_items, batch_size):
            # Проверяем статус задачи
            current_task = await task_repo.find_by_id(task_id)

            if current_task.status == TaskStatus.PAUSED:
                logger.info(f"Task {task_id} paused at index {i}")

                # Сохраняем ВСЁ перед паузой
                await task_repo.save_intermediate_results(
                    task_id=task_id,
                    results=all_results,
                    processed_count=len(all_results),
                    last_index=i,
                )

                logger.info(f"Saved {len(all_results)} results before pause")

                return {
                    "task_id": task_id,
                    "status": "paused",
                    "last_processed_index": i,
                    "saved_results": len(all_results),
                }

            if current_task.status == TaskStatus.CANCELLED:
                logger.info(f"Task {task_id} cancelled at index {i}")

                # Сохраняем перед отменой
                await task_repo.save_intermediate_results(
                    task_id=task_id,
                    results=all_results,
                    processed_count=len(all_results),
                    last_index=i,
                )

                logger.info(f"Saved {len(all_results)} results before cancellation")

                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "saved_results": len(all_results),
                }

            batch = items[i : i + batch_size]

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

                # Генерируем вариации
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

                    logger.info(f"Generated {len(variations)} variations for item {i}")

                # ВЫЗЫВАЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ с нужными параметрами
                if ab_variants:
                    # Режим A/B тестирования
                    results = await _process_with_ab_test(
                        item=item,
                        prompts_to_process=prompts_to_process,
                        adapters=adapters,
                        ab_variants=ab_variants,
                        task=current_task,
                        judge_service=judge_service,
                        rta_evaluator=rta_evaluator,
                        task_id=task_id,  # НОВЫЙ параметр
                        task_repo=task_repo,  # НОВЫЙ параметр
                        item_index=i,  # НОВЫЙ параметр
                    )
                else:
                    # Обычный режим
                    results = await _process_standard(
                        item=item,
                        prompts_to_process=prompts_to_process,
                        models=models,
                        adapters=adapters,
                        task=current_task,
                        judge_service=judge_service,
                        rta_evaluator=rta_evaluator,
                        task_id=task_id,  # НОВЫЙ параметр
                        task_repo=task_repo,  # НОВЫЙ параметр
                        item_index=i,  # НОВЫЙ параметр
                    )

                all_results.extend(results)
                results_since_last_save += len(results)

                # ПЕРИОДИЧЕСКОЕ СОХРАНЕНИЕ
                if results_since_last_save >= SAVE_INTERVAL:
                    logger.info(
                        f"Saving intermediate results: {len(all_results)} total"
                    )

                    await task_repo.save_intermediate_results(
                        task_id=task_id,
                        results=all_results,
                        processed_count=len(all_results),
                        last_index=i,
                    )

                    results_since_last_save = 0

            # Обновляем прогресс после батча
            processed = len(all_results)
            progress = (
                (processed / expected_total_inferences * 100)
                if expected_total_inferences > 0
                else 0
            )

            await task_repo.update_progress(task_id, progress, processed)
            await task_repo.update(
                task_id, {"last_processed_index": min(i + batch_size, total_items)}
            )

            # Сохраняем после каждого батча
            if results_since_last_save > 0:
                await task_repo.save_intermediate_results(
                    task_id=task_id,
                    results=all_results,
                    processed_count=len(all_results),
                    last_index=min(i + batch_size, total_items),
                )
                results_since_last_save = 0

            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "processed": processed,
                    "total": expected_total_inferences,
                },
            )

            logger.info(
                f"Task {task_id}: {processed}/{expected_total_inferences} ({progress:.1f}%)"
            )

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

        # Include/Exclude оценка
        if dataset.include_column or dataset.exclude_column:
            ie_metrics = IncludeExcludeEvaluator.evaluate_results(all_results)
            aggregated_metrics["include_exclude"] = ie_metrics

        # A/B тест анализ
        if task.config.ab_test.enabled and all_results:
            logger.info(f"Running A/B analysis on {len(all_results)} results")

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

        # ФИНАЛЬНОЕ сохранение
        logger.info(f"Saving final results: {len(all_results)} total")
        await task_repo.set_results(task_id, all_results, aggregated_metrics)

        # Завершаем задачу
        await task_repo.update_status(task_id, TaskStatus.COMPLETED)

        logger.info(f"Task {task_id} completed successfully")

        return {
            "task_id": task_id,
            "status": "completed",
            "total_samples": expected_total_inferences,
            "total_results": len(all_results),
            "models_tested": len(models),
        }

    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}", exc_info=True)
        await task_repo.update_status(task_id, TaskStatus.FAILED, error=str(e))
        raise
