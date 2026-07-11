# src/core/tasks/inference_task.py

import asyncio
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Task as CeleryTask
from pymongo import MongoClient

from src.adapters.factory import LLMFactory
from src.adapters.ollama_adapter import OllamaAdapter
from src.config.settings import get_settings
from src.core.schemas.task import Task, TaskStatus
from src.core.tasks.celery_app import celery_app
from src.core.tasks.inference_pipeline import InferencePipeline
from src.core.tasks.post_processing import run_post_processing
from src.core.tasks.recovery import PipelineRecovery, RecoveryState
from src.database.repositories.dataset_repository import DatasetRepository
from src.database.repositories.model_repository import ModelRepository
from src.database.repositories.task_repository import TaskRepository
from src.database.repositories.task_result_repository import TaskResultRepository
from src.utils.logger import logger

settings = get_settings()


# ── Helper functions ───────────────────────────────────────────────────────


def _build_completion_summary(
    *,
    model_ids: List[str],
    per_model_generated: Dict[str, int],
    per_model_errors: Dict[str, int],
    expected_total: int,
    final_count: int,
) -> Dict[str, Any]:

    failed_models = [m for m in model_ids if per_model_generated.get(m, 0) == 0]
    n_with_results = len(model_ids) - len(failed_models)
    return {
        "is_complete": final_count > 0 and not failed_models,
        "final_count": final_count,
        "expected_total": expected_total,
        "n_models_total": len(model_ids),
        "n_models_with_results": n_with_results,
        "failed_models": failed_models,
        "per_model_generated": dict(per_model_generated),
        "per_model_errors": dict(per_model_errors),
    }


async def _load_models(
    task: Task, model_repo: ModelRepository
) -> Tuple[List[Any], Dict[str, Any]]:
    """Load all models in parallel"""

    async def load_one(model_id: str):
        model = await model_repo.find_by_id(model_id)
        if not model:
            logger.warning(f"Model {model_id} not found, skipping")
            return None, None
        adapter = LLMFactory.create(model)
        logger.info(f"Loaded model: {model.name} ({model.provider})")
        return model, adapter

    results = await asyncio.gather(
        *[load_one(mid) for mid in task.model_ids],
        return_exceptions=True,
    )

    models, adapters = [], {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Failed to load model: {result}")
            continue
        model, adapter = result
        if model and adapter:
            models.append(model)
            adapters[model.id] = adapter

    return models, adapters


async def _init_variation_generator(
    task: Task, model_repo: ModelRepository
) -> Optional[Any]:
    if not (task.config.variations.enabled and task.config.variations.strategies):
        return None
    from src.core.services.variation_pipeline import VariationPipeline
    from src.core.services.variation_validator import VariationValidator

    var_model_id = task.config.variations.model_id
    validator = VariationValidator(
        {
            "generator_model": (
                var_model_id or "tier-a-symbolic"
            ),
        }
    )
    pipeline = VariationPipeline(
        model_id=var_model_id,
        validator=validator,
        bypass_validation=task.config.variations.bypass_validation,
        keep_rejected=task.config.variations.keep_rejected,
    )
    if task.config.variations.bypass_validation:
        logger.warning(
            "Variation pipeline: bypass_validation=True (DIAGNOSTIC) — the 3-layer "
            "validator is SKIPPED; variants are NOT verified. Use for coverage "
            "health-checks only, never for real experiments."
        )
    if task.config.variations.keep_rejected:
        logger.info(
            "Variation pipeline: keep_rejected=True (validate-but-keep) — the "
            "validator runs and records per-variant verdict + layers, but REJECTed "
            "variants are RETAINED (valid=False) rather than dropped (keep-all "
            "metric-inclusion policy)."
        )
    logger.info("Variation pipeline initialized")
    return pipeline


async def _init_judge_service(task: Task) -> Optional[Any]:
    if not (task.config.judge.enabled and task.config.judge.model_id):
        return None
    from src.core.services.judge_service import LLMJudgeService

    svc = LLMJudgeService(task.config.judge.model_id)
    await svc.initialize()
    logger.info("Judge service initialized")
    return svc


async def _init_rta_evaluator(task: Task) -> Optional[Any]:
    if not (task.config.rta.enabled and task.config.rta.rta_judge_model_id):
        return None
    from src.core.services.rta_evaluator import RTAEvaluator

    ev = RTAEvaluator(
        task.config.rta.rta_judge_model_id,
        task.config.rta.rta_prompt_template,
    )
    await ev.initialize()
    logger.info("RTA evaluator initialized")
    return ev


async def _unload_ollama_adapters(adapters: Dict[str, Any]) -> None:
    for adapter in adapters.values():
        if isinstance(adapter, OllamaAdapter):
            try:
                await adapter.unload()
            except Exception as e:
                logger.warning(f"Unload failed: {e}")


def _prepare_ab_test_variants(ab_config: Any, models: List[Any]) -> List[Dict]:
    from src.core.schemas.task import ABTestStrategy

    variants = []

    if ab_config.strategy == ABTestStrategy.PROMPT_VARIANTS:
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


def calculate_total_inferences(
    total_items: int,
    num_models: int,
    variations_enabled: bool,
    variations_per_prompt: int,
    num_variation_strategies: int,
    ab_test_enabled: bool,
    ab_variants_count: int,
) -> int:
    # A/B variant count already folds in the per-model factor (variants are
    # built per model in _prepare_ab_test_variants), so it must NOT be
    # multiplied by num_models again. When variations are also enabled the
    # pipeline applies BOTH axes (cached variants × A/B variants), so the
    # variation multiplier has to be combined in here too (H4).
    if ab_test_enabled:
        base = total_items * ab_variants_count
        if variations_enabled:
            variations_count = variations_per_prompt * num_variation_strategies
            base *= 1 + variations_count
        return base

    if variations_enabled:
        variations_count = variations_per_prompt * num_variation_strategies
        return total_items * (1 + variations_count) * num_models

    return total_items * num_models


def _update_task_status_sync(task_id: str, status: str, error: str = "") -> None:
    """Synchronous status update (for on_failure and SoftTimeLimitExceeded)"""
    try:
        client = MongoClient(settings.MONGODB_URL)
        db = client.get_database(settings.MONGODB_DB_NAME)
        update = {
            "status": status,
            "completed_at": datetime.utcnow(),
        }
        if error:
            update["error"] = error
        db["tasks"].update_one({"id": task_id}, {"$set": update})
        client.close()
    except Exception as e:
        logger.error(f"Sync status update failed: {e}")


# ── Celery task ────────────────────────────────────────────────────────────


class InferenceTask(CeleryTask):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Celery task {task_id} failed: {exc}")
        if args:
            _update_task_status_sync(args[0], "failed", str(exc))


@celery_app.task(bind=True, base=InferenceTask)
def run_inference_task(self, task_id: str):
    try:
        return asyncio.run(_run_inference_async(self, task_id))
    except SoftTimeLimitExceeded:
        logger.warning(f"SoftTimeLimitExceeded for task {task_id}")
        _update_task_status_sync(
            task_id,
            "failed",
            "SoftTimeLimitExceeded: partial results preserved",
        )
        return {
            "task_id": task_id,
            "status": "time_limit_exceeded",
        }


# ── Main async function ───────────────────────────────────────────────────


async def _run_inference_async(celery_task: Any, task_id: str) -> Dict:
    task_repo = TaskRepository()
    dataset_repo = DatasetRepository()
    model_repo = ModelRepository()
    result_repo = TaskResultRepository()

    task = await task_repo.find_by_id(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    logger.info(f"Starting inference: task={task_id} name={task.name}")
    await task_repo.update_status(task_id, TaskStatus.RUNNING)

    try:
        # Load dataset
        dataset = await dataset_repo.find_by_id(task.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {task.dataset_id} not found")

        # Load models and initialize services in parallel
        (
            (models, adapters),
            variation_generator,
            judge_service,
            rta_evaluator,
        ) = await asyncio.gather(
            _load_models(task, model_repo),
            _init_variation_generator(task, model_repo),
            _init_judge_service(task),
            _init_rta_evaluator(task),
        )

        if not models:
            raise ValueError("No valid models found for task")

        logger.info(f"Loaded {len(models)} models")

        # A/B variants
        ab_variants = None
        if task.config.ab_test.enabled:
            ab_variants = _prepare_ab_test_variants(task.config.ab_test, models)
            logger.info(f"A/B test: {len(ab_variants)} variants")

        # Get dataset items
        max_samples = task.config.max_samples or dataset.size
        items = await dataset_repo.get_items(task.dataset_id, limit=max_samples)
        total_items = len(items)

        # Calculate expected number of inferences
        expected_total = calculate_total_inferences(
            total_items=total_items,
            num_models=len(models),
            variations_enabled=task.config.variations.enabled,
            variations_per_prompt=(
                task.config.variations.count_per_strategy
                if task.config.variations.enabled
                else 0
            ),
            num_variation_strategies=(
                len(task.config.variations.strategies)
                if task.config.variations.enabled
                else 0
            ),
            ab_test_enabled=task.config.ab_test.enabled,
            ab_variants_count=len(ab_variants) if ab_variants else 0,
        )
        await task_repo.update(task_id, {"total_samples": expected_total})
        logger.info(
            f"Expected inferences: {expected_total} "
            f"({total_items} items x {len(models)} models)"
        )

        # ── State recovery on resume ───────────────────────────────────────
        recovery_state: Optional[RecoveryState] = None
        is_resume = bool(task.resumed_at or task.processed_samples > 0)

        if is_resume:
            logger.info(f"Task {task_id} is a resume, analyzing state...")
            recovery = PipelineRecovery(task_repo, result_repo)
            expected_per_model = expected_total // len(models) if models else 0
            recovery_state = await recovery.recover(
                task_id=task_id,
                model_ids=[m.id for m in models],
                total_items=total_items,
                expected_per_model=expected_per_model,
                judge_enabled=task.config.judge.enabled,
                rta_enabled=task.config.rta.enabled,
            )
            logger.info(
                f"Recovery: {recovery_state.sequence_counter} results, "
                f"{len(recovery_state.completed_model_ids)} models done"
            )

        # ── Pipeline launch ────────────────────────────────────────────────
        pipeline = InferencePipeline(
            task=task,
            task_id=task_id,
            models=models,
            adapters=adapters,
            judge_service=judge_service,
            rta_evaluator=rta_evaluator,
            variation_generator=variation_generator,
            api_concurrency=getattr(settings, "API_MODEL_CONCURRENCY", 10),
            ollama_concurrency=getattr(settings, "OLLAMA_NUM_PARALLEL", 5),
            judge_concurrency=getattr(settings, "JUDGE_CONCURRENCY", 10),
            write_batch_size=getattr(settings, "WRITE_BATCH_SIZE", 50),
        )

        final_count = await pipeline.run(
            items=items,
            dataset=dataset,
            result_repo=result_repo,
            task_repo=task_repo,
            celery_task=celery_task,
            recovery_state=recovery_state,
            ab_variants=ab_variants,
            expected_total=expected_total,
        )

        # ── Post-processing ────────────────────────────────────────────────
        logger.info(f"Running post-processing for task {task_id}")
        await run_post_processing(
            task=task,
            task_id=task_id,
            task_repo=task_repo,
            result_repo=result_repo,
        )

        # WEB-5/WEB-6: record per-model coverage + completeness so the UI can
        # distinguish a complete run from one where a model silently dropped out.
        completion_summary = _build_completion_summary(
            model_ids=[m.id for m in models],
            per_model_generated=pipeline.stats.per_model_generated,
            per_model_errors=pipeline.stats.per_model_errors,
            expected_total=expected_total,
            final_count=final_count,
        )
        await task_repo.set_completion_summary(task_id, completion_summary)
        await task_repo.update_status(task_id, TaskStatus.COMPLETED)
        await _unload_ollama_adapters(adapters)

        if not completion_summary["is_complete"]:
            logger.warning(
                f"Task {task_id} completed PARTIALLY: {final_count} results, "
                f"{completion_summary['n_models_with_results']}/"
                f"{completion_summary['n_models_total']} models produced rows; "
                f"failed models: {completion_summary['failed_models']}"
            )
        else:
            logger.info(
                f"Task {task_id} completed: {final_count} results, "
                f"{len(models)} models"
            )

        return {
            "task_id": task_id,
            "status": "completed",
            "is_complete": completion_summary["is_complete"],
            "total_results": final_count,
            "models_tested": len(models),
            "failed_models": completion_summary["failed_models"],
            "throughput": f"{pipeline.stats.throughput():.1f}/s",
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"SoftTimeLimitExceeded for task {task_id}")
        try:
            await task_repo.update_status(
                task_id,
                TaskStatus.FAILED,
                error="Time limit exceeded. Partial results preserved.",
            )
        except Exception:
            pass
        if "adapters" in locals():
            await _unload_ollama_adapters(adapters)
        return {
            "task_id": task_id,
            "status": "time_limit_exceeded",
        }

    except Exception as e:
        logger.error(
            f"Task {task_id} failed: {e}\n{traceback.format_exc()}",
            exc_info=True,
        )
        try:
            await task_repo.update_status(task_id, TaskStatus.FAILED, error=str(e))
        except Exception:
            pass
        if "adapters" in locals():
            await _unload_ollama_adapters(adapters)
        raise
