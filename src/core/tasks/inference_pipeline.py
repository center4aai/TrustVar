# src/core/tasks/inference_pipeline.py

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.adapters.ollama_adapter import OllamaAdapter
from src.core.schemas.dataset import DatasetItem
from src.core.schemas.task import Task, TaskResult, TaskStatus
from src.core.tasks.progress_tracker import ProgressTracker
from src.core.tasks.recovery import PipelineRecovery, RecoveryState
from src.core.tasks.retry_policy import (
    API_RETRY_POLICY,
    JUDGE_RETRY_POLICY,
    OLLAMA_RETRY_POLICY,
    WRITE_RETRY_POLICY,
    with_retry,
)
from src.core.tasks.variation_cache import CachedPrompt, prebuild_variation_cache
from src.core.taxonomy import normalize_task_type
from src.utils.logger import logger


@dataclass
class InferenceJob:
    item: DatasetItem
    item_index: int
    prompt_data: Dict
    model: Any
    adapter: Any
    ab_variant: Optional[Dict] = None


@dataclass
class EvalJob:
    result: TaskResult
    item: DatasetItem
    prompt_data: Dict
    task: Task
    item_index: int


@dataclass
class PipelineStats:
    generated: int = 0
    evaluated: int = 0
    saved: int = 0
    errors: int = 0
    total_expected: int = 0
    start_time: float = field(default_factory=time.time)
    # WEB-6: per-model success/error tallies so a model that produced zero rows
    # is detected as silently dropped rather than shown as "no data".
    per_model_generated: Dict[str, int] = field(default_factory=dict)
    per_model_errors: Dict[str, int] = field(default_factory=dict)

    def throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.saved / elapsed if elapsed > 0 else 0.0


def _gpu_info_from_adapter(adapter: Any) -> Optional[Dict]:
    if not isinstance(adapter, OllamaAdapter):
        return None
    status = adapter.last_runtime_status
    if status is None:
        return {"provider": "ollama", "status": "unknown"}
    return {
        "provider": "ollama",
        "on_gpu": status.on_gpu,
        "size_vram_mb": status.size_vram_mb,
        "size_total_mb": status.size_total_mb,
        "fallback_detected": status.fallback_detected,
    }


class InferencePipeline:
    def __init__(
        self,
        task: Task,
        task_id: str,
        models: List[Any],
        adapters: Dict[str, Any],
        judge_service: Optional[Any] = None,
        rta_evaluator: Optional[Any] = None,
        variation_generator: Optional[Any] = None,
        api_concurrency: int = 10,
        ollama_concurrency: int = 5,
        judge_concurrency: int = 10,
        write_batch_size: int = 50,
    ):
        self.task = task
        self.task_id = task_id
        self.models = models
        self.adapters = adapters
        self.judge_service = judge_service
        self.rta_evaluator = rta_evaluator
        self.variation_generator = variation_generator
        self.api_concurrency = api_concurrency
        self.judge_concurrency = judge_concurrency
        self.write_batch_size = write_batch_size

        self._api_semaphore = asyncio.Semaphore(api_concurrency)
        self._ollama_semaphore = asyncio.Semaphore(ollama_concurrency)
        self._judge_semaphore = asyncio.Semaphore(judge_concurrency)

        self._eval_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        self.stats = PipelineStats()
        self._shutdown = asyncio.Event()

    def _is_ollama(self, adapter: Any) -> bool:
        return isinstance(adapter, OllamaAdapter)

    def _split_models(self) -> Tuple[List[Any], List[Any]]:
        ollama, api = [], []
        for model in self.models:
            adapter = self.adapters[model.id]
            (ollama if self._is_ollama(adapter) else api).append(model)
        return ollama, api

    # ── Inference ──────────────────────────────────────────────────────────

    async def _generate_single(self, job: InferenceJob) -> Optional[TaskResult]:
        sem = (
            self._ollama_semaphore
            if self._is_ollama(job.adapter)
            else self._api_semaphore
        )
        policy = (
            OLLAMA_RETRY_POLICY if self._is_ollama(job.adapter) else API_RETRY_POLICY
        )

        async with sem:
            try:
                start_time = time.time()

                gen_params = {}
                if job.ab_variant:
                    if job.ab_variant.get("temperature") is not None:
                        gen_params["temperature"] = job.ab_variant["temperature"]
                    if job.ab_variant.get("system_prompt"):
                        gen_params["system_prompt"] = job.ab_variant["system_prompt"]

                response = await with_retry(
                    job.adapter.generate,
                    policy,
                    job.prompt_data["text"],
                    context=f"model={job.model.name} item={job.item_index}",
                    **gen_params,
                )

                execution_time = time.time() - start_time
                gpu_info = _gpu_info_from_adapter(job.adapter)

                metadata = {
                    **(job.item.metadata or {}),
                    "model_name": job.model.name,
                    "model_provider": job.model.provider,
                    "operation_type": (
                        "variation"
                        if job.prompt_data.get("variation_type")
                        else "standard"
                    ),
                    # Critical for resume
                    "item_index": job.item_index,
                    "language": job.prompt_data.get("language", "en"),
                    "task_type": job.prompt_data.get(
                        "task_type",
                        normalize_task_type(
                            (job.item.metadata or {}).get("task_type", "classification")
                        ),
                    ),
                }
                # Surface operator metadata (e.g. mcq_option_permutation's
                # ``permutation``) into the persisted record so EAR can
                # canonicalize permuted answers (eval_service._group_results).
                op_meta = job.prompt_data.get("operator_metadata")
                if op_meta:
                    metadata["operator_metadata"] = op_meta
                if gpu_info:
                    metadata["gpu_info"] = gpu_info

                result = TaskResult(
                    input=job.prompt_data["text"],
                    output=response,
                    model_id=job.model.id,
                    model_name=job.model.name,
                    target=job.prompt_data.get("target", job.item.target),
                    execution_time=execution_time,
                    metrics=self.task.config.evaluation_metrics,
                    metadata=metadata,
                    original_input=job.prompt_data["original"],
                    variation_type=job.prompt_data.get("variation_type"),
                    # keep-all (validate-but-keep): per-variant validator verdict +
                    # layers survive into the persisted result for EVERY variant,
                    # including REJECT (valid=False) — annotation, not a filter.
                    valid=job.prompt_data.get("valid"),
                    validator_verdict=job.prompt_data.get("validator_verdict"),
                    validator_layers=job.prompt_data.get("validator_layers"),
                    ab_variant=(job.ab_variant["name"] if job.ab_variant else None),
                )

                self.stats.generated += 1
                self.stats.per_model_generated[job.model.id] = (
                    self.stats.per_model_generated.get(job.model.id, 0) + 1
                )
                return result

            except Exception as e:
                self.stats.errors += 1
                self.stats.per_model_errors[job.model.id] = (
                    self.stats.per_model_errors.get(job.model.id, 0) + 1
                )
                logger.error(
                    f"Generation failed [{job.model.name}] "
                    f"item {job.item_index}: {e}\n{traceback.format_exc()}"
                )
                return None

    async def _generate_and_enqueue(self, job: InferenceJob) -> None:
        result = await self._generate_single(job)
        if result is None:
            return

        if self.judge_service or self.rta_evaluator:
            await self._eval_queue.put(
                EvalJob(
                    result=result,
                    item=job.item,
                    prompt_data=job.prompt_data,
                    task=self.task,
                    item_index=job.item_index,
                )
            )
        else:
            await self._write_queue.put(result)

    # ── Evaluation ─────────────────────────────────────────────────────────

    async def _eval_worker(self, worker_id: int) -> None:
        logger.debug(f"Eval worker {worker_id} started")

        while not (self._shutdown.is_set() and self._eval_queue.empty()):
            try:
                eval_job: EvalJob = await asyncio.wait_for(
                    self._eval_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            result = eval_job.result
            try:
                async with self._judge_semaphore:
                    judge_model_id = (
                        self.task.config.judge.model_id if self.judge_service else None
                    )
                    rta_model_id = (
                        self.task.config.rta.rta_judge_model_id
                        if self.rta_evaluator
                        else None
                    )
                    same_model = (
                        judge_model_id
                        and rta_model_id
                        and judge_model_id == rta_model_id
                    )

                    if same_model:
                        await self._eval_sequential(eval_job, result)
                    else:
                        await self._eval_parallel(eval_job, result)

                self.stats.evaluated += 1
                await self._write_queue.put(result)

            except Exception as e:
                logger.error(f"Eval worker {worker_id} unhandled error: {e}")
                # Don't lose result — write without evaluation
                await self._write_queue.put(result)
            finally:
                self._eval_queue.task_done()

        logger.debug(f"Eval worker {worker_id} finished")

    async def _eval_sequential(self, eval_job: EvalJob, result: TaskResult) -> None:
        """Judge and RTA sequentially (one model under the hood)"""
        if self.judge_service:
            try:
                judge_result = await with_retry(
                    self._run_judge,
                    JUDGE_RETRY_POLICY,
                    eval_job,
                    context=f"judge item={eval_job.item_index}",
                )
                self._apply_judge_result(result, judge_result)
            except Exception as e:
                logger.warning(
                    f"Judge failed for item {eval_job.item_index}, "
                    f"saving without score: {e}"
                )

        if self.rta_evaluator:
            try:
                rta_result = await with_retry(
                    self._run_rta,
                    JUDGE_RETRY_POLICY,
                    eval_job,
                    context=f"rta item={eval_job.item_index}",
                )
                self._apply_rta_result(result, rta_result)
            except Exception as e:
                logger.warning(
                    f"RTA failed for item {eval_job.item_index}, "
                    f"saving without score: {e}"
                )

    async def _eval_parallel(self, eval_job: EvalJob, result: TaskResult) -> None:
        """Judge and RTA parallel (different models)"""
        tasks = []
        if self.judge_service:
            tasks.append(
                with_retry(
                    self._run_judge,
                    JUDGE_RETRY_POLICY,
                    eval_job,
                    context=f"judge item={eval_job.item_index}",
                )
            )
        if self.rta_evaluator:
            tasks.append(
                with_retry(
                    self._run_rta,
                    JUDGE_RETRY_POLICY,
                    eval_job,
                    context=f"rta item={eval_job.item_index}",
                )
            )

        eval_results = await asyncio.gather(*tasks, return_exceptions=True)

        for eval_result in eval_results:
            if isinstance(eval_result, Exception):
                logger.warning(f"Eval failed: {eval_result}")
                continue
            if eval_result.get("type") == "judge":
                self._apply_judge_result(result, eval_result)
            elif eval_result.get("type") == "rta":
                self._apply_rta_result(result, eval_result)

    async def _run_judge(self, eval_job: EvalJob) -> Dict:
        result = await self.judge_service.evaluate_output(
            input_prompt=eval_job.prompt_data["text"],
            model_output=eval_job.result.output,
            task_description=eval_job.task.name,
            reference_output=eval_job.item.target,
            criteria=eval_job.task.config.judge.criteria,
            custom_template=eval_job.task.config.judge.custom_prompt_template,
            metadata=eval_job.item.metadata,
        )
        return {"type": "judge", **result}

    async def _run_rta(self, eval_job: EvalJob) -> Dict:
        result = await self.rta_evaluator.evaluate_output(
            input_prompt=eval_job.prompt_data["text"],
            model_output=eval_job.result.output,
        )
        return {"type": "rta", **result}

    def _apply_judge_result(self, result: TaskResult, judge_result: Dict) -> None:
        result.judge_score = judge_result["overall_score"]
        result.judge_results = judge_result["results"]
        # JUDGE-note: preserve raw_response for /export diagnostics
        if (
            isinstance(result.judge_results, dict)
            and "response" not in result.judge_results
        ):
            raw = judge_result.get("raw_response")
            if raw:
                result.judge_results["response"] = raw
        result.metadata["judge_criteria_scores"] = judge_result["criteria_scores"]

        reasoning: Dict[str, Any] = {}
        data = judge_result.get("results") or {}
        if isinstance(data, dict):
            verdict_reasoning = data.get("verdict_reasoning")
            if verdict_reasoning:
                reasoning["verdict_reasoning"] = verdict_reasoning
            scores = data.get("scores")
            if isinstance(scores, dict):
                for k, v in scores.items():
                    if k.endswith("_reasoning") and v:
                        reasoning[k] = v
        if reasoning:
            result.metadata["judge_reasoning"] = reasoning

    def _apply_rta_result(self, result: TaskResult, rta_result: Dict) -> None:
        result.refused = rta_result["refused"]
        result.metadata["rta_reasoning"] = rta_result["reasoning"]
        result.metadata["rta_confidence"] = rta_result.get("confidence")

    # ── Re-evaluation on resume ───────────────────────────────────────────

    async def _run_missing_evaluations(
        self,
        recovery_state: RecoveryState,
        result_repo: Any,
    ) -> None:
        # N2: re-evaluation updates already-persisted rows in place; those rows
        # were counted into the tracker via set_saved(sequence_start), so we
        # must NOT call tracker.record_result() here — doing so double-counts
        # progress and pushes _saved past total_expected on resume.
        all_missing: Dict[str, Set] = {}

        for model_id, pairs in recovery_state.needs_judge_eval.items():
            all_missing.setdefault(model_id, set()).update(pairs)
        for model_id, pairs in recovery_state.needs_rta_eval.items():
            all_missing.setdefault(model_id, set()).update(pairs)

        if not all_missing:
            return

        total = sum(len(p) for p in all_missing.values())
        logger.info(f"Re-evaluating {total} results missing judge/rta")

        for model_id, pairs in all_missing.items():
            item_indices = list({p[0] for p in pairs})
            page_size = 100

            for i in range(0, len(item_indices), page_size):
                batch_indices = item_indices[i : i + page_size]

                # Use repository method
                docs = await result_repo.find_docs_for_reevaluation(
                    task_id=self.task_id,
                    model_id=model_id,
                    item_indices=batch_indices,
                    need_judge=bool(self.judge_service),
                    need_rta=bool(self.rta_evaluator),
                )

                await asyncio.gather(
                    *[self._reevaluate_single(doc, result_repo) for doc in docs],
                    return_exceptions=True,
                )

    async def _reevaluate_single(self, doc: Dict, result_repo: Any) -> None:
        update = {}
        try:
            if self.judge_service and doc.get("judge_score") is None:
                judge_result = await with_retry(
                    self.judge_service.evaluate_output,
                    JUDGE_RETRY_POLICY,
                    input_prompt=doc["input"],
                    model_output=doc["output"],
                    task_description=self.task.name,
                    # F4: the reference lives at the top-level `target`, not in
                    # metadata; and metadata must be forwarded so the judge sees
                    # the row's real task_type (else open_qa resume rows hit the
                    # reference-free branch + the "generation" default — a
                    # different rubric than the fresh run).
                    reference_output=doc.get("target"),
                    criteria=self.task.config.judge.criteria,
                    custom_template=self.task.config.judge.custom_prompt_template,
                    metadata=doc.get("metadata"),
                    context=f"re-eval judge doc={doc['_id']}",
                )
                update["judge_score"] = judge_result["overall_score"]
                update["judge_results"] = judge_result["results"]
                update["metadata.judge_criteria_scores"] = judge_result[
                    "criteria_scores"
                ]

            if self.rta_evaluator and doc.get("refused") is None:
                rta_result = await with_retry(
                    self.rta_evaluator.evaluate_output,
                    JUDGE_RETRY_POLICY,
                    input_prompt=doc["input"],
                    model_output=doc["output"],
                    context=f"re-eval rta doc={doc['_id']}",
                )
                update["refused"] = rta_result["refused"]
                update["metadata.rta_reasoning"] = rta_result["reasoning"]
                update["metadata.rta_confidence"] = rta_result.get("confidence")

            # Use repository method
            await result_repo.update_evaluation(doc["_id"], update)

        except Exception as e:
            logger.error(f"Re-evaluation failed for doc {doc['_id']}: {e}")

    # ── Writing results ────────────────────────────────────────────────────

    async def _batch_writer(
        self,
        result_repo: Any,
        sequence_start: int,
        tracker: ProgressTracker,
    ) -> int:
        sequence_counter = sequence_start
        pending: List[TaskResult] = []

        while not (self._shutdown.is_set() and self._write_queue.empty()):
            try:
                result = await asyncio.wait_for(self._write_queue.get(), timeout=0.5)
                pending.append(result)
                self._write_queue.task_done()
            except asyncio.TimeoutError:
                pass

            while (
                not self._write_queue.empty() and len(pending) < self.write_batch_size
            ):
                try:
                    result = self._write_queue.get_nowait()
                    pending.append(result)
                    self._write_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            if not pending:
                continue

            success = await self._write_batch_with_retry(
                result_repo, pending, sequence_counter
            )

            if success:
                sequence_counter += len(pending)
                for _ in pending:
                    tracker.record_result()
                self.stats.saved += len(pending)
                pending.clear()
            else:
                logger.error(
                    f"Batch write failed after retries, "
                    f"{len(pending)} results pending retry"
                )
                await asyncio.sleep(5.0)

        # Final flush
        if pending:
            success = await self._write_batch_with_retry(
                result_repo, pending, sequence_counter
            )
            if success:
                sequence_counter += len(pending)
                for _ in pending:
                    tracker.record_result()
                self.stats.saved += len(pending)
            else:
                logger.error(f"Final batch write failed: {len(pending)} results lost")

        logger.info(f"BatchWriter finished: total written={sequence_counter}")
        return sequence_counter

    async def _write_batch_with_retry(
        self,
        result_repo: Any,
        pending: List[TaskResult],
        sequence_counter: int,
    ) -> bool:
        from src.core.schemas.task_result import StoredTaskResult

        for attempt in range(WRITE_RETRY_POLICY.max_attempts):
            try:
                stored = [
                    StoredTaskResult(
                        task_id=self.task_id,
                        sequence_num=sequence_counter + idx,
                        **r.model_dump(),
                    )
                    for idx, r in enumerate(pending)
                ]
                await result_repo.insert_many(stored, ordered=False)
                return True
            except Exception as e:
                delay = WRITE_RETRY_POLICY.calculate_delay(attempt)
                logger.warning(
                    f"Batch write attempt {attempt + 1}/"
                    f"{WRITE_RETRY_POLICY.max_attempts} failed: {e}. "
                    f"Retry in {delay:.1f}s..."
                )
                if attempt < WRITE_RETRY_POLICY.max_attempts - 1:
                    await asyncio.sleep(delay)
        return False

    # ── Model processing ───────────────────────────────────────────────────

    def _get_prompts_from_cache(
        self,
        item: DatasetItem,
        item_index: int,
        variation_cache: Dict[int, CachedPrompt],
    ) -> List[Dict]:
        if item_index in variation_cache:
            return variation_cache[item_index].all_prompts()

        # Fallback without variations
        rendered = item.prompt
        if item.template and item.variables:
            try:
                from src.utils.template import render_template

                rendered = render_template(item.template, item.variables)
            except Exception as e:
                logger.warning(f"Template render failed item {item_index}: {e}")

        lang = (item.metadata or {}).get("language", "en")
        # S1.1: store canonical task_type for the scoring/eval path. This
        # fallback (no variations) does not invoke operators, so the fine
        # task_semantics is not needed here.
        task_type = normalize_task_type(
            (item.metadata or {}).get("task_type", "classification")
        )

        return [
            {
                "text": rendered,
                # P0.1: baseline original = rendered for unified grouping key with variations
                "original": rendered,
                "variation_type": None,
                "target": item.target,
                "language": lang,
                "task_type": task_type,
                # Baseline (no-variation fallback): symmetric with
                # CachedPrompt.all_prompts — no verdict/layers, valid by construction.
                "validator_verdict": None,
                "validator_layers": None,
                "valid": True,
            }
        ]

    def _apply_variant_to_prompt(self, prompt_data: Dict, variant: Dict) -> Dict:
        result = dict(prompt_data)
        if variant.get("prompt_template"):
            result["text"] = variant["prompt_template"].format(
                input=prompt_data["text"]
            )
        return result

    async def _process_api_models(
        self,
        api_models: List[Any],
        items: List[DatasetItem],
        task_repo: Any,
        ab_variants: Optional[List[Dict]],
        variation_cache: Dict[int, CachedPrompt],
        tracker: ProgressTracker,
        completed_pairs_by_model: Dict[str, Set[Tuple[int, Optional[str]]]],
    ) -> None:
        """All API models in parallel"""

        async def process_one_model(model: Any) -> None:
            adapter = self.adapters[model.id]
            completed_pairs = completed_pairs_by_model.get(model.id, set())
            tasks = []

            for item_index, item in enumerate(items):
                prompts = self._get_prompts_from_cache(
                    item, item_index, variation_cache
                )
                for prompt_data in prompts:
                    variation_type = prompt_data.get("variation_type")
                    if (item_index, variation_type) in completed_pairs:
                        continue

                    if ab_variants:
                        model_variants = [
                            v for v in ab_variants if v["model_id"] == model.id
                        ]
                        for variant in model_variants:
                            tasks.append(
                                self._generate_and_enqueue(
                                    InferenceJob(
                                        item=item,
                                        item_index=item_index,
                                        prompt_data=self._apply_variant_to_prompt(
                                            prompt_data, variant
                                        ),
                                        model=model,
                                        adapter=adapter,
                                        ab_variant=variant,
                                    )
                                )
                            )
                    else:
                        tasks.append(
                            self._generate_and_enqueue(
                                InferenceJob(
                                    item=item,
                                    item_index=item_index,
                                    prompt_data=prompt_data,
                                    model=model,
                                    adapter=adapter,
                                )
                            )
                        )

            await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.gather(
            *[process_one_model(m) for m in api_models],
            return_exceptions=True,
        )

    async def _process_single_ollama_model(
        self,
        model: Any,
        items: List[DatasetItem],
        task_repo: Any,
        ab_variants: Optional[List[Dict]],
        variation_cache: Dict[int, CachedPrompt],
        tracker: ProgressTracker,
        completed_pairs: Set[Tuple[int, Optional[str]]],
    ) -> None:
        adapter = self.adapters[model.id]
        logger.info(f"Processing Ollama model: {model.name}")

        # Collect all jobs into a list, then run in parallel
        tasks = []
        for item_index, item in enumerate(items):
            tracker.set_last_item_index(item_index)

            if item_index % 50 == 0:
                current_task = await task_repo.find_by_id(self.task_id)
                if current_task.status in (
                    TaskStatus.PAUSED,
                    TaskStatus.CANCELLED,
                ):
                    logger.info(
                        f"Task {self.task_id} {current_task.status.value} "
                        f"at item {item_index}"
                    )
                    # Run what we've collected so far and exit
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    return

            prompts = self._get_prompts_from_cache(item, item_index, variation_cache)

            for prompt_data in prompts:
                variation_type = prompt_data.get("variation_type")
                if (item_index, variation_type) in completed_pairs:
                    continue

                if ab_variants:
                    model_variants = [
                        v for v in ab_variants if v["model_id"] == model.id
                    ]
                    for variant in model_variants:
                        tasks.append(
                            self._generate_and_enqueue(
                                InferenceJob(
                                    item=item,
                                    item_index=item_index,
                                    prompt_data=self._apply_variant_to_prompt(
                                        prompt_data, variant
                                    ),
                                    model=model,
                                    adapter=adapter,
                                    ab_variant=variant,
                                )
                            )
                        )
                else:
                    tasks.append(
                        self._generate_and_enqueue(
                            InferenceJob(
                                item=item,
                                item_index=item_index,
                                prompt_data=prompt_data,
                                model=model,
                                adapter=adapter,
                            )
                        )
                    )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ── Main method ────────────────────────────────────────────────────────

    async def run(
        self,
        items: List[DatasetItem],
        dataset: Any,
        result_repo: Any,
        task_repo: Any,
        celery_task: Any,
        recovery_state: Optional[RecoveryState] = None,
        ab_variants: Optional[List[Dict]] = None,
        expected_total: Optional[int] = None,
    ) -> int:
        # WEB-1: expected_total must be set BEFORE the tracker is created,
        # otherwise the tracker is initialized with total_expected == 0 and
        # progress_percent stays 0% for the whole run.
        if expected_total is not None:
            self.stats.total_expected = expected_total

        # Initialize tracker
        tracker = ProgressTracker(
            task_id=self.task_id,
            celery_task=celery_task,
            task_repo=task_repo,
            total_expected=self.stats.total_expected,
        )

        sequence_start = 0
        completed_model_ids: Set[str] = set()

        if recovery_state and not recovery_state.is_fresh_start:
            sequence_start = recovery_state.sequence_counter
            completed_model_ids = recovery_state.completed_model_ids
            tracker.set_saved(sequence_start)
            for mid in completed_model_ids:
                tracker.mark_model_completed(mid)
            logger.info(
                f"Resuming: {sequence_start} done, "
                f"{len(completed_model_ids)} models completed"
            )

        tracker_task = asyncio.create_task(tracker.run())

        has_eval = bool(self.judge_service or self.rta_evaluator)
        eval_worker_tasks = (
            [
                asyncio.create_task(self._eval_worker(i))
                for i in range(self.judge_concurrency)
            ]
            if has_eval
            else []
        )

        writer_task = asyncio.create_task(
            self._batch_writer(result_repo, sequence_start, tracker)
        )

        ollama_models, api_models = self._split_models()
        api_models = [m for m in api_models if m.id not in completed_model_ids]
        ollama_models = [m for m in ollama_models if m.id not in completed_model_ids]

        total_models = len(api_models) + len(ollama_models)
        logger.info(
            f"Pipeline: {len(api_models)} API + {len(ollama_models)} Ollama models"
        )

        final_counter = 0
        try:
            # Build variation cache once for all models
            variation_cache: Dict[int, CachedPrompt] = {}
            if self.variation_generator:
                logger.info("Pre-building variation cache...")

                self.variation_generator.progress_tracker = tracker

                variation_cache = await prebuild_variation_cache(
                    items=items,
                    dataset=dataset,
                    task_config=self.task.config,
                    variation_generator=self.variation_generator,
                )
                logger.info(f"Variation cache: {len(variation_cache)} items")

            # Phase 0: re-evaluation on resume
            if recovery_state and not recovery_state.is_fresh_start:
                await self._run_missing_evaluations(recovery_state, result_repo)

            # Load completed_pairs for incomplete models
            recovery = PipelineRecovery(task_repo, result_repo)

            api_completed_pairs: Dict[str, Set] = {}
            for model in api_models:
                if recovery_state and not recovery_state.is_fresh_start:
                    api_completed_pairs[
                        model.id
                    ] = await recovery.load_completed_item_indices(
                        self.task_id, model.id
                    )
                else:
                    api_completed_pairs[model.id] = set()

            # Phase 1+2: API and Ollama models in parallel
            api_task = None
            ollama_task = None

            if api_models:
                model_idx_offset = 0
                tracker.set_current_model(
                    "API models (parallel)", model_idx_offset, total_models
                )
                api_task = asyncio.create_task(
                    self._process_api_models(
                        api_models=api_models,
                        items=items,
                        task_repo=task_repo,
                        ab_variants=ab_variants,
                        variation_cache=variation_cache,
                        tracker=tracker,
                        completed_pairs_by_model=api_completed_pairs,
                    )
                )

            if ollama_models:

                async def _run_ollama_sequential() -> None:
                    for model_idx, model in enumerate(ollama_models):
                        global_idx = len(api_models) + model_idx
                        tracker.set_current_model(model.name, global_idx, total_models)

                        current_task = await task_repo.find_by_id(self.task_id)
                        if current_task.status in (
                            TaskStatus.PAUSED,
                            TaskStatus.CANCELLED,
                        ):
                            logger.info(
                                f"Task {self.task_id} {current_task.status.value}, "
                                f"stopping before model {model.name}"
                            )
                            return

                        completed_pairs: Set = set()
                        if recovery_state and not recovery_state.is_fresh_start:
                            completed_pairs = (
                                await recovery.load_completed_item_indices(
                                    self.task_id, model.id
                                )
                            )
                            if completed_pairs:
                                logger.info(
                                    f"Model {model.name}: skipping "
                                    f"{len(completed_pairs)} done pairs"
                                )

                        await self._process_single_ollama_model(
                            model=model,
                            items=items,
                            task_repo=task_repo,
                            ab_variants=ab_variants,
                            variation_cache=variation_cache,
                            tracker=tracker,
                            completed_pairs=completed_pairs,
                        )

                        tracker.mark_model_completed(model.id)

                        adapter = self.adapters[model.id]
                        if isinstance(adapter, OllamaAdapter):
                            try:
                                await adapter.unload()
                                logger.info(f"Unloaded {model.name} from VRAM")
                            except Exception as e:
                                logger.warning(f"Unload failed {model.name}: {e}")

                ollama_task = asyncio.create_task(_run_ollama_sequential())

            await asyncio.gather(*[t for t in (api_task, ollama_task) if t is not None])

            if api_task:
                for model in api_models:
                    tracker.mark_model_completed(model.id)

            # Wait for eval_queue to drain
            if has_eval:
                await self._eval_queue.join()

        finally:
            self._shutdown.set()

            # NEW-1: drain eval workers and write queue and WAIT for the writer
            # BEFORE stopping the tracker. Otherwise the tracker's terminal flush writes
            # a _saved snapshot before BatchWriter finishes the last batch (and calls
            # record_result) → completed task progress is recorded as <100%.
            if eval_worker_tasks:
                await asyncio.gather(*eval_worker_tasks, return_exceptions=True)

            await self._write_queue.join()
            final_counter = await writer_task

            # Tracker stops LAST — its final flush sees the complete _saved.
            await tracker.stop()
            tracker_task.cancel()
            try:
                await tracker_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"Pipeline complete: {final_counter} total results, "
            f"throughput={self.stats.throughput():.1f}/s"
        )
        return final_counter
