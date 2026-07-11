# src/core/tasks/recovery.py

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.logger import logger


@dataclass
class RecoveryState:
    sequence_counter: int
    completed_model_ids: Set[str]
    warnings: List[str]
    needs_judge_eval: Dict[str, Set[Tuple[int, Optional[str]]]] = field(
        default_factory=dict
    )
    needs_rta_eval: Dict[str, Set[Tuple[int, Optional[str]]]] = field(
        default_factory=dict
    )

    @property
    def is_fresh_start(self) -> bool:
        return self.sequence_counter == 0


class PipelineRecovery:
    def __init__(self, task_repo: Any, result_repo: Any):
        self.task_repo = task_repo
        self.result_repo = result_repo

    async def recover(
        self,
        task_id: str,
        model_ids: List[str],
        total_items: int,
        expected_per_model: int,
        judge_enabled: bool = False,
        rta_enabled: bool = False,
    ) -> RecoveryState:
        warnings = []
        task = await self.task_repo.find_by_id(task_id)

        # Source of truth
        real_count = await self.result_repo.count_by_task(task_id)

        if real_count == 0:
            return RecoveryState(
                sequence_counter=0,
                completed_model_ids=set(),
                warnings=[],
            )

        if task.processed_samples != real_count:
            warnings.append(
                f"processed_samples mismatch: "
                f"task={task.processed_samples} vs actual={real_count}. Fixed."
            )
            await self.task_repo.update(task_id, {"processed_samples": real_count})

        # Count per model in parallel
        counts = await asyncio.gather(
            *[
                self.result_repo.count_by_task_and_model(task_id, mid)
                for mid in model_ids
            ],
            return_exceptions=True,
        )

        completed_model_ids: Set[str] = set()
        model_counts: Dict[str, int] = {}

        for model_id, count in zip(model_ids, counts):
            if isinstance(count, Exception):
                warnings.append(f"Count failed for {model_id}: {count}")
                count = 0
            model_counts[model_id] = count
            if count >= expected_per_model:
                completed_model_ids.add(model_id)

        logger.info(
            f"Recovery: total={real_count}, "
            f"completed={len(completed_model_ids)}/{len(model_ids)}, "
            f"per_model={model_counts}"
        )

        # H7: skip on resume is driven entirely by completed_pairs
        # (get_completed_pairs), not by an item index. We only log which models
        # are incomplete; the previous "from item ~N" estimate was unused dead
        # code and the message was misleading.
        incomplete = [m for m in model_ids if m not in completed_model_ids]
        for mid in incomplete:
            warnings.append(
                f"Resume model {mid}: "
                f"{model_counts.get(mid, 0)}/{expected_per_model} done, "
                f"backfilling missing pairs"
            )

        # Find results without judge/rta — use repository methods
        needs_judge_eval: Dict[str, Set] = {}
        needs_rta_eval: Dict[str, Set] = {}

        eval_tasks = []

        for model_id in model_ids:
            if judge_enabled:
                eval_tasks.append(self._get_missing_judge(task_id, model_id))
            if rta_enabled:
                eval_tasks.append(self._get_missing_rta(task_id, model_id))

        if eval_tasks:
            eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

            idx = 0
            for model_id in model_ids:
                if judge_enabled:
                    result = eval_results[idx]
                    idx += 1
                    if isinstance(result, Exception):
                        warnings.append(f"Judge check failed for {model_id}: {result}")
                    elif result:
                        needs_judge_eval[model_id] = result
                        warnings.append(
                            f"Model {model_id}: {len(result)} results missing judge"
                        )

                if rta_enabled:
                    result = eval_results[idx]
                    idx += 1
                    if isinstance(result, Exception):
                        warnings.append(f"RTA check failed for {model_id}: {result}")
                    elif result:
                        needs_rta_eval[model_id] = result
                        warnings.append(
                            f"Model {model_id}: {len(result)} results missing rta"
                        )

        for w in warnings:
            logger.warning(f"Recovery: {w}")

        return RecoveryState(
            sequence_counter=real_count,
            completed_model_ids=completed_model_ids,
            warnings=warnings,
            needs_judge_eval=needs_judge_eval,
            needs_rta_eval=needs_rta_eval,
        )

    async def load_completed_item_indices(
        self,
        task_id: str,
        model_id: str,
    ) -> Set[Tuple[int, Optional[str]]]:
        # Delegate to repository
        return await self.result_repo.get_completed_pairs(task_id, model_id)

    async def _get_missing_judge(
        self, task_id: str, model_id: str
    ) -> Set[Tuple[int, Optional[str]]]:
        return await self.result_repo.get_pairs_without_judge(task_id, model_id)

    async def _get_missing_rta(
        self, task_id: str, model_id: str
    ) -> Set[Tuple[int, Optional[str]]]:
        return await self.result_repo.get_pairs_without_rta(task_id, model_id)
