# src/core/tasks/progress_tracker.py

import asyncio
import time
from typing import Any, Dict, List, Optional

from src.utils.logger import logger


class ProgressTracker:
    """
    Централизованный трекер прогресса.
    Запускается как фоновая корутина через asyncio.create_task().
    """

    def __init__(
        self,
        task_id: str,
        celery_task: Any,
        task_repo: Any,
        total_expected: int,
        update_interval_seconds: float = 3.0,
        checkpoint_interval_seconds: float = 30.0,
    ):
        self.task_id = task_id
        self.celery_task = celery_task
        self.task_repo = task_repo
        self.total_expected = total_expected
        self.update_interval = update_interval_seconds
        self.checkpoint_interval = checkpoint_interval_seconds

        self._saved: int = 0
        self._errors: int = 0
        self._current_model_name: str = ""
        self._current_model_idx: int = 0
        self._total_models: int = 0
        self._last_item_index: int = 0
        self._completed_model_ids: List[str] = []

        self._start_time: float = time.time()
        self._last_update_time: float = 0.0
        self._last_checkpoint_time: float = 0.0

        self._shutdown = asyncio.Event()
        self._dirty = asyncio.Event()

    # ── Public API ─────────────────────────────────────────────────────────

    def record_result(self) -> None:
        self._saved += 1
        self._dirty.set()

    def record_error(self) -> None:
        self._errors += 1

    def set_current_model(
        self,
        model_name: str,
        model_idx: int,
        total_models: int,
    ) -> None:
        self._current_model_name = model_name
        self._current_model_idx = model_idx
        self._total_models = total_models
        self._dirty.set()

    def set_last_item_index(self, idx: int) -> None:
        self._last_item_index = idx

    def mark_model_completed(self, model_id: str) -> None:
        if model_id not in self._completed_model_ids:
            self._completed_model_ids.append(model_id)

    def set_saved(self, count: int) -> None:
        self._saved = count

    # ── Computed properties ────────────────────────────────────────────────

    @property
    def progress_percent(self) -> float:
        if self.total_expected <= 0:
            return 0.0
        return min(self._saved / self.total_expected * 100, 100.0)

    @property
    def throughput(self) -> float:
        elapsed = time.time() - self._start_time
        return self._saved / elapsed if elapsed > 0 else 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.throughput <= 0:
            return None
        remaining = self.total_expected - self._saved
        return remaining / self.throughput if remaining > 0 else 0.0

    def get_meta(self) -> Dict:
        eta = self.eta_seconds
        return {
            "progress": round(self.progress_percent, 2),
            "processed": self._saved,
            "total": self.total_expected,
            "errors": self._errors,
            "current_model": self._current_model_name,
            "model_progress": (
                f"{self._current_model_idx + 1}/{self._total_models}"
                if self._total_models > 0
                else ""
            ),
            "throughput_per_sec": round(self.throughput, 2),
            "eta_seconds": round(eta, 0) if eta is not None else None,
        }

    # ── Background worker ──────────────────────────────────────────────────

    async def run(self) -> None:
        logger.debug(f"ProgressTracker started for task {self.task_id}")

        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._dirty.wait(),
                    timeout=self.update_interval,
                )
                self._dirty.clear()
            except asyncio.TimeoutError:
                pass

            now = time.time()

            if now - self._last_update_time >= self.update_interval:
                await self._flush_progress()
                self._last_update_time = now

            if now - self._last_checkpoint_time >= self.checkpoint_interval:
                await self._save_checkpoint()
                self._last_checkpoint_time = now

        await self._flush_progress()
        await self._save_checkpoint()
        logger.debug(f"ProgressTracker stopped for task {self.task_id}")

    async def stop(self) -> None:
        self._shutdown.set()
        self._dirty.set()
        await asyncio.sleep(0.1)

    async def _flush_progress(self) -> None:
        meta = self.get_meta()

        try:
            await self.task_repo.update_progress(
                self.task_id,
                meta["progress"],
                meta["processed"],
            )
            await self.task_repo.update(
                self.task_id,
                {
                    "last_processed_index": self._last_item_index,
                    "current_execution": {
                        "model_name": self._current_model_name,
                        "model_progress": meta["model_progress"],
                        "throughput": meta["throughput_per_sec"],
                        "eta_seconds": meta["eta_seconds"],
                        "last_item_index": self._last_item_index,
                    },
                },
            )
        except Exception as e:
            logger.warning(f"Progress flush failed: {e}")

        try:
            self.celery_task.update_state(
                state="PROGRESS",
                meta=meta,
            )
        except Exception as e:
            logger.warning(f"Celery state update failed: {e}")

    async def _save_checkpoint(self) -> None:
        checkpoint = {
            "sequence_counter": self._saved,
            "completed_model_ids": list(self._completed_model_ids),
            "last_item_index": self._last_item_index,
            "checkpoint_at": time.time(),
        }
        try:
            await self.task_repo.update(
                self.task_id,
                {"recovery_checkpoint": checkpoint},
            )
            logger.debug(
                f"Checkpoint saved: {self._saved} results, "
                f"models={self._completed_model_ids}"
            )
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
