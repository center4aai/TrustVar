# src/database/repositories/task_result_repository.py

from typing import Any, Dict, List, Optional, Set, Tuple

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING

from src.core.schemas.task_result import StoredTaskResult
from src.database.mongodb import get_database
from src.utils.logger import logger


class TaskResultRepository:
    """
    Repository for the 'task_results' collection.

    Does NOT inherit from BaseRepository for two reasons:
    1. StoredTaskResult uses sequence_num instead of id as key
    2. Specific methods needed (get_completed_pairs, bulk insert, etc.)
    """

    COLLECTION = "task_results"

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._indexes_ensured = False

    @property
    def collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            raise RuntimeError(
                "Collection not initialized. "
                "Call await _get_collection() first or use any public method."
            )
        return self._collection

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_database()
            self._collection = db[self.COLLECTION]
            await self._ensure_indexes()
        return self._collection

    async def _ensure_indexes(self) -> None:
        """
        Basic indexes as fallback.
        Full set created via indexes.py on application startup.
        Names match indexes.py — no duplicates will occur.
        """
        if self._indexes_ensured:
            return
        try:
            await self._collection.create_index(
                [("task_id", ASCENDING), ("sequence_num", ASCENDING)],
                name="task_id_sequence_num",
            )
            await self._collection.create_index(
                [("task_id", ASCENDING), ("model_id", ASCENDING)],
                name="task_id_model_id",
            )
            await self._collection.create_index(
                [("task_id", ASCENDING)],
                name="task_id",
            )
            self._indexes_ensured = True
            logger.info("task_results indexes ensured")
        except Exception as exc:
            logger.warning(f"Could not ensure task_results indexes: {exc}")

    # ── Write ──────────────────────────────────────────────────────────────

    async def insert_one(self, result: StoredTaskResult) -> None:
        collection = await self._get_collection()
        await collection.insert_one(result.model_dump(mode="json"))

    async def insert_many(
        self,
        results: List[StoredTaskResult],
        ordered: bool = False,
    ) -> int:
        """
        ordered=False: if one document errors, others
        continue to be written. Important for batch reliability.
        """
        if not results:
            return 0
        collection = await self._get_collection()
        docs = [r.model_dump(mode="json") for r in results]
        outcome = await collection.insert_many(docs, ordered=ordered)
        return len(outcome.inserted_ids)

    # ── Read ───────────────────────────────────────────────────────────────

    async def find_by_task(
        self,
        task_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[StoredTaskResult]:
        collection = await self._get_collection()
        cursor = (
            collection.find({"task_id": task_id})
            .sort("sequence_num", ASCENDING)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._from_doc(d) for d in docs]

    async def find_by_task_and_model(
        self,
        task_id: str,
        model_id: str,
        skip: int = 0,
        limit: int = 500,
    ) -> List[StoredTaskResult]:
        collection = await self._get_collection()
        cursor = (
            collection.find({"task_id": task_id, "model_id": model_id})
            .sort("sequence_num", ASCENDING)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [self._from_doc(d) for d in docs]

    async def find_all_by_task(self, task_id: str) -> List[StoredTaskResult]:
        """
        All task results.
        Only for final operations (export, evaluation).
        For API use paginated find_by_task().
        """
        collection = await self._get_collection()
        cursor = collection.find({"task_id": task_id}).sort("sequence_num", ASCENDING)
        docs = await cursor.to_list(length=None)
        return [self._from_doc(d) for d in docs]

    async def find_by_task_with_ab(self, task_id: str) -> List[StoredTaskResult]:
        """
        Results with ab_variant != None.
        Used in post_processing for A/B analysis.
        Requires task_id_ab_variant index.
        """
        collection = await self._get_collection()
        cursor = collection.find(
            {"task_id": task_id, "ab_variant": {"$ne": None}}
        ).sort("sequence_num", ASCENDING)
        docs = await cursor.to_list(length=None)
        return [self._from_doc(d) for d in docs]

    # ── Counters ───────────────────────────────────────────────────────────

    async def count_by_task(self, task_id: str) -> int:
        collection = await self._get_collection()
        return await collection.count_documents({"task_id": task_id})

    async def count_by_task_and_model(self, task_id: str, model_id: str) -> int:
        collection = await self._get_collection()
        return await collection.count_documents(
            {"task_id": task_id, "model_id": model_id}
        )

    async def get_max_sequence_num(self, task_id: str) -> int:
        """Maximum sequence_num for task, or -1 if no results."""
        collection = await self._get_collection()
        doc = await collection.find_one(
            {"task_id": task_id},
            sort=[("sequence_num", DESCENDING)],
        )
        return doc["sequence_num"] if doc else -1

    # ── Recovery ───────────────────────────────────────────────────────────

    async def get_completed_pairs(
        self,
        task_id: str,
        model_id: str,
    ) -> Set[Tuple[int, Optional[str]]]:
        """
        Returns Set[(item_index, variation_type)] of already recorded
        results for given model.

        Used during resume:
            if (item_index, variation_type) in completed_pairs:
                continue  # skip already processed

        Requires task_id_model_item_var index.
        Reads only index fields (covered query) — fast.
        """
        collection = await self._get_collection()
        cursor = collection.find(
            {"task_id": task_id, "model_id": model_id},
            projection={
                "_id": 0,
                "metadata.item_index": 1,
                "variation_type": 1,
            },
        )

        pairs: Set[Tuple[int, Optional[str]]] = set()
        async for doc in cursor:
            item_index = doc.get("metadata", {}).get("item_index")
            if item_index is None:
                continue
            pairs.add((item_index, doc.get("variation_type")))

        logger.debug(
            f"Loaded {len(pairs)} completed pairs [task={task_id} model={model_id}]"
        )
        return pairs

    async def get_pairs_without_judge(
        self,
        task_id: str,
        model_id: str,
    ) -> Set[Tuple[int, Optional[str]]]:
        """
        Pairs (item_index, variation_type) where output exists
        but judge_score is missing.
        Used in recovery for re-evaluation during resume.
        """
        collection = await self._get_collection()
        cursor = collection.find(
            {
                "task_id": task_id,
                "model_id": model_id,
                "judge_score": None,
            },
            projection={
                "_id": 0,
                "metadata.item_index": 1,
                "variation_type": 1,
            },
        )
        pairs: Set[Tuple[int, Optional[str]]] = set()
        async for doc in cursor:
            item_index = doc.get("metadata", {}).get("item_index")
            if item_index is not None:
                pairs.add((item_index, doc.get("variation_type")))
        return pairs

    async def get_pairs_without_rta(
        self,
        task_id: str,
        model_id: str,
    ) -> Set[Tuple[int, Optional[str]]]:
        """
        Similar to get_pairs_without_judge, but for refused (RTA).
        """
        collection = await self._get_collection()
        cursor = collection.find(
            {
                "task_id": task_id,
                "model_id": model_id,
                "refused": None,
            },
            projection={
                "_id": 0,
                "metadata.item_index": 1,
                "variation_type": 1,
            },
        )
        pairs: Set[Tuple[int, Optional[str]]] = set()
        async for doc in cursor:
            item_index = doc.get("metadata", {}).get("item_index")
            if item_index is not None:
                pairs.add((item_index, doc.get("variation_type")))
        return pairs

    async def find_docs_for_reevaluation(
        self,
        task_id: str,
        model_id: str,
        item_indices: List[int],
        need_judge: bool = False,
        need_rta: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Documents for re-evaluation during resume.
        Returns raw dict (not StoredTaskResult) —
        _id needed for subsequent update_evaluation().
        """
        if not item_indices or not (need_judge or need_rta):
            return []

        conditions = []
        if need_judge:
            conditions.append({"judge_score": None})
        if need_rta:
            conditions.append({"refused": None})

        query: Dict[str, Any] = {
            "task_id": task_id,
            "model_id": model_id,
            "metadata.item_index": {"$in": item_indices},
            "$or": conditions,
        }

        collection = await self._get_collection()
        docs = await collection.find(
            query,
            projection={
                "_id": 1,
                "input": 1,
                "output": 1,
                # F4: project the reference so resume re-eval can pass it to the
                # judge (top-level field, not inside metadata).
                "target": 1,
                "metadata": 1,
                "variation_type": 1,
                "judge_score": 1,
                "refused": 1,
            },
        ).to_list(length=len(item_indices) * 10)

        return docs

    async def update_evaluation(
        self,
        doc_id: Any,
        update: Dict[str, Any],
    ) -> None:
        """
        Update evaluation fields of existing result (judge/rta).
        Used in _reevaluate_single() for re-evaluation after resume.
        """
        if not update:
            return
        collection = await self._get_collection()
        await collection.update_one(
            {"_id": doc_id},
            {"$set": update},
        )

    # ── Delete ─────────────────────────────────────────────────────────────

    async def delete_by_task(self, task_id: str) -> int:
        collection = await self._get_collection()
        result = await collection.delete_many({"task_id": task_id})
        logger.info(f"Deleted {result.deleted_count} results for task {task_id}")
        return result.deleted_count

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _from_doc(doc: dict) -> StoredTaskResult:
        doc.pop("_id", None)
        return StoredTaskResult(**doc)
