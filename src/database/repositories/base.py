# src/database/repositories/base.py
# CHANGE: update() now always returns the document,
# even if data didn't change (upsert-like behavior)

from typing import Any, Dict, Generic, List, Optional, TypeVar

from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

from src.database.mongodb import get_database

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """Base repository for MongoDB operations"""

    def __init__(self, collection_name: str, model_class: type[T]):
        self.collection_name = collection_name
        self.model_class = model_class
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            db = await get_database()
            self._collection = db[self.collection_name]
        return self._collection

    async def create(self, obj: T) -> T:
        collection = await self._get_collection()
        doc = obj.model_dump(mode="json")
        await collection.insert_one(doc)
        return obj

    async def find_by_id(self, id: str) -> Optional[T]:
        collection = await self._get_collection()
        doc = await collection.find_one({"id": id})
        if doc:
            doc.pop("_id", None)
            return self.model_class(**doc)
        return None

    async def find_all(
        self,
        filters: Dict[str, Any] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[T]:
        collection = await self._get_collection()
        cursor = (
            collection.find(filters or {})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        results = []
        for doc in docs:
            doc.pop("_id", None)
            results.append(self.model_class(**doc))
        return results

    async def update(self, id: str, update_data: Dict[str, Any]) -> Optional[T]:
        """
        Update document.

        Difference from original: always returns document after update,
        even if modified_count == 0 (data didn't change).

        This is important for progress tracker and recovery — they may write
        the same values repeatedly (idempotent operations).
        """
        collection = await self._get_collection()
        await collection.update_one({"id": id}, {"$set": update_data})
        # Don't check modified_count — document may not have changed,
        # but we still need to return current state
        return await self.find_by_id(id)

    async def delete(self, id: str) -> bool:
        collection = await self._get_collection()
        result = await collection.delete_one({"id": id})
        return result.deleted_count > 0

    async def count(self, filters: Dict[str, Any] = None) -> int:
        collection = await self._get_collection()
        return await collection.count_documents(filters or {})
