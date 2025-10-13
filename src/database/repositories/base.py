# src/database/repositories/base.py
from typing import Any, Dict, Generic, List, Optional, TypeVar

from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

from src.database.mongodb import get_database

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """Базовый репозиторий для работы с MongoDB"""

    def __init__(self, collection_name: str, model_class: type[T]):
        self.collection_name = collection_name
        self.model_class = model_class
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Получить коллекцию"""
        if self._collection is None:
            db = await get_database()
            self._collection = db[self.collection_name]
        return self._collection

    async def create(self, obj: T) -> T:
        """Создать документ"""
        collection = await self._get_collection()
        doc = obj.model_dump(mode="json")
        await collection.insert_one(doc)
        return obj

    async def find_by_id(self, id: str) -> Optional[T]:
        """Найти по ID"""
        collection = await self._get_collection()
        doc = await collection.find_one({"id": id})
        if doc:
            doc.pop("_id", None)
            return self.model_class(**doc)
        return None

    async def find_all(
        self, filters: Dict[str, Any] = None, skip: int = 0, limit: int = 100
    ) -> List[T]:
        """Найти все документы"""
        collection = await self._get_collection()
        filters = filters or {}

        cursor = collection.find(filters).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        results = []
        for doc in docs:
            doc.pop("_id", None)
            results.append(self.model_class(**doc))

        return results

    async def update(self, id: str, update_data: Dict[str, Any]) -> Optional[T]:
        """Обновить документ"""
        collection = await self._get_collection()

        result = await collection.update_one({"id": id}, {"$set": update_data})

        if result.modified_count > 0:
            return await self.find_by_id(id)
        return None

    async def delete(self, id: str) -> bool:
        """Удалить документ"""
        collection = await self._get_collection()
        result = await collection.delete_one({"id": id})
        return result.deleted_count > 0

    async def count(self, filters: Dict[str, Any] = None) -> int:
        """Подсчитать количество документов"""
        collection = await self._get_collection()
        filters = filters or {}
        return await collection.count_documents(filters)
