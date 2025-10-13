# src/database/repositories/dataset_repository.py
from typing import List, Optional

from src.utils.logger import logger

from src.core.models.dataset import Dataset, DatasetItem

from .base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    """Репозиторий для датасетов"""

    def __init__(self):
        super().__init__("datasets", Dataset)

    async def find_by_name(self, name: str) -> Optional[Dataset]:
        """Найти датасет по имени"""
        collection = await self._get_collection()
        doc = await collection.find_one({"name": name})
        if doc:
            doc.pop("_id", None)
            return Dataset(**doc)
        return None

    async def get_items(
        self, dataset_id: str, skip: int = 0, limit: int = 100
    ) -> List[DatasetItem]:
        """Получить элементы датасета"""
        collection = await self._get_collection()
        dataset_items_collection = collection.database["dataset_items"]

        cursor = (
            dataset_items_collection.find({"dataset_id": dataset_id})
            .skip(skip)
            .limit(limit)
        )

        items = await cursor.to_list(length=limit)
        return [DatasetItem(**item) for item in items]

    async def add_items(self, dataset_id: str, items: List[DatasetItem]):
        """Добавить элементы в датасет"""
        collection = await self._get_collection()
        dataset_items_collection = collection.database["dataset_items"]

        docs = [{**item.model_dump(), "dataset_id": dataset_id} for item in items]

        if docs:
            await dataset_items_collection.insert_many(docs)

            # Обновляем размер датасета
            await self.update(
                dataset_id,
                {
                    "size": await dataset_items_collection.count_documents(
                        {"dataset_id": dataset_id}
                    )
                },
            )

        logger.info(f"Added {len(items)} items to dataset {dataset_id}")
