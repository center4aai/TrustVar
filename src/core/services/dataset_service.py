# src/core/services/dataset_service.py
import csv
import io
import json
from typing import BinaryIO, List, Optional

from src.core.schemas.dataset import Dataset, DatasetItem
from src.database.repositories.dataset_repository import DatasetRepository
from src.utils.logger import logger


class DatasetService:
    """Сервис для работы с датасетами"""

    def __init__(self):
        self.repository = DatasetRepository()

    async def create_dataset(
        self, name: str, description: str, task_type: str, tags: List[str] = None
    ) -> Dataset:
        """Создать новый датасет"""
        dataset = Dataset(
            name=name, description=description, task_type=task_type, tags=tags or []
        )

        created = await self.repository.create(dataset)
        logger.info(f"Dataset created: {created.id}")

        return created

    async def upload_from_file(
        self,
        dataset_id: str,
        file: BinaryIO,
        file_format: str,
        prompt_column: str = "prompt",
        target_column: Optional[str] = None,
        include_column: Optional[str] = None,
        exclude_column: Optional[str] = None,
    ) -> int:
        """Загрузить данные из файла"""
        # Сохраняем конфигурацию столбцов в датасете
        await self.repository.update(
            dataset_id,
            {
                "prompt_column": prompt_column,
                "target_column": target_column,
                "include_column": include_column,
                "exclude_column": exclude_column,
            },
        )
        items = []

        if file_format == "jsonl":
            items = self._parse_jsonl(
                file, prompt_column, target_column, include_column, exclude_column
            )
        elif file_format == "json":
            items = self._parse_json(file)
        elif file_format == "csv":
            items = self._parse_csv(file)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Сохраняем items
        dataset_items = [
            DatasetItem.from_row(
                row=item,
                prompt_column=prompt_column,
                target_column=target_column,
                include_column=include_column,
                exclude_column=exclude_column,
            )
            for item in items
        ]
        await self.repository.add_items(dataset_id, dataset_items)

        logger.info(f"Uploaded {len(items)} items to dataset {dataset_id}")

        return len(items)

    def _parse_jsonl(self, file: BinaryIO, dataset: Dataset) -> List[dict]:
        items = []
        content = file.read().decode("utf-8")
        for line in content.strip().split("\n"):
            if line:
                raw = json.loads(line)
                item = {
                    "prompt": raw.get(dataset.prompt_column, ""),
                    "target": raw.get(dataset.target_column)
                    if dataset.target_column
                    else None,
                    "metadata": {},
                }

                if dataset.include_column and dataset.include_column in raw:
                    item["metadata"]["include_list"] = raw[dataset.include_column]

                if dataset.exclude_column and dataset.exclude_column in raw:
                    item["metadata"]["exclude_list"] = raw[dataset.exclude_column]

                items.append(item)
        return items

    def _parse_json(self, file: BinaryIO) -> List[dict]:
        """Парсинг JSON"""
        content = file.read().decode("utf-8")
        data = json.loads(content)

        # Если это массив, возвращаем его
        if isinstance(data, list):
            return data
        # Если это объект с ключом 'data' или 'items'
        elif isinstance(data, dict):
            return data.get("data", data.get("items", []))

        return []

    def _parse_csv(self, file: BinaryIO) -> List[dict]:
        """Парсинг CSV"""
        content = file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Получить датасет"""
        return await self.repository.find_by_id(dataset_id)

    async def list_datasets(self, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """Список датасетов"""
        return await self.repository.find_all(skip=skip, limit=limit)

    async def get_items(
        self, dataset_id: str, skip: int = 0, limit: int = 100
    ) -> List[DatasetItem]:
        """Получить элементы датасета"""
        return await self.repository.get_items(dataset_id, skip, limit)

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Удалить датасет"""
        result = await self.repository.delete(dataset_id)

        if result:
            # cache.clear_pattern(f"dataset:{dataset_id}*")
            logger.info(f"Dataset deleted: {dataset_id}")

        return result

    async def get_stats(self, dataset_id: str) -> dict:
        """Получить статистику по датасету"""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return {}

        items = await self.get_items(dataset_id, limit=10000)

        # Вычисляем статистику
        total_prompts = len(items)
        avg_prompt_length = (
            sum(len(item.prompt) for item in items) / total_prompts
            if total_prompts > 0
            else 0
        )

        has_expected = sum(1 for item in items if item.target)

        return {
            "total_items": total_prompts,
            "avg_prompt_length": avg_prompt_length,
            "items_with_target": has_expected,
            "coverage": (has_expected / total_prompts * 100)
            if total_prompts > 0
            else 0,
        }
