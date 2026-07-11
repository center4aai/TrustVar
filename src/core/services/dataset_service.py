# src/core/services/dataset_service.py
import csv
import json
from io import StringIO
from typing import BinaryIO, List, Optional, Dict

from src.core.schemas.dataset import Dataset, DatasetItem
from src.core.taxonomy import normalize_task_type
from src.database.repositories.dataset_repository import DatasetRepository
from src.utils.logger import logger


class DatasetService:
    """Service for working with datasets"""

    def __init__(self):
        self.repository = DatasetRepository()

    async def create_dataset(
        self, name: str, description: str, task_type: str,
        task_semantics: Optional[str] = None, tags: List[str] = None
    ) -> Dataset:
        dataset = Dataset(
            name=name, description=description, task_type=task_type,
            task_semantics=task_semantics, tags=tags or []
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
        template_column: Optional[str] = None,
        variables_columns: Optional[List[str]] = None,
    ) -> int:
        """Upload data from file with preservation of all columns in metadata"""

        await self.repository.update(
            dataset_id,
            {
                "prompt_column": prompt_column,
                "target_column": target_column,
                "include_column": include_column,
                "exclude_column": exclude_column,
                "template_column": template_column,
                "variables_columns": variables_columns,
            },
        )

        if file_format == "jsonl":
            items = self._parse_jsonl(file)
        elif file_format == "json":
            items = await self._parse_json(file)
        elif file_format == "csv":
            items = self._parse_csv(file)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        if not items:
            return 0

        # WEB-1/WEB-2: the dataset-level task_type is already canonical, but rows
        # uploaded from JSONL/CSV may still carry a raw label (e.g.
        # "multi_label_classification") in their metadata. Scoring reads
        # result.metadata["task_type"] and would hit the silent fallback if left
        # raw. Normalize per-row metadata here so scoring and operator semantics
        # agree with the dataset declaration.
        dataset = await self.repository.find_by_id(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        canon_task_type = normalize_task_type(
            dataset.task_type, default=dataset.task_type
        )
        task_semantics = getattr(dataset, "task_semantics", None)

        for item in items:
            item["task_type"] = canon_task_type
            if task_semantics:
                item["task_semantics"] = task_semantics

        dataset_items = [
            DatasetItem.from_row(
                row=item,
                dataset_id=dataset_id,
                prompt_column=prompt_column,
                target_column=target_column,
                include_column=include_column,
                exclude_column=exclude_column,
                template_column=template_column,
                variables_columns=variables_columns,
            )
            for item in items
        ]

        count = await self.repository.add_items(dataset_id, dataset_items)

        logger.info(f"Successfully uploaded {count} items to dataset {dataset_id}")
        return count

    # ====================== Parsers ======================

    def _parse_jsonl(self, file: BinaryIO) -> List[Dict]:
        items = []
        for line in file:
            if line.strip():
                items.append(json.loads(line.decode("utf-8")))
        return items

    async def _parse_json(self, file: BinaryIO) -> List[Dict]:
        content = file.read()
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, list):
            return data
        return data.get("data") or data.get("items") or [data]

    def _parse_csv(self, file: BinaryIO) -> List[Dict]:
        file.seek(0)
        content = file.read().decode("utf-8")
        reader = csv.DictReader(StringIO(content))
        return [row for row in reader]

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset"""
        return await self.repository.find_by_id(dataset_id)

    async def list_datasets(self, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """List datasets"""
        return await self.repository.find_all(skip=skip, limit=limit)

    async def get_items(
        self, dataset_id: str, skip: int = 0, limit: int = 100
    ) -> List[DatasetItem]:
        """Get dataset items"""
        return await self.repository.get_items(dataset_id, skip, limit)

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset and all its items"""
        # First delete all dataset items
        deleted_items = await self.repository.delete_items(dataset_id)

        # Then delete the dataset itself
        result = await self.repository.delete(dataset_id)

        if result:
            logger.info(f"Dataset deleted: {dataset_id} ({deleted_items} items removed)")

        return result

    async def get_stats(self, dataset_id: str) -> dict:
        """Get dataset statistics"""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return {}

        items = await self.get_items(dataset_id, limit=10000)

        # Calculate statistics
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
