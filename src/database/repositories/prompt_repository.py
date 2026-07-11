from typing import List, Optional

from src.core.schemas.prompt import Prompt, PromptType
from src.database.mongodb import get_database
from .base import BaseRepository


class PromptRepository(BaseRepository[Prompt]):
    """Repository for working with prompts in MongoDB"""

    def __init__(self):
        self.collection_name = "prompts"

    async def _get_collection(self):
        db = await get_database()
        return db[self.collection_name]

    async def create(self, prompt: Prompt) -> Prompt:
        """Create prompt"""
        collection = await self._get_collection()
        doc = prompt.model_dump(mode="json")
        await collection.insert_one(doc)
        return prompt

    async def find_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Find prompt by ID"""
        collection = await self._get_collection()
        doc = await collection.find_one({"id": prompt_id})
        if doc:
            doc.pop("_id", None)
            return Prompt(**doc)
        return None

    async def find_all(
        self, prompt_type: Optional[PromptType] = None, skip: int = 0, limit: int = 100
    ) -> List[Prompt]:
        """Get all prompts, optionally filtered by type"""
        collection = await self._get_collection()
        filters = {}
        if prompt_type:
            filters["prompt_type"] = (
                prompt_type.value
                if isinstance(prompt_type, PromptType)
                else prompt_type
            )

        cursor = collection.find(filters).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        results = []
        for doc in docs:
            doc.pop("_id", None)
            results.append(Prompt(**doc))

        return results

    async def update(self, prompt_id: str, update_data: dict) -> Optional[Prompt]:
        """Update prompt"""
        collection = await self._get_collection()
        from datetime import datetime

        update_data["updated_at"] = datetime.now()

        result = await collection.update_one({"id": prompt_id}, {"$set": update_data})

        if result.modified_count > 0:
            return await self.find_by_id(prompt_id)
        return None

    async def delete(self, prompt_id: str) -> bool:
        """Delete prompt"""
        collection = await self._get_collection()
        result = await collection.delete_one({"id": prompt_id})
        return result.deleted_count > 0
