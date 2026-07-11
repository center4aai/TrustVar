# src/api/routes/prompts.py
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.core.schemas.prompt import Prompt, PromptBase, PromptType
from src.database.repositories.prompt_repository import PromptRepository

router = APIRouter()

# Initialize repository
prompt_repo = PromptRepository()


class PromptUpdate(BaseModel):
    """Model for updating a prompt"""

    name: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    prompt_type: Optional[PromptType] = None


@router.post("/", response_model=Prompt, status_code=201)
async def create_prompt(prompt_data: PromptBase):
    """Create a new prompt"""
    try:
        prompt = Prompt(**prompt_data.model_dump())
        return await prompt_repo.create(prompt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[Prompt])
async def list_prompts(skip: int = 0, limit: int = 100):
    """List all prompts"""
    return await prompt_repo.find_all(skip=skip, limit=limit)


@router.get("/{prompt_id}", response_model=Prompt)
async def get_prompt(prompt_id: str):
    """Get a prompt by ID"""
    prompt = await prompt_repo.find_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.put("/{prompt_id}", response_model=Prompt)
async def update_prompt(prompt_id: str, update_data: PromptUpdate):
    """Update a prompt"""
    updated = await prompt_repo.update(
        prompt_id, {k: v for k, v in update_data.model_dump().items() if v is not None}
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return updated


@router.delete("/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: str):
    """Delete a prompt"""
    deleted = await prompt_repo.delete(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {}
