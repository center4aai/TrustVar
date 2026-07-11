# src/adapters/base.py
from abc import ABC, abstractmethod
from typing import List

from src.core.schemas.model import Model


class BaseLLMAdapter(ABC):
    def __init__(self, model: Model):
        self.model = model
        self.config = model.config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass

    async def health_check(self) -> bool:
        pass

    async def delete_weights(self) -> bool:
        return True
