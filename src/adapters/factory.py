# src/adapters/factory.py
from src.adapters.base import BaseLLMAdapter
from src.config.constants import ModelProvider
from src.core.schemas.model import Model


class LLMFactory:

    @staticmethod
    def create(model: Model) -> BaseLLMAdapter:
        if model.provider == ModelProvider.OLLAMA:
            from src.adapters.ollama_adapter import OllamaAdapter
            return OllamaAdapter(model)

        if model.provider == ModelProvider.HUGGINGFACE:
            from src.adapters.hf_adapter import HuggingFaceAdapter
            return HuggingFaceAdapter(model)

        from src.adapters.api_adapter import OpenAIAdapter
        return OpenAIAdapter(model)
