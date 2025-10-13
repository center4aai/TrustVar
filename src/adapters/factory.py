# src/adapters/factory.py
from src.adapters.api_adapter import AnthropicAdapter, OpenAIAdapter
from src.adapters.base import BaseLLMAdapter
from src.adapters.hf_adapter import HuggingFaceAdapter
from src.adapters.ollama_adapter import OllamaAdapter
from src.config.constants import ModelProvider
from src.core.models.model import Model

# def get_adapter(model: Model) -> BaseLLMAdapter:
#     """Фабрика адаптеров"""
#     adapters = {
#         ModelProvider.OLLAMA: OllamaAdapter,
#         ModelProvider.HUGGINGFACE: HuggingFaceAdapter,
#         ModelProvider.OPENAI: OpenAIAdapter,
#         ModelProvider.ANTHROPIC: AnthropicAdapter,
#     }

#     adapter_class = adapters.get(model.provider)
#     if not adapter_class:
#         raise ValueError(f"Unknown provider: {model.provider}")

#     return adapter_class(model)


class LLMFactory:
    """Фабрика адаптеров"""

    @staticmethod
    def create(model: Model) -> BaseLLMAdapter:
        adapters = {
            ModelProvider.OLLAMA: OllamaAdapter,
            ModelProvider.HUGGINGFACE: HuggingFaceAdapter,
            ModelProvider.OPENAI: OpenAIAdapter,
            ModelProvider.ANTHROPIC: AnthropicAdapter,
        }

        adapter_class = adapters.get(model.provider)
        if not adapter_class:
            raise ValueError(f"Unknown provider: {model.provider}")

        return adapter_class(model)
