# src/adapters/ollama_adapter.py
from typing import List

import aiohttp
from src.utils.logger import logger

from src.adapters.base import BaseLLMAdapter
from src.config.settings import get_settings

settings = get_settings()


class OllamaAdapter(BaseLLMAdapter):
    """Адаптер для Ollama"""

    def __init__(self, model):
        super().__init__(model)
        self.base_url = settings.OLLAMA_BASE_URL

    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерация через Ollama API"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "repeat_penalty": self.config.repeat_penalty,
                "stop": self.config.stop_sequences,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error = await response.text()
                        logger.error(f"Ollama API error: {error}")
                        raise Exception(f"Ollama API error: {response.status}")
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Пакетная генерация"""
        import asyncio

        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Проверка доступности Ollama"""
        try:
            url = f"{self.base_url}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return self.model.model_name in models
            return False
        except:
            return False
