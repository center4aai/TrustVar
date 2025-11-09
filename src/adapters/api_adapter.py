# src/adapters/api_adapter.py
from typing import List

import aiohttp

from src.adapters.base import BaseLLMAdapter
from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()


class OpenAIAdapter(BaseLLMAdapter):
    """Адаптер для OpenAI API"""

    def __init__(self, model):
        super().__init__(model)
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL

    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерация через OpenAI API"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error = await response.text()
                        logger.error(f"OpenAI API error: {error}")
                        raise Exception(f"OpenAI API error: {response.status}")
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Пакетная генерация"""
        import asyncio

        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
