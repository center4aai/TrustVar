# src/adapters/api_adapter.py
import asyncio
import re
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

    async def generate(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Генерация через OpenAI API"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = (
            [{"role": "system", "content": kwargs.get("system_prompt")}]
            if kwargs.get("system_prompt")
            else []
        )

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = result["choices"][0]["message"]["content"]
                            clean_text = re.sub(r"<think>[\s\S]*?</think>", "", text)
                            return clean_text
                        else:
                            error = await response.text()
                            logger.error(f"OpenAI API error: {error}")
                            raise Exception(f"OpenAI API error: {response.status}")
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries}: Network error: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Пакетная генерация"""
        import asyncio

        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Генерация через OpenAI API"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": "Write '1' symbol"}]

        payload = {
            "model": self.model.model_name,
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 10,
            "top_p": 0.9,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return True

                    else:
                        error = await response.text()
                        logger.error(
                            f"OpenAI API error: {error}. Status: {response.status}"
                        )
                        return False
        except Exception as e:
            logger.warning(f"API Model is not available: {e}")
            return False
