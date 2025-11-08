# src/adapters/huggingface_adapter.py
import asyncio
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.adapters.base import BaseLLMAdapter
from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()


class HuggingFaceAdapter(BaseLLMAdapter):
    """Адаптер для HuggingFace моделей"""

    def __init__(self, model):
        super().__init__(model)
        self.tokenizer = None
        self.hf_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def _load_model(self):
        """Асинхронная загрузка модели"""
        if self.hf_model is None:
            logger.info(f"Loading HuggingFace model: {self.model.model_name}")

            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.model.model_name,
                cache_dir=settings.HF_CACHE_DIR,
                token=settings.HF_TOKEN if settings.HF_TOKEN else None,
            )

            self.hf_model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                self.model.model_name,
                cache_dir=settings.HF_CACHE_DIR,
                token=settings.HF_TOKEN if settings.HF_TOKEN else None,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            self.hf_model.to(self.device)

            logger.info(f"Model loaded on {self.device}")

    async def download_model(self):
        """Скачивает модель через Transformers в отдельном потоке"""
        logger.info(f"Downloading HuggingFace model: {self.model.model_name}")

        # Запускаем синхронные вызовы в отдельном потоке
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            self.model.model_name,
            cache_dir=settings.HF_CACHE_DIR,
            token=settings.HF_TOKEN if settings.HF_TOKEN else None,
        )

        hf_model = await asyncio.to_thread(
            AutoModelForCausalLM.from_pretrained,
            self.model.model_name,
            cache_dir=settings.HF_CACHE_DIR,
            token=settings.HF_TOKEN if settings.HF_TOKEN else None,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.tokenizer = tokenizer
        self.hf_model = hf_model.to(self.device)

        logger.info(
            f"Model {self.model.model_name} downloaded and loaded on {self.device}"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Генерация с HuggingFace моделью"""
        await self._load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                top_k=kwargs.get("top_k", self.config.top_k),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Пакетная генерация"""
        await self._load_model()

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(
                output[inputs.input_ids[i].shape[0] :], skip_special_tokens=True
            )
            results.append(generated_text.strip())

        return results

    async def health_check(self) -> bool:
        try:
            if self.hf_model is None or self.tokenizer is None:
                await self._load_model()
            return self.hf_model is not None and self.tokenizer is not None
        except Exception as e:
            logger.error(f"Health check failed during model loading: {e}")
            return False
