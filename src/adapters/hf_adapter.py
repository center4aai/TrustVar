# src/adapters/huggingface_adapter.py
import asyncio
import re
import traceback
from typing import List
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.adapters.base import BaseLLMAdapter
from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()


class HuggingFaceAdapter(BaseLLMAdapter):

    def __init__(self, model):
        super().__init__(model)
        self.tokenizer = None
        self.hf_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def _load_model(self):
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

    async def download_model(self) -> bool:
        logger.info(f"Downloading HuggingFace model: {self.model.model_name}")

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

        if self.tokenizer and self.hf_model:
            logger.info(
                f"Model {self.model.model_name} downloaded and loaded on {self.device}"
            )
            return True
        else:
            logger.error(
                f"Downloading Error: Model {self.model.model_name} was NOT downloaded and loaded"
            )
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with HuggingFace model"""
        await self._load_model()

        message = kwargs.get("system_prompt", "")

        if message:
            message = message + "\n" + prompt
        else:
            message = prompt

        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)

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
        clean_text = re.sub(r"<think>[\s\S]*?</think>", "", generated_text)

        return clean_text.strip()

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation"""
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
            # Remove think tags — same as in generate()
            clean_text = re.sub(r"<think>[\s\S]*?</think>", "", generated_text)
            results.append(clean_text.strip())

        return results

    async def health_check(self) -> bool:
        try:
            if self.hf_model is None or self.tokenizer is None:
                await self._load_model()
            return self.hf_model is not None and self.tokenizer is not None
        except Exception as e:
            logger.error(
                f"Health check failed during model loading: {e}. Traceback: {traceback.format_exc()}"
            )
            return False

    async def delete_weights(self) -> bool:
        """Delete loaded HuggingFace model weights"""
        try:
            # HuggingFace cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            # HF stores models as models--org--model_name
            model_dir_name = "models--" + self.model.model_name.replace("/", "--")
            model_path = cache_dir / model_dir_name

            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"HF weights deleted: {model_path}")
                return True
            else:
                logger.warning(f"HF weights not found: {model_path}")
                return True  # Nothing to delete — not an error
        except Exception as e:
            logger.error(
                f"Failed to delete HF weights: {e}. Traceback: {traceback.format_exc()}"
            )
            return False
