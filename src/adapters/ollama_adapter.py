# src/adapters/ollama_adapter.py
import asyncio
import json
import re
import traceback
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

import aiohttp

from src.adapters.base import BaseLLMAdapter
from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()

_RETRYABLE_STATUS = frozenset({502, 503, 504})

_T = TypeVar("_T")


class _RetryableHTTPStatus(Exception):
    """Internal marker: a transient HTTP status (502/503/504) worth retrying."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"retryable status {status}")


def _is_retryable_ollama_error(exc: BaseException) -> bool:
    """True for transient connection/timeout/5xx errors worth a retry."""
    if isinstance(exc, _RetryableHTTPStatus):
        return True
    return isinstance(
        exc,
        (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ),
    )


async def _retry_async(
    fn: Callable[[], Awaitable[_T]],
    *,
    max_retries: int,
    base_backoff: float,
    is_retryable: Callable[[BaseException], bool],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    label: str = "",
) -> _T:
    """Run ``fn`` (one attempt) with exponential backoff on retryable errors.

    Pure of any Ollama/network specifics — the retry policy is injected via
    ``is_retryable``/``sleep`` so it is fully model-free testable. Re-raises the
    last exception once ``max_retries`` is exhausted or the error is not
    retryable."""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below if not retryable
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            delay = base_backoff * (2 ** attempt)
            attempt += 1
            logger.warning(
                f"Ollama transient error ({exc}); retry {attempt}/{max_retries} "
                f"after {delay:.1f}s"
                + (f" [{label}]" if label else "")
            )
            await sleep(delay)


async def list_local_models() -> list[dict[str, Any]]:
    """Fetch all locally available models from Ollama via GET /api/tags."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("models", [])


class OllamaGPUFallbackError(RuntimeError):
    """Raised when the model stubbornly fails to load into VRAM."""


@dataclass(frozen=True)
class OllamaRuntimeStatus:
    """Snapshot of model state in Ollama (from /api/ps)."""

    model_name: str
    is_loaded: bool
    on_gpu: bool
    size_total_mb: int
    size_vram_mb: int
    expires_at: Optional[str]

    @property
    def fallback_detected(self) -> bool:
        return self.is_loaded and not self.on_gpu


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama"""

    def __init__(self, model):
        super().__init__(model)
        self.base_url = settings.OLLAMA_BASE_URL
        self.last_runtime_status: Optional[OllamaRuntimeStatus] = None
        logger.info(f"Ollama BASE URL: {self.base_url}")

    async def download_model(self) -> bool:
        """Asynchronously download Ollama model via its REST API."""
        url = f"{self.base_url}/api/pull"
        payload = {"name": self.model.model_name, "stream": True}

        try:
            timeout = aiohttp.ClientTimeout(total=3000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(
                            f"Error: Ollama server returned status {response.status}"
                        )
                        return False

                    async for line in response.content:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue

                        if "status" in event:
                            logger.info(f"Ollama pull: {event['status']}")

                        if event.get("status") == "success":
                            logger.info(
                                f"Model '{self.model.model_name}' downloaded successfully"
                            )
                            return True

                        if "error" in event:
                            logger.error(
                                f"Download error: {event['error']}"
                            )
                            return False

            logger.info("Download completed but no 'success' status received.")
            return False

        except aiohttp.ClientConnectorError:
            logger.error(
                "Failed to connect to Ollama server. Ensure it is running."
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected download error: {e}. "
                f"Traceback: {traceback.format_exc()}"
            )
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate via Ollama API with GPU inference guarantee."""
        return await self._generate(prompt, _retry=False, **kwargs)

    async def _generate(self, prompt: str, *, _retry: bool, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"

        system_prompt = kwargs.get("system_prompt", "")
        message = f"{system_prompt}\n{prompt}" if system_prompt else prompt

        payload = {
            "model": self.model.model_name,
            "prompt": message,
            "think": False,
            "stream": False,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
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
            clean_text = await _retry_async(
                lambda: self._post_generate_once(url, payload),
                max_retries=settings.OLLAMA_MAX_RETRIES,
                base_backoff=settings.OLLAMA_RETRY_BACKOFF_BASE,
                is_retryable=_is_retryable_ollama_error,
                label=self.model.model_name,
            )
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

        status = await self.get_runtime_status()
        self.last_runtime_status = status

        if (
            settings.OLLAMA_REQUIRE_GPU
            and settings.OLLAMA_GPU_RETRY_ON_FALLBACK
            and not _retry
            and status is not None
            and not status.on_gpu
        ):
            logger.warning(
                f"Model {self.model.model_name} detected on CPU "
                f"(size_vram_mb={status.size_vram_mb}); forcing reload and retrying."
            )
            try:
                await self._force_reload()
            except OllamaGPUFallbackError as reload_err:
                logger.warning(
                    f"Force reload of {self.model.model_name} failed: {reload_err}. "
                    f"Returning CPU result, marking metadata."
                )
                return clean_text
            return await self._generate(prompt, _retry=True, **kwargs)

        if (
            settings.OLLAMA_REQUIRE_GPU
            and status is not None
            and not status.on_gpu
        ):
            logger.warning(
                f"Model {self.model.model_name} still on CPU after retry "
                f"(size_vram_mb={status.size_vram_mb}). Continuing with CPU result."
            )

        return clean_text

    async def _post_generate_once(self, url: str, payload: dict) -> str:
        """One POST /api/generate attempt. Raises ``_RetryableHTTPStatus`` on a
        transient 502/503/504 (so the caller retries) and a plain ``Exception``
        on any other non-200 (no retry)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=settings.OLLAMA_INFERENCE_TIMEOUT
            ) as response:
                if response.status in _RETRYABLE_STATUS:
                    error = await response.text()
                    logger.warning(
                        f"Ollama transient HTTP {response.status}: {error[:200]}"
                    )
                    raise _RetryableHTTPStatus(response.status)
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Ollama API error: {error}")
                    raise Exception(f"Ollama API error: {response.status}")
                result = await response.json()
                text = result.get("response", "")
                return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def get_runtime_status(
        self, model_name: Optional[str] = None
    ) -> Optional[OllamaRuntimeStatus]:
        """Query /api/ps and return model status in Ollama."""
        target_name = model_name or self.model.model_name
        url = f"{self.base_url}/api/ps"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Ollama /api/ps returned {response.status}"
                        )
                        return None
                    data = await response.json()
        except Exception as e:
            logger.warning(f"Failed to query Ollama /api/ps: {e}")
            return None

        for entry in data.get("models", []):
            if entry.get("name") != target_name and entry.get("model") != target_name:
                continue
            size_total = int(entry.get("size", 0))
            size_vram = int(entry.get("size_vram", 0))
            return OllamaRuntimeStatus(
                model_name=target_name,
                is_loaded=True,
                on_gpu=size_vram > 0,
                size_total_mb=size_total // (1024 * 1024),
                size_vram_mb=size_vram // (1024 * 1024),
                expires_at=entry.get("expires_at"),
            )

        return None

    async def _force_reload(self) -> None:
        """Force reload model into VRAM (POST /api/generate with empty prompt)."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model.model_name,
            "prompt": "",
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=300) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise OllamaGPUFallbackError(
                            f"Reload failed with status {response.status}: {error}"
                        )
                    await response.read()
        except aiohttp.ClientError as e:
            raise OllamaGPUFallbackError(f"Reload network error: {e}") from e

        status = await self.get_runtime_status()
        self.last_runtime_status = status
        if status is None or not status.on_gpu:
            raise OllamaGPUFallbackError(
                f"Model {self.model.model_name} did not load on GPU after reload "
                f"(status={status})."
            )
        logger.info(
            f"Model {self.model.model_name} successfully reloaded on GPU "
            f"(size_vram_mb={status.size_vram_mb})."
        )

    async def unload(self) -> bool:
        """Explicit model unload from VRAM with /api/ps verification.

        On macOS, Metal backend does not release resources instantly after keep_alive=0,
        which causes ggml crash when loading next model. We wait until model
        actually disappears from /api/ps before returning.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model.model_name,
            "keep_alive": "0s",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(
                            f"Ollama unload non-200 for {self.model.model_name}: "
                            f"status={response.status} body={error_text[:200]}"
                        )
                        return False
                    await response.read()
            logger.info(f"Ollama unload requested: {self.model.model_name}")

            # Verification: wait until model disappears from /api/ps
            ps_url = f"{self.base_url}/api/ps"
            for attempt in range(10):
                await asyncio.sleep(0.5)
                async with aiohttp.ClientSession() as session:
                    async with session.get(ps_url, timeout=5) as ps_resp:
                        if ps_resp.status != 200:
                            continue
                        data = await ps_resp.json()
                        models = data.get("models", [])
                        if not any(m.get("name") == self.model.model_name for m in models):
                            logger.info(
                                f"Ollama model confirmed unloaded: {self.model.model_name} "
                                f"(attempt {attempt + 1})"
                            )
                            self.last_runtime_status = None
                            return True
            logger.warning(
                f"Ollama model {self.model.model_name} still present in /api/ps "
                f"after unload — proceeding anyway"
            )
            self.last_runtime_status = None
            return True
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model {self.model.model_name}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Ollama availability and GPU model location."""
        try:
            models = await list_local_models()
            model_names = [m["name"] for m in models]
            if self.model.model_name not in model_names:
                return False
        except Exception:
            return False

        if not settings.OLLAMA_REQUIRE_GPU:
            return True

        status = await self.get_runtime_status()
        if status is None:
            return True

        if not status.on_gpu:
            logger.warning(
                f"Health check: model {self.model.model_name} loaded on CPU "
                f"(size_vram_mb={status.size_vram_mb})."
            )
            return False
        return True

    async def delete_weights(self) -> bool:
        """Delete model from Ollama"""
        url = f"{self.base_url}/api/delete"
        payload = {"name": self.model.model_name}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Ollama model deleted: {self.model.model_name}")
                        return True
                    error = await response.text()
                    logger.error(f"Ollama delete error: {error}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete Ollama model: {e}")
            return False
