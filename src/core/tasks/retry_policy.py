# src/core/tasks/retry_policy.py

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Type

from src.utils.logger import logger


class RetryStrategy(str, Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [ConnectionError, TimeoutError, OSError]
    )
    non_retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [ValueError, PermissionError]
    )

    def calculate_delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay_seconds * (attempt + 1)
        else:
            delay = self.base_delay_seconds * (2**attempt)

        delay = min(delay, self.max_delay_seconds)

        if self.jitter and delay > 0:
            delay *= 0.5 + random.random() * 0.5

        return delay

    def is_retryable(self, exc: Exception) -> bool:
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exc, exc_type):
                return False
        for exc_type in self.retryable_exceptions:
            if isinstance(exc, exc_type):
                return True
        return True


OLLAMA_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay_seconds=2.0,
    max_delay_seconds=30.0,
)

API_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay_seconds=1.0,
    max_delay_seconds=120.0,
)

JUDGE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    base_delay_seconds=2.0,
    max_delay_seconds=20.0,
)

WRITE_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay_seconds=1.0,
    max_delay_seconds=30.0,
    retryable_exceptions=[ConnectionError, TimeoutError, OSError, Exception],
    non_retryable_exceptions=[],
)


async def with_retry(
    func: Callable,
    policy: RetryPolicy,
    *args,
    context: str = "",
    **kwargs,
) -> Any:
    last_exception = None

    for attempt in range(policy.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc

            if not policy.is_retryable(exc):
                logger.warning(
                    f"Non-retryable error [{context}] "
                    f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                )
                raise

            if attempt == policy.max_attempts - 1:
                break

            delay = policy.calculate_delay(attempt)
            logger.warning(
                f"Retryable error [{context}] "
                f"attempt {attempt + 1}/{policy.max_attempts}: "
                f"{type(exc).__name__}: {exc}. "
                f"Retry in {delay:.1f}s..."
            )
            if delay > 0:
                await asyncio.sleep(delay)

    logger.error(
        f"All {policy.max_attempts} attempts failed [{context}]: "
        f"{type(last_exception).__name__}: {last_exception}"
    )
    raise last_exception
