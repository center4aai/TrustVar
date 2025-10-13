# src/utils/cache.py
import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional

import redis

from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()


class RedisCache:
    """Простой кэш на основе Redis"""

    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL)

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Сохранить значение в кэш"""
        try:
            self.client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def delete(self, key: str):
        """Удалить значение из кэша"""
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    def clear_pattern(self, pattern: str):
        """Удалить все ключи по паттерну"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Глобальный экземпляр кэша
cache = RedisCache()


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Декоратор для кэширования результатов функций"""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Генерируем ключ кэша
            cache_key = f"{key_prefix}:{func.__name__}:{_hash_args(args, kwargs)}"

            # Проверяем кэш
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Вызываем функцию
            result = await func(*args, **kwargs)

            # Сохраняем в кэш
            cache.set(cache_key, result, ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{_hash_args(args, kwargs)}"

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        # Проверяем, асинхронная ли функция
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _hash_args(args, kwargs) -> str:
    """Хэширование аргументов для ключа кэша"""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()
