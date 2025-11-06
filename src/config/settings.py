# src/config/settings.py
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "LLM Testing Framework"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Api
    API_IP: str = "0.0.0.0"
    API_PORT: int = 8000

    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "llm_framework"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # HuggingFace
    HF_TOKEN: str = ""
    HF_CACHE_DIR: str = "./cache/huggingface"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Anthropic
    ANTHROPIC_API_KEY: str = ""

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
