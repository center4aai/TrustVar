# src/core/tasks/celery_app.py
from celery import Celery

from src.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "llm_framework",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "src.core.tasks.inference_task",
        "src.core.tasks.model_download_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 час
    task_soft_time_limit=3300,  # 55 минут
)
