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
        "src.core.tasks.health_check_task",  # ADDED
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=144000,  # 40 hours
    task_soft_time_limit=142000,  # ~40 hours
    # Additional result settings
    result_expires=142000,  # Results stored for 1 hour
    result_extended=True,  # Extended result information
)
