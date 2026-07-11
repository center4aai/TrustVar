# src/core/services/model_service.py
import traceback
from typing import List, Optional

from celery.result import AsyncResult

from src.config.constants import ModelProvider
from src.core.schemas.model import Model, ModelConfig, ModelStatus
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class ModelService:
    """Service for working with models"""

    def __init__(self):
        self.repository = ModelRepository()

    async def register_model(
        self,
        name: str,
        provider: str,
        model_name: str,
        description: str = None,
        config: ModelConfig = None,
    ) -> Model:
        """Register a model"""
        needs_download = provider in (ModelProvider.HUGGINGFACE, ModelProvider.OLLAMA)

        model = Model(
            name=name,
            provider=provider,
            model_name=model_name,
            description=description,
            config=config or ModelConfig(),
            status=ModelStatus.DOWNLOADING if needs_download else ModelStatus.REGISTERED,
        )

        created = await self.repository.create(model)

        # Auto download for HF and Ollama
        if (
            model.provider == ModelProvider.HUGGINGFACE
            or model.provider == ModelProvider.OLLAMA
        ):
            from src.core.tasks.model_download_task import run_download_model_task

            logger.info(f"Starting downloading model {model.name}")
            # Start Celery task
            celery_task = run_download_model_task.delay(created.id)

            # Save Celery task ID
            await self.repository.update(created.id, {"celery_task_id": celery_task.id})

        logger.info(f"Model registered: {created.id} - {created.name}")

        return created

    async def get_model(self, model_id: str) -> Optional[Model]:
        """Get model"""
        return await self.repository.find_by_id(model_id)

    async def list_models(self, active_only: bool = False) -> List[Model]:
        """List models"""
        if active_only:
            return await self.repository.find_active()
        return await self.repository.find_all()

    async def update_model(self, model_id: str, **kwargs) -> Optional[Model]:
        """Update model"""
        result = await self.repository.update(model_id, kwargs)

        if result:
            logger.info(f"Model updated: {model_id}")

        return result

    async def delete_model(self, model_id: str) -> bool:
        """Delete model with weights"""
        model = await self.repository.find_by_id(model_id)
        if not model:
            logger.warning(f"Model not found: {model_id}")
            return False

        if model.provider in (ModelProvider.OLLAMA, ModelProvider.HUGGINGFACE):
            # Delete weights via adapter
            try:
                from src.adapters.factory import LLMFactory

                adapter = LLMFactory.create(model)
                weights_deleted = await adapter.delete_weights()
                if not weights_deleted:
                    logger.warning(
                        f"Failed to delete weights for model {model_id}, "
                        f"proceeding with DB deletion"
                    )
            except Exception as e:
                logger.error(
                    f"Error deleting weights for {model_id}: {e}. Traceback: {traceback.format_exc()}"
                )

        # Delete from DB anyway
        result = await self.repository.delete(model_id)

        if result:
            logger.info(f"Model deleted: {model_id}")

        return result

    def test_model(
        self, model_id: str, test_prompt: str = "Hello, how are you?"
    ) -> dict:
        """
        Test model via Celery (asynchronously)

        Returns:
            Dict with celery_task_id for tracking
        """
        from src.core.tasks.health_check_task import run_test_inference_task

        logger.info(f"Scheduling test inference for model {model_id}")

        # Start Celery task
        celery_task = run_test_inference_task.delay(model_id, test_prompt)

        return {
            "celery_task_id": celery_task.id,
            "status": "scheduled",
            "message": "Test inference task has been scheduled",
        }

    def get_test_result(self, celery_task_id: str) -> dict:
        """
        Get test inference result

        Args:
            celery_task_id: Celery task ID

        Returns:
            Dict with results or status
        """
        from src.core.tasks.celery_app import celery_app

        result = AsyncResult(celery_task_id, app=celery_app)

        if result.ready():
            if result.successful():
                return {"status": "completed", "result": result.result}
            else:
                return {"status": "failed", "error": str(result.result)}
        else:
            return {"status": "pending", "state": result.state}

    def health_check(self, model_id: str) -> dict:
        """
        Check model availability via Celery (asynchronously)

        Returns:
            Dict with celery_task_id for tracking
        """
        from src.core.tasks.health_check_task import run_health_check_task

        logger.info(f"Scheduling health check for model {model_id}")

        # Start Celery task
        celery_task = run_health_check_task.delay(model_id)

        return {
            "celery_task_id": celery_task.id,
            "status": "scheduled",
            "message": "Health check task has been scheduled",
        }

    async def list_available_models(self) -> list[dict]:
        from src.adapters.ollama_adapter import list_local_models
        return await list_local_models()

    async def bulk_register_models(
        self,
        model_names: list[str],
        config: ModelConfig,
        description: str | None = None,
        provider: ModelProvider = ModelProvider.OLLAMA,
        pull_if_missing: bool = True,
    ) -> dict:
        if not model_names:
            return {"created": [], "skipped": [], "downloading": []}

        existing = await self.repository.find_all(
            {"provider": provider.value}, limit=1000
        )
        existing_names = {m.model_name for m in existing}

        ollama_names: set[str] = set()
        if provider == ModelProvider.OLLAMA:
            from src.adapters.ollama_adapter import list_local_models
            ollama_models = await list_local_models()
            ollama_names = {m["name"] for m in ollama_models}

        created = []
        skipped = []
        downloading = []

        for name in model_names:
            if name in existing_names:
                skipped.append(name)
                continue

            needs_pull = (
                provider == ModelProvider.OLLAMA
                and name not in ollama_names
                and pull_if_missing
            )

            model = Model(
                name=name,
                provider=provider,
                model_name=name,
                description=description,
                config=config,
                status=ModelStatus.DOWNLOADING if needs_pull else ModelStatus.REGISTERED,
            )
            await self.repository.create(model)
            created.append(model)
            existing_names.add(name)

            if needs_pull:
                from src.core.tasks.model_download_task import run_download_model_task
                celery_task = run_download_model_task.delay(model.id)
                await self.repository.update(model.id, {"celery_task_id": celery_task.id})
                downloading.append(name)

        return {"created": created, "skipped": skipped, "downloading": downloading}

    def get_health_check_result(self, celery_task_id: str) -> dict:
        """
        Get health check result

        Args:
            celery_task_id: Celery task ID

        Returns:
            Dict with results or status
        """
        from src.core.tasks.celery_app import celery_app

        result = AsyncResult(celery_task_id, app=celery_app)

        if result.ready():
            if result.successful():
                return {"status": "completed", "result": result.result}
            else:
                return {"status": "failed", "error": str(result.result)}
        else:
            return {"status": "pending", "state": result.state}
