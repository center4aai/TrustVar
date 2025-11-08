# src/ui/api_client.py
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from requests.exceptions import RequestException

from src.config.settings import get_settings
from src.core.schemas.dataset import Dataset, DatasetItem
from src.core.schemas.model import Model
from src.core.schemas.task import Task, TaskStatus

settings = get_settings()


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            st.error(
                "API_BASE_URL is not set. Please configure it in your environment."
            )

    def _handle_response(self, response: requests.Response):
        """Обрабатывает ответ от API и вызывает исключение в случае ошибки."""
        # print(response.json())
        if not response.ok:
            try:
                detail = response.json().get("detail", "Unknown API error")
            except requests.JSONDecodeError:
                detail = response.text
            st.error(f"API Error ({response.status_code}): {detail}")
            raise RequestException(f"API Error: {response.status_code}")
        if response.status_code == 204:  # No Content
            return None
        return response.json()

    # --- Datasets ---
    def list_datasets(self) -> List[Dataset]:
        response = requests.get(f"{self.base_url}/api/v1/datasets/")
        data = self._handle_response(response)
        return [Dataset(**item) for item in data]

    def create_dataset_and_upload(
        self, name, description, task_type, tags, file, file_format
    ) -> Dict[str, Any]:
        # 1. Создаем запись о датасете
        dataset_data = {
            "name": name,
            "description": description,
            "task_type": task_type,
            "tags": tags,
        }
        create_resp = requests.post(
            f"{self.base_url}/api/v1/datasets/", data=dataset_data
        )
        dataset_info = self._handle_response(create_resp)
        dataset_id = dataset_info["id"]

        # 2. Загружаем файл
        files = {"file": (file.name, file, file.type)}
        upload_data = {"file_format": file_format}
        upload_resp = requests.post(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/upload",
            files=files,
            data=upload_data,
        )
        return self._handle_response(upload_resp)

    def get_dataset(self, dataset_id: str) -> Dataset:
        response = requests.get(f"{self.base_url}/api/v1/datasets/{dataset_id}")
        data = self._handle_response(response)
        return Dataset(**data)

    def get_dataset_items(
        self, dataset_id: str, skip: int = 0, limit: int = 10
    ) -> List[DatasetItem]:
        response = requests.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/items",
            params={"skip": skip, "limit": limit},
        )
        data = self._handle_response(response)
        return [DatasetItem(**item) for item in data]

    def get_dataset_stats(self, dataset_id: str) -> Dict:
        response = requests.get(f"{self.base_url}/api/v1/datasets/{dataset_id}/stats")
        return self._handle_response(response)

    def delete_dataset(self, dataset_id: str):
        response = requests.delete(f"{self.base_url}/api/v1/datasets/{dataset_id}")
        self._handle_response(response)

    # --- Models ---
    def list_models(self) -> List[Model]:
        response = requests.get(f"{self.base_url}/api/v1/models/")
        data = self._handle_response(response)
        return [Model(**item) for item in data]

    def register_model(self, model_data: Dict) -> Model:
        response = requests.post(f"{self.base_url}/api/v1/models/", json=model_data)
        data = self._handle_response(response)
        return Model(**data)

    def delete_model(self, model_id: str):
        response = requests.delete(f"{self.base_url}/api/v1/models/{model_id}")
        self._handle_response(response)

    def get_model(self, model_id: str):
        model_data = {"model_id": model_id}
        response = requests.get(
            f"{self.base_url}/api/v1/models/{model_id}/get", json=model_data
        )
        data = self._handle_response(response)
        return Model(**data)

    def test_model(self, model_id: str, test_prompt: str) -> Dict:
        """
        Запустить асинхронный тест модели через Celery

        Returns:
            Dict с celery_task_id для отслеживания
        """
        test_data = {"test_prompt": test_prompt}
        response = requests.post(
            f"{self.base_url}/api/v1/models/{model_id}/test", json=test_data
        )
        return self._handle_response(response)

    def get_test_result(self, model_id: str, celery_task_id: str) -> Dict:
        """
        Получить результат асинхронного теста модели

        Returns:
            Dict со статусом и результатом
        """
        response = requests.get(
            f"{self.base_url}/api/v1/models/{model_id}/test/{celery_task_id}"
        )
        return self._handle_response(response)

    def health_check_async(self, model_id: str) -> Dict:
        """
        Запустить асинхронную проверку здоровья модели

        Returns:
            Dict с celery_task_id для отслеживания
        """
        response = requests.post(f"{self.base_url}/api/v1/models/{model_id}/health")
        return self._handle_response(response)

    def get_health_check_result(self, model_id: str, celery_task_id: str) -> Dict:
        """
        Получить результат асинхронной проверки здоровья

        Returns:
            Dict со статусом и результатом
        """
        response = requests.get(
            f"{self.base_url}/api/v1/models/{model_id}/health/{celery_task_id}"
        )
        return self._handle_response(response)

    def compare_results(self, task_id) -> Dict:
        """
        Сравнение результатов моделей

        Returns:
            Dict с результатами моделей
        """
        response = requests.get(
            f"{self.base_url}/api/v1/tasks/{task_id}/compare-models"
        )
        return self._handle_response(response)

    # --- Tasks ---
    def list_tasks(
        self, status: Optional[TaskStatus] = None, skip: int = 0, limit: int = 100
    ) -> List[Task]:
        task_data = {"status": status, "skip": skip, "limit": limit}
        response = requests.get(f"{self.base_url}/api/v1/tasks/", json=task_data)

        data = self._handle_response(response)
        return [Task(**item) for item in data]

    def create_task(self, task_data: Dict) -> Task:
        response = requests.post(f"{self.base_url}/api/v1/tasks/", json=task_data)
        data = self._handle_response(response)
        return Task(**data)

    def get_task(self, task_id: str) -> Task:
        response = requests.get(f"{self.base_url}/api/v1/tasks/{task_id}")
        data = self._handle_response(response)
        return Task(**data)

    def cancel_task(self, task_id: str):
        response = requests.post(f"{self.base_url}/api/v1/tasks/{task_id}/cancel")
        self._handle_response(response)


# Функция для получения синглтона клиента
@st.cache_resource
def get_api_client():
    return ApiClient(
        base_url=os.getenv("API_BASE_URL", "http://localhost:8000")
    )  # settings.API_IP + settings.API_PORT
