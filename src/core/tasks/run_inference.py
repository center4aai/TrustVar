# # src/core/tasks/run_inference.py
# from celery import Celery
# from src.database.repositories.task_repository import TaskRepository

# from src.adapters import get_adapter

# celery_app = Celery("llm_tasks", broker="redis://localhost:6379/0")


# @celery_app.task(bind=True)
# def run_llm_inference(self, task_id: str):
#     """Выполняет инференс модели на датасете"""
#     task_repo = TaskRepository()
#     task = task_repo.find_by_id(task_id)

#     try:
#         # Обновляем статус
#         task_repo.update_status(task_id, "running")

#         # Получаем адаптер для модели
#         adapter = get_adapter(task.model_id)

#         # Загружаем датасет
#         dataset = get_dataset(task.dataset_id)

#         results = []
#         total = len(dataset)

#         for idx, item in enumerate(dataset):
#             # Генерируем ответ
#             response = adapter.generate(item["prompt"])
#             results.append({"input": item, "output": response})

#             # Обновляем прогресс
#             progress = (idx + 1) / total * 100
#             self.update_state(state="PROGRESS", meta={"progress": progress})

#         # Сохраняем результаты
#         task_repo.save_results(task_id, results)
#         task_repo.update_status(task_id, "completed")

#     except Exception as e:
#         task_repo.update_status(task_id, "failed", error=str(e))
#         raise
