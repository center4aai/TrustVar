================================================================================
PROJECT STRUCTURE
================================================================================

```
Project: trustvar2

├── src
│   ├── adapters
│   │   ├── api_adapter.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── hf_adapter.py
│   │   └── ollama_adapter.py
│   ├── config
│   │   ├── constants.py
│   │   └── settings.py
│   ├── core
│   │   ├── schemas
│   │   │   ├── dataset.py
│   │   │   ├── model.py
│   │   │   └── task.py
│   │   ├── services
│   │   │   ├── dataset_service.py
│   │   │   ├── eval_service.py
│   │   │   ├── model_service.py
│   │   │   └── task_service.py
│   │   ├── tasks
│   │   │   ├── celery_app.py
│   │   │   └── inference_task.py
│   ├── database
│   │   ├── repositories
│   │   │   ├── base.py
│   │   │   ├── dataset_repository.py
│   │   │   ├── model_repository.py
│   │   │   └── task_repository.py
│   │   └── mongodb.py
│   ├── ui
│   │   ├── components
│   │   │   ├── dataset_uploader.py
│   │   │   ├── datasets_section.py
│   │   │   ├── models_section.py
│   │   │   ├── results_section.py
│   │   │   ├── task_monitor.py
│   │   │   └── tasks_section.py
│   │   └── app.py
│   └── utils
│       └── logger.py
├── Dockerfile.celery
├── Dockerfile.streamlit
├── README.md
├── docker-compose.yml
├── pyproject.toml
```
