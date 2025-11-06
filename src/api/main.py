# src/api/main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.routes import datasets, models, tasks
from src.config.settings import get_settings
from src.database.mongodb import MongoDB


# Управляем жизненным циклом приложения (подключение к БД)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Действия при старте
    await MongoDB.connect()
    yield
    # Действия при выключении
    await MongoDB.close()


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

# Подключаем роутеры
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host=settings.API_IP, port=settings.API_PORT, reload=True
    )
