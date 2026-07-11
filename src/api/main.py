# src/api/main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import datasets, models, prompts, tasks
from src.config.default_prompts import seed_default_prompts
from src.config.settings import get_settings
from src.database.mongodb import MongoDB
from src.utils.logger import logger


# Manage application lifecycle (DB connection)
@asynccontextmanager
async def lifespan(app: FastAPI):
    await MongoDB.connect()
    await MongoDB.ensure_indexes()
    try:
        await seed_default_prompts()
    except Exception as e:
        logger.error(f"Default prompt seeding failed (non-fatal): {e}")
    yield
    await MongoDB.close()


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS - first thing!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["Prompts"])


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host=settings.API_IP, port=settings.API_PORT, reload=False
    )
