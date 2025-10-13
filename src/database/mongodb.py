# src/database/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from src.utils.logger import logger

from src.config.settings import get_settings

settings = get_settings()


class MongoDB:
    """Singleton для MongoDB соединения"""

    _client: AsyncIOMotorClient = None
    _db: AsyncIOMotorDatabase = None

    @classmethod
    async def connect(cls):
        """Подключение к MongoDB"""
        if cls._client is None:
            logger.info(f"Connecting to MongoDB: {settings.MONGODB_URL}")
            cls._client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls._db = cls._client[settings.MONGODB_DB_NAME]
            logger.info("MongoDB connected successfully")

    @classmethod
    async def close(cls):
        """Закрытие соединения"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB connection closed")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """Получить экземпляр БД"""
        if cls._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls._db


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency для получения БД"""
    await MongoDB.connect()
    return MongoDB.get_db()
