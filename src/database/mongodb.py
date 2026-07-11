# src/database/mongodb.py
from typing import Any, Dict, List

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()

# ── Index descriptions ─────────────────────────────────────────────────────

INDEXES: Dict[str, List[Dict[str, Any]]] = {
    "task_results": [
        {
            # Matches what the repository created —
            # use same name to avoid duplicates
            "keys": [("task_id", 1), ("sequence_num", 1)],
            "options": {"name": "task_id_sequence_num"},
            "description": (
                "Paginated reading of results: "
                "find_by_task(skip, limit), post_processing"
            ),
        },
        {
            "keys": [("task_id", 1), ("model_id", 1)],
            "options": {"name": "task_id_model_id"},
            "description": (
                "Per-model count: count_by_task_and_model(), "
                "find_by_task_and_model(), recovery"
            ),
        },
        {
            "keys": [("task_id", 1)],
            "options": {"name": "task_id"},
            "description": (
                "Basic task search: count_by_task(), "
                "find_all_by_task(), delete_by_task()"
            ),
        },
        # ── New indexes for recovery and post_processing ──────────────
        {
            "keys": [
                ("task_id", 1),
                ("model_id", 1),
                ("metadata.item_index", 1),
                ("variation_type", 1),
            ],
            "options": {"name": "task_id_model_item_var"},
            "description": (
                "Resume: get_completed_pairs() builds Set "
                "of already processed (item_index, variation_type)"
            ),
        },
        {
            "keys": [("task_id", 1), ("judge_score", 1)],
            "options": {"name": "task_id_judge_score", "sparse": True},
            "description": (
                "Post-processing: aggregate judge_score by models. "
                "sparse=True — documents without judge_score not indexed"
            ),
        },
        {
            "keys": [("task_id", 1), ("refused", 1)],
            "options": {"name": "task_id_refused", "sparse": True},
            "description": (
                "Post-processing: aggregate RTA refusal_rate. "
                "sparse=True — documents without refused not indexed"
            ),
        },
        {
            "keys": [("task_id", 1), ("ab_variant", 1)],
            "options": {"name": "task_id_ab_variant", "sparse": True},
            "description": (
                "A/B analysis: find_by_task_with_ab() filters "
                "only results with ab_variant != None. "
                "sparse=True — documents without ab_variant not indexed"
            ),
        },
    ],
    "tasks": [
        {
            "keys": [("id", 1)],
            "options": {"name": "idx_task_id", "unique": True},
            "description": (
                "Primary key: find_by_id() called very often — "
                "status check, recovery, progress tracker"
            ),
        },
        {
            "keys": [("status", 1)],
            "options": {"name": "idx_task_status"},
            "description": (
                "Status filtering: find_by_status(), GET /tasks?status=running"
            ),
        },
    ],
    "dataset_items": [
        {
            # DatasetRepository.get_items() does find by dataset_id
            "keys": [("dataset_id", 1)],
            "options": {"name": "dataset_id"},
            "description": (
                "get_items(dataset_id): search dataset items. "
                "Without index — full scan on every request"
            ),
        },
    ],
}


class MongoDB:
    """Singleton for MongoDB connection"""

    _client: AsyncIOMotorClient = None
    _db: AsyncIOMotorDatabase = None

    @classmethod
    async def connect(cls):
        """Connect to MongoDB"""
        if cls._client is None:
            logger.info(f"Connecting to MongoDB: {settings.MONGODB_URL}")
            cls._client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls._db = cls._client[settings.MONGODB_DB_NAME]
            logger.info("MongoDB connected successfully")

    @classmethod
    async def close(cls):
        """Close connection"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB connection closed")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if cls._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls._db

    @classmethod
    async def ensure_indexes(cls, drop_existing: bool = False) -> Dict[str, int]:
        """
        Create indexes if they don't exist.
        Uses MongoDB singleton — connect() is idempotent.

        Safe to run repeatedly: MongoDB will not recreate
        index if it already exists with the same name and keys.

        Returns:
            {"created": N, "skipped": N, "failed": N}
        """

        stats = {"created": 0, "skipped": 0, "failed": 0}

        for collection_name, indexes in INDEXES.items():
            collection = cls._db[collection_name]

            if drop_existing:
                logger.warning(
                    f"Dropping all indexes for '{collection_name}' (except _id)..."
                )
                await collection.drop_indexes()

            existing_names: set = set()
            async for idx in collection.list_indexes():
                existing_names.add(idx["name"])

            for index_config in indexes:
                name = index_config["options"]["name"]
                description = index_config["description"]

                if name in existing_names and not drop_existing:
                    logger.debug(f"Index already exists [{collection_name}] {name}")
                    stats["skipped"] += 1
                    continue

                try:
                    await collection.create_index(
                        index_config["keys"],
                        **index_config["options"],
                    )
                    logger.info(
                        f"Index created [{collection_name}] {name}: {description}"
                    )
                    stats["created"] += 1
                except Exception as e:
                    logger.error(
                        f"Failed to create index '{name}' on '{collection_name}': {e}"
                    )
                    stats["failed"] += 1

        logger.info(
            f"Indexes ensured: "
            f"created={stats['created']}, "
            f"skipped={stats['skipped']}, "
            f"failed={stats['failed']}"
        )
        return stats

    @classmethod
    async def list_indexes(cls) -> Dict[str, List[Dict]]:
        """Return all existing indexes for our collections"""

        result: Dict[str, List[Dict]] = {}
        for collection_name in INDEXES:
            collection = cls._db[collection_name]
            indexes = []
            async for idx in collection.list_indexes():
                indexes.append(
                    {
                        "name": idx["name"],
                        "key": dict(idx["key"]),
                        "unique": idx.get("unique", False),
                        "sparse": idx.get("sparse", False),
                    }
                )
            result[collection_name] = indexes

        return result

    @classmethod
    async def drop_indexes(cls, collection_name: str = None) -> None:
        """
        Drop indexes.

        Args:
            collection_name: if None — drops in all collections.
        """
        targets = [collection_name] if collection_name else list(INDEXES.keys())
        for name in targets:
            await cls._db[name].drop_indexes()
            logger.warning(f"Dropped all indexes for '{name}'")


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency for getting database"""
    await MongoDB.connect()
    return MongoDB.get_db()
