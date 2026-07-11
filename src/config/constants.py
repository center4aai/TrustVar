# src/config/constants.py
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


# Supported tasks
SUPPORTED_TASKS = [
    "text-generation",
    "question-answering",
    "summarization",
    "translation",
    "classification",
]

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "rta",
]
