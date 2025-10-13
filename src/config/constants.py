# src/config/constants.py
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


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
    "bleu",
    "rouge",
    "perplexity",
    "exact_match",
    "f1_score",
]
