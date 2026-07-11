# src/core/schemas/task_result.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class StoredTaskResult(BaseModel):
    """Result document stored in the separate 'task_results' collection.

    Kept separate from the Task document to prevent hitting MongoDB's 16 MB
    BSON limit when a task accumulates thousands of results.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    sequence_num: int  # 0-based insertion order; used for pagination and resume

    # Core inference fields
    input: str
    output: str
    model_id: str
    model_name: str
    target: Optional[str] = None
    metrics: List[str] = []
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}

    # Variation fields
    original_input: Optional[str] = None
    variation_type: Optional[str] = None

    valid: Optional[bool] = None
    validator_verdict: Optional[str] = None
    validator_layers: Optional[Dict[str, Any]] = None

    # LLM judge fields
    judge_score: Optional[float] = None
    judge_results: Optional[Dict[str, Any]] = None

    # RTA (Refuse-to-Answer) field
    refused: Optional[str] = None

    # Include/Exclude fields
    include_score: Optional[float] = None
    exclude_violations: Optional[int] = None

    # A/B test field
    ab_variant: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
