# src/core/schemas/dataset.py
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar
from uuid import uuid4
import json

from pydantic import BaseModel, Field

TDatasetItem = TypeVar("TDatasetItem", bound="DatasetItem")


class DatasetItem(BaseModel):
    """Dataset item"""

    # id: str = Field(default_factory=lambda: str(uuid4()), alias="_id")
    dataset_id: str

    prompt: str
    target: Optional[str] = None
    include_list: Optional[List[str]] = None
    exclude_list: Optional[List[str]] = None

    # Template data
    template: Optional[str] = None  # Prompt template (original, if from CSV)
    variables: Optional[Dict[str, Any]] = None  # Variables for template substitution

    # All columns from source file (including prompt, target, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @classmethod
    def from_row(
        cls,
        row: Dict[str, Any],
        dataset_id: str,
        prompt_column: str,
        target_column: Optional[str] = None,
        include_column: Optional[str] = None,
        exclude_column: Optional[str] = None,
        template_column: Optional[str] = None,
        variables_columns: Optional[List[str]] = None,
    ) -> TDatasetItem:
        if prompt_column not in row or not str(row.get(prompt_column)).strip():
            raise ValueError(f"Prompt column '{prompt_column}' is missing or empty")

        metadata = {k: v for k, v in row.items()}

        prompt = str(row[prompt_column]).strip()

        template = None
        if template_column and row.get(template_column):
            template = str(row[template_column]).strip()

        variables = None
        if variables_columns:
            variables = {}
            for col in variables_columns:
                if col in row:
                    value = row[col]
                    if isinstance(value, str):
                        value = value.strip()
                        if value.startswith("{") and value.endswith("}"):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                try:
                                    import ast
                                    value = ast.literal_eval(value)
                                except (ValueError, SyntaxError):
                                    pass
                    
                    if isinstance(value, dict) and col == "inputs":
                        variables.update(value)
                    else:
                        variables[col] = value

        target = None
        if target_column and row.get(target_column) is not None:
            target = str(row[target_column]).strip()

        include_list = cls._parse_list(row.get(include_column))
        exclude_list = cls._parse_list(row.get(exclude_column))

        return cls(
            dataset_id=dataset_id,
            prompt=prompt,
            target=target,
            include_list=include_list,
            exclude_list=exclude_list,
            template=template,
            variables=variables,
            metadata=metadata,
        )

    @staticmethod
    def _parse_list(value: Any) -> Optional[List[str]]:
        """Convert possible list formats to List[str]"""
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [
                            str(item).strip() for item in parsed if str(item).strip()
                        ]
                except:
                    pass
            return [x.strip() for x in value.split(",") if x.strip()]
        return [str(value).strip()]


class Dataset(BaseModel):
    """Dataset model"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    format: str = "jsonl"
    size: int = 0
    # Scoring canon (mcq/classification/open_qa/generation) — drives evaluation,
    # answer-preservation and EAR. Normalised at the boundary via
    # taxonomy.normalize_task_type. Free-form input accepted for back-compat.
    task_type: str
    # Operator-precondition semantics (fine, open vocabulary: set_membership,
    # sentiment_classification, summarization, negation_detection, …) — consulted
    # by transformation operators to decide applicability. Distinct axis from
    # task_type (S1.1, decision b). When None, operators fall back to the raw
    # task_type label via taxonomy.resolve_task_semantics.
    task_semantics: Optional[str] = None
    tags: List[str] = []

    # Column configuration
    prompt_column: str = "prompt"  # Column name for prompts
    target_column: Optional[str] = None  # Column name for target values (if exists)
    include_column: Optional[str] = None  # Column name for include_list
    exclude_column: Optional[str] = None  # Column name for exclude_list

    # Template configuration
    template_column: Optional[str] = None  # Column name for prompt template (for Jinja2)
    variables_columns: Optional[List[str]] = None  # List of columns with variables for substitution

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Dataset",
                "description": "Test dataset for QA",
                "format": "jsonl",
                "task_type": "question-answering",
                "tags": ["qa", "test"],
                "prompt_column": "prompt",
                "target_column": "answer",
            }
        }
