from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PromptType(str, Enum):
    """Prompt types"""

    JUDGE = "judge"
    RTA = "rta"
    VARIATION = "variation"


class PromptBase(BaseModel):
    """Base prompt model"""

    name: str
    content: str
    prompt_type: PromptType
    description: Optional[str] = None
    output_schema: Optional[dict] = None
    input_variables: Optional[List[str]] = None


# PromptCreate no longer needed, use PromptBase directly


class Prompt(PromptBase):
    """Prompt model from DB"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    class Config:
        use_enum_values = True
