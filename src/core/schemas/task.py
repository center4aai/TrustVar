# src/core/schemas/task.py
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.constants import TaskStatus


class TaskType(str, Enum):
    """Task types"""

    STANDARD = "standard"  # Normal inference
    VARIATION = "variation"  # With prompt variations
    COMPARISON = "comparison"  # Model comparison
    JUDGED = "judged"  # Using LLM judge
    RTA = "refuse_to_answer"  # Refuse-to-Answer tasks
    AB_TEST = "ab_test"  # A/B testing


class VariationStrategy(str, Enum):
    # ── Tier A ──
    FORMAT_NORMALIZATION = "format_normalization"
    ORTHOGRAPHIC_NORMALIZATION_RU = "orthographic_normalization_ru"
    MCQ_OPTION_PERMUTATION = "mcq_option_permutation"
    LIST_REORDERING = "list_reordering"
    TYPED_PARAMETRIC_SUBSTITUTION = "typed_parametric_substitution"

    # ── Tier B ──
    ACTIVE_PASSIVE_VOICE = "active_passive_voice"
    MONOSEMIC_SYNONYM_SUBSTITUTION = "monosemic_synonym_substitution"
    NOMINALISATION = "nominalisation"
    CONTROLLED_SYNTACTIC_TRANSFORMATIONS = "controlled_syntactic_transformations"
    SENTENCE_SPLIT_MERGE = "sentence_split_merge"
    CONTROLLED_DESCRIPTIVE_MODIFIER_INSERTION = (
        "controlled_descriptive_modifier_insertion"
    )

    # ── Tier C ──
    PARAPHRASE_LEXICO_SYNTACTIC_CONSTRAINED = "paraphrase_lexico_syntactic_constrained"
    PARAPHRASE_FREE = "paraphrase_free"
    LENGTH_VARIATION = "length_variation"
    REGISTER_FORMAL_INFORMAL = "register_formal_informal"
    TONE_SHIFT = "tone_shift"
    NEGATION_SCOPE_PRESERVING_REPHRASING = "negation_scope_preserving_rephrasing"
    WSD_SYNONYM_SUBSTITUTION = "wsd_synonym_substitution"
    BACK_TRANSLATION_SINGLE_PIVOT = "back_translation_single_pivot"

    # ── Legacy (deprecated, kept for backward compat with old tasks in MongoDB) ──
    INCREASE_SENTENCE_LEN = "increase_sentence_len"
    SHORTEN_SENTENCE_LEN = "shorten_sentence_len"
    PARAPHRASING = "paraphrasing"
    SYNONYMY = "synonymy"
    STYLE_CHANGE = "style_change"
    DISCOURSE_CONNECTIVE_VAR = "discourse_connective_var"
    # Typographic typo (Cyrillic 'с') in old tasks
    DISCOURSE_CONNECTIVE_VAR_TYPO = "discourse_сonnective_var"
    SPLIT_MERGE_SENT = "split_merge_sent"
    POLITENESS_HEDGING = "politeness_hedging"
    PUNCTUATION_NOISE = "punctuation_noise"
    TRANSLATE_RU = "translate_ru"
    TRANSLATE_EN = "translate_en"

    @property
    def tier(self) -> str:
        """Return operator tier: 'A', 'B' or 'C'."""
        if self in _TIER_A:
            return "A"
        if self in _TIER_B:
            return "B"
        return "C"

    @property
    def requires_llm(self) -> bool:
        """Whether strategy requires LLM for generation (Tier B/C)."""
        return self.tier in ("B", "C")


_TIER_A = frozenset({
    VariationStrategy.FORMAT_NORMALIZATION,
    VariationStrategy.ORTHOGRAPHIC_NORMALIZATION_RU,
    VariationStrategy.MCQ_OPTION_PERMUTATION,
    VariationStrategy.LIST_REORDERING,
    VariationStrategy.TYPED_PARAMETRIC_SUBSTITUTION,
})

_TIER_B = frozenset({
    VariationStrategy.ACTIVE_PASSIVE_VOICE,
    VariationStrategy.MONOSEMIC_SYNONYM_SUBSTITUTION,
    VariationStrategy.NOMINALISATION,
    VariationStrategy.CONTROLLED_SYNTACTIC_TRANSFORMATIONS,
    VariationStrategy.SENTENCE_SPLIT_MERGE,
    VariationStrategy.CONTROLLED_DESCRIPTIVE_MODIFIER_INSERTION,
})


class ABTestStrategy(str, Enum):
    """A/B testing strategies"""

    PROMPT_VARIANTS = "prompt_variants"  # Compare different prompts
    MODEL_COMPARISON = "model_comparison"  # Compare models on same data
    TEMPERATURE_TEST = "temperature_test"  # Test temperatures
    # SYSTEM_PROMPT_TEST = "system_prompt_test"  # Test system prompts
    # PARAMETER_SWEEP = "parameter_sweep"  # Parameter sweep


class VariationConfig(BaseModel):
    """Variation configuration"""

    enabled: bool = False
    model_id: Optional[str] = None  # Model for generating variations
    strategies: List[VariationStrategy] = []
    count_per_strategy: int = 1
    bypass_validation: bool = False
    keep_rejected: bool = True


class JudgeConfig(BaseModel):
    """LLM Judge configuration"""

    enabled: bool = False
    model_id: Optional[str] = None
    criteria: List[str] = ["accuracy", "relevance", "completeness"]
    custom_prompt_template: Optional[str] = None
    custom_prompt_id: Optional[str] = None
    custom_prompt_output_schema: Optional[dict] = None


class RTAConfig(BaseModel):
    """Refuse-to-Answer configuration"""

    enabled: bool = False
    rta_judge_model_id: Optional[str] = None
    rta_prompt_template: Optional[str] = None  # Prompt for RTA judge
    rta_prompt_id: Optional[str] = None
    rta_prompt_output_schema: Optional[dict] = None
    # refusal_keywords: List[str] = [
    #     "I cannot",
    #     "I can't",
    #     "I'm not able to",
    #     "I refuse",
    #     "I apologize, but",
    #     "I'm sorry, but",
    #     "I cannot assist",
    # ]


class ABTestConfig(BaseModel):
    """A/B test configuration"""

    enabled: bool = False
    strategy: Optional[ABTestStrategy] = None

    # For PROMPT_VARIANTS
    prompt_variants: Optional[Dict[str, str]] = None  # variant_name -> prompt

    # For TEMPERATURE_TEST
    temperatures: Optional[List[float]] = None  # [0.3, 0.7, 1.0]

    # For SYSTEM_PROMPT_TEST
    system_prompts: Optional[Dict[str, str]] = None  # variant_name -> system_prompt

    # For PARAMETER_SWEEP
    parameter_ranges: Optional[Dict[str, List[Any]]] = None  # param_name -> [values]

    # General settings
    sample_size_per_variant: Optional[int] = None  # Sample size for each variant
    statistical_test: str = "t_test"  # t_test, chi_square, mann_whitney

    balance_variants: bool = True  # Automatic balancing
    random_seed: Optional[int] = None  # For reproducibility


class TaskResult(BaseModel):
    """Task execution result"""

    input: str
    output: str
    model_id: str  # ID of model that generated the result
    model_name: str  # Name of model that generated the result
    target: Optional[str] = None
    metrics: List[str] = []  # Dict[str, str] = {}
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}

    # For variations
    original_input: Optional[str] = None  # Original prompt
    variation_type: Optional[str] = None  # Variation type

    valid: Optional[bool] = None  # False = REJECT retained under keep_rejected
    validator_verdict: Optional[str] = None  # validation_status (accept/flag_*/reject_*/bypassed)
    validator_layers: Optional[Dict[str, Any]] = None  # validation_metadata (Layer 1/2/3 detail)

    # For LLM judge and RTA
    judge_score: Optional[float] = None
    judge_results: Optional[Dict[str, Any]] = None

    # For RTA
    refused: Optional[str] = None  # Whether model refused to answer

    # For Include/Exclude
    include_score: Optional[float] = None
    exclude_violations: Optional[int] = None

    # For A/B tests
    ab_variant: Optional[str] = None  # A/B test variant


class TaskConfig(BaseModel):
    """Task configuration"""

    # Basic settings
    batch_size: int = 1
    max_samples: Optional[int] = None

    # Evaluation
    evaluate: bool = True
    evaluation_metrics: List[str] = []

    # Variations
    variations: VariationConfig = Field(default_factory=VariationConfig)

    # LLM Judge
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # Refuse-to-Answer
    rta: RTAConfig = Field(default_factory=RTAConfig)

    # A/B tests
    ab_test: ABTestConfig = Field(default_factory=ABTestConfig)


class Task(BaseModel):
    """Task model"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_ids: List[str]  # List of models
    task_type: TaskType = TaskType.STANDARD
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0

    # Configuration
    config: TaskConfig = Field(default_factory=TaskConfig)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results (data stored in 'task_results' collection)
    total_samples: int = 0
    processed_samples: int = 0
    aggregated_metrics: Dict[str, Any] = {}  # model_id -> metrics or _trustvar
    # WEB-5/WEB-6: run completeness + per-model coverage (None until the run ends)
    completion_summary: Optional[Dict[str, Any]] = None

    # For A/B tests
    ab_test_results: Optional[Dict[str, Any]] = None  # Statistical analysis

    # Errors
    error: Optional[str] = None
    # RUN-3: post-processing trustvar metrics error (None = OK). Separate from `error`
    # (error = task/inference failure). See closed_decisions/2026-07-01_run3_...
    metrics_error: Optional[str] = None

    # Metadata
    celery_task_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    recovery_checkpoint: Optional[Dict[str, Any]] = None
    last_processed_index: int = 0  # Index of last processed item
    current_execution: Optional[Dict[str, Any]] = None  # Currently executing task
    recent_executions: List[Dict[str, Any]] = []  # Last 2 executed tasks

    class Config:
        use_enum_values = True
