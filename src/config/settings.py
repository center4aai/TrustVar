# src/config/settings.py
from functools import lru_cache
from typing import Dict, Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TrustVar"
    VERSION: str = "0.1.1"
    DEBUG: bool = False

    # Api
    API_IP: str = "api"
    API_PORT: int = 8000

    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "llm_framework"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Ollama
    OLLAMA_BASE_URL: str = Field(json_schema_extra={"env": "OLLAMA_BASE_URL"})
    OLLAMA_KEEP_ALIVE: str = Field(
        default="-1s", json_schema_extra={"env": "OLLAMA_KEEP_ALIVE"}
    )
    OLLAMA_REQUIRE_GPU: bool = Field(
        default=True, json_schema_extra={"env": "OLLAMA_REQUIRE_GPU"}
    )
    OLLAMA_GPU_RETRY_ON_FALLBACK: bool = Field(
        default=True, json_schema_extra={"env": "OLLAMA_GPU_RETRY_ON_FALLBACK"}
    )
    OLLAMA_INFERENCE_TIMEOUT: int = Field(
        default=3000, json_schema_extra={"env": "OLLAMA_INFERENCE_TIMEOUT"}
    )
    OLLAMA_NUM_PARALLEL: int = Field(
        default=5, json_schema_extra={"env": "OLLAMA_NUM_PARALLEL"}
    )
    OLLAMA_MAX_RETRIES: int = Field(
        default=3, json_schema_extra={"env": "OLLAMA_MAX_RETRIES"}
    )
    OLLAMA_RETRY_BACKOFF_BASE: float = Field(
        default=1.0, json_schema_extra={"env": "OLLAMA_RETRY_BACKOFF_BASE"}
    )
    # HuggingFace
    HF_TOKEN: str = Field(json_schema_extra={"env": "HF_TOKEN"})
    HF_CACHE_DIR: str = "./cache/huggingface"

    # RuWordNet (RU lexical-semantic DB for Tier B/C synonym operators).
    # Project-managed cache (git-ignored ./cache/), mirroring HF_CACHE_DIR /
    # MINICHECK_CACHE_DIR: env-overridable, auto-provisioned on first use.
    # In hermetic/offline CI set RUWORDNET_AUTO_DOWNLOAD=0 and point
    # RUWORDNET_DB_PATH at a pre-baked copy (e.g. a mounted Docker volume).
    RUWORDNET_DB_PATH: str = "./cache/ruwordnet/ruwordnet.db"
    RUWORDNET_DB_URL: str = (
        "https://github.com/avidale/python-ruwordnet/releases/download/0.0.4/ruwordnet-2021.db"
    )
    RUWORDNET_AUTO_DOWNLOAD: bool = True

    # OpenAI
    OPENAI_API_KEY: str = Field(json_schema_extra={"env": "OPENAI_API_KEY"})
    OPENAI_BASE_URL: str = Field(json_schema_extra={"env": "OPENAI_BASE_URL"})

    # vLLM
    VLLM_BASE_URL: str = Field(json_schema_extra={"env": "VLLM_BASE_URL"})

    # LLAMACPP
    LLAMACPP_BASE_URL: str = Field(json_schema_extra={"env": "LLAMACPP_BASE_URL"})

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # ── Validation NLI models ──────────────────────────────────────
    EN_NLI_MODEL_PRIMARY: str = "cross-encoder/nli-deberta-v3-large"
    EN_NLI_MODEL_SECONDARY: str = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    RU_NLI_MODEL_PRIMARY: str = "cointegrated/rubert-base-cased-nli-threeway"
    RU_NLI_MODEL_SECONDARY: str = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    EMBEDDING_MODEL: str = "sentence-transformers/LaBSE"
    SENTIMENT_MODEL_EN: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    SENTIMENT_MODEL_RU: str = "blanchefort/rubert-base-cased-sentiment"

    MAX_NLI_TOKENS: int = 512

    # ── Paraphrase excluded task types ─────────────────────────────
    PARAPHRASE_EXCLUDED_TASK_TYPES: Set[str] = {
        "paraphrase_detection",
        "paraphrase_identification",
        "paraphrase_generation",
        "lexical_substitution",
        "lexical_simplification",
        "text_simplification",
    }

    # ── Operator-specific thresholds ───────────────────────────────
    WSD_CONFIDENCE_THRESHOLD_EN: float = 0.70
    WSD_CONFIDENCE_THRESHOLD_RU: float = 0.65
    WSD_THRESHOLD: float = 0.35

    SENTENCE_SPLIT_MIN_TOKENS: int = 20
    SENTENCE_MERGE_MAX_TOKENS: int = 15
    SENTENCE_MERGE_MAX_GROUP: int = 3
    SENTENCE_WORD_OVERLAP_FLAG: float = 0.40

    BACK_TRANSLATION_MAX_LENGTHEN_RATIO: float = 2.0
    BACK_TRANSLATION_MAX_SHORTEN_RATIO: float = 0.50

    LAYER1_CONTENT_JACCARD_MIN_FORMAT: float = 0.80
    LAYER1_FORMAT_NORMALIZATION_MIN_OVERLAP: float = 0.80


    OPERATOR_LAYER1: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "paraphrase_lexico_syntactic_constrained": {
                "jaccard_min": 0.30,
                "length_min": 0.80,
                "length_max": 1.25,
                "cosine_min": 0.85,
                "bertscore_min": 0.75,
                "chi2_max": 0.05,
                "nli_min": 0.80,
            },
            "paraphrase_free": {
                "jaccard_min": 0.20,
                "length_min": 0.70,
                "length_max": 1.40,
                "cosine_min": 0.80,
                "bertscore_min": 0.65,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "length_variation": {
                "jaccard_min": 0.25,
                "jaccard_min_compression": 0.35,
                "length_min": 0.65,
                "length_max": 1.55,
                "cosine_min": 0.80,
                "cosine_min_compression": 0.85,
                "bertscore_min": 0.65,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "register_formal_informal": {
                "jaccard_min": 0.20,
                "length_min": 0.85,
                "length_max": 1.20,
                "cosine_min": 0.78,
                "bertscore_min": 0.65,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "tone_shift": {
                "jaccard_min": 0.20,
                "length_min": 0.85,
                "length_max": 1.25,
                "cosine_min": 0.78,
                "bertscore_min": 0.65,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "negation_scope_preserving_rephrasing": {
                "jaccard_min": 0.25,
                "length_min": 0.80,
                "length_max": 1.30,
                "cosine_min": 0.80,
                "bertscore_min": 0.70,
                "chi2_max": 0.08,
                "nli_min": 0.80,
            },
            "active_passive_voice": {
                "jaccard_min": 0.30,
                "length_min": 0.65,
                "length_max": 1.40,
                "cosine_min": 0.70,
                "bertscore_min": 0.75,
                "chi2_max": 0.08,
                "nli_min": 0.75,
            },
            "monosemic_synonym_substitution": {
                "jaccard_min": 0.25,
                "length_min": 0.70,
                "length_max": 1.30,
                "cosine_min": 0.85,
                "bertscore_min": 0.75,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "wsd_synonym_substitution": {
                "jaccard_min": 0.30,
                "length_min": 0.85,
                "length_max": 1.15,
                "cosine_min": 0.85,
                "bertscore_min": 0.75,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "nominalisation": {
                "jaccard_min": 0.25,
                "length_min": 0.70,
                "length_max": 1.40,
                "cosine_min": 0.70,
                "bertscore_min": 0.75,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "controlled_syntactic_transformations": {
                "jaccard_min": 0.25,
                "length_min": 0.65,
                "length_max": 1.60,
                "cosine_min": 0.70,
                "bertscore_min": 0.75,
                "chi2_max": 0.08,
                "nli_min": 0.75,
            },
            "back_translation_single_pivot": {
                "jaccard_min": 0.20,
                "length_min": 0.70,
                "length_max": 1.40,
                "cosine_min": 0.85,
                "bertscore_min": 0.75,
                "chi2_max": 0.10,
                "nli_min": 0.75,
            },
            "sentence_split_merge": {
                "jaccard_min": 0.40,
                "length_min": 0.80,
                "length_max": 1.30,
                "cosine_min": 0.85,
                "bertscore_min": 0.75,
                "chi2_max": 0.08,
                "nli_min": 0.75,
            },
            "controlled_descriptive_modifier_insertion": {
                "jaccard_min": 0.50,
                "length_min": 0.85,
                "length_max": 1.50,
                "cosine_min": 0.85,   
                "bertscore_min": 0.80, 
                "chi2_max": 0.05,
                "nli_min": 0.75,
            },
        }
    )

    TIER_C_DEFAULTS: Dict[str, float] = Field(
        default_factory=lambda: {
            "jaccard_min": 0.20,
            "length_min": 0.70,
            "length_max": 1.40,
            "cosine_min": 0.80,
            "bertscore_min": 0.65,
            "chi2_max": 0.10,
            "nli_min": 0.75,
        }
    )

    TIER_B_DEFAULTS: Dict[str, float] = Field(
        default_factory=lambda: {
            "jaccard_min": 0.25,
            "length_min": 0.80,
            "length_max": 1.30,
            "cosine_min": 0.85,
            "bertscore_min": 0.75,
            "chi2_max": 0.08,
            "nli_min": 0.75,
        }
    )

    OPERATOR_NLI_DIRECTION: Dict[str, str] = Field(
        default_factory=lambda: {
            "controlled_descriptive_modifier_insertion": "backward_only",
        }
    )

    DISAGREEMENT_THRESHOLD: float = 0.20
    MAX_REGISTER_GAP: int = 1

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
