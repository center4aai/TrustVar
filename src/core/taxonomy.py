# src/core/taxonomy.py

from typing import Dict, Optional

from src.utils.logger import logger

# ── Scoring canon ─────────────────────────────────────────────────────────

CANONICAL_TASK_TYPES = ("mcq", "classification", "open_qa", "generation")

_TASK_TYPE_ALIASES: Dict[str, str] = {
    # mcq
    "mcq": "mcq",
    "multiple_choice": "mcq",
    "multiple-choice": "mcq",
    "multiplechoice": "mcq",
    "choice": "mcq",
    "single_choice": "mcq",
    # classification
    "classification": "classification",
    "class": "classification",
    "classify": "classification",
    "label": "classification",
    "labeling": "classification",
    "labelling": "classification",
    "sentiment": "classification",
    # open-qa
    "open_qa": "open_qa",
    "open-qa": "open_qa",
    "openqa": "open_qa",
    "qa": "open_qa",
    "open_question": "open_qa",
    "open_question_answering": "open_qa",
    "free_qa": "open_qa",
    "short_answer": "open_qa",
    # generation
    "generation": "generation",
    "gen": "generation",
    "free_generation": "generation",
    "free-generation": "generation",
    "freeform": "generation",
    "free_form": "generation",
    "text_generation": "generation",
    "generate": "generation",
    "summarization": "generation",
    # ── backward-compat: old UI vocabulary + common variants ────────
    "question_answering": "open_qa",
    "question-answering": "open_qa",
    "translation": "generation",
    "other": "open_qa",
    "text_generation": "generation",
    "text-generation": "generation",
    "summarisation": "generation",
}


def normalize_task_type(raw: Optional[str], default: Optional[str] = "classification") -> Optional[str]:
    """Normalize a raw ``task_type`` label to the canonical scoring taxonomy.

    Maps known aliases ('gen', 'free_generation', 'open-qa', …) to canonical
    types and validates at the boundary. Unknown non-empty values:

    * If ``default`` is a string → fall back to ``default`` with a warning.
    * If ``default`` is ``None`` → return ``None`` (caller should reject with an
      explicit error; this is the API-boundary validation path).

    NOTE (decision *b*): this function is for the **scoring** axis only. Do NOT
    feed it operator-precondition labels (``set_membership`` etc.) — those
    belong to ``task_semantics`` and are resolved by
    :func:`resolve_task_semantics`.
    """
    if raw is None:
        return default
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if key in _TASK_TYPE_ALIASES:
        return _TASK_TYPE_ALIASES[key]
    if key in CANONICAL_TASK_TYPES:
        return key
    if default is not None:
        logger.warning(
            f"Unknown task_type '{raw}'; defaulting to '{default}'. "
            f"Add it to _TASK_TYPE_ALIASES if it is a real scoring type, or pass it "
            f"as task_semantics if it is an operator-precondition label."
        )
    return default


# ── Operator-precondition semantics ───────────────────────────────────────


def resolve_task_semantics(
    task_type: Optional[str], task_semantics: Optional[str]
) -> Optional[str]:
    """Resolve the effective **operator-precondition** signal (the fine axis).

    Operators gate on the fine ``task_semantics``. When a dataset does not
    declare ``task_semantics`` explicitly, fall back to the raw ``task_type``
    label so that legacy datasets which encoded fine information in
    ``task_type`` (e.g. ``set_membership``, ``sentiment_classification``)
    keep working unchanged. Returns ``None`` only when neither is provided.

    This is the operator-side counterpart of :func:`normalize_task_type`
    (the scoring-side normaliser); the two never share a value.
    """
    return task_semantics or task_type
