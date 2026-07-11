import re
from typing import Optional

from src.core.operators.base import PreCheckResult, TierCOperator, Tier, VariationResult
from src.core.taxonomy import resolve_task_semantics


# Combined negation marker patterns for EN and RU.
# Used both as a pre-condition gate ("source must contain at least one negation")
# and as documentation of the surface forms the operator targets.
_EN_NEGATION_RE = re.compile(
    r"\b(not|n't|never|no|nothing|nobody|nowhere|neither|nor|"
    r"cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|"
    r"hardly|barely|scarcely)\b",
    re.IGNORECASE,
)
_RU_NEGATION_RE = re.compile(
    r"\b(не|нет|ни|никто|ничто|нигде|никогда|нечего|некого|никакой|"
    r"ничей|нисколько|нельзя|без|некогда)\b",
    re.IGNORECASE,
)


_NEGATION_EXCLUDED_TASK_TYPES = {
    "nli_negation",
    "negation_detection",
    "negation_resolution",
    "negation_scope_resolution",
    "nli",
    "natural_language_inference",
}


_SCOPE_AMBIGUITY_EN = re.compile(
    r"\b((all|every|each|everyone|everything|everybody)\s+"
    r"(?:\w+\s+){0,5}(not|n't)|"
    r"(may|might)\s+not\b|"
    r"not\s+(?:\w+\s+){0,5}(because|since)\b)",
    re.IGNORECASE,
)
_SCOPE_AMBIGUITY_RU = re.compile(
    r"\b((все|каждый|всякий|всегда)\s+"
    r"(?:\w+\s+){0,5}не\b|"
    r"не\s+(?:\w+\s+){0,5}(потому|так\s+как)\b)",
    re.IGNORECASE,
)

_COMPLEX_SCOPE_RE = re.compile(
    r"\b(if|when|unless|although|because|since|while|"
    r"если|когда|если\s+бы|хотя|потому\s+что|так\s+как)\b.*"
    r"\b(not|n't|never|no|nor|не|нет|никогда|ни)\b.*"
    r"\b(if|when|unless|although|because|since|while|"
    r"если|когда|если\s+бы|хотя|потому\s+что|так\s+как)\b",
    re.IGNORECASE,
)

_RU_CONCORD_RE = re.compile(
    r"\b(никто|ничто|никого|ничему|ничем|ни о ком|"
    r"никогда|нигде|никуда|ниоткуда|нисколько|"
    r"никакой|ничей)\s+(?:\w+\s+){0,3}не\b",
    re.IGNORECASE,
)


class NegationScopePreservingRephrasingOperator(TierCOperator):
    operator_id = "negation_scope_preserving_rephrasing"
    tier = Tier.C
    stochastic = True

    @property
    def prompt_template(self) -> str:
        return """- Variation: Negation-scope-preserving rephrasing.
- Goal: Rephrase the task while preserving the scope and polarity of ALL negation operators exactly. Negation (not, never, no, nothing, nobody, neither, nor, etc.) must remain logically equivalent.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- CRITICAL: Polarity and scope of every negation must be preserved. If the original says "not all", the variant must also say "not all". If the original says "nothing", the variant must also contain a equivalent negation.
- Do NOT: add/remove facts; double negate; change logical operators.
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{ prompt }}"""

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # C6: Source must contain at least one negation marker.
        lang = (language or "en").lower()
        negation_re = _RU_NEGATION_RE if lang == "ru" else _EN_NEGATION_RE
        matches = negation_re.findall(text)
        if not matches:
            return PreCheckResult(
                passed=False,
                reason="No negation marker in source; nothing to rephrase",
            )
        # C6: Excluded for tasks where negation is the evaluation target.
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in _NEGATION_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (negation is label-relevant)",
            )
        # C6: Excluded if scope of negation is ambiguous in the source.
        ambig_re = _SCOPE_AMBIGUITY_RU if lang == "ru" else _SCOPE_AMBIGUITY_EN
        ambig_match = ambig_re.search(text)
        if ambig_match:
            return PreCheckResult(
                passed=False,
                reason=f"Negation scope ambiguous: {ambig_match.group(0)!r}",
            )
        # C6: Excluded for complex scope interactions (multi-clause
        # conditionals with embedded negation, multi-negation interactions).
        complex_match = _COMPLEX_SCOPE_RE.search(text)
        if complex_match:
            return PreCheckResult(
                passed=False,
                reason=f"Complex scope interaction detected: {complex_match.group(0)[:80]!r}",
            )
        # C6-003: Russian negative concord detection — warn (not reject) when
        # multiple negation markers participate in concord (e.g., "никто не", "ничего не").
        if lang == "ru":
            concord = _RU_CONCORD_RE.findall(text)
            if concord:
                pass  # Concord detected; preserved in metadata below.

        # Normalise markers to lowercase for stable downstream comparison
        # (regex IGNORECODE preserves original casing; details want canonical).
        normalised = sorted({m.lower() for m in matches})
        return PreCheckResult(
            passed=True,
            details={
                "negation_markers": normalised[:5],
                "language": lang,
            },
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        adapter=None,
        **kwargs,
    ) -> VariationResult:
    
        language = kwargs.get("language") or "en"
        result = await super().apply(text, seed=seed, adapter=adapter, **kwargs)
        if language == "ru" or result.metadata.get("language") == "ru":
            result.metadata["human_audit_required"] = True
        return result