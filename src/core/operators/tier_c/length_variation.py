from typing import Optional

from src.core.operators.base import PreCheckResult, TierCOperator, Tier, VariationResult
from src.core.taxonomy import resolve_task_semantics


_LENGTHEN_TEMPLATE = """- Variation: Lengthen the task.
- Goal: Make the task noticeably longer via paraphrasing, synonym expansion, and semantically neutral scaffolding without changing semantics, intent, tone, register, constraints, entities, or formatting.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- Techniques:
  • Insert semantically neutral discourse markers or frame-setters ("In this task,", "Please note that")
  • Replace concise phrases with longer but equivalent expressions
  • Expand clauses into fuller constructions
  • Use same-register synonyms that lengthen text naturally
- Target: noticeably longer (+20–50%).
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{ prompt }}"""


_SHORTEN_TEMPLATE = """- Variation: Shorten the task.
- Goal: Make the task more concise via paraphrasing, concise synonyms, and condensation without changing semantics, intent, tone, register, constraints, entities, or formatting.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- Techniques:
  • Remove discourse fillers and verbosity
  • Replace wordy phrases with concise equivalents ("in order to" → "to")
  • Collapse redundant modifiers; prefer precise, same-register synonyms
  • Convert clause→phrase or passive→active if shorter and meaning unchanged
- Target: noticeably shorter (15–30%).
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{ prompt }}"""


_LENGTH_EXCLUDED_TASK_TYPES = {
    "summarization",
    "text_compression",
    "headline_generation",
    "title_generation",
}


class LengthVariationOperator(TierCOperator):
    operator_id = "length_variation"
    tier = Tier.C
    stochastic = True

    # Prompt templates indexed by sub-strategy direction.
    _TEMPLATES = {
        "lengthen": _LENGTHEN_TEMPLATE,
        "shorten": _SHORTEN_TEMPLATE,
    }

    def __init__(self, direction: str = "lengthen"):
        if direction not in self._TEMPLATES:
            raise ValueError(
                f"LengthVariationOperator: direction must be one of "
                f"{list(self._TEMPLATES)}, got {direction!r}"
            )
        self.direction = direction

    @property
    def prompt_template(self) -> str:
        return self._TEMPLATES[self.direction]

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # ARCH-002: accept direction kwarg so gate matches the actual apply() direction.
        direction = kwargs.get("direction", self.direction)
        tokens = text.strip().split()
        n_tokens = len(tokens)
        # C3: Text length ≥ 6 tokens (below this, compression becomes uninformative)
        if n_tokens < 6:
            return PreCheckResult(
                passed=False,
                reason=f"Text too short for length variation ({n_tokens} tokens, min 6)",
            )
        # C3: Extension excluded if text is already verbose (> 500 tokens)
        if direction == "lengthen" and n_tokens > 500:
            return PreCheckResult(
                passed=False,
                reason="Text already verbose (>500 tokens); lengthen excluded",
            )
        # C3: Compression excluded if text is already terse (< 10 tokens)
        if direction == "shorten" and n_tokens < 10:
            return PreCheckResult(
                passed=False,
                reason="Text already terse (<10 tokens); shorten excluded",
            )
        # C3: Excluded for tasks evaluating text length, conciseness, or summarization
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in _LENGTH_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (evaluates length/summarization)",
            )
        return PreCheckResult(
            passed=True,
            details={"direction": direction, "n_tokens": n_tokens},
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        adapter=None,
        **kwargs,
    ) -> VariationResult:
        from jinja2 import Template

        # ARCH-001: build prompt from effective direction without mutating self.direction —
        # safe for concurrent asyncio coroutines sharing the same operator instance.
        direction = kwargs.pop("direction", None) or self.direction
        prompt = Template(self._TEMPLATES[direction]).render(prompt=text)

        if adapter is None:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_adapter"},
                original_text=text,
            )

        variation = await adapter.generate(prompt, temperature=0.8)
        return VariationResult(
            variant_text=variation.strip(),
            metadata={"strategy": self.operator_id},
            original_text=text,
        )