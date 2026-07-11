import re
from typing import Optional

from src.config.settings import get_settings
from src.core.operators.base import PreCheckResult, TierCOperator, Tier
from src.core.taxonomy import resolve_task_semantics

settings = get_settings()


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|<code>[\s\S]*?</code>", re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MATH_EXPR_RE = re.compile(
    r"(\$\$[\s\S]+?\$\$|\\\[.+?\\\]|\\\(.*?\\\)|"
    r"\$[^$\n]+\$|"  # LaTeX inline $...$
    r"\\[a-zA-Z]+|"
    r"=\s*[\d\.]+\s*[+\-*/]\s*[\d\.]+)"
)


class ParaphraseLexicoSyntacticConstrainedOperator(TierCOperator):
    operator_id = "paraphrase_lexico_syntactic_constrained"
    tier = Tier.C
    stochastic = True

    _PROMPT_TEMPLATE = """- Variation: Constrained paraphrase (lexico-syntactic).
- Goal: Restate the task in different words while keeping semantics, intent, tone, register, constraints, entities, and formatting unchanged.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints in the task text
  • If the task text includes multiple-choice options, reproduce the options block exactly as given (text, markers/labels, order, punctuation unchanged); apply edits only to the stem/context.
- Techniques (ONLY if semantics remain identical):
  • Use natural same-register synonyms and equivalent phrasing
  • Reorder words/phrases or switch active↔passive where appropriate without changing scope
  • Convert clause↔phrase; adjust punctuation for fluency
  • Keep length approximately similar (±10–20%)
- Do NOT: add/remove facts or examples; change definitions; reorder logical steps.
- If strict semantic equivalence cannot be maintained, return the original task exactly.

Original task:
{{ prompt }}"""

    @property
    def prompt_template(self) -> str:
        return self._PROMPT_TEMPLATE

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # C1: Text length ≥ 4 tokens (inherited from base).
        n_tokens = len(text.strip().split())
        if n_tokens < 4:
            return PreCheckResult(
                passed=False, reason="Text too short (min 4 tokens)"
            )
        # C1: Excluded if text contains code blocks, math expressions.
        if _CODE_BLOCK_RE.search(text):
            return PreCheckResult(
                passed=False, reason="Text contains code block"
            )
        if _INLINE_CODE_RE.search(text):
            return PreCheckResult(
                passed=False, reason="Text contains inline code"
            )
        if _MATH_EXPR_RE.search(text):
            return PreCheckResult(
                passed=False, reason="Text contains formal mathematical expression"
            )
        # C1: Excluded for tasks evaluating paraphrase/lexical-substitution.
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in settings.PARAPHRASE_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (evaluates paraphrase/lexical-substitution)",
            )
        return PreCheckResult(passed=True, details={"n_tokens": n_tokens})