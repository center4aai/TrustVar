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
    r"\$[^$\n]+\$|"
    r"\\[a-zA-Z]+|"
    r"=\s*[\d\.]+\s*[+\-*/]\s*[\d\.]+)"
)


class ParaphraseFreeOperator(TierCOperator):
    operator_id = "paraphrase_free"
    tier = Tier.C
    stochastic = True

    _PROMPT_TEMPLATE = """- Variation: Free paraphrase.
- Goal: Rewrite the task in completely different wording while preserving the core meaning, intent, and answering constraints.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If the task text includes multiple-choice options, reproduce the options block exactly as given.
- Techniques:
  • May use substantial lexical and syntactic restructuring
  • May change clause order, voice, or discourse structure
  • May add or remove discourse markers for fluency
  • Length may vary more freely (±30%) but do not add or remove factual content
- Do NOT: change the gold answer label; add external knowledge or examples.
- If semantic equivalence cannot be maintained, return the original task exactly.

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
        n_tokens = len(text.strip().split())
        if n_tokens < 4:
            return PreCheckResult(
                passed=False, reason="Text too short (min 4 tokens)"
            )
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
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in settings.PARAPHRASE_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (evaluates paraphrase/lexical-substitution)",
            )
        return PreCheckResult(passed=True, details={"n_tokens": n_tokens})