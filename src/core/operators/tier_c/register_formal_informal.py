from typing import Optional

from src.core.operators.base import PreCheckResult, TierCOperator, Tier, VariationResult
from src.core.taxonomy import resolve_task_semantics


_FORMAL_TEMPLATE = """- Variation: Style change to formal.
- Goal: Rewrite the task in a formal register while keeping meaning, intent, constraints, entities, and formatting unchanged.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- Techniques:
  • Adjust lexicon and phrasing to the formal register
  • Expand contractions; prefer Latinate vocabulary; avoid colloquialisms
  • Reshape sentence structure to suit the register
  • Keep length approximately similar (±10–20%)
- For Russian (RU): use formal 'Вы' pronouns and corresponding verb morphology; avoid 'ты' forms.
- Do NOT: add/remove facts; reorder logical steps; alter fixed expressions.
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{ prompt }}"""


_INFORMAL_TEMPLATE = """- Variation: Style change to informal.
- Goal: Rewrite the task in an informal register while keeping meaning, intent, constraints, entities, and formatting unchanged.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- Techniques:
  • Adjust lexicon and phrasing to the informal register
  • Use contractions; prefer everyday vocabulary; avoid overly formal terms
  • Reshape sentence structure to suit the register
  • Keep length approximately similar (±10–20%)
- For Russian (RU): use informal 'ты' pronouns and corresponding verb morphology.
- Do NOT: add/remove facts; reorder logical steps; alter fixed expressions.
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{ prompt }}"""


_REGISTER_EXCLUDED_TASK_TYPES = {
    "formality_classification",
    "register_identification",
    "style_classification",
    "social_relationship_detection",
    "dialogue_social_inference",
    "politeness_detection",
}


class RegisterFormalInformalOperator(TierCOperator):
    operator_id = "register_formal_informal"
    tier = Tier.C
    stochastic = True

    _TEMPLATES = {
        "formal": _FORMAL_TEMPLATE,
        "informal": _INFORMAL_TEMPLATE,
    }

    def __init__(self, direction: str = "formal", skip_same_register: bool = False):
        if direction not in self._TEMPLATES:
            raise ValueError(
                f"RegisterFormalInformalOperator: direction must be one of "
                f"{list(self._TEMPLATES)}, got {direction!r}"
            )
        self.direction = direction
        self.skip_same_register = skip_same_register

    @property
    def prompt_template(self) -> str:
        return self._TEMPLATES[self.direction]

    @staticmethod
    def _detect_source_register(text: str, language: str) -> str:
        import re as _re
        words = _re.findall(r"[а-яёa-z]+", text.lower())
        if language == "ru":
            formal_ru = {"вы", "вас", "вам", "ваш", "ваши", "пожалуйста",
                         "уважаемый", "благодарю", "прошу"}
            informal_ru = {"ты", "тебя", "тебе", "твой", "привет",
                          "пока", "давай", "спс"}
            formal_score = sum(1 for w in words if w in formal_ru)
            informal_score = sum(1 for w in words if w in informal_ru)
        else:
            formal_en = {"please", "kindly", "regarding", "hereby", "therein",
                        "thereafter", "pursuant", "request", "require", "provide"}
            informal_en = {"hey", "hi", "hello", "cool", "awesome", "gonna",
                          "wanna", "yeah", "nah", "thanks"}
            formal_score = sum(1 for w in words if w in formal_en)
            informal_score = sum(1 for w in words if w in informal_en)

        if formal_score > informal_score:
            return "formal"
        if informal_score > formal_score:
            return "informal"
        return "neutral"

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # ARCH-002: accept direction kwarg so gate matches the actual apply() direction.
        direction = kwargs.get("direction", self.direction)
        # C4: Text length ≥ 8 tokens (for reliable register detection)
        n_tokens = len(text.strip().split())
        if n_tokens < 8:
            return PreCheckResult(
                passed=False,
                reason=f"Text too short for register shift ({n_tokens} tokens, min 8)",
            )
        # C4: Excluded for tasks where register is the evaluation target
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in _REGISTER_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (evaluates register/role)",
            )
        # C4-001: Source register detection (lightweight heuristic)
        lang = (language or "en").lower()
        source_register = self._detect_source_register(text, lang)
        # METH-004: skip if source already matches target direction (avoids trivial no-change variants)
        if self.skip_same_register and source_register == direction:
            return PreCheckResult(
                passed=False,
                reason="same_register",
                details={"source_register": source_register, "direction": direction},
            )
        return PreCheckResult(
            passed=True,
            details={
                "direction": direction,
                "n_tokens": n_tokens,
                "source_register": source_register,
            },
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