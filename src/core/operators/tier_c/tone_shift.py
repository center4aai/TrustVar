import re
from typing import Optional

from src.core.operators.base import PreCheckResult, TierCOperator, Tier, VariationResult
from src.core.taxonomy import resolve_task_semantics


def _tone_template(target_tone: str) -> str:
    # TECH-003: "Avoid emotionally charged language" only applies when neutralising;
    # for emotional_positive/negative targets it contradicts the transformation goal.
    avoid_emotional = (
        "\n  • Avoid emotionally charged language unless already present"
        if target_tone == "neutral"
        else ""
    )
    return f"""- Variation: Tone shift to {target_tone}.
- Goal: Change the emotional tone of the task to {target_tone} while keeping the factual content, intent, constraints, entities, and formatting unchanged.
- Output language must equal input language.
- Preserve EXACTLY:
  • Placeholders in curly braces
  • Named entities/terms, technical symbols, equations/code, citations/URLs, quoted strings
  • Numbers, units, dates, negations, quantifiers, conditionals, comparisons, logical relations
  • Any embedded format or answering constraints
  • If options are present, reproduce the options block exactly.
- Techniques:
  • Adjust affective vocabulary to suit {target_tone} tone{avoid_emotional}
  • Keep length approximately similar (±10–20%)
- Do NOT: add/remove facts; change the gold answer; reorder logical steps.
- If strict equivalence cannot be maintained, return the original exactly.

Original task:
{{{{ prompt }}}}"""


_TONE_EXCLUDED_TASK_TYPES = {
    "sentiment_classification",
    "sentiment_analysis",
    "hate_speech_detection",
    "toxicity_classification",
    "emotion_recognition",
    "subjectivity_classification",
    "affect_classification",
    "polarity_classification",
    "sarcasm_detection",
    "irony_detection",
    "offensive_language_detection",
}


_EN_AFFECTIVE_MARKERS = re.compile(
    r"\b(amazing|awful|terrible|horrible|fantastic|outstanding|disgusting|"
    r"wonderful|appalling|excellent|brilliant|hilarious|furious|enraged|"
    r"ecstatic|devastated|heartbroken|thrilled|disappointed|angry|happy|"
    r"sad|love|hate|adore|despise|terrific|horrendous|phenomenal)\b",
    re.IGNORECASE,
)
_RU_AFFECTIVE_MARKERS = re.compile(
    r"\b(ужасн\w+|прекрасн\w+|отвратительн\w+|великолепн\w+|восхитительн\w+|"
    r"потрясающ\w+|кошмарн\w+|омерзительн\w+|невероятн\w+|фантастическ\w+|"
    r"обожа\w+|ненави\w+|любл\w+|презира\w+|разочаров\w+|счастлив\w+|"
    r"грустн\w+|зл\w+|довол\w+)\b",
    re.IGNORECASE,
)


class ToneShiftOperator(TierCOperator):
    operator_id = "tone_shift"
    tier = Tier.C
    stochastic = True

    _VALID_TARGET_TONES = ("neutral", "emotional_positive", "emotional_negative")

    def __init__(self, target_tone: str = "neutral"):
        if target_tone not in self._VALID_TARGET_TONES:
            raise ValueError(
                f"ToneShiftOperator: target_tone must be one of "
                f"{self._VALID_TARGET_TONES}, got {target_tone!r}"
            )
        self.target_tone = target_tone

    @property
    def prompt_template(self) -> str:
        return _tone_template(self.target_tone)

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # C5: Text length ≥ 8 tokens
        n_tokens = len(text.strip().split())
        if n_tokens < 8:
            return PreCheckResult(
                passed=False,
                reason=f"Text too short for tone shift ({n_tokens} tokens, min 8)",
            )
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in _TONE_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics {task_semantics!r} excluded (affect is label-relevant)",
            )

        lang = (language or "en").lower()
        target_tone = kwargs.get("target_tone", self.target_tone)
        if target_tone != "neutral":
            markers = (
                _RU_AFFECTIVE_MARKERS if lang == "ru" else _EN_AFFECTIVE_MARKERS
            )
            match = markers.search(text)
            if match:
                return PreCheckResult(
                    passed=False,
                    reason=f"Source already contains strong affective language: {match.group(0)!r}",
                )
        return PreCheckResult(
            passed=True,
            details={"target_tone": self.target_tone, "language": lang},
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        adapter=None,
        **kwargs,
    ) -> VariationResult:
        from jinja2 import Template

        # ARCH-001: build prompt from effective tone without mutating self.target_tone —
        # safe for concurrent asyncio coroutines sharing the same operator instance.
        target_tone = kwargs.pop("target_tone", None) or self.target_tone
        prompt = Template(_tone_template(target_tone)).render(prompt=text)

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