import re
from typing import Any, Dict, Optional, Set, Tuple

from src.core.operators.base import TierCOperator, Tier, PreCheckResult, VariationResult
from src.core.taxonomy import resolve_task_semantics
from src.core.operators.utils.nlp_utils import detect_lang


# Lazy spaCy loader (avoids eager import failure when model not installed)
_nlp_en = None


def _get_nlp_en():
    global _nlp_en
    if _nlp_en is None:
        import spacy as _spacy
        try:
            _nlp_en = _spacy.load("en_core_web_sm")
        except OSError:
            _nlp_en = None
    return _nlp_en


class BackTranslationSinglePivotOperator(TierCOperator):
    """C8: back_translation_single_pivot.
    The operator's apply() is fully overridden (two-leg MT round-trip).
    prompt_template exists only to satisfy TierCOperator abstract interface.
    Validation is delegated to the Tier C three-layer cascade; operator-level validate() keeps
    lightweight symbolic sanity checks (NE/numeric/length) as fast
    rejection pre-filters.
    """
    operator_id = "back_translation_single_pivot"
    tier = Tier.C
    _NO_LLM_PROMPT = "[back_translation_single_pivot] MT-based operator; no LLM prompt template."
    _CROSS_LINGUAL_EXCLUDED_TASK_TYPES = {
        "translation", "machine_translation", "multilingual_benchmark",
        "cross_lingual_retrieval", "cross_lingual_nli", "cross_lingual_qa",
        "mt_evaluation",
    }

    @property
    def prompt_template(self) -> str:
        return self._NO_LLM_PROMPT

    CODE_RE = re.compile(r"(```|`{1,2}[^`]+`{1,2}|<code>)", re.IGNORECASE)
    URL_RE = re.compile(r"https?://\S+")
    EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    NUM_RE = re.compile(r"\b\d+\b")
    ENTITY_CAP_RE = re.compile(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*")
    ENTITY_CAP_RU_RE = re.compile(r"[А-Я][а-яё]+(?:\s[А-Я][а-яё]+)*")
    QUOTED_RE = re.compile(r""".*?"|'.*?'|«.*?»""")
    PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]{}<>]")

    # Technical term detection (mirrors C7 pattern, adapted for C8 MT context).
    # Used in check_preconditions to gate against high-density technical text.
    TECH_AFFIX_RE = re.compile(
        r"\b\w*(?:ology|itis|osis|genic|lysis|trophic|"
        r"ectomy|ostomy|otomy|plasty|rrhaphy|rrhea|rrhexis|"
        r"scope|graphy|gram|metry|nomy|asis|esis|iasis|"
        r"algia|dynia|oma|pathy|penia|phagia|phasia|"
        r"phobia|plasia|plegia|pnea|ptosis|rrhagia|spasm|stasis|"
        r"trophy|uria|"
        r"ware|code|data|stack|queue|poly|nomial|"
        r"throughput|endpoint|bandwidth|protocol|"
        r"aintiff|efendant|antiff|"
        r"gorithm|rithm|tcp|udp|dhcp)\b",
        re.IGNORECASE,
    )
    TECH_DENSITY_THRESHOLD = 0.30

    def __init__(self):
        super().__init__()
        # Instance-level translate prompt cache (was class-level mutable dict — NEW-6 fix)
        self._translate_prompts: Dict[str, str] = {}

    async def _get_prompt(self, text: str, src: str, tgt: str) -> str:
        key = f"{src}_{tgt}"
        if key not in self._translate_prompts:
            if tgt == "ru":
                self._translate_prompts[key] = (
                    "Translate the following text to Russian. "
                    "Preserve ALL named entities, numbers, dates, and technical terms exactly. "
                    "Output ONLY the translation, nothing else:\n\n{text}"
                )
            else:
                self._translate_prompts[key] = (
                    "Translate the following text to English. "
                    "Preserve ALL named entities, numbers, dates, and technical terms exactly. "
                    "Output ONLY the translation, nothing else:\n\n{text}"
                )
        return self._translate_prompts[key].format(text=text)

    def _extract_entities(self, text: str, lang: str) -> Dict[str, Set[str]]:
        result = {"named": set(), "numbers": set(), "emails": set(), "urls": set()}
        result["numbers"] = set(self.NUM_RE.findall(text))
        result["emails"] = set(self.EMAIL_RE.findall(text))
        result["urls"] = set(self.URL_RE.findall(text))

        if lang == "ru":
            try:
                from natasha import NewsNERTagger, NewsMorphoTagger, Segmenter, NewsEmbedding, Doc
                emb = NewsEmbedding()
                segmenter = Segmenter()
                ner_tagger = NewsNERTagger(emb)
                natasha_doc = Doc(text)
                natasha_doc.segment(segmenter)
                natasha_doc.tag_ner(ner_tagger)
                result["named"] = {sp.text for sp in natasha_doc.spans if sp.type in ("PER", "ORG", "LOC")}
            except Exception:
                result["named"] = set(self.ENTITY_CAP_RU_RE.findall(text))
        else:
            nlp = _get_nlp_en()
            if nlp is not None:
                try:
                    doc = nlp(text)
                    result["named"] = {ent.text for ent in doc.ents}
                except Exception:
                    pass
            if not result["named"]:
                result["named"] = set(self.ENTITY_CAP_RE.findall(text))

        return result

    def _restore_entities(
        self, original: str, back: str, ents: Dict[str, Set[str]]
    ) -> Tuple[str, Dict[str, Any]]:
        result = back
        restore_meta: Dict[str, Any] = {}

        missing_numbers = [
            num for num in sorted(ents["numbers"], key=len, reverse=True)
            if num not in result
        ]
        if missing_numbers:
            restore_meta["numbers_lost"] = missing_numbers
            restore_meta["numbers_flag"] = True

        # NAMED ENTITIES: case-preserving substitution for entities that appear
        # in the back-translation with wrong casing (common MT artifact).
        for entity in sorted(ents["named"], key=len, reverse=True):
            entity_lower_in_back = entity.lower() in result.lower()
            entity_exact_in_back = entity in result
            if not entity_exact_in_back and entity_lower_in_back:
                idx = result.lower().find(entity.lower())
                if idx >= 0:
                    result = result[:idx] + entity + result[idx + len(entity):]


        missing_emails = [email for email in ents["emails"] if email not in result]
        if missing_emails:
            restore_meta["emails_lost"] = missing_emails

        return result, restore_meta

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)

        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in self._CROSS_LINGUAL_EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Excluded for cross-lingual task semantics: {task_semantics}",
            )

        if len(text.split()) < 4:
            return PreCheckResult(passed=False, reason="Too short")
        if self.CODE_RE.search(text):
            return PreCheckResult(passed=False, reason="Code block")
        if self.URL_RE.search(text):
            return PreCheckResult(passed=False, reason="URL")
        if lang not in ("en", "ru"):
            return PreCheckResult(passed=False, reason=f"Unsupported language: {lang}")


        words = text.strip().split()
        if len(words) >= 10:
            tech_count = sum(1 for w in words if self.TECH_AFFIX_RE.search(w))
            if tech_count / len(words) >= self.TECH_DENSITY_THRESHOLD:
                return PreCheckResult(
                    passed=False,
                    reason=f"High technical term density ({tech_count}/{len(words)})",
                )

        return PreCheckResult(passed=True, details={"language": lang})

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        adapter=None,
        **kwargs,
    ) -> VariationResult:
        lang = language or detect_lang(text)
        pivot = "ru" if lang == "en" else "en"

        if adapter is None:
            return VariationResult(variant_text=text, metadata={"skipped": "no_adapter"}, original_text=text)

        ents_before = self._extract_entities(text, lang)

        leg1 = await self._get_prompt(text, lang, pivot)
        try:
            intermediate = (await adapter.generate(leg1, temperature=0.3, max_tokens=512)).strip()
        except Exception:
            return VariationResult(variant_text=text, metadata={"skipped": "first_leg_failed"}, original_text=text)

        leg2 = await self._get_prompt(intermediate, pivot, lang)
        try:
            back = (await adapter.generate(leg2, temperature=0.3, max_tokens=512)).strip()
        except Exception:
            return VariationResult(variant_text=text, metadata={"skipped": "second_leg_failed"}, original_text=text)

        back, restore_meta = self._restore_entities(text, back, ents_before)

        ents_after = self._extract_entities(back, lang)
        ne_restored = len(ents_before["named"] & ents_after["named"])
        num_restored = len(ents_before["numbers"] & ents_after["numbers"])

        metadata: Dict[str, Any] = {
            "src": lang,
            "pivot": pivot,
            "ne_before": len(ents_before["named"]),
            "ne_after": len(ents_after["named"]),
            "ne_restored": ne_restored,
            "num_restored": num_restored,
        }
        metadata.update(restore_meta)

        return VariationResult(
            variant_text=back,
            metadata=metadata,
            original_text=text,
        )


