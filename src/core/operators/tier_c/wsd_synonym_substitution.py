import random
import re
from typing import Any, Dict, List, Optional

from src.config.settings import get_settings
from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierCOperator,
    VariationResult,
)
from src.core.operators.utils.nlp_utils import (
    _EN_REGISTER_MAP,
    _EN_SUBSTITUTION_FILTER_WORDS,
    _RU_REGISTER_MAP,
    _RU_SUBSTITUTION_FILTER_WORDS,
    check_ru_lexicon,
    detect_lang,
    en_inflect,
    get_synset_count_en,
    get_synset_count_ru,
    has_fixed_expression,
    parse_ud,
    pick_synonym,
    ru_inflect_synonym,
)
from src.core.operators.utils.wsd import (
    disambiguate,
    get_context_window,
)

settings = get_settings()


class WsdSynonymSubstitutionOperator(TierCOperator):
    operator_id = "wsd_synonym_substitution"
    tier = Tier.C
    stochastic = True
    # WSD is fully symbolic; apply() is fully overridden and does not call
    # self.prompt_template. The property below exists only to satisfy the
    # TierCOperator abstract interface contract.
    _NO_LLM_PROMPT = (
        "[wsd_synonym_substitution] Symbolic operator; no LLM prompt template."
    )

    WSD_CONFIDENCE_THRESHOLD = {
        "en": settings.WSD_CONFIDENCE_THRESHOLD_EN,
        "ru": settings.WSD_CONFIDENCE_THRESHOLD_RU,
    }

    TECH_AFFIX_RE = re.compile(
        r"\b\w*(?:ology|itis|osis|genic|lysis|trophic|"
        r"ectomy|ostomy|otomy|plasty|rrhaphy|rrhea|rrhexis|"
        r"scope|graphy|gram|metry|nomy|asis|esis|iasis|"
        r"algia|dynia|oma|pathy|penia|phagia|phasia|"
        r"phobia|plasia|plegia|pnea|ptosis|rrhagia|spasm|stasis|"
        r"trophy|uria|"
        r"ware|code|data|stack|queue|poly|nomial|"
        r"throughput|endpoint|bandwidth|protocol|"
        r"aintiff|efendant|antiff)\b",
        re.IGNORECASE,
    )

    def __init__(self, max_substitutions: int = 2, wsd_confidence_min: Optional[float] = None):
        super().__init__()
        self.max_substitutions = max_substitutions
        self.wsd_confidence_min = wsd_confidence_min

    @property
    def prompt_template(self) -> str:
        return self._NO_LLM_PROMPT

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)
        if len(text.split()) < 5:
            return PreCheckResult(passed=False, reason="Text too short")

        expr = has_fixed_expression(text, lang)
        if expr:
            return PreCheckResult(
                passed=False, reason=f"Fixed expression detected: {expr}"
            )

        # C7: Terminology dictionary gate — reject if text contains high
        # density of technical terms (≥30% of content words). This prevents
        # WSD from mis-substituting domain-locked vocabulary.
        words = text.strip().split()
        if len(words) >= 10:
            tech_count = sum(1 for w in words if self.TECH_AFFIX_RE.search(w))
            if tech_count / len(words) >= 0.30:
                return PreCheckResult(
                    passed=False,
                    reason=f"High technical term density ({tech_count}/{len(words)})",
                )

        doc = parse_ud(text, lang)
        if doc is None:
            return PreCheckResult(passed=False, reason="Stanza pipeline unavailable")

        content_words = [
            t
            for sent in doc.sentences
            for t in sent.words
            if t.upos in ("NOUN", "VERB", "ADJ", "ADV") and t.upos != "PROPN"
        ]
        if not content_words:
            return PreCheckResult(passed=False, reason="No content words found")

        viable = 0
        for t in content_words:
            lemma = (t.lemma or t.text or "").lower()
            filter_words = (
                _RU_SUBSTITUTION_FILTER_WORDS
                if lang == "ru"
                else _EN_SUBSTITUTION_FILTER_WORDS
            )
            if lemma in filter_words:
                continue
            if lang == "ru":
                synset_count = get_synset_count_ru(lemma, t.upos)
            else:
                synset_count = get_synset_count_en(lemma, t.upos)
            if synset_count <= 1:
                continue
            viable += 1
            if viable >= self.max_substitutions:
                break

        if viable == 0:
            return PreCheckResult(
                passed=False, reason="No viable polysemous substitution candidates"
            )

        return PreCheckResult(
            passed=True,
            details={
                "content_words": len(content_words),
                "language": lang,
            },
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        lang = language or detect_lang(text)
        rng = random.Random(seed)
        conf_min = (
            self.wsd_confidence_min
            if self.wsd_confidence_min is not None
            else self.WSD_CONFIDENCE_THRESHOLD.get(lang, 0.65)
        )

        doc = parse_ud(text, lang)
        if doc is None:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_stanza"},
                original_text=text,
            )

        candidates: List[tuple] = []

        for sent in doc.sentences:
            for word in sent.words:
                if word.upos not in ("NOUN", "VERB", "ADJ", "ADV"):
                    continue
                if word.upos == "PROPN":
                    continue
                if word.text and word.text[0].isupper() and word.id > sent.words[0].id:
                    continue

                lemma = (word.lemma or word.text or "").lower()
                filter_words = (
                    _RU_SUBSTITUTION_FILTER_WORDS
                    if lang == "ru"
                    else _EN_SUBSTITUTION_FILTER_WORDS
                )
                if not lemma or lemma in filter_words:
                    continue

                if lang == "ru":
                    synset_count = get_synset_count_ru(lemma, word.upos)
                else:
                    synset_count = get_synset_count_en(lemma, word.upos)
                if synset_count <= 1:
                    continue

                ctx = get_context_window(word, sent)
                wsd = disambiguate(lemma, word.upos, ctx, lang)
                if wsd is None:
                    continue

                wsd_score = float(wsd.get("score", 1.0))
                if wsd_score < conf_min:
                    continue

                synonyms = wsd.get("synonyms", [])
                register_map = _RU_REGISTER_MAP if lang == "ru" else _EN_REGISTER_MAP
                chosen = pick_synonym(lemma, synonyms, register_map, rng)
                if chosen is None or chosen == lemma:
                    continue

                inflected = (
                    ru_inflect_synonym(chosen, str(word.feats))
                    if lang == "ru"
                    else en_inflect(chosen, word.upos, str(word.feats))
                )
                if inflected.lower() == lemma:
                    continue

                candidates.append((word, inflected, lemma, wsd["synset_id"]))

        if not candidates:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_substitutions"},
                original_text=text,
            )

        rng.shuffle(candidates)
        selected = candidates[: self.max_substitutions]

        result = text
        substitutions: List[Dict[str, Any]] = []
        for word, replacement, orig_lemma, synset_id in reversed(selected):
            if word.text and word.text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result = result[: word.start_char] + replacement + result[word.end_char :]
            substitutions.append(
                {
                    "original": word.text,
                    "replacement": replacement,
                    "pos": word.upos,
                    "synset_id": synset_id,
                }
            )

        return VariationResult(
            variant_text=result,
            metadata={
                "substitutions": substitutions,
                "count": len(substitutions),
                "language": lang,
                "max_substitutions": self.max_substitutions,
                "wsd_confidence_min": conf_min,
            },
            original_text=text,
        )


