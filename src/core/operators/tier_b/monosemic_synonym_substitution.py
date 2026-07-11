import random
import re
from typing import Any, Dict, List, Optional

from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierBOperator,
    VariationResult,
)
from src.core.operators.utils.nlp_utils import (
    _EN_REGISTER_MAP,
    _EN_SUBSTITUTION_FILTER_WORDS,
    _RU_REGISTER_MAP,
    _RU_SUBSTITUTION_FILTER_WORDS,
    detect_lang,
    en_inflect,
    get_synonyms_from_synset_en,
    get_synonyms_from_synset_ru,
    get_unique_synset_en,
    get_unique_synset_ru,
    has_fixed_expression,
    parse_ud,
    pick_synonym,
    ru_inflect_synonym,
)


class MonosemicSynonymSubstitutionOperator(TierBOperator):
    operator_id = "monosemic_synonym_substitution"
    tier = Tier.B
    stochastic = True

    def __init__(self, max_substitutions: int = 2):
        super().__init__()
        self.max_substitutions = max_substitutions

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
            synset = (
                get_unique_synset_ru(lemma, t.upos)
                if lang == "ru"
                else get_unique_synset_en(lemma, t.upos)
            )
            if synset is None:
                continue
            synonyms = (
                get_synonyms_from_synset_ru(synset)
                if lang == "ru"
                else get_synonyms_from_synset_en(synset)
            )
            register_map = _RU_REGISTER_MAP if lang == "ru" else _EN_REGISTER_MAP
            candidate = pick_synonym(lemma, synonyms, register_map, random.Random())
            if candidate is None or candidate == lemma:
                continue
            viable += 1
            if viable >= self.max_substitutions:
                break

        if viable == 0:
            return PreCheckResult(
                passed=False, reason="No viable monosemic substitution candidates"
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

                synset = (
                    get_unique_synset_ru(lemma, word.upos)
                    if lang == "ru"
                    else get_unique_synset_en(lemma, word.upos)
                )
                if synset is None:
                    continue

                synonyms = (
                    get_synonyms_from_synset_ru(synset)
                    if lang == "ru"
                    else get_synonyms_from_synset_en(synset)
                )
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

                synset_id = synset.id if hasattr(synset, "id") else synset.name()
                candidates.append((word, inflected, lemma, synset_id))

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
            },
            original_text=text,
        )


