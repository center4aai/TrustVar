import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from nltk.corpus import wordnet as _wn

from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierBOperator,
    VariationResult,
)
from src.core.taxonomy import resolve_task_semantics
from src.core.operators.utils.nlp_utils import (
    detect_lang,
    get_stanza_pipeline,
    inflect,
    parse_feats,
    parse_ud,
)
from src.utils.logger import logger

_STANZA_TO_PYMORPHY3_CASE = {
    "Nom": "nomn",
    "Acc": "accs",
    "Gen": "gent",
    "Dat": "datv",
    "Ins": "ablt",
    "Loc": "loct",
    "Voc": "voct",
    "Par": "gent",
}

_STANZA_TO_PYMORPHY3_GENDER = {
    "Masc": "masc",
    "Fem": "femn",
    "Neut": "neut",
}

_STANZA_TO_PYMORPHY3_NUMBER = {
    "Sing": "sing",
    "Plur": "plur",
}


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_adjective_registry() -> Dict[str, Dict[str, List[str]]]:
    registry: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("en", "ru"):
        path = os.path.join(_DATA_DIR, f"{lang}_neutral_adjectives.yaml")
        with open(path, encoding='utf-8') as f:
            registry[lang] = yaml.safe_load(f) or {}
    return registry


_ADJ_REGISTRY: Dict[str, Dict[str, List[str]]] = _load_adjective_registry()

_EXCLUDED_TASK_TYPES = {"sentiment", "toxicity", "hate_speech", "affective", "emotion"}


def _classify_noun_en_wordnet(token) -> Optional[str]:
    lemma = getattr(token, "lemma", None) or getattr(token, "text", None)
    if not lemma:
        return None
    synsets = _wn.synsets(lemma, pos=_wn.NOUN)
    if not synsets:
        return None
    lexname = synsets[0].lexname()
    if lexname in ("noun.person", "noun.animal"):
        return "animate"
    if lexname in {
        "noun.cognition", "noun.communication", "noun.event",
        "noun.feeling", "noun.attribute", "noun.quantity",
        "noun.relation", "noun.time", "noun.process", "noun.state",
        "noun.act", "noun.group", "noun.possession", "noun.phenomenon",
        "noun.motive", "noun.Tops",
    }:
        return "abstract"
    return "concrete_inanimate"


def _classify_noun(token, lang: str) -> str:
    # PROPN is filtered out upstream in _find_noun_slots (NOUN-only candidates),
    # so _classify_noun only ever sees common nouns (B7-CLEANUP, S1.6).
    if token is None:
        return "concrete_inanimate"
    upos = getattr(token, "upos", None)
    if upos == "NOUN":
        feats = str(getattr(token, "feats", "") or "")
        if "Anim" in feats:
            return "animate"
        if "Abstr" in feats:
            return "abstract"
        if lang == "en":
            wn_result = _classify_noun_en_wordnet(token)
            if wn_result is not None:
                return wn_result
        return "concrete_inanimate"
    return "concrete_inanimate"


def _find_noun_slots(text: str, lang: str) -> List[Dict[str, Any]]:
    """Find candidate NP slots for modifier insertion.
    """
    if get_stanza_pipeline(lang) is None:
        return []

    doc = parse_ud(text, lang)
    if doc is None:
        return []

    nouns: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        for token in sentence.words:
            if token.upos not in ("NOUN",):
                continue
            has_amod = any(
                w.head == token.id and w.deprel == "amod" for w in sentence.words
            )
            if has_amod:
                continue
            entry: Dict[str, Any] = {
                "text": token.text,
                "token": token,
                "lang": lang,
                "start_char": token.start_char
                if hasattr(token, "start_char")
                else None,
                "end_char": token.end_char if hasattr(token, "end_char") else None,
                "feats": parse_feats(str(token.feats or "")),
            }
            nouns.append(entry)
    return nouns


def _all_registry_adjectives(lang: str) -> Set[str]:
    reg = _ADJ_REGISTRY.get(lang, _ADJ_REGISTRY["en"])
    result: Set[str] = set()
    for cat in reg.values():
        result.update(w.lower() for w in cat)
    return result


def _select_category_adjectives(
    registry: Dict[str, List[str]], requested: str
) -> Tuple[List[str], str]:
    """Pick adjective pool for a noun category.

    """
    requested_pool = list(registry.get(requested, []))
    if requested_pool:
        return requested_pool, requested
    fallback_pool = list(registry.get("concrete_inanimate", []))
    return fallback_pool, "concrete_inanimate"


class ControlledDescriptiveModifierInsertionOperator(TierBOperator):
    operator_id = "controlled_descriptive_modifier_insertion"
    tier = Tier.B

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        # S1.1: gate on the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics and task_semantics.lower() in _EXCLUDED_TASK_TYPES:
            return PreCheckResult(
                passed=False, reason=f"Excluded for task semantics: {task_semantics}"
            )

        lang = language or detect_lang(text)
        if get_stanza_pipeline(lang) is None:
            return PreCheckResult(passed=False, reason="no_parser")

        nouns = _find_noun_slots(text, lang)
        if not nouns:
            return PreCheckResult(passed=False, reason="No suitable noun slot found")
        return PreCheckResult(
            passed=True, details={"nouns": len(nouns), "language": lang}
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        lang = language or detect_lang(text)

        if get_stanza_pipeline(lang) is None:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_parser"},
                original_text=text,
            )

        rng = random.Random(seed)

        nouns = _find_noun_slots(text, lang)
        if not nouns:
            return VariationResult(
                variant_text=text, metadata={"skipped": "no_nouns"}, original_text=text
            )

        registry = _ADJ_REGISTRY.get(lang, _ADJ_REGISTRY["en"])
        all_adjs: List[str] = []
        for cat in registry.values():
            all_adjs.extend(cat)
        if not all_adjs:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_adjectives"},
                original_text=text,
            )

        rng.shuffle(nouns)
        target = nouns[0]
        noun_type = _classify_noun(target.get("token"), lang)
        requested_category = (
            "animate"
            if noun_type == "animate"
            else "abstract"
            if noun_type == "abstract"
            else "concrete_inanimate"
        )
        candidates, actual_category = _select_category_adjectives(
            registry, requested_category
        )
        if not candidates:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_adjectives_for_category"},
                original_text=text,
            )
        if requested_category == "animate" and actual_category != "animate":
            logger.debug(
                "B7: animate noun category has empty registry; "
                "falling back to concrete_inanimate (implicature risk per spec)."
            )
        rng.shuffle(candidates)
        adj = candidates[0]

        noun_text = target["text"]
        start_char = target.get("start_char")

        if start_char is not None:
            idx = start_char
        else:
            idx = text.find(noun_text)
            if idx < 0:
                idx = text.lower().find(noun_text.lower())
            if idx < 0 and noun_text:
                for word in noun_text.split():
                    idx2 = text.lower().find(word.lower())
                    if idx2 >= 0:
                        idx = idx2
                        noun_text = word
                        break

        if idx is None or idx < 0:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "noun_not_found_in_text"},
                original_text=text,
            )

        if lang == "ru":
            noun_feats = target.get("feats", {})
            target_case_raw = noun_feats.get("Case", "Nom")
            target_gender_raw = noun_feats.get("Gender", "Masc")
            target_number_raw = noun_feats.get("Number", "Sing")
            target_case = _STANZA_TO_PYMORPHY3_CASE.get(
                target_case_raw, target_case_raw.lower()
            )
            target_gender = _STANZA_TO_PYMORPHY3_GENDER.get(
                target_gender_raw, target_gender_raw.lower()
            )
            target_number = _STANZA_TO_PYMORPHY3_NUMBER.get(
                target_number_raw, target_number_raw.lower()
            )
            adj_form = inflect(
                adj,
                {
                    "case": target_case,
                    "gender": target_gender,
                    "number": target_number,
                },
                lang="ru",
            )
            if adj_form == adj:
                adj_form = inflect(
                    adj,
                    {
                        "case": target_case,
                        "number": target_number,
                    },
                    lang="ru",
                )
        else:
            adj_form = adj

        if idx > 0 and not text[idx - 1].isspace():
            variant = text[:idx] + " " + adj_form + " " + text[idx:]
        else:
            variant = text[:idx] + adj_form + " " + text[idx:]

        meta: Dict[str, Any] = {
            "adjective": adj_form,
            "noun": noun_text,
            "noun_category": actual_category,
            "noun_category_requested": requested_category,
            "language": lang,
        }
        if lang == "ru":
            nf = target.get("feats", {})
            meta["noun_case"] = nf.get("Case", "Nom")
            meta["noun_gender"] = nf.get("Gender", "Masc")
            meta["noun_number"] = nf.get("Number", "Sing")
        return VariationResult(
            variant_text=variant,
            metadata=meta,
            original_text=text,
        )

