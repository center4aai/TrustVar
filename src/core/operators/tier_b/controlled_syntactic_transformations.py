import os
import random
import re
from typing import Any, Dict, List, Optional

import yaml

from src.core.operators.base import TierBOperator, Tier, PreCheckResult, VariationResult
from src.core.operators.utils.nlp_utils import (
    detect_lang,
    get_stanza_pipeline,
    parse_ud,
    get_np_tokens,
    has_scope_operators,
    has_fixed_expression,
    tokenize,
)
from src.utils.logger import logger


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_dative_verbs_registry() -> Dict[str, Dict[str, Any]]:
    """Load per-verb dative alternation metadata.

    """
    path = os.path.join(_DATA_DIR, "en_dative_verbs.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


_DATIVE_VERBS_REGISTRY: Dict[str, Dict[str, Any]] = _load_dative_verbs_registry()


def _dative_verb_allowed(verb_lemma: str, target_form: str) -> bool:
    """Check if a verb accepts a specific dative alternation form.

    `target_form` ∈ {"double_object", "prepositional", "both"}.
    """
    entry = _DATIVE_VERBS_REGISTRY.get(verb_lemma.lower())
    if entry is None:
        return True
    allowed = entry.get("allowed_forms", "both")
    if allowed == "both":
        return True
    return allowed == target_form


_DATIVE_VERBS_FALLBACK = {
    "give", "send", "tell", "bring", "offer", "pass", "pay", "sell",
    "show", "teach", "lend", "buy", "get", "make", "write", "read",
    "throw", "hand", "award", "grant", "mail", "ship", "serve",
    "feed", "assign", "owe", "promise",
}

_EN_AUX_LEMMAS = {"have", "has", "had", "be", "am", "is", "are", "was", "were", "been", "being"}
_EN_MODAL_LEMMAS = {
    "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "do", "does", "did",
}

_WH_WORDS_EN = {"what", "who", "whom", "whose", "which", "where", "when", "why", "how"}
_WH_WORDS_RU = {"что", "кто", "кого", "кому", "кем", "чей", "какой", "который", "где", "куда", "откуда", "когда", "почему", "зачем", "как", "сколько"}

_SUBTRANSFORMATIONS = ("clefting", "dative_alternation", "topicalization", "rc_reduction", "rc_expansion", "wh_fronting")


def _np_text(head: Any, sentence: Any) -> str:
    tokens = get_np_tokens(head, sentence) if sentence else [head]
    return " ".join(w.text for w in tokens)


def _np_text_clean(head: Any, sentence: Any) -> str:
    tokens = get_np_tokens(head, sentence) if sentence else [head]
    return " ".join(w.text for w in tokens if not (hasattr(w, "deprel") and w.deprel == "case"))


def _verb_lemma(verb: Any) -> Optional[str]:
    if verb is None:
        return None
    return getattr(verb, "lemma", None)


def _find_clefting_candidates(doc) -> List[Dict[str, Any]]:
    """Find subject-verb-object candidates suitable for it-clefting."""
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words = {w.id: w for w in sentence.words}
        for token in sentence.words:
            if token.deprel != "nsubj" or token.upos not in ("NOUN", "PROPN"):
                continue
            if token.head not in words:
                continue
            verb = words[token.head]
            if verb.upos not in ("VERB", "AUX"):
                continue
            obj = next((w for w in sentence.words if w.head == verb.id and w.deprel == "obj"), None)
            if obj is not None:
                candidates.append({
                    "type": "clefting",
                    "verb": verb,
                    "subj": token,
                    "obj": obj,
                    "sentence": sentence,
                })
                break
    return candidates


def _find_clefting_candidates_ru(doc) -> List[Dict[str, Any]]:
    """Find subject-verb-object candidates for RU clefting via 'именно'."""
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words = {w.id: w for w in sentence.words}
        for token in sentence.words:
            if token.deprel != "nsubj" or token.upos not in ("NOUN", "PROPN"):
                continue
            if token.head not in words:
                continue
            verb = words[token.head]
            if verb.upos not in ("VERB", "AUX"):
                continue
            obj = next((w for w in sentence.words if w.head == verb.id and w.deprel == "obj"), None)
            if obj is not None:
                candidates.append({
                    "type": "clefting_ru",
                    "verb": verb,
                    "subj": token,
                    "obj": obj,
                    "sentence": sentence,
                })
                break
    return candidates


def _find_dative_candidates(text: str, lang: str, doc) -> List[Dict[str, Any]]:
    """Find dative-alternation candidates (double object ↔ prepositional)."""
    if lang == "ru":
        return []
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        for token in sentence.words:
            if token.upos not in ("VERB", "AUX"):
                continue
            lemma = token.lemma.lower()
            registry_verbs = (
                set(_DATIVE_VERBS_REGISTRY.keys())
                if _DATIVE_VERBS_REGISTRY
                else _DATIVE_VERBS_FALLBACK
            )
            if lemma not in registry_verbs:
                continue
            obj_direct = None
            obj_indirect = None
            is_prepositional = False
            for child in sentence.words:
                if child.head != token.id:
                    continue
                if child.deprel == "obj":
                    obj_direct = child
                elif child.deprel == "iobj":
                    obj_indirect = child
                elif child.deprel == "obl":
                    has_to = any(
                        gc.deprel == "case" and gc.text.lower() == "to"
                        for gc in sentence.words if gc.head == child.id
                    )
                    if has_to:
                        obj_indirect = child
                        is_prepositional = True
                    elif obj_indirect is None:
                        obj_indirect = child
            if obj_direct is not None and obj_indirect is not None:
                if not _dative_verb_allowed(
                    lemma, "prepositional" if is_prepositional else "double_object"
                ):
                    continue
                candidates.append({
                    "type": "dative_alternation",
                    "verb": token,
                    "subj": next(
                        (w for w in sentence.words if w.head == token.id and w.deprel == "nsubj"),
                        None,
                    ),
                    "obj_direct": obj_direct,
                    "obj_indirect": obj_indirect,
                    "is_prepositional": is_prepositional,
                    "sentence": sentence,
                })
    return candidates


def _find_topicalization_candidates(doc) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words = {w.id: w for w in sentence.words}
        for token in sentence.words:
            if token.deprel != "nsubj" or token.upos not in ("NOUN", "PROPN"):
                continue
            if token.head not in words:
                continue
            verb = words[token.head]
            if verb.upos not in ("VERB", "AUX"):
                continue
            obj = next((w for w in sentence.words if w.head == verb.id and w.deprel == "obj"), None)
            if obj is not None:
                candidates.append({
                    "type": "topicalization",
                    "verb": verb,
                    "subj": token,
                    "obj": obj,
                    "sentence": sentence,
                })
                break
    return candidates


def _find_rc_reduction_candidates(doc) -> List[Dict[str, Any]]:
    """Find restrictive relative clauses (with explicit relative marker)."""
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words_by_id = {w.id: w for w in sentence.words}
        for token in sentence.words:
            if token.deprel == "acl:relcl" and token.upos in ("VERB", "AUX"):
                head = words_by_id.get(token.head)
                if head is not None:
                    mark_token = _find_mark_token_for_relcl(token, sentence)
                    candidates.append({
                        "type": "rc_reduction",
                        "relcl_token": token,
                        "head": head,
                        "mark_token": mark_token,
                        "sentence": sentence,
                    })
                    break
    return candidates


def _find_wh_questions(doc, lang: str) -> List[Dict[str, Any]]:
    """Find sentences with wh-words in non-initial position (in-situ wh)."""
    wh_words = _WH_WORDS_RU if lang == "ru" else _WH_WORDS_EN
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words = sentence.words
        if not words:
            continue
        wh_tokens = [w for w in words if w.lemma.lower() in wh_words and w.text.lower() in wh_words]
        if not wh_tokens:
            continue
        wh_token = wh_tokens[0]
        if wh_token.id == 1:
            continue
        verb = next(
            (w for w in words if w.upos in ("VERB", "AUX") and w.deprel == "root"),
            None,
        )
        has_aux_or_modal = any(
            w.upos == "AUX" or (w.lemma and w.lemma.lower() in _EN_MODAL_LEMMAS)
            for w in words
        )
        if lang != "ru" and has_aux_or_modal and verb is not None and verb.upos != "AUX":
            continue
        subj = next(
            (w for w in words if w.head == (verb.id if verb else -1) and w.deprel in ("nsubj",)),
            None,
        )
        candidates.append({
            "type": "wh_fronting",
            "wh_token": wh_token,
            "verb": verb,
            "subj": subj,
            "sentence": sentence,
        })
    return candidates


def _apply_clefting(cand: Dict[str, Any]) -> Optional[str]:
    verb = cand["verb"]
    subj = cand["subj"]
    obj = cand["obj"]
    sentence = cand.get("sentence")
    sent_text = sentence.text if sentence is not None and hasattr(sentence, "text") else None
    if sent_text is None:
        return None
    subj_text = _np_text(subj, sentence)
    obj_text = _np_text(obj, sentence)
    if subj_text and subj_text[0].isupper():
        subj_text = subj_text[0].lower() + subj_text[1:]
    obj_end = max(
        (w.end_char for w in (get_np_tokens(obj, sentence) if sentence else [obj]) if hasattr(w, "end_char") and w.end_char is not None),
        default=None,
    )
    if obj_end is None or obj_end <= 0 or obj_end >= len(sent_text):
        return None
    tail = sent_text[obj_end:].lstrip()
    result = f"It was {obj_text} that {subj_text} {verb.text} {tail}".strip()
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)
    return result


def _apply_clefting_ru(cand: Dict[str, Any]) -> Optional[str]:
    verb = cand["verb"]
    subj = cand["subj"]
    obj = cand["obj"]
    sentence = cand.get("sentence")
    sent_text = sentence.text if sentence is not None and hasattr(sentence, "text") else None
    if sent_text is None:
        return None
    subj_text = _np_text(subj, sentence)
    obj_text = _np_text(obj, sentence)
    verb_text = verb.text
    tail_start = max(
        (w.end_char for w in (get_np_tokens(obj, sentence) if sentence else [obj]) if hasattr(w, "end_char") and w.end_char is not None),
        default=None,
    )
    tail = ""
    if tail_start is not None and 0 < tail_start < len(sent_text):
        tail = sent_text[tail_start:].rstrip()
    if tail:
        return f"Именно {obj_text} {verb_text} {subj_text}{tail}".strip()
    return f"Именно {obj_text} {verb_text} {subj_text}".strip()


def _apply_dative(cand: Dict[str, Any]) -> Optional[str]:
    verb = cand["verb"]
    subj = cand.get("subj")
    obj_direct = cand.get("obj_direct")
    obj_indirect = cand.get("obj_indirect")
    is_prepositional = cand.get("is_prepositional", False)
    if obj_direct is None or obj_indirect is None:
        return None
    sentence = cand.get("sentence")
    subj_text = _np_text(subj, sentence) if subj is not None else ""
    obj_direct_text = _np_text_clean(obj_direct, sentence)
    if is_prepositional:
        obj_indirect_text = _np_text_clean(obj_indirect, sentence)
        return f"{subj_text} {verb.text} {obj_indirect_text} {obj_direct_text}".strip()
    obj_indirect_text = _np_text_clean(obj_indirect, sentence)
    return f"{subj_text} {verb.text} {obj_direct_text} to {obj_indirect_text}".strip()


def _apply_topicalization(cand: Dict[str, Any]) -> Optional[str]:
    obj = cand["obj"]
    subj = cand["subj"]
    verb = cand["verb"]
    sentence = cand.get("sentence")
    obj_text = _np_text(obj, sentence)
    subj_text = _np_text(subj, sentence)
    return f"{obj_text}, {subj_text} {verb.text}".strip()


_RELATIVE_PRONOUN_LEMMAS = {
    "that", "which", "who", "whom", "whose",
    "который", "которого", "которому", "которым",
    "которой", "которую", "которая", "которое",
    "которые", "которых", "которыми",
}


def _is_relative_pronoun_lemma(lemma: Optional[str]) -> bool:
    if not lemma:
        return False
    return lemma.lower() in _RELATIVE_PRONOUN_LEMMAS


def _has_relative_pronoun_child(relcl: Any, sentence: Any) -> bool:
    """True iff `relcl` has a child with a relative-pronoun lemma
    (`mark`, `obj`, `nsubj`, or `iobj`).
    """
    if relcl is None or sentence is None:
        return False
    for w in sentence.words:
        if w.head != relcl.id:
            continue
        if w.deprel in ("mark", "obj", "nsubj", "iobj") and _is_relative_pronoun_lemma(
            getattr(w, "lemma", None)
        ):
            return True
    return False


def _find_mark_token_for_relcl(relcl: Any, sentence: Any) -> Optional[Any]:
    """Return the relative-pronoun token that introduces an `acl:relcl`.


    Stanza quirk: in subject-/object-relative constructions, the
    relative pronoun can be tagged as `obj`/`nsubj`/`iobj` instead of
    the canonical `mark` (e.g. "The book *that* John wrote" — "that"
    is parsed as `obj` of "wrote"). We accept any child whose lemma is
    a relative pronoun.
    """
    if relcl is None or sentence is None:
        return None
    for w in sentence.words:
        if w.head == relcl.id and w.deprel == "mark":
            return w
    for w in sentence.words:
        if w.head == relcl.id and w.deprel in ("obj", "nsubj", "iobj"):
            if _is_relative_pronoun_lemma(getattr(w, "lemma", None)):
                return w
    return None


def _apply_rc_reduction(cand: Dict[str, Any], source_text: str) -> Optional[str]:
    """Reduce a relative clause by removing its `mark` child.

  
    """
    relcl = cand.get("relcl_token")
    sentence = cand.get("sentence")
    mark_token = cand.get("mark_token") or _find_mark_token_for_relcl(relcl, sentence)
    if mark_token is None:
        return None
    if mark_token.start_char is None or mark_token.end_char is None:
        return None
    if mark_token.end_char > len(source_text):
        return None
    result = source_text[: mark_token.start_char] + source_text[mark_token.end_char :]
    result = re.sub(r"\s+", " ", result).strip()
    return result if result != source_text else None


def _apply_rc_expansion(
    cand: Dict[str, Any], source_text: str, lang: str
) -> Optional[str]:
    """Expand a reduced relative clause by inserting a relative marker.

    """
    relcl = cand.get("relcl_token")
    head = cand.get("head")
    if relcl is None or head is None:
        return None
    if cand.get("mark_token") is not None:
        return None
    if head.end_char is None:
        return None
    rel_marker = "который" if lang == "ru" else "that"
    insert_pos = head.end_char
    while insert_pos < len(source_text) and source_text[insert_pos] in (" ", "\t"):
        insert_pos += 1
    if insert_pos >= len(source_text):
        return None
    result = (
        source_text[:insert_pos] + rel_marker + " " + source_text[insert_pos:]
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result if result != source_text else None


def _apply_wh_fronting(cand: Dict[str, Any], lang: str) -> Optional[str]:
    wh_token = cand["wh_token"]
    verb = cand.get("verb")
    subj = cand.get("subj")
    sentence = cand.get("sentence")
    sent_text = sentence.text if sentence is not None and hasattr(sentence, "text") else None
    if sent_text is None:
        return None

    sentence_words = sentence.words if sentence is not None else []

    def _token_text(tok):
        return tok.text

    wh_word = _token_text(wh_token)
    subj_text = _np_text(subj, sentence) if subj is not None else ""

    if lang == "ru":
        remaining = sent_text.replace(wh_word, "", 1).lstrip()
        remaining = re.sub(r"^[,.!?;:\s]+", "", remaining)
        remaining = remaining[0].lower() + remaining[1:] if remaining else remaining
        result = f"{wh_word} {remaining}"
        result = result[0].upper() + result[1:] if result else result
        result = re.sub(r'\s+([.!?,;:])', r'\1', result)
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    non_wh_words = [w for w in sentence_words if w.id != wh_token.id]
    tail_tokens = [
        w for w in non_wh_words
        if w.id != (subj.id if subj else -1)
        and w.id != (verb.id if verb else -1)
    ]
    tail_text = " ".join(_token_text(w) for w in tail_tokens if _token_text(w).lower() not in (verb.lemma.lower() if verb else ""))
    tail_text = re.sub(r"\s+", " ", tail_text).strip()
    verb_lemma = verb.lemma if verb is not None else ""
    subj_lower = subj_text[0].lower() + subj_text[1:] if subj_text else ""
    result = f"{wh_word} did {subj_lower} {verb_lemma} {tail_text}"
    result = re.sub(r"\s+", " ", result).strip()
    result = result[0].upper() + result[1:] if result else result
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)
    return result


class ControlledSyntacticTransformationsOperator(TierBOperator):
    operator_id = "controlled_syntactic_transformations"
    tier = Tier.B

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)
        if get_stanza_pipeline(lang) is None:
            return PreCheckResult(passed=False, reason="No parser available")

        fixed_expr = has_fixed_expression(text, lang)
        if fixed_expr is not None:
            return PreCheckResult(passed=False, reason=f"Fixed expression detected: {fixed_expr}")

        tokens = tokenize(text, lang)
        if has_scope_operators(tokens, lang):
            return PreCheckResult(passed=False, reason="Scope operators detected")

        candidates = self._find_applicable(text, lang)
        if not candidates:
            return PreCheckResult(passed=False, reason="No applicable transformation found")

        return PreCheckResult(passed=True, details={"candidates": candidates, "language": lang})

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        lang = language or detect_lang(text)
        rng = random.Random(seed)

        candidates = self._find_applicable(text, lang)
        if not candidates:
            return VariationResult(variant_text=text, metadata={"skipped": "no_applicable"}, original_text=text)

        rc_candidates = [c for c in candidates if c["type"] == "rc_reduction"]
        for cand in rc_candidates:
            variant = _apply_rc_reduction(cand, text)
            if variant and variant != text:
                return VariationResult(
                    variant_text=variant,
                    metadata={
                        "subtransformation": "rc_reduction",
                        "language": lang,
                        "verb_lemma": _verb_lemma(cand.get("relcl_token")),
                    },
                    original_text=text,
                )
            if cand.get("mark_token") is None:
                variant = _apply_rc_expansion(cand, text, lang)
                if variant and variant != text:
                    return VariationResult(
                        variant_text=variant,
                        metadata={
                            "subtransformation": "rc_expansion",
                            "language": lang,
                            "verb_lemma": _verb_lemma(cand.get("relcl_token")),
                        },
                        original_text=text,
                    )

        rc_expansion_only = [c for c in candidates if c["type"] == "rc_expansion"]
        for cand in rc_expansion_only:
            variant = _apply_rc_expansion(cand, text, lang)
            if variant and variant != text:
                return VariationResult(
                    variant_text=variant,
                    metadata={
                        "subtransformation": "rc_expansion",
                        "language": lang,
                        "verb_lemma": _verb_lemma(cand.get("relcl_token")),
                    },
                    original_text=text,
                )

        other = [c for c in candidates if c["type"] not in ("rc_reduction", "rc_expansion")]
        rng.shuffle(other)
        topicalization_cands = [c for c in other if c["type"] == "topicalization"]
        non_topo_cands = [c for c in other if c["type"] != "topicalization"]

        for cand in non_topo_cands:
            sub_type = cand["type"]
            variant: Optional[str] = None
            if sub_type == "clefting":
                variant = _apply_clefting(cand)
            elif sub_type == "clefting_ru":
                variant = _apply_clefting_ru(cand)
            elif sub_type == "dative_alternation":
                variant = _apply_dative(cand)
            elif sub_type == "wh_fronting":
                variant = _apply_wh_fronting(cand, lang)
            if variant and variant != text:
                return VariationResult(
                    variant_text=variant,
                    metadata={
                        "subtransformation": sub_type,
                        "language": lang,
                        "verb_lemma": _verb_lemma(cand.get("verb")),
                    },
                    original_text=text,
                )

        if topicalization_cands:
            logger.debug(
                "B4: topicalization has lower priority per spec B4.Pre-conditions "
                "(limited applicability due to information-structure constraints)."
            )
            for cand in topicalization_cands:
                variant = _apply_topicalization(cand)
                if variant and variant != text:
                    return VariationResult(
                        variant_text=variant,
                        metadata={
                            "subtransformation": "topicalization",
                            "language": lang,
                            "verb_lemma": _verb_lemma(cand.get("verb")),
                        },
                        original_text=text,
                    )

        return VariationResult(variant_text=text, metadata={"skipped": "no_applicable"}, original_text=text)

    @staticmethod
    def _verb_lemma_in_doc(doc, expected_lemma: Optional[str]) -> bool:
        if expected_lemma is None:
            return True
        expected_lower = expected_lemma.lower()
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ("VERB", "AUX") and (word.lemma or "").lower() == expected_lower:
                    return True
        return False




    def _find_applicable(self, text: str, lang: str) -> List[Dict[str, Any]]:
        if get_stanza_pipeline(lang) is None:
            return []
        doc = parse_ud(text, lang)
        if doc is None:
            return []
        candidates: List[Dict[str, Any]] = []
        if lang == "ru":
            candidates.extend(_find_clefting_candidates_ru(doc))
            if not any(c["type"] == "clefting_ru" for c in candidates):
                candidates.extend(_find_topicalization_candidates(doc))
            candidates.extend(_find_wh_questions(doc, lang))
        else:
            candidates.extend(_find_clefting_candidates(doc))
            if not any(c["type"] == "clefting" for c in candidates):
                candidates.extend(_find_topicalization_candidates(doc))
            candidates.extend(_find_dative_candidates(text, lang, doc))
            candidates.extend(_find_wh_questions(doc, lang))
        rc_cands = _find_rc_reduction_candidates(doc)
        for c in rc_cands:
            candidates.append(c)
            if c.get("mark_token") is None:
                candidates.append({**c, "type": "rc_expansion"})
        return candidates
