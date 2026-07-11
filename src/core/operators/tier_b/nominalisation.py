import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import lemminflect
import pymorphy3
import yaml

from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierBOperator,
    VariationResult,
)
from src.core.taxonomy import resolve_task_semantics
from src.core.operators.utils.nlp_utils import (
    detect_lang,
    get_active_verb_form_ru,
    get_clause_span,
    get_np_tokens,
    get_stanza_pipeline,
    has_fixed_expression,
    has_scope_operators,
    np_case_inflect_ru,
    parse_feats,
    parse_ud,
)

_ru_morph = pymorphy3.MorphAnalyzer()


def _ru_lemma_variants(lemma: str) -> List[str]:
    variants = [lemma]
    lowered = lemma.lower().rstrip("ть").rstrip("ти")
    if not lowered:
        return variants

    for impf_suffix in ("ва", "ыва", "ива", "ева", "а", "я"):
        if lowered.endswith(impf_suffix) and len(lowered) > len(impf_suffix) + 1:
            stem = lowered[: -len(impf_suffix)]
            candidate = stem + "ть"
            if candidate != lemma.lower():
                variants.append(candidate)
            candidate2 = stem + "ти"
            if candidate2 != lemma.lower():
                variants.append(candidate2)
            if len(stem) > 1:
                variants.append(stem)

    try:
        parsed = _ru_morph.parse(lowered + "ть")
        if parsed:
            normal = parsed[0].normal_form
            if normal != lemma.lower():
                variants.append(normal)
    except Exception:
        pass

    return list(set(variants))


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_EN_LEXICON_PATH = os.path.join(_DATA_DIR, "en_nomlex_deverbatives.yaml")
_RU_LEXICON_PATH = os.path.join(_DATA_DIR, "ru_morphynet_deverbatives.yaml")


def _load_lexicon_en() -> Dict[str, str]:
    if not os.path.exists(_EN_LEXICON_PATH):
        return {}
    with open(_EN_LEXICON_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return {}
    return {k.lower().strip(): v for k, v in raw.items() if isinstance(v, str)}


def _load_lexicon_ru() -> Dict[str, List[str]]:
    if not os.path.exists(_RU_LEXICON_PATH):
        return {}
    with open(_RU_LEXICON_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for k, v in raw.items():
        k_lower = k.lower().strip()
        if isinstance(v, list):
            result[k_lower] = [x.lower().strip() for x in v if isinstance(x, str)]
        elif isinstance(v, str):
            result[k_lower] = [v.lower().strip()]
    return result


def _build_reverse_lexicon(forward: Dict, lang: str) -> Dict[str, str]:
    rev = {}
    if lang == "ru":
        for verb, nouns in forward.items():
            if isinstance(nouns, list):
                for n in nouns:
                    rev[n] = verb
            elif isinstance(nouns, str):
                rev[nouns] = verb
    else:
        for verb, noun in forward.items():
            rev[noun] = verb
    return rev


_LEXICON_EN: Dict[str, str] = _load_lexicon_en()
_LEXICON_RU: Dict[str, List[str]] = _load_lexicon_ru()
_REV_LEXICON_EN: Dict[str, str] = _build_reverse_lexicon(_LEXICON_EN, "en")
_REV_LEXICON_RU: Dict[str, str] = _build_reverse_lexicon(_LEXICON_RU, "ru")


def _get_lexicon(lang: str):
    return _LEXICON_RU if lang == "ru" else _LEXICON_EN


def _get_rev_lexicon(lang: str):
    return _REV_LEXICON_RU if lang == "ru" else _REV_LEXICON_EN


def _lemma_in_lexicon(lexicon, lemma: str, lang: str) -> bool:
    if lemma in lexicon:
        return True
    if lang == "ru":
        return any(alt in lexicon for alt in _ru_lemma_variants(lemma))
    return False


def _get_noun_from_lexicon(lexicon, lemma: str, lang: str, rng: Optional[random.Random] = None) -> Optional[str]:
    val = lexicon.get(lemma)
    if val is not None:
        return _pick_noun(val, rng)
    if lang == "ru":
        for alt in _ru_lemma_variants(lemma):
            val = lexicon.get(alt)
            if val is not None:
                return _pick_noun(val, rng)
    return None


def _pick_noun(val, rng: Optional[random.Random] = None) -> Optional[str]:
    if isinstance(val, str):
        return val
    if isinstance(val, list) and val:
        candidates = sorted(val)
        if rng:
            rng.shuffle(candidates)
        return candidates[0]
    return None


def _np_text(tokens: List[Any]) -> str:
    return " ".join(t.text for t in tokens if getattr(t, "text", ""))


def _lc_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1:]
    return s


def _en_present_3sg(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBZ")
    return forms[0] if forms else verb_lemma + "s"


def _en_past_tense(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBD")
    return forms[0] if forms else verb_lemma + "ed"


def _en_present_non3sg(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBP")
    return forms[0] if forms else verb_lemma


def _find_nominalisation_candidates(doc, lexicon, lang: str, rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        for token in sentence.words:
            if token.upos not in ("VERB", "AUX"):
                continue
            if not _lemma_in_lexicon(lexicon, token.lemma.lower(), lang):
                continue
            subj = next(
                (w for w in sentence.words if w.head == token.id and w.deprel == "nsubj"),
                None,
            )
            if subj is None:
                continue
            obj = next(
                (w for w in sentence.words if w.head == token.id and w.deprel == "obj"),
                None,
            )
            noun = _get_noun_from_lexicon(lexicon, token.lemma.lower(), lang, rng)
            if noun is None:
                continue
            candidates.append({
                "verb": token,
                "subject": subj,
                "object": obj,
                "sentence": sentence,
                "voice": "active",
                "deverbal_noun": noun,
                "lang": lang,
            })
    return candidates


def _ru_could_be_instrumental(surface: str) -> bool:
    for p in _ru_morph.parse(surface):
        if p.tag.POS in ("NOUN", "ADJF", "PRTF", "NPRO") and p.tag.case == "ablt":
            return True
    return False


def _find_denominalisation_candidates(doc, rev_lexicon, lang: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        for token in sentence.words:
            if token.upos != "NOUN":
                continue
            verb_lemma = rev_lexicon.get(token.lemma.lower())
            if verb_lemma is None:
                continue
            deprel = (token.deprel or "").split(":", 1)[0]
            if deprel not in ("nsubj", "nsubj:pass", "obj", "iobj", "root", "nmod"):
                continue

            subj = None
            obj = None
            if lang == "ru":
                # Step 1: look for case-marked dependents of the deverbal noun
                for w in sentence.words:
                    if w.head != token.id:
                        continue
                    w_dep = (w.deprel or "").split(":", 1)[0]
                    if w_dep not in ("nmod", "obl"):
                        continue
                    w_feats = parse_feats(getattr(w, "feats", None) or "")
                    w_case = w_feats.get("Case", "").lower()
                    if w_case in ("ablt", "ins"):
                        subj = w
                    elif w_case in ("gent", "gen"):
                        obj = w

                # Step 2: if no instrumental subject found among direct dependents,
                # look for a noun between the deverbal noun and the root that
                # either has Instrumental case in UD or can be instrumental per pymorphy3
                if subj is None:
                    root = None
                    for w in sentence.words:
                        if w.deprel == "root":
                            root = w
                            break
                    if root is not None:
                        token_end = getattr(token, "end_char", 0) or 0
                        root_start = getattr(root, "start_char", 0) or 0
                        for w in sentence.words:
                            if w.id == token.id or w.id == root.id:
                                continue
                            w_start = getattr(w, "start_char", None)
                            if w_start is None:
                                continue
                            if w_start < token_end or w_start >= root_start:
                                continue
                            w_upos = (w.upos or "")
                            if w_upos not in ("NOUN", "PROPN", "ADJF"):
                                continue
                            w_feats = parse_feats(getattr(w, "feats", None) or "")
                            w_case = w_feats.get("Case", "").lower()
                            if w_case in ("ablt", "ins"):
                                subj = w
                                break
                            # UD may misparse instrumental as dative
                            if w_case in ("datv", "dat"):
                                if _ru_could_be_instrumental(getattr(w, "text", "")):
                                    subj = w
                                    break
            else:
                for w in sentence.words:
                    if w.head == token.id and w.deprel == "nmod":
                        for child in sentence.words:
                            if child.head == w.id:
                                if w.lemma == "of":
                                    obj = child
                                elif w.lemma == "by":
                                    subj = child

            candidates.append({
                "deverbal_noun_token": token,
                "verb_lemma": verb_lemma,
                "subject_from_oblique": subj,
                "object_from_oblique": obj,
                "sentence": sentence,
                "lang": lang,
            })
    return candidates


# Task types where nominalisation strips tense (all languages, unconditional reject)
_TENSE_SENSITIVE_TASK_TYPES = {
    "tense_discrimination",
    "verb_conjugation",
    "grammatical_tense",
    "time_reference",
    "sequence_of_events",
}

# Task types where only RU aspect can be collapsed (conditional: only if collapsed)
_ASPECT_SENSITIVE_TASK_TYPES = {
    "aspect_discrimination",
    "aspectual_pair",
}

# Noun → list of verbs index for RU aspect collapse detection
_RU_NOUN_TO_VERBS: Dict[str, List[str]] = {}
for _verb, _nouns in _LEXICON_RU.items():
    for _noun in (_nouns if isinstance(_nouns, list) else [_nouns]):
        _RU_NOUN_TO_VERBS.setdefault(_noun, []).append(_verb)


def _ru_aspect_collapsed(verb_lemma: str, lexicon) -> bool:
    """Check if another verb in the lexicon maps to the same deverbal noun (aspect collapsed)."""
    val = lexicon.get(verb_lemma)
    if val is None:
        return False
    nouns = val if isinstance(val, list) else [val]
    for noun in nouns:
        verbs_for_noun = _RU_NOUN_TO_VERBS.get(noun, [])
        if any(v != verb_lemma for v in verbs_for_noun):
            return True
    return False


def _recover_tense_features(sentence, deverbal_token) -> Dict[str, str]:
    """Extract tense/gender/number from the clause hosting a deverbal noun."""
    copula = None
    root_verb = None
    for w in sentence.words:
        if w.head == deverbal_token.id and w.deprel == "cop":
            copula = w
        if w.deprel == "root" and w.upos in ("VERB", "AUX"):
            root_verb = w
    source = copula or root_verb
    if source is None:
        return {}
    return parse_feats(getattr(source, "feats", None) or "")


def _verb_domain_tokens(verb_token, sentence) -> List[Any]:
    tokens = [verb_token]
    for w in sentence.words:
        if w.head == verb_token.id:
            tokens.append(w)
    return tokens


def _splice_clause(
    text: str,
    cand: Dict[str, Any],
    new_clause: str,
    lang: str,
) -> str:
    sentence = cand.get("sentence")
    verb = cand.get("verb")
    if (
        sentence is None
        or verb is None
        or getattr(verb, "start_char", None) is None
        or getattr(verb, "end_char", None) is None
    ):
        return new_clause

    start, end = get_clause_span(cand, text)
    if start == 0 and end == len(text):
        out = new_clause
    else:
        out = text[:start] + new_clause + text[end:]

    if start == 0 and out:
        for i, ch in enumerate(out):
            if ch.isalpha():
                out = out[:i] + ch.upper() + out[i + 1:]
                break

    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


class NominalisationOperator(TierBOperator):
    operator_id = "nominalisation"
    tier = Tier.B

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)
        # S1.1: tense/aspect sensitivity is an operator-precondition property,
        # carried by the fine task_semantics (falls back to task_type).
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))

        if fixed := has_fixed_expression(text.strip().lower(), lang):
            return PreCheckResult(passed=False, reason=f"Fixed expression: {fixed}")

        if get_stanza_pipeline(lang) is None:
            return PreCheckResult(passed=False, reason="Stanza UD parser not available")

        doc = parse_ud(text, lang)
        if doc is None:
            return PreCheckResult(passed=False, reason="Stanza parse failed")

        lexicon = _get_lexicon(lang)
        if not lexicon:
            return PreCheckResult(passed=False, reason=f"No derivational lexicon for {lang}")

        # Stage 1: tense-sensitive tasks (all langauges, unconditional — applies to both directions)
        if task_semantics is not None and task_semantics.lower() in _TENSE_SENSITIVE_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics '{task_semantics}' is tense-sensitive; nominalisation strips finite tense",
            )

        found = _find_nominalisation_candidates(doc, lexicon, lang)
        if found:
            cand = found[0]
            verb = cand["verb"]
            domain = _verb_domain_tokens(verb, cand["sentence"])
            if has_scope_operators(domain, lang):
                return PreCheckResult(passed=False, reason="Scope-bearing operator in verb's domain")
            for arg_key in ("subject", "object"):
                arg = cand.get(arg_key)
                if arg is not None and has_scope_operators([arg], lang):
                    return PreCheckResult(
                        passed=False, reason=f"Scope-bearing operator in {arg_key}"
                    )

            # Stage 2: aspect-sensitive tasks (RU only, conditional — forward only)
            if (
                lang == "ru"
                and task_semantics is not None
                and task_semantics.lower() in _ASPECT_SENSITIVE_TASK_TYPES
                and _ru_aspect_collapsed(verb.lemma.lower(), lexicon)
            ):
                return PreCheckResult(
                    passed=False,
                    reason=f"Russian verb '{verb.lemma}' collapses aspect in deverbal noun; excluded for task semantics '{task_semantics}'",
                )

            return PreCheckResult(passed=True, details={"verb": verb.lemma, "language": lang})

        # Fallback: check reverse (denominalisation) candidates
        rev_lexicon = _get_rev_lexicon(lang)
        found_rev = _find_denominalisation_candidates(doc, rev_lexicon, lang) if rev_lexicon else []
        if found_rev:
            cand_rev = found_rev[0]
            if has_scope_operators(list(cand_rev["sentence"].words), lang):
                return PreCheckResult(passed=False, reason="Scope-bearing operator in denominalisation sentence")
            return PreCheckResult(passed=True, details={"direction": "denominalisation", "language": lang})

        return PreCheckResult(passed=False, reason="No verb with documented deverbal noun or denominalisation candidate")

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        lang = language or detect_lang(text)

        pre = await self.check_preconditions(
            text,
            task_type=kwargs.get("task_type"),
            task_semantics=kwargs.get("task_semantics"),
            language=lang,
        )
        if not pre.passed:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": pre.reason},
                original_text=text,
            )

        doc = parse_ud(text, lang)
        if doc is None:
            return VariationResult(variant_text=text, metadata={"error": "parse_failed"}, original_text=text)

        lexicon = _get_lexicon(lang)
        if not lexicon:
            return VariationResult(variant_text=text, metadata={"error": "empty_lexicon"}, original_text=text)

        rng = random.Random(seed) if seed is not None else None

        # --- try forward nominalisation (verb -> noun) ---
        found = _find_nominalisation_candidates(doc, lexicon, lang, rng)
        if found:
            cand = found[0]
            verb = cand["verb"]
            subj = cand["subject"]
            obj = cand["object"]
            noun = cand["deverbal_noun"]

            # re-check scope in apply
            domain = _verb_domain_tokens(verb, cand["sentence"])
            if has_scope_operators(domain, lang):
                return VariationResult(
                    variant_text=text, metadata={"skipped": "scope_on_verb"}, original_text=text
                )
            for arg_key in ("subject", "object"):
                arg = cand.get(arg_key)
                if arg is not None and has_scope_operators([arg], lang):
                    return VariationResult(
                        variant_text=text,
                        metadata={"skipped": f"scope_in_{arg_key}"},
                        original_text=text,
                    )

            if lang == "ru":
                new_clause, ok = self._ru_nominalise(cand, text)
            else:
                new_clause, ok = self._en_nominalise(cand, text)

            if ok and new_clause:
                variant = _splice_clause(text, cand, new_clause, lang)
                if variant != text:
                    return VariationResult(
                        variant_text=variant,
                        metadata={"verb": verb.lemma, "deverbal_noun": noun, "direction": "nominalisation", "language": lang},
                        original_text=text,
                    )

        # --- try reverse de-nominalisation (noun -> verb) ---
        rev_lexicon = _get_rev_lexicon(lang)
        if rev_lexicon:
            found_rev = _find_denominalisation_candidates(doc, rev_lexicon, lang)
            if found_rev:
                cand = found_rev[0]
                if lang == "ru":
                    new_clause, ok = self._ru_denominalise(cand, text)
                else:
                    new_clause, ok = self._en_denominalise(cand, text)

                if ok and new_clause:
                    variant = _splice_clause(text, cand, new_clause, lang)
                    if variant != text:
                        return VariationResult(
                            variant_text=variant,
                            metadata={"deverbal_noun": cand["deverbal_noun_token"].lemma, "verb": cand["verb_lemma"], "direction": "denominalisation", "language": lang},
                            original_text=text,
                        )

        return VariationResult(variant_text=text, metadata={"skipped": "no_applicable_transformation"}, original_text=text)



    # ── Forward nominalisation (verb -> noun) ──

    @staticmethod
    def _en_nominalise(cand: Dict[str, Any], text: str) -> Tuple[Optional[str], bool]:
        verb = cand["verb"]
        subj = cand["subject"]
        obj = cand["object"]
        noun = cand["deverbal_noun"]
        sentence = cand["sentence"]

        feats = parse_feats(getattr(verb, "feats", None))
        tense = feats.get("Tense", "Pres")

        subj_np = get_np_tokens(subj, sentence) if sentence else [subj]
        obj_np = get_np_tokens(obj, sentence) if sentence else [obj] if obj else None

        subj_text = _np_text(subj_np)
        obj_text = _np_text(obj_np) if obj_np else None

        # subject -> "by X" form, preserve PROPN capital, lowercase common nouns
        if subj.upos == "PROPN":
            subj_obl = "by " + subj_text
        else:
            subj_obl = "by " + _lc_first_alpha(subj_text)

        # object -> "of X"
        obj_phrase = ("of " + _lc_first_alpha(obj_text)) if obj_text else ""

        parts = [f"the {noun}"]
        if obj_phrase:
            parts.append(obj_phrase)
        parts.append(subj_obl)
        return " ".join(parts), True

    @staticmethod
    def _ru_nominalise(cand: Dict[str, Any], text: str) -> Tuple[Optional[str], bool]:
        verb = cand["verb"]
        subj = cand["subject"]
        obj = cand["object"]
        noun = cand["deverbal_noun"]
        sentence = cand["sentence"]

        # object -> genitive
        obj_tokens: List[Any] = []
        if obj:
            obj_np = get_np_tokens(obj, sentence) if sentence else [obj]
            obj_tokens = np_case_inflect_ru(obj_np, "gent")

        # subject -> instrumental
        subj_tokens: List[Any] = []
        if subj:
            subj_np = get_np_tokens(subj, sentence) if sentence else [subj]
            subj_tokens = np_case_inflect_ru(subj_np, "ablt")

        parts = [noun]
        if subj_tokens:
            parts.append(_np_text(subj_tokens))
        if obj_tokens:
            parts.append(_np_text(obj_tokens))
        return " ".join(parts), True

    # ── Reverse de-nominalisation (noun -> verb) ──

    @staticmethod
    def _en_denominalise(cand: Dict[str, Any], text: str) -> Tuple[Optional[str], bool]:
        verb_lemma = cand["verb_lemma"]
        sentence = cand["sentence"]
        deverbal_token = cand["deverbal_noun_token"]

        subj = cand.get("subject_from_oblique")
        obj = cand.get("object_from_oblique")

        # NEW-2: with neither a recovered oblique subject nor object, the result
        # collapses to a contentless agent stub ("someone <verb>") that destroys
        # the prompt's content and is identical across prompts. Reject.
        if subj is None and obj is None:
            return None, False

        subj_np = get_np_tokens(subj, sentence) if sentence and subj else []
        obj_np = get_np_tokens(obj, sentence) if sentence and obj else []

        subj_text = _np_text(subj_np) if subj_np else "someone"
        obj_text = _np_text(obj_np) if obj_np else None

        # Recover tense from surrounding context; use subject number for verb agreement
        tense_feats = _recover_tense_features(sentence, deverbal_token) if sentence else {}
        tense = tense_feats.get("Tense", "Past")
        if subj is not None:
            subj_feats = parse_feats(getattr(subj, "feats", None) or "")
            number = subj_feats.get("Number", tense_feats.get("Number", "Sing"))
        else:
            number = tense_feats.get("Number", "Sing")

        if tense == "Past":
            verb_form = _en_past_tense(verb_lemma)
        elif number == "Sing":
            verb_form = _en_present_3sg(verb_lemma)
        else:
            verb_form = _en_present_non3sg(verb_lemma)

        # NEW-2 (defensive): subject recovered but produced no surface text and
        # no object → still a contentless stub. Reject.
        if subj_text == "someone" and not obj_text:
            return None, False

        parts = [subj_text, verb_form]
        if obj_text:
            parts.append(_lc_first_alpha(obj_text))
        return " ".join(parts), True

    @staticmethod
    def _ru_denominalise(cand: Dict[str, Any], text: str) -> Tuple[Optional[str], bool]:
        verb_lemma = cand["verb_lemma"]
        sentence = cand["sentence"]
        deverbal_token = cand["deverbal_noun_token"]

        subj = cand.get("subject_from_oblique")
        obj = cand.get("object_from_oblique")

        # NEW-2: with neither a recovered oblique subject nor object, the result
        # collapses to a contentless agent stub (e.g., "кто-то <verb>") that destroys
        # the prompt's content and is identical across prompts. Reject.
        if subj is None and obj is None:
            return None, False

        # Resolve subject gender/number: if UD parsed as dative but word can be
        # instrumental, use pymorphy3 instrumental parse features for agreement
        subj_gender = "Masc"
        subj_number = "Sing"
        if subj is not None:
            subj_feats = parse_feats(getattr(subj, "feats", None) or "")
            subj_case = subj_feats.get("Case", "").lower()
            subj_gender = subj_feats.get("Gender", "Masc")
            subj_number = subj_feats.get("Number", "Sing")
            if subj_case in ("datv", "dat"):
                p_parses = _ru_morph.parse(getattr(subj, "text", ""))
                ins_parse = next(
                    (p for p in p_parses if p.tag.case == "ablt" and p.tag.POS == "NOUN"),
                    None,
                )
                if ins_parse is not None:
                    subj_gender = ins_parse.tag.gender or "Masc"
                    subj_number = ins_parse.tag.number or "Sing"

        # Inflect recovered subject from oblique back to nominative
        subj_np_raw = get_np_tokens(subj, sentence) if sentence and subj else []
        subj_text = ""
        if subj_np_raw:
            target_num = subj_number
            target_gndr = subj_gender
            subj_infl = np_case_inflect_ru(
                subj_np_raw, "nomn",
                target_number=target_num, target_gender=target_gndr,
            )
            subj_text = _np_text(subj_infl)
        if not subj_text:
            subj_text = "кто-то"

        # Inflect recovered object from genitive back to accusative
        obj_np_raw = get_np_tokens(obj, sentence) if sentence and obj else []
        obj_text = ""
        if obj_np_raw:
            obj_infl = np_case_inflect_ru(obj_np_raw, "accs")
            obj_text = _np_text(obj_infl)

        # Recover tense from surrounding context
        tense_feats = _recover_tense_features(sentence, deverbal_token) if sentence else {}
        tense = tense_feats.get("Tense", "Past")

        verb_form = get_active_verb_form_ru(verb_lemma, subj_gender, subj_number)
        if not verb_form:
            verb_form = verb_lemma + "л"

        # NEW-2 (defensive): subject collapsed to the placeholder and no object →
        # contentless stub (e.g., "кто-то <verb>"). Reject.
        if subj_text == "кто-то" and not obj_text:
            return None, False

        parts = [subj_text, verb_form]
        if obj_text:
            parts.append(obj_text)
        return " ".join(parts), True
