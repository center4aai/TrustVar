import re
from typing import Any, Dict, List, Optional, Tuple

import lemminflect

from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierBOperator,
    VariationResult,
)
from src.core.operators.utils.nlp_utils import (
    normalize_gender,
    normalize_number,
    detect_lang,
    get_active_verb_form_ru,
    get_be_form_ru,
    get_cc_prefix,
    get_clause_span,
    get_np_tokens,
    get_passive_participle_ru,
    get_present_passive_ru,
    get_stanza_pipeline,
    get_transitive_voice_candidates,
    has_scope_operators,
    inflect,
    InflectedToken,
    is_perfective_verb,
    is_reflexive_verb,
    np_case_inflect_ru,
    parse_feats,
)

_FIXED_EXPRESSIONS = {
    "en": [
        "it was meant to be",
        "it is said",
        "it is believed",
        "it is known",
        "it was decided",
        "it is well known",
        "it is widely believed",
        "it remains to be seen",
        "as was mentioned",
        "as has been noted",
    ],
    "ru": [
        "было решено",
        "было сказано",
        "было известно",
        "считается",
        "было объявлено",
        "как было сказано",
        "как отмечалось",
    ],
}


def _voice_has_fixed_expression(text_lower: str, lang: str) -> Optional[str]:
    for expr in _FIXED_EXPRESSIONS.get(lang, []):
        if expr in text_lower:
            return expr
    return None


def _en_past_participle(verb_lemma: str) -> str:
    """English past participle via lemminflect (handles ~280 irregulars + regular rules)."""
    forms = lemminflect.getInflection(verb_lemma, tag="VBN")
    if forms:
        return forms[0]
    return verb_lemma + "ed"


def _en_present_3sg(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBZ")
    if forms:
        return forms[0]
    return verb_lemma + "s"


def _en_past_tense(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBD")
    if forms:
        return forms[0]
    return verb_lemma + "ed"


def _en_present_non3sg(verb_lemma: str) -> str:
    forms = lemminflect.getInflection(verb_lemma, tag="VBP")
    if forms:
        return forms[0]
    return verb_lemma


def _en_be(tense: str, number: str) -> str:
    """EN auxiliary 'be' form for passive construction."""
    t = tense or "Pres"
    n = number or "Sing"
    if t == "Past":
        return "were" if n == "Plur" else "was"
    return "are" if n == "Plur" else "is"


def _np_text(tokens: List[Any]) -> str:
    return " ".join(t.text for t in tokens if getattr(t, "text", ""))


def _lc_first_alpha(s: str) -> str:
    """Lowercase the first alphabetic character of ``s``.

    Used when an NP that was sentence-initial in the source is repositioned
    mid-clause by the voice flip (e.g. EN agent "by the company" / "by the
    chef") — its determiner must drop the sentence-initial capital.
    Leaves the string unchanged if it has no alphabetic character.
    """
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def _text_between_tokens(
    prev: Any, curr: Any, original_text: str, sentence: Any = None
) -> str:
    """Return the substring of ``original_text`` between ``prev``'s end and the
    START of the next conjunct's NP span.

    ``prev`` may be a single token OR a list of tokens; when a list is given, the
    LAST token's ``end_char`` is used (so multi-word NPs like
    "туроператор Fun&Sun" do not duplicate the modifiers into the between-text).

    The between-text is clipped at the FIRST token of the next conjunct's full
    NP (computed via ``get_np_tokens``) rather than the head's start_char, so
    that modifiers that precede the next head (e.g. the nummod "три" in
    "статьи и три книги") are not duplicated into the between.
    """
    if isinstance(prev, (list, tuple)):
        if not prev:
            return " "
        last = max(prev, key=lambda t: getattr(t, "end_char", -1))
    else:
        last = prev
    end1 = getattr(last, "end_char", None)
    if end1 is None:
        return " "

    if sentence is not None:
        next_np = get_np_tokens(curr, sentence)
        starts = [
            getattr(t, "start_char")
            for t in next_np
            if getattr(t, "start_char", None) is not None
        ]
        if starts:
            start2 = min(starts)
            return original_text[end1:start2]

    start2 = getattr(curr, "start_char", None)
    if start2 is None:
        return " "
    return original_text[end1:start2]


def _extract_advmod_text(subj: Any, verb: Any, text: str, sentence: Any = None) -> str:
    """Return the advmod-style text (whitespace-stripped) between ``subj`` and ``verb``.

    This preserves adverbs like ``уже`` / ``Ранее`` / ``always`` that attach to
    the verb but appear between the subject and the verb in surface order. The
    caller embeds this string between the new subject and the new
    (passive/active) verb form. Returns an empty string when the offsets are
    unavailable.

    Aux tokens (``aux`` / ``aux:pass``) are NOT treated as advmods: in passive
    constructions like "The meal was prepared by ..." the ``was`` is the
    auxiliary that pairs with the past participle, and must be DROPPED (not
    carried over) when transforming to active voice. Same for copula ``был``
    in Russian passive.
    """
    if subj is None or verb is None:
        return ""
    subj_end = getattr(subj, "end_char", None)
    verb_start = getattr(verb, "start_char", None)
    if subj_end is None or verb_start is None:
        return ""
    between = text[subj_end:verb_start].strip()
    if not between or sentence is None:
        return between

    words = getattr(sentence, "words", None) or []
    for w in words:
        deprel_base = (w.deprel or "").split(":", 1)[0]
        if (
            deprel_base in ("aux", "cop")
            and getattr(w, "text", "").lower() in between.lower().split()
        ):
            return ""
    return between


def _build_coordinated_subject(
    cand: Dict[str, Any],
    sentence: Any,
    original_text: str,
    target_case: str,
    lang: str,
    target_number: Optional[str] = None,
    target_gender: Optional[str] = None,
) -> str:
    """Build the new (post-voice-flip) text for a coordinated subject/object span.

    For ``X и Y`` the result is ``inflect(X, target_case) + " и " +
    inflect(Y, target_case)``. For ``X, Y и Z`` the serial comma is preserved
    via the surface text between conjuncts. For ``ни X, ни Y`` the first
    ``ни`` (prefix-cc on the head) is prepended manually; the second ``ни``
    is captured by the between-text of the second conjunct.

    In English (no case inflection), each conjunct's NP text is preserved
    as-is. Capitalization of the resulting string is left to the splice layer
    (sentence-initial position triggers uppercase).

    ``target_number`` / ``target_gender`` (optional): if provided, they
    override the per-conjunct Stanza features for case inflection — used
    when the new subject must be Plur regardless of individual conjunct
    numbers (e.g. "две статьи и три книги" → "Две статьи и три книги").
    """
    conjuncts = cand.get("object_conjuncts") or []
    if len(conjuncts) < 2:
        head = cand.get("object") or cand.get("subject")
        if head is None:
            return ""
        np_tokens = get_np_tokens(head, sentence) if sentence else [head]
        if lang == "ru":
            inflected = np_case_inflect_ru(
                np_tokens, target_case, target_number, target_gender
            )
            return _np_text(inflected)
        return _np_text(np_tokens)

    parts: List[str] = []
    prev_np: Optional[List[Any]] = None
    for i, c in enumerate(conjuncts):
        np_tokens = get_np_tokens(c, sentence) if sentence else [c]
        if lang == "ru":
            has_nummod = any(
                (getattr(t, "deprel") or "").split(":", 1)[0] == "nummod"
                for t in np_tokens
            )
            head_feats = parse_feats(getattr(c, "feats", None))
            head_number = head_feats.get("Number", "Sing")
            per_conj_target_number = "Plur" if has_nummod else head_number
            per_conj_target_gender = head_feats.get("Gender", "Masc")
            inflected = np_case_inflect_ru(
                np_tokens, target_case, per_conj_target_number, per_conj_target_gender
            )
            piece = _np_text(inflected)
        else:
            piece = _np_text(np_tokens)

        prefix = ""
        if i == 0:
            prefix_cc = get_cc_prefix(c, sentence)
            if prefix_cc is not None:
                prefix = (prefix_cc.text or "") + " "

        if i == 0:
            parts.append(prefix + piece)
        else:
            between = _text_between_tokens(prev_np, c, original_text, sentence)
            parts.append(between + piece)

        prev_np = np_tokens

    return "".join(parts)


class ActivePassiveVoiceOperator(TierBOperator):
    operator_id = "active_passive_voice"
    tier = Tier.B

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)
        if not text.strip():
            return PreCheckResult(passed=False, reason="Empty text")

        if fixed := _voice_has_fixed_expression(text.strip().lower(), lang):
            return PreCheckResult(passed=False, reason=f"Fixed expression: {fixed}")

        if get_stanza_pipeline(lang) is None:
            return PreCheckResult(
                passed=False, reason=f"Stanza pipeline unavailable for lang={lang}"
            )

        candidates = get_transitive_voice_candidates(text, lang)
        if not candidates:
            return PreCheckResult(
                passed=False,
                reason="No transitive verb with nsubj+obj (or passive equivalent) found",
            )

        cand = candidates[0]
        verb = cand.get("verb")
        if verb is not None and has_scope_operators([verb], lang):
            return PreCheckResult(
                passed=False, reason="Scope-bearing operator detected on verb"
            )
        for arg_key in ("subject", "object", "agent"):
            arg = cand.get(arg_key)
            if arg is not None and has_scope_operators([arg], lang):
                return PreCheckResult(
                    passed=False, reason=f"Scope-bearing operator in {arg_key}"
                )

        if lang == "ru" and is_reflexive_verb(verb, lang="ru"):
            return PreCheckResult(
                passed=False,
                reason="Russian reflexive verb (-ся): passive vs. reflexive ambiguity",
            )

        return PreCheckResult(
            passed=True,
            details={"candidates": len(candidates), "language": lang},
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
                metadata={"skipped": "no_stanza"},
                original_text=text,
            )

        candidates = get_transitive_voice_candidates(text, lang)
        if not candidates:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_candidates"},
                original_text=text,
            )

        cand = candidates[0]
        verb = cand.get("verb")
        if verb is not None and has_scope_operators([verb], lang):
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "scope_on_verb"},
                original_text=text,
            )
        for arg_key in ("subject", "object", "agent"):
            arg = cand.get(arg_key)
            if arg is not None and has_scope_operators([arg], lang):
                return VariationResult(
                    variant_text=text,
                    metadata={"skipped": f"scope_in_{arg_key}"},
                    original_text=text,
                )

        if lang == "ru" and is_reflexive_verb(verb, lang="ru"):
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "reflexive_verb"},
                original_text=text,
            )

        if cand["voice"] == "active":
            new_clause, ok = self._active_to_passive(cand, text, lang)
            direction = "active_to_passive"
        else:
            new_clause, ok = self._passive_to_active(cand, text, lang)
            direction = "passive_to_active"

        if not ok or not new_clause:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "transformation_failed"},
                original_text=text,
            )

        new_text = self._splice_clause(text, cand, new_clause, lang)

        if not new_text or new_text == text:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_change_after_splice"},
                original_text=text,
            )

        return VariationResult(
            variant_text=new_text,
            metadata={
                "direction": direction,
                "language": lang,
                "verb_lemma": verb.lemma if verb else None,
                "new_clause": new_clause,
            },
            original_text=text,
        )



    @staticmethod
    def _active_to_passive(
        cand: Dict[str, Any], text: str, lang: str
    ) -> Tuple[Optional[str], bool]:
        verb = cand["verb"]
        subj = cand["subject"]
        obj = cand.get("object")
        if obj is None:
            return None, False

        sentence = cand.get("sentence")
        feats = parse_feats(getattr(verb, "feats", None))

        is_coord = cand.get("is_coordinated_object", False)
        new_number = cand.get("new_subject_number", "Sing")
        advmod_text = _extract_advmod_text(subj, verb, text, sentence)

        if is_coord:
            new_subj = _build_coordinated_subject(
                cand,
                sentence,
                text,
                target_case="nomn",
                lang=lang,
            )
        else:
            obj_np = get_np_tokens(obj, sentence) if sentence else [obj]
            if lang == "ru":
                obj_feats = parse_feats(getattr(obj, "feats", None))
                t_num = obj_feats.get("Number", "Sing")
                t_gen = obj_feats.get("Gender", "Masc")
                new_subj = _np_text(np_case_inflect_ru(obj_np, "nomn", t_num, t_gen))
            else:
                new_subj = _np_text(obj_np)

        subj_np = get_np_tokens(subj, sentence) if sentence else [subj]

        if lang == "en":
            tense = feats.get("Tense", "Pres")
            pp = _en_past_participle(verb.lemma)
            be = _en_be(tense, new_number)
            new_verb = f"{be} {pp}"
            new_agent = "by " + _lc_first_alpha(_np_text(subj_np))
        elif lang == "ru":
            if is_perfective_verb(verb, lang="ru") and feats.get("Tense") == "Past":
                if new_number == "Plur":
                    target_gender = "Masc"
                else:
                    obj_feats = parse_feats(getattr(obj, "feats", None))
                    target_gender = obj_feats.get("Gender", "Masc")
                pp = get_passive_participle_ru(verb.lemma, target_gender, new_number)
                if not pp:
                    return None, False
                be = get_be_form_ru("Past", target_gender, new_number)
                new_verb = f"{be} {pp}" if be else pp
            elif feats.get("Tense") == "Pres":
                pp = get_present_passive_ru(verb.lemma, new_number)
                if not pp:
                    return None, False
                new_verb = pp
            else:
                return None, False
            new_agent = _np_text(np_case_inflect_ru(subj_np, "ablt"))
        else:
            return None, False

        parts = [new_subj]
        if advmod_text:
            parts.append(advmod_text)
        parts.append(new_verb)
        parts.append(new_agent)
        return " ".join(parts), True

    @staticmethod
    def _passive_to_active(
        cand: Dict[str, Any], text: str, lang: str
    ) -> Tuple[Optional[str], bool]:
        verb = cand["verb"]
        subj = cand["subject"]
        agent = cand.get("agent")
        if agent is None:
            return None, False

        sentence = cand.get("sentence")
        feats = parse_feats(getattr(verb, "feats", None))

        is_coord_subj = cand.get("is_coordinated_subject", False)
        advmod_text = _extract_advmod_text(subj, verb, text, sentence)

        if lang == "en":
            aux_token = None
            if sentence is not None:
                for w in sentence.words:
                    deprel_base = (w.deprel or "").split(":", 1)[0]
                    if w.head == verb.id and deprel_base == "aux" and w.upos == "AUX":
                        aux_token = w
                        aux_feats = parse_feats(w.feats)
                        aux_tense = aux_feats.get("Tense", feats.get("Tense", "Pres"))
                        aux_number = aux_feats.get(
                            "Number", feats.get("Number", "Sing")
                        )
                        break
            if aux_token is not None:
                tense = aux_tense
                number = aux_number
            else:
                tense = feats.get("Tense", "Pres")
                number = feats.get("Number", "Sing")

            if tense == "Past":
                active_form = _en_past_tense(verb.lemma)
            elif number == "Sing":
                active_form = _en_present_3sg(verb.lemma)
            else:
                active_form = _en_present_non3sg(verb.lemma)

            agent_np = get_np_tokens(agent, sentence) if sentence else [agent]
            new_subj = _np_text(agent_np)

            if is_coord_subj:
                coord_cand = {
                    "object": subj,
                    "object_conjuncts": cand.get("subject_conjuncts", []),
                    "object_conjunction": cand.get("subject_conjunction"),
                }
                new_obj = _build_coordinated_subject(
                    coord_cand, sentence, text, target_case="nomn", lang=lang
                )
            else:
                subj_np = get_np_tokens(subj, sentence) if sentence else [subj]
                new_obj = _np_text(subj_np)
            new_verb = active_form
            new_agent = ""

        elif lang == "ru":
            agent_feats = parse_feats(getattr(agent, "feats", None))
            target_gender = agent_feats.get("Gender", "masc")
            target_number = agent_feats.get("Number", "sing")

            agent_np = get_np_tokens(agent, sentence) if sentence else [agent]
            new_subj = _np_text(np_case_inflect_ru(agent_np, "nomn"))

            if is_coord_subj:
                coord_cand = {
                    "object": subj,
                    "object_conjuncts": cand.get("subject_conjuncts", []),
                    "object_conjunction": cand.get("subject_conjunction"),
                }
                new_obj = _build_coordinated_subject(
                    coord_cand, sentence, text, target_case="accs", lang=lang
                )
            else:
                subj_np = get_np_tokens(subj, sentence) if sentence else [subj]
                new_obj = _np_text(np_case_inflect_ru(subj_np, "accs"))

            active_form = get_active_verb_form_ru(
                verb.lemma,
                target_gender,
                target_number,
                source_form=verb.text,
            )
            if not active_form:
                return None, False
            new_verb = active_form
            new_agent = ""
        else:
            return None, False

        parts = [new_subj]
        if advmod_text:
            parts.append(advmod_text)
        parts.append(new_verb)
        if new_agent:
            parts.append(new_agent)
        if new_obj:
            parts.append(new_obj)
        return " ".join(parts), True

    @staticmethod
    def _splice_clause(
        text: str,
        cand: Dict[str, Any],
        new_clause: str,
        lang: str,
    ) -> str:
        """Replace the source clause in `text` with `new_clause`.

        Single-clause sentences: full text is replaced (with leading cap + trailing
        period handling).
        Multi-clause sentences: only the targeted clause is replaced; surrounding
        context (conjunctions, other clauses, terminal punctuation) is preserved.

        If the clause span starts at position 0 (the new subject becomes
        sentence-initial), the first alphabetic character of the spliced result
        is uppercased. Otherwise the case of the first character of ``new_clause``
        is preserved verbatim — important for cases like "Ранее RWB уже
        приобрела ..." where the new subject ("туроператор ...") is mid-sentence.
        """
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
                    out = out[:i] + ch.upper() + out[i + 1 :]
                    break

        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        return out


__all__ = ["ActivePassiveVoiceOperator"]
