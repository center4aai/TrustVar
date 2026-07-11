import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import lemminflect
import pymorphy3
import ruwordnet as _ruwn
import stanza
import yaml
from nltk.corpus import wordnet as _wn

try:
    import torch
except Exception:  # pragma: no cover - optional dep
    torch = None

from src.config.settings import get_settings
from src.utils.logger import logger

settings = get_settings()

_ru_morph = pymorphy3.MorphAnalyzer()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_yaml(filename: str) -> Any:
    with open(_DATA_DIR / filename, encoding='utf-8') as f:
        return yaml.safe_load(f)


_PYMORPHY_CASE = {
    "nomn": "nomn",
    "gent": "gent",
    "datv": "datv",
    "accs": "accs",
    "ablt": "ablt",
    "loct": "loct",
    "NOMN": "nomn",
    "GENT": "gent",
    "DATV": "datv",
    "ACCS": "accs",
    "ABLT": "ablt",
    "LOCT": "loct",
}
_PYMORPHY_GENDER = {
    "fema": "femn",
    "FEMA": "femn",
    "masc": "masc",
    "MASC": "masc",
    "neut": "neut",
    "NEUT": "neut",
}
_PYMORPHY_NUMBER = {"plur": "plur", "PLUR": "plur"}


_stanza_pipelines: Dict[str, stanza.Pipeline] = {}
_STANZA_PROCESSORS = "tokenize,pos,lemma,depparse"


def get_stanza_pipeline(lang: str) -> Optional[stanza.Pipeline]:
    if lang not in ("en", "ru"):
        return None
    if lang not in _stanza_pipelines:
        try:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            _stanza_pipelines[lang] = stanza.Pipeline(
                lang,
                processors=_STANZA_PROCESSORS,
                device=device,
                verbose=False,
                logging_level="WARNING",
                download_method=None,
            )
        except Exception as e:
            logger.warning(f"Failed to load Stanza pipeline for {lang}: {e}")
            return None
    return _stanza_pipelines.get(lang)


def detect_lang(text: str) -> str:
    cyr = len(re.findall(r"[а-яА-ЯёЁ]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyr > lat else "en"


def parse_ud(text: str, lang: Optional[str] = None) -> Optional[stanza.Document]:
    lang = lang or detect_lang(text)
    nlp = get_stanza_pipeline(lang)
    if nlp is None:
        return None
    try:
        return nlp(text)
    except Exception as e:
        logger.warning(f"Stanza parse failed for lang={lang}: {e}")
        return None


def get_nouns(text: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    lang = lang or detect_lang(text)
    doc = parse_ud(text, lang)
    if doc is None:
        return []
    return [
        {
            "text": t.text,
            "lemma": t.lemma,
            "feats": str(t.feats),
            "id": t.id,
            "head": t.head,
        }
        for sent in doc.sentences
        for t in sent.words
        if t.upos in ("NOUN", "PROPN")
    ]


def get_verbs(text: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    lang = lang or detect_lang(text)
    doc = parse_ud(text, lang)
    if doc is None:
        return []
    return [
        {
            "text": t.text,
            "lemma": t.lemma,
            "feats": str(t.feats),
            "id": t.id,
            "head": t.head,
            "upos": t.upos,
        }
        for sent in doc.sentences
        for t in sent.words
        if t.upos in ("VERB", "AUX")
    ]


def get_transitive_voice_candidates(
    text: str, lang: Optional[str] = None
) -> List[Dict[str, Any]]:
    lang = lang or detect_lang(text)
    doc = parse_ud(text, lang)
    if doc is None:
        return []
    candidates: List[Dict[str, Any]] = []
    for sentence in doc.sentences:
        words = {w.id: w for w in sentence.words}
        for token in sentence.words:
            if (
                token.deprel == "nsubj"
                and token.upos in ("NOUN", "PROPN", "PRON")
                and token.head in words
            ):
                verb = words[token.head]
                if verb.upos in ("VERB", "AUX"):
                    obj = next(
                        (
                            w
                            for w in sentence.words
                            if w.head == verb.id and w.deprel == "obj"
                        ),
                        None,
                    )
                    if obj is not None:
                        coord = get_coordinated_object(obj, sentence)
                        cand: Dict[str, Any] = {
                            "voice": "active",
                            "verb": verb,
                            "subject": token,
                            "object": obj,
                            "object_conjuncts": coord["conjuncts"],
                            "object_conjunction": coord["cc"],
                            "object_conjuncts_cc_tokens": coord["cc_tokens"],
                            "is_coordinated_object": coord["is_coordinated"],
                            "is_ne_ni": coord["is_ne_ni"],
                            "agent": None,
                            "sentence": sentence,
                        }
                        cand["new_subject_number"] = infer_new_subject_number(
                            cand, lang
                        )
                        candidates.append(cand)
            elif token.deprel == "nsubj:pass" and token.head in words:
                verb = words[token.head]
                if verb.upos in ("VERB", "AUX"):
                    agent = next(
                        (
                            w
                            for w in sentence.words
                            if w.head == verb.id and w.deprel == "obl:agent"
                        ),
                        None,
                    )
                    if agent is None and lang == "ru":
                        agent = next(
                            (
                                w
                                for w in sentence.words
                                if w.head == verb.id
                                and w.deprel == "obl"
                                and "Ins" in (w.feats or "")
                            ),
                            None,
                        )
                    subj_coord = get_coordinated_object(token, sentence)
                    cand = {
                        "voice": "passive",
                        "verb": verb,
                        "subject": token,
                        "object": None,
                        "object_conjuncts": [],
                        "object_conjunction": None,
                        "is_coordinated_object": False,
                        "is_ne_ni": False,
                        "agent": agent,
                        "subject_conjuncts": subj_coord["conjuncts"],
                        "subject_conjunction": subj_coord["cc"],
                        "is_coordinated_subject": subj_coord["is_coordinated"],
                        "sentence": sentence,
                    }
                    candidates.append(cand)
    return candidates


def has_scope_operators(tokens: List[Any], lang: str = "en") -> bool:
    scope_en = {
        "may",
        "might",
        "must",
        "shall",
        "should",
        "can",
        "could",
        "will",
        "would",
        "not",
        "n't",
        "everyone",
        "everybody",
        "everything",
        "noone",
        "nobody",
        "nothing",
        "all",
        "each",
        "every",
        "some",
        "any",
        "none",
        "neither",
        "either",
        "want",
        "wish",
        "seek",
        "try",
        "need",
        "require",
    }
    scope_ru = {
        "может",
        "могут",
        "мог",
        "должен",
        "должна",
        "должны",
        "следует",
        "нужно",
        "надо",
        "не",
        "ни",
        "все",
        "каждый",
        "никто",
        "ничто",
        "хотеть",
        "хотел",
        "пытаться",
    }
    words: set = set()
    for t in tokens:
        for k in ("text", "lemma"):
            if isinstance(t, dict):
                v = t.get(k)
            else:
                v = getattr(t, k, None)
            if v:
                words.add(str(v).lower())
    if lang == "ru":
        return bool(words & scope_ru)
    return bool(words & scope_en)


def parse_feats(feats: Optional[str]) -> Dict[str, str]:
    if not feats:
        return {}
    out: Dict[str, str] = {}
    for pair in feats.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def get_np_tokens(head: Any, sentence: Any) -> List[Any]:
    head_id = getattr(head, "id", None)
    if head_id is None:
        return [head]

    np_deprel_prefixes = (
        "det",
        "amod",
        "nmod",
        "compound",
        "flat",
        "nummod",
        "appos",
        "obl",
    )

    if hasattr(sentence, "words"):
        words = {w.id: w for w in sentence.words}
    else:
        words = sentence

    included_ids: set = set()
    out: List[Any] = []
    if head_id in words:
        out.append(words[head_id])
        included_ids.add(head_id)
    for w in words.values():
        if w.id == head_id:
            continue
        deprel = w.deprel or ""
        deprel_base = deprel.split(":", 1)[0]
        if w.head == head_id and deprel_base in np_deprel_prefixes:
            out.append(w)
            included_ids.add(w.id)

    for w in words.values():
        if w.id in included_ids:
            continue
        deprel = w.deprel or ""
        deprel_base = deprel.split(":", 1)[0]
        if (
            deprel == "case"
            and w.head in included_ids
            and any(
                (words[i].deprel or "").split(":", 1)[0] in ("nmod", "obl")
                and words[i].head == head_id
                for i in included_ids
            )
        ):
            out.append(w)
            included_ids.add(w.id)

    case_obj_head_ids = {
        w.id for w in words.values() if w.deprel == "case" and w.head == head_id
    }
    if case_obj_head_ids:
        extra: List[Any] = []
        for case_id in sorted(case_obj_head_ids):
            if case_id not in included_ids:
                extra.append(words[case_id])
                included_ids.add(case_id)
        last_case_id = max(case_obj_head_ids)
        for wid in sorted(w.id for w in words.values() if w.id > last_case_id):
            if wid in included_ids or wid in case_obj_head_ids:
                continue
            wobj = words[wid]
            if wobj.upos in ("PUNCT", "CCONJ", "SCONJ", "VERB", "AUX"):
                break
            if wobj.id == head_id:
                break
            if wobj.head == head_id and wobj.deprel in ("conj",):
                break
            extra.append(wobj)
            included_ids.add(wobj.id)
        out.extend(extra)

    out.sort(key=lambda t: t.id)
    return out


def get_coordinated_object(head: Any, sentence: Any) -> Dict[str, Any]:
    if hasattr(sentence, "words"):
        words = {w.id: w for w in sentence.words}
    else:
        words = sentence

    head_id = head.id
    conjuncts: List[Any] = [head]

    for w in words.values():
        if (
            w.head == head_id
            and w.deprel == "conj"
            and w.upos in ("NOUN", "PROPN", "PRON")
        ):
            conjuncts.append(w)

    is_coordinated = len(conjuncts) > 1
    if not is_coordinated:
        return {
            "conjuncts": [head],
            "cc": None,
            "cc_tokens": [],
            "is_coordinated": False,
            "is_ne_ni": False,
        }

    cc_token = None
    cc_tokens: List[Any] = []
    for c in conjuncts:
        for w in words.values():
            if w.head == c.id and w.deprel == "cc":
                cc_tokens.append(w)
                if c is conjuncts[-1]:
                    cc_token = w
                break

    is_ne_ni = bool(cc_tokens) and all(
        (c.text or "").lower() in ("ни", "nor") for c in cc_tokens
    )

    return {
        "conjuncts": conjuncts,
        "cc": cc_token,
        "cc_tokens": cc_tokens,
        "is_coordinated": True,
        "is_ne_ni": is_ne_ni,
    }


def get_cc_prefix(token: Any, sentence: Any) -> Optional[Any]:
    if hasattr(sentence, "words"):
        words = list(sentence.words)
    else:
        words = sentence

    head_id = getattr(token, "id", None)
    head_start = getattr(token, "start_char", None)
    if head_id is None:
        return None

    for w in words:
        if w.head == head_id and w.deprel == "cc":
            if head_start is None or w.end_char <= head_start:
                return w
    return None


def infer_new_subject_number(cand: Dict[str, Any], lang: str) -> str:
    if not cand.get("is_coordinated_object"):
        obj = cand.get("object")
        if obj is None:
            return "Sing"
        feats = parse_feats(getattr(obj, "feats", None))
        n = feats.get("Number", "Sing")
        return "Plur" if n == "Plur" else "Sing"

    cc = cand.get("object_conjunction")
    cc_text = cc.text.lower() if cc is not None else ""

    if lang == "ru":
        if cc_text in ("или", "либо"):
            return "Sing"
        return "Plur"

    if lang == "en":
        if cc_text in ("or", "nor"):
            return "Sing"
        return "Plur"

    return "Sing"


def get_clause_span(
    candidate: Dict[str, Any],
    sentence_text: str,
) -> tuple:
    verb = candidate.get("verb")
    sentence = candidate.get("sentence")
    if verb is None:
        return 0, len(sentence_text)

    relevant_tokens: List[Any] = []

    for arg_key in ("subject", "object", "agent"):
        arg = candidate.get(arg_key)
        if arg is None:
            continue
        np_tokens = get_np_tokens(arg, sentence) if sentence else [arg]
        relevant_tokens.extend(np_tokens)

        if arg_key == "object" and candidate.get("is_coordinated_object"):
            for c in candidate.get("object_conjuncts", [])[1:]:
                c_tokens = get_np_tokens(c, sentence) if sentence else [c]
                relevant_tokens.extend(c_tokens)
            prefix_cc = get_cc_prefix(candidate["object"], sentence)
            if prefix_cc is not None:
                relevant_tokens.append(prefix_cc)
            cc = candidate.get("object_conjunction")
            if cc is not None:
                relevant_tokens.append(cc)

        if arg_key == "subject" and candidate.get("is_coordinated_subject"):
            for c in candidate.get("subject_conjuncts", [])[1:]:
                c_tokens = get_np_tokens(c, sentence) if sentence else [c]
                relevant_tokens.extend(c_tokens)
            prefix_cc = get_cc_prefix(candidate["subject"], sentence)
            if prefix_cc is not None:
                relevant_tokens.append(prefix_cc)
            cc = candidate.get("subject_conjunction")
            if cc is not None:
                relevant_tokens.append(cc)

    if verb is not None:
        relevant_tokens.append(verb)

    span_starts = [
        t.start_char
        for t in relevant_tokens
        if getattr(t, "start_char", None) is not None
    ]
    span_ends = [
        t.end_char for t in relevant_tokens if getattr(t, "end_char", None) is not None
    ]
    if not span_starts or not span_ends:
        return 0, len(sentence_text)

    raw_start = min(span_starts)
    raw_end = max(span_ends)

    pre_context = sentence_text[:raw_start]
    comma_pos = pre_context.rfind(",")
    if comma_pos != -1 and (raw_start - comma_pos) < 12:
        after = sentence_text[comma_pos + 1 : comma_pos + 6].lower()
        if after.lstrip().startswith(("и ", "но ", "а ", "or ", "and ", "but ")):
            raw_start = comma_pos + 1
            while raw_start < len(sentence_text) and sentence_text[raw_start] == " ":
                raw_start += 1

    return raw_start, raw_end


def is_perfective_verb(verb: Any, lang: str = "ru") -> bool:
    if lang != "ru":
        return True

    feats = parse_feats(getattr(verb, "feats", None))
    aspect = feats.get("Aspect")
    if aspect == "Perf":
        return True
    if aspect == "Imp":
        return False

    lemma = getattr(verb, "lemma", None)
    if lemma:
        try:
            parsed = _ru_morph.parse(lemma)
            for p in parsed:
                if "perf" in (p.tag.aspect or ""):
                    return True
                if "impf" in (p.tag.aspect or ""):
                    return False
        except Exception as e:
            logger.warning(f"pymorphy3 aspect lookup failed for {lemma!r}: {e}")

    return False


def is_reflexive_verb(verb: Any, lang: str = "ru") -> bool:
    if lang != "ru":
        return False

    feats = parse_feats(getattr(verb, "feats", None))
    if feats.get("Reflex") == "Yes":
        return True

    for attr in ("text", "lemma"):
        s = getattr(verb, attr, None)
        if s and (s.endswith("ся") or s.endswith("сь")):
            return True
    return False


def normalize_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    g = g.lower()
    return {"masc": "masc", "fem": "femn", "neut": "neut"}.get(g, g)


def normalize_number(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    n = n.lower()
    return {"sing": "sing", "plur": "plur"}.get(n, n)


class InflectedToken:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def np_case_inflect_ru(
    tokens: List[Any],
    target_case: str,
    target_number: Optional[str] = None,
    target_gender: Optional[str] = None,
) -> List[Any]:
    inflectable_upos = {"NOUN", "PROPN", "PRON", "ADJ", "DET", "NUM"}
    case_value_to_pymorphy = {
        "nomn": "nomn",
        "gent": "gent",
        "datv": "datv",
        "accs": "accs",
        "ablt": "ablt",
        "loct": "loct",
    }
    stanza_case_to_pymorphy = {
        "nom": "nomn",
        "gen": "gent",
        "dat": "datv",
        "acc": "accs",
        "ins": "ablt",
        "loc": "loct",
        "abl": "ablt",
    }
    target_pymorphy = case_value_to_pymorphy.get(target_case, target_case)
    target_stanza = {v: k for k, v in stanza_case_to_pymorphy.items()}.get(
        target_pymorphy, ""
    )

    out: List[Any] = []
    for t in tokens:
        upos = getattr(t, "upos", None)
        original = getattr(t, "text", "")
        if upos in inflectable_upos and original:
            t_feats = parse_feats(getattr(t, "feats", None))
            t_case_stanza = t_feats.get("Case", "")

            if t_case_stanza and (
                t_case_stanza.lower() == target_pymorphy
                or (target_stanza and t_case_stanza.lower() == target_stanza.lower())
            ):
                out.append(t)
                continue

            t_number = normalize_number(t_feats.get("Number", ""))
            t_gender = normalize_gender(t_feats.get("Gender", ""))
            t_number_eff = (
                normalize_number(target_number) if target_number else t_number
            )
            t_gender_eff = (
                normalize_gender(target_gender) if target_gender else t_gender
            )

            target_features: Dict[str, str] = {"case": target_case}
            if upos == "NUM":
                if t_gender:
                    target_features["gender"] = t_gender
            else:
                if t_number_eff == "plur":
                    target_features["number"] = "plur"
                elif t_number_eff == "sing" and t_gender_eff:
                    target_features["number"] = "sing"
                    target_features["gender"] = t_gender_eff
            new_form = inflect(original, target_features, lang="ru")
            if new_form and new_form != original:
                if upos == "PROPN" and original[:1].isupper():
                    new_form = new_form[:1].upper() + new_form[1:]
                out.append(InflectedToken(new_form))
                continue
            if upos == "PROPN":
                stanza_lemma = getattr(t, "lemma", None)
                if (
                    stanza_lemma
                    and stanza_lemma != original
                    and stanza_lemma[:1].isalpha()
                ):
                    out.append(InflectedToken(stanza_lemma))
                    continue
        out.append(t)
    return out


def get_passive_participle_ru(
    verb_lemma: str, gender: str, number: str
) -> Optional[str]:
    g = normalize_gender(gender)
    n = normalize_number(number)
    if not n:
        return None

    try:
        parsed = _ru_morph.parse(verb_lemma)
    except Exception as e:
        logger.warning(f"pymorphy3 parse failed for {verb_lemma!r}: {e}")
        return None

    base = {"PRTS", "past", "pssv", n}
    candidates_set: List[set] = []
    if g and n == "sing":
        candidates_set.append(base | {g})
    candidates_set.append(base)
    for tag_set in candidates_set:
        for p in parsed:
            try:
                inflected = p.inflect(tag_set)
            except Exception:
                inflected = None
            if inflected is not None:
                return inflected.word

    return None


def get_active_verb_form_ru(
    verb_lemma: str,
    gender: str,
    number: str,
    source_form: Optional[str] = None,
) -> Optional[str]:
    g = normalize_gender(gender)
    n = normalize_number(number)
    if not n:
        return None

    candidates_lemma: List[str] = []
    if source_form:
        try:
            for p in _ru_morph.parse(source_form):
                if "perf" in (p.tag.aspect or ""):
                    candidates_lemma.append(p.normal_form)
                    break
        except Exception as e:
            logger.warning(
                f"pymorphy3 source_form parse failed for {source_form!r}: {e}"
            )
    candidates_lemma.append(verb_lemma)

    base = {"past", n}
    tag_sets: List[set] = []
    if g and n == "sing":
        tag_sets.append(base | {g})
    tag_sets.append(base)

    for lemma in candidates_lemma:
        try:
            parsed = _ru_morph.parse(lemma)
        except Exception as e:
            logger.warning(f"pymorphy3 parse failed for {lemma!r}: {e}")
            continue
        for tag_set in tag_sets:
            for p in parsed:
                try:
                    inflected = p.inflect(tag_set)
                except Exception:
                    inflected = None
                if inflected is not None:
                    return inflected.word

    return None


def get_present_passive_ru(
    verb_lemma: str,
    number: str,
) -> Optional[str]:
    n = normalize_number(number)
    if n not in ("sing", "plur"):
        return None
    try:
        parsed_list = _ru_morph.parse(verb_lemma)
    except Exception as e:
        logger.warning(f"pymorphy3 parse failed for {verb_lemma!r}: {e}")
        return None

    base = {"pres", n, "3per"}
    for p in parsed_list:
        try:
            inflected = p.inflect(base)
        except Exception:
            inflected = None
        if inflected is not None:
            return inflected.word + "ся"
    return None


def get_be_form_ru(tense: str, gender: str, number: str) -> str:
    t = (tense or "").lower()
    if t != "past":
        return ""

    n = normalize_number(number) or ""
    g = normalize_gender(gender) or ""

    if n == "plur":
        return "были"
    return {
        "masc": "был",
        "femn": "была",
        "neut": "было",
    }.get(g, "было")


def tokenize(text: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    lang = lang or detect_lang(text)
    doc = parse_ud(text, lang)
    if doc is None:
        return [
            {"text": w, "lemma": w.lower(), "pos": "X", "feats": ""}
            for w in text.split()
        ]
    return [
        {
            "text": t.text,
            "lemma": t.lemma,
            "pos": t.upos,
            "feats": str(t.feats),
            "id": t.id,
            "head": t.head,
        }
        for sent in doc.sentences
        for t in sent.words
    ]


def inflect(word: str, target_features: Dict[str, str], lang: str = "ru") -> str:
    if lang == "ru":
        try:
            parsed = _ru_morph.parse(word)
            if not parsed:
                return word
            required: set = set()
            for key, val in target_features.items():
                val_lower = (val or "").lower()
                if key == "case":
                    required.add(_PYMORPHY_CASE.get(val, val_lower))
                elif key == "gender":
                    required.add(_PYMORPHY_GENDER.get(val, val_lower))
                elif key == "number":
                    required.add(_PYMORPHY_NUMBER.get(val, val_lower))
                else:
                    required.add(val)
            if required:
                for p in parsed:
                    try:
                        inflected = p.inflect(required)
                        if inflected is not None:
                            return inflected.word
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(
                f"pymorphy3 inflect failed for word={word!r} features={target_features}: {e}"
            )
    return word


def get_wordnet_synonyms(lemma: str, pos: Optional[str] = None) -> List[str]:
    wn_pos_map = {"NOUN": _wn.NOUN, "VERB": _wn.VERB, "ADJ": _wn.ADJ, "ADV": _wn.ADV}
    synsets = _wn.synsets(lemma, pos=wn_pos_map.get(pos or ""))
    if not synsets:
        return []
    return [lm.name().lower().replace("_", " ") for lm in synsets[0].lemma_names()]


# ── Shared synonym utilities (for monosemic_synonym_substitution + wsd_synonym_substitution) ──


def _load_fixed_expressions(lang: str) -> List[str]:
    path = _DATA_DIR / f"{lang}_fixed_expressions.yaml"
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


_EN_FIXED_EXPRESSIONS = _load_fixed_expressions("en")
_RU_FIXED_EXPRESSIONS = _load_fixed_expressions("ru")


_filter_words_data = _load_yaml("substitution_filter_words.yaml")
_EN_SUBSTITUTION_FILTER_WORDS: set = set(_filter_words_data["en"])
_RU_SUBSTITUTION_FILTER_WORDS: set = set(_filter_words_data["ru"])

_register_data = _load_yaml("register_maps.yaml")
_EN_REGISTER_MAP: Dict[str, str] = _register_data["en"]
_RU_REGISTER_MAP: Dict[str, str] = _register_data["ru"]
_REGISTER_LEVELS: Dict[str, int] = _register_data["levels"]


def register_compatible(src_reg: str, tgt_reg: str) -> bool:
    return (
        abs(_REGISTER_LEVELS.get(src_reg, 1) - _REGISTER_LEVELS.get(tgt_reg, 1))
        <= settings.MAX_REGISTER_GAP
    )


def has_fixed_expression(text: str, lang: str) -> Optional[str]:
    text_lower = text.lower()
    exprs = _RU_FIXED_EXPRESSIONS if lang == "ru" else _EN_FIXED_EXPRESSIONS
    for expr in exprs:
        if expr in text_lower:
            return expr
    return None


def pick_synonym(
    lemma: str, synonyms: List[str], register_map: Dict[str, str], rng: random.Random
) -> Optional[str]:
    candidates = [
        s
        for s in synonyms
        if s != lemma and " " not in s and "." not in s and "," not in s
    ]
    if not candidates:
        return None
    rng.shuffle(candidates)
    for cand in candidates:
        src_reg = register_map.get(lemma, "neutral")
        tgt_reg = register_map.get(cand, "neutral")
        if register_compatible(src_reg, tgt_reg):
            return cand
    return candidates[0]


def en_inflect(lemma: str, upos: str, feats_str: str) -> str:
    if upos not in ("NOUN", "VERB", "ADJ", "ADV"):
        return lemma
    feats = {}
    for kv in (feats_str or "").split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            feats[k.strip()] = v.strip()

    _DEFAULT_TAG = {"VERB": "VB", "NOUN": "NN", "ADJ": "JJ", "ADV": "RB"}

    if upos == "VERB":
        vform = feats.get("VerbForm", "")
        tense = feats.get("Tense", "")
        number = feats.get("Number", "")
        if vform == "Fin" and tense == "Past":
            tag = "VBD"
        elif vform == "Part":
            tag = "VBN"
        elif vform == "Ger":
            tag = "VBG"
        elif tense == "Pres" and number == "Sing":
            tag = "VBZ"
        elif tense == "Pres" and number == "Plur":
            tag = "VBP"
        else:
            tag = "VB"
    elif upos == "NOUN":
        tag = "NNS" if feats.get("Number") == "Plur" else "NN"
    elif upos == "ADJ":
        degree = feats.get("Degree", "Pos")
        if degree == "Cmp":
            tag = "JJR"
        elif degree == "Sup":
            tag = "JJS"
        else:
            tag = "JJ"
    else:
        tag = _DEFAULT_TAG[upos]

    forms = lemminflect.getInflection(lemma, tag=tag)
    return forms[0] if forms else lemma


def ru_inflect_synonym(lemma: str, feats_str: str) -> str:
    feats = {}
    for kv in (feats_str or "").split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            feats[k.strip()] = v.strip()
    target = {}
    case = feats.get("Case")
    if case:
        target["case"] = case
    gender = feats.get("Gender")
    if gender:
        target["gender"] = gender
    number = feats.get("Number")
    if number:
        target["number"] = number
    if not target:
        return lemma
    return inflect(lemma, target, "ru")


def check_ru_lexicon(word: str) -> bool:
    wn = _get_ruwn()
    return len(wn.get_synsets(word.lower())) > 0


# ── EN WordNet POS mapping for synset queries ──

_RUWN_MODEL: Optional[Any] = None


def _get_ruwn():
    global _RUWN_MODEL
    if _RUWN_MODEL is not None:
        return _RUWN_MODEL
    # Provisioning lives in the dependency-light ruwordnet_db module (resolves the
    # project-managed RUWORDNET_DB_PATH and auto-provisions on first use).
    from src.core.operators.utils.ruwordnet_db import (
        provision_ruwn_db,
        resolve_ruwn_db_path,
    )

    db_path = resolve_ruwn_db_path()
    provision_ruwn_db(db_path)
    _RUWN_MODEL = _ruwn.RuWordNet(str(db_path))
    return _RUWN_MODEL


_WN_POS_CACHE: Optional[Dict[str, Any]] = None

_RUWN_POS: Dict[str, str] = _load_yaml("ruwn_pos.yaml")


def _get_wn_pos() -> Dict[str, Any]:
    global _WN_POS_CACHE
    if _WN_POS_CACHE is None:
        _WN_POS_CACHE = {
            "NOUN": _wn.NOUN,
            "VERB": _wn.VERB,
            "ADJ": _wn.ADJ,
            "ADV": _wn.ADV,
        }
    return _WN_POS_CACHE


def get_synset_count_en(lemma: str, upos: str) -> int:
    wn_pos = _get_wn_pos().get(upos)
    if wn_pos is None:
        return 0
    return len(_wn.synsets(lemma, pos=wn_pos))


def get_synset_count_ru(lemma: str, upos: str) -> int:
    ru_pos = _RUWN_POS.get(upos)
    if ru_pos is None:
        return 0
    wn = _get_ruwn()
    synsets = wn.get_synsets(lemma.lower())
    return sum(1 for s in synsets if s.part_of_speech == ru_pos)


def get_unique_synset_en(lemma: str, upos: str) -> Optional[Any]:
    wn_pos = _get_wn_pos().get(upos)
    if wn_pos is None:
        return None
    synsets = _wn.synsets(lemma, pos=wn_pos)
    if len(synsets) != 1:
        return None
    return synsets[0]


def get_unique_synset_ru(lemma: str, upos: Optional[str] = None) -> Optional[Any]:
    """Unique RuWordNet synset for ``lemma``.

    With ``upos`` → the single synset for that part of speech (monosemic check),
    else ``None``. With ``upos=None`` → a POS-agnostic *membership* probe: returns
    a synset if the lemma is known to RuWordNet at all (used by the validator's
    lexicon check, which only cares whether a word exists), else ``None``.
    """
    wn = _get_ruwn()
    synsets = wn.get_synsets(lemma.lower())
    if upos is None:
        return synsets[0] if synsets else None
    ru_pos = _RUWN_POS.get(upos)
    if ru_pos is None:
        return None
    pos_synsets = [s for s in synsets if s.part_of_speech == ru_pos]
    if len(pos_synsets) != 1:
        return None
    return pos_synsets[0]


def get_synonyms_from_synset_en(synset) -> List[str]:
    return [lm.lower().replace("_", " ") for lm in synset.lemma_names()]


def get_synonyms_from_synset_ru(synset) -> List[str]:
    seen: set = set()
    result: List[str] = []
    for se in synset.senses:
        name = se.name.lower().replace("_", " ")
        if name not in seen:
            seen.add(name)
            result.append(name)
        lemma = (se.lemma or "").lower().replace("_", " ")
        if lemma and lemma not in seen:
            seen.add(lemma)
            result.append(lemma)
    return result


__all__ = [
    "detect_lang",
    "normalize_gender",
    "normalize_number",
    "get_nouns",
    "get_verbs",
    "get_transitive_voice_candidates",
    "get_stanza_pipeline",
    "parse_ud",
    "has_scope_operators",
    "parse_feats",
    "get_np_tokens",
    "get_coordinated_object",
    "get_cc_prefix",
    "infer_new_subject_number",
    "get_clause_span",
    "is_perfective_verb",
    "is_reflexive_verb",
    "get_passive_participle_ru",
    "get_active_verb_form_ru",
    "get_present_passive_ru",
    "get_be_form_ru",
    "tokenize",
    "inflect",
    "InflectedToken",
    "np_case_inflect_ru",
    "get_wordnet_synonyms",
    # Shared synonym utilities
    "register_compatible",
    "has_fixed_expression",
    "pick_synonym",
    "en_inflect",
    "ru_inflect_synonym",
    "check_ru_lexicon",
    "get_synset_count_en",
    "get_synset_count_ru",
    "get_unique_synset_en",
    "get_unique_synset_ru",
    "get_synonyms_from_synset_en",
    "get_synonyms_from_synset_ru",
    "_EN_REGISTER_MAP",
    "_RU_REGISTER_MAP",
    "_EN_SUBSTITUTION_FILTER_WORDS",
    "_RU_SUBSTITUTION_FILTER_WORDS",
]
