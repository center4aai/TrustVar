import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import get_settings
from src.core.operators.base import (
    PreCheckResult,
    Tier,
    TierBOperator,
    VariationResult,
)
from src.core.operators.utils.nlp_utils import (
    detect_lang,
    get_np_tokens,
    parse_ud,
)

settings = get_settings()


DEFAULT_SPLIT_MIN_TOKENS = settings.SENTENCE_SPLIT_MIN_TOKENS
DEFAULT_MERGE_MAX_TOKENS = settings.SENTENCE_MERGE_MAX_TOKENS
DEFAULT_MERGE_MAX_GROUP = settings.SENTENCE_MERGE_MAX_GROUP
DEFAULT_CAUSAL_STYLE = "spec"
VALID_CAUSAL_STYLES = ("spec", "natural")


WORD_OVERLAP_FLAG = settings.SENTENCE_WORD_OVERLAP_FLAG


# ── Lexicons ──────────────────────────────────────────────────────────

# Coordinating conjunctions (cc deprel) that introduce a clausal conjoin.
# Limited to the set we materialise; anything else is left to the LLM-fallback.
_CC_LEXICON: Dict[str, set] = {
    "en": {"and", "but", "or", "nor", "yet", "so", "for"},
    "ru": {"и", "но", "а", "или", "да", "зато"},
}

# Subordinators (mark deprel). Token-level — multi-word units like
# "потому что" / "так как" are reconstructed via the `fixed` multiword
# extension (see `_extend_multiword_mark`).
_MARK_LEXICON: Dict[str, set] = {
    "en": {
        "because",
        "since",
        "as",
        "though",
        "although",
        "while",
        "when",
        "if",
        "after",
        "before",
        "until",
        "unless",
    },
    "ru": {
        "потому",
        "так",
        "поскольку",
        "хотя",
        "когда",
        "если",
        "после",
        "перед",
        "до",
        "раз",
        "будь",
    },
}

# Discourse markers at the head of the second sentence — used to choose
# the appropriate connective on merge. Only initial-position triggers
# (lowercase, no leading capital) are listed; the check is case-insensitive
# at the call site.
_DISCOURSE_CONTRAST: Dict[str, set] = {
    "en": {"however", "nevertheless", "nonetheless", "still", "yet", "but"},
    "ru": {"однако", "тем не менее", "все же", "всё же", "зато", "но"},
}
_DISCOURSE_SEQUENTIAL: Dict[str, set] = {
    "en": {
        "then",
        "next",
        "afterwards",
        "subsequently",
        "finally",
        "and",
        "also",
        "additionally",
    },
    "ru": {"затем", "потом", "далее", "наконец", "и", "также", "кроме того"},
}
_DISCOURSE_CAUSAL: Dict[str, set] = {
    "en": {"therefore", "thus", "hence", "consequently", "so", "accordingly"},
    "ru": {"поэтому", "следовательно", "значит", "в итоге", "оттого", "потому"},
}

_CAUSAL_MATERIALISATION: Dict[str, Dict[str, str]] = {
    "en": {
        "spec": "This is because",
        "natural": "After all,",
    },
    "ru": {
        "spec": "Это потому что",
        "natural": "Ведь",
    },
}

# Connectives used to join two sentences on merge.
_JOIN_CONTRAST: Dict[str, str] = {"en": "but", "ru": "но"}
_JOIN_SEQUENTIAL: Dict[str, str] = {"en": "and", "ru": "и"}

_PRONOUNS: Dict[str, set] = {
    "en": {
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "themselves",
    },
    "ru": {
        "я",
        "ты",
        "он",
        "она",
        "оно",
        "мы",
        "вы",
        "они",
        "меня",
        "тебя",
        "его",
        "её",
        "ее",
        "нас",
        "вас",
        "их",
        "мой",
        "твой",
        "наш",
        "ваш",
        "свой",
        "этот",
        "эта",
        "это",
        "эти",
        "тот",
        "та",
        "те",
        "такой",
        "себя",
        "сам",
        "сама",
        "само",
        "сами",
    },
}


# ── Text helpers ──────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise_words(text: str) -> List[str]:
    return text.split()


def _word_overlap(a: str, b: str) -> float:
    """Jaccard overlap on word sets (case-folded, unicode-aware)."""
    wa = set(_WORD_RE.findall(a.lower()))
    wb = set(_WORD_RE.findall(b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(len(wa | wb), 1)


def _pronoun_count(text: str, lang: str) -> int:
    pronouns = _PRONOUNS.get(lang, _PRONOUNS["en"])
    return sum(1 for w in _WORD_RE.findall(text.lower()) if w in pronouns)


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip().rstrip(",;:")


def _strip_leading_punct(s: str) -> str:
    return s.lstrip().lstrip(",;:")


def _uc_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def _lc_first_alpha(s: str, lang: str = "en") -> str:
    """Lowercase the first alphabetic character of ``s``.

    The English pronoun ``"I"`` is a hard exception — it is always
    capitalised regardless of mid-sentence position. Russian pronouns are
    normally lowercase mid-sentence (so ``"Я"`` becomes ``"я"`` here);
    a sentence-initial Russian pronoun is handled by ``_uc_first_alpha``
    via the other branch of ``_materialise_clause_split``.
    """
    if not s:
        return s
    tokens = s.split(maxsplit=1)
    if not tokens:
        return s
    first = tokens[0]
    if first == "I":
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def _normalise_whitespace(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()


# ── UD-driven clause boundary detection ──────────────────────────────


def _extend_multiword_mark(mark_token: Any, sentence_obj: Any) -> Tuple[int, int, str]:
    """Extend a single ``mark`` token to cover a Russian multiword subordinator
    (e.g. «потому что», «так как», «после того как»).

    Stanza tags «потому» / «так» as ``deprel=mark`` and «что» / «как» as
    ``deprel=fixed`` attached to it. This helper returns the full
    ``(start_char, end_char, text)`` span of the multiword marker.

    Returns the mark token's own span unchanged when no ``fixed`` follower
    is present (single-word case).
    """
    if not hasattr(sentence_obj, "words"):
        return mark_token.start_char, mark_token.end_char, mark_token.text
    words = list(sentence_obj.words)
    span_start = mark_token.start_char
    span_end = mark_token.end_char
    span_text = mark_token.text
    for w in words:
        if (
            w.head == mark_token.id
            and w.deprel == "fixed"
            and w.start_char >= mark_token.end_char
        ):
            gap = (
                sentence_obj.text[span_end : w.start_char]
                if hasattr(sentence_obj, "text")
                else " "
            )
            span_end = w.end_char
            span_text = (
                sentence_obj.text[span_start:span_end]
                if hasattr(sentence_obj, "text")
                else f"{span_text}{gap}{w.text}"
            )
            break
    return span_start, span_end, span_text


def _is_nsubj(token: Any) -> bool:
    deprel = (token.deprel or "").split(":", 1)[0]
    return deprel == "nsubj"


def _is_clausal_root(verb: Any) -> bool:
    """A verb token is clausal if it's the root, a conj, or an advcl/acl head."""
    deprel = verb.deprel or ""
    if deprel in ("root", "conj", "advcl", "acl", "acl:relcl", "parataxis"):
        return True
    return False


def _find_clause_subject(verb: Any, sentence_obj: Any) -> Optional[Any]:
    """Return the surface subject of ``verb`` in ``sentence_obj``.

    For ordinary verbs the subject is the ``nsubj`` dependent. Russian
    copular constructions of the form «мне нужно X» / «ему было холодно»
    surface the dative argument as ``iobj`` (UD-2 convention) even though
    it is semantically the subject of the copular adjective. To keep the
    split well-formed (avoiding double-subject artefacts like
    «…я, потому что я мне нужно…»), we accept ``iobj`` as a fallback
    subject only for copular heads (ADJ with ``Variant=Short`` or verbs
    with a ``cop`` dependent).
    """
    if sentence_obj is None or verb is None:
        return None
    for w in sentence_obj.words:
        if w.head == verb.id and _is_nsubj(w):
            return w
    if verb.upos in ("ADJ", "AUX", "VERB"):
        is_copular = (verb.upos == "ADJ" and "Short" in (verb.feats or "")) or any(
            w.head == verb.id and (w.deprel or "").split(":", 1)[0] == "cop"
            for w in sentence_obj.words
        )
        if is_copular:
            for w in sentence_obj.words:
                if w.head == verb.id and (w.deprel or "").split(":", 1)[0] == "iobj":
                    return w
    return None


def _find_clause_controller(second_verb: Any, sentence_obj: Any) -> Optional[Any]:
    """Поднимаемся от ``second_verb`` по цепочке ``.head`` до ближайшего
    предка-глагола, у которого есть ровно один прямой ``nsubj``.

    Если по пути такого нет, либо контролёр неоднозначен (>1 nsubj) —
    возвращаем ``None``.  Это предотвращает копирование глобально-левого
    подлежащего (например, «Аня» вместо «Петя» для эллиптической
    сочинённой клаузы).

    Обрабатывает gapped coordination где UD присоединяет второй conj
    напрямую к root, минуя промежуточный conj («Аня приготовила ужин,
    и Петя устал и ушёл»): sibling-поиск conj того же головы, имеющих
    nsubj.
    """
    if sentence_obj is None or second_verb is None:
        return None

    def _unique_nsubj(verb: Any) -> Optional[Any]:
        nsubjs = [w for w in sentence_obj.words if w.head == verb.id and _is_nsubj(w)]
        if len(nsubjs) == 1:
            return nsubjs[0]
        return None

    def _find_sibling_nsubj(root_or_conj: Any, exclude_id: int) -> Optional[Any]:
        """Search siblings of ``root_or_conj`` that are ``conj``/``parataxis``
        with an ``nsubj`` and lie before ``exclude_id``."""
        siblings = [
            w
            for w in sentence_obj.words
            if w.head == root_or_conj.id
            and w.id != root_or_conj.id
            and w.upos in ("VERB", "AUX", "ADJ")
            and (w.deprel or "").split(":", 1)[0] in ("conj", "parataxis")
            and w.id < exclude_id
        ]
        for sib in siblings:
            n = _unique_nsubj(sib)
            if n is not None:
                return n
        return None

    visited: set = set()
    current = second_verb
    while current is not None:
        if current.id in visited:
            break
        visited.add(current.id)

        if current.upos in ("VERB", "AUX", "ADJ"):
            dep = (current.deprel or "").split(":", 1)[0]

            # Intermediate conj (not the starting verb) — its nsubj is
            # a valid controller for sibling-gapped clauses.
            if dep in ("conj", "parataxis") and current.id != second_verb.id:
                nsubj = _unique_nsubj(current)
                if nsubj is not None:
                    return nsubj

            # Root: prefer a sibling conj with its own nsubj (gapped
            # coordination case), fall back to root's own nsubj.
            if dep == "root":
                sibling_ns = _find_sibling_nsubj(current, second_verb.id)
                if sibling_ns is not None:
                    return sibling_ns
                nsubj = _unique_nsubj(current)
                if nsubj is not None:
                    return nsubj

        head_id = getattr(current, "head", current.id)
        if head_id == current.id:
            break
        head = _verb_at(sentence_obj, head_id)
        if head is not None and head.upos not in ("VERB", "AUX", "ADJ"):
            if (head.deprel or "").split(":", 1)[0] not in ("conj", "parataxis", "advcl"):
                break
        current = head
    return None


def _verb_at(sentence_obj: Any, token_id: int) -> Optional[Any]:
    if sentence_obj is None:
        return None
    for w in sentence_obj.words:
        if w.id == token_id:
            return w
    return None


def _find_clause_boundaries(sentence_obj: Any, lang: str) -> List[Dict[str, Any]]:
    """Return all clausal boundaries in ``sentence_obj`` as a list of dicts
    (sorted left-to-right by span start):

      {
        "kind": "cc" | "mark",
        "connector": str,                # text of the connector
        "span_start": int,               # inclusive start char in the text
        "span_end": int,                 # exclusive end char of the connector
        "second_clause_head_id": int,    # Stanza id of the conjoined/advcl verb
        "first_clause_end": int,         # char where the first clause ends
                                         # (== span_start for mark; span_start
                                         #  for cc as well — kept for clarity)
      }

    Only boundaries that:
      • correspond to a known coordinator (``cc``) or subordinator (``mark``)
        for the target language;
      • have a clausal head (VERB / AUX / copular ADJ) that is itself a
        ``conj`` of the root (cc) or an ``advcl/acl`` of the root (mark);
      • have a non-zero ``span_end - span_start`` (filters empty markers).

    are returned. This deliberately rejects `cc` in noun-list coordinations
    («apples, oranges, and bananas») where the head is a NOUN, not a VERB.
    """
    if sentence_obj is None or not hasattr(sentence_obj, "words"):
        return []
    cc_set = _CC_LEXICON.get(lang, set())
    mark_set = _MARK_LEXICON.get(lang, set())
    if not cc_set and not mark_set:
        return []
    out: List[Dict[str, Any]] = []
    for w in sentence_obj.words:
        text_lower = (w.text or "").lower()
        if w.deprel == "cc" and text_lower in cc_set:
            head = _verb_at(sentence_obj, w.head)
            if head is None or head.upos not in ("VERB", "AUX"):
                continue
            if not _is_clausal_root(head):
                continue
            out.append(
                {
                    "kind": "cc",
                    "connector": w.text,
                    "span_start": w.start_char,
                    "span_end": w.end_char,
                    "second_clause_head_id": w.head,
                    "first_clause_end": w.start_char,
                }
            )
        elif w.deprel == "mark" and text_lower in mark_set:
            head = _verb_at(sentence_obj, w.head)
            if head is None:
                continue
            if head.upos not in ("VERB", "AUX", "ADJ"):
                continue
            if not _is_clausal_root(head):
                continue
            span_start, span_end, span_text = _extend_multiword_mark(w, sentence_obj)
            if span_start <= 0:
                continue
            out.append(
                {
                    "kind": "mark",
                    "connector": span_text,
                    "span_start": span_start,
                    "span_end": span_end,
                    "second_clause_head_id": w.head,
                    "first_clause_end": span_start,
                }
            )
    out.sort(key=lambda b: b["span_start"])
    return out


# ── Discourse-relation detection for merge candidates ────────────────


def _detect_discourse_relation(second_sentence: str, lang: str) -> str:
    s = second_sentence.strip().lower()
    if not s:
        return "neutral"
    first_word = re.split(r"[\s,;:.!?]", s, maxsplit=1)[0]
    for marker in _DISCOURSE_CAUSAL.get(lang, set()):
        if (
            first_word == marker
            or s.startswith(marker + " ")
            or s.startswith(marker + ",")
        ):
            return "causal"
    for marker in _DISCOURSE_CONTRAST.get(lang, set()):
        if (
            first_word == marker
            or s.startswith(marker + " ")
            or s.startswith(marker + ",")
        ):
            return "contrastive"
    for marker in _DISCOURSE_SEQUENTIAL.get(lang, set()):
        if (
            first_word == marker
            or s.startswith(marker + " ")
            or s.startswith(marker + ",")
        ):
            return "sequential"
    return "neutral"


def _strip_initial_discourse_marker(sentence: str, lang: str) -> str:
    """Remove a leading discourse marker (and its trailing comma) from
    ``sentence``. Used during merge to avoid doubled contrastives
    (``"… but however, …"``).

    Markers tested, in order: causal → contrastive → sequential (the
    causal set is included so the relation still gets classified; after
    classification causal is rejected by the merge caller, so this
    stripping path is dead code for causal but kept for robustness).
    """
    s = sentence.lstrip()
    if not s:
        return s
    for marker_set in (
        _DISCOURSE_CAUSAL.get(lang, set()),
        _DISCOURSE_CONTRAST.get(lang, set()),
        _DISCOURSE_SEQUENTIAL.get(lang, set()),
    ):
        for marker in marker_set:
            if " " in marker:
                continue
            low = s.lower()
            if low.startswith(marker + " "):
                s = s[len(marker) :].lstrip()
                if s.startswith(","):
                    s = s[1:].lstrip()
                return s
            if low.startswith(marker + ","):
                s = s[len(marker) + 1 :].lstrip()
                return s
    multi_contrast = _DISCOURSE_CONTRAST.get(lang, set())
    for marker in multi_contrast:
        if " " not in marker:
            continue
        low = s.lower()
        if low.startswith(marker + " "):
            s = s[len(marker) :].lstrip()
            if s.startswith(","):
                s = s[1:].lstrip()
            return s
    return s


def _find_merge_groups(
    sentence_texts: List[str],
    sentence_objs: List[Any],
    max_tokens: int,
    max_group: int,
    lang: str,
) -> List[Tuple[int, int]]:

    if len(sentence_texts) < 2:
        return []
    counts = [len(_tokenise_words(s)) for s in sentence_texts]
    groups: List[Tuple[int, int]] = []
    i = 0
    while i < len(sentence_texts) - 1:
        if counts[i] > max_tokens:
            i += 1
            continue
        j = i + 1
        while (
            j < len(sentence_texts) and (j - i) < max_group and counts[j] <= max_tokens
        ):
            rel = _detect_discourse_relation(sentence_texts[j], lang)
            if rel == "causal":
                break
            j += 1
        if j > i + 1:
            groups.append((i, j - 1))
            i = j
        else:
            i += 1
    return groups


# ── Operator ─────────────────────────────────────────────────────────


class SentenceSplitMergeOperator(TierBOperator):


    operator_id = "sentence_split_merge"
    tier = Tier.B

    DEFAULT_CONFIG: Dict[str, Any] = {
        "split_min_tokens": DEFAULT_SPLIT_MIN_TOKENS,
        "merge_max_tokens": DEFAULT_MERGE_MAX_TOKENS,
        "merge_max_group": DEFAULT_MERGE_MAX_GROUP,
        "causal_style": DEFAULT_CAUSAL_STYLE,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        merged = dict(self.DEFAULT_CONFIG)
        if config:
            merged.update(config)
        self.split_min_tokens = int(merged["split_min_tokens"])
        self.merge_max_tokens = int(merged["merge_max_tokens"])
        self.merge_max_group = int(merged["merge_max_group"])
        causal_style = str(merged.get("causal_style", DEFAULT_CAUSAL_STYLE))
        if causal_style not in VALID_CAUSAL_STYLES:
            raise ValueError(
                f"Invalid causal_style={causal_style!r}; expected one of {VALID_CAUSAL_STYLES}"
            )
        self.causal_style = causal_style

    # ── Public API ────────────────────────────────────────────────

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> PreCheckResult:
        lang = language or detect_lang(text)
        clean = (text or "").strip()
        if len(clean) < 10:
            return PreCheckResult(passed=False, reason="Text too short (< 10 chars)")

        sentences = self._ud_sentences(clean, lang)
        if not sentences:
            return PreCheckResult(passed=False, reason="No parseable sentences")

        can_split = self._can_split(sentences, lang)
        merge_groups = _find_merge_groups(
            [s[0] for s in sentences],
            [s[1] for s in sentences],
            self.merge_max_tokens,
            self.merge_max_group,
            lang,
        )
        can_merge = bool(merge_groups)

        if not (can_split or can_merge):
            return PreCheckResult(
                passed=False,
                reason="Neither split (no clausal boundary) nor merge (no short adjacent pair) applicable",
            )

        return PreCheckResult(
            passed=True,
            details={
                "can_split": can_split,
                "can_merge": can_merge,
                "language": lang,
                "split_min_tokens": self.split_min_tokens,
                "merge_max_tokens": self.merge_max_tokens,
                "n_sentences": len(sentences),
            },
        )

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> VariationResult:
        lang = language or detect_lang(text)
        clean = (text or "").strip()
        if not clean:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "empty_text"},
                original_text=text,
            )

        causal_style = kwargs.get("causal_style", self.causal_style)
        if causal_style not in VALID_CAUSAL_STYLES:
            causal_style = self.causal_style

        rng = random.Random(seed)
        ops = ["split", "merge"]
        rng.shuffle(ops)

        for op in ops:
            if op == "split":
                result = self._apply_split(clean, lang, causal_style)
                if result is not None:
                    return result
            else:
                result = self._apply_merge(clean, lang)
                if result is not None:
                    return result

        return VariationResult(
            variant_text=text,
            metadata={"skipped": "no_operation_applicable"},
            original_text=text,
        )



    # ── Sentence segmentation (UD-based) ──────────────────────────

    def _ud_sentences(self, text: str, lang: str) -> List[Tuple[str, Any]]:
        """Tokenise into sentences and return ``(sentence_text, stanza_sentence)``
        tuples. Falls back to a regex split with ``None`` Stanza objects when
        the parser is unavailable.
        """
        doc = parse_ud(text, lang)
        if doc is None:
            parts = re.split(r"(?<=[.!?…])\s+", text)
            return [(p.strip(), None) for p in parts if p.strip()]
        out: List[Tuple[str, Any]] = []
        for sent in doc.sentences:
            if not sent.tokens:
                continue
            s_start = sent.tokens[0].start_char
            s_end = sent.tokens[-1].end_char
            out.append((text[s_start:s_end], sent))
        return out

    def _sentence_count(self, text: str, lang: str) -> int:
        return len(self._ud_sentences(text, lang))

    # ── Precondition helpers ──────────────────────────────────────

    def _can_split(self, sentences: List[Tuple[str, Any]], lang: str) -> bool:
        for sent_text, sent_obj in sentences:
            if len(_tokenise_words(sent_text)) > self.split_min_tokens:
                if sent_obj is not None:
                    verbs = [w for w in sent_obj.words if w.upos in ("VERB", "AUX")]
                    if len(verbs) >= 2:
                        return True
                if _find_clause_boundaries(sent_obj, lang):
                    return True
            else:
                if sent_obj is not None and _find_clause_boundaries(sent_obj, lang):
                    verbs = [w for w in sent_obj.words if w.upos in ("VERB", "AUX")]
                    if len(verbs) >= 2:
                        return True
        return False

    # ── Split ─────────────────────────────────────────────────────

    def _apply_split(
        self, text: str, lang: str, causal_style: str
    ) -> Optional[VariationResult]:
        doc = parse_ud(text, lang)
        if doc is None:
            return None

        any_split_done = False
        new_sentence_parts: List[str] = []
        total_splits = 0
        introduced_subjects: List[Dict[str, Any]] = []
        boundary_kinds: List[str] = []

        for sent in doc.sentences:
            if not sent.tokens:
                continue
            s_start = sent.tokens[0].start_char
            s_end = sent.tokens[-1].end_char
            sent_text = text[s_start:s_end]
            verbs = [w for w in sent.words if w.upos in ("VERB", "AUX")]
            boundaries = _find_clause_boundaries(sent, lang)
            if not boundaries or len(verbs) < 2:
                new_sentence_parts.append(sent_text)
                continue

            new_sent_text, n_done, intros = self._split_sentence(
                sent_text, sent, boundaries, lang, causal_style
            )
            new_sentence_parts.append(new_sent_text)
            total_splits += n_done
            introduced_subjects.extend(intros)
            boundary_kinds.extend([_find_kind_label(b) for b in boundaries[:n_done]])
            if n_done > 0:
                any_split_done = True

        if not any_split_done:
            return None

        new_text = " ".join(p for p in new_sentence_parts if p).strip()
        new_text = _normalise_whitespace(new_text)
        if not new_text or new_text == text:
            return None

        sent_count_orig = self._sentence_count(text, lang)
        sent_count_new = self._sentence_count(new_text, lang)
        if sent_count_new <= sent_count_orig:
            return None

        return VariationResult(
            variant_text=new_text,
            metadata={
                "operation": "split",
                "n_boundaries": total_splits,
                "boundary_kinds": boundary_kinds,
                "introduced_subjects": introduced_subjects,
                "language": lang,
                "causal_style": causal_style,
                "sentence_count_orig": sent_count_orig,
                "sentence_count_new": sent_count_new,
            },
            original_text=text,
        )

    def _split_sentence(
        self,
        sent_text: str,
        sent_obj: Any,
        boundaries: List[Dict[str, Any]],
        lang: str,
        causal_style: str,
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Split a single sentence at one or more clausal boundaries.

        Returns ``(new_sentence_text, n_splits_applied, introduced_subjects)``.
        Processes boundaries right-to-left so that earlier (left-side)
        positions in ``sent_text`` remain valid as we splice.
        """
        if not boundaries:
            return sent_text, 0, []
        current = sent_text
        splits_done = 0
        intros: List[Dict[str, Any]] = []

        for b in sorted(boundaries, key=lambda x: x["span_start"], reverse=True):
            span_start = b["span_start"]
            span_end = b["span_end"]
            kind = b["kind"]
            connector = b["connector"]
            second_verb = _verb_at(sent_obj, b["second_clause_head_id"])

            if span_start >= len(current) or span_end > len(current):
                continue
            if span_start < 0 or span_end <= span_start:
                continue

            controller = _find_clause_controller(second_verb, sent_obj)

            before_text = _strip_trailing_punct(current[:span_start])
            after_text = _strip_leading_punct(current[span_end:])

            new_after, subject_intro = self._materialise_clause_split(
                after_text=after_text,
                second_verb=second_verb,
                sent_obj=sent_obj,
                controller=controller,
                kind=kind,
                connector=connector,
                lang=lang,
                causal_style=causal_style,
            )
            new_materialisation = self._split_materialisation(
                kind=kind, connector=connector, lang=lang, causal_style=causal_style
            )
            current = (before_text + new_materialisation + new_after).strip()
            splits_done += 1
            if subject_intro is not None:
                intros.append(subject_intro)

        current = _normalise_whitespace(current)
        if not current:
            return sent_text, 0, []
        return current, splits_done, intros

    def _split_materialisation(
        self,
        kind: str,
        connector: str,
        lang: str,
        causal_style: str,
    ) -> str:
        """Return the string inserted in place of the connector for a split."""
        if kind == "cc":
            cl = connector.lower()
            if lang == "en" and cl in ("but",):
                return ". However, "
            if lang == "ru" and cl in ("но", "а"):
                return ". Однако, "
            return ". "
        if kind == "mark":
            style = (
                causal_style
                if causal_style in VALID_CAUSAL_STYLES
                else DEFAULT_CAUSAL_STYLE
            )
            tmpl = _CAUSAL_MATERIALISATION[lang][style]
            return ". " + tmpl + (" " if lang == "en" else " ")
        return ". "

    def _materialise_clause_split(
        self,
        after_text: str,
        second_verb: Optional[Any],
        sent_obj: Any,
        controller: Optional[Any],
        kind: str,
        connector: str,
        lang: str,
        causal_style: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Build the text of the new (post-split) clause.

        The materialisation prefix (e.g. \``. However, "\`) is supplied by
        ``_split_materialisation``; this function only constructs the
        remainder of the clause — including a copied subject NP when the
        second clause lacks its own nsubj, and the right capitalisation
        of the new sentence-initial token.

        Returns ``(new_clause_text, subject_intro_or_None)``.
        """
        own_subject = (
            _find_clause_subject(second_verb, sent_obj) if second_verb else None
        )

        copied: Optional[str] = None
        intro_meta: Optional[Dict[str, Any]] = None
        if own_subject is None and controller is not None:
            np_tokens = get_np_tokens(controller, sent_obj)
            copied = " ".join(
                getattr(t, "text", "") for t in np_tokens if getattr(t, "text", "")
            ).strip()
            if copied:
                intro_meta = {
                    "controller_lemma": controller.lemma,
                    "controller_text": copied,
                    "copied_to_clause_lemma": (
                        second_verb.lemma if second_verb else None
                    ),
                }

        if copied:
            clause_body = f"{copied} {after_text}".strip()
        else:
            clause_body = after_text

        if kind == "cc" and connector.lower() in (
            ("but",) if lang == "en" else ("но", "а")
        ):
            clause_body = _lc_first_alpha(clause_body, lang)
        elif kind == "mark":
            clause_body = _lc_first_alpha(clause_body, lang)
        else:
            clause_body = _uc_first_alpha(clause_body)

        return clause_body, intro_meta

    # ── Merge ─────────────────────────────────────────────────────

    def _apply_merge(self, text: str, lang: str) -> Optional[VariationResult]:
        sentences = self._ud_sentences(text, lang)
        if len(sentences) < 2:
            return None
        sent_texts = [s[0] for s in sentences]
        sent_objs = [s[1] for s in sentences]
        groups = _find_merge_groups(
            sent_texts,
            sent_objs,
            self.merge_max_tokens,
            self.merge_max_group,
            lang,
        )
        if not groups:
            return None

        relations: List[str] = []
        for start, end in groups:
            for j in range(start, end):
                relations.append(_detect_discourse_relation(sent_texts[j + 1], lang))

        new_parts: List[str] = [s for s in sent_texts]
        for start, end in sorted(groups, key=lambda g: g[0], reverse=True):
            merged = self._merge_group(sent_texts[start : end + 1], lang)
            if merged is None:
                continue
            new_parts[start : end + 1] = [merged]
        new_text = _normalise_whitespace(" ".join(new_parts))
        if not new_text or new_text == text:
            return None
        sent_count_orig = self._sentence_count(text, lang)
        sent_count_new = self._sentence_count(new_text, lang)
        if sent_count_new >= sent_count_orig:
            return None

        return VariationResult(
            variant_text=new_text,
            metadata={
                "operation": "merge",
                "n_groups": len(groups),
                "language": lang,
                "discourse_relations": relations,
                "sentence_count_orig": sent_count_orig,
                "sentence_count_new": sent_count_new,
            },
            original_text=text,
        )

    def _merge_group(self, sentences: List[str], lang: str) -> Optional[str]:
        """Join N short adjacent sentences into one. Relations between
        adjacent pairs are read from the second sentence's leading
        discourse marker; defaulting to sequential when neutral. Causal
        relations are pre-filtered by ``_find_merge_groups``.

        When the second sentence already carries an explicit discourse
        marker (e.g. ``"However, I was tired."``) the marker is stripped
        before joining — the merge connective we insert (``but`` /
        ``но`` / ``and`` / ``и``) already carries the same information, so
        keeping the marker would produce doubled contrastives
        (``"… but however, …"``).
        """
        if not sentences:
            return None
        if len(sentences) == 1:
            return sentences[0]
        out = sentences[0].rstrip().rstrip(".!?")
        for k in range(1, len(sentences)):
            nxt = sentences[k].lstrip()
            relation = _detect_discourse_relation(nxt, lang)
            if relation == "causal":
                return None
            nxt = _strip_initial_discourse_marker(nxt, lang)
            if not nxt:
                continue
            if relation == "contrastive":
                connective = _JOIN_CONTRAST[lang]
            else:
                connective = _JOIN_SEQUENTIAL[lang]
            tail = _lc_first_alpha(nxt, lang)
            if not tail:
                continue
            if connective in ("and", "и"):
                out = f"{out}, {connective} {tail}"
            else:
                out = f"{out} {connective} {tail}"
        out = out.rstrip()
        if not out.endswith((".", "!", "?")):
            out = out + "."
        return out


def _find_kind_label(b: Dict[str, Any]) -> str:
    return f"{b['kind']}:{b['connector']}"


__all__ = ["SentenceSplitMergeOperator"]
