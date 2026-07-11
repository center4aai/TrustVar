import logging
from typing import Any, Dict, List, Optional

import numpy as np
from nltk.corpus import wordnet as _wn

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_LABSE_MODEL: Optional[Any] = None
_RUWN_MODEL: Optional[Any] = None
_WN_POS_CACHE: Optional[Dict[str, Any]] = None

_RUWN_POS = {"NOUN": "N", "VERB": "V", "ADJ": "Adj"}


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


_WSD_THRESHOLD = settings.WSD_THRESHOLD


def _get_labse():
    global _LABSE_MODEL
    if _LABSE_MODEL is not None:
        return _LABSE_MODEL
    from sentence_transformers import (
        SentenceTransformer,  # lazy: avoid module-level torch import
    )

    _LABSE_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _LABSE_MODEL


def _get_ruwn():
    # Single source of truth: delegate to nlp_utils, which resolves the
    # project-managed RUWORDNET_DB_PATH and auto-provisions the DB on first use
    # (avoids the package default that points at the ephemeral venv static/ dir).
    global _RUWN_MODEL
    if _RUWN_MODEL is not None:
        return _RUWN_MODEL
    from src.core.operators.utils.nlp_utils import _get_ruwn as _shared_get_ruwn

    _RUWN_MODEL = _shared_get_ruwn()
    return _RUWN_MODEL


def _cosine(a, b) -> float:
    a_vec = a.flatten()
    b_vec = b.flatten()
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom < 1e-9:
        return 0.0
    return float(a_vec @ b_vec) / denom


def _embed(texts):
    model = _get_labse()
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, show_progress_bar=False)


def get_context_window(word, sentence, window: int = 5) -> str:
    words = sentence.words
    target_id = word.id
    idx = next((i for i, w in enumerate(words) if w.id == target_id), -1)
    if idx == -1:
        return sentence.text
    start = max(0, idx - window)
    end = min(len(words), idx + window + 1)
    return " ".join(w.text for w in words[start:end])


def _best_synset(synsets, ctx_emb):
    if ctx_emb is None:
        return synsets[0], 0.0

    glosses = [s.definition() for s in synsets]
    if not any(glosses):
        return synsets[0], 0.0

    gloss_embs = _embed(glosses)
    if gloss_embs is None:
        return synsets[0], 0.0

    sims = [_cosine(ctx_emb, ge) for ge in gloss_embs]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    confidence = sims[best_idx]

    if confidence < _WSD_THRESHOLD:
        return None, confidence

    return synsets[best_idx], confidence


def disambiguate_en(lemma: str, pos: str, context: str) -> Optional[Dict[str, Any]]:
    wn_pos = _get_wn_pos().get(pos)
    synsets = _wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        return None

    ctx_emb = _embed(context) if context else None
    best, confidence = _best_synset(synsets, ctx_emb)

    if best is None:
        return None

    return {
        "synset_id": best.name(),
        "synonyms": [lm.lower().replace("_", " ") for lm in best.lemma_names()],
        "confidence": confidence,
        "gloss": best.definition(),
        "lang": "en",
    }


def _collect_ru_synonyms(synset) -> List[str]:
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


def disambiguate_ru(lemma: str, pos: str, context: str) -> Optional[Dict[str, Any]]:
    ru_pos = _RUWN_POS.get(pos)
    if ru_pos is None:
        return None

    wn = _get_ruwn()
    synsets = wn.get_synsets(lemma.lower())
    synsets = [s for s in synsets if s.part_of_speech == ru_pos]
    if not synsets:
        return None

    ctx_emb = _embed(context) if context else None

    if ctx_emb is None:
        best = synsets[0]
        return {
            "synset_id": best.id,
            "synonyms": _collect_ru_synonyms(best),
            "confidence": 0.0,
            "gloss": best.definition or "",
            "lang": "ru",
        }

    defs = [(s, s.definition) for s in synsets if s.definition]
    if not defs:
        best = synsets[0]
        return {
            "synset_id": best.id,
            "synonyms": _collect_ru_synonyms(best),
            "confidence": 0.0,
            "gloss": best.definition or "",
            "lang": "ru",
        }

    glosses = [d for _, d in defs]
    gloss_embs = _embed(glosses)
    if gloss_embs is None:
        best = defs[0][0]
        return {
            "synset_id": best.id,
            "synonyms": _collect_ru_synonyms(best),
            "confidence": 0.0,
            "gloss": best.definition or "",
            "lang": "ru",
        }

    sims = [_cosine(ctx_emb, ge) for ge in gloss_embs]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    confidence = sims[best_idx]

    if confidence < _WSD_THRESHOLD:
        return None

    best = defs[best_idx][0]
    return {
        "synset_id": best.id,
        "synonyms": _collect_ru_synonyms(best),
        "confidence": confidence,
        "gloss": best.definition or "",
        "lang": "ru",
    }


def disambiguate(
    lemma: str, pos: str, context: str, lang: str
) -> Optional[Dict[str, Any]]:
    if lang == "ru":
        return disambiguate_ru(lemma, pos, context)
    return disambiguate_en(lemma, pos, context)
