import asyncio
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import bert_score
except ImportError:
    bert_score = None

try:
    from scipy.stats import chi2_contingency
except ImportError:
    chi2_contingency = None

from src.config.settings import get_settings
from src.core.operators.base import Tier
from src.core.operators.registry import OperatorRegistry
from src.core.services.model_cache import model_cache
from src.utils.logger import logger

settings = get_settings()


class ValidationStatus(Enum):
    ACCEPT = "accept"
    REJECT_LEXICAL = "reject_lexical"
    REJECT_SEMANTIC = "reject_semantic"
    REJECT_LOGIC = "reject_logic"
    REJECT_LINEAGE = "reject_lineage"
    FLAG_DISAGREEMENT = "flag_disagreement"
    FLAG_MARGINAL = "flag_marginal"
    ERROR = "error"


class TaskType(str, Enum):
    MCQ = "mcq"
    OPEN_QA = "open_qa"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    UNKNOWN = "unknown"


from pathlib import Path

import yaml

_DATA_DIR = Path(__file__).resolve().parent.parent / "operators" / "data"


def _load_yaml_data(filename: str) -> Any:
    with open(_DATA_DIR / filename, encoding='utf-8') as f:
        return yaml.safe_load(f)


# Load sentence starters from YAML
_sentence_data = _load_yaml_data("sentence_starters.yaml")
_SENTENCE_STARTERS = frozenset(
    set(_sentence_data.get("en", [])) | set(_sentence_data.get("ru", []))
)

# Model family registry for lineage enforcement (loaded from YAML)
MODEL_FAMILIES = _load_yaml_data("model_families.yaml")


def _resolve_family(model_name_or_tag: str) -> str:
    key = model_name_or_tag.lower().split("/")[-1].split("-")[0]
    return MODEL_FAMILIES.get(key, key)


# ── Module-level helpers for controlled_descriptive_modifier_insertion ──

_ADJ_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "operators", "data")


def _load_adjective_registry() -> Dict[str, Dict[str, List[str]]]:
    import yaml

    registry: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("en", "ru"):
        path = os.path.join(_ADJ_DATA_DIR, f"{lang}_neutral_adjectives.yaml")
        with open(path, encoding='utf-8') as f:
            registry[lang] = yaml.safe_load(f) or {}
    return registry


_ADJ_REGISTRY: Dict[str, Dict[str, List[str]]] = _load_adjective_registry()


_B7_LAYER1_REJECT_REASONS = frozenset({"too_many_adjectives", "multiple_amod_children"})

_OPERATOR_FLAG_METAS = (
    "idiom_flattening_flag",    # C8 idiom-flattening
    "verb_lemma_flag",          # active_passive_voice (B) — verb lemma changed
    "adj_registry_flag",        # controlled_descriptive_modifier_insertion (B7)
    "adj_ud_flag",              # controlled_descriptive_modifier_insertion (B7)
    "nominalisation_flag",      # nominalisation (B3)
    "backtranslation_flag",     # back_translation_single_pivot (B)
    "l3_openqa_fallback_flag",  # Open-QA free-text Layer 3 backend unavailable (M2)
    "register_shift_flag",      # register_formal_informal (C)
)


def _validator_all_registry_adjectives(lang: str) -> Set[str]:
    reg = _ADJ_REGISTRY.get(lang, _ADJ_REGISTRY["en"])
    result: Set[str] = set()
    for cat in reg.values():
        result.update(w.lower() for w in cat)
    return result


def _validator_check_ud_amod_slot(
    variant: str,
    lang: str,
    inserted_adj: str,
    target_noun: str,
) -> Tuple[bool, Dict[str, Any]]:
    from src.core.operators.utils.nlp_utils import (
        parse_feats,
        parse_ud,
    )

    doc = parse_ud(variant, lang)
    if doc is None:
        return False, {"reason": "ud_parse_failed", "check": "ud_amod_slot"}

    adj_token = None
    head_token = None
    head_sentence = None
    for sentence in doc.sentences:
        words = list(sentence.words)
        words_by_id = {w.id: w for w in words}
        for token in words:
            if token.text.lower() != inserted_adj.lower():
                continue
            if token.deprel != "amod":
                continue
            head_id = token.head
            if head_id not in words_by_id:
                continue
            head_cand = words_by_id[head_id]
            if head_cand.text.lower() != target_noun.lower():
                continue
            adj_token = token
            head_token = head_cand
            head_sentence = sentence
            break
        if adj_token is not None:
            break

    if adj_token is None:
        return False, {
            "reason": "amod_slot_not_found",
            "adjective": inserted_adj,
            "target_noun": target_noun,
            "check": "ud_amod_slot",
        }

    if lang == "ru":
        noun_feats = parse_feats(str(getattr(head_token, "feats", "") or ""))
        adj_feats = parse_feats(str(getattr(adj_token, "feats", "") or ""))
        for feat in ("Case", "Gender", "Number"):
            n_val = noun_feats.get(feat)
            a_val = adj_feats.get(feat)
            if n_val and a_val and n_val != a_val:
                return False, {
                    "reason": f"ru_morphological_disagreement_{feat.lower()}",
                    "expected": n_val,
                    "got": a_val,
                    "check": "ud_amod_slot",
                }

    amod_children_count = sum(
        1 for w in head_sentence.words if w.head == head_token.id and w.deprel == "amod"
    )
    if amod_children_count > 1:
        return False, {
            "reason": "multiple_amod_children",
            "count": amod_children_count,
            "check": "ud_amod_slot",
        }

    return True, {
        "adjective": inserted_adj,
        "noun": target_noun,
        "deprel": "amod",
        "check": "ud_amod_slot",
    }


def _validator_check_lexical(
    original: str,
    variant: str,
    lang: str,
) -> Tuple[bool, Dict[str, Any]]:
    all_valid = _validator_all_registry_adjectives(lang)
    words_orig = set(re.findall(r"[a-zа-яё]+", original.lower()))
    words_var = set(re.findall(r"[a-zа-яё]+", variant.lower()))
    new_words = words_var - words_orig
    inserted_adjs = [w for w in new_words if w in all_valid]

    if not inserted_adjs:
        return False, {
            "reason": "no_registry_adjective_found",
            "new_words": list(new_words)[:5],
            "check": "lexical_fallback",
        }

    orig_adj_count = sum(1 for w in words_orig if w in all_valid)
    var_adj_count = sum(1 for w in words_var if w in all_valid)
    if var_adj_count - orig_adj_count > 1:
        return False, {
            "reason": "too_many_adjectives",
            "delta": var_adj_count - orig_adj_count,
            "check": "lexical_fallback",
        }

    return True, {
        "inserted": inserted_adjs[:3],
        "adj_delta": var_adj_count - orig_adj_count,
        "check": "lexical_fallback",
    }


# ── Module-level helpers for controlled_syntactic_transformations ──

_WH_WORDS_DATA = _load_yaml_data("wh_words.yaml")
_WH_WORDS_EN: Set[str] = set(_WH_WORDS_DATA.get("en", []))
_WH_WORDS_RU: Set[str] = set(_WH_WORDS_DATA.get("ru", []))
_RELATIVE_PRONOUN_DATA = _load_yaml_data("relative_pronoun_lemmas.yaml")
_RELATIVE_PRONOUN_LEMMAS: Set[str] = set(_RELATIVE_PRONOUN_DATA.get("en", [])) | set(
    _RELATIVE_PRONOUN_DATA.get("ru", [])
)


def _validator_is_relative_pronoun_lemma(lemma: Optional[str]) -> bool:
    if not lemma:
        return False
    return lemma.lower() in _RELATIVE_PRONOUN_LEMMAS


def _validator_has_relative_pronoun_child(relcl: Any, sentence: Any) -> bool:
    if relcl is None or sentence is None:
        return False
    for w in sentence.words:
        if w.head != relcl.id:
            continue
        if w.deprel in (
            "mark",
            "obj",
            "nsubj",
            "iobj",
        ) and _validator_is_relative_pronoun_lemma(getattr(w, "lemma", None)):
            return True
    return False


def _validator_verb_lemma_in_doc(doc, expected_lemma: Optional[str]) -> bool:
    if expected_lemma is None:
        return True
    expected_lower = expected_lemma.lower()
    for sentence in doc.sentences:
        for word in sentence.words:
            if (
                word.upos in ("VERB", "AUX")
                and (word.lemma or "").lower() == expected_lower
            ):
                return True
    return False


def _validator_check_ud_subtransformation(
    variant: str,
    lang: str,
    subtransformation: Optional[str],
    verb_lemma: Optional[str],
) -> Tuple[bool, Dict[str, Any]]:
    from src.core.operators.utils.nlp_utils import parse_ud

    doc = parse_ud(variant, lang)
    if doc is None:
        return False, {"reason": "ud_parse_failed", "check": "ud_subtransformation"}

    if verb_lemma and not _validator_verb_lemma_in_doc(doc, verb_lemma):
        return False, {
            "reason": "verb_lemma_lost",
            "expected_lemma": verb_lemma,
            "check": "ud_subtransformation",
        }

    if subtransformation in ("clefting", "clefting_ru"):
        marker = "Именно" if lang == "ru" else "It was"
        if marker.lower() not in variant.lower():
            return False, {
                "reason": "clefting_marker_missing",
                "expected_marker": marker,
                "check": "ud_subtransformation",
            }
        has_relcl = any(
            w.deprel == "acl:relcl"
            for sentence in doc.sentences
            for w in sentence.words
        )
        if not has_relcl:
            return False, {
                "reason": "clefting_relcl_missing",
                "check": "ud_subtransformation",
            }
        return True, {
            "subtransformation": subtransformation,
            "check": "ud_subtransformation",
        }

    if subtransformation == "dative_alternation":
        for sentence in doc.sentences:
            for token in sentence.words:
                if (
                    token.upos in ("VERB", "AUX")
                    and (token.lemma or "").lower() == (verb_lemma or "").lower()
                ):
                    children_deprels = {
                        w.deprel for w in sentence.words if w.head == token.id
                    }
                    has_obj = "obj" in children_deprels
                    has_iobj = "iobj" in children_deprels
                    has_obl = "obl" in children_deprels
                    if has_obj and (has_iobj or has_obl):
                        return True, {
                            "subtransformation": subtransformation,
                            "check": "ud_subtransformation",
                        }
        return False, {
            "reason": "dative_object_structure_lost",
            "check": "ud_subtransformation",
        }

    if subtransformation == "rc_reduction":
        for sentence in doc.sentences:
            for token in sentence.words:
                if token.deprel == "acl:relcl":
                    has_marker = _validator_has_relative_pronoun_child(token, sentence)
                    if has_marker:
                        continue
                    return True, {
                        "subtransformation": subtransformation,
                        "check": "ud_subtransformation",
                    }
        return False, {
            "reason": "rc_clause_missing_after_reduction",
            "check": "ud_subtransformation",
        }

    if subtransformation == "rc_expansion":
        for sentence in doc.sentences:
            for token in sentence.words:
                if token.deprel == "acl:relcl":
                    has_marker = _validator_has_relative_pronoun_child(token, sentence)
                    if has_marker:
                        return True, {
                            "subtransformation": subtransformation,
                            "check": "ud_subtransformation",
                        }
        return False, {
            "reason": "rc_marker_missing_after_expansion",
            "check": "ud_subtransformation",
        }

    if subtransformation == "wh_fronting":
        wh_words = _WH_WORDS_RU if lang == "ru" else _WH_WORDS_EN
        first_token_text = ""
        for sentence in doc.sentences:
            if sentence.words:
                first_token_text = sentence.words[0].text
                break
        if first_token_text.lower() not in wh_words:
            return False, {
                "reason": "wh_not_at_sentence_start",
                "first_token": first_token_text,
                "check": "ud_subtransformation",
            }
        return True, {
            "subtransformation": subtransformation,
            "check": "ud_subtransformation",
        }

    if subtransformation == "topicalization":
        if "," not in variant:
            return False, {
                "reason": "topicalization_comma_missing",
                "check": "ud_subtransformation",
            }
        return True, {
            "subtransformation": subtransformation,
            "check": "ud_subtransformation",
        }

    return False, {
        "reason": "unknown_subtransformation",
        "subtransformation": subtransformation,
        "check": "ud_subtransformation",
    }


def _validator_validate_jaccard(
    original: str,
    variant: str,
    lang: str,
) -> Tuple[bool, Dict[str, Any]]:
    if lang == "ru":
        orig_content = set(re.findall(r"[а-яё]+", original.lower()))
        var_content = set(re.findall(r"[а-яё]+", variant.lower()))
    else:
        orig_content = set(re.findall(r"[a-z]+", original.lower()))
        var_content = set(re.findall(r"[a-z]+", variant.lower()))
    if not orig_content and not var_content:
        return False, {"reason": "no_content_tokens"}
    shared = orig_content & var_content
    ratio = len(shared) / max(len(orig_content | var_content), 1)
    if ratio < 0.4:
        return False, {
            "reason": "content_too_different",
            "content_overlap": round(ratio, 3),
            "check": "jaccard_fallback",
        }
    return True, {
        "content_overlap": round(ratio, 3),
        "check": "jaccard_fallback",
    }


class VariationValidator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.disagreement_threshold = self.config.get(
            "disagreement_threshold", settings.DISAGREEMENT_THRESHOLD
        )
        self.placeholder_pattern = re.compile(r"\{[a-zA-Z0-9_]+\}")

        # Lineage config: families, not specific model names
        self._lineage_configured = "generator_model" in self.config
        self.generator_family = _resolve_family(
            self.config.get("generator_model", "openai")
        )
        self.verifier_families_raw = self.config.get(
            "verifier_models",
            [
                settings.EN_NLI_MODEL_PRIMARY,
                settings.EN_NLI_MODEL_SECONDARY,
            ],
        )
        self.verifier_families = [
            _resolve_family(m) for m in self.verifier_families_raw
        ]

        # NLI model names per language
        self.en_nli_models = [
            self.config.get("en_nli_model_primary", settings.EN_NLI_MODEL_PRIMARY),
            self.config.get("en_nli_model_secondary", settings.EN_NLI_MODEL_SECONDARY),
        ]
        self.ru_nli_models = [
            self.config.get("ru_nli_model_primary", settings.RU_NLI_MODEL_PRIMARY),
            self.config.get("ru_nli_model_secondary", settings.RU_NLI_MODEL_SECONDARY),
        ]

        
        self.audit_queue: List[Dict[str, Any]] = []

        # ── Function-word chi-square ───────────────────────────────────

        self._EN_FUNCTION_WORDS = frozenset(_load_yaml_data("en_function_words.yaml"))
        self._RU_FUNCTION_WORDS = frozenset(_load_yaml_data("ru_function_words.yaml"))

    # ── NLI pipelines ──────────────────────────────────────────────

    def _get_nli_pipelines(self, language: str) -> List:
        return model_cache.get_nli_pipelines(language)

    # ── Embedding model (LaBSE) ────────────────────────────────────

    def _get_embedding_model(self):
        return model_cache.get_embedding_model()

    # ── Sentiment classifier (for C5 Layer 3 affect-preservation) ──

    def _get_sentiment_polarity(self, text: str, lang: str) -> Optional[int]:
        """Returns polarity: -1 (negative), 0 (neutral), 1 (positive), or None on failure.

        Uses a transformer-based sentiment classifier when available;
        falls back to the keyword heuristic.
        """
        classifier = model_cache.get_sentiment_classifier(lang)
        if classifier is not None:
            try:
                result = classifier(text, top_k=None)
                if isinstance(result, list):
                    scores = {r["label"].lower(): r["score"] for r in result}
                    if lang == "ru":
                        pos = scores.get("positive", 0.0)
                        neg = scores.get("negative", 0.0)
                    else:
                        pos = scores.get("positive", 0.0)
                        neg = scores.get("negative", 0.0)
                else:
                    label = (
                        result[0]["label"].lower() if isinstance(result, list) else ""
                    )
                    pos = 1.0 if "positive" in label else 0.0
                    neg = 1.0 if "negative" in label else 0.0
                if pos > 0.5 and pos > neg:
                    return 1
                if neg > 0.5 and neg > pos:
                    return -1
                return 0
            except Exception as e:
                logger.warning(f"Sentiment classifier inference failed: {e}")

        # Fallback: keyword heuristic
        _sentiment_data = _load_yaml_data("sentiment_words.yaml")
        _POS_WORDS = frozenset(
            w.lower() for w in _sentiment_data.get("positive", {}).get(lang, [])
        )
        _NEG_WORDS = frozenset(
            w.lower() for w in _sentiment_data.get("negative", {}).get(lang, [])
        )
        words = set(text.lower().split())
        pos_count = len(words & _POS_WORDS)
        neg_count = len(words & _NEG_WORDS)
        if pos_count == 0 and neg_count == 0:
            return 0
        return 1 if pos_count > neg_count else -1

    # ── Language detection ─────────────────────────────────────────

    @staticmethod
    def _detect_dominant_lang(text: str) -> str:
        cyr = len(re.findall(r"[а-яА-ЯёЁ]", text))
        lat = len(re.findall(r"[a-zA-Z]", text))
        return "ru" if cyr > lat else "en"

    # ── Jaccard ────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        wa = set(re.findall(r"\w+", a.lower()))
        wb = set(re.findall(r"\w+", b.lower()))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # ── Embedding cosine ───────────────────────────────────────────

    def _cosine_similarity(self, a: str, b: str) -> Optional[float]:
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            emb = model.encode([a, b], show_progress_bar=False)
            vec_a, vec_b = emb[0], emb[1]
            denom = (vec_a @ vec_a) ** 0.5 * (vec_b @ vec_b) ** 0.5
            return float(vec_a @ vec_b / denom) if denom > 0 else 0.0
        except Exception as e:
            logger.warning(f"Embedding cosine failed: {e}")
            return None

    def _function_word_chi2(self, a: str, b: str, lang: str) -> Optional[float]:
        if chi2_contingency is None:
            return None
        fw = self._RU_FUNCTION_WORDS if lang == "ru" else self._EN_FUNCTION_WORDS
        words_a = re.findall(r"[а-яёa-z]+", a.lower())
        words_b = re.findall(r"[а-яёa-z]+", b.lower())
        fw_a = sum(1 for w in words_a if w in fw)
        fw_b = sum(1 for w in words_b if w in fw)
        other_a = len(words_a) - fw_a
        other_b = len(words_b) - fw_b
        table = [[fw_a, other_a], [fw_b, other_b]]
        if min(table[0][1], table[1][1]) < 1:
            return 1.0
        try:
            _, p, _, _ = chi2_contingency(table)
            return float(p)
        except Exception:
            return None

    # ── BERTScore F1 ───────────────────────────────────────────────

    @staticmethod
    def _bertscore_f1(a: str, b: str, lang: str) -> Optional[float]:
        if bert_score is None:
            return None
        try:
            # bert_score.score expects list of candidates, list of references
            P, R, F1 = bert_score.score(
                [a], [b], lang="ru" if lang == "ru" else "en", verbose=False
            )
            return float(F1[0])
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return None

    # ── NLI entailment extraction (full distribution) ──────────────

    @staticmethod
    def _extract_entailment_score(pipeline_output, model_name: str = "") -> float:
        if (
            isinstance(pipeline_output, list)
            and pipeline_output
            and isinstance(pipeline_output[0], list)
        ):
            pipeline_output = pipeline_output[0]
        # Full distribution: list of {label, score} dicts (top_k=None).
        if isinstance(pipeline_output, list):
            scores = {r["label"].lower(): r["score"] for r in pipeline_output}
            entail = scores.get("entailment", scores.get("label_2", None))
            if entail is not None:
                return entail
            logger.warning(
                "NLI entailment class not found in distribution for model '%s' "
                "(labels=%s); fail-closed → 0.0",
                model_name,
                list(scores.keys()),
            )
            return 0.0
        # Single top label (fallback — less accurate, class still identifiable)
        label = pipeline_output.get("label", "").lower()
        score = pipeline_output.get("score", 0.0)
        if "entail" in label or "label_2" in label:
            return score
        if "contradict" in label or "label_0" in label:
            return 1.0 - score
        logger.warning(
            "NLI output label '%s' unrecognized for model '%s'; fail-closed → 0.0",
            label,
            model_name,
        )
        return 0.0

    @staticmethod
    def _extract_verdict_label(pipeline_output, model_name: str = "") -> str:
        
        if (
            isinstance(pipeline_output, list)
            and pipeline_output
            and isinstance(pipeline_output[0], list)
        ):
            pipeline_output = pipeline_output[0]

        def _canon(label: str) -> Optional[str]:
            label = label.lower()
            if "entail" in label or "label_2" in label:
                return "entailment"
            if "contradict" in label or "label_0" in label:
                return "contradiction"
            if "neutral" in label or "label_1" in label:
                return "neutral"
            return None

        if isinstance(pipeline_output, list):
            best_label = None
            best_score = -1.0
            for r in pipeline_output:
                score = r.get("score", 0.0)
                if score > best_score:
                    best_score = score
                    best_label = r.get("label", "")
            canon = _canon(best_label or "")
            if canon is not None:
                return canon
            logger.warning(
                "NLI verdict label not identifiable from distribution for model "
                "'%s' (argmax='%s'); fail-closed → 'contradiction'",
                model_name,
                best_label,
            )
            return "contradiction"
        # Single top label
        canon = _canon(pipeline_output.get("label", ""))
        if canon is not None:
            return canon
        logger.warning(
            "NLI verdict label '%s' unrecognized for model '%s'; fail-closed → "
            "'contradiction'",
            pipeline_output.get("label", ""),
            model_name,
        )
        return "contradiction"

    # ── Truncation policy ───────────────────────────────────────

    @staticmethod
    def _truncate_for_nli(text: str, max_tokens: int = settings.MAX_NLI_TOKENS) -> str:
        """Head truncation for NLI inference. Preserves head of prompt."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    # ── Tier dispatch ──────────────────────────────────────────────

    @staticmethod
    def _resolve_operator_id(strategy: str) -> Tuple[str, Tier]:
        """Map strategy name to canonical ID + tier via OperatorRegistry."""
        tier = VariationValidator._resolve_tier(strategy)
        return strategy, tier

    @staticmethod
    def _resolve_tier(operator_id: str) -> Tier:
        """Resolve tier from operator_id via OperatorRegistry."""
        try:
            from src.core.schemas.task import VariationStrategy

            strat_enum = VariationStrategy(operator_id)
            reg_tier = OperatorRegistry.get_tier(strat_enum)
            if reg_tier is not None:
                return reg_tier
        except (ValueError, KeyError):
            pass
        return Tier.C

    @staticmethod
    def _get_layer1_params(operator_id: str) -> Dict[str, float]:
        params = settings.OPERATOR_LAYER1.get(operator_id)
        if params is not None:
            return params
        tier = VariationValidator._resolve_tier(operator_id)
        if tier == Tier.B:
            return settings.TIER_B_DEFAULTS
        return settings.TIER_C_DEFAULTS

    # ── Lineage enforcement ────────────────────────────────────────

    def _check_lineage(self, meta: Dict[str, Any]) -> bool:
        if not self._lineage_configured:
            # T4 fix: strict by default — no generator_model → REJECT (fail-closed).
            # Set TRUSTVAR_LINEAGE_STRICT=0 to allow permissive mode for local debugging.
            if os.environ.get("TRUSTVAR_LINEAGE_STRICT", "1") != "0":
                meta["lineage_error"] = (
                    "generator_model not configured; lineage enforcement is strict "
                    "by default. "
                    "Set 'generator_model' in validator config, or set "
                    "TRUSTVAR_LINEAGE_STRICT=0 for permissive local debugging."
                )
                return False
            meta["lineage_warning"] = (
                "generator_model not explicitly configured; using default "
                f"'{self.generator_family}' which may not reflect the actual generator. "
                "Set 'generator_model' in validator config for accurate lineage enforcement."
            )
        if self.generator_family in self.verifier_families:
            logger.error(
                "Lineage violation: generator family '%s' overlaps with "
                "verifier families %s. REJECT.",
                self.generator_family,
                self.verifier_families,
            )
            meta["lineage_error"] = (
                f"generator={self.generator_family} overlaps verifier"
            )
            return False
        meta["generator_family"] = self.generator_family
        meta["verifier_families"] = self.verifier_families
        return True

    # ── Public API ─────────────────────────────────────────────────

    async def validate_batch(
        self,
        original: str,
        variations: List[Dict[str, Any]],
        target: Optional[str] = None,
        language: str = "en",
        task_type: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Validate all variations in a batch. Uses asyncio.gather for parallel NLI.
        """
        tasks = [
            self.validate(
                original=original,
                variation=var["text"],
                strategy=var.get("strategy", "paraphrase_lexico_syntactic_constrained"),
                target=target,
                language=language,
                task_type=task_type,
                operator_metadata=var.get("operator_metadata"),
            )
            for var in variations
        ]
        results = await asyncio.gather(*tasks)

        validated = []
        for var, (passed, status, meta) in zip(variations, results):
            validated.append(
                {
                    **var,
                    "valid": passed,
                    "validation_status": status.value,
                    "validation_metadata": meta,
                }
            )
        return validated

    async def validate(
        self,
        original: str,
        variation: str,
        strategy: str,
        target: Optional[str] = None,
        language: str = "en",
        task_type: str = "unknown",
        operator_metadata: Optional[Dict[str, Any]] = None,
        template_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, ValidationStatus, Dict[str, Any]]:
        """
        Full validation cascade with Tier dispatch.
        """
        meta: Dict[str, Any] = {}
        operator_id, tier = self._resolve_operator_id(strategy)

        meta["operator_id"] = operator_id
        meta["tier"] = tier.value

        # ── Layer 0: Lineage enforcement (all tiers need generator/verifier check) ──
        if not self._check_lineage(meta):
            return False, ValidationStatus.REJECT_LINEAGE, meta

        # ── Layer 1: Symbolic / Lexical-Semantic Screening ──
        layer1_ok, layer1_meta = self._check_layer1(
            original,
            variation,
            operator_id,
            tier,
            language,
            strategy,
            operator_metadata=operator_metadata,
            template_schema=template_schema,
        )
        meta.update(layer1_meta)
        if not layer1_ok:
            return False, ValidationStatus.REJECT_LEXICAL, meta

        # Tier A: no further checks
        if tier == Tier.A:
            return True, ValidationStatus.ACCEPT, meta

        # ── Layer 2: Bidirectional NLI Ensemble ──
        layer2_ok, layer2_meta = await self._check_layer2(
            original, variation, operator_id, language
        )
        meta.update(layer2_meta)
        if not layer2_ok:
            return False, ValidationStatus.REJECT_SEMANTIC, meta

        # Tier B: FLAG on NLI disagreement or a soft operator flag, but ACCEPT.
        # Operator-level FLAG metas (verb_lemma / adj_registry / adj_ud /
        # nominalisation / backtranslation …) are set during Layer 1 for Tier B
        # operators and were previously dropped here — route them to FLAG_MARGINAL
        # via the shared helper, same as the Tier C verdict below.
        if tier == Tier.B:
            if layer2_meta.get("disagreement_flag", False):
                self._enqueue_audit(original, variation, operator_id, meta)
                return True, ValidationStatus.FLAG_DISAGREEMENT, meta
            if self._flag_marginal_verdict(original, variation, operator_id, meta):
                return True, ValidationStatus.FLAG_MARGINAL, meta
            return True, ValidationStatus.ACCEPT, meta

        # ── Layer 3: Task-aware answer preservation (Tier C only) ──
        layer3_ok, layer3_meta = self._check_layer3(
            original, variation, operator_id, target, task_type, language=language
        )
        meta.update(layer3_meta)
        if not layer3_ok:
            return False, ValidationStatus.REJECT_LOGIC, meta

        # ── Verdict ──
        if layer2_meta.get("disagreement_flag", False):
            self._enqueue_audit(original, variation, operator_id, meta)
            return True, ValidationStatus.FLAG_DISAGREEMENT, meta
        if self._flag_marginal_verdict(original, variation, operator_id, meta):
            return True, ValidationStatus.FLAG_MARGINAL, meta
        return True, ValidationStatus.ACCEPT, meta

    def _enqueue_audit(
        self,
        original: str,
        variation: str,
        operator_id: str,
        meta: Dict[str, Any],
    ) -> None:
       
        self.audit_queue.append(
            {
                "original": original,
                "variation": variation,
                "operator_id": operator_id,
                "meta": meta,
            }
        )

    def _flag_marginal_verdict(
        self,
        original: str,
        variation: str,
        operator_id: str,
        meta: Dict[str, Any],
    ) -> bool:
        if any(meta.get(flag, False) for flag in _OPERATOR_FLAG_METAS):
            self._enqueue_audit(original, variation, operator_id, meta)
            return True
        return False

    # ── Layer 1: Symbolic & Lexical-Semantic Gate ──────────────────

    def _check_layer1(
        self,
        original: str,
        variation: str,
        operator_id: str,
        tier: Tier,
        language: str,
        strategy: str,
        operator_metadata: Optional[Dict[str, Any]] = None,
        template_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        params = self._get_layer1_params(operator_id)
        orig = original.strip()
        var = variation.strip()

        # 1. Identity guard (skip for Tier A — normalisation operators legitimately
        #    produce no change when input is already canonical; for B/C identity
        #    means generation failure).
        if orig == var and tier != Tier.A:
            meta["reason"] = "identity_match"
            return False, meta

        # 2. Trivial length (skip for parametric substitution — a single digit is valid)
        if operator_id != "typed_parametric_substitution" and len(var) < 5:
            meta["reason"] = "trivial_length"
            return False, meta

        # 3. Placeholder integrity (skip for typed_parametric_substitution — it
        #    substitutes placeholders with values, which is its purpose)
        if operator_id != "typed_parametric_substitution":
            orig_ph = set(self.placeholder_pattern.findall(original))
            var_ph = set(self.placeholder_pattern.findall(variation))
            if not orig_ph.issubset(var_ph):
                meta["reason"] = "missing_placeholders"
                meta["missing"] = list(orig_ph - var_ph)
                return False, meta

        # 4. Language ID guard
        detected_lang = self._detect_dominant_lang(var)
        target_lang = language
        if strategy == "translate_ru":
            target_lang = "ru"
        elif strategy == "translate_en":
            target_lang = "en"
        # Only reject on mismatch for Tier B/C (Tier A operators are structural / language-independent)
        # Skip for back_translation_single_pivot — cross-lingual by design: original in one
        # language, back-translated variant in another.
        if (
            tier != Tier.A
            and detected_lang != target_lang
            and strategy != "back_translation_single_pivot"
        ):
            meta["reason"] = "language_mismatch"
            meta["detected"] = detected_lang
            meta["expected"] = target_lang
            return False, meta

        # 5. Jaccard similarity (sub-strategy-aware for length_variation)
        #    Skip for typed_parametric_substitution / active_passive_voice —
        #    the former replaces placeholders with values, the latter completely
        #    restructures sentence (word forms change, Jaccard ~0 for inflected langs).
        _skip_content_checks = operator_id in (
            "typed_parametric_substitution",
            "active_passive_voice",
            "controlled_descriptive_modifier_insertion",
            "controlled_syntactic_transformations",
            "nominalisation",
            "back_translation_single_pivot",
        )
        if not _skip_content_checks:
            jaccard = self._jaccard(orig, var)
            meta["jaccard"] = jaccard
            if operator_id == "length_variation":
                _lv_ratio = len(var.split()) / max(len(orig.split()), 1)
                _lv_jaccard_min = (
                    params.get("jaccard_min_compression", params["jaccard_min"])
                    if _lv_ratio < 0.95
                    else params["jaccard_min"]
                )
                if jaccard < _lv_jaccard_min:
                    meta["reason"] = "jaccard_below_min"
                    meta["threshold"] = _lv_jaccard_min
                    meta["detected_mode"] = (
                        "compression" if _lv_ratio < 0.95 else "extension"
                    )
                    return False, meta
            elif jaccard < params["jaccard_min"]:
                meta["reason"] = "jaccard_below_min"
                meta["threshold"] = params["jaccard_min"]
                return False, meta

        # 6. Length ratio (sub-strategy-aware for length_variation)
        if not _skip_content_checks:
            ratio = len(var) / max(len(orig), 1)
            meta["length_ratio"] = ratio
            if operator_id == "length_variation":
                _lv_word_ratio = len(var.split()) / max(len(orig.split()), 1)
                if _lv_word_ratio > 1.05:  # extension mode
                    if _lv_word_ratio < 1.15 or _lv_word_ratio > 1.55:
                        meta["reason"] = "length_ratio_out_of_bounds"
                        meta["bounds"] = (1.15, 1.55)
                        meta["detected_mode"] = "extension"
                        return False, meta
                elif _lv_word_ratio < 0.95:  # compression mode
                    if _lv_word_ratio < 0.65 or _lv_word_ratio > 0.90:
                        meta["reason"] = "length_ratio_out_of_bounds"
                        meta["bounds"] = (0.65, 0.90)
                        meta["detected_mode"] = "compression"
                        return False, meta
            elif ratio < params["length_min"] or ratio > params["length_max"]:
                meta["reason"] = "length_ratio_out_of_bounds"
                meta["bounds"] = (params["length_min"], params["length_max"])
                return False, meta

        _skip_neural_similarity = operator_id == "controlled_descriptive_modifier_insertion"
        if tier != Tier.A and not _skip_neural_similarity:
            # 8. Embedding cosine — surface-invariant (Tier B/C only)
            cos = self._cosine_similarity(orig, var)
            meta["cosine"] = cos
            if cos is not None:
                if operator_id == "length_variation":
                    _lv_cosine_min = (
                        params.get("cosine_min_compression", params["cosine_min"])
                        if meta.get("detected_mode") == "compression"
                        else params["cosine_min"]
                    )
                    if cos < _lv_cosine_min:
                        meta["reason"] = "cosine_below_min"
                        meta["threshold"] = _lv_cosine_min
                        return False, meta
                elif cos < params["cosine_min"]:
                    meta["reason"] = "cosine_below_min"
                    meta["threshold"] = params["cosine_min"]
                    return False, meta

            # 9. BERTScore F1 — surface-invariant (Tier B/C only)
            bs = self._bertscore_f1(orig, var, detected_lang)
            meta["bertscore_f1"] = bs
            if bs is not None and bs < params["bertscore_min"]:
                meta["reason"] = "bertscore_below_min"
                meta["threshold"] = params["bertscore_min"]
                return False, meta

        if not _skip_content_checks:
            # 7. Function-word chi-square (skip on error/missing dep)
            chi2_p = self._function_word_chi2(orig, var, detected_lang)
            meta["chi2_p"] = chi2_p
            if chi2_p is not None and chi2_p < params["chi2_max"]:
                meta["reason"] = "function_word_distribution_shift"
                meta["threshold"] = params["chi2_max"]
                return False, meta

            # 10. Named-entity preservation
            if operator_id not in (
                "orthographic_normalization_ru",
                "controlled_descriptive_modifier_insertion",
                "controlled_syntactic_transformations",
                "back_translation_single_pivot",
            ):
                _NE_MULTI = re.compile(r"[A-ZА-Я][a-zа-яё]+(?:\s+[A-ZА-Я][a-zа-яё]+)+")
                # Single-word NE: capitalized word not at beginning of sentence (fixes NEW-8).
                # TECH-002: filter out common sentence-starters to avoid false positives at doc start.
                _NE_SINGLE = re.compile(r"(?<![.\n]\s)[A-ZА-Я][a-zа-яё]+")
                orig_nes = set(_NE_MULTI.findall(original)) | {
                    w
                    for w in _NE_SINGLE.findall(original)
                    if w not in _SENTENCE_STARTERS
                }
                var_nes = set(_NE_MULTI.findall(variation)) | {
                    w
                    for w in _NE_SINGLE.findall(variation)
                    if w not in _SENTENCE_STARTERS
                }
                if orig_nes and not orig_nes.issubset(var_nes):
                    missing = orig_nes - var_nes
                    # Allow partial preservation (at least one word of multi-word NE)
                    partial_ok = True
                    for ne in missing:
                        words = ne.split()
                        if not any(w in " ".join(var_nes) for w in words):
                            partial_ok = False
                            break
                    if not partial_ok:
                        meta["reason"] = "ne_preservation_failed"
                        meta["missing_nes"] = list(missing)
                        return False, meta

        # 11. Operator-specific hooks (resolved per operator)
        hooks_ok, hooks_meta = self._apply_operator_layer1_hooks(
            original,
            variation,
            operator_id,
            language,
            operator_metadata=operator_metadata,
            template_schema=template_schema,
        )
        meta.update(hooks_meta)
        if not hooks_ok:
            meta["reason"] = hooks_meta.get("reason", "operator_hook_failed")
            return False, meta

        return True, meta

    # ── Operator-specific Layer 1 hooks ────────────────────────────

    @staticmethod
    def _apply_operator_layer1_hooks(
        original: str,
        variation: str,
        operator_id: str,
        language: str,
        operator_metadata: Optional[Dict[str, Any]] = None,
        template_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        # ────────────────────────────────────────────────────────────
        # Tier A — format_normalization (A1)
        # ────────────────────────────────────────────────────────────
        if operator_id == "format_normalization":
            # Banned patterns: REJECT if variation introduces MCQ markers or code fences
            _BANNED = [r"(?<!\S)[A-Da-dА-Га-г][\)\.](?!\S)", r"```"]
            orig_banned = [bool(re.search(p, original)) for p in _BANNED]
            var_banned = [bool(re.search(p, variation)) for p in _BANNED]
            for i, (ob, vb) in enumerate(zip(orig_banned, var_banned)):
                if vb and not ob:
                    meta["reason"] = "banned_token_introduced"
                    meta["pattern_idx"] = i
                    return False, meta

            # Strict content Jaccard ≥ 0.80
            _ct = lambda t: set(re.sub(r"[^\w\s]", "", t.lower()).split())
            ct_orig = _ct(original)
            ct_var = _ct(variation)
            if ct_orig and ct_var:
                overlap = len(ct_orig & ct_var) / max(len(ct_orig | ct_var), 1)
                meta["content_jaccard"] = round(overlap, 3)
                if overlap < 0.80:
                    meta["reason"] = "content_token_drift"
                    return False, meta

            # Round-trip stability: re-apply normalizations; must produce same result
            try:
                import unicodedata

                re_norm = unicodedata.normalize("NFC", variation)
                re_norm = re.sub(r"\s+", " ", re_norm).strip()
                re_norm = re.sub(r"[ \t]*\n[ \t]*", "\n", re_norm)
                if re_norm != variation:
                    meta["reason"] = "round_trip_unstable"
                    return False, meta
                meta["round_trip_stable"] = True
            except Exception:
                meta["round_trip_stable"] = "skipped"

            meta["format_normalization_check"] = "passed"
            return True, meta

        # ────────────────────────────────────────────────────────────
        # Tier A — list_reordering (A4)
        # ────────────────────────────────────────────────────────────
        if operator_id == "list_reordering":
            # Detect enumerated list items and verify item count + bijection
            _LIST_LINE = re.compile(
                r"^\s*(?:\d+[\.\)]|[a-zA-Zа-яА-Я][\.\)]|[→▪•●○■□])\s+(.+)",
                re.MULTILINE,
            )
            orig_items = _LIST_LINE.findall(original)
            var_items = _LIST_LINE.findall(variation)

            if not orig_items and not var_items:
                # Fallback: line‑based reordering (comma‑separated lists)
                orig_lines = [
                    l.strip() for l in original.strip().split("\n") if l.strip()
                ]
                var_lines = [
                    l.strip() for l in variation.strip().split("\n") if l.strip()
                ]
                if len(orig_lines) > 1:
                    orig_items = orig_lines
                    var_items = var_lines

            if len(orig_items) != len(var_items):
                meta["reason"] = "item_count_mismatch"
                meta["orig_count"] = len(orig_items)
                meta["var_count"] = len(var_items)
                return False, meta

            if sorted(orig_items) != sorted(var_items):
                meta["reason"] = "item_text_set_mismatch"
                return False, meta

            orig_pos = {t: i for i, t in enumerate(orig_items)}
            var_pos = {t: i for i, t in enumerate(var_items)}

            # Bijection: each orig item → exactly one var position
            perm = [var_pos.get(t, -1) for t in orig_items]
            if -1 in perm:
                meta["reason"] = "missing_text_in_variant"
                return False, meta
            if len(set(perm)) != len(orig_items):
                meta["reason"] = "not_a_bijection"
                return False, meta

            meta["item_count"] = len(orig_items)
            meta["permuted"] = orig_items != var_items
            meta["list_reordering_check"] = "passed"
            return True, meta

        # ────────────────────────────────────────────────────────────
        # Tier A — mcq_option_permutation (A3)
        # ────────────────────────────────────────────────────────────
        if operator_id == "mcq_option_permutation":
            _LC_EN = [c + s for c in "ABCDEFG" for s in ").:"] + [
                c + s for c in "abcdefg" for s in ").:"
            ]
            _LC_RU = [c + s for c in "АБВГДЕЁЖ" for s in ").:"] + [
                c + s for c in "абвгдеёж" for s in ").:"
            ]
            # start at 0 so 0-indexed schemes (e.g. BBQ: "0.", "1.", "2.") parse.
            _NUM = [f"{n}{s}" for n in range(0, 21) for s in ").:"]
            _PAREN_LETTER_EN = [f"({c})" for c in "ABCDEFGabcdefg"]
            _PAREN_LETTER_RU = [f"({c})" for c in "АБВГДЕЁЖабвгдеёж"]
            _PAREN_NUM = [f"({n})" for n in range(0, 11)]
            _ALL_MARKERS = frozenset(
                _LC_EN + _LC_RU + _NUM + _PAREN_LETTER_EN + _PAREN_LETTER_RU + _PAREN_NUM
            )

            _OPTION_MARKER = re.compile(
                r"^("
                r"\([A-Ga-gА-Жа-жёЁ]\)"
                r"|"
                r"\([0-9]\d?\)"
                r"|"
                r"[A-Ga-gА-Жа-жёЁ][\)\.:]"
                r"|"
                r"[0-9]\d{0,1}[\)\.:]"
                r")\s*(.+)",
                re.MULTILINE,
            )

            orig_opts = _OPTION_MARKER.findall(original)
            var_opts = _OPTION_MARKER.findall(variation)

            if len(orig_opts) != len(var_opts):
                meta["reason"] = "option_count_mismatch"
                return False, meta

            orig_texts = sorted(t.lower().strip() for _, t in orig_opts)
            var_texts = sorted(t.lower().strip() for _, t in var_opts)
            if orig_texts != var_texts:
                meta["reason"] = "option_text_set_mismatch"
                return False, meta

            for marker, _ in var_opts:
                if marker not in _ALL_MARKERS:
                    meta["reason"] = "invalid_marker"
                    meta["marker"] = marker
                    return False, meta

            orig_order = [m for m, _ in orig_opts]
            var_order = [m for m, _ in var_opts]
            if orig_order != var_order:
                meta["reason"] = "marker_order_mismatch"
                return False, meta

            meta["option_count"] = len(orig_opts)
            meta["mcq_check"] = "passed"
            return True, meta

        # ────────────────────────────────────────────────────────────
        # Tier A — orthographic_normalization_ru (A2)
        # ────────────────────────────────────────────────────────────
        if operator_id == "orthographic_normalization_ru":
            _YO_RE = re.compile(r"[ёЁ]")
            var_has_yo = bool(_YO_RE.search(variation))

            # Forward+reverse: apply basic yo→e to both; should match
            _basic_yo_to_e = lambda t: re.sub(
                r"[ёЁ]", lambda m: "е" if m.group() == "ё" else "Е", t
            )
            norm_orig = _basic_yo_to_e(original)
            norm_var = _basic_yo_to_e(variation)
            if norm_orig != norm_var:
                meta["reason"] = "forward_reverse_mismatch"
                return False, meta

            # No non‑Cyrillic content tokens introduced
            _CYRILLIC_ONLY = re.compile(r"^[а-яё]+$", re.IGNORECASE)
            orig_tokens = {
                t.lower().strip('.,!?;:"()[]{}«»—–-')
                for t in original.split()
                if t.strip('.,!?;:"()[]{}«»—–-')
            }
            var_tokens = {
                t.lower().strip('.,!?;:"()[]{}«»—–-')
                for t in variation.split()
                if t.strip('.,!?;:"()[]{}«»—–-')
            }
            new_tokens = var_tokens - orig_tokens
            non_cyrillic = [t for t in new_tokens if t and not _CYRILLIC_ONLY.match(t)]
            if non_cyrillic:
                meta["reason"] = "non_cyrillic_content_introduced"
                meta["new_tokens"] = non_cyrillic[:10]
                return False, meta

            meta["yo_folding"] = True
            meta["var_has_yo"] = var_has_yo
            meta["orthographic_check"] = "passed"
            return True, meta

        # ────────────────────────────────────────────────────────────
        # Tier C — wsd_synonym_substitution (C7)
        # ────────────────────────────────────────────────────────────
        if operator_id == "wsd_synonym_substitution":
            orig_w = set(re.findall(r"\w+", original.lower()))
            var_w = set(re.findall(r"\w+", variation.lower()))
            new_w = var_w - orig_w
            if new_w:
                unknown = []
                try:
                    from nltk.corpus import wordnet as _wn

                    for nw in list(new_w)[:10]:
                        if language == "ru":
                            try:
                                from src.core.operators.utils.nlp_utils import (
                                    check_ru_lexicon,
                                )

                                ok = check_ru_lexicon(nw)
                            except Exception:
                                ok = False
                        else:
                            # Check direct synsets first
                            synsets = _wn.synsets(nw)
                            if not synsets:
                                # Try common inflection endings as lemma hints
                                for suffix in ("ed", "ing", "s", "es", "ies"):
                                    if (
                                        nw.endswith(suffix)
                                        and len(nw) > len(suffix) + 2
                                    ):
                                        synsets = _wn.synsets(nw[: -len(suffix)])
                                        if synsets:
                                            break
                            ok = bool(synsets)
                        if not ok:
                            unknown.append(nw)
                except Exception:
                    meta["lexicon_skipped"] = "wordnet_unavailable"
                if unknown:
                    meta["reason"] = "new_word_not_in_lexicon"
                    meta["unknown_words"] = unknown
                    return False, meta
            meta["wsd_lexicon_check"] = "passed"
            return True, meta

        if operator_id == "back_translation_single_pivot":
            # Length ratio check — flag extreme deviations (per original operator spec)
            _orig_len = len(original.split())
            _var_len = len(variation.split())
            _ratio = _var_len / max(_orig_len, 1)
            meta["length_ratio"] = round(_ratio, 3)
            if _ratio < 0.70 or _ratio > 1.40:
                meta["reason"] = "length_ratio_out_of_bounds"
                meta["ratio"] = round(_ratio, 3)
                return False, meta
            # NE preservation: only check for same-language back-translation
            # (cross-lingual transliteration makes regex NE matching unreliable).
            _CYR_RE = re.compile(r"[а-яёА-ЯЁ]")
            orig_has_cyr = bool(_CYR_RE.search(original))
            var_has_cyr = bool(_CYR_RE.search(variation))
            _same_lang = orig_has_cyr == var_has_cyr  # both Latin or both Cyrillic
            if _same_lang:
                _C8_NE = re.compile(r"[A-ZА-ЯЁ][a-zа-яё]+")
                _SENTENCE_STARTERS_RE = re.compile(r"(?:^|[.!?]\s+)[A-ZА-ЯЁ][a-zа-яё]+")
                orig_ne = set(_C8_NE.findall(original))
                var_ne = set(_C8_NE.findall(variation))
                if orig_ne and not orig_ne.issubset(var_ne):
                    missing = orig_ne - var_ne
                    meta["reason"] = "ne_preservation_failed"
                    meta["missing_ne"] = list(missing)[:5]
                    return False, meta
            # Numeric preservation (digits are language-independent)
            orig_num = set(re.findall(r"\b\d+\b", original))
            var_num = set(re.findall(r"\b\d+\b", variation))
            if orig_num and not orig_num.issubset(var_num):
                meta["reason"] = "numeric_preservation_failed"
                meta["missing_numbers"] = list(orig_num - var_num)
                return False, meta
            meta["ne_preserved"] = True
            meta["numeric_preserved"] = True
            _IDIOM_STOP = frozenset(_load_yaml_data("idiom_stop_words.yaml"))
            try:
                from src.core.operators.utils.nlp_utils import (
                    _EN_FIXED_EXPRESSIONS,
                    _RU_FIXED_EXPRESSIONS,
                )

                idioms = (
                    _EN_FIXED_EXPRESSIONS if language == "en" else _RU_FIXED_EXPRESSIONS
                )
                orig_lower = original.lower()
                for idiom in idioms:
                    if idiom in orig_lower:
                        idiom_keywords = {
                            w
                            for w in idiom.split()
                            if w not in _IDIOM_STOP and len(w) > 3
                        }
                        var_lower = variation.lower()
                        preserved = any(w in var_lower for w in idiom_keywords)
                        if not preserved:
                            meta["idiom_flattening_flag"] = True
                            meta["idiom_possible_flattening"] = idiom
                        break
            except Exception:
                pass

        elif operator_id == "active_passive_voice":
            # ── B2: Voice flip + verb lemma preservation via UD ──
            try:
                from src.core.operators.utils.nlp_utils import (
                    get_stanza_pipeline,
                    get_transitive_voice_candidates,
                )

                lang = language
                stz = get_stanza_pipeline(lang)
                if stz is not None:
                    orig_candidates = get_transitive_voice_candidates(original, lang)
                    var_candidates = get_transitive_voice_candidates(variation, lang)
                    orig_voices = {c["voice"] for c in orig_candidates}
                    var_voices = {c["voice"] for c in var_candidates}
                    if orig_voices and var_voices and not (orig_voices ^ var_voices):
                        meta["reason"] = "voice_not_flipped"
                        meta["orig_voices"] = list(orig_voices)
                        meta["var_voices"] = list(var_voices)
                        return False, meta
                    orig_verbs = {
                        c["verb"].lemma
                        for c in orig_candidates
                        if c.get("verb") and c["verb"].lemma
                    }
                    var_verbs = {
                        c["verb"].lemma
                        for c in var_candidates
                        if c.get("verb") and c["verb"].lemma
                    }
                    if orig_verbs and not (orig_verbs & var_verbs):
                        meta["reason"] = "verb_lemma_changed"
                        meta["orig_verbs"] = list(orig_verbs)
                        meta["var_verbs"] = list(var_verbs)
                        meta["verb_lemma_flag"] = True
                        # Don't return False — original operator FLAG'd, not REJECT'd
                    meta["voice_flip"] = list(orig_voices ^ var_voices)
                    meta["verbs_preserved"] = list(orig_verbs & var_verbs)
                else:
                    meta["ud_skipped"] = "no_stanza_pipeline"
            except Exception:
                meta["ud_skipped"] = "stanza_import_failed"

        elif operator_id == "nominalisation":
            # ── B3: Nominalisation pattern + case agreement via UD ──
            try:
                from src.core.operators.tier_b.nominalisation import (
                    _find_denominalisation_candidates,
                    _find_nominalisation_candidates,
                    _get_lexicon,
                    _get_rev_lexicon,
                )
                from src.core.operators.utils.nlp_utils import (
                    get_stanza_pipeline,
                    parse_ud,
                )
                from src.core.operators.utils.nlp_utils import (
                    parse_feats as _parse_ud_feats,
                )

                lang = language
                orig_doc = parse_ud(original, lang)
                var_doc = parse_ud(variation, lang)
                if orig_doc is not None and var_doc is not None:
                    orig_lemmas = set()
                    for s in orig_doc.sentences:
                        for w in s.words:
                            l = (w.lemma or "").lower().strip()
                            if l and w.upos not in ("PUNCT", "SYM", "X"):
                                orig_lemmas.add(l)
                    var_lemmas = set()
                    for s in var_doc.sentences:
                        for w in s.words:
                            l = (w.lemma or "").lower().strip()
                            if l and w.upos not in ("PUNCT", "SYM", "X"):
                                var_lemmas.add(l)
                    if orig_lemmas and var_lemmas:
                        overlap = len(orig_lemmas & var_lemmas) / max(
                            len(orig_lemmas | var_lemmas), 1
                        )
                    else:
                        overlap = 0.0
                    meta["lemma_jaccard"] = round(overlap, 3)
                    if overlap < 0.25:
                        meta["reason"] = "content_drift_lemma"
                        meta["nominalisation_flag"] = True
                        # Don't return False — original returned FLAG, not REJECT
                stz = get_stanza_pipeline(lang)
                if stz is not None:
                    var_doc2 = parse_ud(variation, lang)
                    if var_doc2 is not None:
                        lexicon = _get_lexicon(lang)
                        fwd = _find_nominalisation_candidates(var_doc2, lexicon, lang)
                        if fwd:
                            cand = fwd[0]
                            noun = cand["deverbal_noun"]
                            subj = cand["subject"]
                            obj = cand["object"]
                            if lang == "en":
                                if obj is not None:
                                    obj_feats = _parse_ud_feats(
                                        getattr(obj, "feats", None)
                                    )
                                    if obj_feats.get("Case", "").lower() not in (
                                        "nomn",
                                        "nom",
                                        "acc",
                                        "accs",
                                    ):
                                        meta["reason"] = "obj_not_in_expected_case"
                                        meta["nominalisation_flag"] = True
                                meta["nominalisation_check"] = (
                                    "forward_nominalisation_ud"
                                )
                                meta["deverbal_noun"] = str(noun)
                            elif lang == "ru":
                                checks_ok = True
                                if obj is not None:
                                    obj_feats = _parse_ud_feats(
                                        getattr(obj, "feats", None)
                                    )
                                    if obj_feats.get("Case", "").lower() not in (
                                        "gent",
                                        "gen",
                                    ):
                                        checks_ok = False
                                if subj is not None:
                                    subj_feats = _parse_ud_feats(
                                        getattr(subj, "feats", None)
                                    )
                                    if subj_feats.get("Case", "").lower() not in (
                                        "ablt",
                                        "ins",
                                    ):
                                        checks_ok = False
                                if not checks_ok:
                                    meta["reason"] = "ru_case_mismatch"
                                    meta["nominalisation_flag"] = True
                                meta["nominalisation_check"] = (
                                    "forward_nominalisation_ud"
                                )
                                meta["deverbal_noun"] = str(noun)
                        else:
                            rev_lexicon = _get_rev_lexicon(lang)
                            if rev_lexicon:
                                rev = _find_denominalisation_candidates(
                                    var_doc2, rev_lexicon, lang
                                )
                                if rev:
                                    meta["nominalisation_check"] = (
                                        "reverse_denominalisation_ud"
                                    )
                            if "nominalisation_check" not in meta:
                                meta["reason"] = "no_nominalisation_pattern_detected"
                                meta["nominalisation_flag"] = True
                                # Don't return False — original returned FLAG, not REJECT
                    else:
                        meta["ud_skipped"] = "variant_parse_failed"
                else:
                    meta["ud_skipped"] = "no_stanza_pipeline"
            except Exception:
                meta["ud_skipped"] = "nominalisation_import_failed"

        elif operator_id == "monosemic_synonym_substitution":
            # ── B5: WordNet/RuWordNet lexicon verification ──
            try:
                import nltk

                nltk.data.find("corpora/wordnet")
                from nltk.corpus import wordnet as wn

                HAS_WN = True
            except Exception:
                HAS_WN = False
            try:
                from src.core.operators.utils.nlp_utils import (
                    get_unique_synset_ru,
                )

                HAS_RUWN = True
            except Exception:
                HAS_RUWN = False
            orig_words = set(re.findall(r"[a-zа-яё]+", original.lower()))
            var_words = set(re.findall(r"[a-zа-яё]+", variation.lower()))
            new_words = var_words - orig_words
            if new_words:
                unknown = []
                for w in list(new_words)[:10]:
                    if language == "ru" and HAS_RUWN:
                        ss = get_unique_synset_ru(w)
                        if ss is None:
                            unknown.append(w)
                    elif language != "ru" and HAS_WN:
                        if not wn.synsets(w):
                            unknown.append(w)
                if unknown:
                    meta["reason"] = "new_words_not_in_lexicon"
                    meta["unknown_words"] = unknown
                    return False, meta
            meta["lexicon_check"] = "passed"

        elif operator_id == "typed_parametric_substitution":
            # ── A5: Slot preservation + type checking + constraints ──
            # Identity: no change is valid (text already matches constraints)
            if variation == original:
                meta["identity_accepted"] = True
                return True, meta

            _TEMPLATE_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
            orig_slots = set(_TEMPLATE_SLOT_RE.findall(original))
            var_slots = set(_TEMPLATE_SLOT_RE.findall(variation))
            added = var_slots - orig_slots
            if added:
                meta["reason"] = "new_slots_added"
                meta["added"] = list(added)
                return False, meta
            meta["slots_preserved"] = True

            # Delegate complex validation to operator utilities
            try:
                from src.core.operators.tier_a.typed_parametric_substitution import (
                    SlotType,
                    _evaluate_constraint,
                    _extract_type_annotations,
                    _parse_math_constraint,
                    _split_template,
                )
                from src.core.operators.tier_a.typed_parametric_substitution import (
                    TypedParametricSubstitutionOperator as _TPSOp,
                )

                _HAS_TYPED_UTILS = True
            except Exception:
                _HAS_TYPED_UTILS = False

            orig_template = original
            var_template = variation
            constraints_str = ""

            if _HAS_TYPED_UTILS:
                orig_template, constraints_str = _split_template(original)
                var_template, _ = _split_template(variation)

                if constraints_str:
                    type_annotations = _extract_type_annotations(constraints_str)
                    constraints = _parse_math_constraint(constraints_str)
                    extracted = {}
                    for slot_name in orig_slots:
                        raw_val = _TPSOp._extract_current(
                            orig_template, variation, slot_name
                        )
                        if raw_val == "":
                            continue
                        st = type_annotations.get(slot_name, SlotType.INT)
                        try:
                            extracted[slot_name] = _TPSOp._coerce_value(raw_val, st)
                        except (ValueError, TypeError):
                            meta["reason"] = f"type_violation_inline_{slot_name}"
                            meta["slot"] = slot_name
                            meta["expected"] = st.value
                            meta["got"] = raw_val
                            return False, meta

                    if constraints:
                        for expr in constraints:
                            if not _evaluate_constraint(expr, extracted):
                                meta["reason"] = "inline_constraint_violation"
                                meta["expr"] = expr
                                meta["values"] = extracted
                                return False, meta

                    meta["type_check"] = "inline_mode"
                    return True, meta

            # Template-schema mode (explicit TemplateSchema passed—e.g. from tests)
            if _HAS_TYPED_UTILS and template_schema is not None:
                try:
                    ts_slots = template_schema.get("slots", [])
                except AttributeError:
                    ts_slots = []
                if ts_slots:
                    
                    extracted = {}
                    for slot_def in ts_slots:
                        sname = slot_def.get("name", "")
                        stype = slot_def.get("type", "str")
                        if sname not in orig_slots:
                            continue
                        raw_val = _TPSOp._extract_current(
                            orig_template, variation, sname
                        )
                        if raw_val == "":
                            continue
                        st = SlotType(stype)
                        try:
                            extracted[sname] = _TPSOp._coerce_value(raw_val, st)
                        except (ValueError, TypeError):
                            meta["reason"] = f"type_violation_schema_{sname}"
                            meta["slot"] = sname
                            meta["expected"] = st.value
                            meta["got"] = raw_val
                            return False, meta

                        # Value pool check (str_enum)
                        spool = slot_def.get("value_pool")
                        if spool is not None and extracted[sname] not in spool:
                            meta["reason"] = f"value_not_in_pool_{sname}"
                            meta["slot"] = sname
                            meta["expected_pool"] = spool
                            meta["got"] = extracted[sname]
                            return False, meta

                   
                    for slot_def in ts_slots:
                        sname = slot_def.get("name", "")
                        sconstraints = slot_def.get("constraints") or []
                        for expr in sconstraints:
                            if not _evaluate_constraint(expr, extracted):
                                meta["reason"] = "constraint_violation"
                                meta["expr"] = expr
                                meta["slot"] = sname
                                meta["values"] = extracted
                                return False, meta

                    
                    ts_template_constraints = (
                        template_schema.get("template_constraints") or []
                    )
                    for expr in ts_template_constraints:
                        if not _evaluate_constraint(expr, extracted):
                            meta["reason"] = "template_constraint_violation"
                            meta["expr"] = expr
                            meta["values"] = extracted
                            return False, meta

                    meta["type_check"] = "schema_mode"
                    return True, meta

            # Fallback: slot‑only check (no type system or no inline constraints)
            meta["type_check"] = "slots_only"

        elif operator_id == "sentence_split_merge":
            # ── B6: Sentence count change + controller recoverability ──
            _words = lambda t: set(re.findall(r"\w+", t.lower()))
            orig_w = _words(original)
            var_w = _words(variation)
            if orig_w and var_w:
                overlap = len(orig_w & var_w) / max(len(orig_w | var_w), 1)
                meta["word_overlap"] = round(overlap, 3)
                if overlap < settings.SENTENCE_WORD_OVERLAP_FLAG:
                    meta["reason"] = "word_overlap_below_threshold"
                    return False, meta

            lang = language
            # Sentence count via regex (no UD dependency)
            _sentences = lambda t: [
                s.strip() for s in re.split(r"(?<=[.!?…])\s+", t) if s.strip()
            ]
            sent_orig = len(_sentences(original))
            sent_var = len(_sentences(variation))
            meta["sentence_count_orig"] = sent_orig
            meta["sentence_count_var"] = sent_var
            if sent_orig == sent_var:
                meta["reason"] = "sentence_count_unchanged"
                return False, meta
            meta["operation"] = "split" if sent_var > sent_orig else "merge"

            # Controller recoverability (split only): introduced_subjects must be unambiguously resolved
            if meta["operation"] == "split" and operator_metadata:
                intros = operator_metadata.get("introduced_subjects") or []
                for intro in intros:
                    copying_type = intro.get("copying_type")
                    if copying_type in ("ambiguous_controller", None):
                        meta["reason"] = "subject_copy_unrecoverable"
                        meta["copying_type"] = copying_type
                        meta["controller_text"] = intro.get("controller_text")
                        return False, meta

            # Pronoun density heuristic (existing check)
            orig_pronouns = len(
                re.findall(
                    r"\b(it|he|she|they|this|that|он|она|оно|они|это)\b",
                    original.lower(),
                )
            )
            var_pronouns = len(
                re.findall(
                    r"\b(it|he|she|they|this|that|он|она|оно|они|это)\b",
                    variation.lower(),
                )
            )
            meta["pronoun_delta"] = abs(orig_pronouns - var_pronouns)

        elif operator_id == "controlled_descriptive_modifier_insertion":
            op_meta = operator_metadata or {}
            inserted_adj = op_meta.get("adjective")
            target_noun = op_meta.get("noun")

            if inserted_adj and target_noun:
                from src.core.operators.utils.nlp_utils import get_stanza_pipeline

                if get_stanza_pipeline(language) is not None:
                    ok, m = _validator_check_ud_amod_slot(
                        variation, language, inserted_adj, target_noun
                    )
                    if not ok:
                        m["original_reason"] = m.get("reason")
                        meta.update(m)
                        # Count-limit (multiple_amod_children) is load-bearing → REJECT;
                        # amod-slot / morphology mismatches → FLAG (retain with audit).
                        if m.get("reason") in _B7_LAYER1_REJECT_REASONS:
                            return False, meta
                        meta["adj_ud_flag"] = True
                    else:
                        meta.update(m)
                else:
                    ok, m = _validator_check_lexical(original, variation, language)
                    if not ok:
                        m["original_reason"] = m.get("reason")
                        meta.update(m)
                        if m.get("reason") in _B7_LAYER1_REJECT_REASONS:
                            return False, meta
                        meta["adj_registry_flag"] = True
                    else:
                        meta.update(m)
            else:
                ok, m = _validator_check_lexical(original, variation, language)
                if not ok:
                    m["original_reason"] = m.get("reason")
                    meta.update(m)
                    # too_many_adjectives (Δ>1 registry) is load-bearing → REJECT;
                    # no_registry_adjective_found → FLAG (retain with audit).
                    if m.get("reason") in _B7_LAYER1_REJECT_REASONS:
                        return False, meta
                    meta["adj_registry_flag"] = True
                else:
                    meta.update(m)

        elif operator_id == "controlled_syntactic_transformations":
            op_meta = operator_metadata or {}
            subtransformation = op_meta.get("subtransformation")
            verb_lemma = op_meta.get("verb_lemma")

            if subtransformation:
                from src.core.operators.utils.nlp_utils import get_stanza_pipeline

                if get_stanza_pipeline(language) is not None:
                    ok, m = _validator_check_ud_subtransformation(
                        variation, language, subtransformation, verb_lemma
                    )
                    if not ok:
                        m["original_reason"] = m.get("reason")
                        meta.update(m)
                        return False, meta
                    meta.update(m)
                else:
                    ok, m = _validator_validate_jaccard(original, variation, language)
                    if not ok:
                        meta.update(m)
                        return False, meta
                    meta.update(m)
            else:
                ok, m = _validator_validate_jaccard(original, variation, language)
                if not ok:
                    meta.update(m)
                    return False, meta
                meta.update(m)

        elif operator_id == "negation_scope_preserving_rephrasing":
            # Negation polarity check with scope token heuristic.
            _C6_NEG_RE = re.compile(
                r"\b(not|n't|never|no|nothing|nobody|neither|nor|"
                r"не|нет|никогда|ничего|никто|нигде)\b",
                re.IGNORECASE,
            )
            orig_neg = _C6_NEG_RE.findall(original.lower())
            var_neg = _C6_NEG_RE.findall(variation.lower())
            has_orig = len(orig_neg) > 0
            has_var = len(var_neg) > 0
            if has_orig != has_var:
                meta["reason"] = "negation_polarity_shift"
                meta["orig_negations"] = len(orig_neg)
                meta["var_negations"] = len(var_neg)
                return False, meta
            _SCOPE_TOKEN_RE = re.compile(
                r"\b(?:not|n't|never|no|nor|не|нет)\s+(\w+)", re.IGNORECASE
            )
            orig_scope_tokens = set(_SCOPE_TOKEN_RE.findall(original.lower()))
            var_scope_tokens = set(_SCOPE_TOKEN_RE.findall(variation.lower()))
            if orig_scope_tokens:
                _confidence = len(orig_scope_tokens & var_scope_tokens) / len(
                    orig_scope_tokens
                )
            else:
                _confidence = 1.0
            meta["scope_token_heuristic_confidence"] = round(_confidence, 2)
            if orig_scope_tokens and not orig_scope_tokens & var_scope_tokens:
                meta["reason"] = "negation_scope_content_lost"
                meta["orig_scope_tokens"] = list(orig_scope_tokens)[:3]
                meta["var_scope_tokens"] = list(var_scope_tokens)[:3]
                return False, meta

        elif operator_id == "paraphrase_lexico_syntactic_constrained":
            # ── C1: Constrained paraphrase — content word preservation ──
            # Lexico-syntactic constrained paraphrasing must retain most content
            # words (nouns, verbs, adjectives, adverbs).  Failure → REJECT.
            _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
            try:
                from src.core.operators.utils.nlp_utils import get_stanza_pipeline

                stz = get_stanza_pipeline(language)
                if stz is not None:
                    orig_doc = stz(original)
                    var_doc = stz(variation)
                    orig_cw = (
                        {
                            w.lemma.lower()
                            for w in orig_doc.sentences[0].words
                            if w.upos in _CONTENT_POS
                            and not w.feats.get("Typo", "") == "Yes"
                        }
                        if orig_doc.sentences
                        else set()
                    )
                    var_cw = {
                        w.lemma.lower()
                        for s in var_doc.sentences
                        for w in s.words
                        if w.upos in _CONTENT_POS
                    }
                    if orig_cw:
                        preserved_ratio = len(orig_cw & var_cw) / len(orig_cw)
                        meta["content_word_preservation"] = round(preserved_ratio, 3)
                        meta["missing_content_words"] = list(orig_cw - var_cw)[:5]
                        if preserved_ratio < 0.70:
                            meta["reason"] = "content_words_lost"
                            meta["preserved_ratio"] = round(preserved_ratio, 3)
                            return False, meta
            except Exception:
                # Fallback: use regex-based POS heuristic
                _WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)
                orig_words = set(_WORD_RE.findall(original.lower()))
                var_words = set(_WORD_RE.findall(variation.lower()))
                if orig_words:
                    overlap = len(orig_words & var_words) / len(orig_words)
                    meta["content_word_preservation_fallback"] = round(overlap, 3)
                    if overlap < 0.60:
                        meta["reason"] = "content_words_lost"
                        meta["preserved_ratio"] = round(overlap, 3)
                        return False, meta

        elif operator_id == "paraphrase_free":
            # ── C2: Free paraphrase — content word preservation (looser) ──
            # Free paraphrasing is more aggressive; check with a lower threshold.
            _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
            try:
                from src.core.operators.utils.nlp_utils import get_stanza_pipeline

                stz = get_stanza_pipeline(language)
                if stz is not None:
                    orig_doc = stz(original)
                    var_doc = stz(variation)
                    orig_cw = (
                        {
                            w.lemma.lower()
                            for s in orig_doc.sentences
                            for w in s.words
                            if w.upos in _CONTENT_POS
                        }
                        if orig_doc.sentences
                        else set()
                    )
                    var_cw = {
                        w.lemma.lower()
                        for s in var_doc.sentences
                        for w in s.words
                        if w.upos in _CONTENT_POS
                    }
                    if orig_cw:
                        preserved_ratio = len(orig_cw & var_cw) / len(orig_cw)
                        meta["content_word_preservation"] = round(preserved_ratio, 3)
                        meta["missing_content_words"] = list(orig_cw - var_cw)[:5]
                        if preserved_ratio < 0.60:
                            meta["reason"] = "content_words_lost"
                            meta["preserved_ratio"] = round(preserved_ratio, 3)
                            return False, meta
            except Exception:
                _WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)
                orig_words = set(_WORD_RE.findall(original.lower()))
                var_words = set(_WORD_RE.findall(variation.lower()))
                if orig_words:
                    overlap = len(orig_words & var_words) / len(orig_words)
                    meta["content_word_preservation_fallback"] = round(overlap, 3)
                    if overlap < 0.50:
                        meta["reason"] = "content_words_lost"
                        meta["preserved_ratio"] = round(overlap, 3)
                        return False, meta

        elif operator_id == "register_formal_informal":
            # ── C3: Register shift — formality marker detection ──
            # Verify the register actually changed (or was intended to).
            _marker_data = _load_yaml_data("formal_informal_markers.yaml")
            _FORMAL_MARKERS = {
                "en": set(_marker_data.get("formal", {}).get("en", [])),
                "ru": set(_marker_data.get("formal", {}).get("ru", [])),
            }
            _INFORMAL_MARKERS = {
                "en": set(_marker_data.get("informal", {}).get("en", [])),
                "ru": set(_marker_data.get("informal", {}).get("ru", [])),
            }
            lang = language if language in _FORMAL_MARKERS else "en"
            orig_lower = original.lower()
            var_lower = variation.lower()
            orig_words = set(re.findall(r"[a-zа-яё]+", orig_lower))
            var_words = set(re.findall(r"[a-zа-яё]+", var_lower))
            formal_markers = _FORMAL_MARKERS[lang]
            informal_markers = _INFORMAL_MARKERS[lang]
            orig_formal = len(orig_words & formal_markers)
            orig_informal = len(orig_words & informal_markers)
            var_formal = len(var_words & formal_markers)
            var_informal = len(var_words & informal_markers)
            orig_register = (
                "formal"
                if orig_formal > orig_informal
                else ("informal" if orig_informal > orig_formal else "neutral")
            )
            var_register = (
                "formal"
                if var_formal > var_informal
                else ("informal" if var_informal > var_formal else "neutral")
            )
            meta["orig_register"] = orig_register
            meta["var_register"] = var_register
            meta["register_changed"] = orig_register != var_register
            # FLAG (not REJECT) if register didn't change — operator may have
            # made lexical changes that don't surface in simple marker counts
            if orig_register != "neutral" and var_register == orig_register:
                meta["register_shift_flag"] = True

        elif operator_id == "length_variation":
            # ── C4: Length variation — direction-aware length check ──
            # Verify the variation achieved the intended length change.
            # operator_metadata may contain {"direction": "lengthen"/"shorten"}
            direction = None
            if operator_metadata:
                direction = operator_metadata.get("direction")
            orig_len = len(original.split())
            var_len = len(variation.split())
            ratio = var_len / max(orig_len, 1)
            if direction == "lengthen":
                meta["length_direction"] = "lengthen"
                meta["length_ratio"] = round(ratio, 3)
                if ratio < 1.10:
                    meta["reason"] = "lengthen_insufficient"
                    meta["ratio"] = round(ratio, 3)
                    return False, meta
            elif direction == "shorten":
                meta["length_direction"] = "shorten"
                meta["length_ratio"] = round(ratio, 3)
                if ratio > 0.90:
                    meta["reason"] = "shorten_insufficient"
                    meta["ratio"] = round(ratio, 3)
                    return False, meta
            else:
                # No direction metadata — accept (generic checks already applied)
                meta["length_ratio"] = round(ratio, 3)

        return True, meta

    # ── Layer 2: Bidirectional NLI Ensemble ────────────────────────

    async def _check_layer2(
        self,
        original: str,
        variation: str,
        operator_id: str,
        language: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        params = self._get_layer1_params(operator_id)
        nli_min = params["nli_min"]
        direction = settings.OPERATOR_NLI_DIRECTION.get(operator_id, "bidirectional")
        meta["nli_direction"] = direction
        pipes = self._get_nli_pipelines(language)

        if not pipes:
            # Fail-closed: no NLI models loaded → REJECT for B/C
            meta["reason"] = "nli_models_unavailable"
            meta["nli_mode"] = "no_models"
            meta["diagnostic"] = (
                "NLI models not loaded. Run `python scripts/download_validator_models.py` "
                "to install required transformer models (cross-encoder/nli-deberta-v3-large, "
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7). "
                "Until models are available, ALL Tier B/C variations will be REJECTED "
                "by Layer 2 (fail-closed). This is expected in CI/test environments "
                "without transformers/torch."
            )
            return False, meta

        orig_trunc = self._truncate_for_nli(original, settings.MAX_NLI_TOKENS)
        var_trunc = self._truncate_for_nli(variation, settings.MAX_NLI_TOKENS)

        scores = []  # list of min_scores per model
        raw = []
        for i, pipe in enumerate(pipes):
            model_name = (
                self.en_nli_models[i] if language != "ru" else self.ru_nli_models[i]
            )
            try:
                fwd = pipe(
                    {"text": orig_trunc, "text_pair": var_trunc},
                    top_k=None,
                    truncation=True,
                )
                bwd = pipe(
                    {"text": var_trunc, "text_pair": orig_trunc},
                    top_k=None,
                    truncation=True,
                )
                fwd_score = self._extract_entailment_score(fwd, model_name)
                bwd_score = self._extract_entailment_score(bwd, model_name)
                min_score = min(fwd_score, bwd_score)
                scores.append(min_score)
                entry = {
                    "model": model_name,
                    "forward": fwd_score,
                    "backward": bwd_score,
                    "min": min_score,
                }
                # For the directional (backward_only) gate we also need the
                # FORWARD argmax verdict — the ERAP oracle rejects on forward
                # contradiction (argmax form; no numeric threshold).
                if direction == "backward_only":
                    entry["forward_label"] = self._extract_verdict_label(
                        fwd, model_name
                    )
                raw.append(entry)
            except Exception as e:
                logger.error(f"NLI inference failed for {model_name}: {e}")
                meta["reason"] = "nli_inference_error"
                meta["error"] = str(e)
                return False, meta

        meta["nli_raw"] = raw
        meta["nli_min_scores"] = scores

        # min(forward_NLI_m1, backward_NLI_m1, forward_NLI_m2, backward_NLI_m2)
        all_raw_scores = []
        for r in raw:
            all_raw_scores.append(r["forward"])
            all_raw_scores.append(r["backward"])
        meta["MIS"] = min(all_raw_scores) if all_raw_scores else None

        if len(scores) < 2:
            meta["reason"] = "insufficient_nli_models"
            return False, meta

        meta["nli_min_required"] = nli_min

        if direction == "backward_only":
            backward_scores = [r["backward"] for r in raw]
            if not all(s >= nli_min for s in backward_scores):
                meta["reason"] = "nli_backward_below_threshold"
                return False, meta
            if any(r.get("forward_label") == "contradiction" for r in raw):
                meta["reason"] = "nli_forward_contradiction"
                return False, meta
            gate_scores = backward_scores
        else:
            # Bidirectional (default): strict-paraphrase criterion — both models
            # must pass on min(forward, backward).
            if not all(s >= nli_min for s in scores):
                meta["reason"] = "nli_below_threshold"
                return False, meta
            gate_scores = scores

        # Disagreement diagnostic (over the direction-appropriate scores)
        disagreement = max(gate_scores) - min(gate_scores)
        meta["disagreement"] = disagreement
        meta["disagreement_flag"] = disagreement > self.disagreement_threshold

        meta["nli_mode"] = "ensemble_active"
        return True, meta

    # ── Layer 3: Task-aware answer preservation ────────────────────

    def _check_layer3(
        self,
        original: str,
        variation: str,
        operator_id: str,
        target: Optional[str],
        task_type: str,
        language: str = "en",
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        ttype = TaskType.UNKNOWN
        for candidate in TaskType:
            if candidate.value == task_type.lower():
                ttype = candidate
                break

        # ── (1) Task-aware answer preservation (only when target provided) ──
        if target is not None:
            target_clean = target.strip().lower()

            if ttype == TaskType.MCQ:
                # Option-set equality heuristic (basic)
                orig_options = set(re.findall(r"[A-D]\)\s*[^A-D)]+", original))
                var_options = set(re.findall(r"[A-D]\)\s*[^A-D)]+", variation))
                if orig_options and var_options and orig_options != var_options:
                    meta["reason"] = "mcq_option_set_changed"
                    return False, meta

            elif ttype == TaskType.OPEN_QA:
                if target_clean in ("yes", "no", "true", "false"):
                    orig_neg = len(
                        re.findall(
                            r"\b(not|no|never|n't|не|нет|никогда)\b", original.lower()
                        )
                    )
                    var_neg = len(
                        re.findall(
                            r"\b(not|no|never|n't|не|нет|никогда)\b", variation.lower()
                        )
                    )
                    if (orig_neg % 2) != (var_neg % 2) and "negation" not in operator_id:
                        meta["reason"] = "negation_polarity_shift"
                        return False, meta
                else:
                    l3_result = self._check_openqa_backend(
                        original, variation, target, language, meta
                    )
                    if l3_result is not None:
                        if not l3_result[0]:
                            return l3_result
                        # backend passed; meta already updated with scores
                    else:
                        meta["l3_openqa_fallback_flag"] = True
                        meta["layer3_status"] = (
                            f"{language}_openqa_free_text_fallback; "
                            "backend unavailable"
                        )

            elif ttype == TaskType.CLASSIFICATION:
                orig_label_words = set(original.lower().split())
                var_label_words = set(variation.lower().split())
                if (
                    target_clean in orig_label_words
                    and target_clean not in var_label_words
                ):
                    meta["reason"] = "classification_label_lost"
                    return False, meta
                if (
                    target_clean not in orig_label_words
                    and target_clean in var_label_words
                ):
                    meta["reason"] = "classification_label_added"
                    return False, meta

  
        if operator_id == "tone_shift":
            lang = self._detect_dominant_lang(original)
            orig_class = self._get_sentiment_polarity(original, lang)
            var_class = self._get_sentiment_polarity(variation, lang)
            if orig_class is not None and var_class is not None:
                class_shift = abs(var_class - orig_class)
                meta["tone_polarity_orig_class"] = orig_class
                meta["tone_polarity_var_class"] = var_class
                meta["tone_polarity_class_shift"] = class_shift
                meta["tone_polarity_method"] = (
                    "classifier"
                    if model_cache.get_sentiment_classifier(lang) is not None
                    else "keyword_heuristic"
                )
                if class_shift > 1:
                    meta["reason"] = "tone_polarity_class_reversal"
                    return False, meta
                # Normalised L3 score from class_shift: 1.0 (no shift) → 0.5 (max allowed shift)
                meta["L3_score"] = max(0.0, 1.0 - class_shift / 2.0)

        # C6 operator-specific Layer 3 polarity check
        if operator_id == "negation_scope_preserving_rephrasing":
            orig_n = set(
                re.findall(
                    r"\b(not|n't|never|no|nothing|nobody|neither|nor|не|нет|никогда|ничего|никто|нигде)\b",
                    original.lower(),
                )
            )
            var_n = set(
                re.findall(
                    r"\b(not|n't|never|no|nothing|nobody|neither|nor|не|нет|никогда|ничего|никто|нигде)\b",
                    variation.lower(),
                )
            )
            if (len(orig_n) > 0) != (len(var_n) > 0):
                meta["reason"] = "c6_polarity_flip"
                return False, meta

        meta.setdefault("L3_score", None)
        # Preserve an already-set status (e.g. the Open-QA backend-unavailable
        # fallback) rather than clobbering it with "passed".
        meta.setdefault("layer3_status", "passed")
        return True, meta

    # ── Open-QA backend dispatch (Layer 3) ──────────────────────────

    @staticmethod
    def _check_openqa_backend(
        original: str,
        variation: str,
        target: str,
        language: str,
        meta: Dict[str, Any],
    ) -> Optional[Tuple[bool, Dict[str, Any]]]:
        try:
            from src.core.services.open_qa_equivalence import (
                OpenQABackendUnavailable,
                OpenQAEquivalence,
            )
        except ImportError:
            return None

        if language == "ru":
            try:
                scores = OpenQAEquivalence.score_ruRoberta(
                    contexts=[original, variation],
                    claims=[target, target],
                )
                if scores and len(scores) == 2:
                    meta["ruroberta_scores"] = list(scores)
                    meta["L3_score"] = min(scores)
                    meta["layer3_backend"] = "ruroberta"
                    if min(scores) < OpenQAEquivalence.THETA_RUROBERTA:
                        meta["reason"] = "ruroberta_answer_preservation_failed"
                        return False, meta
            except OpenQABackendUnavailable:
                return None
        else:
            try:
                scores = OpenQAEquivalence.score_minicheck(
                    contexts=[original, variation],
                    claims=[target, target],
                )
                if scores and len(scores) == 2:
                    meta["minicheck_scores"] = list(scores)
                    meta["L3_score"] = min(scores)
                    meta["layer3_backend"] = "minicheck"
                    if min(scores) < OpenQAEquivalence.THETA_MINICHECK:
                        meta["reason"] = "minicheck_answer_preservation_failed"
                        return False, meta
            except OpenQABackendUnavailable:
                return None

        return True, meta

    # ── Audit queue ────────────────────────────────────────────────

    def drain_audit_queue(self) -> List[Dict[str, Any]]:
        items = list(self.audit_queue)
        self.audit_queue.clear()
        return items
