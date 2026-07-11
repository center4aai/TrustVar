# src/core/services/eval_service.py
import csv
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, precision_score, recall_score

from src.core.schemas.task import TaskResult, VariationStrategy
from src.core.taxonomy import (  # noqa: F401  (re-exported for back-compat)
    CANONICAL_TASK_TYPES,
    _TASK_TYPE_ALIASES,
    normalize_task_type,
)
from src.core.services.include_exclude_evaluator import IncludeExcludeEvaluator
from src.core.services.open_qa_equivalence import (
    OpenQABackendUnavailable,
    OpenQAEquivalence,
)
from src.utils.logger import logger

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    _HAS_STATSMODELS = True
except ImportError:
    MixedLM = None  # type: ignore
    ConvergenceWarning = Warning  # type: ignore
    _HAS_STATSMODELS = False

try:
    from pymer4.models import Lmer as _Lmer  # type: ignore
    _HAS_PYMER4 = True
except Exception:
    _Lmer = None
    _HAS_PYMER4 = False


# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_BOOTSTRAP_SEED = 42

CV_UNRELIABLE_MU_THRESHOLD = 0.1
VC_BOUNDARY_EPS = 1e-6


# ── Tier mapping ──────────────────────────────────────────────────────────

_TIER_A_STRATEGIES = {
    VariationStrategy.FORMAT_NORMALIZATION,
    VariationStrategy.ORTHOGRAPHIC_NORMALIZATION_RU,
    VariationStrategy.MCQ_OPTION_PERMUTATION,
    VariationStrategy.LIST_REORDERING,
    VariationStrategy.TYPED_PARAMETRIC_SUBSTITUTION,
}

_TIER_B_STRATEGIES = {
    VariationStrategy.ACTIVE_PASSIVE_VOICE,
    VariationStrategy.MONOSEMIC_SYNONYM_SUBSTITUTION,
    VariationStrategy.NOMINALISATION,
    VariationStrategy.CONTROLLED_SYNTACTIC_TRANSFORMATIONS,
    VariationStrategy.SENTENCE_SPLIT_MERGE,
    VariationStrategy.CONTROLLED_DESCRIPTIVE_MODIFIER_INSERTION,
}

_TIER_C_STRATEGIES = {
    VariationStrategy.PARAPHRASE_LEXICO_SYNTACTIC_CONSTRAINED,
    VariationStrategy.PARAPHRASE_FREE,
    VariationStrategy.LENGTH_VARIATION,
    VariationStrategy.REGISTER_FORMAL_INFORMAL,
    VariationStrategy.TONE_SHIFT,
    VariationStrategy.NEGATION_SCOPE_PRESERVING_REPHRASING,
    VariationStrategy.WSD_SYNONYM_SUBSTITUTION,
    VariationStrategy.BACK_TRANSLATION_SINGLE_PIVOT,
}

_TIER_MAP: Dict[VariationStrategy, str] = {}
for _s in _TIER_A_STRATEGIES:
    _TIER_MAP[_s] = "A"
for _s in _TIER_B_STRATEGIES:
    _TIER_MAP[_s] = "B"
for _s in _TIER_C_STRATEGIES:
    _TIER_MAP[_s] = "C"


def get_tier(variation_type: Optional[str]) -> Optional[str]:
    """Map a variation_type string to its tier (A/B/C). Returns None for non-variation results."""
    if variation_type is None:
        return None
    try:
        strategy = VariationStrategy(variation_type)
        return _TIER_MAP.get(strategy)
    except ValueError:
        return None


# ── Data structures for grouped results ───────────────────────────────────


@dataclass
class VariantScores:
    """Scores for a single variant across models."""

    task_id: str
    tier: str
    variant_type: str
    variant_index: int
    scores: Dict[str, float] = field(default_factory=dict)  # model_id -> score
    outputs: Dict[str, str] = field(default_factory=dict)  # model_id -> output
    targets: Dict[str, str] = field(default_factory=dict)  # model_id -> target
    question: str = ""
    operator_metadata: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "classification"
    language: str = "en"
    option_labels: List[str] = field(default_factory=list)  # MCQ label set (positional)
    options: List[str] = field(default_factory=list)  # MCQ option texts (original order)

# ── Answer label extraction (for MCQ / multi-label classification) ──────

_COMMON_SUFFIXES: Tuple[str, ...] = (
    "ment", "tion", "sion", "ing", "ed", "ly", "ness", "ity", "ive", "al",
    "ic", "ism", "ist", "able", "ible", "ful", "less", "ous", "er", "est",
    "ize", "ise", "ify", "en", "ate", "ion", "or",
)


def _stem_word(word: str) -> str:
    r"""Strip a common suffix from *word* so that ``\bstem\w*\b`` catches
    morphological variants (``entailed`` ← ``entailment``, ``contradicts`` ←
    ``contradiction``).  Applies only to all-alpha labels ≥ 3 characters;
    single-letter or numeric labels are returned unchanged."""
    w = word.strip().lower()
    for suffix in sorted(_COMMON_SUFFIXES, key=len, reverse=True):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _extract_answer_labels(
    output: str,
    option_labels: List[str],
    task_type: str,
    options: Optional[List[str]] = None,
    multi_label: bool = False,
    allow_stem_fallback: bool = False,
) -> str:
    """Extract answer labels from a model's free-form output.

    For single-answer tasks (mcq, and classification when ``multi_label`` is
    ``False``), returns the *last* isolated occurrence of any valid label
    (e.g. ``"C"`` from ``"...Answer: C) 24"``).

    For multi-label tasks (``multi_label=True``), collects *all* distinct
    valid labels found, sorts them, and joins them into a concatenated string
    (e.g. ``"1,2,6"`` → ``"126"``).

    If no label tokens are found, falls back to searching for option **values**
    (e.g. ``"24"`` in ``"…\\boxed{24}"``) and maps them back to the corresponding
    label via ``options`` / ``option_labels``.

    If *still* nothing is found and ``allow_stem_fallback`` is ``True``, a
    second pass matches morphological variants of multi-letter word labels by
    stripping common suffixes and matching ``\\b{stem}\\w*\\b`` — e.g.
    ``"entailed"`` is recognised for label ``"entailment"``.

    Labels and values are matched as whole tokens (word-boundary delimited)."""

    if not option_labels or not output:
        return output.strip()

    escaped = sorted(
        [re.escape(str(lbl)) for lbl in option_labels],
        key=len,
        reverse=True,
    )
    pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
    found = re.findall(pattern, output, re.IGNORECASE)

    if not found and options and len(options) == len(option_labels):
        escaped_vals = sorted(
            [re.escape(str(v)) for v in options],
            key=len,
            reverse=True,
        )
        val_pattern = r'\b(?:' + '|'.join(escaped_vals) + r')\b'
        found_vals = re.findall(val_pattern, output)
        if found_vals:
            found = []
            for val in found_vals:
                try:
                    idx = next(
                        i for i, o in enumerate(options)
                        if str(o).strip() == val.strip()
                    )
                    found.append(option_labels[idx])
                except StopIteration:
                    pass

    if not found and allow_stem_fallback and output:
        all_alpha = all(
            str(lbl).isalpha() and len(str(lbl)) >= 3 for lbl in option_labels
        )
        if all_alpha:
            stem_to_label: Dict[str, str] = {}
            for lbl in option_labels:
                stem_to_label.setdefault(_stem_word(str(lbl)).lower(), str(lbl))
            stems = sorted(
                {re.escape(s) for s in stem_to_label},
                key=len,
                reverse=True,
            )
            stem_pattern = r'\b(?:' + '|'.join(stems) + r')\w*\b'
            stem_found = re.findall(stem_pattern, output, re.IGNORECASE)
            if stem_found:
                mapped: List[str] = []
                for w in stem_found:
                    w_lower = w.lower()
                    for stem, label in sorted(
                        stem_to_label.items(), key=lambda kv: -len(kv[0])
                    ):
                        if w_lower.startswith(stem):
                            mapped.append(label)
                            break
                    else:
                        mapped.append(w)
                # F1: the multi-label join applies to mcq too — gate on
                # multi_label only. Single-answer (multi_label=False) keeps the
                # last-isolated-label semantics for both mcq and classification.
                if not multi_label:
                    found = [mapped[-1]]
                else:
                    found = list(dict.fromkeys(mapped))

    if not found:
        return output.strip()

    if not multi_label:
        return found[-1].strip()

    return ''.join(sorted(set(f.strip() for f in found)))


def _get_option_labels_for_result(r: TaskResult) -> Optional[List[str]]:
    """Derive option_labels from result metadata, falling back to ``classes`` dict."""
    meta = r.metadata or {}
    labels = meta.get("option_labels")
    if labels:
        return labels
    classes = meta.get("classes")
    if isinstance(classes, dict) and classes:
        return sorted(classes.values(), key=lambda v: str(v))
    return None


def _get_options_for_result(r: TaskResult) -> Optional[List[str]]:
    """Derive options from result metadata, falling back to ``classes`` keys."""
    meta = r.metadata or {}
    opts = meta.get("options")
    if opts:
        return opts
    classes = meta.get("classes")
    if isinstance(classes, dict) and classes:
        return sorted(classes.keys(), key=lambda k: str(k))
    return None


def _normalize_target_with_classes(r: TaskResult) -> str:
    """Map a class-index target (e.g. ``"1"``) to its label via ``metadata.classes``."""
    target = str(r.target).strip()
    classes = (r.metadata or {}).get("classes")
    if isinstance(classes, dict):
        for k, v in classes.items():
            if str(k).strip() == target:
                return str(v).strip()
    return target


def _is_multi_label(r: TaskResult) -> bool:
    """Check whether this result belongs to a multi-label classification task."""
    meta = r.metadata or {}
    semantics = meta.get("task_semantics")
    if isinstance(semantics, str) and "multi_label" in semantics.lower():
        return True
    return False


# ── EvaluationService ─────────────────────────────────────────────────────


class EvaluationService:
    """Service for evaluating results (basic + TrustVar metrics)"""

    # ── Basic metrics (preserved) ─────────────────────────────────────────

    def evaluate_results(
        self, results: List[TaskResult], metrics: List[str]
    ) -> Dict[str, float]:
        """Evaluate results by basic metrics (backward compatibility)."""
        aggregated: Dict[str, float] = {}

        for metric in metrics:
            if metric == "bleu":
                aggregated[metric] = self._bleu_score(results)
            elif metric == "rouge":
                aggregated[metric] = self._rouge_score(results)
            elif metric == "accuracy":
                aggregated[metric] = self._accuracy(results)
            elif metric == "f1_score":
                aggregated[metric] = self._f1_score(results)
            elif metric == "precision":
                aggregated[metric] = self._precision(results)
            elif metric == "recall":
                aggregated[metric] = self._recall(results)
            elif metric == "cv":
                scores = [float(r.metadata.get("score", 0)) for r in results]
                aggregated[metric] = self._calculate_cv(scores)
            elif metric == "rta":
                aggregated[metric] = self._rta_score(results)
            elif metric == "iqr_cv":
                scores = [float(r.metadata.get("score", 0)) for r in results]
                aggregated[metric] = self._calculate_iqr_cv(scores)
            elif metric == "jsd":
                scores = [float(r.metadata.get("score", 0)) for r in results]
                aggregated[metric] = self._calculate_jsd_divergence(scores)
            elif metric == "include_exclude":
                ie_results = IncludeExcludeEvaluator.evaluate_results(results)
                aggregated["include_exclude_score"] = ie_results[
                    "include_exclude_score"
                ]
                aggregated["include_success_rate"] = ie_results[
                    "include_success_rate"
                ]
                aggregated["exclude_violation_rate"] = ie_results[
                    "exclude_violation_rate"
                ]

        return aggregated

    # ── TrustVar metrics ──────────────────────────────────────────────────

    def compute_trustvar_metrics(
        self,
        results: List[TaskResult],
        task_type: Optional[str] = None,
        language: str = "en",
        n_resamples: int = 1000,
        ci_level: float = 0.95,
        n_models: Optional[int] = None,
        seed: int = DEFAULT_BOOTSTRAP_SEED,
    ) -> Dict[str, Any]:
        """
        Compute the full TrustVar metrics suite:
        - Stratified TSI per task per tier
        - Model-centric CV*_j
        - EAR per task per tier (task-type-specific equivalence)
        - Bootstrap BCa CI for TSI and EAR
        - Variance decomposition (mixed-effects)
        - Aggregate TSI_τ(D) = mean ± std

        Args:
            results: All TaskResult objects across all models
            task_type: Task type (mcq, classification, open_qa, generation).
                       If None, inferred from metadata.
            language: Language code ('en' or 'ru') for Open-QA equivalence backend.
                      EN uses MiniCheck, RU uses AlignScore.
            n_resamples: Number of bootstrap replicates (default 1000)
            ci_level: Confidence level for BCa CI (default 0.95)
            n_models: Expected model pool size (for reporting)
            seed: RNG seed for the model-bootstrap (C7 — exposed for
                  seed-stability robustness checks; default 42).

        Returns:
            Dictionary with all computed metrics
        """
        if not results:
            return self._empty_trustvar_result()

        grouped = self._group_results(results)
        if not grouped:
            return self._empty_trustvar_result()

        all_models = sorted({r.model_id for r in results})
        if n_models is None:
            n_models = len(all_models)

        # ── Per-tier, per-task metrics ──
        per_task_tsi: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_task_ear: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_task_cv: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_task_iqr_cv: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_task_uninformative: Dict[str, Dict[str, bool]] = defaultdict(dict)
        per_task_ear_flags: Dict[str, Dict[str, str]] = defaultdict(dict)
        per_task_cv_unreliable: Dict[str, Dict[str, bool]] = defaultdict(dict)

        # ── Bootstrap replicates for aggregate CI ──
        boot_tsi_bench: Dict[str, List[float]] = {}
        boot_ear_bench: Dict[str, List[float]] = {}
        boot_tsi_per_task: Dict[str, Dict[str, List[float]]] = {}
        boot_ear_per_task: Dict[str, Dict[str, List[float]]] = {}

        for tier in ("A", "B", "C"):
            tier_groups = {
                k: v for k, v in grouped.items() if v.tier == tier
            }
            if not tier_groups:
                continue

            # Group tier_groups by task_id
            task_to_variants: Dict[str, List[VariantScores]] = defaultdict(list)
            for g in tier_groups.values():
                task_to_variants[g.task_id].append(g)

            # Per-task task_type map (normalized — C5), reused for EAR + bootstrap
            task_type_map: Dict[str, str] = {}
            for ttid, vlist in task_to_variants.items():
                task_type_map[ttid] = normalize_task_type(
                    task_type or self._infer_task_type(vlist[0])
                )

            # Per-task language map (C4): resolved from metadata, with global
            # fallback for tasks that don't carry language in their metadata.
            language_map: Dict[str, str] = {}
            for ttid, vlist in task_to_variants.items():
                lang = self._infer_language(vlist[0], language)
                language_map[ttid] = lang

            # Per-task TSI and EAR
            for task_id, variant_list in task_to_variants.items():
                # Need at least 2 variants and 2 models for TSI
                all_models_in_task = sorted(
                    {m for g in variant_list for m in g.scores.keys()}
                )
                if len(all_models_in_task) < 2 or len(variant_list) < 2:
                    continue

                # TSI_τ(t) = mean_j CV*_τ^{j,t}
                tsi_val = self._compute_tsi_per_task(
                    task_id, tier, grouped
                )
                if np.isnan(tsi_val):
                    per_task_uninformative[task_id][tier] = True
                else:
                    per_task_tsi[task_id][tier] = tsi_val

    
                cv_val = self._compute_avg_cv_per_tier_task(
                    task_id, tier, grouped
                )
                per_task_cv[task_id][tier] = cv_val

                iqr_cv_val = self._compute_avg_iqr_cv_per_tier_task(
                    task_id, tier, grouped
                )
                per_task_iqr_cv[task_id][tier] = iqr_cv_val

                cell_scores = [
                    s for g in variant_list for s in g.scores.values()
                    if not np.isnan(s)
                ]
                if cell_scores and float(np.mean(cell_scores)) < CV_UNRELIABLE_MU_THRESHOLD:
                    per_task_cv_unreliable[task_id][tier] = True

                # EAR (+ A1/B3/B4 diagnostic flag)
                ear_val, ear_flag, _ = self._build_ear_cell(
                    variant_list, task_type_map[task_id], language_map.get(task_id, language)
                )
                per_task_ear[task_id][tier] = ear_val
                if ear_flag:
                    per_task_ear_flags[task_id][tier] = ear_flag

            # Bootstrap CI for TSI and EAR (shared G_b, precomputed tensor)
            pt_tsi, pt_ear, bench_tsi, bench_ear, dropped_frac = self._bootstrap_ci_per_tier(
                grouped, tier, n_resamples, ci_level, all_models,
                task_type_map, language_map, seed
            )
            boot_tsi_bench[tier] = bench_tsi
            boot_ear_bench[tier] = bench_ear
            boot_tsi_per_task[tier] = pt_tsi
            boot_ear_per_task[tier] = pt_ear

            # Mark uninformative if >5% dropped
            for task_id in per_task_tsi:
                if dropped_frac.get(task_id, 0) > 0.05:
                    per_task_uninformative[task_id][tier] = True

        # ── Model-centric dispersion CV*_j ──
        cv_star = self._compute_model_reliability_profile(grouped)

        # ── Aggregate TSI_τ(D) = mean ± std ──

        aggregate_tsi = self._compute_aggregate_tsi(per_task_tsi, boot_tsi_bench, ci_level)
        aggregate_ear = self._compute_aggregate_ear(per_task_ear, boot_ear_bench, ci_level)

        # ── Variance decomposition ──
        var_decomp = self._compute_variance_decomposition(grouped, task_type)

        # ── Tier comparison (Kruskal-Wallis) ──
        tier_comparison = self._compute_tier_comparison(per_task_tsi)

        return {
            "per_task_tsi": dict(per_task_tsi),
            "per_task_ear": dict(per_task_ear),
            "per_task_cv": dict(per_task_cv),
            "per_task_iqr_cv": dict(per_task_iqr_cv),
            "per_task_uninformative": dict(per_task_uninformative),
            "per_task_ear_flags": dict(per_task_ear_flags),
            "per_task_cv_unreliable": dict(per_task_cv_unreliable),
            "model_cv_star": cv_star,
            "aggregate_tsi": aggregate_tsi,
            "aggregate_ear": aggregate_ear,
            "variance_decomposition": var_decomp,
            "tier_comparison": tier_comparison,
            "bootstrap_replicates": {
                "tsi": {
                    "benchmark": {t: v for t, v in boot_tsi_bench.items()},
                    "per_task": {t: v for t, v in boot_tsi_per_task.items()},
                },
                "ear": {
                    "benchmark": {t: v for t, v in boot_ear_bench.items()},
                    "per_task": {t: v for t, v in boot_ear_per_task.items()},
                },
            },
            "n_models": n_models,
            "n_resamples": n_resamples,
            "ci_level": ci_level,
        }

    # ── Grouping ──────────────────────────────────────────────────────────

    def _group_results(
        self, results: List[TaskResult]
    ) -> Dict[str, VariantScores]:
        """Group results by (task_id, tier, variant_type, variant_prompt).

        Different variants of the same type are distinguished by their
        input prompt (the variant text).
        """
        groups: Dict[str, VariantScores] = {}

        for r in results:
            task_id = r.original_input or r.input
            tier = get_tier(r.variation_type)
            variant_type = r.variation_type or "original"
            variant_prompt = r.input
            key = f"{task_id}||{tier or 'none'}||{variant_type}||{variant_prompt}"

            if key not in groups:
                op_meta = r.metadata.get("operator_metadata", {}) if r.metadata else {}
                rt_task_type = r.metadata.get("task_type", "classification") if r.metadata else "classification"
                rt_language = r.metadata.get("language", "en") if r.metadata else "en"
                groups[key] = VariantScores(
                    task_id=task_id,
                    tier=tier or "original",
                    variant_type=variant_type,
                    variant_index=0,
                    question=r.input,
                    operator_metadata=op_meta,
                    task_type=rt_task_type,
                    language=rt_language,
                    option_labels=_get_option_labels_for_result(r) or [],
                    options=_get_options_for_result(r) or [],
                )
            elif r.metadata and "operator_metadata" in r.metadata:
                groups[key].operator_metadata = r.metadata["operator_metadata"]

            g = groups[key]
            g.scores[r.model_id] = self._extract_score(r)
            g.outputs[r.model_id] = r.output
            if r.target:
                g.targets[r.model_id] = r.target

        # Assign variant indices within each (task, tier)
        variant_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        for g in groups.values():
            if g.tier != "original":
                idx_key = (g.task_id, g.tier)
                g.variant_index = variant_counter[idx_key]
                variant_counter[idx_key] += 1

        return groups

    def _extract_score(self, r: TaskResult) -> float:
        if r.judge_score is not None:
            return r.judge_score
        if r.include_score is not None:
            return r.include_score
        if r.target is not None and r.output is not None:
            tt = normalize_task_type(
                (r.metadata or {}).get("task_type") if r.metadata else None
            )
            if tt in ("mcq", "classification"):
                target = _normalize_target_with_classes(r)
                output = r.output.strip() if r.output else ""
                option_labels = _get_option_labels_for_result(r)
                if option_labels:
                    extracted = _extract_answer_labels(
                        output, option_labels, tt,
                        options=_get_options_for_result(r),
                        multi_label=_is_multi_label(r),
                        allow_stem_fallback=(tt == "classification"),
                    )
                    return 1.0 if target.lower() == extracted.lower() else 0.0
                return 1.0 if target.lower() == output.lower() else 0.0
            # open_qa / generation: exact-match is not a valid score → uninformative.
            return float("nan")
        return float(r.metadata.get("score", 0.0))

    # ── TSI computation ───────────────────────────────────────────────────

    def _compute_tsi_per_task(
        self, task_id: str, tier: str, grouped: Dict[str, VariantScores]
    ) -> float:
        """
        TSI_τ(t) = (1/N) × Σ_j CV*_τ^{j,t}

        CV*_τ^{j,t} = std(m_{τ,1}^{j,t}, …, m_{τ,k_τ}^{j,t}) /
                       mean(m_{τ,1}^{j,t}, …, m_{τ,k_τ}^{j,t})
        """
        # Collect all models and their scores across variants for this (task, tier)
        variant_groups = [
            g
            for g in grouped.values()
            if g.task_id == task_id and g.tier == tier
        ]

        if len(variant_groups) < 2:
            return np.nan

        all_models = sorted(
            {m for g in variant_groups for m in g.scores.keys()}
        )
        if len(all_models) < 2:
            return np.nan

        cv_star_values = []
        for model_id in all_models:
            model_scores = [
                g.scores.get(model_id, np.nan) for g in variant_groups
            ]
            model_scores = [s for s in model_scores if not np.isnan(s)]
            if len(model_scores) < 2:
                continue
            cv_star = self._calculate_corrected_cv(model_scores)
            if not np.isnan(cv_star):
                cv_star_values.append(cv_star)

        if not cv_star_values:
            return np.nan

        return float(np.mean(cv_star_values))

    def _compute_avg_cv_per_tier_task(
        self, task_id: str, tier: str, grouped: Dict[str, VariantScores]
    ) -> float:
        """Average variation-centric CV across variants for a (task, tier) cell."""
        variant_groups = [
            g
            for g in grouped.values()
            if g.task_id == task_id and g.tier == tier
        ]
        cv_values = []
        for g in variant_groups:
            scores = list(g.scores.values())
            if len(scores) >= 2:
                cv = self._calculate_corrected_cv(scores)
                if not np.isnan(cv):
                    cv_values.append(cv)
        return float(np.mean(cv_values)) if cv_values else np.nan

    def _compute_avg_iqr_cv_per_tier_task(
        self, task_id: str, tier: str, grouped: Dict[str, VariantScores]
    ) -> float:
        """Average IQR-CV across variants for a (task, tier) cell."""
        variant_groups = [
            g
            for g in grouped.values()
            if g.task_id == task_id and g.tier == tier
        ]
        iqr_values = []
        for g in variant_groups:
            scores = list(g.scores.values())
            if len(scores) >= 2:
                iqr_cv = self._calculate_iqr_cv(scores)
                if not np.isnan(iqr_cv):
                    iqr_values.append(iqr_cv)
        return float(np.mean(iqr_values)) if iqr_values else np.nan

    # ── EAR computation ───────────────────────────────────────────────────

    def _build_ear_cell(
        self,
        variant_groups: List["VariantScores"],
        task_type: str,
        language: str = "en",
    ) -> Tuple[float, Optional[str], List[Dict[str, float]]]:
        """Build the EAR cell for one (task, tier): point estimate + diagnostics.

        EAR_τ(t) = average over variant pairs of P(equivalence) across models.
        For each pair (i, i'), the proportion of models where answer_i ≡ answer_i'.

        
        Returns:
            (point_value, flag, pairs)
            - point_value: EAR_τ(t), or NaN if generation / <2 variants /
              unavailable.
            - flag: diagnostic marker for the output —
                * "unavailable_backend"       
                * "unavailable_mcq_metadata"  
                * "provisional"               
                * None                         (fully determined)
            - pairs: per variant-pair {model_id: indicator∈{0,1}} for models
              present in both variants — reused by the bootstrap (C2) so the
              neural equivalence is computed exactly once.
        """
        tt = normalize_task_type(task_type)
        if tt == "generation" or len(variant_groups) < 2:
            return np.nan, None, []

        models = sorted({m for g in variant_groups for m in g.outputs.keys()})
        if not models:
            return np.nan, None, []

        pairs: List[Dict[str, float]] = []
        unavailable = False
        for i in range(len(variant_groups)):
            for j in range(i + 1, len(variant_groups)):
                g_i, g_j = variant_groups[i], variant_groups[j]
                indicators: Dict[str, float] = {}
                for model_id in models:
                    if model_id not in g_i.outputs or model_id not in g_j.outputs:
                        continue  # model absent in one variant → exclude, not unavailable
                    equiv = self._check_equivalence(
                        g_i.outputs[model_id],
                        g_j.outputs[model_id],
                        g_i.targets.get(model_id, ""),
                        g_j.targets.get(model_id, ""),
                        tt,
                        g_i.question,
                        language,
                        g_i.operator_metadata,
                        g_j.operator_metadata,
                        # Variants of one task share the original (positional)
                        # label set; the per-variant permutation lives in meta.
                        option_labels=g_i.option_labels or g_j.option_labels,
                    )
                    if equiv is None:
                        unavailable = True
                    else:
                        indicators[model_id] = 1.0 if equiv else 0.0
                pairs.append(indicators)

        if unavailable:
            flag = (
                "unavailable_backend"
                if tt == "open_qa"
                else "unavailable_mcq_metadata"
            )
            return np.nan, flag, pairs

        pair_means = [
            float(np.mean(list(d.values()))) for d in pairs if d
        ]
        point = float(np.mean(pair_means)) if pair_means else np.nan
        
        flag = "provisional" if tt == "open_qa" else None
        return point, flag, pairs

    def _check_equivalence(
        self,
        output_i: str,
        output_j: str,
        target_i: str,
        target_j: str,
        task_type: str,
        question: str = "",
        language: str = "en",
        meta_i: Optional[Dict[str, Any]] = None,
        meta_j: Optional[Dict[str, Any]] = None,
        option_labels: Optional[List[str]] = None,
    ) -> Optional[bool]:
        """Task-type-specific answer equivalence.

        Returns True/False, or **None** when equivalence cannot be determined
        (open-QA backend unavailable, or MCQ permutation metadata malformed) —
        the caller marks such cells ``unavailable`` rather than guessing.
        """
        tt = normalize_task_type(task_type)
        if tt == "mcq":
            return self._equiv_mcq(
                output_i, output_j, target_i, target_j, meta_i, meta_j, option_labels
            )
        elif tt == "classification":
            return self._equiv_classification(output_i, output_j)
        elif tt == "open_qa":
            return self._equiv_open_qa(output_i, output_j, question, language)
        else:

            return output_i.strip().lower() == output_j.strip().lower()

    @staticmethod
    def _letter_index(s: str) -> Optional[int]:
        """Map a clean option letter ('A'..'Z', optionally wrapped) to 0-based
        index, else None. 'A'→0, '(b)'→1, 'C.'→2."""
        t = s.strip().lower().strip("().").rstrip(".)")
        if len(t) == 1 and "a" <= t <= "z":
            return ord(t) - ord("a")
        return None

    @staticmethod
    def _canonical_option_index(
        position: int, meta: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Map a displayed option position to its canonical (original) content
        index using the variant's ``permutation`` (A1).

        Operator semantics (mcq_option_permutation): displayed position ``i``
        shows original content ``permutation[i]``. A non-reordering operator
        emits no permutation → the position IS the canonical index (identity).
        A present-but-malformed permutation → None (unrecoverable).
        """
        perm = meta.get("permutation") if meta else None
        if perm is None:
            return position  # no reordering declared → identity
        if not isinstance(perm, (list, tuple)) or position >= len(perm):
            return None  # malformed / out of range → unavailable
        try:
            return int(perm[position])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _answer_position(output: str, option_labels: Optional[List[str]]) -> Optional[int]:
        """Displayed option position the model chose, or None if not a clean label.

        Uses the shared label extractor over the variant's ``option_labels`` so
        both letter (``A``..) and numeric (``0``,``1``,..) schemes work and a
        label is recovered even from verbose output (e.g. ``"## Answer: 1 ..."``).
        Deliberately does NOT use option *text* matching: under permutation the
        displayed text order differs, so a text match would map to the wrong
        displayed position. Pure-text answers fall through to the caller's
        position-invariant comparison.
        """
        if option_labels:
            label = _extract_answer_labels(output, option_labels, "mcq")
            if label in option_labels:
                return option_labels.index(label)
        return EvaluationService._letter_index(output.strip().lower())

    @classmethod
    def _equiv_mcq(
        cls,
        output_i: str,
        output_j: str,
        target_i: str = "",
        target_j: str = "",
        meta_i: Optional[Dict[str, Any]] = None,
        meta_j: Optional[Dict[str, Any]] = None,
        option_labels: Optional[List[str]] = None,
    ) -> Optional[bool]:
        def _norm(s: str) -> str:
            return s.strip().lower()

        out_i, out_j = _norm(output_i), _norm(output_j)

        # Both outputs equal their (remapped) gold → both chose the gold content,
        # equivalent by construction (also the legacy metadata-free remap signal).
        if target_i and target_j and out_i == _norm(target_i) and out_j == _norm(target_j):
            return True

        # Precise path: recover the displayed label position (letter or digit,
        # incl. from verbose output) and map it through the variant's permutation
        # to a canonical content index.
        pos_i = cls._answer_position(output_i, option_labels)
        pos_j = cls._answer_position(output_j, option_labels)
        if pos_i is not None and pos_j is not None:
            canon_i = cls._canonical_option_index(pos_i, meta_i)
            canon_j = cls._canonical_option_index(pos_j, meta_j)
            if canon_i is None or canon_j is None:
                return None  # label-mode but permutation unrecoverable → unavailable
            return canon_i == canon_j

        # No clean displayed label on one side (e.g. full-text answers) →
        # order-independent direct content comparison.
        return out_i == out_j

    @staticmethod
    def _equiv_classification(output_i: str, output_j: str) -> bool:
        """Classification equivalence: exact label match (case-insensitive)."""
        return output_i.strip().lower() == output_j.strip().lower()

    @staticmethod
    def _equiv_open_qa(
        output_i: str, output_j: str, question: str, language: str = "en"
    ) -> Optional[bool]:
        if not output_i or not output_j:
            return False

        try:
            lang = language.lower()
            if lang in ("en", "eng", "english"):
                return bool(
                    OpenQAEquivalence.are_equivalent_en(question, output_i, output_j)
                )
            elif lang in ("ru", "rus", "russian"):
                return bool(
                    OpenQAEquivalence.are_equivalent_ru(question, output_i, output_j)
                )
            else:
                return bool(
                    OpenQAEquivalence.are_equivalent(
                        question, output_i, output_j, language
                    )
                )
        except OpenQABackendUnavailable:
            return None

    # ── Model reliability profile ─────────────────────────────────────────

    def _compute_model_reliability_profile(
        self, grouped: Dict[str, VariantScores]
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-model CV*_τ by tier: std(scores across variants) / mean(scores across variants).
        Returns {model_id: {tier: cv_star}}.
        """
        all_models = sorted({m for g in grouped.values() for m in g.scores.keys()})
        profile: Dict[str, Dict[str, float]] = {}

        for model_id in all_models:
            profile[model_id] = {}
            for tier in ("A", "B", "C"):
                variant_groups = [
                    g
                    for g in grouped.values()
                    if g.tier == tier and model_id in g.scores
                ]
                if len(variant_groups) < 2:
                    profile[model_id][tier] = np.nan
                    continue
                scores = [g.scores[model_id] for g in variant_groups]
                cv_star = self._calculate_corrected_cv(scores)
                profile[model_id][tier] = cv_star

        return profile

    # ── Bootstrap CI ──────────────────────────────────────────────────────

    def _bootstrap_ci_per_tier(
        self,
        grouped: Dict[str, VariantScores],
        tier: str,
        n_resamples: int,
        ci_level: float,
        all_models: List[str],
        task_type_map: Dict[str, str] = {},
        language_map: Dict[str, str] = {},
        seed: int = DEFAULT_BOOTSTRAP_SEED,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, List[float]],
        List[float],
        List[float],
        Dict[str, float],
    ]:
        rng = np.random.default_rng(seed)

        tier_tasks = sorted(
            {g.task_id for g in grouped.values() if g.tier == tier}
        )
        if not tier_tasks or not all_models:
            return {}, {}, [], [], {}

        # ── Precompute (once): per-task CV* per model + EAR indicator pairs ──
        cv_star_by_task: Dict[str, Dict[str, float]] = {}
        ear_pairs_by_task: Dict[str, List[Dict[str, float]]] = {}
        tsi_eligible: Dict[str, bool] = {}
        ear_eligible: Dict[str, bool] = {}

        for task_id in tier_tasks:
            variants = [
                g for g in grouped.values()
                if g.task_id == task_id and g.tier == tier
            ]
            models = sorted({m for g in variants for m in g.scores.keys()})

            cvs: Dict[str, float] = {}
            for model_id in models:
                scores = [
                    g.scores[model_id] for g in variants
                    if model_id in g.scores and not np.isnan(g.scores[model_id])
                ]
                if len(scores) >= 2:
                    cv = self._calculate_corrected_cv(scores)
                    if not np.isnan(cv):
                        cvs[model_id] = cv
            cv_star_by_task[task_id] = cvs
            tsi_eligible[task_id] = len(models) >= 2 and len(variants) >= 2

            tt = task_type_map.get(task_id, "classification")
            task_lang = language_map.get(task_id, "en")
            _, flag, pairs = self._build_ear_cell(variants, tt, task_lang)
            ear_pairs_by_task[task_id] = pairs
            ear_eligible[task_id] = (
                len(variants) >= 2
                and bool(pairs)
                and not (flag or "").startswith("unavailable")
                and normalize_task_type(tt) != "generation"
            )

        # ── Shared-G_b bootstrap (pure arithmetic) ──
        per_task_tsi: Dict[str, List[float]] = {t: [] for t in tier_tasks}
        per_task_ear: Dict[str, List[float]] = {t: [] for t in tier_tasks}
        bench_tsi_reps: List[float] = []
        bench_ear_reps: List[float] = []
        n_global = len(all_models)

        for _ in range(n_resamples):
            boot_idx = rng.choice(n_global, size=n_global, replace=True)
            boot_models = [all_models[i] for i in boot_idx]  # one G_b for all tasks

            rep_tsi_vals: List[float] = []
            rep_ear_vals: List[float] = []

            for task_id in tier_tasks:
                if tsi_eligible[task_id]:
                    cvs = cv_star_by_task[task_id]
                    vals = [cvs[m] for m in boot_models if m in cvs]
                    if vals:
                        tsi_val = float(np.mean(vals))
                        per_task_tsi[task_id].append(tsi_val)
                        rep_tsi_vals.append(tsi_val)

                if ear_eligible[task_id]:
                    pair_means = []
                    for indicators in ear_pairs_by_task[task_id]:
                        gathered = [
                            indicators[m] for m in boot_models if m in indicators
                        ]
                        if gathered:
                            pair_means.append(float(np.mean(gathered)))
                    if pair_means:
                        ear_val = float(np.mean(pair_means))
                        per_task_ear[task_id].append(ear_val)
                        rep_ear_vals.append(ear_val)

            if rep_tsi_vals:
                bench_tsi_reps.append(float(np.mean(rep_tsi_vals)))
            if rep_ear_vals:
                bench_ear_reps.append(float(np.mean(rep_ear_vals)))

        dropped_frac: Dict[str, float] = {}
        for task_id in tier_tasks:
            if tsi_eligible[task_id]:
                n_actual = len(per_task_tsi[task_id])
                dropped_frac[task_id] = 1.0 - (n_actual / n_resamples)
            else:
                dropped_frac[task_id] = 1.0

        return per_task_tsi, per_task_ear, bench_tsi_reps, bench_ear_reps, dropped_frac

    def _bca_ci(
        self, replicates: List[float], ci_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute BCa (Bias-Corrected and Accelerated) confidence interval."""
        if not replicates:
            return (np.nan, np.nan)

        arr = np.array(replicates, dtype=float)
        n = len(arr)
        alpha = (1 - ci_level) / 2

        # Degenerate bootstrap distribution (all replicates identical, e.g. a
        # task where every model scores 0/1 → CV*≡0): BCa bias-correction is
        # undefined (z0 = ±inf). Return the point as a zero-width interval
        # instead of crashing the whole metrics pipeline.
        if np.ptp(arr) == 0:
            return (float(arr[0]), float(arr[0]))

        if n < 10:
            # Fallback to percentile CI for small samples
            return (
                float(np.percentile(arr, alpha * 100)),
                float(np.percentile(arr, (1 - alpha) * 100)),
            )

        theta_hat = np.mean(arr)

        # Bias correction (guard the boundary: all reps on one side of the mean)
        prop_less = np.mean(arr < theta_hat)
        prop_less = min(max(prop_less, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
        z0 = stats.norm.ppf(prop_less)

        # Acceleration (jackknife)
        jackknife = np.array(
            [np.mean(np.delete(arr, i)) for i in range(n)]
        )
        theta_jack = np.mean(jackknife)
        num = np.sum((theta_jack - jackknife) ** 3)
        den = 6 * (np.sum((theta_jack - jackknife) ** 2)) ** 1.5
        a = num / den if den != 0 else 0.0

        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha)
        z_1alpha = stats.norm.ppf(1 - alpha)

        p_low = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_high = stats.norm.cdf(
            z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha))
        )

        # Final guard: if anything went non-finite, fall back to percentile CI.
        if not (np.isfinite(p_low) and np.isfinite(p_high)):
            return (
                float(np.percentile(arr, alpha * 100)),
                float(np.percentile(arr, (1 - alpha) * 100)),
            )

        return (
            float(np.percentile(arr, np.clip(p_low, 0.0, 1.0) * 100)),
            float(np.percentile(arr, np.clip(p_high, 0.0, 1.0) * 100)),
        )

    # ── Aggregate TSI_τ(D) ───────────────────────────────────────────────

    def _compute_aggregate_tsi(
        self,
        per_task_tsi: Dict[str, Dict[str, float]],
        bench_reps: Dict[str, List[float]],
        ci_level: float,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for tier in ("A", "B", "C"):
            values = [
                v
                for task_dict in per_task_tsi.values()
                if tier in task_dict
                for v in [task_dict[tier]]
                if not np.isnan(v)
            ]
            if not values:
                result[tier] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_tasks": 0,
                }
                continue

            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            # CI from benchmark-level bootstrap replicates (not per-task values)
            tier_reps = bench_reps.get(tier, [])
            if tier_reps:
                ci_low, ci_high = self._bca_ci(tier_reps, ci_level)
            else:
                ci_low, ci_high = np.nan, np.nan

            result[tier] = {
                "mean": mean_val,
                "std": std_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_tasks": len(values),
            }

        return result

    def _compute_aggregate_ear(
        self,
        per_task_ear: Dict[str, Dict[str, float]],
        bench_reps: Dict[str, List[float]],
        ci_level: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate EAR_τ(D) = mean_t EAR_τ(t) ± std.
        BCa CI on the mean from benchmark-level bootstrap replicates."""
        result: Dict[str, Dict[str, Any]] = {}
        for tier in ("A", "B", "C"):
            values = [
                v
                for task_dict in per_task_ear.values()
                if tier in task_dict
                for v in [task_dict[tier]]
                if not np.isnan(v)
            ]
            if not values:
                result[tier] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_tasks": 0,
                }
                continue

            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            tier_reps = bench_reps.get(tier, [])
            if tier_reps:
                ci_low, ci_high = self._bca_ci(tier_reps, ci_level)
            else:
                ci_low, ci_high = np.nan, np.nan

            result[tier] = {
                "mean": mean_val,
                "std": std_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_tasks": len(values),
            }

        return result

    # ── Variance decomposition ────────────────────────────────────────────

    def _fit_variance_model(
        self, df: "pd.DataFrame", label: str = "",
        fixed_fe: str = "C(task_type)",
    ) -> Dict[str, Any]:
        return self._fit_crossed_statsmodels(df, label, fixed_fe)

    # ── PRIMARY: statsmodels single-group crossed RE ──────────────────────

    def _fit_crossed_statsmodels(
        self, df: "pd.DataFrame", label: str = "",
        fixed_fe: str = "C(task_type)",
    ) -> Dict[str, Any]:
        """Fit the spec model with TRUE crossed random effects via the
        canonical statsmodels construction: a single constant group with
        every random effect supplied as a variance component.

            groups = const, re_formula = "0",
            vc_formula = {task, model, variant}

        Why this and not ``groups="task"``: with a single group the ``model``
        VC has one shared random effect per model across the whole dataset
        (genuinely crossed (1|model)); ``variant`` is nested-by-unique-label
        within task; ``task`` is its own component. The ``groups="task"`` form
        instead estimates model effects *within* each task group, so σ²_model
        absorbs the model×task interaction and σ²_residual is deflated —
        the bug this method replaces (B2). Ground-truth simulation confirms
        this construction recovers known components.

        Non-convergence / boundary-pinned components are surfaced as
        ``converged`` / ``boundary`` flags (fail-closed); component values are
        clamped to ≥0 for reporting only, never silently.
        """
        if not _HAS_STATSMODELS:
            return {"error": "statsmodels not installed", "engine": "statsmodels_crossed"}
        if df.empty or len(df) < 10:
            return {"error": f"Insufficient data for {label}", "engine": "statsmodels_crossed"}

        work = df.copy()
        work["_grp"] = 1
        # Keep only fixed-effect factors that actually vary (a single-level
        # C(col) gives a singular design); applies to both C(task_type) per-tier
        # and C(tier)+C(task_type) pooled.
        fe_terms = []
        for term in (t.strip() for t in fixed_fe.split("+") if t.strip()):
            col = term[2:-1] if term.startswith("C(") and term.endswith(")") else term
            if col in work.columns and work[col].nunique() > 1:
                fe_terms.append(term)
        fe_part = " + ".join(fe_terms)
        formula = f"score ~ {fe_part}" if fe_part else "score ~ 1"
        vc_formula = {
            "task": "0 + C(task)",
            "model": "0 + C(model)",
            "variant": "0 + C(variant)",
        }

        converged = True
        fit_warnings: List[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = MixedLM.from_formula(
                    formula,
                    groups="_grp",
                    re_formula="0",
                    vc_formula=vc_formula,
                    data=work,
                )
                result = model.fit(reml=True)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    converged = False
                    fit_warnings.append(str(w.message))
        except Exception as e:
            logger.error(f"Crossed variance model fit failed for {label}: {e}")
            return {"error": str(e), "label": label, "engine": "statsmodels_crossed"}

        raw = self._extract_crossed_components(result)
        # Boundary detection BEFORE clamping: any RE component pinned at/below
        # eps (or negative) signals a singular fit.
        boundary = self._boundary_pinned(raw)
        if not converged or boundary:
            logger.warning(
                f"Variance decomposition for {label}: "
                f"converged={converged}, boundary={boundary} — "
                f"components may be unreliable."
            )
        vc = {k: max(0.0, v) for k, v in raw.items()}  # clamp ≥0 for report only

        total_var = sum(vc.values())
        return {
            "variance_components": vc,
            "icc": {k: v / total_var for k, v in vc.items()} if total_var > 0 else {},
            "total_variance": total_var,
            "fixed_effects": {
                coef: float(val) for coef, val in result.fe_params.items()
            },
            "n_observations": int(result.nobs),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "engine": "statsmodels_crossed",
            "converged": converged,
            "boundary": boundary,
            "fit_warnings": fit_warnings,
        }

    @staticmethod
    def _boundary_pinned(raw_components: Dict[str, float]) -> bool:
        """True if any random-effect variance (task/model/variant) is pinned at
        or below ``VC_BOUNDARY_EPS`` (or negative) — a singular / boundary fit.
        Residual is excluded (it is the error term, never a boundary signal)."""
        return any(
            raw_components.get(k, 0.0) <= VC_BOUNDARY_EPS
            for k in ("task", "model", "variant")
        )

    @staticmethod
    def _extract_crossed_components(result: Any) -> Dict[str, float]:
        """Extract task/model/variant/residual variances from a single-group
        crossed MixedLMResults.

        With ``re_formula="0"`` there is no per-group random intercept, so all
        random-effect variances live in ``result.vcomp`` (absolute data units,
        ordered per ``result.model.exog_vc.names``) and the residual variance is
        ``result.scale``.

        NOTE: ``vcomp`` being in absolute units (not residual-scaled) is pinned
        by ``test_vcomp_units_are_absolute`` so a statsmodels upgrade cannot
        silently change the units underneath us.
        """
        vcomp = np.asarray(result.vcomp, dtype=float)
        names = list(getattr(result.model.exog_vc, "names", []))

        def _vc(name: str) -> float:
            return float(vcomp[names.index(name)]) if name in names else 0.0

        return {
            "task": _vc("task"),
            "model": _vc("model"),
            "variant": _vc("variant"),
            "residual": float(result.scale),
        }



    def _fit_variance_model_pymer4(
        self, df: "pd.DataFrame", label: str = "",
        fixed_fe: str = "C(task_type)",
    ) -> Dict[str, Any]:
        """Independent crossed-RE fit via pymer4 → lme4 (R).

            score ~ task_type + (1|task) + (1|model) + (1|variant)

        pymer4 exposes random-effect variances on ``Lmer.ranef_var`` — a
        DataFrame indexed by grouping factor (plus a ``Residual`` row) with a
        ``Var`` column. (The earlier dict-by-key access silently returned
        zeros — fixed here.)
        """
        if not _HAS_PYMER4:
            return {"error": "pymer4 not installed", "engine": "pymer4"}
        if df.empty or len(df) < 10:
            return {"error": f"Insufficient data for {label}", "engine": "pymer4"}

        # lme4 treats string columns as factors; strip patsy C(...) wrappers.
        fe_raw = fixed_fe.replace("C(", "").replace(")", "")
        fe_part = fe_raw if df["task_type"].nunique() > 1 else ""
        re_parts = ["(1|task)", "(1|model)", "(1|variant)"]
        rhs = " + ".join(([fe_part] if fe_part else []) + re_parts)
        formula = f"score ~ {rhs}"

        try:
            model = _Lmer(formula, data=df)
            model.fit(summarize=False, REML=True)
            rv = model.ranef_var  # DataFrame: index=factor (+ 'Residual'), col 'Var'

            def _var(idx_name: str) -> float:
                try:
                    return float(rv.loc[idx_name, "Var"])
                except (KeyError, TypeError, ValueError):
                    return 0.0

            vc = {
                "task": _var("task"),
                "model": _var("model"),
                "variant": _var("variant"),
                "residual": _var("Residual"),
            }
        except Exception as e:
            logger.error(f"pymer4 fit failed for {label}: {e}")
            return {"error": str(e), "label": label, "engine": "pymer4"}

        total_var = sum(vc.values())
        fixed_effects: Dict[str, float] = {}
        coefs = getattr(model, "coefs", None)
        if coefs is not None and hasattr(coefs, "index"):
            for name in coefs.index:
                col = "Estimate" if "Estimate" in coefs.columns else coefs.columns[0]
                fixed_effects[str(name)] = float(coefs.loc[name, col])

        return {
            "variance_components": vc,
            "icc": {k: v / total_var for k, v in vc.items()} if total_var > 0 else {},
            "total_variance": total_var,
            "fixed_effects": fixed_effects,
            "n_observations": len(df),
            "aic": float(getattr(model, "AIC", np.nan)),
            "bic": float(getattr(model, "BIC", np.nan)),
            "engine": "pymer4",
        }

    # ── DIAGNOSTIC-ONLY: task-nested fit (must NOT feed reported numbers) ──

    def _fit_variance_model_nested_diagnostic(
        self, df: "pd.DataFrame", label: str = ""
    ) -> Dict[str, Any]:
        """DIAGNOSTIC-ONLY task-nested fit (``groups="task"`` + vc_formula).

        Retained solely to demonstrate the B2 bug in tests: it nests ``model``
        within ``task``, so σ²_model absorbs the model×task interaction and
        σ²_residual is deflated. It is NOT wired into any reporting path and
        must never be used for published metrics.
        """
        if not _HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}
        if df.empty or len(df) < 10:
            return {"error": f"Insufficient data for {label}"}

        formula = (
            "score ~ C(task_type)"
            if df["task_type"].nunique() > 1
            else "score ~ 1"
        )
        try:
            model = MixedLM.from_formula(
                formula,
                groups="task",
                re_formula="1",
                vc_formula={
                    "model": "0 + C(model)",
                    "variant": "0 + C(variant)",
                },
                data=df,
            )
            result = model.fit(reml=True)
            vc = self._extract_nested_components(result)
        except Exception as e:
            return {"error": str(e), "label": label}

        total_var = sum(vc.values())
        return {
            "variance_components": vc,
            "icc": {k: v / total_var for k, v in vc.items()} if total_var > 0 else {},
            "total_variance": total_var,
            "n_observations": int(result.nobs),
            "engine": "statsmodels_nested_diagnostic",
        }

    @staticmethod
    def _extract_nested_components(result: Any) -> Dict[str, float]:
        """Variance components for the DIAGNOSTIC task-nested fit (cov_re holds
        the per-task random intercept; vcomp holds model/variant nested in task)."""
        vcomp = np.asarray(result.vcomp, dtype=float)
        names = list(getattr(result.model.exog_vc, "names", []))

        def _vc(name: str, fallback_idx: int) -> float:
            if name in names:
                return float(vcomp[names.index(name)])
            return float(vcomp[fallback_idx]) if vcomp.size > fallback_idx else 0.0

        return {
            "task": float(result.cov_re.iloc[0, 0]),
            "model": _vc("model", 0),
            "variant": _vc("variant", 1),
            "residual": float(result.scale),
        }

    def _compute_variance_decomposition(
        self,
        grouped: Dict[str, VariantScores],
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            import pandas as pd
        except ImportError:
            return {"error": "pandas not installed"}

        rows = []
        for g in grouped.values():
            resolved_tt = normalize_task_type(task_type or g.task_type)
            for model_id, score in g.scores.items():
                if not np.isnan(score):
                    rows.append(
                        {
                            "score": score,
                            "tier": g.tier,
                            "task_type": resolved_tt,
                            "task": g.task_id,
                            "model": model_id,
                            "variant": f"{g.task_id}_{g.tier}_{g.variant_index}",
                        }
                    )

        if not rows:
            return {"error": "No valid data for variance decomposition"}

        df = pd.DataFrame(rows)

        # ── Per-tier models ──
        per_tier: Dict[str, Any] = {}
        for tier in ("A", "B", "C"):
            df_tier = df[df.tier == tier]
            per_tier[tier] = self._fit_variance_model(df_tier, f"tier={tier}")

        # ── Pooled model (all tiers, with tier as fixed effect) ──
        # Same crossed engine as per-tier; tier enters as a fixed effect.
        pooled_fe = []
        if df["tier"].nunique() > 1:
            pooled_fe.append("C(tier)")
        if df["task_type"].nunique() > 1:
            pooled_fe.append("C(task_type)")
        pooled_fe_str = " + ".join(pooled_fe) if pooled_fe else "C(task_type)"

        pooled = self._fit_crossed_statsmodels(df, "pooled", fixed_fe=pooled_fe_str)

        return {
            "per_tier": per_tier,
            "pooled": pooled,
        }

    # ── Tier comparison (Kruskal-Wallis) ──────────────────────────────────

    def _compute_tier_comparison(
        self, per_task_tsi: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:

        tier_values: Dict[str, List[float]] = defaultdict(list)
        for task_dict in per_task_tsi.values():
            for tier, val in task_dict.items():
                if not np.isnan(val):
                    tier_values[tier].append(val)

        present_tiers = [t for t in ("A", "B", "C") if len(tier_values[t]) >= 2]
        if len(present_tiers) < 2:
            return {"kruskal_wallis": None, "post_hoc": {}}

        all_vals = [v for t in present_tiers for v in tier_values[t]]
        if len(set(all_vals)) <= 1:
            # Degenerate: every value identical → kruskal raises; TSI meaningless.
            return {
                "kruskal_wallis": {
                    "h_statistic": None,
                    "p_value": None,
                    "significant": False,
                    "degenerate": True,
                },
                "post_hoc": {},
            }

        # Kruskal-Wallis
        try:
            kw_groups = [tier_values[t] for t in present_tiers]
            h_stat, kw_pvalue = stats.kruskal(*kw_groups)
        except ValueError:
            return {
                "kruskal_wallis": {
                    "h_statistic": None,
                    "p_value": None,
                    "significant": False,
                    "degenerate": True,
                },
                "post_hoc": {},
            }

        post_hoc = self._posthoc_dunn(tier_values, present_tiers)

        return {
            "kruskal_wallis": {
                "h_statistic": float(h_stat),
                "p_value": float(kw_pvalue),
                "significant": bool(kw_pvalue < 0.05),
            },
            "post_hoc": post_hoc,
        }

    @staticmethod
    def _posthoc_dunn(
        tier_values: Dict[str, List[float]],
        present_tiers: List[str],
    ) -> Dict[str, Any]:
        """Dunn's (1964) post-hoc test with Benjamini-Hochberg FDR.

        Unlike pairwise Mann-Whitney (which re-ranks each pair independently),
        Dunn ranks the **pooled** sample once — the same ranking the
        Kruskal-Wallis omnibus uses — so the post-hoc stays consistent with the
        omnibus. Includes the standard tie correction.
        """
        labels: List[str] = []
        values: List[float] = []
        for t in present_tiers:
            for v in tier_values[t]:
                labels.append(t)
                values.append(v)

        arr = np.asarray(values, dtype=float)
        n_total = arr.size
        ranks = stats.rankdata(arr)  # average ranks for ties

        # Tie correction term: Σ(τ³ − τ) over tie groups
        _, tie_counts = np.unique(arr, return_counts=True)
        tie_sum = float(np.sum(tie_counts**3 - tie_counts))

        mean_rank: Dict[str, float] = {}
        n_in: Dict[str, int] = {}
        ranks_arr = np.asarray(ranks)
        labels_arr = np.asarray(labels)
        for t in present_tiers:
            mask = labels_arr == t
            mean_rank[t] = float(ranks_arr[mask].mean())
            n_in[t] = int(mask.sum())

        sigma2 = (n_total * (n_total + 1) / 12.0) - (
            tie_sum / (12.0 * (n_total - 1))
        ) if n_total > 1 else 0.0

        comparisons: List[Tuple[str, str, float, float]] = []
        p_values: List[float] = []
        for i in range(len(present_tiers)):
            for j in range(i + 1, len(present_tiers)):
                t1, t2 = present_tiers[i], present_tiers[j]
                se = np.sqrt(sigma2 * (1.0 / n_in[t1] + 1.0 / n_in[t2]))
                if se == 0:
                    z, p = 0.0, 1.0
                else:
                    z = (mean_rank[t1] - mean_rank[t2]) / se
                    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
                comparisons.append((t1, t2, float(z), float(p)))
                p_values.append(p)

        post_hoc: Dict[str, Any] = {}
        if p_values:
            reject, p_adjusted = statsmodels_multipletests(p_values)
            for idx, (t1, t2, z, p) in enumerate(comparisons):
                post_hoc[f"{t1}_vs_{t2}"] = {
                    "z_statistic": z,
                    "p_raw": float(p),
                    "p_adjusted": float(p_adjusted[idx]),
                    "significant": bool(reject[idx]),
                }
        return post_hoc

    # ── Task type inference ───────────────────────────────────────────────

    def _infer_task_type(self, variant_group: VariantScores) -> str:
        """Infer task type from grouped result data.

        Priority:
        1. Stored task_type from dataset metadata (if propagated through pipeline)
        2. Operator metadata: MCQ-specific keys → 'mcq'
        3. Heuristic: single-letter targets (A-D) → 'mcq'
        4. Default: 'classification'
        """
        if variant_group.task_type != "classification":
            return normalize_task_type(variant_group.task_type)

        op_meta = variant_group.operator_metadata
        if op_meta and any(k in op_meta for k in ("option_count", "permutation", "inverse_permutation", "new_gold_text")):
            return "mcq"

        targets = [t.strip().lower() for t in variant_group.targets.values() if t]
        if targets:
            single_letter = sum(1 for t in targets if len(t) <= 3 and t.isalpha())
            if single_letter / len(targets) > 0.5:
                return "mcq"

        return "classification"

    # ── Language inference ─────────────────────────────────────────────────

    @staticmethod
    def _infer_language(
        variant_group: VariantScores, default_lang: str = "en"
    ) -> str:
        """Resolve per-task language from metadata, falling back to global default.

        Priority:
        1. variant_group.language — propagated from inference_task.py via TaskResult.metadata (C4)
        2. Global default (from compute_trustvar_metrics argument)
        """
        lang = (variant_group.language or "").lower()[:2]
        if lang in ("en", "ru"):
            return lang
        return default_lang

    # ── Export ────────────────────────────────────────────────────────────

    def export_to_csv(
        self,
        metrics: Dict[str, Any],
        filepath: str,
    ) -> None:
        """
        Export per-task metrics to CSV statistical analysis.

        Columns: task_id, tier, tsi, ear, cv, iqr_cv, uninformative
        """
        rows = []
        per_task_tsi = metrics.get("per_task_tsi", {})
        per_task_ear = metrics.get("per_task_ear", {})
        per_task_cv = metrics.get("per_task_cv", {})
        per_task_iqr_cv = metrics.get("per_task_iqr_cv", {})
        uninformative = metrics.get("per_task_uninformative", {})

        for task_id in per_task_tsi:
            for tier in ("A", "B", "C"):
                tsi = per_task_tsi.get(task_id, {}).get(tier)
                ear = per_task_ear.get(task_id, {}).get(tier)
                cv = per_task_cv.get(task_id, {}).get(tier)
                iqr_cv = per_task_iqr_cv.get(task_id, {}).get(tier)
                unin = uninformative.get(task_id, {}).get(tier, False)

                if tsi is not None or ear is not None:
                    rows.append(
                        {
                            "task_id": task_id,
                            "tier": tier,
                            "tsi": tsi if tsi is not None else "",
                            "ear": ear if ear is not None else "",
                            "cv": cv if cv is not None else "",
                            "iqr_cv": iqr_cv if iqr_cv is not None else "",
                            "uninformative": unin,
                        }
                    )

        if not rows:
            logger.warning("No data to export to CSV")
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task_id", "tier", "tsi", "ear", "cv", "iqr_cv", "uninformative"]
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} rows to {filepath}")

    def export_to_json(
        self,
        metrics: Dict[str, Any],
        filepath: str,
    ) -> None:

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj) if not np.isnan(obj) else None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(
            json.dumps(metrics, default=_serialize)
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported metrics to {filepath}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _empty_trustvar_result(self) -> Dict[str, Any]:
        return {
            "per_task_tsi": {},
            "per_task_ear": {},
            "per_task_cv": {},
            "per_task_iqr_cv": {},
            "per_task_uninformative": {},
            "per_task_ear_flags": {},
            "per_task_cv_unreliable": {},
            "model_cv_star": {},
            "aggregate_tsi": {},
            "aggregate_ear": {},
            "variance_decomposition": {"per_tier": {}, "pooled": {}},
            "tier_comparison": {},
            "bootstrap_replicates": {
                "tsi": {"benchmark": {}, "per_task": {}},
                "ear": {"benchmark": {}, "per_task": {}},
            },
            "n_models": 0,
            "n_resamples": 0,
            "ci_level": 0.95,
        }

    # ── Basic metric methods (preserved from original) ────────────────────

    def _calculate_cv(self, values: List[float]) -> float:
        """Coefficient of variation: CV = std/mean × 100%."""
        if not values or len(values) < 2:
            return np.nan
        mean_val = np.mean(values)
        if mean_val == 0:
            return np.nan
        return (np.std(values) / mean_val) * 100

    def _calculate_corrected_cv(self, values: List[float]) -> float:
        """Bias-corrected CV for small samples: CV* = CV × (1 + 1/(4N))."""
        if not values or len(values) < 2:
            return np.nan
        arr = np.array(values)
        n = arr.size
        mean_val = arr.mean()
        std_val = arr.std(ddof=1)
        if mean_val == 0:
            return np.nan
        cv = std_val / mean_val
        return (1 + 1 / (4 * n)) * cv * 100

    def _calculate_iqr_cv(self, values: List[float]) -> float:
        if not values or len(values) < 2:
            return np.nan
        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        median_val = np.median(arr)
        if median_val == 0:
            return np.nan
        return ((q3 - q1) / median_val) * 100

    def _calculate_jsd_divergence(self, values: List[float]) -> float:
        """Jensen-Shannon Divergence for distribution heterogeneity."""
        if not values or len(values) < 2:
            return np.nan
        arr = np.array(values)
        arr_sum = arr.sum()
        if arr_sum == 0:
            return np.nan
        P = arr / arr_sum
        P_mean = P.mean()
        jsd = jensenshannon(P, [P_mean] * len(P)) ** 2
        return jsd * 100

    def _accuracy(self, results: List[TaskResult]) -> float:
        if not results:
            return 0.0
        correct = 0
        total = 0
        for r in results:
            if not r.target:
                continue
            tt = normalize_task_type(
                (r.metadata or {}).get("task_type") if r.metadata else None
            )
            if tt in ("mcq", "classification"):
                target_str = _normalize_target_with_classes(r)
                output = r.output.strip() if r.output else ""
                option_labels = _get_option_labels_for_result(r)
                if option_labels:
                    extracted = _extract_answer_labels(
                        output, option_labels, tt,
                        options=_get_options_for_result(r),
                        multi_label=_is_multi_label(r),
                        allow_stem_fallback=(tt == "classification"),
                    )
                    if target_str.lower() == extracted.lower():
                        correct += 1
                else:
                    if target_str.lower() == output.lower():
                        correct += 1
                total += 1
            elif tt in ("open_qa", "generation"):
                if r.judge_score is not None:
                    if r.judge_score >= 4.0:
                        correct += 1
                    total += 1
            else:
                if str(r.target).strip().lower() == r.output.strip().lower():
                    correct += 1
                total += 1
        if total == 0:
            return float("nan")
        return (correct / total * 100)

    def _bleu_score(self, results: List[TaskResult]) -> float:
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            scores = []
            smooth = SmoothingFunction()
            for r in results:
                if r.target:
                    reference = [str(r.target).split()]
                    candidate = r.output.split()
                    score = sentence_bleu(
                        reference, candidate, smoothing_function=smooth.method1
                    )
                    scores.append(score)
            return (sum(scores) / len(scores) * 100) if scores else 0.0
        except ImportError:
            logger.warning("NLTK not installed, BLEU score unavailable")
            return 0.0

    def _rouge_score(self, results: List[TaskResult]) -> float:
        try:
            from rouge import Rouge

            rouge = Rouge()
            scores = []
            for r in results:
                if r.target and r.output:
                    score = rouge.get_scores(r.output, str(r.target))[0]
                    scores.append(score["rouge-l"]["f"])
            return (sum(scores) / len(scores) * 100) if scores else 0.0
        except ImportError:
            logger.warning("rouge not installed, ROUGE score unavailable")
            return 0.0

    def _rta_score(self, results: List[TaskResult]) -> float:
        if not results:
            return 0.0
        refused = sum(int(r.refused) for r in results if r.refused)
        total = sum(1 for r in results if r.refused)
        return (refused / total * 100) if total > 0 else 0.0

    def _f1_score(self, results: List[TaskResult]) -> float:
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1 * 100

    def _precision(self, results: List[TaskResult]) -> float:
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        precision = precision_score(y_true, y_pred, average="macro")
        return precision * 100

    def _recall(self, results: List[TaskResult]) -> float:
        y_pred = [r.output for r in results]
        y_true = [r.target for r in results]
        recall = recall_score(y_true, y_pred, average="macro")
        return recall * 100


# ── Module-level helper ───────────────────────────────────────────────────


def statsmodels_multipletests(
    pvalues: List[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (reimplemented to avoid circular import).

    Returns:
        (reject, p_adjusted) — boolean array of rejections and adjusted p-values.
    """
    pvals = np.array(pvalues)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    # Sort
    order = np.argsort(pvals)
    sorted_pvals = pvals[order]

    # BH adjusted p-values
    adjusted = np.minimum(1, sorted_pvals * n / np.arange(1, n + 1))

    # Enforce monotonicity (from largest to smallest)
    adjusted_reversed = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Unsort
    p_adjusted = np.empty(n)
    p_adjusted[order] = adjusted_reversed

    reject = p_adjusted < alpha

    return reject, p_adjusted
