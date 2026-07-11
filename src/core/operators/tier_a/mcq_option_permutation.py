import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.core.operators.base import (
    AbstractOperator,
    PreCheckResult,
    Tier,
    VariationResult,
)

_ORDINAL_IMPLICATION_PATTERNS = {
    "en": [
        r"(?i)\b(never|rarely|sometimes|often|always|usually|seldom|frequently|occasionally)\b",
        r"(?i)\b(strongly disagree|disagree|neutral|agree|strongly agree)\b",
        r"(?i)\b(very (low|poor|bad)|low|poor|bad|medium|average|high|good|very (high|good|well))\b",
    ],
    "ru": [
        r"(?i)\b(никогда|редко|иногда|часто|всегда|обычно|изредка|постоянно)\b",
        r"(?i)\b(категорически не согласен|не согласен|нейтрально|согласен|полностью согласен)\b",
        r"(?i)\b(очень низкий|низкий|средний|высокий|очень высокий)\b",
    ],
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _expand_marker_config(cfg: dict) -> set:
    markers: set = set()
    letter = cfg.get("letter", {})
    for char in letter.get("values", []):
        for sfx in letter.get("suffix_chars", [")"]):
            markers.add(f"{char}{sfx}")
    number = cfg.get("number", {})
    for n in range(number.get("start", 1), number.get("end", 0) + 1):
        for sfx in number.get("suffix_chars", [")"]):
            markers.add(f"{n}{sfx}")
    for v in cfg.get("paren_lower", {}).get("values", []):
        markers.add(f"({v})")
    for v in cfg.get("paren_upper", {}).get("values", []):
        markers.add(f"({v})")
    for n in range(
        cfg.get("paren_number", {}).get("start", 1),
        cfg.get("paren_number", {}).get("end", 0) + 1,
    ):
        markers.add(f"({n})")
    return markers


class McqOptionPermutationOperator(AbstractOperator):
    operator_id = "mcq_option_permutation"
    tier = Tier.A

    def __init__(self):
        self._combination_patterns: Dict[str, List[Dict]] = self._load_patterns()
        self._markers_en, self._markers_ru, self._all_markers = self._load_markers()
        self._inline_marker_re = re.compile(
            r'(?:(?<=^)|(?<=\s))('
            + '|'.join(re.escape(m) for m in sorted(self._all_markers, key=len, reverse=True))
            + r')(?=\s|$)'
        )

    @staticmethod
    def _load_patterns() -> Dict[str, List[Dict]]:
        path = _DATA_DIR / "mcq_combination_patterns.yaml"
        if path.exists():
            with open(path, encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"en": [], "ru": []}

    @staticmethod
    def _load_markers() -> Tuple[set, set, set]:
        path = _DATA_DIR / "mcq_marker_patterns.yaml"
        if not path.exists():
            return set(), set(), set()
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        en_markers = _expand_marker_config(data.get("en", {}))
        ru_markers = _expand_marker_config(data.get("ru", {}))
        all_markers = en_markers | ru_markers
        return en_markers, ru_markers, all_markers

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        ordinal_threshold: float = 0.5,
        **kwargs,
    ) -> PreCheckResult:
        options, _ = self._parse_options(text)
        if len(options) < 2:
            return PreCheckResult(passed=False, reason="Less than 2 options found")

        option_texts = [t for _, t in options]
        lang = language or "en"

        if self._has_duplicate_texts(option_texts):
            return PreCheckResult(passed=False, reason="Duplicate option texts")

        if self._has_combination_option(option_texts, lang):
            return PreCheckResult(passed=False, reason="Combination option detected")

        if self._has_ordinal_implication(option_texts, lang, ordinal_threshold):
            return PreCheckResult(passed=False, reason="Ordinal-implication options detected")

        return PreCheckResult(passed=True, details={"option_count": len(options)})

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        original_gold_index: int = 0,
        target_text: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        options, _ = self._parse_options(text)
        if len(options) < 2:
            return VariationResult(variant_text=text, metadata={"skipped": "not_enough_options"})

        original_gold_index = self._resolve_gold_index(
            options, target_text, original_gold_index
        )

        rng = random.Random(seed)
        n = len(options)
        perm, inv_perm = self._make_permutation(rng, n)


        new_gold_index = inv_perm[original_gold_index]

        markers = [m for m, _ in options]
        texts = [t for _, t in options]
 
        new_texts = [texts[perm[i]] for i in range(n)]

        variant = self._render_permuted(text, options, new_texts)

        return VariationResult(
            variant_text=variant,
            metadata={
                "permutation": perm,
                "inverse_permutation": inv_perm,
                "original_gold_index": original_gold_index,
                "new_gold_index": new_gold_index,
                "new_gold_text": texts[original_gold_index],
                "new_gold_label": self._normalize_label(markers[new_gold_index]),
                "option_count": n,
            },
            original_text=text,
        )

    @staticmethod
    def _make_permutation(rng: random.Random, n: int) -> Tuple[List[int], List[int]]:
        """Return a non-identity permutation ``perm`` and its inverse.

        ``perm[i]`` = original content index displayed at position ``i``;
        ``inv_perm[original_index]`` = displayed position of that content.
        """
        perm = list(range(n))
        while perm == list(range(n)):
            rng.shuffle(perm)
        inv_perm = [0] * n
        for displayed_pos, orig_idx in enumerate(perm):
            inv_perm[orig_idx] = displayed_pos
        return perm, inv_perm

    @staticmethod
    def _normalize_label(marker: str) -> str:
        """Reduce an option marker to its bare label ("2." -> "2", "C)" -> "C",
        "(a)" -> "a"), matching the dataset's ``option_labels`` form."""
        return marker.strip().strip("().").rstrip(".):").strip()

    @staticmethod
    def _resolve_gold_index(
        options: List[Tuple[str, str]],
        target_text: Optional[str],
        default: int,
    ) -> int:
        """Resolve the gold option index from the dataset target (option text or
        label), falling back to ``default`` when it cannot be matched."""
        if target_text is None:
            return default
        for i, (_, opt_text) in enumerate(options):
            if opt_text.strip() == target_text.strip():
                return i
        # Fallback: match against marker/label (e.g. "A", "2")
        target_stripped = target_text.strip().strip("().").rstrip(".)")
        for i, (marker, _) in enumerate(options):
            if marker.strip().strip("().").rstrip(".)") == target_stripped:
                return i
        return default

    def _render_permuted(
        self,
        text: str,
        options: List[Tuple[str, str]],
        new_texts: List[str],
    ) -> str:
        """Rebuild the prompt with option texts permuted **in place**.

        Each option keeps its marker at its original line position; every
        non-option line (stem before, answer instruction after) is preserved
        verbatim. Only the single-line inline layout falls back to a
        stem+block reconstruction.
        """
        normalized = text.replace("\r\n", "\n")
        lines = normalized.split("\n")
        opt_line_indices = [
            idx for idx, line in enumerate(lines)
            if self._match_marker(line.strip()) is not None
        ]

        if len(opt_line_indices) == len(options):
            out = list(lines)
            for pos, line_idx in enumerate(opt_line_indices):
                marker = options[pos][0]
                out[line_idx] = f"{marker} {new_texts[pos]}"
            return "\n".join(out)

        # Inline (single-line) options: preserve the stem, re-emit options as a block.
        _, inline_stem = self._parse_options_inline(normalized)
        markers = [m for m, _ in options]
        new_block = "\n".join(
            f"{markers[i]} {new_texts[i]}" for i in range(len(options))
        )
        stem_stripped = inline_stem.rstrip()
        return new_block if not stem_stripped else stem_stripped + "\n" + new_block

    def _match_marker(self, stripped_line: str) -> Optional[str]:
        """Return the option marker a stripped line starts with, else None."""
        for marker_set in (self._markers_en, self._markers_ru):
            for marker in marker_set:
                if stripped_line.startswith(marker):
                    return marker
        return None

    def _parse_options(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        text = text.replace("\r\n", "\n")
        lines = text.strip().split("\n")
        options = []
        stem_lines = []

        for line in lines:
            marker = self._match_marker(line.strip())
            if marker is not None:
                opt_text = line.strip()[len(marker):].strip()
                options.append((marker, opt_text))
            else:
                stem_lines.append(line)

        if len(options) < 2:
            inline_opts, inline_stem = self._parse_options_inline(text)
            if len(inline_opts) >= 2:
                return inline_opts, inline_stem

        return options, "\n".join(stem_lines)

    def _parse_options_inline(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        text = text.replace("\r\n", "\n").strip()
        matches = list(self._inline_marker_re.finditer(text))
        if len(matches) < 2:
            return [], ""
        stem = text[:matches[0].start()].strip()
        options = []
        for i, m in enumerate(matches):
            marker = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            opt_text = text[start:end].strip()
            options.append((marker, opt_text))
        return options, stem

    @staticmethod
    def _has_duplicate_texts(option_texts: List[str]) -> bool:
        return len(option_texts) != len(set(option_texts))

    def _has_combination_option(self, option_texts: List[str], language: str) -> bool:
        patterns = self._combination_patterns.get(language, [])
        for entry in patterns:
            for pat in entry["patterns"]:
                for opt_text in option_texts:
                    if re.search(pat, opt_text, re.IGNORECASE):
                        return True
        return False

    def _has_ordinal_implication(
        self,
        option_texts: List[str],
        language: str,
        threshold: float = 0.5,
    ) -> bool:
        patterns = _ORDINAL_IMPLICATION_PATTERNS.get(language, _ORDINAL_IMPLICATION_PATTERNS["en"])
        for pat in patterns:
            matches = sum(1 for opt in option_texts if re.search(pat, opt))
            if matches >= len(option_texts) * threshold:
                return True
        return False
