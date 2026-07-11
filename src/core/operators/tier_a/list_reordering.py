import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import yaml

from src.core.operators.base import (
    AbstractOperator,
    PreCheckResult,
    Tier,
    VariationResult,
)
from src.core.taxonomy import resolve_task_semantics

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMMUTATIVE_TASK_TYPES: Set[str] = {
    "set_membership",
    "multi_label_classification",
    "multi_fact_retrieval",
    "set_comparison",
    "which_of_the_following",
}


_LIST_ITEM_RE = re.compile(
    r"((?:^|\n)[ \t]*([-*•]|\d+[\)\.])[ \t]+)([^\n]+)"
)


_MCQ_MARKER_RE = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:[A-Ga-gА-Жа-ж][\)\.:]"
    r"|\d+[\)\.:]"
    r"|\([A-Ga-gА-Жа-ж]\)"
    r"|\(\d+\))"
    r"[ \t]+[^\n]+"
)

# Comma-separated + conjunction patterns
_COMMA_LIST_EN_RE = re.compile(
    r'(\w+(?:,\s+\w+)+,?\s+(?:and|or)\s+\w+)', re.IGNORECASE
)
_COMMA_LIST_EN_2_RE = re.compile(
    r'(\w+\s+(?:and|or)\s+\w+)', re.IGNORECASE
)
_COMMA_LIST_RU_RE = re.compile(
    r'(\w+(?:,\s+\w+)+,?\s+(?:и|или|а\s+также)\s+\w+)', re.IGNORECASE
)
_COMMA_LIST_RU_2_RE = re.compile(
    r'(\w+\s+(?:и|или)\s+\w+)', re.IGNORECASE
)

_CONJ_EN_RE = re.compile(r'\s+(and|or)\s+', re.IGNORECASE)
_CONJ_RU_RE = re.compile(r'\s+(и|или|а\s+также)\s+', re.IGNORECASE)

# Irreversible binomials — phrases whose order is frozen; swapping breaks semantics
_FROZEN_BINOMIALS_EN: Set[str] = {
    "bread and butter",
    "law and order",
    "pros and cons",
    "cause and effect",
    "supply and demand",
    "trial and error",
    "life and death",
    "peace and quiet",
    "safe and sound",
    "back and forth",
    "black and white",
    "rise and fall",
    "profit and loss",
    "input and output",
    "up and down",
    "in and out",
    "on and off",
    "step by step",
    "hand in hand",
    "face to face",
    "side by side",
    "one by one",
    "little by little",
    "day by day",
    "door to door",
    "knife and fork",
    "salt and pepper",
    "husband and wife",
    "mother and father",
    "brother and sister",
    "son and daughter",
    "men and women",
    "boys and girls",
    "none of the above",
    "all of the above",
}

_FROZEN_BINOMIALS_RU: Set[str] = {
    "хлеб и масло",
    "муж и жена",
    "мать и отец",
    "брат и сестра",
    "день и ночь",
    "причина и следствие",
    "вверх и вниз",
    "туда и обратно",
    "вперёд и назад",
    "вопросы и ответы",
    "чёрным по белому",
    "ни свет ни заря",
    "шаг за шагом",
    "время от времени",
    "бок о бок",
    "рука об руку",
    "лицом к лицу",
}


class ListReorderingOperator(AbstractOperator):
    operator_id = "list_reordering"
    tier = Tier.A

    def __init__(self):
        self._discourse_markers: dict = self._load_discourse_markers()

    @staticmethod
    def _load_discourse_markers() -> dict:
        result = {}
        for lang in ("en", "ru"):
            path = _DATA_DIR / f"{lang}_discourse_markers.yaml"
            if path.exists():
                with open(path, encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    result[lang] = data if isinstance(data, list) else []
            else:
                result[lang] = []
        return result

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        task_semantics = resolve_task_semantics(task_type, kwargs.get("task_semantics"))
        if task_semantics is None:
            return PreCheckResult(
                passed=False,
                reason="task_semantics not specified — cannot confirm commutativity",
            )
        if task_semantics not in COMMUTATIVE_TASK_TYPES:
            return PreCheckResult(
                passed=False,
                reason=f"Task semantics '{task_semantics}' not in commutative whitelist",
            )

        lang = language or "en"
        frozen_set = _FROZEN_BINOMIALS_RU if lang[:2] == "ru" else _FROZEN_BINOMIALS_EN
        text_lower = text.lower()
        for frozen in frozen_set:
            if frozen in text_lower:
                return PreCheckResult(
                    passed=False,
                    reason=f"Frozen binomial detected: '{frozen}'",
                )

        if self._looks_like_mcq(text):
            return PreCheckResult(passed=False, reason="Text looks like MCQ, not commutative list")

        blocks = self._detect_list_blocks(text, lang)
        if len(blocks) == 0:
            return PreCheckResult(passed=False, reason="Less than 2 list items found")

        all_texts = []
        for b in blocks:
            all_texts.extend(b["items"])

        if len(all_texts) < 2:
            return PreCheckResult(passed=False, reason="Less than 2 list items found")

        if self._has_duplicate_texts(all_texts):
            return PreCheckResult(passed=False, reason="Duplicate list item texts")

        discourse = self._discourse_markers.get(lang, self._discourse_markers.get("en", []))
        for marker in discourse:
            if re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE):
                return PreCheckResult(
                    passed=False,
                    reason=f"Discourse marker '{marker}' detected — list likely ordered",
                )

        return PreCheckResult(passed=True, details={"item_count": len(all_texts)})

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> VariationResult:
        lang = kwargs.get("language", "en")
        blocks = self._detect_list_blocks(text, lang)
        if len(blocks) == 0:
            return VariationResult(variant_text=text, metadata={"skipped": "not_enough_items"})

        all_texts = []
        for b in blocks:
            all_texts.extend(b["items"])

        if len(all_texts) < 2:
            return VariationResult(variant_text=text, metadata={"skipped": "not_enough_items"})

        rng = random.Random(seed)
        n = len(all_texts)
        perm = list(range(n))
        while perm == list(range(n)):
            rng.shuffle(perm)

        flat_permuted = [all_texts[perm[i]] for i in range(n)]

        variant = text
        consumed = 0
        for b in reversed(blocks):
            k = len(b["items"])
            new_block_texts = flat_permuted[n - consumed - k:n - consumed]
            new_block = b["reconstruct"](new_block_texts)
            variant = variant[:b["start"]] + new_block + variant[b["end"]:]
            consumed += k

        return VariationResult(
            variant_text=variant,
            metadata={
                "item_count": n,
                "permutation": perm,
                "shuffled": True,
            },
            original_text=text,
        )



    def _detect_list_blocks(
        self,
        text: str,
        language: str = "en",
    ) -> List[dict]:
        text = text.replace("\r\n", "\n")

        # Try per-line format first
        line_blocks = self._detect_line_blocks(text)
        if line_blocks:
            return line_blocks

        # Fallback: comma-separated
        comma_block = self._detect_comma_block(text, language)
        if comma_block:
            return [comma_block]

        # Fallback: 2-item conjunction
        conj_block = self._detect_conj_block(text, language)
        if conj_block:
            return [conj_block]

        return []

    def _detect_line_blocks(self, text: str) -> List[dict]:
        matches = list(_LIST_ITEM_RE.finditer(text))
        if len(matches) < 2:
            return []

        items = []
        prefixes = []
        for m in matches:
            prefix = m.group(1)  # e.g. "\n- " or "\n1) "
            text_content = m.group(3)
            items.append(text_content)
            prefixes.append(prefix)

        def reconstruct(new_texts: List[str]) -> str:
            return "".join(prefixes[i] + new_texts[i] for i in range(len(new_texts)))

        return [{
            "start": matches[0].start(),
            "end": matches[-1].end(),
            "items": items,
            "reconstruct": reconstruct,
        }]

    def _detect_comma_block(self, text: str, language: str) -> Optional[dict]:
        if language == "ru":
            pattern = _COMMA_LIST_RU_RE
            conj_re = _CONJ_RU_RE
        else:
            pattern = _COMMA_LIST_EN_RE
            conj_re = _CONJ_EN_RE

        match = pattern.search(text)
        if not match:
            return None

        match_text = match.group(0)
        conj_match = conj_re.search(match_text)
        if not conj_match:
            return None

        conjunction = conj_match.group(1).strip()
        parts = conj_re.split(match_text, maxsplit=1)
        first_part = parts[0]
        items = [x.strip() for x in first_part.split(",") if x.strip()]
        items.append(parts[-1].strip())

        if len(items) < 2:
            return None

        if self._has_frozen_binomial_in_items(items, language):
            return None

        def reconstruct(new_texts: List[str]) -> str:
            return ", ".join(new_texts[:-1]) + f" {conjunction} " + new_texts[-1]

        return {
            "start": match.start(),
            "end": match.end(),
            "items": items,
            "reconstruct": reconstruct,
        }

    def _detect_conj_block(self, text: str, language: str) -> Optional[dict]:
        if language == "ru":
            pattern = _COMMA_LIST_RU_2_RE
            conj_re = _CONJ_RU_RE
        else:
            pattern = _COMMA_LIST_EN_2_RE
            conj_re = _CONJ_EN_RE

        match = pattern.search(text)
        if not match:
            return None

        match_text = match.group(0)
        conj_match = conj_re.search(match_text)
        if not conj_match:
            return None

        conjunction = conj_match.group(1).strip()
        parts = conj_re.split(match_text, maxsplit=1)
        items = [p.strip() for p in parts if p.strip()]

        if self._has_frozen_binomial_in_items(items, language):
            return None

        if len(items) < 2:
            return None

        def reconstruct(new_texts: List[str]) -> str:
            return f" {conjunction} ".join(new_texts)

        return {
            "start": match.start(),
            "end": match.end(),
            "items": items,
            "reconstruct": reconstruct,
        }

    @staticmethod
    def _has_duplicate_texts(texts: List[str]) -> bool:
        return len(texts) != len(set(texts))

    @staticmethod
    def _has_frozen_binomial_in_items(items: List[str], language: str) -> bool:
        frozen_set = _FROZEN_BINOMIALS_RU if language[:2] == "ru" else _FROZEN_BINOMIALS_EN
        for item in items:
            item_lower = item.lower()
            for frozen in frozen_set:
                if frozen in item_lower:
                    return True
        return False

    @staticmethod
    def _looks_like_mcq(text: str) -> bool:
        return len(_MCQ_MARKER_RE.findall(text)) >= 2
