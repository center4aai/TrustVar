import re
from pathlib import Path
from typing import Optional, Set

from natasha import (
    Doc,
    NewsEmbedding,
    NewsNERTagger,
    Segmenter,
)
import yaml

from src.core.operators.base import (
    AbstractOperator,
    PreCheckResult,
    Tier,
    VariationResult,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class OrthographicNormalizationRuOperator(AbstractOperator):
    operator_id = "orthographic_normalization_ru"
    tier = Tier.A
    stochastic = False

    _ambiguous_pairs: Optional[Set[str]] = None
    _yo_dict: Optional[Set[str]] = None

    def __init__(self):
        if self.__class__._ambiguous_pairs is None:
            self.__class__._load_resources()
        self._ambiguous_pairs = self.__class__._ambiguous_pairs
        self._yo_dict = self.__class__._yo_dict
        self._natasha_segmenter = None
        self._natasha_ner = None

    @classmethod
    def _load_resources(cls):
        cls._ambiguous_pairs = set()
        cls._yo_dict = set()
        pairs_path = _DATA_DIR / "ru_orthographic_ambiguous_pairs.yaml"
        if pairs_path.exists():
            with open(pairs_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
                for entry in data.get("ambiguous_pairs", []):
                    for pair in entry.get("pairs", []):
                        cls._ambiguous_pairs.add(pair["form_e"])

        yo_path = _DATA_DIR / "ru_yo_dictionary.yaml"
        if yo_path.exists():
            with open(yo_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
                cls._yo_dict = set(data.get("yo_words", []))

    def _ensure_natasha_ner(self):
        if self._natasha_ner is None:
            self._natasha_segmenter = Segmenter()
            emb = NewsEmbedding()
            self._natasha_ner = NewsNERTagger(emb)

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        if language and language not in ("ru", "ru-RU", "ru-ru"):
            return PreCheckResult(passed=False, reason="Not a Russian-language task")

        if not text or len(text.strip()) < 2:
            return PreCheckResult(passed=False, reason="Text too short")

        return PreCheckResult(passed=True)

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        sub_rules: Optional[list[str]] = None,
        **kwargs,
    ) -> VariationResult:
        if sub_rules is None:
            sub_rules = ["yo_to_e"]

        result = text
        applied = []

        if "yo_to_e" in sub_rules:
            result = self._yo_to_e(result)
            applied.append("yo_to_e")

        if "e_to_yo" in sub_rules:
            result = self._e_to_yo(result)
            applied.append("e_to_yo")

        return VariationResult(
            variant_text=result,
            metadata={"sub_rules_applied": applied},
            original_text=text,
        )



    def _get_ner_words(self, text: str) -> Set[str]:
        self._ensure_natasha_ner()
        doc = Doc(text)
        doc.segment(self._natasha_segmenter)
        doc.tag_ner(self._natasha_ner)
        return {text[span.start:span.stop].lower() for span in doc.spans}

    def _iter_word(self, text: str, pos: int):
        start = pos
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        end = pos
        while end < len(text) - 1 and text[end + 1].isalpha():
            end += 1
        return start, end, text[start:end + 1].lower()

    def _yo_to_e(self, text: str) -> str:
        ner_words = self._get_ner_words(text)
        result = list(text)
        i = 0
        while i < len(result):
            if result[i] in ("ё", "Ё"):
                _, _, word = self._iter_word(text, i)
                if word not in ner_words:
                    result[i] = "е" if result[i] == "ё" else "Е"
            i += 1
        return "".join(result)

    def _e_to_yo(self, text: str) -> str:
        if not self._yo_dict:
            return text
        ner_words = self._get_ner_words(text)
        result = list(text)
        i = 0
        while i < len(result):
            if result[i] in ("е", "Е"):
                _, _, word = self._iter_word(text, i)
                if word in ner_words:
                    i += 1
                    continue
                word_yo = word.replace("е", "ё")
                if (word in self._yo_dict or word_yo in self._yo_dict) and word not in self._ambiguous_pairs:
                    result[i] = "ё" if result[i] == "е" else "Ё"
            i += 1
        return "".join(result)
