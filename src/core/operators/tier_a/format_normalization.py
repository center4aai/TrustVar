import re
import unicodedata
from typing import Optional

import spacy
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Segmenter,
)

from src.core.operators.base import (
    AbstractOperator,
    PreCheckResult,
    Tier,
    VariationResult,
)


_MINOR_WORDS_EN = {
    # articles
    "a", "an", "the",
    # conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    # short prepositions
    "in", "on", "at", "by", "to", "of", "with", "from", "up", "as",
    "off", "out", "via",
    # longer prepositions
    "into", "onto", "upon", "over", "under", "down", "without",
    "within", "through", "about", "across", "after", "before",
    "between", "during", "until",
    # auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "not",
}

_MINOR_WORDS_RU = {
    # prepositions
    "в", "во", "на", "по", "за", "у", "из", "от", "до", "при", "про",
    "без", "безо", "для", "о", "об", "обо", "под", "подо", "над",
    "надо", "через", "чрез", "сквозь", "вокруг", "около", "после",
    "перед", "передо", "среди", "между", "ради", "кроме", "вместо",
    "насчёт", "благодаря", "вследствие", "вроде", "касательно",
    "помимо", "сверх", "согласно",
    # conjunctions
    "и", "а", "но", "или", "да", "что", "чтобы", "чтоб", "если",
    "когда", "как", "будто", "словно", "хотя", "хоть", "потому",
    "поэтому", "так", "также", "то", "либо", "ни", "чем", "тем",
    "едва", "пока", "лишь", "раз", "ведь", "зато", "причём",
    "притом", "несмотря",
    # particles
    "не", "бы", "б", "же", "ж", "ли", "ль", "то", "таки", "вот",
    "вон",
    # short pronouns
    "я", "мы", "ты", "вы", "он", "она", "оно", "они",
    "мне", "меня", "нам", "нас", "тебя", "вас",
    "его", "её", "ее", "их", "себя", "себе", "собой",
    "этот", "эта", "это", "эти", "того", "той", "тех",
    "мой", "моя", "моё", "мои", "мое", "твой", "твоя",
    "твоё", "твое", "твои", "наш", "наша", "наше", "наши",
    "ваш", "ваша", "ваше", "ваши",
    # short adverbs
    "уже", "ещё", "еще", "только", "даже", "именно", "ведь",
    "все", "всё", "всего", "всех",
    "где", "там", "тут", "сюда", "туда", "здесь", "всюду",
    "всегда", "иногда", "теперь", "сейчас", "потом", "затем",
    "опять", "снова", "вновь",
    "чуть", "чуть-чуть",
    "почти", "совсем", "совершенно",
    "очень", "слишком", "довольно", "достаточно",
}


_ABBR_RE = re.compile(r'\b([A-ZА-ЯЁ]{2,})\b')
_MIXED_CASE_RE = re.compile(
    r"\b(?:"
    r"[a-zA-Zа-яА-ЯёЁ'-]*[a-zа-яё][a-zA-Zа-яА-ЯёЁ'-]*[A-ZА-ЯЁ][a-zA-Zа-яА-ЯёЁ'-]*"
    r'|O\'[A-ZА-ЯЁ][a-zA-Zа-яА-ЯёЁ\'-]*'
    r")\b"
)


def _is_full_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.5


def _is_title_case(text: str) -> bool:
    words = text.split()
    if len(words) < 2:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped / len(words) > 0.5


def _to_sentence_case(text: str) -> str:
    words = text.split()
    if not words:
        return text
    result = words[0].capitalize()
    for w in words[1:]:
        w2 = w.lower()
        if w2 and w2[0].isalpha():
            result += " " + w2
        else:
            result += " " + w
    return result


def _to_title_case(text: str, minor_words: set = _MINOR_WORDS_EN) -> str:
    words = text.split()
    if not words:
        return text
    result = words[0].capitalize()
    for w in words[1:]:
        w_lower = w.lower()
        if w_lower in minor_words:
            result += " " + w_lower
        else:
            result += " " + w.capitalize()
    return result


class FormatNormalizationOperator(AbstractOperator):
    operator_id = "format_normalization"
    tier = Tier.A
    stochastic = False

    CODE_TASK_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"write\s+(?:a\s+)?(?:function|method|class|program|script|code|module|routine|subroutine)"
            r"|complete\s+the\s+(?:code|function|implementation|snippet)"
            r"|implement\s+(?:a\s+)?(?:function|method|algorithm|class|interface)"
            r"|generate\s+(?:code|a\s+program|a\s+function|a\s+class)"
            r"|fill\s+in\s+the\s+(?:missing\s+)?code"
            r"|code\s+(?:completion|generation|writing)"
            r"|programming\s+task"
            r"|output\s+(?:only\s+)?the\s+code"
            # Russian
            r"|напиши(?:те)?\s+(?:функцию|метод|класс|программу|скрипт|код|модуль)"
            r"|дополни(?:те)?\s+код"
            r"|реализуй(?:те)?\s+(?:функцию|метод|алгоритм|класс|интерфейс)"
            r"|сгенерируй(?:те)?\s+(?:код|программу|функцию|класс)"
            r"|заполни(?:те)?\s+пропуски?\s+(?:в\s+)?код(?:е|а)"
            r"|заверши(?:те)?\s+(?:написание\s+)?код(?:а|е)"
            r"|дополнение\s+код(?:а|е)"
            r"|генерация\s+код(?:а|е)"
            r"|выведи(?:те)?\s+только\s+код"
            r"|задача\s+на\s+(?:программирование|кодирование|написание\s+кода)"
        r")"
    )

    STRUCTURED_OUTPUT_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"output\s+(?:as\s+)?(?:json|yaml|xml|structured\s+data|a\s+json\s+object)"
            r"|respond\s+with\s+(?:json|yaml|xml|a\s+valid\s+json)"
            r"|return\s+(?:json|yaml|xml)"
            r"|in\s+(?:json|yaml|xml)\s*format"
            r"|format\s*(?:as\s*)?(?:json|yaml|xml|structured)"
            r"|schema"
            r"|key\s*[-:]\s*value\s+pairs?"
            r"|structured\s+output"
            r"|serialize\s+(?:as\s+)?(?:json|yaml|xml)"
            # Russian
            r"|выведи(?:те)?\s+(?:в\s+формате\s+)?(?:json|yaml|xml)"
            r"|ответь(?:те)?\s+(?:в\s+формате\s+)?(?:json|yaml|xml)"
            r"|верни(?:те)?\s+(?:json|yaml|xml)"
            r"|представь(?:те)?\s+(?:ответ\s+)?в\s+(?:виде\s+)?(?:json|yaml|xml)"
            r"|в\s+формате\s+(?:json|yaml|xml)"
            r"|структурирован(?:ный|ные)\s+(?:вывод|данные|ответ)"
            r"|схема\s+(?:json|yaml|xml)?"
            r"|сериализуй(?:те)?\s+в\s+(?:json|yaml|xml)"
        r")"
    )

    FORMAT_FOLLOWING_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"respond\s+in\s+markdown"
            r"|output\s+as\s+(?:json|yaml|xml|markdown|table)"
            r"|format\s*(?:.{0,20})?\s*(?:markdown|json|table|yaml|xml|csv)"
            r"|use\s+(?:markdown|json|yaml|xml)\s+format"
            r"|reply\s+in\s+(?:a\s+)?table"
            r"|present\s+(?:your\s+)?answer\s+(?:as|in)\s+(?:a\s+)?(?:table|markdown|json|yaml|xml)"
            # Russian
            r"|ответь(?:те)?\s+в\s+(?:формате\s+)?(?:markdown|таблиц[ыуе]|json|yaml|xml|csv)"
            r"|выведи(?:те)?\s+(?:ответ|результат)\s+в\s+(?:виде\s+)?(?:таблиц[ыуе]|markdown|json|yaml|xml)"
            r"|оформи(?:те)?\s+(?:ответ\s+)?(?:как\s+)?(?:таблицу|markdown|json|yaml|xml)"
            r"|используй(?:те)?\s+(?:формат\s+)?(?:markdown|таблицу|json|yaml|xml)"
            r"|формат\s+(?:вывода|ответа)\s*[-:]\s*(?:markdown|таблица|json|yaml|xml|csv)"
            r"|представь(?:те)?\s+(?:ответ|данные)\s+в\s+(?:виде\s+)?(?:таблицы|markdown|json|yaml)"
            r"|в\s+(?:формате\s+)?(?:markdown|таблицы|json|yaml|xml)\b"
        r")"
    )

    SPELL_GRAMMAR_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"correct\s+(?:the\s+)?(?:spell(?:ing)?|grammar|punctuation|orthography)"
            r"|fix\s+(?:the\s+)?(?:spell(?:ing)?|grammar|punctuation|typos)"
            r"|spell(?:ing)?\s+check"
            r"|grammar\s+correction"
            r"|proofread"
            r"|check\s+for\s+(?:spelling|grammar)\s+errors"
            r"|identify\s+and\s+correct\s+(?:spelling|grammar)\s+mistakes"
            # Russian
            r"|исправь(?:те)?\s+(?:орфографи(?:ческие|ю)|граммати(?:ческие|ку)|пунктуац(?:ионные|ию)|опечатки)"
            r"|провер(?:ь|ьте)\s+(?:орфографию|грамматику|пунктуацию|правописание)"
            r"|найди(?:те)?\s+и\s+исправь(?:те)?\s+(?:орфографические|грамматические|пунктуационные)\s+ошибки"
            r"|коррекц(?:ия|ию)\s+(?:орфографии|грамматики|правописания|текста)"
            r"|исправление\s+(?:ошибок|опечаток)"
            r"|проверка\s+(?:орфографии|грамматики|правописания)"
            r"|отредактируй(?:те)?\s+(?:текст|ошибки)"
            r"|напиши(?:те)?\s+правильно"
            r"|ошиб[ок|ки|ку]\s+(?:в\s+)?(?:слов(?:е|ах)|текст(?:е|ах))"
        r")"
    )

    NUMERIC_FORMAT_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"format\s+(?:the\s+)?number|numeric\s+format"
            r"|currency\s+format"
            r"|decimal\s+separator|thousands?\s+separator"
            r"|locale\s*(?:-|\s)specific\s+number\s+format"
            r"|use\s+(?:comma|period|space)\s+as\s+(?:decimal|thousands?\s+separator)"
            r"|format\s+as\s+(?:currency|percentage)"
            r"|output\s+numbers?\s+(?:with|using)\s+(?:comma|dot|space)"
            r"|comma\s*[-]\s*separated\s+values\s*\(\s*CSV\s*\)"
            r"|digit\s+grouping"
            # Russian
            r"|формат\s+(?:числ(?:а|ел)|числовой\s+формат)"
            r"|денежный\s+формат|формат\s+валюты"
            r"|разделитель\s+(?:целой\s+и\s+дробной|десятичный|тысяч(?:ных)?)"
            r"|десятичный\s+разделитель"
            r"|локальный\s+формат\s+чис(?:ел|ла)"
            r"|используй(?:те)?\s+(?:запятую|точку|пробел)\s+(?:как\s+)?(?:десятичный\s+)?разделитель"
            r"|выведи(?:те)?\s+числ(?:о|а)\s+(?:с\s+)?(?:запятой|точкой|пробелом)"
            r"|разделение\s+(?:на\s+)?группы\s+разрядов"
            r"|числовой\s+вывод"
            r"|с\s+(?:запятой|точкой)\s+в\s+(?:качестве\s+)?(?:десятичного\s+)?разделителя"
        r")"
    )

    CONTAINS_CODE_BLOCK_REGEX = re.compile(
        r"(?i)"
        r"```[\s\S]*?```"
        r"|`{1,2}[^`]+`{1,2}"
        r"|блок(?:а|е|ов)?\s+код(?:а|у|е|ов)?"
        r"|встав(?:ь|ьте)\s+код"
        r"|привед(?:и|ите)\s+(?:пример\s+)?код(?:а)?"
    )
    
    NER_TASK_REGEX = (
        r"(?i)"
        r"(?:"
            # English
            r"named\s+entity\s+(?:recognition|extraction|detection|tagging|classification)"
            r"|NER\b"
            r"|extract\s+(?:named\s+)?entities"
            r"|find\s+(?:all\s+)?(?:named\s+)?entities"
            r"|identify\s+(?:the\s+)?(?:named\s+)?entities"
            r"|entity\s+(?:recognition|extraction|tagging|span\s+detection)"
            r"|tag\s+(?:the\s+)?(?:named\s+)?entities"
            r"|label\s+(?:the\s+)?(?:named\s+)?entities"
            r"|sequence\s+labeling\s+(?:for\s+)?(?:named\s+)?entit(?:y|ies)"
            r"|token\s+classification\s+(?:for\s+)?(?:named\s+)?entit(?:y|ies)"
            r"|BIO\s*(?:tagging|labeling|format|scheme)"
            r"|BIOSE?\s*(?:tagging|labeling|format|scheme)"
            r"|IOB\s*(?:tagging|labeling|format|scheme)"
            r"|entity\s+span\s+(?:detection|extraction|recognition)"
            # Russian
            r"|распознавани(?:е|я|ю|ем)\s+именован(?:ных|ного)\s+сущност(?:ей|и|ям|ями)"
            r"|выделени(?:е|я|ю|ем)\s+именован(?:ных|ного)\s+сущност(?:ей|и|ям|ями)"
            r"|извлечени(?:е|я|ю|ем)\s+именован(?:ных|ного)\s+сущност(?:ей|и|ям|ями)"
            r"|поиск\s+именован(?:ных|ного)\s+сущност(?:ей|и|ям|ями)"
            r"|най(?:ти|дите?)\s+именован(?:ные|ного)\s+сущност(?:и|ей|ям|ями)"
            r"|определи(?:ть|те)?\s+именован(?:ные|ного)\s+сущност(?:и|ей|ям|ями)"
            r"|разметь(?:ть|те)?\s+(?:именован(?:ные|ного)\s+)?сущност(?:и|ей|ям|ями)"
            r"|отметь(?:ть|те)?\s+(?:именован(?:ные|ного)\s+)?сущност(?:и|ей|ям|ями)"
            r"|классификац(?:ия|ию|ией)\s+именован(?:ных|ного)\s+сущност(?:ей|и|ям|ями)"
            r"|тег(?:гирование|ировани(?:е|я|ю|ем))\s+сущност(?:ей|и|ям|ями)"
            r"|NER\b"
            r"|BIO\s*(?:разметк(?:а|и|е|ой)|формат|схем(?:а|ы|е|ой))"
            r"|BIOSE?\s*(?:разметк(?:а|и|е|ой)|формат|схем(?:а|ы|е|ой))"
            r"|последовательн(?:ая|ой)\s+разметк(?:а|и|е)"
            r"|токен\s*(?:-|\s)?классификац(?:ия|ии|ией)"
        r")"
    )

    _nlp_en = None
    _natasha_segmenter = None
    _natasha_ner = None
    _natasha_morph = None

    def __init__(self, locale: str = "en_US"):
        self.locale = locale

    @classmethod
    def _get_nlp_en(cls):
        if cls._nlp_en is None:
            cls._nlp_en = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer"])
        return cls._nlp_en

    @classmethod
    def _get_natasha_ner(cls):
        if cls._natasha_ner is None:
            cls._natasha_segmenter = Segmenter()
            cls._natasha_morph = MorphVocab()
            emb = NewsEmbedding()
            cls._natasha_ner = NewsNERTagger(emb)
        return cls._natasha_segmenter, cls._natasha_ner, cls._natasha_morph

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        if not text or len(text.strip()) < 3:
            return PreCheckResult(passed=False, reason="Text too short")

        if re.search(self.STRUCTURED_OUTPUT_REGEX, text):
            return PreCheckResult(passed=False, reason="Structured-output task")

        if re.search(self.CODE_TASK_REGEX, text):
            return PreCheckResult(passed=False, reason="Code task")

        if re.search(self.FORMAT_FOLLOWING_REGEX, text):
            return PreCheckResult(passed=False, reason="Format-following task")

        if re.search(self.SPELL_GRAMMAR_REGEX, text):
            return PreCheckResult(passed=False, reason="Spelling/grammar task")

        if re.search(self.NUMERIC_FORMAT_REGEX, text):
            return PreCheckResult(passed=False, reason="Numeric-format task")

        return PreCheckResult(passed=True)

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> VariationResult:
        result = text
        applied = []

        result = unicodedata.normalize("NFC", result)
        applied.append("unicode_nfc")

        result = self._normalize_whitespace(result)
        applied.append("whitespace")

        result = self._normalize_punctuation(result)
        applied.append("punctuation")

        if not re.search(self.NER_TASK_REGEX, result) and not self._has_named_entity(result, language):
            result = self._normalize_case(result, language=language)
            applied.append("case")
        else:
            applied.append("case_skipped_ner_detected")

        if not self.CONTAINS_CODE_BLOCK_REGEX.search(result):
            result = self._normalize_markdown(result)
            applied.append("markdown")
        else:
            applied.append("markdown_skipped_informational")

        result = self._normalize_numeric(result)
        applied.append("numeric")

        return VariationResult(
            variant_text=result,
            metadata={
                "normalizations_applied": applied,
                "locale": self.locale,
            },
            original_text=text,
        )



    def _has_named_entity(self, text: str, language: Optional[str] = None) -> bool:
        lang = (language or self.locale or "en_US")[:2]
        try:
            if lang == "ru":
                segmenter, ner_tagger, morph_vocab = self._get_natasha_ner()
                doc = Doc(text)
                doc.segment(segmenter)
                doc.tag_ner(ner_tagger)
                return len(doc.spans) > 0
            nlp = self._get_nlp_en()
            doc = nlp(text)
            return len(doc.ents) > 0
        except Exception:
            return False

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    def _normalize_punctuation(self, text: str) -> str:
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        return text

    def _normalize_case(self, text: str, language: Optional[str] = None) -> str:
        if _is_full_uppercase(text):
            return text

        # Preserve mixed-case proper nouns before case transform
        mixed_case = {}

        def _save_mixed(m):
            idx = len(mixed_case)
            key = f"_mc{idx}_"
            mixed_case[key] = m.group(0)
            return key

        text = _MIXED_CASE_RE.sub(_save_mixed, text)

        # Preserve abbreviations before case transform
        abbreviations = {}

        def _save_abbr(m):
            idx = len(abbreviations)
            key = f"_ab{idx}_"
            abbreviations[key] = m.group(1)
            return key

        text = _ABBR_RE.sub(_save_abbr, text)

        minor_words = _MINOR_WORDS_RU if (language or "en_US")[:2] == "ru" else _MINOR_WORDS_EN

        if _is_title_case(text):
            text = _to_sentence_case(text)
        else:
            text = _to_title_case(text, minor_words=minor_words)

        for key, original in abbreviations.items():
            text = text.replace(key, original)

        for key, original in mixed_case.items():
            text = text.replace(key, original)

        return text

    def _normalize_markdown(self, text: str) -> str:
        # Protect inline math spans from accidental *-stripping
        _math = {}

        def _save_math(m):
            key = f"\x00M{len(_math)}\x00"
            _math[key] = m.group(0)
            return key

        text = re.sub(r'\$\$[\s\S]+?\$\$', _save_math, text)
        text = re.sub(r'\$[^\n$]+\$', _save_math, text)

        text = re.sub(r"(?<!\w)\*\*(?!\s)(.+?)(?<!\s)\*\*(?!\w)", r"\1", text)
        text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
        text = re.sub(r"__([^_]+?)__", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        for key, original in _math.items():
            text = text.replace(key, original)
        return text

    def _normalize_numeric(self, text: str) -> str:
        # Thousands separator removal (space, NBSP, comma before 3-digit groups)
        # — locale-agnostic, always safe
        text = re.sub(r"(?<=\d)[ ,\u00A0](?=\d{3}(?:[^\d]|$))", "", text)
        # Comma→decimal dot — locale-sensitive, skip if text contains
        # patterns like "3,14" (1–3 digits + comma + 2 digits)
        if not re.search(r"(?<!\d)\d{1,3},\d{2}(?!\d)", text):
            text = re.sub(r"(?<=\d),(?=\d)", ".", text)
        return text
