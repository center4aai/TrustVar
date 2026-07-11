# src/core/tasks/variation_cache.py

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.taxonomy import normalize_task_type, resolve_task_semantics
from src.utils.logger import logger


@dataclass
class CachedPrompt:
    original: str
    rendered: str
    language: str = "en"
    task_type: str = "classification"
    target: Any = None
    variations: List[Dict] = field(default_factory=list)

    def all_prompts(self) -> List[Dict]:
        # P0.1: baseline grouping key = rendered_prompt, matching variations so
        # eval_service._group_results derives the same task_id for baseline and
        # its variants. task_type mirrors the variation dicts so the baseline
        # carries the same C(task_type) level in variance decomposition.
        # N1: baseline carries the item's reference answer (self.target) — fixing
        # it at the source keeps it symmetric with the fallback path and avoids
        # forcing item.target onto variations (whose target may legitimately
        # differ, e.g. MCQ permutation, or be None).
        base = [
            {
                "text": self.rendered,
                "original": self.rendered,
                "variation_type": None,
                "target": self.target,
                "language": self.language,
                "task_type": self.task_type,
                # Baseline = unmodified reference prompt, never a variation: it
                # carries no validator verdict/layers and is valid by construction
                # (always in the ACCEPT∪FLAG companion subset). Mirrors the
                # no-variation fallback in inference_pipeline._get_prompts_from_cache.
                "validator_verdict": None,
                "validator_layers": None,
                "valid": True,
            }
        ]
        return base + self.variations


class VariationCache:
    """
    Кеш вариаций промптов.
    Ключ = hash(rendered_prompt + strategies + count).
    Один и тот же промпт не генерируется дважды.
    """

    def __init__(self):
        self._cache: Dict[str, List[Dict]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _make_key(
        self,
        rendered_prompt: str,
        strategies: List[str],
        count_per_strategy: int,
    ) -> str:
        # [AUGMENT 2026-07-11 default-prompts-seed] `custom_prompt` dropped from
        # the key — it never reached the operators (dead field, now removed).
        payload = json.dumps(
            {
                "prompt": rendered_prompt,
                "strategies": sorted(strategies),
                "count": count_per_strategy,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    async def get_or_generate(
        self,
        rendered_prompt: str,
        variation_generator: Any,
        strategies: List[str],
        count_per_strategy: int,
        extra_params: Dict,
    ) -> List[Dict]:
        key = self._make_key(rendered_prompt, strategies, count_per_strategy)

        if key in self._cache:
            return self._cache[key]

        # Get or create lock for this key
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            key_lock = self._locks[key]

        async with key_lock:
            # Double-check after acquiring lock
            if key in self._cache:
                return self._cache[key]

            variations = await variation_generator.generate_and_validate(
                prompt=rendered_prompt,
                strategies=strategies,
                count_per_strategy=count_per_strategy,
                custom_params=extra_params,
            )

            language = extra_params.get("language", "en")
            task_type = extra_params.get("task_type", "classification")
            formatted = [
                {
                    "text": var["text"],
                    "original": rendered_prompt,
                    "variation_type": var["strategy"],
                    "target": var.get("target"),
                    "language": language,
                    "task_type": task_type,
        
                    "validator_verdict": var.get("validation_status"),
                    "validator_layers": var.get("validation_metadata"),
                    "valid": var.get("valid"),
      
                    "operator_metadata": var.get("operator_metadata"),
                }
                for var in variations
            ]

            self._cache[key] = formatted
            return formatted

    def stats(self) -> Dict:
        return {
            "cached_prompts": len(self._cache),
            "total_variations": sum(len(v) for v in self._cache.values()),
        }


async def prebuild_variation_cache(
    items: List[Any],
    dataset: Any,
    task_config: Any,
    variation_generator: Any,
    concurrency: int = 3,
) -> Dict[int, CachedPrompt]:
    """
    Генерирует все вариации один раз до старта инференса.
    Возвращает Dict[item_index -> CachedPrompt].
    """
    cache = VariationCache()
    result: Dict[int, CachedPrompt] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def process_item(idx: int, item: Any) -> None:
        async with semaphore:
            rendered = item.prompt
            if item.template and item.variables:
                try:
                    from src.utils.template import render_template

                    rendered = render_template(item.template, item.variables)
                except Exception as e:
                    logger.warning(f"Template render failed for item {idx}: {e}")

            lang = _detect_language(item, dataset)
            # S1.1 (decision b): canonical task_type drives scoring/eval; the
            # fine task_semantics label drives operator preconditions. They are
            # resolved from separate dataset fields and never share a value.
            canon_task_type = normalize_task_type(dataset.task_type)
            task_semantics = resolve_task_semantics(
                dataset.task_type, getattr(dataset, "task_semantics", None)
            )
            cached = CachedPrompt(
                original=item.prompt,
                rendered=rendered,
                language=lang,
                task_type=canon_task_type,
                target=item.target,
            )

            if variation_generator:
                try:
                    cached.variations = await cache.get_or_generate(
                        rendered_prompt=rendered,
                        variation_generator=variation_generator,
                        strategies=task_config.variations.strategies,
                        count_per_strategy=task_config.variations.count_per_strategy,
                        extra_params={
                            "target_text": item.target,
                            "target": item.target,
                            "task_type": canon_task_type,      # scoring/eval + validator
                            "task_semantics": task_semantics,  # operator preconditions (fine)
                            "language": lang,
                            "template_text": item.template,     # A5: unresolved template with {slots}
                            "template_variables": item.variables,  # A5: original slot values
                        },
                    )
                except Exception as e:
                    logger.warning(f"Variation generation failed for item {idx}: {e}", exc_info=True)

            result[idx] = cached

    await asyncio.gather(
        *[process_item(idx, item) for idx, item in enumerate(items)],
        return_exceptions=True,
    )

    logger.info(
        f"Variation cache built: {len(result)} items, cache stats: {cache.stats()}"
    )
    return result


def _detect_language(item: Any, dataset: Any) -> str:
    lang = (item.metadata or {}).get("language", "")
    if not lang:
        lang = (dataset.metadata or {}).get("language", "")
    if not lang:
        for tag in dataset.tags or []:
            tl = tag.lower()[:2]
            if tl in ("ru", "en"):
                lang = tl
                break
    return (lang or "en")[:2]
