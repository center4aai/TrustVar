# src/core/services/variation_pipeline.py
"""
VariationPipeline — orchestrator for variation generation and validation.

Responsibilities (SRP):
  1. Get operator implementation from OperatorRegistry
  2. Execute operator.check_preconditions()
  3. Execute operator.apply() (with adapter for LLM-based operators)
  4. Execute VariationValidator.validate() — single validation point
  5. Deduplication and result formation

Validation occurs ONLY in VariationValidator.
"""

import hashlib
import traceback
from typing import Any, Dict, List, Optional

from src.adapters.factory import LLMFactory
from src.core.operators.base import AbstractOperator, Tier
from src.core.operators.registry import OperatorRegistry
from src.core.schemas.task import VariationStrategy
from src.core.services.variation_validator import (
    VariationValidator,
)
from src.database.repositories.model_repository import ModelRepository
from src.utils.logger import logger


class VariationPipeline:
    """
    Unified pipeline for generating prompt variations with subsequent validation.

    Usage:
        pipeline = VariationPipeline(model_id=..., validator=validator)
        await pipeline.initialize()
        results = await pipeline.generate_and_validate(
            prompt="...",
            strategies=[...],
            count_per_strategy=3,
            custom_params={...},
        )
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        validator: VariationValidator = None,
        bypass_validation: bool = False,
        keep_rejected: bool = False,
        progress_tracker: Any = None,
    ):
        self.model_id = model_id
        self.model = None
        self.adapter = None
        self.validator = validator
        self.bypass_validation = bypass_validation
        self.keep_rejected = keep_rejected
        self.progress_tracker = progress_tracker

    async def initialize(self):
        """Load model for variation generation (LLM-based operators Tier B/C).

        For Tier-A-only tasks model_id can be None — initialize() becomes no-op.
        """
        if not self.model_id:
            logger.info("VariationPipeline: no model_id, Tier-A-only mode (no LLM)")
            return
        model_repo = ModelRepository()
        self.model = await model_repo.find_by_id(self.model_id)
        if not self.model:
            raise ValueError(f"Variation model {self.model_id} not found")
        self.adapter = LLMFactory.create(self.model)
        logger.info(f"VariationPipeline initialized with model: {self.model.name}")

    async def generate_and_validate(
        self,
        prompt: str,
        strategies: List[VariationStrategy],
        count_per_strategy: int = 1,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate and validate variations for each operator.

        Args:
            prompt: Original prompt for variation.
            strategies: List of variation strategies.
            count_per_strategy: Number of variations per strategy.
            custom_params: Additional parameters (target, task_type, language, etc.).

        Returns:
            List of dicts with keys:
                text, strategy, target, valid, validation_status, metadata, operator_metadata
        """
        if not self.adapter:
            await self.initialize()

        custom_params = custom_params or {}
        variations: List[Dict[str, Any]] = []

        op_cache: Dict[VariationStrategy, Optional[AbstractOperator]] = {
            s: OperatorRegistry.get(s)() for s in strategies if OperatorRegistry.has(s)
        }

        for strategy in strategies:
            strategy_items: List[Dict[str, Any]] = []
            for i in range(count_per_strategy):
                try:
                    result = await self._process_single(
                        prompt=prompt,
                        strategy=strategy,
                        iteration=i,
                        custom_params=custom_params,
                        operator=op_cache.get(strategy),
                    )

                    if result is None:
                        continue

                    strategy_items.append(result)
                    logger.info(f"Generated variation {i + 1} using {strategy.value}")

                except Exception as e:
                    logger.error(
                        f"Failed to generate variation with {strategy}: {e}. "
                        f"Traceback: {traceback.format_exc()}"
                    )
                    continue

            # Deduplication
            seen = set()
            deduped = []
            for v in strategy_items:
                if v["text"] not in seen:
                    seen.add(v["text"])
                    deduped.append(v)

            if len(deduped) < len(strategy_items):
                logger.warning(
                    f"Deduplicated {len(strategy_items) - len(deduped)} duplicate variations "
                    f"for {strategy.value} ({len(deduped)} unique of {len(strategy_items)} total)"
                )

            logger.info(
                f"k_A for {strategy.value}: {len(deduped)} unique "
                f"(requested: {count_per_strategy})"
            )
            variations.extend(deduped)

        return variations

    async def _process_single(
        self,
        prompt: str,
        strategy: VariationStrategy,
        iteration: int,
        custom_params: Dict[str, Any],
        operator: Optional[AbstractOperator] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Full cycle for one variation:
          check_preconditions → apply → VariationValidator.validate
        """
        operator_cls = OperatorRegistry.get(strategy)

        # Deterministic operators skip repeated iterations
        if (
            hasattr(operator_cls, "stochastic")
            and not operator_cls.stochastic
            and iteration > 0
        ):
            logger.info(
                f"{strategy.value} is deterministic — skipping iteration {iteration + 1}"
            )
            return None

        if operator is None:
            operator = operator_cls()

        # 1. Preconditions
        # S1.1 (decision b): operators gate on the fine task_semantics; pass it
        # alongside the canonical task_type. apply() receives both via
        # custom_params (apply_kwargs below).
        precheck = await operator.check_preconditions(
            prompt,
            task_type=custom_params.get("task_type"),
            task_semantics=custom_params.get("task_semantics"),
            language=custom_params.get("language"),
        )
        if not precheck.passed:
            logger.info(
                f"Preconditions failed for {strategy.value}: {precheck.reason}; skipping"
            )
            return None

        # 2. Apply (generate variation)
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]
        operator_id = getattr(operator_cls, "operator_id", strategy.value)
        seed_str = f"{prompt_hash}:{operator_id}:{iteration}"
        seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)

        op_tier = getattr(operator_cls, "tier", None)

        # For Tier B/C pass adapter, for Tier A it's not needed
        apply_kwargs = {**custom_params}
        if op_tier in (Tier.B, Tier.C):
            apply_kwargs["adapter"] = self.adapter

        result = await operator.apply(prompt, seed=seed, **apply_kwargs)

        if self.progress_tracker:
            self.progress_tracker.record_result()

        # 3. Validation — single validation point via VariationValidator.
        # DIAGNOSTIC bypass: keep every candidate so the coverage health-check can
        # measure generation mechanics independently of validator thresholds.
        if self.bypass_validation:
            val_passed, val_status_str, val_meta = True, "bypassed", {}
        else:
            val_passed, val_status, val_meta = await self.validator.validate(
                original=prompt,
                variation=result.variant_text,
                strategy=strategy.value,
                target=custom_params.get("target"),
                language=custom_params.get("language", "en"),
                task_type=custom_params.get("task_type", "unknown"),
                operator_metadata=result.metadata,
            )
            val_status_str = val_status.value
            if not val_passed:
                if not self.keep_rejected:
                    logger.info(
                        f"Validator rejected {strategy.value}: {val_status_str}; skipping"
                    )
                    return None
                # validate-but-keep: retain the REJECTed variant as an annotated
                # datapoint (valid=False + full verdict/layers below) instead of
                # dropping it before the model loop. The verdict is logged, not
                # used as a filter (keep-all policy).
                logger.info(
                    f"Validator rejected {strategy.value}: {val_status_str}; "
                    f"keeping (validate-but-keep, valid=False)"
                )
            elif val_status_str.startswith("flag"):
                logger.info(f"Validator flagged {strategy.value}: {val_status_str}")

        # 4. Target resolution (operator may change gold answer)
        # Prefer the remapped LABEL (e.g. "1", "C") so the variant target stays in
        # the same namespace as the baseline / option_labels — the scorer compares
        # output-label == target-label (eval_service._extract_score). new_gold_text
        # (option text) is kept only as a legacy fallback.
        new_target = custom_params.get("target")
        if "new_gold_label" in result.metadata:
            new_target = result.metadata["new_gold_label"]
        elif "new_gold_text" in result.metadata:
            new_target = result.metadata["new_gold_text"]
        elif "new_gold" in result.metadata:
            new_target = str(result.metadata["new_gold"])

        return {
            "text": result.variant_text,
            "strategy": strategy.value,
            "target": new_target,
            "valid": val_passed,
            "validation_status": val_status_str,
            "validation_metadata": val_meta,
            "metadata": {
                "original": prompt,
                "iteration": iteration,
                "params": custom_params.get(strategy.value, {}),
            },
            "operator_metadata": result.metadata,
        }
