from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass
class PreCheckResult:
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariationResult:
    variant_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_text: str = ""


@dataclass
class OperatorOutput:
    text: str
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractOperator(ABC):
    operator_id: str
    tier: Tier
    stochastic: bool = True  # False → operator produces identical output for same input

    @abstractmethod
    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        ...

    @abstractmethod
    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> VariationResult:
        ...


class TierAOperator(AbstractOperator):
    """Symbolic operator — full verification without neural network."""

    tier = Tier.A


class TierBOperator(AbstractOperator):
    """Symbolic operator with neural gate — may have LLM fallback."""

    tier = Tier.B

    async def get_llm_fallback(
        self,
        text: str,
        language: str,
        adapter=None,
    ) -> Optional[str]:
        return None


class TierCOperator(AbstractOperator):
    """LLM-based operator with Jinja2 prompt template."""

    tier = Tier.C

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        ...

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> PreCheckResult:
        if len(text.strip().split()) < 4:
            return PreCheckResult(passed=False, reason="Text too short")
        return PreCheckResult(passed=True)

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        adapter=None,
        **kwargs,
    ) -> VariationResult:
        from jinja2 import Template

        tpl = Template(self.prompt_template)
        prompt = tpl.render(prompt=text)

        if adapter is None:
            return VariationResult(
                variant_text=text,
                metadata={"skipped": "no_adapter"},
                original_text=text,
            )

        variation = await adapter.generate(
            prompt,
            temperature=0.8,
        )
        return VariationResult(
            variant_text=variation.strip(),
            metadata={"strategy": self.operator_id},
            original_text=text,
        )