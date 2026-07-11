from typing import Optional

from src.core.operators.base import AbstractOperator, Tier
from src.core.schemas.task import VariationStrategy


class OperatorRegistry:
    _operators: dict[VariationStrategy, type[AbstractOperator]] = {}

    @classmethod
    def register(cls, strategy: VariationStrategy, operator_cls: type[AbstractOperator]):
        cls._operators[strategy] = operator_cls

    @classmethod
    def get(cls, strategy: VariationStrategy) -> type[AbstractOperator]:
        impl = cls._operators.get(strategy)
        if impl is None:
            raise KeyError(f"No operator registered for strategy: {strategy}")
        return impl

    @classmethod
    def has(cls, strategy: VariationStrategy) -> bool:
        return strategy in cls._operators

    @classmethod
    def get_tier(cls, strategy: VariationStrategy) -> Optional[Tier]:
        impl = cls._operators.get(strategy)
        if impl is None:
            return None
        return impl.tier if hasattr(impl, "tier") else None


def _register_all_operators():
    # ── Tier A ──
    from src.core.operators.tier_a.format_normalization import FormatNormalizationOperator
    from src.core.operators.tier_a.list_reordering import ListReorderingOperator
    from src.core.operators.tier_a.mcq_option_permutation import McqOptionPermutationOperator
    from src.core.operators.tier_a.orthographic_normalization_ru import OrthographicNormalizationRuOperator
    from src.core.operators.tier_a.typed_parametric_substitution import TypedParametricSubstitutionOperator

    OperatorRegistry.register(VariationStrategy.FORMAT_NORMALIZATION, FormatNormalizationOperator)
    OperatorRegistry.register(VariationStrategy.ORTHOGRAPHIC_NORMALIZATION_RU, OrthographicNormalizationRuOperator)
    OperatorRegistry.register(VariationStrategy.MCQ_OPTION_PERMUTATION, McqOptionPermutationOperator)
    OperatorRegistry.register(VariationStrategy.LIST_REORDERING, ListReorderingOperator)
    OperatorRegistry.register(VariationStrategy.TYPED_PARAMETRIC_SUBSTITUTION, TypedParametricSubstitutionOperator)

    # ── Tier B ──
    from src.core.operators.tier_b.active_passive_voice import ActivePassiveVoiceOperator
    from src.core.operators.tier_b.controlled_descriptive_modifier_insertion import ControlledDescriptiveModifierInsertionOperator
    from src.core.operators.tier_b.controlled_syntactic_transformations import ControlledSyntacticTransformationsOperator
    from src.core.operators.tier_b.nominalisation import NominalisationOperator
    from src.core.operators.tier_b.sentence_split_merge import SentenceSplitMergeOperator
    from src.core.operators.tier_b.monosemic_synonym_substitution import MonosemicSynonymSubstitutionOperator

    OperatorRegistry.register(VariationStrategy.ACTIVE_PASSIVE_VOICE, ActivePassiveVoiceOperator)
    OperatorRegistry.register(VariationStrategy.MONOSEMIC_SYNONYM_SUBSTITUTION, MonosemicSynonymSubstitutionOperator)
    OperatorRegistry.register(VariationStrategy.NOMINALISATION, NominalisationOperator)
    OperatorRegistry.register(VariationStrategy.CONTROLLED_SYNTACTIC_TRANSFORMATIONS, ControlledSyntacticTransformationsOperator)
    OperatorRegistry.register(VariationStrategy.SENTENCE_SPLIT_MERGE, SentenceSplitMergeOperator)
    OperatorRegistry.register(VariationStrategy.CONTROLLED_DESCRIPTIVE_MODIFIER_INSERTION, ControlledDescriptiveModifierInsertionOperator)

    # ── Tier C ──
    from src.core.operators.tier_c.back_translation_single_pivot import BackTranslationSinglePivotOperator
    from src.core.operators.tier_c.length_variation import LengthVariationOperator
    from src.core.operators.tier_c.negation_scope_preserving_rephrasing import NegationScopePreservingRephrasingOperator
    from src.core.operators.tier_c.paraphrase_free import ParaphraseFreeOperator
    from src.core.operators.tier_c.paraphrase_lexico_syntactic_constrained import ParaphraseLexicoSyntacticConstrainedOperator
    from src.core.operators.tier_c.register_formal_informal import RegisterFormalInformalOperator
    from src.core.operators.tier_c.tone_shift import ToneShiftOperator
    from src.core.operators.tier_c.wsd_synonym_substitution import WsdSynonymSubstitutionOperator

    OperatorRegistry.register(VariationStrategy.PARAPHRASE_LEXICO_SYNTACTIC_CONSTRAINED, ParaphraseLexicoSyntacticConstrainedOperator)
    OperatorRegistry.register(VariationStrategy.PARAPHRASE_FREE, ParaphraseFreeOperator)
    OperatorRegistry.register(VariationStrategy.LENGTH_VARIATION, LengthVariationOperator)
    OperatorRegistry.register(VariationStrategy.REGISTER_FORMAL_INFORMAL, RegisterFormalInformalOperator)
    OperatorRegistry.register(VariationStrategy.TONE_SHIFT, ToneShiftOperator)
    OperatorRegistry.register(VariationStrategy.NEGATION_SCOPE_PRESERVING_REPHRASING, NegationScopePreservingRephrasingOperator)
    OperatorRegistry.register(VariationStrategy.WSD_SYNONYM_SUBSTITUTION, WsdSynonymSubstitutionOperator)
    OperatorRegistry.register(VariationStrategy.BACK_TRANSLATION_SINGLE_PIVOT, BackTranslationSinglePivotOperator)


_register_all_operators()
