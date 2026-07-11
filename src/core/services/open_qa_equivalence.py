# src/core/services/open_qa_equivalence.py
from __future__ import annotations

import os
from typing import Dict

from src.utils.logger import logger


class OpenQABackendUnavailable(RuntimeError):
    """Raised when an Open-QA neural backend (MiniCheck/AlignScore) cannot
    produce a real score — model not installed, checkpoint missing, or a
    scoring exception.

    Per the fail-closed principle : equivalence must NOT silently degrade
    to ``0.0`` (false "all answers diverge") or an uncalibrated heuristic.
    Callers catch this and mark the affected EAR cells ``unavailable`` (NaN +
    diagnostic flag) instead of contaminating the headline metric.
    """


class OpenQAEquivalence:
    """Lazy-loaded singleton for Open-QA answer equivalence scoring."""

    _minicheck = None
    _alignscore = None
    _ruroberta = None
    _ruroberta_tokenizer = None
    _minicheck_loaded = False
    _alignscore_loaded = False
    _ruroberta_loaded = False

    THETA_MINICHECK = 0.75   
    THETA_RUROBERTA = 0.75   
    THETA_ALIGN = 0.75       


    MINICHECK_MODEL_ENV = "MINICHECK_MODEL"
    MINICHECK_MODEL_DEFAULT = "flan-t5-large"
    RUROBERTA_MODEL_ENV = "RUROBERTA_MODEL"
    RUROBERTA_MODEL_DEFAULT = "s-nlp/ruRoberta-large-paraphrase-v1"

    @classmethod
    def resolved_backend_ids(cls) -> Dict[str, str]:
        """Configured Open-QA L3 backend model ids (env override → default).

        Single inventory used both by the loaders and by run-manifest pinning
        (Task 3.1 F7.1). NOTE: MiniCheck downloads into its OWN ``MINICHECK_CACHE_DIR``
        (``./cache/minicheck``), not the HF hub cache — an HF-cache scan of this id
        will therefore report ``not_cached`` even when the backend is installed;
        the recorded value pins WHICH backend model is configured, which is the
        reproducibility fact of interest. ruRoberta is a real HF repo id and does
        land in the HF cache.
        """
        return {
            "minicheck_en": os.environ.get(
                cls.MINICHECK_MODEL_ENV, cls.MINICHECK_MODEL_DEFAULT
            ),
            "ruroberta_ru": os.environ.get(
                cls.RUROBERTA_MODEL_ENV, cls.RUROBERTA_MODEL_DEFAULT
            ),
        }

    @classmethod
    def _load_minicheck(cls) -> None:
        if cls._minicheck_loaded:
            return
        cls._minicheck_loaded = True
        try:
            from minicheck.minicheck import MiniCheck

            cache_dir = os.environ.get(
                "MINICHECK_CACHE_DIR", "./cache/minicheck"
            )
            model_name = os.environ.get(
                cls.MINICHECK_MODEL_ENV, cls.MINICHECK_MODEL_DEFAULT
            )
            cls._minicheck = MiniCheck(
                model_name=model_name, cache_dir=cache_dir
            )
            logger.info(f"MiniCheck loaded: {model_name}")
        except ImportError as e:
            # An ImportError here may be the minicheck package OR a missing tokenizer
            # dep (e.g. sentencepiece for the flan-T5 tokenizer) raised inside the
            # constructor — surface the real message so it is diagnosable.
            logger.warning(
                f"MiniCheck unavailable (ImportError): {e}. "
                "If the package is present, check tokenizer deps (sentencepiece) — "
                'install via: uv sync --extra celery'
            )
        except Exception as e:
            logger.error(f"Failed to load MiniCheck: {e}")

    @classmethod
    def _load_alignscore(cls) -> None:
        if cls._alignscore_loaded:
            return
        cls._alignscore_loaded = True
        try:
            from alignscore import AlignScore

            ckpt_path = os.environ.get(
                "ALIGNSCORE_CKPT_PATH",
                "./cache/alignscore/alignscore-large.ckpt",
            )
            device = os.environ.get("ALIGNSCORE_DEVICE", "cpu")
            cls._alignscore = AlignScore(
                model="roberta-large",
                batch_size=32,
                device=device,
                ckpt_path=ckpt_path,
                evaluation_mode="nli_sp",
                verbose=False,
            )
            logger.info(f"AlignScore loaded on {device}")
        except ImportError:
            logger.warning(
                "alignscore-SpeedOfMagic not installed. Install via: "
                "pip install alignscore-SpeedOfMagic"
            )
        except Exception as e:
            logger.error(f"Failed to load AlignScore: {e}")

    @classmethod
    def score_minicheck(
        cls, contexts: list[str], claims: list[str]
    ) -> list[float]:
        """
        Score factuality using MiniCheck (EN only).

        MiniCheck(document, sentence) -> [0, 1].


        Args:
            contexts: grounding documents / questions
            claims: claims / answers to verify

        Returns:
            List of scores in [0, 1].

        Raises:
            OpenQABackendUnavailable: if MiniCheck is not loaded or scoring
            fails — fail-closed rather than returning a silent [0.0] (B3).
        """
        cls._load_minicheck()
        if cls._minicheck is None:
            raise OpenQABackendUnavailable(
                "MiniCheck backend not available (not installed or failed to load)"
            )

        try:
            pred_label, raw_prob, _, _ = cls._minicheck.score(
                docs=contexts, claims=claims
            )
            return [float(p) for p in raw_prob]
        except Exception as e:
            logger.error(f"MiniCheck scoring failed: {e}")
            raise OpenQABackendUnavailable(f"MiniCheck scoring failed: {e}") from e

    @classmethod
    def score_alignscore(
        cls, contexts: list[str], claims: list[str]
    ) -> list[float]:
        """
        Score factual consistency using AlignScore (multilingual).

        AlignScore(context, claim) -> float in [0, 1].
        Measures whether all information in claim is contained in context.

        Args:
            contexts: reference texts / questions + gold answers
            claims: candidate answers to verify

        Returns:
            List of scores in [0, 1].

        Raises:
            OpenQABackendUnavailable: if AlignScore is not loaded or scoring
            fails — fail-closed rather than returning a silent [0.0] (B3).
        """
        cls._load_alignscore()
        if cls._alignscore is None:
            raise OpenQABackendUnavailable(
                "AlignScore backend not available (not installed or failed to load)"
            )

        try:
            scores = cls._alignscore.score(
                contexts=contexts, claims=claims
            )
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"AlignScore scoring failed: {e}")
            raise OpenQABackendUnavailable(f"AlignScore scoring failed: {e}") from e

    @classmethod
    def _load_ruroberta(cls) -> None:
        if cls._ruroberta_loaded:
            return
        cls._ruroberta_loaded = True
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_name = os.environ.get(
                cls.RUROBERTA_MODEL_ENV, cls.RUROBERTA_MODEL_DEFAULT
            )
            device = os.environ.get(
                "RUROBERTA_DEVICE",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            )
            cls._ruroberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._ruroberta = (
                AutoModelForSequenceClassification.from_pretrained(model_name)
                .to(device)
                .eval()
            )
            logger.info(f"ruRoberta loaded: {model_name} on {device}")
        except ImportError:
            logger.warning(
                "transformers/torch not installed. Install via: "
                "uv sync --extra celery"
            )
        except Exception as e:
            logger.error(f"Failed to load ruRoberta: {e}")

    @classmethod
    def score_ruRoberta(
        cls, contexts: list[str], claims: list[str]
    ) -> list[float]:
        """
        Score paraphrase equivalence using ruRoberta-large-paraphrase-v1 (RU).

        Returns P(paraphrase) = softmax(logits)[:, 1] for each
        (context_i, claim_i) sentence pair. Bidirectionality is handled by the
        caller (``_are_equivalent_bidirectional``), which passes both
        directions and requires both to clear θ.

        Args:
            contexts: reference texts / questions + gold answers
            claims: candidate answers to verify

        Returns:
            List of paraphrase probabilities in [0, 1].

        Raises:
            OpenQABackendUnavailable: if ruRoberta is not loaded or scoring
            fails — fail-closed rather than returning a silent [0.0] (B3).
        """
        cls._load_ruroberta()
        if cls._ruroberta is None:
            raise OpenQABackendUnavailable(
                "ruRoberta backend not available (not installed or failed to load)"
            )

        try:
            import torch

            device = cls._ruroberta.device
            batch_size = int(os.environ.get("RUROBERTA_BATCH_SIZE", "16"))
            probs: list[float] = []
            with torch.inference_mode():
                for i in range(0, len(contexts), batch_size):
                    enc = cls._ruroberta_tokenizer(
                        contexts[i : i + batch_size],
                        claims[i : i + batch_size],
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(device)
                    logits = cls._ruroberta(**enc).logits
                    proba = torch.softmax(logits, dim=-1)[:, 1]
                    probs.extend(float(p) for p in proba.cpu().tolist())
            return probs
        except Exception as e:
            logger.error(f"ruRoberta scoring failed: {e}")
            raise OpenQABackendUnavailable(f"ruRoberta scoring failed: {e}") from e

    @classmethod
    def _are_equivalent_bidirectional(
        cls,
        context: str,
        answer_a: str,
        answer_b: str,
        scorer: str,
    ) -> bool:
        """
        Bidirectional equivalence check via doc→claim scoring.

        Two answers are equivalent iff both directions pass the threshold:
          forward:  scorer(q + answer_a, q + answer_b) ≥ θ
          backward: scorer(q + answer_b, q + answer_a) ≥ θ

        The min of the two scores is the conservative estimate.
        """
        doc_a = f"{context} {answer_a}".strip()
        doc_b = f"{context} {answer_b}".strip()

        if scorer == "minicheck":
            scores = cls.score_minicheck(
                contexts=[doc_a, doc_b],
                claims=[doc_b, doc_a],
            )
        elif scorer == "ruroberta":
            scores = cls.score_ruRoberta(
                contexts=[doc_a, doc_b],
                claims=[doc_b, doc_a],
            )
        elif scorer == "alignscore":
            scores = cls.score_alignscore(
                contexts=[doc_a, doc_b],
                claims=[doc_b, doc_a],
            )
        else:
            return False

        if not scores or len(scores) < 2:
            return False

        theta = {
            "minicheck": cls.THETA_MINICHECK,
            "ruroberta": cls.THETA_RUROBERTA,
            "alignscore": cls.THETA_ALIGN,
        }[scorer]
        return scores[0] >= theta and scores[1] >= theta

    @classmethod
    def are_equivalent_en(
        cls, context: str, answer_a: str, answer_b: str
    ) -> bool:
        """
        Check Open-QA equivalence for English using MiniCheck (bidirectional).
        """
        return cls._are_equivalent_bidirectional(
            context, answer_a, answer_b, scorer="minicheck"
        )

    @classmethod
    def are_equivalent_ru(
        cls, context: str, answer_a: str, answer_b: str
    ) -> bool:
        """
        Check Open-QA equivalence for Russian using ruRoberta (bidirectional).
        """
        return cls._are_equivalent_bidirectional(
            context, answer_a, answer_b, scorer="ruroberta"
        )

    @classmethod
    def are_equivalent(
        cls,
        context: str,
        answer_a: str,
        answer_b: str,
        language: str = "en",
    ) -> bool:
        """
        Check Open-QA equivalence using the appropriate backend.
        """
        if language.lower() in ("en", "eng", "english"):
            return cls.are_equivalent_en(context, answer_a, answer_b)
        elif language.lower() in ("ru", "rus", "russian"):
            return cls.are_equivalent_ru(context, answer_a, answer_b)
        else:
            # Unknown language: try MiniCheck, fall back to AlignScore only if
            # MiniCheck is unavailable. If both are unavailable, propagate.
            try:
                return cls._are_equivalent_bidirectional(
                    context, answer_a, answer_b, scorer="minicheck"
                )
            except OpenQABackendUnavailable:
                return cls._are_equivalent_bidirectional(
                    context, answer_a, answer_b, scorer="alignscore"
                )
