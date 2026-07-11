"""Global singleton cache for AI models used in variation validation.
Models are loaded once and shared across all VariationValidator instances.
This avoids redundant model loading when multiple validators are created.
Usage:
    from src.core.services.model_cache import model_cache
    model_cache.preload_all()
Or via CLI for CI/CD:
    python -m src.core.services.model_cache
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception as _e:  # pragma: no cover - optional dep
    logging.getLogger(__name__).error(f"torch import failed: {_e}", exc_info=True)
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as _e:  # pragma: no cover - optional dep
    logging.getLogger(__name__).error(
        f"sentence-transformers import failed: {_e}", exc_info=True
    )
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception as _e:  # pragma: no cover - optional dep
    logging.getLogger(__name__).error(
        f"transformers.pipeline import failed: {_e}", exc_info=True
    )
    pipeline = None

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelCache:
    """Thread-safe singleton cache for AI models."""

    _instance: Optional["ModelCache"] = None
    _nli_pipelines: Dict[str, list] = {}
    _embedding_model: Optional[SentenceTransformer] = None
    _sentiment_classifiers: Dict[str, Any] = {}

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_nli_pipelines(self, language: str) -> List:
        """Get or create NLI pipelines for a language."""
        if language not in self._nli_pipelines:
            if pipeline is None or torch is None:
                logger.error(
                    "transformers/torch unavailable; no NLI pipelines loaded for "
                    "'%s'. Tier B/C validation will fail-close (REJECT).",
                    language,
                )
                self._nli_pipelines[language] = []
                return self._nli_pipelines[language]
            model_names = (
                (settings.RU_NLI_MODEL_PRIMARY, settings.RU_NLI_MODEL_SECONDARY)
                if language == "ru"
                else (settings.EN_NLI_MODEL_PRIMARY, settings.EN_NLI_MODEL_SECONDARY)
            )
            pipes = []
            device = 0 if torch.cuda.is_available() else -1
            for name in model_names:
                try:
                    logger.info(f"Loading NLI model: {name}")
                    pipes.append(
                        pipeline("text-classification", model=name, device=device)
                    )
                    logger.info(f"  OK: {name} loaded on device={device}")
                except Exception as e:
                    logger.error(f"  FAIL: {name}: {e}", exc_info=True)
            self._nli_pipelines[language] = pipes
            logger.info(
                f"  NLI ({language}): {len(pipes)}/{len(model_names)} pipelines loaded"
            )
        return self._nli_pipelines[language]

    def get_embedding_model(self) -> Optional[SentenceTransformer]:
        """Get or create the embedding model (LaBSE)."""
        if self._embedding_model is None:
            if SentenceTransformer is None:
                logger.warning(
                    "sentence-transformers unavailable; embedding cosine check disabled."
                )
                return None
            try:
                logger.info("Loading LaBSE embedding model")
                self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"  OK: LaBSE loaded ({settings.EMBEDDING_MODEL})")
            except Exception as e:
                logger.error(f"  FAIL: LaBSE: {e}", exc_info=True)
                self._embedding_model = None
        return self._embedding_model

    def get_sentiment_classifier(self, language: str) -> Any:
        """Get or create sentiment classifier for a language."""
        if language not in self._sentiment_classifiers:
            if pipeline is None or torch is None:
                logger.warning(
                    "transformers/torch unavailable; sentiment classifier disabled "
                    "for '%s' (Layer 3 tone check falls back to keyword heuristic).",
                    language,
                )
                self._sentiment_classifiers[language] = None
                return None
            model_name = (
                settings.SENTIMENT_MODEL_RU
                if language == "ru"
                else settings.SENTIMENT_MODEL_EN
            )
            try:
                device = 0 if torch.cuda.is_available() else -1
                self._sentiment_classifiers[language] = pipeline(
                    "text-classification",
                    model=model_name,
                    device=device,
                )
                logger.info(f"  OK: {model_name} loaded on device={device}")
            except Exception as e:
                logger.warning(f"  FAIL: {model_name}: {e}", exc_info=True)
                self._sentiment_classifiers[language] = None
        return self._sentiment_classifiers.get(language)

    def preload_all(self) -> None:
        """Preload all models + NLP tools (for CI/CD or startup).

        NLTK and stanza models are baked into the Docker image at build time
        (Dockerfile.celery downloads them to /opt/seeds/). The entrypoint
        seed-copies them to the volume (/app/cache/) on first run. Here we
        only verify they are resolvable — no runtime download needed.
        HF-backed models (NLI, LaBSE, sentiment) are loaded from cache or
        downloaded from huggingface.co (reachable on the runner).
        """
        import os

        logger.info("Preloading all validation models...")
        logger.info("  NLI (EN)...")
        self.get_nli_pipelines("en")
        logger.info("  NLI (RU)...")
        self.get_nli_pipelines("ru")
        logger.info("  Embedding (LaBSE)...")
        self.get_embedding_model()
        logger.info("  Sentiment (EN)...")
        self.get_sentiment_classifier("en")
        logger.info("  Sentiment (RU)...")
        self.get_sentiment_classifier("ru")

        # ── NLTK ─────────────────────────────────────────────────────
        # wordnet/punkt are baked into the image at build time
        # (Dockerfile.celery downloads to /usr/local/share/nltk_data).
        # Just verify they are resolvable; no runtime download needed.
        logger.info("  NLTK data...")
        try:
            import nltk

            wordnet_ok = True
            punkt_ok = True
            punkt_tab_ok = True
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                wordnet_ok = False
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                punkt_ok = False
            # punkt_tab: required by NLTK>=3.9 sent_tokenize (MiniCheck Open-QA
            # L3 EAR backend). Its absence degrades EAR to `unavailable_backend`.
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                punkt_tab_ok = False
            logger.info(
                f"  NLTK: wordnet={'OK' if wordnet_ok else 'MISSING'} "
                f"punkt={'OK' if punkt_ok else 'MISSING'} "
                f"punkt_tab={'OK' if punkt_tab_ok else 'MISSING'}"
            )
            if not (wordnet_ok and punkt_ok and punkt_tab_ok):
                logger.warning(
                    "  NLTK: some packages missing. They should be baked into "
                    "the image (Dockerfile.celery). If running outside Docker, run: "
                    "python -c \"import nltk; nltk.download('wordnet'); "
                    "nltk.download('punkt'); nltk.download('punkt_tab')\""
                )
        except Exception as e:
            logger.error(f"  FAIL: NLTK check: {e}", exc_info=True)

        # ── Stanza ───────────────────────────────────────────────────
        # en/ru models are baked into the image (Dockerfile.celery downloads
        # to /opt/seeds/stanza_resources, entrypoint seed-copies to volume).
        # Just verify they are present; no runtime download needed.
        logger.info("  Stanza models...")
        try:
            stanza_dir = os.environ.get(
                "STANZA_RESOURCES_DIR", os.path.expanduser("~/stanza_resources")
            )
            en_ok = os.path.isdir(os.path.join(stanza_dir, "en"))
            ru_ok = os.path.isdir(os.path.join(stanza_dir, "ru"))
            logger.info(
                f"  Stanza: en={'OK' if en_ok else 'MISSING'} "
                f"ru={'OK' if ru_ok else 'MISSING'} (dir={stanza_dir})"
            )
            if not (en_ok and ru_ok):
                logger.warning(
                    "  Stanza: models missing. They should be baked into "
                    "the image (Dockerfile.celery). If running outside Docker, "
                    "run: python -c \"import stanza; stanza.download('en'); stanza.download('ru')\""
                )
        except Exception as e:
            logger.error(f"  FAIL: Stanza check: {e}", exc_info=True)

        # ── Open-QA Layer-3 EAR backends (MiniCheck EN / ruRoberta RU) ──────
        # These are NOT part of the validator model_cache, but are smoke-verified
        # here so a scoring-time dependency gap (e.g. the NLTK punkt_tab rename in
        # 3.9, which crashes MiniCheck) surfaces LOUDLY at deploy time instead of
        # silently degrading every open-QA EAR cell to `unavailable_backend`
        # mid-run. Non-fatal: fail-closed at runtime is by design (B3) — the point
        # is visibility. Loads the on-volume checkpoints; no network if cached.
        l3_en_ok = l3_ru_ok = False
        try:
            from src.core.services.open_qa_equivalence import OpenQAEquivalence

            logger.info("  Open-QA L3 (MiniCheck EN)...")
            en_score = OpenQAEquivalence.score_minicheck(
                ["Paris is the capital of France."],
                ["The capital of France is Paris."],
            )
            l3_en_ok = bool(en_score)
            logger.info(f"  OK: MiniCheck EN smoke score={en_score}")
        except Exception as e:
            logger.error(
                f"  FAIL: MiniCheck EN Open-QA L3 backend (EAR will be "
                f"unavailable_backend on EN open_qa): {e}",
                exc_info=True,
            )
        try:
            from src.core.services.open_qa_equivalence import OpenQAEquivalence

            logger.info("  Open-QA L3 (ruRoberta RU)...")
            ru_score = OpenQAEquivalence.score_ruRoberta(
                ["Париж — столица Франции."],
                ["Столица Франции — Париж."],
            )
            l3_ru_ok = bool(ru_score)
            logger.info(f"  OK: ruRoberta RU smoke score={ru_score}")
        except Exception as e:
            logger.error(
                f"  FAIL: ruRoberta RU Open-QA L3 backend (EAR will be "
                f"unavailable_backend on RU open_qa): {e}",
                exc_info=True,
            )

        logger.info("=== Preload summary ===")
        logger.info(f"  NLI EN:  {len(self._nli_pipelines.get('en', []))} pipelines")
        logger.info(f"  NLI RU:  {len(self._nli_pipelines.get('ru', []))} pipelines")
        logger.info(f"  LaBSE:   {'OK' if self._embedding_model is not None else 'FAIL'}")
        logger.info(
            f"  Sent EN: {'OK' if self._sentiment_classifiers.get('en') is not None else 'FAIL'}"
        )
        logger.info(
            f"  Sent RU: {'OK' if self._sentiment_classifiers.get('ru') is not None else 'FAIL'}"
        )
        logger.info(f"  L3 EN (MiniCheck): {'OK' if l3_en_ok else 'FAIL'}")
        logger.info(f"  L3 RU (ruRoberta): {'OK' if l3_ru_ok else 'FAIL'}")
        logger.info("All validation models loaded.")


# Global singleton instance
model_cache = ModelCache()
if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_cache.preload_all()
    # Force-exit once verification/warming is done. The L3 smoke loads MiniCheck
    # (internal DataLoader) + ruRoberta on CUDA, whose teardown can leave a
    # non-daemon thread that STALLS interpreter shutdown. This CLI is a one-shot
    # verifier run from the celery entrypoint BEFORE `exec celery`; if the process
    # hangs on exit, the entrypoint (set -e) blocks forever and the worker never
    # starts → tasks stuck PENDING. os._exit bypasses the hanging teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
