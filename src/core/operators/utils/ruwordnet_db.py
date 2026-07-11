# src/core/operators/utils/ruwordnet_db.py
"""Production-ready provisioning for the RuWordNet database.

The ``ruwordnet`` package looks for its DB inside its own (ephemeral) ``site-
packages/ruwordnet/static`` directory, which is lost on every venv re-create and
is not reproducible across machines / CI / Docker. This module relocates the DB
to a **project-managed cache** that mirrors the existing model-cache convention
(``HF_CACHE_DIR`` / ``MINICHECK_CACHE_DIR``): a git-ignored ``./cache/`` path,
env-overridable, auto-provisioned on first use.

Kept deliberately dependency-light (pathlib + settings + logger + lazy urllib)
so it can be imported and unit-tested WITHOUT pulling in the heavy NLP stack
(stanza / pymorphy3 / torch) that ``nlp_utils`` carries.
"""

from pathlib import Path

from src.config.settings import get_settings
from src.utils.logger import logger


def resolve_ruwn_db_path() -> Path:
    """Project-managed RuWordNet DB path (``RUWORDNET_DB_PATH`` setting/env)."""
    return Path(get_settings().RUWORDNET_DB_PATH)


def provision_ruwn_db(path: Path) -> None:
    """Ensure the RuWordNet DB exists at ``path`` (idempotent, atomic download).

    Auto-downloads on first use (same UX as HF/MiniCheck). The download writes to
    a ``.part`` temp file and is renamed into place only on success, so an
    interrupted download never leaves a corrupt DB that looks present.

    Disabled by ``RUWORDNET_AUTO_DOWNLOAD=0`` (hermetic / offline CI): a missing
    DB then raises an actionable error instead of attempting network access.
    """
    if path.exists():
        return
    settings = get_settings()
    if not settings.RUWORDNET_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            f"RuWordNet DB not found at '{path}' and auto-download is disabled "
            f"(RUWORDNET_AUTO_DOWNLOAD=0). Provision it by setting "
            f"RUWORDNET_DB_PATH to a pre-downloaded copy, or enable auto-download."
        )
    url = settings.RUWORDNET_DB_URL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.info(f"Provisioning RuWordNet DB (first use): {url} → {path}")
    try:
        import urllib.request

        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)  # only a complete download lands at the final path
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to provision RuWordNet DB from '{url}': {e}. Set "
            f"RUWORDNET_DB_PATH to a pre-downloaded copy or retry with network."
        ) from e
    logger.info(f"RuWordNet DB ready at {path}")
