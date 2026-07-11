#!/bin/bash
set -e

# Seed-copy: populate volume from image on first run
# Docker volume overwrites image content at mount point on first mount.
# Data downloaded in Dockerfile lives in /opt/seeds/ (not affected by volume).
# Copy to /app/cache/ (volume) if volume is empty. On subsequent runs,
# volume already contains data — copy is skipped.
seed_copy() {
    local src="$1"
    local dst="$2"
    local name="$3"
    if [ -d "$src" ]; then
        if [ -z "$(ls -A "$dst" 2>/dev/null)" ]; then
            echo "=== Seeding $name: $src -> $dst (first run) ==="
            cp -a "$src/." "$dst/"
            echo "    $name seeded ($(du -sh "$dst" | cut -f1))"
        else
            echo "=== $name: already present in volume ($dst), skip seed ==="
        fi
    fi
}

# Merge-seed: add resources bundled in a newer image that are MISSING from an
# existing volume, without clobbering what is already there (no-clobber). The
# all-or-nothing seed_copy above skips entirely when the volume is non-empty, so
# a resource added in a later image (e.g. NLTK punkt_tab for MiniCheck) would
# never land on a pre-existing volume. Idempotent; safe on first run too.
seed_merge() {
    local src="$1"
    local dst="$2"
    local name="$3"
    if [ -d "$src" ]; then
        echo "=== Merge-seeding $name: $src -> $dst (no-clobber) ==="
        cp -an "$src/." "$dst/"
        echo "    $name present ($(du -sh "$dst" | cut -f1))"
    fi
}

mkdir -p /app/cache/nltk_data /app/cache/stanza_resources
seed_merge /opt/seeds/nltk_data /app/cache/nltk_data "NLTK data"
seed_copy /opt/seeds/stanza_resources /app/cache/stanza_resources "Stanza models"

echo "=== Preloading/verifying models (best-effort; never blocks worker start) ==="
# Bounded + non-fatal: model_cache is a one-shot verifier/warmer. It must NEVER
# gate the worker — a hang or non-zero exit here (e.g. a heavy L3 smoke stalling
# on CUDA/DataLoader teardown) would otherwise abort the entrypoint under `set -e`
# and leave every task stuck PENDING. `timeout` bounds a pathological stall;
# `|| echo` neutralises set -e so we always fall through to `exec celery`.
timeout 600 python -m src.core.services.model_cache \
    || echo "WARN: model preload/verify failed or timed out (exit $?) — starting worker anyway"

echo "=== Starting Celery worker ==="
exec "$@"
