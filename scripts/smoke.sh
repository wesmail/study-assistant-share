#!/usr/bin/env bash
# A free check that the install works: no keys, no hosted model, nothing of yours touched.
#
#     scripts/smoke.sh
#
# In a scratch folder, with your config but its own library and data, it writes a small PDF
# and a two-slide deck, ingests them with the real worker, asks Fast one question that must
# be answered from page 2 and cite it, runs Python and SymPy in the sandbox, and checks
# SearXNG (a warning only). Each failure says how to fix it. A few minutes, most of it
# loading models; run it while nothing else is using the GPU. The scratch folder is removed
# when it passes and kept, for a look, when it fails.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "FAIL  there is no virtualenv at .venv"
    echo "      Fix: INSTALL.md, the Python environment"
    exit 1
fi

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/study-smoke.XXXXXX")
# Hugging Face stays offline: a missing parser model fails here instead of starting a download.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
if PYTHONPATH=backend "$PYTHON" -W ignore scripts/smoke.py "$SCRATCH"; then
    rm -rf "$SCRATCH"
else
    status=$?
    echo "The scratch folder is kept for a look: $SCRATCH"
    exit $status
fi
