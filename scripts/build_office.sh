#!/usr/bin/env bash
# Builds the LibreOffice image that renders Word and PowerPoint files and reads decks
# (roadmap item 7). Needs Docker and a network, once; running needs neither.
#
#     scripts/build_office.sh
#
# The context is office/ alone, so nothing else in the project can reach the image, and
# no font from the reader's machine is ever copied into it (they are mounted read-only
# at run time).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
IMAGE=$(PYTHONPATH=backend .venv/bin/python -c \
    'from app.config import load_settings as s; print(s().office_image)' 2>/dev/null)
docker build --tag "$IMAGE" office
echo "Built $IMAGE"
