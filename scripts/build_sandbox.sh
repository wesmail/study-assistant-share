#!/usr/bin/env bash
# Builds the image that runs model-written code, TeX and pandoc. Needs Docker and a
# network, once; running the app needs neither the network nor this script again.
#
#     scripts/build_sandbox.sh
#
# The context is sandbox/ alone, so nothing else in the project can reach the image.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
IMAGE=$(PYTHONPATH=backend .venv/bin/python -c \
    'from app.config import load_settings as s; print(s().sandbox_image)' 2>/dev/null)
docker build --tag "$IMAGE" sandbox
echo "Built $IMAGE"
