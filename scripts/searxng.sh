#!/usr/bin/env bash
# The SearXNG container behind the web_search tool.
#
#     scripts/searxng.sh start     create it the first time, or start it again
#     scripts/searxng.sh stop
#     scripts/searxng.sh status
#
# It listens on 127.0.0.1 only, so nothing outside this machine can use it. The image is
# pinned; pull it first with: docker pull searxng/searxng:<tag below>. A new secret key is
# made when the container is created; it never leaves the container.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
IMAGE="searxng/searxng:2026.10.19-81371861f"
NAME="study-assistant-searxng"
PORT="${SEARXNG_PORT:-8888}"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == "true" ]]; }

case "${1:-status}" in
start)
    if running; then
        echo "SearXNG is running: http://127.0.0.1:$PORT"
        exit 0
    fi
    if docker inspect "$NAME" >/dev/null 2>&1; then
        docker start "$NAME" >/dev/null
    else
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            echo "No SearXNG image: run docker pull $IMAGE" >&2
            exit 1
        fi
        docker run -d --name "$NAME" \
            -p "127.0.0.1:$PORT:8080" \
            -v "$PWD/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
            -e SEARXNG_SECRET="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" \
            -e SEARXNG_BASE_URL="http://127.0.0.1:$PORT/" \
            --cap-drop ALL --cap-add CHOWN --cap-add SETGID --cap-add SETUID \
            --memory 512m \
            "$IMAGE" >/dev/null
    fi
    for _ in $(seq 1 30); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null; then
            echo "SearXNG is running: http://127.0.0.1:$PORT"
            exit 0
        fi
        sleep 1
    done
    echo "SearXNG didn't answer within 30 s: docker logs $NAME" >&2
    exit 1
    ;;
stop)
    docker stop "$NAME" >/dev/null 2>&1 || true
    echo "SearXNG stopped"
    ;;
status)
    if running; then echo "running"; else echo "stopped"; fi
    ;;
*)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
