#!/usr/bin/env bash
# Starts the study assistant in this terminal; Ctrl+C stops it.
#
#     scripts/start.sh
#
# It rebuilds the frontend when its source has changed, then starts the web server,
# which runs the ingestion worker beside it (app/supervisor.py). On startup the worker
# scans library/ and queues new or changed PDFs; then its watcher picks up files as they
# arrive. Ingestion output is prefixed with [worker]. Ctrl+C stops both; a job that was
# running resumes at the next start.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="$PWD/.venv/bin/python"
export PATH="$PWD/.venv/bin:$PATH"

fail() {
    echo "Can't start: $1" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "there is no virtualenv at .venv (see backend/requirements.txt)."

read -r PORT OLLAMA < <(PYTHONPATH=backend "$PYTHON" -c \
    'from app.config import load_settings as s; print(s().port, s().ollama_url)' 2>/dev/null)

curl --silent --fail --max-time 3 "$OLLAMA/api/tags" >/dev/null \
    || fail "Ollama isn't answering at $OLLAMA. Start it with: sudo systemctl start ollama"

if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    fail "port $PORT is already in use. Is the app already running? Try http://localhost:$PORT"
fi

BUILT=frontend/dist/index.html
if [[ ! -f $BUILT ]] || [[ -n $(find frontend/src frontend/index.html frontend/package.json -newer "$BUILT" -print -quit) ]]; then
    echo "Building the frontend..."
    [[ -d frontend/node_modules ]] || npm --prefix frontend ci --no-fund --no-audit
    npm --prefix frontend run build
fi

read -r IMAGE OFFICE < <(PYTHONPATH=backend "$PYTHON" -c \
    'from app.config import load_settings as s; print(s().sandbox_image, s().office_image)' 2>/dev/null)
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "No sandbox image: PDF, Word, PowerPoint, code and SymPy tools won't work until you run scripts/build_sandbox.sh."
fi
if ! docker image inspect "$OFFICE" >/dev/null 2>&1; then
    echo "No office image: decks can't be ingested, and Word and PowerPoint files have no preview, until you run scripts/build_office.sh."
fi

# Web search (Phase 5d) needs SearXNG; it keeps running after Ctrl+C: scripts/searxng.sh stop.
if ! scripts/searxng.sh start >/dev/null 2>&1; then
    echo "SearXNG isn't running, so web search won't work. scripts/searxng.sh start says why."
fi

# Backups (scripts/backup.sh): one in the background when the newest is older than
# backup_max_age_hours, in its own session so Ctrl+C here doesn't cut it short.
if DUE=$(PYTHONPATH=backend "$PYTHON" -m app.backup due 2>/dev/null); then
    LOG=$(PYTHONPATH=backend "$PYTHON" -c \
        'from app.config import load_settings as s; print(s().backup_local_dir / "backup.log")')
    mkdir -p "$(dirname "$LOG")"
    echo "$DUE Backing up in the background (log: $LOG)."
    echo "== $(date -Is) from start.sh" >>"$LOG"
    setsid -f nice -n 10 scripts/backup.sh create --if-older >>"$LOG" 2>&1 </dev/null
fi

echo "Study assistant: http://localhost:$PORT, with the library worker (Ctrl+C stops both)"
cd backend
exec "$PYTHON" -m app.main
