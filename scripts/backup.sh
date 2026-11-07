#!/usr/bin/env bash
# Backups of the study assistant's data (backend/app/backup.py says what is in one).
#
#     scripts/backup.sh create              # one now
#     scripts/backup.sh list                # what there is, and what retention keeps
#     scripts/backup.sh verify [ARCHIVE]    # check one end to end (the newest by default)
#     scripts/backup.sh copy /mnt/e/backups # the newest to an external drive, checked there
#     scripts/backup.sh restore ARCHIVE --to DIR [--env]
#     scripts/backup.sh install-timer       # daily, through a systemd user timer
#     scripts/backup.sh remove-timer
#
# WSL stops when idle, so the timer is Persistent: a day it missed runs when WSL next
# starts. start.sh also starts one when the newest is older than backup_max_age_hours.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="$PWD/.venv/bin/python"
UNITS="$HOME/.config/systemd/user"
NAME=study-assistant-backup

case "${1:-}" in
install-timer)
    mkdir -p "$UNITS"
    cat > "$UNITS/$NAME.service" <<EOF
[Unit]
Description=Study assistant backup (scripts/backup.sh)

[Service]
Type=oneshot
ExecStart=$PWD/scripts/backup.sh create --if-older
Nice=10
IOSchedulingClass=idle
EOF
    cat > "$UNITS/$NAME.timer" <<EOF
[Unit]
Description=Daily study assistant backup

[Timer]
OnCalendar=*-*-* 12:30
Persistent=true

[Install]
WantedBy=timers.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now "$NAME.timer"
    systemctl --user list-timers "$NAME.timer" --no-pager
    echo "Its log: journalctl --user -u $NAME.service"
    ;;
remove-timer)
    systemctl --user disable --now "$NAME.timer" || true
    rm -f "$UNITS/$NAME.service" "$UNITS/$NAME.timer"
    systemctl --user daemon-reload
    echo "Timer removed; the backups themselves are untouched."
    ;;
*)
    PYTHONPATH=backend exec "$PYTHON" -m app.backup "$@"
    ;;
esac
