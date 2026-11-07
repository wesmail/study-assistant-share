"""What the builders share: the work folder, running tools, and TeX's error reports."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("SANDBOX_WORK", "/work"))
OUT = WORK / "out"
ONLINE = bool(os.environ.get("SANDBOX_BAKING"))     # only while the image is being built

MISSING = re.compile(r"Missing character: There is no (.+?) in font", re.I)
TEX_ERROR = re.compile(r"^(?:error|!)[: ]\s*(.+)$", re.M)
LINE_ERROR = re.compile(r"\.tex:(\d+):\s*(.+)")
MAX_ERRORS = 6


def finish(**result) -> None:
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)


def run(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd or WORK, capture_output=True, text=True, errors="replace")


def tectonic(tex: str) -> tuple[subprocess.CompletedProcess, str]:
    """Compile with only what's baked into the image. Returns the run and its log."""
    argv = ["tectonic", "--untrusted", "--keep-logs", "--chatter", "minimal"]
    done = run(argv + ([] if ONLINE else ["--only-cached"]) + [tex])
    log = WORK / Path(tex).with_suffix(".log")
    return done, log.read_text(errors="replace") if log.exists() else ""


def tex_errors(done: subprocess.CompletedProcess, log: str) -> str:
    found = TEX_ERROR.findall(done.stderr) + TEX_ERROR.findall(log)
    unique = list(dict.fromkeys(line.strip() for line in found))
    return "; ".join(unique[:MAX_ERRORS]) or done.stderr.strip()[-400:] or "TeX failed"


def missing_characters(log: str) -> list[str]:
    return sorted(set(MISSING.findall(log)))
