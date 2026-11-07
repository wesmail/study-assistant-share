"""Run each dev question's reference_check in the sandbox and report ok or not.

    .venv/bin/python eval/check_dev_references.py [--id dev-deriv-03 ...]

The reference values in eval/derivation_dev.yaml are what the claim checker is measured
against, so they are checked themselves first. Each check runs in its own sandbox
container (no network, only a new empty /work), like every other piece of SymPy here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_settings  # noqa: E402
from app.tools.sandbox import run_container, work_folder  # noqa: E402

QUESTIONS = ROOT / "eval" / "derivation_dev.yaml"
SECONDS = 60
FOOTER = '\nimport json as _json\nprint(_json.dumps({"ok": bool(ok)}))\n'


async def check(settings, question: dict) -> tuple[bool | None, str]:
    with work_folder() as work:
        (work / "check.py").write_text(question["reference_check"] + FOOTER, encoding="utf-8")
        done = await run_container(
            settings, work, ["timeout", "--signal=KILL", str(SECONDS), "python", "/work/check.py"], SECONDS)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return None, (done.stderr or done.stdout).strip()[-400:]
    return json.loads(lines[-1])["ok"], f"{done.seconds:.1f} s"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", nargs="*", help="only these questions")
    args = parser.parse_args()
    settings = load_settings()
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    failed = 0
    for question in questions:
        if args.id and question["id"] not in args.id:
            continue
        ok, note = await check(settings, question)
        failed += ok is not True
        print(f"{question['id']}: {'ok' if ok else 'FAILED' if ok is False else 'ERROR'}  {note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
