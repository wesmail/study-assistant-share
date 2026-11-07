"""Check the claims blocks of an earlier run again, with the checker as it is now.

    .venv/bin/python eval/recheck_claims.py --run dev-checks-fast-1 [--save]

For tuning sandbox/verify_claims.py without a new image: the working copy of the script
goes into the container's /work beside the block and runs there, so the blocks (model
text) are still read only inside the sandbox. --save writes the new reports into the
run's records, keeping the old one under checks_before.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_settings  # noqa: E402
from app.tools.sandbox import give_nonce, run_container, work_folder  # noqa: E402
from app.verify import Report, read_report, says_none  # noqa: E402

RESULTS = ROOT / "eval" / "results"
CHECKER = ROOT / "sandbox" / "verify_claims.py"


async def recheck(settings, block: str, source: str) -> Report:
    if says_none(block):
        return Report(source=source, block=block)
    with work_folder() as work:
        (work / "checks.txt").write_text(block[:settings.verify_block_chars], encoding="utf-8")
        for script in (CHECKER, CHECKER.with_name("safe_sympify.py")):      # this checkout's checker
            (work / script.name).write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
        nonce = give_nonce(work)
        done = await run_container(settings, work, ["timeout", "--signal=KILL", str(settings.verify_seconds),
                                                    "python", "/work/verify_claims.py"], settings.verify_seconds)
    report = read_report(done.stdout, nonce)
    return report.model_copy(update={"source": source, "block": block, "seconds": round(done.seconds, 2)})


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="answers-RUN.jsonl")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    path = RESULTS / f"answers-{args.run}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        old = record.get("checks") or {}
        if not old.get("block"):
            print(f"{record['id']}: no block")
            continue
        new = await recheck(settings, old["block"], old["source"])
        new = new.model_copy(update={key: old.get(key) for key in ("extraction_seconds", "extraction_cost_usd")
                                     if old.get(key) is not None})
        before = Report.model_validate(old).summary()
        after = new.summary()
        print(f"{record['id']}: {before['badge']}  ->  {after['badge']}   "
              + ", ".join(f"{k} {v}" for k, v in after.items() if k not in ("badge", "total", "passed") and v))
        for claim in new.claims:
            print(f"    {claim.verdict_text():40s} {claim.line[:78]}  | {claim.note[:100]}")
        if args.save:
            record.setdefault("checks_before", old)
            record["checks"] = new.model_dump()
            record["checks_summary"] = after
    if args.save:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
