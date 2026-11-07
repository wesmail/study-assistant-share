"""Score the checker on the claim test set (verification layer, gate item 1).

    .venv/bin/python eval/check_claim_testset.py                  the checker in the image
    .venv/bin/python eval/check_claim_testset.py --working-copy   sandbox/verify_claims.py as it is now
    ... --save NAME                                               eval/results/claim-testset-NAME.json
    ... --set eval/claim_testset_heldout.yaml                     another set in the same format

Every block runs in the sandbox, one container per item, like an answer's block. Each block
ends with the one claim under test, and that last claim is what's scored. Gate: no true
claim marked failed; at least 90% of the wrong claims marked failed; at most 20% of all
claims undecided, where ≈ (numerically_equal) and error count as undecided.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_settings  # noqa: E402
from app.verify import check_block  # noqa: E402
from recheck_claims import recheck  # noqa: E402

TESTSET = ROOT / "eval" / "claim_testset.yaml"
RESULTS = ROOT / "eval" / "results"
UNDECIDED = {"undecided", "numerically_equal", "error"}


def score(rows: list[dict]) -> dict:
    true = [r for r in rows if r["truth"]]
    wrong = [r for r in rows if not r["truth"]]
    caught = sum(r["verdict"] == "failed" for r in wrong)
    undecided = sum(r["verdict"] in UNDECIDED for r in rows)
    result = {
        "claims": len(rows), "true": len(true), "wrong": len(wrong),
        "true_marked_failed": [r["id"] for r in true if r["verdict"] == "failed"],
        "wrong_marked_failed": caught,
        "wrong_caught_share": round(caught / len(wrong), 3) if wrong else None,
        "wrong_marked_passed": [r["id"] for r in wrong if r["verdict"] == "passed"],
        "true_marked_passed": sum(r["verdict"] == "passed" for r in true),
        "undecided": [r["id"] for r in rows if r["verdict"] in UNDECIDED],
        "undecided_share": round(undecided / len(rows), 3) if rows else None,
        "verdicts_true": dict(Counter(r["verdict"] for r in true)),
        "verdicts_wrong": dict(Counter(r["verdict"] for r in wrong)),
        "extra_claims": [r["id"] for r in rows if r["claims_in_block"] != 1],
    }
    result["gate"] = {
        "no_true_claim_failed": not result["true_marked_failed"],
        "wrong_caught_at_least_90": bool(wrong) and caught / len(wrong) >= 0.9,
        "undecided_at_most_20": bool(rows) and undecided / len(rows) <= 0.2,
    }
    result["gate_met"] = all(result["gate"].values())
    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-copy", action="store_true", help="run sandbox/verify_claims.py as it is now")
    parser.add_argument("--save", metavar="NAME")
    parser.add_argument("--set", type=Path, default=TESTSET)
    args = parser.parse_args()
    settings = load_settings()
    items = yaml.safe_load(args.set.read_text(encoding="utf-8"))["items"]
    rows = []
    for item in items:
        block = item["block"].strip()
        report = await (recheck(settings, block, "answer") if args.working_copy
                        else check_block(settings, block, "answer"))
        if report.failure or not report.claims:
            verdict, text, note = "error", "error", report.failure or "no claim in the block"
        else:
            claim = report.claims[-1]
            verdict, text, note = claim.verdict, claim.verdict_text(), claim.note
        expected_ok = (verdict == "passed") if item["truth"] else (verdict == "failed")
        mark = "ok " if expected_ok else ("BAD" if (verdict == "failed") == item["truth"] or
                                          (verdict == "passed" and not item["truth"]) else " ? ")
        print(f"{mark} {item['id']:4s} {'true ' if item['truth'] else 'wrong'} {text:40s} "
              f"{block.splitlines()[-1][:70]}")
        if not expected_ok:
            print(f"       note: {note[:160]}")
        rows.append({"id": item["id"], "question": item["question"], "truth": item["truth"],
                     "slip": item.get("slip"), "verdict": verdict, "shown": text, "note": note,
                     "basis": report.claims[-1].basis if report.claims else [],
                     "claims_in_block": len(report.claims), "seconds": report.seconds})
    result = score(rows)
    print()
    for key in ("claims", "true", "wrong", "true_marked_failed", "wrong_marked_failed", "wrong_caught_share",
                "wrong_marked_passed", "true_marked_passed", "undecided", "undecided_share",
                "verdicts_true", "verdicts_wrong", "extra_claims", "gate", "gate_met"):
        print(f"{key}: {result[key]}")
    if args.save:
        path = RESULTS / f"claim-testset-{args.save}.json"
        path.write_text(json.dumps({"score": result, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"saved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
