"""Gate item 2 of the verification layer: are an answer's checked claims what the answer says?

    .venv/bin/python eval/faithfulness.py pack --name NAME --runs RUN1 RUN2 ... --seed S [--batches 3]
    .venv/bin/python eval/faithfulness.py score --name NAME

`pack` writes eval/grading/NAME/ (gitignored): for every answer shown in each run that has
checked claims, the question, the answer as the reader saw it and its claims, numbered.
Items get random ids, shuffled with the seed, so graders can't tell runs apart; key.json
holds the mapping. The quality graders (run_eval.py blind-pack) never see the claims; these
graders see nothing but this. `score` unblinds the grades: the share of claims judged
faithful, per run and in all.

A claim is faithful when the answer states it, as written, whether it's true or not: the
same quantity with the same formula or value. Unfaithful: the answer doesn't state it, or
states something else (a changed value, a correction, an added step).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
GRADING = ROOT / "eval" / "grading"

BRIEF = """You are checking whether lists of claims match the answers they were taken from. Each item
below is a study question, an answer to it as a reader saw it, and numbered claims written in
SymPy syntax for a checker. Items come from different sources in a shuffled order; don't try to
tell them apart. Read this whole file and no other file.

For each numbered claim, decide whether it is FAITHFUL to the answer:
- faithful: the answer states this claim, as written: the same quantity with the same formula or
  the same value (rounding as the answer gives it is fine, and so are names chosen for the
  answer's symbols). Whether the claim is TRUE does not matter: a wrong result the answer states
  is faithful. Lines the claim relies on (let, assume, definitions) are shown for context.
- unfaithful: the answer doesn't state it, or states something different: a changed value or
  formula, a correction of the answer, a step or result the answer doesn't contain.

Write valid JSON to {grades}, one entry per item:
[{{"item": "...", "claims": [{{"n": 1, "faithful": true, "note": "short, for unfaithful ones: what differs"}}, ...]}}]
This batch has {count} items."""


def load_run(name: str) -> dict[str, dict]:
    path = RESULTS / f"answers-{name}.jsonl"
    return {record["id"]: record for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def item_text(item_id: str, record: dict) -> str:
    claims = record["checks"]["claims"]
    numbered = "\n".join(f"{n}. {claim['line']}" for n, claim in enumerate(claims, 1))
    return (f"=== ITEM {item_id}\nQUESTION: {record['question']}\n\nANSWER:\n{record['answer']}\n\n"
            f"ALL LINES OF THE CLAIMS LIST, AS WRITTEN:\n{record['checks'].get('block') or ''}\n\n"
            f"CLAIMS TO JUDGE:\n{numbered}\n=== END OF ITEM {item_id}")


def cmd_pack(args) -> None:
    folder = GRADING / args.name
    if folder.exists():
        raise SystemExit(f"{folder} exists: a new pack needs a new name")
    rng = random.Random(args.seed)
    items, key = [], {}
    for run in args.runs:
        for question_id, record in load_run(run).items():
            if not (record.get("checks") or {}).get("claims"):
                continue
            item_id = f"{rng.randrange(16**6):06x}"
            while item_id in key:
                item_id = f"{rng.randrange(16**6):06x}"
            key[item_id] = {"run": run, "question": question_id, "claims": len(record["checks"]["claims"]),
                            "source": record["checks"]["source"]}
            items.append((item_id, item_text(item_id, record)))
    rng.shuffle(items)
    folder.mkdir(parents=True)
    for batch in range(1, args.batches + 1):
        chosen = items[batch - 1::args.batches]
        brief = BRIEF.format(grades=folder / f"grades-{batch}.json", count=len(chosen))
        (folder / f"batch-{batch}.txt").write_text(brief + "\n\n" + "\n\n".join(text for _, text in chosen),
                                                    encoding="utf-8")
    (folder / "key.json").write_text(json.dumps({"runs": args.runs, "seed": args.seed, "items": key}, indent=1))
    print(f"{len(items)} items, {sum(entry['claims'] for entry in key.values())} claims, "
          f"in {args.batches} batches in {folder}")


def cmd_score(args) -> None:
    folder = GRADING / args.name
    key = json.loads((folder / "key.json").read_text())
    grades = {entry["item"]: entry for path in sorted(folder.glob("grades-*.json"))
              for entry in json.loads(path.read_text(encoding="utf-8"))}
    missing = sorted(set(key["items"]) - set(grades))
    if missing:
        raise SystemExit(f"no grades for {missing}")
    per_run: dict[str, Counter] = {run: Counter() for run in key["runs"]}
    unfaithful = []
    for item_id, entry in key["items"].items():
        judged = grades[item_id]["claims"]
        if len(judged) != entry["claims"]:
            raise SystemExit(f"{item_id}: {len(judged)} claims judged, {entry['claims']} given")
        for claim in judged:
            per_run[entry["run"]]["faithful" if claim["faithful"] else "unfaithful"] += 1
            if not claim["faithful"]:
                unfaithful.append({"run": entry["run"], "question": entry["question"], "n": claim["n"],
                                   "source": entry["source"], "note": claim.get("note", "")})
    total = sum(per_run.values(), Counter())
    summary = {run: {**counts, "share": round(counts["faithful"] / max(1, sum(counts.values())), 3)}
               for run, counts in per_run.items()}
    summary["all"] = {**total, "share": round(total["faithful"] / max(1, sum(total.values())), 3)}
    for run, counts in summary.items():
        print(f"{run}: {counts}")
    for entry in unfaithful:
        print(f"  unfaithful: {entry}")
    out = RESULTS / f"faithfulness-{args.name}.json"
    out.write_text(json.dumps({"summary": summary, "unfaithful": unfaithful, "key": key}, indent=1) + "\n")
    print(f"saved {out.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--name", required=True)
    pack.add_argument("--runs", nargs="+", required=True)
    pack.add_argument("--seed", type=int, required=True)
    pack.add_argument("--batches", type=int, default=3)
    pack.set_defaults(run=cmd_pack)
    score = commands.add_parser("score")
    score.add_argument("--name", required=True)
    score.set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
