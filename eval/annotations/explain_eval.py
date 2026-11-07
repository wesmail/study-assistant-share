"""Gate item 4 of roadmap item 4: Explain, line by line.

    .venv/bin/python eval/annotations/explain_eval.py items          writes explain_items.yaml (once)
    ... run --option deep --name exp-deep-1 [--max-usd 0.7]
    ... pack --name exp-1 --runs exp-deep-1 exp-balanced-1 --seed S
    ... score --name exp-1

Ten items: five equations from the gold derivation questions' pages, as the book's LaTeX (what
the viewer sends for a selected equation), each with its page's text as context; and five code
blocks, UDL's PyTorch listing (p. 72) and four short snippets on the same topics. Each is
explained at the intermediate depth through app/explain.py, as the panel does. Graded blind:
good = every line explained and nothing wrong; weak = a part left vague or a small slip;
wrong = a wrong statement, or lines missing. Gate: Deep at least 8 of 10 good.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import fitz
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import explain  # noqa: E402
from app.config import load_settings  # noqa: E402

ITEMS = Path(__file__).with_name("explain_items.yaml")
RESULTS = ROOT / "eval" / "results"
GRADING = ROOT / "eval" / "grading"

EQUATIONS = [  # (gold question, document, page, a piece of the LaTeX that picks the block)
    ("deriv-004", "2606.10868v2.pdf", 7, r"\underbrace"),
    ("deriv-003", "2606.10868v2.pdf", 8, r"\text{SRR}"),
    ("book-udl-02", "Simon J. D. Prince - Understanding Deep Learning-The MIT Press (2024)-55-386.pdf", 32, r"\delta[y - y_{i}]"),
    ("book-udl-05", "Simon J. D. Prince - Understanding Deep Learning-The MIT Press (2024)-55-386.pdf", 312, r"\mathbf{z}_{1}}&"),
    ("book-rub-02", "André Rubbia - Phenomenology of Particle Physics-Cambridge University Press (2022)-pages-2.pdf", 16, r"M_{W}^{2}W_{\mu}^{+}"),
]

SNIPPETS = [
    ("backprop toy model (UDL 7.3)", """import numpy as np

def forward(x, phi):
    b0, w0, b1, w1, b2, w2, b3, w3 = phi
    f0 = b0 + w0 * x
    h1 = np.sin(f0)
    f1 = b1 + w1 * h1
    h2 = np.exp(f1)
    f2 = b2 + w2 * h2
    h3 = np.cos(f2)
    return b3 + w3 * h3, (f0, h1, f1, h2, f2, h3)

def grad_b0(x, y, phi):
    out, (f0, h1, f1, h2, f2, h3) = forward(x, phi)
    b0, w0, b1, w1, b2, w2, b3, w3 = phi
    dl_df = 2 * (out - y)
    df_dh3 = w3
    dh3_df2 = -np.sin(f2)
    df2_dh2 = w2
    dh2_df1 = np.exp(f1)
    df1_dh1 = w1
    dh1_df0 = np.cos(f0)
    return dl_df * df_dh3 * dh3_df2 * df2_dh2 * dh2_df1 * df1_dh1 * dh1_df0"""),
    ("scaled dot-product attention", """import torch

def attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights"""),
    ("beta-binomial estimates (Murphy 4.6)", """from scipy import stats

a, b = 2, 2              # Beta prior
n1, n0 = 7, 3            # heads, tails
post = stats.beta(a + n1, b + n0)
mle = n1 / (n1 + n0)
map_ = (a + n1 - 1) / (a + b + n1 + n0 - 2)
mean = post.mean()
low, high = post.interval(0.95)
print(mle, map_, round(mean, 3), (round(low, 3), round(high, 3)))"""),
    ("W mass from the vacuum value (Rubbia 25)", """import sympy as sp

g2, v, GF = sp.symbols("g_2 v G_F", positive=True)
M_W = g2 * v / 2
v_value = 1 / sp.sqrt(sp.sqrt(2) * GF)
numbers = {GF: 1.1664e-5, g2: 0.652}
print(sp.N(v_value.subs(numbers)))
print(sp.N(M_W.subs(v, v_value).subs(numbers)))"""),
]


def cmd_items(args) -> None:
    if ITEMS.exists():
        raise SystemExit(f"{ITEMS} exists: the items are fixed once made")
    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'app.sqlite'}?mode=ro", uri=True)
    items = []
    for number, (question, doc, page, marker) in enumerate(EQUATIONS, 1):
        doc_id = conn.execute("SELECT id FROM documents WHERE path = ?", (doc,)).fetchone()[0]
        text = next(t for (t,) in conn.execute("SELECT text FROM blocks WHERE doc_id = ? AND page = ? AND kind = 'equation'",
                                                (doc_id, page)) if marker in t)
        latex = text.strip().removeprefix("$$").removesuffix("$$").strip()
        with fitz.open(ROOT / "library" / doc) as pdf:
            context = pdf[page - 1].get_text().strip()
        items.append({"id": f"e{number:02d}", "kind": "equation", "source": f"{question}: {doc}, p. {page}",
                      "text": latex, "context": context})
    code = next(t for (t,) in conn.execute("SELECT text FROM blocks WHERE kind = 'code' AND doc_id = 3 AND page = 72"))
    lines = code.strip().removeprefix("```Python").removesuffix("```").strip().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("import"))
    items.append({"id": "c01", "kind": "code", "source": "UDL p. 72 (book listing)", "text": "\n".join(lines[start:]).strip(),
                  "context": ""})
    for number, (source, text) in enumerate(SNIPPETS, 2):
        items.append({"id": f"c{number:02d}", "kind": "code", "source": source, "text": text, "context": ""})
    ITEMS.write_text(yaml.safe_dump({"items": items}, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    print(f"{len(items)} items in {ITEMS.relative_to(ROOT)}")


def cmd_run(args) -> None:
    settings = load_settings()
    option = next(o for o in settings.answer_options if o.id == args.option)
    items = yaml.safe_load(ITEMS.read_text(encoding="utf-8"))["items"]
    path = RESULTS / f"explain-{args.name}.jsonl"
    done = {json.loads(line)["id"]: json.loads(line) for line in path.read_text().splitlines()} if path.exists() else {}
    spent = sum(r["cost_usd"] for r in done.values())
    with open(path, "a", encoding="utf-8") as out:
        for item in items:
            if item["id"] in done:
                continue
            if args.max_usd is not None and spent >= args.max_usd:
                print(f"Stopped at ${spent:.2f}: the limit is ${args.max_usd:.2f}")
                break
            started = time.monotonic()
            try:
                result = asyncio.run(explain.explain(option.apply(settings), option, item["text"], item["kind"],
                                                     "intermediate", item["context"]))
            except Exception as error:      # noqa: BLE001 - recorded as a failure, the run goes on
                record = {"id": item["id"], "option": args.option, "failed": str(error)[:300], "cost_usd": 0.0,
                          "seconds": round(time.monotonic() - started, 1)}
            else:
                record = {"id": item["id"], "option": args.option, "items": result["items"],
                          "cost_usd": result["cost_usd"], "seconds": round(time.monotonic() - started, 1)}
            spent += record["cost_usd"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{item['id']}: {'FAILED ' + record['failed'] if 'failed' in record else str(len(record['items'])) + ' rows'}, "
                  f"{record['seconds']} s, ${record['cost_usd']:.3f}", flush=True)


BRIEF = """You are grading line-by-line explanations of equations and code for a student at an intermediate
level. Items come from different sources in a shuffled order; don't try to tell them apart. Read this whole
file and no other file.

Each item gives the equation (LaTeX) or the code, where it is from, and the explanation as numbered rows:
a piece of the text and what the explainer said about it. An item marked FAILED had no explanation.

Verdict per item:
- good: every part is explained (each row of an equation, each line or small group of lines of code),
  every statement is correct, and the symbols are said what they are;
- weak: correct, but a part is left vague or skipped over, or a small inaccuracy that doesn't mislead;
- wrong: a wrong statement that would mislead the student, or parts missing, or FAILED.
Check the maths and the code yourself; don't trust the explanation.

Write valid JSON to {grades}:
[{{"item": "...", "verdict": "good", "problems": "short; empty for good"}}]
This batch has {count} items."""


def cmd_pack(args) -> None:
    items = {item["id"]: item for item in yaml.safe_load(ITEMS.read_text(encoding="utf-8"))["items"]}
    folder = GRADING / args.name
    if folder.exists():
        raise SystemExit(f"{folder} exists: a new pack needs a new name")
    rng = random.Random(args.seed)
    packed, key = [], {}
    for run in args.runs:
        for line in (RESULTS / f"explain-{run}.jsonl").read_text().splitlines():
            record = json.loads(line)
            item = items[record["id"]]
            item_id = f"{rng.randrange(16 ** 6):06x}"
            key[item_id] = {"run": run, "item": record["id"]}
            rows = ("FAILED" if "failed" in record else
                    "\n".join(f"{n}. TEXT: {row['span']}\n   EXPLANATION: {row['explanation']}"
                              for n, row in enumerate(record["items"], 1)))
            packed.append((item_id, f"=== ITEM {item_id}\nFROM: {item['source']}\n{item['kind'].upper()}:\n{item['text']}\n\n"
                                    f"EXPLANATION ROWS:\n{rows}\n=== END OF ITEM {item_id}"))
    rng.shuffle(packed)
    folder.mkdir(parents=True)
    for batch in range(1, args.batches + 1):
        chosen = packed[batch - 1::args.batches]
        (folder / f"batch-{batch}.txt").write_text(BRIEF.format(grades=folder / f"grades-{batch}.json", count=len(chosen))
                                                   + "\n\n" + "\n\n".join(text for _, text in chosen), encoding="utf-8")
    (folder / "key.json").write_text(json.dumps({"runs": args.runs, "seed": args.seed, "items": key}, indent=1))
    print(f"{len(packed)} items in {args.batches} batches in {folder}")


def cmd_score(args) -> None:
    folder = GRADING / args.name
    key = json.loads((folder / "key.json").read_text())
    grades = {g["item"]: g for path in sorted(folder.glob("grades-*.json")) for g in json.loads(path.read_text())}
    missing = sorted(set(key["items"]) - set(grades))
    if missing:
        raise SystemExit(f"no grades for {missing}")
    summary = {run: {"good": 0, "weak": 0, "wrong": 0, "items": {}} for run in key["runs"]}
    for item_id, entry in key["items"].items():
        grade = grades[item_id]
        summary[entry["run"]][grade["verdict"]] += 1
        summary[entry["run"]]["items"][entry["item"]] = f"{grade['verdict']}: {grade.get('problems', '')}"
    print(json.dumps(summary, indent=1))
    (RESULTS / f"explain-grades-{args.name}.json").write_text(json.dumps(summary, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("items").set_defaults(run=cmd_items)
    run = commands.add_parser("run")
    run.add_argument("--option", required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--max-usd", type=float)
    run.set_defaults(run=cmd_run)
    pack = commands.add_parser("pack")
    pack.add_argument("--name", required=True)
    pack.add_argument("--runs", nargs="+", required=True)
    pack.add_argument("--seed", type=int, required=True)
    pack.add_argument("--batches", type=int, default=2)
    pack.set_defaults(run=cmd_pack)
    score = commands.add_parser("score")
    score.add_argument("--name", required=True)
    score.set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
