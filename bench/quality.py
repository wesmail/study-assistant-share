#!/usr/bin/env python3
"""Phase 0 quality comparison: both main-model candidates, same hand-picked context.

Runs every question in eval/questions.yaml whose source document exists in library/,
saves each answer to bench/quality/raw/, then writes bench/quality/side_by_side.html.
Finished answers are skipped, so an interrupted run can simply be started again.

Usage:
  python3 bench/quality.py
  python3 bench/quality.py --ids deriv-005 figure-002 --force
  python3 bench/quality.py --render-only
"""

import argparse
import html
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
LIBRARY = ROOT / "library"
OUT = ROOT / "bench" / "quality"
OLLAMA = "http://127.0.0.1:11434"

# gpt-oss stays on "medium": on "high" it looped in its reasoning and ran out of tokens
# before answering on 7 of 15 questions (kept in bench/quality/raw_think_high/).
MODELS = {
    "gpt-oss:20b": {"think": "medium", "options": {}},
    "qwen3.6:35b-a3b": {"think": True, "options": {"num_batch": 2048}},
}
OPTIONS = {"num_ctx": 32768, "num_predict": 16384, "temperature": 0.2, "seed": 1}

SYSTEM = """You are a study assistant for physics, mathematics and code. Answer the question using the context provided.

- Explain step by step: the concept first, then the mathematics.
- Show derivations in full and do not skip steps.
- Use the context for facts about the document. You may use general knowledge of mathematics and physics in your reasoning, but say so when a statement goes beyond the context. If the context does not contain something, say that instead of guessing.
- Write mathematics in LaTeX, with $...$ inline and $$...$$ for display equations."""


def load_questions(ids=None):
    questions = yaml.safe_load(QUESTIONS.read_text())["questions"]
    selected = [q for q in questions if (LIBRARY / q["expected_sources"][0]["doc"]).exists()]
    if ids:
        selected = [q for q in selected if q["id"] in ids]
    return selected


def raw_path(question_id, model):
    return OUT / "raw" / f"{question_id}__{model.replace(':', '_')}.json"


def ask(model, question):
    settings = MODELS[model]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context from the document:\n\n{question['context']}\n"
                                        f"\nQuestion: {question['question']}"},
        ],
        "stream": False,
        "think": settings["think"],
        "keep_alive": "10m",
        "options": {**OPTIONS, **settings["options"]},
    }
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        reply = json.loads(r.read())
    return {
        "id": question["id"],
        "model": model,
        "answer": reply["message"].get("content", ""),
        "thinking": reply["message"].get("thinking", ""),
        "done_reason": reply.get("done_reason"),
        "prompt_tokens": reply.get("prompt_eval_count"),
        "generated_tokens": reply.get("eval_count"),
        "wall_s": round(time.perf_counter() - t0, 1),
        "settings": payload["options"] | {"think": settings["think"]},
    }


def unload(model):
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def run(questions, force):
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for q in questions:
            path = raw_path(q["id"], model)
            if path.exists() and not force:
                continue
            record = ask(model, q)
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            print(f"{model:18s} {q['id']:13s} {record['generated_tokens']:5d} tok "
                  f"{record['wall_s']:6.1f} s  {record['done_reason']}", flush=True)
        unload(model)


def to_html(markdown):
    """Markdown with LaTeX to an HTML fragment. Raw HTML/TeX stays visible as text."""
    # pandoc rejects display math with a blank line inside; models sometimes write that
    markdown = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + re.sub(r"\n\s*\n", "\n", m[1]) + "$$",
                      markdown, flags=re.S)
    return subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_single_backslash-raw_html-raw_tex",
         "-t", "html", "--katex"],
        input=markdown, capture_output=True, text=True, check=True).stdout


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality Comparison</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".math").forEach(el => katex.render(el.textContent, el, {
    displayMode: el.classList.contains("display"), throwOnError: false}));
});
</script>
<style>
:root { --bg: #fbfbfa; --ink: #1f2328; --muted: #646b73; --line: #dcdfe3; --panel: #ffffff; --warn: #a33a00; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16181b; --ink: #e4e6e9; --muted: #9aa1a9; --line: #33383e; --panel: #1e2125; --warn: #f0a070; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 1500px; padding: 24px 16px 64px; background: var(--bg); color: var(--ink);
       font: 16px/1.55 system-ui, sans-serif; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 48px 0 4px; }
.meta, details summary { color: var(--muted); font-size: .875rem; }
.question { font-size: 1.05rem; margin: 4px 0 12px; }
.pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }
.answer { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; overflow-x: auto; }
.answer h3 { margin: 0 0 2px; font-size: 1rem; }
.answer h1, .answer h2, .answer h3:not(.model), .answer h4 { font-size: 1rem; margin: 1.2em 0 .4em; }
.warn { color: var(--warn); font-weight: 600; }
table { border-collapse: collapse; margin: 8px 0; max-width: 100%; }
td, th { border: 1px solid var(--line); padding: 3px 8px; text-align: left; vertical-align: top; }
details { margin: 6px 0; }
pre { overflow-x: auto; }
@media (max-width: 800px) { .pair { grid-template-columns: minmax(0, 1fr); } }
</style></head><body>
<h1>Quality comparison: gpt-oss-20b vs Qwen3.6-35B-A3B</h1>
<p class="meta">Same system prompt and context for both models. Generated {date}. Rubric, 1 to 5 each: correctness, depth, clarity.</p>
{summary}
{sections}
</body></html>
"""


def render(questions):
    records = {}
    for q in questions:
        for model in MODELS:
            path = raw_path(q["id"], model)
            if path.exists():
                records[q["id"], model] = json.loads(path.read_text())

    rows, sections = [], []
    for q in questions:
        cells = [f"<td>{q['id']}</td>"]
        columns = []
        for model in MODELS:
            r = records.get((q["id"], model))
            if r is None:
                cells.append("<td>missing</td>")
                columns.append(f'<div class="answer"><h3 class="model">{model}</h3><p>Not generated.</p></div>')
                continue
            truncated = r["done_reason"] != "stop"
            flag = f' <span class="warn">{html.escape(str(r["done_reason"]))}: incomplete</span>' if truncated else ""
            empty = "" if r["answer"].strip() else '<p class="warn">Empty answer.</p>'
            cells.append(f"<td>{r['generated_tokens']} tok, {r['wall_s']:.0f} s{flag}</td>")
            columns.append(f'<div class="answer"><h3 class="model">{model}</h3>'
                           f'<p class="meta">{r["generated_tokens"]} tokens including thinking, '
                           f'{r["wall_s"]:.0f} s{flag}</p>{empty}{to_html(r["answer"])}</div>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
        points = "".join(f"<li>{html.escape(p)}</li>" for p in q["key_points"])
        pages = q["expected_sources"][0]["pages"]
        notes = f'<p class="meta">Note: {html.escape(q["notes"])}</p>' if q.get("notes") else ""
        sections.append(
            f'<section id="{q["id"]}"><h2>{q["id"]}</h2>'
            f'<p class="meta">{q["category"]}, difficulty {q["difficulty"]}, pages {pages}</p>'
            f'<p class="question">{html.escape(q["question"])}</p>'
            f'<details><summary>Context given to both models</summary>{to_html(q["context"])}</details>'
            f'<details><summary>Key points (open after reading the answers)</summary><ul>{points}</ul>{notes}</details>'
            f'<div class="pair">{"".join(columns)}</div></section>')

    header = "".join(f"<th>{m}</th>" for m in MODELS)
    summary = f"<table><tr><th>Question</th>{header}</tr>{''.join(rows)}</table>"
    page = PAGE.replace("{date}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    page = page.replace("{summary}", summary).replace("{sections}", "\n".join(sections))
    (OUT / "side_by_side.html").write_text(page)
    print(f"Wrote {OUT / 'side_by_side.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", help="only these question ids")
    ap.add_argument("--force", action="store_true", help="regenerate existing answers")
    ap.add_argument("--render-only", action="store_true", help="rebuild the HTML, run no models")
    args = ap.parse_args()

    questions = load_questions(args.ids)
    if not args.render_only:
        run(questions, args.force)
    render(load_questions())


if __name__ == "__main__":
    main()
