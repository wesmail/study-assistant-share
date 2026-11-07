"""Phase 5.1, step 1: how should figures be described? Ten figures, several describers, blind.

    python eval/figure_compare.py run [--describers qwen-vl]
    python eval/figure_compare.py pack --name figure-compare-2 --describers qwen-vl opus --seed 20261109

The first pack had Gemma 4 locally with thinking off and on, and Opus 5.5 as a
reference; Gemma misread too much. The second has Qwen3-VL 8B
locally against the same Opus descriptions, and the third Sonnet 5.5 against them, after
Qwen3-VL misread and invented values too. Every describer gets the same prompt, image,
caption and page text; Sonnet and Opus the same effort, thinking and output limit.
The ten figures are held out from eval/figure_questions.yaml, so the gate isn't
measured on the figures the setting was chosen on.

`pack` writes eval/grading/figure-compare/ (gitignored: it holds book images): a review
page with the three descriptions of each figure in a fresh random order per figure,
labelled A, B and C, and grader batches for blind checks against the image. key.json
holds the labels. Opus calls are logged in hosted_calls, purpose eval:figure-compare.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import random
import re
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_settings  # noqa: E402
from app.ingest import figures  # noqa: E402
from app.llm import hosted  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "figure-compare.jsonl"
GRADING = ROOT / "eval" / "grading"
PACK = GRADING / "figure-compare"
SEED = 20261108
PAGE_TEXT_CHARS = 1800
GEMMA = "gemma4:26b-a4b-it-qat"
QWEN_VL = "qwen3-vl:8b-instruct"
TEMPERATURE = 0.3           # both Gemma runs, so thinking is the only difference
# Image, caption and page text take about 1,500 tokens. Thinking off writes about 400; thinking
# on first ran out of an 8K context while still thinking, so it gets room of its own.
LIMITS = {False: {"num_ctx": 8192, "num_predict": 1024}, True: {"num_ctx": 16384, "num_predict": 12000}}
NO_DESCRIPTION = "(No description: the model stopped at its output limit before writing one.)"
PURPOSE = "eval:figure-compare"
# Held out from the figure questions: a schematic, multi-panel plots, data with values, a vector diagram.
FIGURES = [1, 3, 20, 149, 175, 254, 336, 349, 356, 383]
DESCRIBERS = ("gemma-off", "gemma-on", "opus", "qwen-vl", "sonnet")
HOSTED = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}     # same effort, thinking and limit
FIRST_PACK = ("gemma-off", "gemma-on", "opus")
REVEALING = re.compile(r"\b(gemma|claude|opus|sonnet|anthropic|google|deepmind|qwen|alibaba)\b|</?think", re.I)

PROMPT = figures.PROMPT     # the app's prompt: what was compared is what runs


def figure_inputs(conn: sqlite3.Connection, settings, figure_id: int) -> dict:
    row = conn.execute(
        "SELECT f.id, f.page, f.caption, f.image_path, d.sha256, d.title, d.id AS doc_id FROM figures f "
        "JOIN documents d ON d.id = f.doc_id WHERE f.id = ?", (figure_id,)).fetchone()
    paragraphs = [text for (text,) in conn.execute(
        'SELECT text FROM blocks WHERE doc_id = ? AND page = ? AND kind IN ("paragraph", "heading", "list_item") '
        'ORDER BY "order"', (row["doc_id"], row["page"]))]
    image = settings.parsed_dir / row["sha256"] / row["image_path"]
    return {"figure": row["id"], "document": row["title"], "page": row["page"], "caption": row["caption"] or "",
            "page_text": "\n\n".join(paragraphs)[:PAGE_TEXT_CHARS], "image": str(image.relative_to(ROOT))}


def user_text(inputs: dict) -> str:
    return (f"Caption: {inputs['caption'] or '(none)'}\n\nText from the same page:\n{inputs['page_text'] or '(none)'}"
            "\n\nDescribe the figure.")


def image_b64(inputs: dict) -> str:
    return base64.b64encode((ROOT / inputs["image"]).read_bytes()).decode()


async def gemma(inputs: dict, settings, think: bool, model: str = GEMMA) -> dict:
    """A local vision model through Ollama. Qwen3-VL instruct has no thinking switch, so none is sent."""
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": user_text(inputs), "images": [image_b64(inputs)]}],
        "options": {**LIMITS[think], "temperature": TEMPERATURE},
    }
    if model == GEMMA:
        payload["think"] = think
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10)) as client:
        response = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
    message = body.get("message") or {}
    text = (message.get("content") or "").strip()
    return {"text": text or NO_DESCRIPTION, "no_description": not text,
            "thinking_chars": len(message.get("thinking") or ""),
            "seconds": round(time.monotonic() - started, 1), "output_tokens": body.get("eval_count"),
            "done_reason": body.get("done_reason"), "cost_usd": 0.0}


async def hosted_describe(inputs: dict, settings, model: str) -> dict:
    settings = replace(settings, hosted_model=model)
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64(inputs)}},
        {"type": "text", "text": user_text(inputs)}]}]
    started = time.monotonic()
    text, done = "", None
    async for delta in hosted.stream_chat(messages, settings, purpose=PURPOSE):
        text += delta.text
        if delta.done:
            done = delta
    return {"text": text.strip(), "seconds": round(time.monotonic() - started, 1),
            "output_tokens": done.output_tokens if done else None, "done_reason": done.done_reason if done else None,
            "cost_usd": done.cost_usd if done else 0.0}


def load_done() -> dict[tuple[int, str], dict]:
    if not RESULTS.exists():
        return {}
    records = map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines())
    return {(record["figure"], record["describer"]): record for record in records}


async def cmd_run(args) -> None:
    settings = load_settings()
    settings = replace(settings.option("deep").apply(settings), hosted_max_tokens=16000)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    done = load_done()
    RESULTS.parent.mkdir(exist_ok=True)
    # One describer at a time: one local model load, and no switching between figures.
    for describer in args.describers:
        for figure_id in FIGURES:
            if (figure_id, describer) in done:
                continue
            inputs = figure_inputs(conn, settings, figure_id)
            try:
                if describer in HOSTED:
                    result = await hosted_describe(inputs, settings, HOSTED[describer])
                elif describer == "qwen-vl":
                    result = await gemma(inputs, settings, think=False, model=QWEN_VL)
                else:
                    result = await gemma(inputs, settings, think=describer == "gemma-on")
            except (httpx.HTTPError, RuntimeError) as error:
                print(f"figure {figure_id} {describer}: FAILED, not saved: {error}", flush=True)
                continue
            record = {**inputs, "describer": describer, **result}
            with open(RESULTS, "a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"figure {figure_id} {describer}: {result['seconds']} s, {result['output_tokens']} tokens, "
                  f"{len(result['text'].split())} words, {result['done_reason']}, ${result['cost_usd']:.4f}", flush=True)


GRADER_BRIEF = """You are checking descriptions of figures written by {count} unnamed describers, for a study assistant whose
answers will rely on them. Labels are shuffled for each figure, so "A" for one figure has nothing to do with "A"
for another. Don't try to work out who wrote which. Read this file and look at each figure's image with your
image-reading tool; read no other file.

For each description, compare it with the image, the caption and the page text:
- invented: values, labels or features stated that the image doesn't show or that can't be read from it
  (give each one).
- misread: axes, units, ranges, legend entries, curves or values read wrongly (give each one, with the right
  reading).
- missing: what a reader would need to answer questions about the figure that the description leaves out.
- verdict: "good" (accurate, nothing invented, covers the main content), "weak" (accurate but thin, or one small
  error), or "wrong" (an invented value, a misread axis or curve, or the main content wrong).

Write valid JSON to {path}, one entry per figure:
[{{"figure": 1, "A": {{"verdict": "good|weak|wrong", "invented": [...], "misread": [...], "missing": [...],
"note": "one or two sentences"}}, {more}}}]
"""


def cmd_pack(args) -> None:
    describers = tuple(args.describers) if args else FIRST_PACK
    seed = args.seed if args else SEED
    pack = GRADING / args.name if args else PACK
    letters = "ABCD"[:len(describers)]
    records = load_done()
    missing = [(figure, describer) for figure in FIGURES for describer in describers if (figure, describer) not in records]
    if missing:
        raise SystemExit(f"Not every figure has all its descriptions yet: {missing}")
    if pack.exists():
        raise SystemExit(f"{pack} exists. A new pack needs a new folder, so that nobody judges twice.")
    for (figure, describer), record in records.items():
        if describer in describers and REVEALING.search(record["text"]):
            raise SystemExit(f"Figure {figure} ({describer}) names a model or shows its thinking, which would unblind it")
    rng = random.Random(seed)
    key, sections, batches = {}, [], [[], []]
    for number, figure in enumerate(FIGURES):
        order = list(describers)
        rng.shuffle(order)
        key[str(figure)] = dict(zip(letters, order))
        first = records[(figure, order[0])]
        texts = {label: records[(figure, describer)]["text"] for label, describer in zip(letters, order)}
        sections.append((figure, first, texts))
        packet = (f"################ FIGURE {figure} ################\n\nIMAGE: {ROOT / first['image']}\n"
                  f"DOCUMENT: {first['document']}, page {first['page']}\n\nCAPTION: {first['caption']}\n\n"
                  f"PAGE TEXT:\n{first['page_text']}\n\n"
                  + "\n\n".join(f"=== DESCRIPTION {label} ===\n{text}" for label, text in texts.items()))
        batches[number % 2].append(packet)
    pack.mkdir(parents=True)
    (pack / "key.json").write_text(json.dumps({"seed": seed, "describers": list(describers), "labels": key}, indent=1))
    counts = {2: "two", 3: "three", 4: "four"}
    more = ", ".join(f'"{letter}": {{...}}' for letter in letters[1:])
    for number, packets in enumerate(batches, 1):
        brief = GRADER_BRIEF.format(path=pack / f"grades-{number}.json", count=counts[len(describers)], more=more)
        (pack / f"batch-{number}.txt").write_text(brief + "\n\n" + "\n\n".join(packets), encoding="utf-8")
    (pack / "review.html").write_text(review_page(sections, letters), encoding="utf-8")
    print(f"{len(FIGURES)} figures in {pack}: review.html, batch-1.txt, batch-2.txt")


def review_page(sections, letters: str = "ABC") -> str:
    """One figure at a time: the image, its caption, the unnamed descriptions, and a verdict for each."""
    blocks = []
    for figure, first, texts in sections:
        image = base64.b64encode((ROOT / first["image"]).read_bytes()).decode()
        choices = "".join(
            f'<div class="d"><h3>{label}</h3><p>{html.escape(text)}</p>'
            + "".join(f'<label><input type="radio" name="f{figure}{label}" value="{v}"> {v}</label> '
                      for v in ("good", "weak", "wrong"))
            + f'<label><input type="checkbox" name="f{figure}{label}x"> invents or misreads</label>'
            f'<input type="text" name="f{figure}{label}n" placeholder="note"></div>'
            for label, text in texts.items())
        blocks.append(f'<section><h2>Figure {figure} · {html.escape(first["document"])}, page {first["page"]}</h2>'
                      f'<img src="data:image/png;base64,{image}"><p class="c">{html.escape(first["caption"])}</p>'
                      f'<div class="row">{choices}</div></section>')
    figures = json.dumps(FIGURES)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Figure descriptions, blind</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:16px auto;max-width:1400px;background:#fafaf9;color:#1c1917}}
section{{background:#fff;border:1px solid #e7e5e4;border-radius:8px;padding:16px;margin:16px 0}}
img{{max-width:100%;max-height:520px;display:block;margin:auto;border:1px solid #e7e5e4}}
.c{{font-size:13px;color:#57534e}} .row{{display:grid;grid-template-columns:repeat({len(letters)},1fr);gap:12px}}
.d{{border:1px solid #e7e5e4;border-radius:6px;padding:10px}} .d p{{white-space:pre-wrap;font-size:14px}}
input[type=text]{{width:100%;margin-top:6px}} button{{font-size:15px;padding:8px 14px}}
@media (max-width:900px){{.row{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Figure descriptions, blind</h1><p>{len(letters)} describers per figure, labelled {", ".join(letters)} in a different random
order for each figure. Judge each against the image: good, weak or wrong, and tick it if it invents or misreads
anything. Then press the button and paste the result into the chat.</p>
{''.join(blocks)}
<button onclick="copyVerdicts()">Copy my verdicts</button> <span id="s"></span>
<script>function copyVerdicts(){{const out=[];for(const f of {figures}){{const e={{figure:f}};for(const l of "{letters}"){{
const v=document.querySelector(`input[name="f${{f}}${{l}}"]:checked`);e[l]={{verdict:v?v.value:null,
flagged:document.querySelector(`input[name="f${{f}}${{l}}x"]`).checked,
note:document.querySelector(`input[name="f${{f}}${{l}}n"]`).value}}}}out.push(e)}}
const text=JSON.stringify(out);navigator.clipboard.writeText(text).then(()=>document.getElementById("s").textContent=
"Copied.",()=>{{document.getElementById("s").textContent=text}})}}</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--describers", nargs="+", choices=DESCRIBERS, default=list(FIRST_PACK))
    run.set_defaults(run=lambda a: asyncio.run(cmd_run(a)))
    pack = commands.add_parser("pack")
    pack.add_argument("--name", required=True, help="a new folder under eval/grading/")
    pack.add_argument("--describers", nargs="+", choices=DESCRIBERS, required=True)
    pack.add_argument("--seed", type=int, required=True, help="fix it before seeing any results")
    pack.set_defaults(run=cmd_pack)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
