"""Formula spot check: the Phase 1 gate.

    python eval/spot_check.py sample --n 50 --name gate --seed S --min-pages 50 --at-least Tong=15
    python eval/spot_check.py score  --name gate

`sample` fixes a seeded random sample of indexed formulas and writes, under
eval/spot_check/<name>/, a PNG crop of each one from the PDF, the KaTeX compile
result, sample.json and gallery.html. Verdicts then go in verdicts.json, as
{"<id>": {"verdict": "correct" | "wrong" | "excluded", "note": "...",
"bold": "ok" | "lost", "number": "ok" | "lost", "crop": "ok" | "bad"}}. The last
three are optional: set "bold" when the printed formula has a bold symbol, and
"number" when it has a printed equation number, so recovery is measured against
the page. Set "crop" on each unverified formula whose LaTeX is wrong, after
looking at the crop saved at ingest.

Grade in two passes, so the flags can't colour the grading: first every "verdict"
with the LaTeX and the page only, then "crop" with the flags in view.

Correct means the same mathematics as the page: every symbol, operator, index
and structure, plus the equation number if one is printed. Typography alone
(italic or upright, spacing) doesn't count against it, but bold and calligraphic
letters are different symbols from plain ones. Wrong means a changed, missing or
extra symbol, lost sub- or superscript structure, a truncated formula, or LaTeX
that doesn't compile. "excluded" drops a sampled block that isn't a formula.

The gate has two parts, both scored on display formulas (decided
2026-11-01):

    shown correctly  the LaTeX is right, or the formula is unverified and its
                     crop shows it. At least GATE_SHOWN.
    silent error     the LaTeX is wrong and nothing warns the reader. At most
                     GATE_SILENT.

Inline formulas are plain text in the paragraphs, not LaTeX, so the compile
check covers display formulas only. `--n` counts display formulas, inline ones
are sampled on top at INLINE_SHARE and reported separately, and they can't move
the pass or fail.

A formula dropped as junk stays in the population and can be sampled. It was on
the page and the reader failed to deliver it, with no warning, so it counts
against both parts, or as excluded if the block isn't a formula. Otherwise
dropping would raise the score.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import fitz  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.ingest.chunker import DROPPED_EQUATION  # noqa: E402
from app.ingest.formula_layer import junk_reason  # noqa: E402

OUT = ROOT / "eval" / "spot_check"
NODE = ROOT / ".venv" / "bin" / "node"
KATEX_CHECK = ROOT / "eval" / "katex_check.mjs"
INLINE_SHARE = 0.3
URL = re.compile(r"\S*(?:https?://|www\.)\S*|\S+\.\w+/\S*")
INLINE_MATH = re.compile(r"[=≈≠≤≥≡∑∏∫∂∇√±∓×∈∞αβγδεζηθλμνξπρστφχψωΓΔΘΛΣΦΨΩ]")
GATE_SHOWN = 0.90
GATE_SILENT = 0.05


def strip_delimiters(text: str) -> str:
    return re.sub(r"^\$\$\s*|\s*\$\$$", "", text.strip())


def allocate(
    total: int, sizes: dict[int, int], floors: dict[int, int] | None = None
) -> dict[int, int]:
    """Split `total` across documents by size, giving each at least its floor.

    A document under its floor is held there, and the rest share what remains.
    """
    fixed: dict[int, int] = {}
    remaining = dict(sizes)
    while True:
        shares = _proportional(total - sum(fixed.values()), remaining)
        short = [key for key, share in shares.items() if share < (floors or {}).get(key, 0)]
        if not short:
            return {**fixed, **shares}
        for key in short:
            fixed[key] = min(floors[key], sizes[key])
            del remaining[key]


def _proportional(total: int, sizes: dict[int, int]) -> dict[int, int]:
    """Split `total` in proportion to size, by largest remainder."""
    population = sum(sizes.values())
    if population <= total:
        return dict(sizes)
    exact = {key: total * size / population for key, size in sizes.items()}
    counts = {key: min(int(value), sizes[key]) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: exact[key] - counts[key], reverse=True)
    for key in by_remainder[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def floors_by_document(rows: list, at_least: dict[str, int]) -> dict[int, int]:
    """Document ids for each "text in its file name" floor. No match stops the draw."""
    floors = {}
    for text, count in at_least.items():
        matching = {row["doc_id"] for row in rows if text.lower() in row["path"].lower()}
        if len(matching) != 1:
            raise SystemExit(f"--at-least {text}: {len(matching)} documents match, expected 1")
        floors[matching.pop()] = count
    return floors


def draw(
    conn, n: int, seed: int, min_pages: int, at_least: dict[str, int] | None = None,
    inline_share: float = INLINE_SHARE,
) -> list:
    rng = random.Random(seed)
    display = conn.execute(
        "SELECT b.*, d.path, d.sha256 FROM blocks b JOIN documents d ON d.id = b.doc_id "
        "WHERE b.kind IN ('equation', ?) AND d.status = 'indexed' AND d.pages >= ? ORDER BY b.id",
        (DROPPED_EQUATION, min_pages),
    ).fetchall()
    # A block holding only "(2.28)" isn't a formula, so it was never a candidate.
    display = [row for row in display if junk_reason(strip_delimiters(row["text"])) != "number"]
    prose = conn.execute(
        "SELECT b.*, d.path, d.sha256 FROM blocks b JOIN documents d ON d.id = b.doc_id "
        "WHERE b.kind IN ('paragraph', 'list_item') AND d.status = 'indexed' AND d.pages >= ? "
        "ORDER BY b.id",
        (min_pages,),
    ).fetchall()
    inline = [row for row in prose if INLINE_MATH.search(URL.sub(" ", row["text"]))]

    n_display = min(n, len(display))
    n_inline = min(round(n * inline_share), len(inline))

    floors = floors_by_document(display, at_least or {})
    picked = []
    for kind, rows, count in (("display", display, n_display), ("inline", inline, n_inline)):
        by_doc: dict[int, list] = {}
        for row in rows:
            by_doc.setdefault(row["doc_id"], []).append(row)
        shares = allocate(count, {key: len(value) for key, value in by_doc.items()},
                          floors if kind == "display" else None)
        for doc_id, share in shares.items():
            picked += [(kind, row) for row in rng.sample(by_doc[doc_id], share)]
    return picked


def sidecar(settings, sha256: str) -> dict[int, dict]:
    """The merge's own record of each block, by reading order."""
    path = settings.parsed_dir / sha256 / "blocks.jsonl"
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {record["order"]: record for record in records}


def crop(pdf: fitz.Document, regions: list[dict], target: Path, dpi: int = 200) -> None:
    """Render each region, padded, and stack them when a block spans pages."""
    images = []
    for region in regions:
        page = pdf[region["page"] - 1]
        rect = (fitz.Rect(region["bbox"]) + (-6, -6, 6, 6)) & page.rect
        images.append(page.get_pixmap(clip=rect, dpi=dpi))
    if len(images) == 1:
        images[0].save(target)
        return
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + 8 * (len(images) - 1)
    sheet = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    sheet.clear_with(255)
    offset = 0
    for image in images:
        image.set_origin(0, offset)
        sheet.copy(image, image.irect)
        offset += image.height + 8
    sheet.save(target)


def compile_all(latex: list[str]) -> list[dict]:
    result = subprocess.run(
        [str(NODE), str(KATEX_CHECK)],
        input=json.dumps(latex), capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


GALLERY = """<!doctype html><meta charset="utf-8"><title>Formula spot check</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js"></script>
<style>body{{font:15px system-ui;margin:24px;max-width:1000px}}
.item{{border-top:1px solid #ccc;padding:14px 0}}img{{max-width:100%}}
pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px}}.bad{{color:#b00}}</style>
<h1>Formula spot check: {name}</h1><p>{count} formulas. Page crop first, parsed version below.</p>
{items}
<script>addEventListener('DOMContentLoaded',()=>document.querySelectorAll('.tex').forEach(e=>
katex.render(e.dataset.tex,e,{{displayMode:true,throwOnError:false,strict:false}})))</script>
"""


def gallery(name: str, samples: list[dict]) -> str:
    items = []
    for s in samples:
        heading = f"#{s['id']} · {html.escape(s['doc'])} · p. {s['page']} · {s['kind']}"
        parsed = (f'<div class="tex" data-tex="{html.escape(s["latex"], quote=True)}"></div>'
                  f"<pre>{html.escape(s['latex'])}</pre>"
                  if s["kind"] == "display" else f"<pre>{html.escape(s['text'])}</pre>")
        error = f'<p class="bad">KaTeX: {html.escape(s["compile_error"])}</p>' if s.get("compile_error") else ""
        items.append(f'<div class="item"><b>{heading}</b><br><img src="{s["crop"]}">{parsed}{error}</div>')
    return GALLERY.format(name=html.escape(name), count=len(samples), items="\n".join(items))


def cmd_sample(args) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    out = OUT / args.name
    if (out / "sample.json").exists() and not args.force:
        raise SystemExit(f"{out} already has a sample; a gate sample is drawn once. Use --force to redraw.")
    (out / "crops").mkdir(parents=True, exist_ok=True)

    samples = []
    pdfs: dict[str, fitz.Document] = {}
    records: dict[str, dict[int, dict]] = {}
    at_least = dict(item.rsplit("=", 1) for item in args.at_least)
    at_least = {text: int(count) for text, count in at_least.items()}
    picked = draw(conn, args.n, args.seed, args.min_pages, at_least, args.inline_share)
    for index, (kind, row) in enumerate(picked, 1):
        pdf = pdfs.setdefault(row["path"], fitz.open(settings.library_dir / row["path"]))
        record = records.setdefault(row["sha256"], sidecar(settings, row["sha256"])).get(row["order"], {})
        name = f"crops/{index:03d}.png"
        # Every display formula uses the box a saved crop would, so the size of
        # the picture can't give away which ones are flagged.
        regions = json.loads(row["bboxes_json"])
        if record.get("crop_bbox"):
            regions = [{"page": row["page"], "bbox": record["crop_bbox"]}]
        crop(pdf, regions, out / name)
        stored = record.get("crop")
        samples.append({
            "id": index, "kind": kind, "block_id": row["id"], "doc": row["path"],
            "page": row["page"], "crop": name, "text": row["text"],
            "latex": strip_delimiters(row["text"]) if kind == "display" else None,
            "dropped": row["kind"] == DROPPED_EQUATION,
            "drop_reason": record.get("drop_reason"),
            "source": record.get("formula_source"),
            "alt_latex": record.get("alt_latex"),
            "unverified": bool(record.get("unverified")),
            "unverified_reason": record.get("unverified_reason"),
            "stored_crop": str(settings.parsed_dir / row["sha256"] / stored) if stored else None,
        })

    display = [s for s in samples if s["kind"] == "display"]
    for sample, outcome in zip(display, compile_all([s["latex"] for s in display])):
        sample["compile_ok"] = outcome["ok"] and not sample["dropped"]
        sample["compile_error"] = "dropped as junk" if sample["dropped"] else outcome.get("error")

    (out / "sample.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "gallery.html").write_text(gallery(args.name, samples), encoding="utf-8")
    failed = [s["id"] for s in display if not s["compile_ok"]]
    print(f"{len(samples)} sampled ({len(display)} display, {len(samples) - len(display)} inline) "
          f"from {len({s['doc'] for s in samples})} document(s)")
    print(f"KaTeX: {len(display) - len(failed)}/{len(display)} compile; failed ids: {failed}")
    print(f"Review: {out / 'gallery.html'}")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; unlike p ± 2σ it behaves near 100%."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - margin, centre + margin


def judge(sample: dict, verdict: dict) -> tuple[bool, bool, bool]:
    """(the LaTeX is right, the reader is shown the formula, a silent error)."""
    right = verdict["verdict"] == "correct" and sample.get("compile_ok", True)
    flagged = bool(sample.get("unverified"))
    shown = right or (flagged and verdict.get("crop") == "ok")
    return right, shown, not right and not flagged


def rate(label: str, count: int, n: int) -> None:
    low, high = wilson(count, n)
    print(f"{label}: {count}/{n} = {count / n:.1%}  (95% CI {low:.0%}-{high:.0%})")


def cmd_score(args) -> None:
    out = OUT / args.name
    samples = json.loads((out / "sample.json").read_text(encoding="utf-8"))
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    missing = [s["id"] for s in samples if str(s["id"]) not in verdicts]
    if missing:
        raise SystemExit(f"No verdict for ids {missing}")
    excluded = [s["id"] for s in samples if verdicts[str(s["id"])]["verdict"] == "excluded"]
    samples = [s for s in samples if s["id"] not in excluded]
    if excluded:
        print(f"Excluded as not formulas: {excluded}")

    display = [s for s in samples if s["kind"] == "display"]
    inline = [s for s in samples if s["kind"] == "inline"]
    judged = {s["id"]: judge(s, verdicts[str(s["id"])]) for s in display}
    flagged = [s for s in display if s.get("unverified")]
    ungraded = [s["id"] for s in flagged if not judged[s["id"]][0] and "crop" not in verdicts[str(s["id"])]]
    if ungraded:
        raise SystemExit(f"No crop verdict for unverified ids {ungraded}")
    unsaved = [s["id"] for s in flagged if not (s.get("stored_crop") and Path(s["stored_crop"]).exists())]
    if unsaved:
        raise SystemExit(f"Unverified formulas with no crop on disk: {unsaved}")

    n = len(display)
    shown = sum(outcome[1] for outcome in judged.values())
    silent = sum(outcome[2] for outcome in judged.values())
    wrong = [s for s in display if not judged[s["id"]][0]]
    caught = [s for s in wrong if s.get("unverified")]
    rate("GATE shown correctly", shown, n)
    rate("GATE silent errors ", silent, n)
    rate("LaTeX right on its own", n - len(wrong), n)
    rate("Flagged unverified", len(flagged), n)
    if flagged:
        rate("  flagged and truly wrong", len(caught), len(flagged))
    if wrong:
        rate("  wrong and flagged", len(caught), len(wrong))
    compiled = sum(s["compile_ok"] for s in display)
    rate("Compile (not gated)", compiled, n)
    dropped = [s["id"] for s in display if s.get("dropped")]
    if dropped:
        print(f"Dropped as junk, counted against both parts: {dropped}")
    for field, label in (("bold", "Bold recovered"), ("number", "Equation number recovered")):
        marked = [verdicts[str(s["id"])][field] for s in display if field in verdicts[str(s["id"])]]
        if marked:
            rate(label, marked.count("ok"), len(marked))
    if inline:
        rate("Inline right (not gated)", sum(verdicts[str(s["id"])]["verdict"] == "correct" for s in inline), len(inline))
    print("By document:")
    for doc in sorted({s["doc"] for s in display}):
        part = [s for s in display if s["doc"] == doc]
        print(f"  {doc[:40]}: n={len(part)} right={sum(judged[s['id']][0] for s in part)} "
              f"shown={sum(judged[s['id']][1] for s in part)} silent={sum(judged[s['id']][2] for s in part)}")
    print("Wrong:")
    for s in wrong:
        state = "flagged" if s.get("unverified") else "SILENT"
        crop_state = f", crop {verdicts[str(s['id'])].get('crop')}" if s.get("unverified") else ""
        print(f"  #{s['id']} {s['doc'][:12]} p.{s['page']} [{state}{crop_state}]: "
              f"{verdicts[str(s['id'])].get('note', '')}")
    gate = n >= 50 and shown / n >= GATE_SHOWN and silent / n <= GATE_SILENT
    print(f"Gate (n >= 50, shown >= {GATE_SHOWN:.0%}, silent <= {GATE_SILENT:.0%}): "
          f"{'PASS' if gate else 'NOT MET'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--n", type=int, default=50)
    sample.add_argument("--seed", type=int, default=20261031)
    sample.add_argument("--name", default="gate")
    sample.add_argument("--min-pages", type=int, default=0, help="leave out shorter documents")
    sample.add_argument("--at-least", action="append", default=[], metavar="TEXT=COUNT",
                        help="display formulas from the document with TEXT in its file name")
    sample.add_argument("--inline-share", type=float, default=INLINE_SHARE,
                        help="inline formulas to draw on top, as a share of --n; 0 for none")
    sample.add_argument("--force", action="store_true")
    sample.set_defaults(run=cmd_sample)
    score = commands.add_parser("score")
    score.add_argument("--name", default="gate")
    score.set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
